"""Embedding backends.

Both processes call an embedding server over HTTP rather than loading the
model in-process, so the weights live in exactly one place regardless of how
many components are running.

Two wire protocols are supported:

  openai   POST /v1/embeddings   {"model": ..., "input": [...]}
           -> {"data": [{"index": 0, "embedding": [...]}, ...]}
           llama.cpp (llama-server --embeddings), TEI, vLLM, LM Studio,
           localai, and anything else speaking the OpenAI embeddings API.

  ollama   POST /api/embed       {"model": ..., "input": [...]}
           -> {"embeddings": [[...], ...]}
           Ollama only.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from typing import Any, Sequence

import httpx

from Config import EmbeddingConfig
from Pdf import clean_text, is_embeddable

log = logging.getLogger(__name__)


class EmbeddingServerError(RuntimeError):
    """The server answered with an error. The body carries the reason, and it
    is almost always a configuration problem worth reading verbatim."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


_WHITESPACE = re.compile(r"\s+")

# Conservative chars/token for wire sizing. Extract.CHARS_PER_TOKEN (4)
# measures chunk targets; this guards a hard server limit, so it must not
# under-count. bge-m3's XGLM tokenizer lands near 2 chars/token on Finnish.
WIRE_CHARS_PER_TOKEN = 2


def _estimate_tokens(text: str) -> int:
    return len(text) // WIRE_CHARS_PER_TOKEN + 1


def _split_wire_text(text: str, limit_chars: int) -> list[str]:
    """Break one over-long text at whitespace into pieces under limit_chars."""
    words = text.split()
    pieces: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for word in words:
        extra = len(word) + (1 if cur else 0)
        if cur and cur_len + extra > limit_chars:
            pieces.append(" ".join(cur))
            cur = [word]
            cur_len = len(word)
        else:
            cur.append(word)
            cur_len += extra
    if cur:
        pieces.append(" ".join(cur))
    return pieces or [text[:limit_chars]]


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Average piece vectors and re-normalise (bge-m3 embeddings are unit
    length, so the mean of unit vectors only needs rescaling to unit length).
    An average of parts approximates the whole within the pooling model."""
    dim = len(vectors[0])
    mean = [sum(v[d] for v in vectors) / len(vectors) for d in range(dim)]
    norm = math.sqrt(sum(x * x for x in mean))
    if norm > 0:
        mean = [x / norm for x in mean]
    return mean


def wire_safe(text: str) -> str:
    """Flatten a chunk to a single line before sending it to the server.

    Stored chunk text keeps its newlines — procedures need their line breaks
    when read back — but the wire form does not. A sentence embedder derives
    nothing from layout, and some bge-m3 GGUF conversions ship without a
    vocabulary entry for U+000A and without byte fallback, so the tokenizer
    does token_to_id.at('\n'), throws std::out_of_range, and llama-server
    reports the C++ string "_Map_base::at" as an opaque 500.

    Flattening here rather than in pdf.clean_text keeps the two concerns
    apart: what is stored, and what is safe to transmit.
    """
    return _WHITESPACE.sub(" ", text).strip()


def _body_message(resp: httpx.Response) -> str:
    """Pull the human-readable reason out of an error response.

    llama.cpp, Ollama and TEI each nest it differently, and an empty body is
    itself a useful signal.
    """
    text = resp.text.strip()
    if not text:
        return "(empty response body)"
    try:
        data = resp.json()
    except ValueError:
        return text[:500]

    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if isinstance(err, str):
            return err
        for key in ("message", "detail"):
            if key in data:
                return str(data[key])
    return text[:500]


class Embedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        # Resolved against /v1/models at startup. None means "send no model
        # field", which is what a single-model llama-server wants.
        self._wire_model: str | None = config.model or None
        self._client: httpx.AsyncClient | None = None

    def client(self) -> httpx.AsyncClient:
        """The HTTP client, reopened on demand if it was closed.

        The client's lifetime belongs to the process, not to one MCP session.
        Older mcp SDKs re-entered the FastMCP lifespan per streamable-HTTP
        session, so a session ending (client DELETE, idle timeout, crash) ran
        the lifespan teardown and closed this client while the server kept
        serving; every later tool call then died with httpx's "Cannot send a
        request, as the client has been closed." Reopening here makes that
        teardown survivable regardless of SDK version.
        """
        if self._client is None or self._client.is_closed:
            headers = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url.rstrip("/"),
                timeout=httpx.Timeout(self._config.timeout, connect=10.0),
                headers=headers,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def embed(self, texts: Sequence[str]) -> list[list[float] | None]:
        """Embed a batch. Result order matches input order.

        Inputs that would tokenise to nothing are never sent: they yield None
        in the corresponding slot, and the caller drops that chunk. A server
        asked to embed nothing has no result to return, and llama.cpp answers
        that with std::out_of_range rather than an error, which would abort the
        whole batch over one blank page.

        llama.cpp folds every text of one request into a single physical
        batch, so the token SUM across a request must fit the server's
        --ubatch-size. Requests are therefore packed under both
        `batch_size` (count) and `max_tokens_per_request` (summed estimate).
        A single input larger than the request budget is split at whitespace
        and its piece vectors averaged back into one.
        """
        if not texts:
            return []

        cleaned = [wire_safe(clean_text(t)) for t in texts]
        sendable = [i for i, t in enumerate(cleaned) if is_embeddable(t)]

        if len(sendable) != len(cleaned):
            log.warning(
                "Skipping %d of %d inputs with no embeddable content",
                len(cleaned) - len(sendable), len(cleaned),
            )

        results: list[list[float] | None] = [None] * len(cleaned)
        max_tokens = self._config.max_tokens_per_request
        limit_chars = max(max_tokens * WIRE_CHARS_PER_TOKEN, 1)

        # Explode oversized inputs into (result_index, piece) units so the
        # packer never builds a request the server will reject.
        units: list[tuple[int, str]] = []
        for i in sendable:
            text = cleaned[i]
            if _estimate_tokens(text) > max_tokens:
                pieces = _split_wire_text(text, limit_chars)
                log.warning(
                    "Input ~%d tokens exceeds the %d-token request budget; "
                    "split into %d pieces and averaging",
                    _estimate_tokens(text), max_tokens, len(pieces),
                )
                units.extend((i, piece) for piece in pieces)
            else:
                units.append((i, text))

        collected: dict[int, list[list[float]]] = {}
        batch: list[tuple[int, str]] = []
        batch_tokens = 0
        for unit in units:
            est = _estimate_tokens(unit[1])
            if batch and (
                len(batch) >= self._config.batch_size
                or batch_tokens + est > max_tokens
            ):
                await self._embed_units(batch, collected)
                batch = []
                batch_tokens = 0
            batch.append(unit)
            batch_tokens += est
        if batch:
            await self._embed_units(batch, collected)

        for i, vectors in collected.items():
            results[i] = _mean_vector(vectors) if len(vectors) > 1 else vectors[0]
        return results

    async def _embed_units(
        self,
        units: Sequence[tuple[int, str]],
        collected: dict[int, list[list[float]]],
    ) -> None:
        vectors = await self._embed_batch([text for _, text in units])
        for (index, _), vector in zip(units, vectors, strict=True):
            collected.setdefault(index, []).append(vector)

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single string. Raises if it has no embeddable content.

        Unlike batch embedding, there is nothing sensible to skip here: an
        empty query is a caller bug, not a bad page in a PDF. Goes through
        the same packing path so an oversized summary is split rather than
        rejected by the server.
        """
        cleaned = wire_safe(clean_text(text))
        if not is_embeddable(cleaned):
            raise ValueError(f"Nothing embeddable in input: {text!r:.80}")
        return (await self.embed([cleaned]))[0]

    # -- transport ---------------------------------------------------------

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                if self._config.backend == "ollama":
                    vectors = await self._post_ollama(batch)
                else:
                    vectors = await self._post_openai(batch)
                self._check_dimensions(vectors)
                return vectors
            except EmbeddingServerError as exc:
                # A 4xx/5xx with a message from the server is a configuration
                # problem, not a blip. Retrying it three times just delays the
                # error and buries it under retry warnings.
                longest = max((len(t) for t in batch), default=0)
                log.error(
                    "Embedding server rejected the request: %s\n"
                    "  endpoint : %s/v1/embeddings\n"
                    "  batch    : %d texts, longest %d chars (~%d tokens)\n"
                    "  model    : %s\n"
                    "  hint     : llama.cpp needs --embeddings and an explicit "
                    "--pooling (cls for bge-m3). _Map_base::at is a vocabulary "
                    "lookup miss: try a different GGUF conversion.",
                    exc, self._config.base_url.rstrip("/"),
                    len(batch), longest, longest // 4,
                    self._wire_model or "(not sent)",
                )
                raise
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                last_error = exc
                wait = 2**attempt
                log.warning(
                    "Embedding attempt %d failed (%s); retrying in %ds",
                    attempt + 1, exc, wait,
                )
                await asyncio.sleep(wait)
        raise RuntimeError(f"Embedding failed after 3 attempts: {last_error}")

    async def _post_openai(self, batch: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"input": batch}
        # Only send "model" when it names something the server actually knows.
        # llama-server in router mode does a map lookup on this and throws
        # _Map_base::at (a bare std::out_of_range, surfaced as an opaque 500)
        # for an unregistered name. Single-model servers ignore the field
        # entirely, so omitting it is the safe default.
        if self._wire_model:
            payload["model"] = self._wire_model

        resp = await self.client().post("/v1/embeddings", json=payload)
        if resp.status_code >= 400:
            raise EmbeddingServerError(resp.status_code, _body_message(resp))
        data = resp.json()["data"]
        # The spec does not promise response order, and some servers do reorder
        # under concurrency. Sort by index rather than trusting position.
        data.sort(key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in data]

    async def _post_ollama(self, batch: list[str]) -> list[list[float]]:
        resp = await self.client().post(
            "/api/embed", json={"model": self._config.model, "input": batch}
        )
        if resp.status_code >= 400:
            raise EmbeddingServerError(resp.status_code, _body_message(resp))
        return resp.json()["embeddings"]

    def _check_dimensions(self, vectors: list[list[float]]) -> None:
        for v in vectors:
            if len(v) != self._config.dimensions:
                raise ValueError(
                    f"Embedding server returned {len(v)} dimensions but config "
                    f"says {self._config.dimensions}. The vector(N) columns in "
                    f"the schema must match; changing this requires a re-index."
                )

    async def _resolve_model(self) -> None:
        """Reconcile the configured model name with what the server serves.

        Servers disagree about this field: vLLM and TEI require a name that
        matches exactly, llama-server ignores it when serving one model, and
        llama-server in router mode throws on an unknown one. Asking
        /v1/models removes the guesswork.
        """
        if self._config.backend != "openai":
            return

        try:
            resp = await self.client().get("/v1/models")
            resp.raise_for_status()
            ids = [m["id"] for m in resp.json().get("data", []) if m.get("id")]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            log.debug("Could not list models (%s); sending configured name", exc)
            return

        if not ids:
            self._wire_model = None
            return

        if self._config.model in ids:
            self._wire_model = self._config.model
            return

        # Exactly one model served: use its real name whatever it is called.
        if len(ids) == 1:
            log.warning(
                "Config says embedding.model=%r but the server serves %r. "
                "Using the server's name.", self._config.model, ids[0],
            )
            self._wire_model = ids[0]
            return

        raise RuntimeError(
            f"embedding.model={self._config.model!r} is not served by "
            f"{self._config.base_url}. Available: {', '.join(ids)}"
        )

    async def health_check(self) -> None:
        """Verify the server answers and the dimension matches, at startup
        rather than on the first document.

        The probe is deliberately two tokens long: if this fails, batch and
        context sizing are ruled out and the problem is the model or the
        server flags.
        """
        await self._resolve_model()
        vector = await self.embed_one("dimension probe")
        log.info(
            "Embedding backend %s at %s: model=%s dim=%d",
            self._config.backend, self._config.base_url,
            self._wire_model or "(unnamed, single-model server)", len(vector),
        )


# ---------------------------------------------------------------------------
# Asymmetric prefixes
#
# Some models are trained as asymmetric retrievers and expect different
# prefixes on queries and documents. Omitting them does not error; it silently
# degrades retrieval, which is the worst kind of bug to have here.
#
# bge-m3 is symmetric and needs none.
# ---------------------------------------------------------------------------

def query_text_for(text: str, model: str) -> str:
    name = model.lower()
    if "nomic-embed" in name:
        return f"search_query: {text}"
    if "e5" in name:
        return f"query: {text}"
    if "lfm2" in name:
        return f"query: {text}"
    if "qwen3-embedding" in name:
        return (
            "Instruct: Given a search query, retrieve relevant passages\n"
            f"Query: {text}"
        )
    return text


def document_text_for(text: str, model: str) -> str:
    name = model.lower()
    if "nomic-embed" in name:
        return f"search_document: {text}"
    if "e5" in name:
        return f"passage: {text}"
    if "lfm2" in name:
        return f"document: {text}"
    return text
