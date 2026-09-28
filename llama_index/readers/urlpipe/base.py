"""``UrlpipeReader``: web pages as LlamaIndex Documents, through the URLpipe API."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Union

import urlpipe
from llama_index.core.bridge.pydantic import Field, model_validator
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

from ._version import __version__

logger = logging.getLogger(__name__)

ReaderOperation = Literal["markdown", "html", "summarize"]
"""The operations whose result is page text a Document can hold."""

# Named in every request's User-Agent, in front of the client's own:
# "llama-index-readers-urlpipe/0.1.1 urlpipe-python/0.2.0".
USER_AGENT = f"llama-index-readers-urlpipe/{__version__}"

# Errors about the account rather than the URL: every other URL would fail the
# same way, so they are raised even with continue_on_failure.
_ACCOUNT_ERRORS = (
    urlpipe.ConfigurationError,
    urlpipe.AuthenticationError,
    urlpipe.EmailUnverifiedError,
    urlpipe.QuotaExceededError,
)

# Metadata that identifies a request but says nothing about the page, kept out
# of what the embedding model and the LLM see.
_BOOKKEEPING_KEYS = ["operation", "token", "cache"]


class UrlpipeReader(BasePydanticReader):
    """Load web pages as Documents: clean Markdown, rendered HTML or a summary.

    URLpipe opens each page in real Chrome, so JavaScript-rendered content comes
    back complete. One Document per URL, in the order given.

    >>> reader = UrlpipeReader()  # reads URLPIPE_API_KEY
    >>> docs = reader.load_data(["https://example.com"])

    Args:
        api_key: a URLpipe project API key. Defaults to ``URLPIPE_API_KEY``.
        operation: ``"markdown"`` (the main content), ``"html"`` (the HTML after
            JavaScript ran) or ``"summarize"`` (an AI summary, as Markdown).
        max_age: reuse a stored result younger than this, free of charge:
            seconds or a duration such as ``"1 hour"``; ``0`` always fetches.
            ``None`` keeps the API default of 7 days.
        page_options: how the page is loaded and cleaned, passed through as
            given (for example ``{"block_cookie_banners": True}``).
        continue_on_failure: log a URL that fails and skip it, instead of
            raising. Errors about the account itself (a wrong API key, no
            credits left) are raised either way.
        max_concurrency: how many URLs ``aload_data`` fetches at a time. Keep
            it at or below your plan's parallel request limit.
        base_url, timeout, max_retries, wait_timeout: passed to
            ``urlpipe.Client`` / ``urlpipe.AsyncClient``.
    """

    is_remote: bool = True
    api_key: Optional[str] = Field(
        default=None,
        exclude=True,
        repr=False,
        description="URLpipe project API key. Defaults to URLPIPE_API_KEY. Never serialised.",
    )
    operation: ReaderOperation = Field(
        default="markdown",
        description="What each Document holds: markdown, html or summarize.",
    )
    max_age: Optional[Union[int, str]] = Field(
        default=None,
        description="Reuse a stored result younger than this (seconds or a duration). 0 always fetches.",
    )
    page_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Page loading and cleaning options, sent to the API as given.",
    )
    continue_on_failure: bool = Field(
        default=False,
        description="Log and skip a URL that fails instead of raising.",
    )
    max_concurrency: int = Field(
        default=4,
        ge=1,
        description="How many URLs aload_data fetches at a time.",
    )
    base_url: str = Field(default="https://urlpipe.dev", description="The URLpipe API origin.")
    timeout: float = Field(default=90.0, gt=0, description="Seconds per HTTP request.")
    max_retries: int = Field(default=2, ge=0, description="Automatic retries per request.")
    wait_timeout: float = Field(
        default=300.0,
        gt=0,
        description="How long a request that outlives the API's 60 s window keeps polling.",
    )

    @model_validator(mode="after")
    def _require_api_key(self) -> "UrlpipeReader":
        if not (self.api_key or os.environ.get("URLPIPE_API_KEY")):
            raise urlpipe.ConfigurationError(
                "No URLpipe API key: pass api_key=... or set the URLPIPE_API_KEY "
                "environment variable. Each project's key is in its dashboard settings."
            )
        return self

    @classmethod
    def class_name(cls) -> str:
        return "UrlpipeReader"

    # -- loading ----------------------------------------------------------- #

    def lazy_load_data(self, urls: Union[str, Sequence[str]]) -> Iterator[Document]:
        """Fetch each URL in turn and yield its Document as soon as it is ready."""
        targets = _as_list(urls)
        if not targets:
            return
        with urlpipe.Client(**self._client_kwargs()) as client:
            for url in targets:
                try:
                    response = self._fetch(client, url)
                except urlpipe.UrlpipeError as error:
                    self._skip_or_raise(url, error)
                    continue
                yield self._document(url, response)

    def load_data(self, urls: Union[str, Sequence[str]]) -> List[Document]:
        """Fetch each URL and return one Document per URL, in order."""
        return list(self.lazy_load_data(urls))

    async def aload_data(self, urls: Union[str, Sequence[str]]) -> List[Document]:
        """Fetch the URLs concurrently, ``max_concurrency`` at a time, and return
        one Document per URL in the order given."""
        targets = _as_list(urls)
        if not targets:
            return []
        limit = asyncio.Semaphore(self.max_concurrency)
        async with urlpipe.AsyncClient(**self._client_kwargs()) as client:

            async def load(url: str) -> Optional[Document]:
                async with limit:
                    try:
                        response = await self._afetch(client, url)
                    except urlpipe.UrlpipeError as error:
                        self._skip_or_raise(url, error)
                        return None
                return self._document(url, response)

            tasks = [asyncio.ensure_future(load(url)) for url in targets]
            try:
                documents = await asyncio.gather(*tasks)
            except BaseException:
                # One URL failed: stop the rest before the connection pool closes.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        return [doc for doc in documents if doc is not None]

    async def alazy_load_data(self, urls: Union[str, Sequence[str]]) -> Iterable[Document]:
        """The same as :meth:`aload_data`: URLs are fetched concurrently."""
        return await self.aload_data(urls)

    # -- plumbing ---------------------------------------------------------- #

    def _client_kwargs(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "wait_timeout": self.wait_timeout,
            "user_agent": USER_AGENT,
        }

    def _fetch(self, client: urlpipe.Client, url: str) -> urlpipe.Response[str]:
        method = {
            "markdown": client.markdown,
            "html": client.html,
            "summarize": client.summarize,
        }[self.operation]
        return method(url, max_age=self.max_age, page_options=self.page_options)

    async def _afetch(self, client: urlpipe.AsyncClient, url: str) -> urlpipe.Response[str]:
        method = {
            "markdown": client.markdown,
            "html": client.html,
            "summarize": client.summarize,
        }[self.operation]
        return await method(url, max_age=self.max_age, page_options=self.page_options)

    def _document(self, url: str, response: urlpipe.Response[str]) -> Document:
        metadata: Dict[str, Any] = {"url": url, "operation": self.operation}
        # Missing values are left out: several vector stores refuse None.
        if response.token is not None:
            metadata["token"] = response.token
        if response.meta.cache is not None:
            metadata["cache"] = response.meta.cache
        return Document(
            id_=document_id(url, self.operation),
            text=response.data or "",
            metadata=metadata,
            excluded_embed_metadata_keys=list(_BOOKKEEPING_KEYS),
            excluded_llm_metadata_keys=list(_BOOKKEEPING_KEYS),
        )

    def _skip_or_raise(self, url: str, error: urlpipe.UrlpipeError) -> None:
        if not self.continue_on_failure or isinstance(error, _ACCOUNT_ERRORS):
            raise error
        logger.warning("URLpipe could not load %s, skipping it: %s", url, error)


def document_id(url: str, operation: str) -> str:
    """The Document id for ``url`` read with ``operation``.

    A UUID derived from both, so loading the same page again gives the same id
    (an ingestion pipeline's docstore can then spot the duplicate) and every
    vector store accepts it.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"urlpipe:{operation}:{url}"))


def _as_list(urls: Union[str, Sequence[str]]) -> List[str]:
    return [urls] if isinstance(urls, str) else list(urls)

