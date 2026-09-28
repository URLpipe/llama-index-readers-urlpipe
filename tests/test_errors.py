"""Failed URLs: raised by default, logged and skipped with continue_on_failure."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List

import pytest
import urlpipe
from llama_index.core.schema import Document

from llama_index.readers.urlpipe import UrlpipeReader

from .conftest import Reply, StubServer, error_reply

GOOD = "https://example.com/good"
BAD = "https://example.com/bad"
ALSO_GOOD = "https://example.com/also-good"
URLS = [GOOD, BAD, ALSO_GOOD]

Loader = Callable[[UrlpipeReader, List[str]], List[Document]]


def load_sync(reader: UrlpipeReader, urls: List[str]) -> List[Document]:
    return reader.load_data(urls)


def load_lazy(reader: UrlpipeReader, urls: List[str]) -> List[Document]:
    return list(reader.lazy_load_data(urls))


def load_async(reader: UrlpipeReader, urls: List[str]) -> List[Document]:
    return asyncio.run(reader.aload_data(urls))


LOADERS = pytest.mark.parametrize("load", [load_sync, load_lazy, load_async])

PER_URL_FAILURES = [
    (error_reply(422, "The page could not be analysed: net::ERR_NAME_NOT_RESOLVED"), urlpipe.AnalysisFailedError),
    (error_reply(422, "invalid_url", "The url is not a valid http(s) URL."), urlpipe.InvalidRequestError),
    (error_reply(500, "internal_error"), urlpipe.ServerError),
]


@LOADERS
@pytest.mark.parametrize("reply, error_class", PER_URL_FAILURES)
def test_a_failed_url_raises_the_client_error(
    make_reader: Any, server: StubServer, load: Loader, reply: Reply, error_class: "type[urlpipe.UrlpipeError]"
) -> None:
    server.pages[BAD] = reply
    with pytest.raises(error_class) as raised:
        load(make_reader(), URLS)
    assert raised.value.status == reply.status


@LOADERS
def test_the_error_carries_the_api_message(
    make_reader: Any, server: StubServer, load: Loader
) -> None:
    server.pages[BAD] = error_reply(422, "invalid_url", "The url is not a valid http(s) URL.")
    with pytest.raises(urlpipe.InvalidRequestError) as raised:
        load(make_reader(), URLS)
    assert raised.value.code == "invalid_url"
    assert raised.value.message == "The url is not a valid http(s) URL."


@LOADERS
def test_continue_on_failure_skips_and_logs(
    make_reader: Any,
    server: StubServer,
    load: Loader,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server.pages[BAD] = error_reply(422, "The page could not be analysed.")
    with caplog.at_level(logging.WARNING, logger="llama_index.readers.urlpipe"):
        docs = load(make_reader(continue_on_failure=True), URLS)

    assert [doc.metadata["url"] for doc in docs] == [GOOD, ALSO_GOOD]
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert BAD in record.getMessage()
    assert "The page could not be analysed." in record.getMessage()


@LOADERS
def test_continue_on_failure_skips_connection_errors(
    server: StubServer, load: Loader, unused_port: int
) -> None:
    reader = UrlpipeReader(
        api_key="k", base_url=f"http://127.0.0.1:{unused_port}", max_retries=0,
        continue_on_failure=True,
    )
    assert load(reader, URLS) == []


@LOADERS
@pytest.mark.parametrize(
    "reply, error_class",
    [
        (error_reply(401, "unauthorized"), urlpipe.AuthenticationError),
        (error_reply(403, "email_unverified"), urlpipe.EmailUnverifiedError),
        (Reply(429, {"error": "quota_exceeded", "limit": 1000, "used": 1000, "needed": 1}), urlpipe.QuotaExceededError),
    ],
)
def test_account_errors_raise_even_with_continue_on_failure(
    make_reader: Any, server: StubServer, load: Loader, reply: Reply, error_class: type
) -> None:
    server.pages[BAD] = reply
    with pytest.raises(error_class):
        load(make_reader(continue_on_failure=True), URLS)


@LOADERS
def test_retries_a_server_error_before_failing(
    make_reader: Any, server: StubServer, load: Loader
) -> None:
    server.pages[BAD] = error_reply(503, "unavailable")
    with pytest.raises(urlpipe.ServerError):
        load(make_reader(max_retries=2), [BAD])
    assert len(server.requests_for(BAD)) == 3


def test_missing_api_key_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("URLPIPE_API_KEY", raising=False)
    with pytest.raises(urlpipe.ConfigurationError, match="URLPIPE_API_KEY"):
        UrlpipeReader()


def test_an_unknown_operation_is_refused() -> None:
    with pytest.raises(ValueError):
        UrlpipeReader(api_key="k", operation="screenshot")  # type: ignore[arg-type]


@pytest.fixture
def unused_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
