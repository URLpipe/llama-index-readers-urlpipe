"""aload_data fetches concurrently, never more than max_concurrency at a time."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from .conftest import StubServer, error_reply

URLS = [f"https://example.com/page-{n}" for n in range(10)]


@pytest.mark.parametrize("bound", [1, 2, 4])
def test_never_more_than_max_concurrency_in_flight(
    make_reader: Any, server: StubServer, bound: int
) -> None:
    server.delay = 0.05
    docs = asyncio.run(make_reader(max_concurrency=bound).aload_data(URLS))

    assert len(docs) == len(URLS)
    assert server.max_in_flight == bound


def test_default_bound_is_4(make_reader: Any, server: StubServer) -> None:
    server.delay = 0.05
    reader = make_reader()
    asyncio.run(reader.aload_data(URLS))

    assert reader.max_concurrency == 4
    assert server.max_in_flight == 4


def test_runs_concurrently(make_reader: Any, server: StubServer) -> None:
    server.delay = 0.2
    started = time.monotonic()
    asyncio.run(make_reader(max_concurrency=4).aload_data(URLS[:4]))
    assert time.monotonic() - started < 0.6  # one after another would take 0.8 s


def test_documents_keep_the_order_given(make_reader: Any, server: StubServer) -> None:
    server.delay = 0.01
    docs = asyncio.run(make_reader(max_concurrency=5).aload_data(URLS))
    assert [doc.metadata["url"] for doc in docs] == URLS


def test_sync_load_is_one_at_a_time(make_reader: Any, server: StubServer) -> None:
    server.delay = 0.01
    make_reader(max_concurrency=4).load_data(URLS[:3])
    assert server.max_in_flight == 1


def test_a_failure_stops_the_other_requests(make_reader: Any, server: StubServer) -> None:
    import urlpipe

    server.delay = 0.05
    server.pages[URLS[0]] = error_reply(422, "The page could not be analysed.")
    with pytest.raises(urlpipe.AnalysisFailedError):
        asyncio.run(make_reader(max_concurrency=2).aload_data(URLS))

    # Only the first wave started; the rest were cancelled rather than sent.
    assert len(server.requests) < len(URLS)


def test_zero_concurrency_is_refused(make_reader: Any) -> None:
    with pytest.raises(ValueError):
        make_reader(max_concurrency=0)
