"""What the reader sends to the API."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import urlpipe

from llama_index.readers.urlpipe import UrlpipeReader, __version__

from .conftest import API_KEY, StubServer

URL = "https://example.com/docs"


@pytest.mark.parametrize("operation", ["markdown", "html", "summarize"])
def test_posts_the_url_to_the_operation_and_waits(
    make_reader: Any, server: StubServer, operation: str
) -> None:
    make_reader(operation=operation).load_data([URL])

    (request,) = server.requests
    assert request.method == "POST"
    assert request.path == f"/{operation}"
    assert request.json == {"url": URL, "sync": True}


def test_sends_max_age_and_page_options_as_given(make_reader: Any, server: StubServer) -> None:
    options = {"block_cookie_banners": True, "remove_selectors": [".promo", "#related"]}
    make_reader(max_age="1 hour", page_options=options).load_data([URL])

    (request,) = server.requests
    assert request.json == {
        "url": URL,
        "sync": True,
        "max_age": "1 hour",
        "page_options": options,
    }


def test_max_age_zero_is_sent(make_reader: Any, server: StubServer) -> None:
    make_reader(max_age=0).load_data([URL])
    assert server.requests[0].json["max_age"] == 0


def test_authenticates_with_the_api_key(make_reader: Any, server: StubServer) -> None:
    make_reader().load_data([URL])
    assert server.requests[0].headers["authorization"] == f"Bearer {API_KEY}"


def test_reads_the_api_key_from_the_environment(
    server: StubServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("URLPIPE_API_KEY", "env_key_456")
    UrlpipeReader(base_url=server.url).load_data([URL])
    assert server.requests[0].headers["authorization"] == "Bearer env_key_456"


def test_user_agent_names_the_client_and_this_integration(
    make_reader: Any, server: StubServer
) -> None:
    make_reader().load_data([URL])
    asyncio.run(make_reader().aload_data([URL]))

    expected = f"llama-index-readers-urlpipe/{__version__} urlpipe-python/{urlpipe.__version__}"
    assert [r.headers["user-agent"] for r in server.requests] == [expected, expected]


def test_one_request_per_url_in_order(make_reader: Any, server: StubServer) -> None:
    urls = [f"https://example.com/{n}" for n in range(3)]
    make_reader().load_data(urls)
    assert [r.json["url"] for r in server.requests] == urls


def test_a_single_url_string_is_one_url(make_reader: Any, server: StubServer) -> None:
    docs = make_reader().load_data(URL)
    assert len(docs) == 1
    assert [r.json["url"] for r in server.requests] == [URL]


def test_no_urls_sends_nothing(make_reader: Any, server: StubServer) -> None:
    assert make_reader().load_data([]) == []
    assert asyncio.run(make_reader().aload_data([])) == []
    assert server.requests == []
