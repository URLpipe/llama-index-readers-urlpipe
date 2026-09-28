"""The Documents the reader returns."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from llama_index.core.schema import Document, MetadataMode

from llama_index.readers.urlpipe import document_id

from .conftest import Reply, StubServer, page_headers, page_text

URLS = ["https://example.com/a", "https://example.com/b"]


@pytest.mark.parametrize("operation", ["markdown", "html", "summarize"])
def test_one_document_per_url_holding_the_result(
    make_reader: Any, server: StubServer, operation: str
) -> None:
    docs = make_reader(operation=operation).load_data(URLS)

    assert all(isinstance(doc, Document) for doc in docs)
    assert [doc.text for doc in docs] == [page_text(url, operation) for url in URLS]


@pytest.mark.parametrize("operation", ["markdown", "html", "summarize"])
def test_metadata(make_reader: Any, server: StubServer, operation: str) -> None:
    (doc,) = make_reader(operation=operation).load_data(URLS[:1])

    assert doc.metadata == {
        "url": URLS[0],
        "operation": operation,
        "token": page_headers(URLS[0])["X-Result-Token"],
        "cache": "miss",
    }


def test_a_cache_hit_is_reported(make_reader: Any, server: StubServer) -> None:
    server.pages[URLS[0]] = Reply(200, "# Cached", page_headers(URLS[0], cache="hit"))
    (doc,) = make_reader().load_data(URLS[:1])
    assert doc.metadata["cache"] == "hit"


def test_missing_headers_are_left_out_of_metadata(make_reader: Any, server: StubServer) -> None:
    server.pages[URLS[0]] = Reply(200, "# Bare")
    (doc,) = make_reader().load_data(URLS[:1])
    assert doc.metadata == {"url": URLS[0], "operation": "markdown"}


def test_only_the_url_reaches_the_embedding_model_and_the_llm(
    make_reader: Any, server: StubServer
) -> None:
    (doc,) = make_reader().load_data(URLS[:1])
    for mode in (MetadataMode.EMBED, MetadataMode.LLM):
        assert doc.get_metadata_str(mode) == f"url: {URLS[0]}"


def test_ids_are_stable_per_url_and_operation(make_reader: Any, server: StubServer) -> None:
    first = [doc.id_ for doc in make_reader().load_data(URLS)]
    again = [doc.id_ for doc in make_reader().load_data(URLS)]
    html = [doc.id_ for doc in make_reader(operation="html").load_data(URLS)]

    assert first == again == [document_id(url, "markdown") for url in URLS]
    assert len(set(first)) == 2
    assert set(first).isdisjoint(html)
    uuid.UUID(first[0])  # a UUID, which every vector store accepts


def test_lazy_load_yields_each_document_as_it_is_fetched(
    make_reader: Any, server: StubServer
) -> None:
    docs = make_reader().lazy_load_data(URLS)

    first = next(iter(docs))
    assert first.metadata["url"] == URLS[0]
    assert len(server.requests) == 1
    assert [doc.metadata["url"] for doc in docs] == URLS[1:]


def test_async_load_returns_the_same_documents(make_reader: Any, server: StubServer) -> None:
    reader = make_reader()
    sync_docs = reader.load_data(URLS)
    async_docs = asyncio.run(reader.aload_data(URLS))
    lazy_docs = asyncio.run(reader.alazy_load_data(URLS))

    for docs in (async_docs, list(lazy_docs)):
        assert [(d.id_, d.text, d.metadata) for d in docs] == [
            (d.id_, d.text, d.metadata) for d in sync_docs
        ]
