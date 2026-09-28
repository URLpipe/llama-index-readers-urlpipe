"""An optional smoke test against the real API. Runs only when URLPIPE_API_KEY
is set; it spends at most two credits (none on a cache hit)."""

from __future__ import annotations

import asyncio
import os

import pytest

from llama_index.readers.urlpipe import UrlpipeReader

pytestmark = pytest.mark.skipif(
    not os.environ.get("URLPIPE_API_KEY"), reason="set URLPIPE_API_KEY to run the live test"
)


def test_loads_example_com_as_markdown() -> None:
    (doc,) = UrlpipeReader().load_data(["https://example.com"])
    assert "Example Domain" in doc.text
    assert doc.metadata["url"] == "https://example.com"
    assert doc.metadata["token"]
    assert doc.metadata["cache"] in {"hit", "miss"}


def test_loads_asynchronously() -> None:
    docs = asyncio.run(UrlpipeReader().aload_data(["https://example.com", "https://example.org"]))
    assert len(docs) == 2
    assert all("Example Domain" in doc.text for doc in docs)
