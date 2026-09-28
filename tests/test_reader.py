"""The reader as a LlamaIndex component."""

from __future__ import annotations

import pytest
from llama_index.core.readers.base import BasePydanticReader

from llama_index.readers.urlpipe import UrlpipeReader


def test_is_a_remote_pydantic_reader() -> None:
    reader = UrlpipeReader(api_key="secret_key")
    assert isinstance(reader, BasePydanticReader)
    assert reader.is_remote is True
    assert UrlpipeReader.class_name() == "UrlpipeReader"


def test_defaults() -> None:
    reader = UrlpipeReader(api_key="k")
    assert reader.operation == "markdown"
    assert reader.max_age is None
    assert reader.page_options is None
    assert reader.continue_on_failure is False
    assert reader.max_concurrency == 4


def test_the_api_key_is_never_serialised_or_shown() -> None:
    reader = UrlpipeReader(api_key="secret_key", operation="html", max_age=3600)
    assert "secret_key" not in reader.to_json()
    assert "secret_key" not in repr(reader)
    assert "api_key" not in reader.to_dict()


def test_round_trips_through_to_dict_with_the_key_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("URLPIPE_API_KEY", "env_key")
    reader = UrlpipeReader(operation="summarize", max_age="1 day", page_options={"block_ads": True})
    again = UrlpipeReader.from_dict(reader.to_dict())
    assert again.operation == "summarize"
    assert again.max_age == "1 day"
    assert again.page_options == {"block_ads": True}
