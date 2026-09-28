"""A real local HTTP server standing in for the URLpipe API.

It answers each analysis request by the ``url`` in its JSON body, records every
request, and counts how many were in flight at once.
"""

from __future__ import annotations

import json
import threading
import time
import zlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Optional, Union

import pytest

from urlpipe import _core

from llama_index.readers.urlpipe import UrlpipeReader

API_KEY = "test_key_123"


@dataclass
class Recorded:
    method: str
    path: str
    headers: Dict[str, str]
    body: bytes

    @property
    def json(self) -> Any:
        return json.loads(self.body)


@dataclass
class Reply:
    status: int = 200
    body: Union[str, Any] = ""
    headers: Dict[str, str] = field(default_factory=dict)


def page_headers(url: str, cache: str = "miss") -> Dict[str, str]:
    """The headers a completed sync request comes back with."""
    return {
        "X-Result-Token": f"tok_{zlib.crc32(url.encode())}",
        "X-Cache": cache,
        "X-Quota-Cost": "0" if cache == "hit" else "1",
        "X-Quota-Limit": "1000",
        "X-Quota-Remaining": "943",
    }


def error_reply(status: int, error: str, message: Optional[str] = None) -> Reply:
    """An API error body: ``{"error": code_or_sentence, "message": ...}``."""
    body: Dict[str, Any] = {"error": error}
    if message is not None:
        body["message"] = message
    return Reply(status, body)


class StubServer:
    """Answers an analysis by the ``url`` in its body: the Reply set in
    ``pages``, else the page's text built from the URL and operation."""

    def __init__(self) -> None:
        self.pages: Dict[str, Reply] = {}
        self.requests: List[Recorded] = []
        self.delay = 0.0
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                recorded = Recorded(
                    self.command,
                    self.path,
                    {k.lower(): v for k, v in self.headers.items()},
                    self.rfile.read(length) if length else b"",
                )
                with stub._lock:
                    stub.requests.append(recorded)
                    stub.in_flight += 1
                    stub.max_in_flight = max(stub.max_in_flight, stub.in_flight)
                try:
                    time.sleep(stub.delay)
                    self._send(stub.reply_for(recorded))
                finally:
                    with stub._lock:
                        stub.in_flight -= 1

            def _send(self, reply: Reply) -> None:
                if isinstance(reply.body, str):
                    payload, content_type = reply.body.encode(), "text/plain; charset=utf-8"
                else:
                    payload = json.dumps(reply.body).encode()
                    content_type = "application/json; charset=utf-8"
                self.send_response(reply.status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                for name, value in reply.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(payload)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def reply_for(self, recorded: Recorded) -> Reply:
        url = recorded.json["url"]
        if url in self.pages:
            return self.pages[url]
        operation = recorded.path.lstrip("/")
        return Reply(200, page_text(url, operation), page_headers(url))

    def requests_for(self, url: str) -> List[Recorded]:
        return [r for r in self.requests if r.json["url"] == url]

    def reset(self) -> None:
        self.pages.clear()
        self.requests.clear()
        self.delay = 0.0
        self.in_flight = 0
        self.max_in_flight = 0

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def page_text(url: str, operation: str) -> str:
    """What the stub answers for ``url`` read with ``operation``."""
    if operation == "html":
        return f"<html><body><h1>{url}</h1></body></html>"
    if operation == "summarize":
        return f"A summary of {url}."
    return f"# {url}\n\nThe page's main content."


@pytest.fixture(scope="session")
def _server() -> Iterator[StubServer]:
    server = StubServer()
    yield server
    server.close()


@pytest.fixture
def server(_server: StubServer) -> Iterator[StubServer]:
    _server.reset()
    yield _server
    _server.reset()


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """No backoff between the client's retries, so error tests run instantly."""
    monkeypatch.setattr(_core, "BACKOFF_BASE", 0.0)
    monkeypatch.setattr(_core, "RATE_LIMIT_FALLBACK", 0.0)


@pytest.fixture
def make_reader(server: StubServer) -> Any:
    """A factory for a UrlpipeReader pointed at the stub server."""

    def make(**overrides: Any) -> UrlpipeReader:
        kwargs: Dict[str, Any] = {"api_key": API_KEY, "base_url": server.url}
        kwargs.update(overrides)
        return UrlpipeReader(**kwargs)

    return make
