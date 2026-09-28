"""LlamaIndex reader for URLpipe: web pages as Documents, rendered in real Chrome.

    from llama_index.readers.urlpipe import UrlpipeReader

    docs = UrlpipeReader().load_data(["https://example.com"])  # reads URLPIPE_API_KEY

Docs: https://urlpipe.dev/integrations/llamaindex
"""

from ._version import __version__
from .base import ReaderOperation, UrlpipeReader, document_id

__all__ = ["ReaderOperation", "UrlpipeReader", "__version__", "document_id"]
