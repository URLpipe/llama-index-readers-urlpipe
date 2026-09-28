# llama-index-readers-urlpipe

Load any web page into LlamaIndex as clean Markdown, ready to index: URLpipe opens it in real Chrome, so JavaScript-rendered content comes back complete, without the menus, footers and cookie banners.

## Install

```sh
pip install llama-index-readers-urlpipe
```

Python 3.10 or later. It builds on [`urlpipe`](https://pypi.org/project/urlpipe/), the official Python client, which is installed with it.

## Get an API key

Create a project at [urlpipe.dev](https://urlpipe.dev) and copy its API key. The Free plan gives you 1,000 credits a month, no card needed; reading a page as Markdown costs 1 credit, and reading it again within 7 days costs nothing.

```sh
export URLPIPE_API_KEY=your_project_key
```

Or pass it in: `UrlpipeReader(api_key="...")`.

## Load pages and ask questions about them

```python
from llama_index.core import VectorStoreIndex
from llama_index.readers.urlpipe import UrlpipeReader

reader = UrlpipeReader()  # reads URLPIPE_API_KEY
documents = reader.load_data([
    "https://urlpipe.dev/docs",
    "https://urlpipe.dev/pricing",
])

index = VectorStoreIndex.from_documents(documents)
answer = index.as_query_engine().query("How many credits does a Markdown request cost?")
print(answer)
```

`VectorStoreIndex` uses the embedding model and LLM set in `llama_index.core.Settings`, OpenAI by default (`pip install llama-index` brings those in and reads `OPENAI_API_KEY`).

You get one `Document` per URL, in the order you gave them:

| Field | What it holds |
|---|---|
| `text` | the page as Markdown (or HTML, or a summary; see below) |
| `id_` | the same id every time you load that URL with that operation, so an ingestion pipeline's docstore spots a page it already has |
| `metadata["url"]` | the URL you asked for |
| `metadata["operation"]` | `"markdown"`, `"html"` or `"summarize"` |
| `metadata["token"]` | the request's token: fetch the same result again for free with `urlpipe.Client().result(token)` |
| `metadata["cache"]` | `"hit"` when a stored result was reused at no cost, `"miss"` when the page was fetched |

Only `url` goes to the embedding model and the LLM; the other keys stay out of the way.

For a long list of pages, `lazy_load_data` hands you each Document as soon as it is ready:

```python
for doc in reader.lazy_load_data(urls):
    print(doc.metadata["url"], len(doc.text))
```

## Async

`aload_data` fetches several pages at once, 4 at a time by default. Set `max_concurrency` to your plan's parallel request limit:

```python
import asyncio
from llama_index.readers.urlpipe import UrlpipeReader

async def main() -> None:
    reader = UrlpipeReader(max_concurrency=8)
    documents = await reader.aload_data(urls)
    print(len(documents))

asyncio.run(main())
```

## Operations

| `operation` | `Document.text` | Credits |
|---|---|---|
| `"markdown"` (default) | the main content as Markdown | 1 |
| `"html"` | the full HTML after JavaScript ran | 1 |
| `"summarize"` | an AI summary of the page, in Markdown | 17 |

```python
UrlpipeReader(operation="summarize")
```

## Options

```python
reader = UrlpipeReader(
    api_key="...",                # default: URLPIPE_API_KEY
    operation="markdown",
    max_age="1 hour",             # reuse a stored result younger than this; 0 always fetches
    page_options={"block_cookie_banners": True, "remove_selectors": [".promo"]},
    continue_on_failure=True,     # log a page that cannot be read and carry on
    max_concurrency=4,            # pages aload_data fetches at a time
    timeout=90,                   # seconds per HTTP request
    max_retries=2,                # retries on connection errors, rate limits and 5xx
)
```

`page_options` is sent to the API as given; the [page options docs](https://urlpipe.dev/docs/page-options) list what it takes.

## When a page cannot be read

By default the first page that fails raises the client's error, a subclass of `urlpipe.UrlpipeError`:

```python
import urlpipe

try:
    documents = reader.load_data(urls)
except urlpipe.AnalysisFailedError as error:
    print("Could not read the page:", error.message)
```

With `continue_on_failure=True` the reader logs a warning for that page and returns the rest. A wrong API key, an unconfirmed email address or a spent monthly allowance still raises, since every other page would fail the same way. A failed page costs nothing.

## Links

- LlamaIndex guide: https://urlpipe.dev/integrations/llamaindex
- API docs: https://urlpipe.dev/docs
- Python client: https://urlpipe.dev/docs/client-libraries
- MCP server, to use URLpipe from AI assistants: https://github.com/URLpipe/mcp

## License

MIT © Aliat Partner S.L.
