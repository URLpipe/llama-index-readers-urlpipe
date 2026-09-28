# Changelog

## 0.1.0 — 2026-09-28

- `UrlpipeReader`: load web pages into LlamaIndex as Markdown, rendered HTML or a
  summary, one `Document` per URL, with `load_data`, `lazy_load_data` and
  `aload_data` (concurrent, `max_concurrency` at a time).
- Stable Document ids per URL and operation; metadata `url`, `operation`,
  `token`, `cache`.
- `continue_on_failure` to log and skip pages that cannot be read.
