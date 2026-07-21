# Coupled peer task

Implement a tiny item normalization contract split across `src/protocol.py` and
`src/client.py`.

`normalize_item(raw)` must return a dictionary with a string `id` and a tuple of
trimmed, non-empty string `tags`. `build_request(raw)` must consume that exact
normalized representation and return `{"item": normalized}`.

The two components must agree on the interface; run the acceptance tests and
submit the final Git revision.
