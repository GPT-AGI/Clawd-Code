# Configuration requirements

`migrate_config(raw, env)` returns this canonical version 3 shape:

```python
{
    "version": 3,
    "service": {"url": "https://...", "auth": {"token": "..."}},
    "retry": {"max_attempts": 3},
    "features": {...},
}
```

- Version 1 is selected when `version` is absent or `1`; fields are `endpoint`,
  `token`, `retries`, and optional `features`.
- Version 2 fields are `service_url`, `credentials.api_token`, `retry_count`,
  and optional `features`.
- Version 3 already uses the canonical shape. Return a deep copy, never an alias.
- Interpolate `${NAME}` placeholders recursively in string values using the
  supplied `env` mapping. Multiple placeholders in one string are supported;
  a missing name raises `ValueError` naming that variable.
- The canonical service URL must use HTTPS and have a host. Token must be a
  non-empty string. Retry count must be an integer from 0 through 10; booleans
  are not integers for this purpose. Features must be a mapping.
- Unsupported versions and malformed nested structures raise `ValueError`.
- Neither input mapping nor nested values may be mutated.

`public_config(config)` returns a deep detached copy where values under keys
`token`, `password`, `secret`, and `api_key` are replaced with `[REDACTED]` at
every nesting depth, including dictionaries inside lists.
