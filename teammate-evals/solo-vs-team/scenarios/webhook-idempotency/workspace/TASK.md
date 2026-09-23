# Webhook processor repair

Repair `WebhookProcessor` in `src/webhooks.py`. Production deliveries can be
duplicated, concurrent, out of order, or retried after handler failures. Keep
the public API intact and make processing deterministic and tenant-safe.
