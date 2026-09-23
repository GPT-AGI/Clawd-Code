# Webhook processing requirements

- `WebhookEvent.tenant_id` and `event_id` must be non-empty and `sequence` must
  be a positive integer; invalid events raise `ValueError` before the handler.
- Idempotency is scoped by tenant and event ID. The first successful delivery
  returns `processed`; later deliveries return `duplicate` without invoking the
  handler.
- Different event IDs with a sequence less than or equal to the last successful
  sequence for that tenant return `stale` without invoking the handler.
- Handler exceptions propagate. A failed event must remain retryable and must
  not advance ordering state.
- Concurrent calls for the same tenant/event must invoke the handler exactly
  once. All other callers return `duplicate` after the successful call.
- State for one tenant must never suppress or reorder another tenant.
- `snapshot()` returns a detached mapping keyed by tenant with sorted processed
  IDs and the last successful sequence. Callers cannot mutate processor state
  through the returned value.
