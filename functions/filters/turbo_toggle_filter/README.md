# Turbo Toggle Filter

Enables OpenAI's priority processing tier by setting `service_tier` to `priority` on supported requests.

## Behavior

- Applies only to OpenAI model IDs (prefix `openai_responses.` or `openai.`).
- For `openai_responses.*` models, also sets `features.openai_responses.service_tier="priority"` so the manifold request replaces `auto` with `priority`.
- For other providers/models, the filter no-ops and emits a status message (when an event emitter is available).

## Valves

- `priority`: Controls filter execution order (lower runs first).
