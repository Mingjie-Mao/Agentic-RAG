Hotfix: close authz bypass on streaming chat route (deny-by-default + regression test)

Incident: See Slack #incidents thread “authz regression on streaming endpoint” (sources/slack/incidents/1740151207-authz-regression-on-streaming-endpoint.json). During RBAC v2 rollout, we discovered that one streaming code path could start an SSE response without having a permission check applied. This was not an intentional allowlist; it was a routing/middleware gap that effectively bypassed deny-by-default for a specific handler. Impact: unauthorized callers could receive streamed tokens on the affected route if they had valid authentication but lacked the required permission; audit logs still captured request metadata but did not record an authz decision for these responses because evaluation never ran.

Root cause: The streaming handler for chat completions registers under a separate router group (legacy streaming mux) which did not have the gateway authz middleware attached. Non-streaming requests hit the standard stack and were correctly blocked. Streaming requests (Accept: text/event-stream or stream=true) were routed to the legacy mux, which performed authentication but skipped the permission enforcement hook.

Fix: (1) Attach the RBAC v2 permission middleware to the legacy streaming mux/router group to ensure deny-by-default applies consistently. (2) Add an explicit preflight authz check in the streaming handler before writing headers/starting the event stream, to prevent partial responses if middleware ordering is changed in the future. (3) Ensure we emit an authz decision audit record for streaming denies (no bytes written) so that evidence remains complete.

Required permission: redwood.inference.chat.completions.stream (existing permission name aligned with ADR-014 naming conventions; no new permission added in this hotfix to keep scope minimal).

Regression tests: Added an integration test that calls the streaming endpoint with a token missing the streaming permission and asserts: HTTP 403, no SSE frames, and denial reason code is set (admin debug header only when enabled). The test also covers the “headers already sent” edge by verifying the authz check runs before any write/flush.

Rollout: This is a hotfix intended for immediate merge and deployment to Hosted first, followed by Dedicated/Private release branches via normal backport process. No customer-visible behavior change for authorized callers; unauthorized callers now receive consistent 403 responses.

Notes: This PR is intentionally small and does not change role templates or permission inventory. Follow-ups for expanding coverage gates across all streaming routes are tracked in ENG-18492.
Kaitlyn Nguyen (review): “Can we guarantee we don’t write any bytes before the decision? We’ve had ‘headers already sent’ issues on SSE before.” Amara Diallo (author): “Yes—moved the authz evaluation ahead of any WriteHeader/Flush and added a guard that panics in tests if we write prior to decision in this handler.”

Marcus Lin (review): “Please add explicit regression coverage. Also, confirm this doesn’t introduce an allowlist by accident.” Amara: “Added integration test asserting 403 + no SSE frames. No allowlist; middleware remains deny-by-default, handler check is a belt-and-suspenders preflight.”

Jordan Lee (review): “Do we log authz decisions on denied streaming requests? We need evidence for SOC2 and to debug pilot reports.” Amara: “Added audit emit for deny path; includes action + scope + policy version when available. This mirrors non-streaming behavior.”

Leila Farouk (QA): “Integration test looks good. Can we make it resilient to timing?” Amara: “Test asserts on response headers/body only; no timing-sensitive reads. Also validates that the connection closes without ‘data:’ frames.”

Approvals: Kaitlyn Nguyen ✅, Marcus Lin ✅, Jordan Lee ✅, Leila Farouk ✅. Merged via squash for hotfix deployment.