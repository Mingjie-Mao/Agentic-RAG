SSE gap-healer: mid-event rescue and ordering preservation for interleaved frames

Summary: Introduces a 'gap-healer' reconciliation pass in the SSE framing layer and a lightweight sequence-anchor heuristic to recover mid-event partial payloads, prevent duplicate delimiters, and maintain strict event ordering when heartbeats or toolcall frames interleave with content frames. This is targeted at the OpenAI-compat streaming surface (completion/chat/function-call paths) and addresses a class of customer-visible issues where partial JSON objects were emitted or tool/function-call sentinels were observed out-of-order during high-concurrency backpressure.

Motivation and context: We continue to see incidents where network/backpressure-induced splits cause the SSE parser to emit incomplete JSON chunks or duplicate trailing delimiters, which in turn breaks clients (especially OpenAI-compat consumers and SDKs that assume atomic JSON spans). Additionally, we observed ordering anomalies when heartbeat or control frames were merged into the same reassembly window as content, causing tool/function-call boundaries to be observed out of order.

What this PR does (high level):
- Adds a gap-healer reconciliation pass that runs on assembled frame windows prior to emitter stage. The pass detects small sequence gaps or delimiter-loss patterns and attempts conservative repair (stitch prefixes/suffixes, reframe partial JSON) using an anchor fingerprint (last-closed-brace + seq-id heuristic).
- Introduces a sequence-anchor (seq_anchor) tag attached to SSE windows to prevent out-of-order emission when control frames (heartbeats/tool-sentinels) interleave. Emission waits until anchors are reconciled or a bounded timeout elapses.
- Strengthens delimiter normalization so duplicate '\n\n' or trailing 'data: ' markers are collapsed consistently; avoids emitting more than one terminating delimiter per event.
- Adds targeted unit and integration tests that simulate split UTF-8 codepoints and interleaved heartbeat/tool frames, plus a small e2e harness run with the OpenAI-compat test client.

Design notes and tradeoffs:
- Repair logic is intentionally conservative: we only attempt mid-event stitching when the anchor fingerprint matches and the gap size is below the configured threshold (default 64 bytes). Larger gaps fall back to safe truncation + error telemetry to avoid inventing user data.
- The seq_anchor wait window is short (configurable; default 50ms) to avoid adding significant latency to stable streams; flows that exceed the window fall back to the existing fast-path emitter.
- This avoids a large refactor of the streaming state machine but provides a safety net for most customer-visible corruption cases.

Checklist:
- [x] Add gap-healer pass implementation
- [x] Add seq_anchor propagation through assembler and emitter
- [x] Normalize delimiter emission in emitter
- [x] Unit tests for fragmentation and UTF-8 split
- [x] Integration test with OpenAI-compat client (simulated fragments)
- [x] Perf microbenchmark: <3% median latency impact on normal traffic (observed ~+1.2%)
- [x] CI green and manual canary in staging for 48h

Commits (summary):
- feat(sse): introduce gap-healer reconciliation pass and seq_anchor metadata
- fix(sse): collapse duplicate delimiters and avoid double-emitting terminators
- test(sse): add fragmentation/utf8 unit tests and integration harness
- perf(sse): add microbenchmark and guard timer for seq_anchor wait
- docs: annotate streaming contract and rollout instructions

Changed areas (high level):
- sse/parser/assembler.go -- new gap healing functions and seq_anchor propagation (approx +220 LOC)
- sse/emitter/emitter.go -- delimiter normalization and anchor-aware emission
- sse/tests/fragmentation_test.go -- unit/integration cases for splits and heartbeats
- openai_compat/compat_client_test.go -- simulated client against instrumented server
- docs/sse-framing.md -- operational notes and knob descriptions
- telemetry/metrics.go -- add sse.repair_attempts and sse.repair_success gauges

Small illustrative snippet (emitter normalizer):
- before: emitter.Write("\n\n") on terminal events could run multiple times leading to duplicate-delimiters
- after: emitter.Write(normalizeDelimiters(window)) ensures a single terminating delimiter and trims leading spaces on subsequent fragments

Review conversation (selected):
- Rohit Patel (2026-02-16): "Can we quantify how often the gap-healer actually kicks in in staging? Also ensure we emit telemetry when we choose not to repair (large gap)."
  - Author: "Added sse.repair_attempts/repaired_success metrics and a sample run in the PR description (staging median repairs: ~0.6% of streams)."
- Marta Lopez (2026-02-17): "Concerned about adding waits; any impact on tail latency?"
  - Author: "Benchmarked with contended stream microbench; default 50ms window adds ~1.2% median latency and <4% p95 in worst-case; documented knob for operators. Also added fallback to fast-path after timeout."
  - Marta: "OK, can we gate the default in staging flags for rollout?"
  - Author: "Flag added as sse.enable_gap_healer (default true in staging, false in prod for initial rollout)."
- Daniel Kim (2026-02-18): "Please add a test for toolcall sentinel ordering — saw a flaky client repro in the incident log."
  - Author: "Added compat_client_test.go which replays the incident scenario and asserts strict ordering; passes locally and in CI."

CI summary:
- github/actions/sse-unit (initial): lint failed due to missing docstring in assembler.go -> fixed in commit 2
- github/actions/sse-unit (re-run): pass
- github/actions/integration (OpenAI-compat harness): pass
- github/actions/perf: pass (microbench within threshold)
- flaky-checker: pass
- overall: pass

Rollout and deployment plan:
- Canary toggle: enable sse.enable_gap_healer on 10% of staging routers and run compatibility suite for 48h
- Monitor sse.repair_attempts, sse.repair_success, streamer.latency.* and emitter.duplicate_delimiters.*
- If metrics are stable, enable on 25% of production routers for 24h, then ramp to 100% over 3 days
- If any regression observed, revert via feature flag (instant) or roll back the deployment

Manual testing notes:
- Reproduced the original incident scenario locally by forcing a split inside a UTF-8 3-byte codepoint and interleaving a heartbeat frame; before this change client observed a truncated JSON; after change client receives reconstructed JSON and a single terminating delimiter.

Merge outcome: Merged via squash on 2026-02-20 by Rohit Patel after approvals from reviewers. Changelog entry added and docs updated. Backported to release/2026-02 patch with a small fix in emitter normalization.

Related: references incident INT-8092 for the original customer report.
Fixes streaming stability: improves SSE fragment reconciliation to avoid truncated/duplicate JSON emissions, preserves toolcall/event order under heartbeat interleaving, and reduces user-visible delimiter glitches under backpressure.