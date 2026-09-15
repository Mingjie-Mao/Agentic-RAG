Introduce sparse invocation lattice and cohorted cost-blame resolver

Summary: Add a sparse invocation lattice aggregation pipeline and a cohorted cost-blame resolver that attributes token-level cost and latency drift to caller cohorts and callsites. Also introduces a low-overhead mini-trace anchor for sampled invocations and a Grafana 'Invocation Lattice' dashboard with cohort-level sparkline panels and a new alert: 'Cohorted Cost Drift (sample-size aware)'. Motivation: Long tail cost and slow drift across cohorts has been noisy for paging; we need an attribution surface that is 1) low-overhead in production, 2) resistant to high-cardinality explosion, and 3) useful for oncall and SRE runbooks. The lattice groups invocations by coarse prompt-shape, kernel-warmup bucket, and tenant callsite fingerprint to create stable cohorts and computes a weighted token entropy metric to detect meaningful drift. The cost-blame resolver then attributes deviations to cohorts and upstream callsites, with confidence scores and a sample-size-aware alert gating to reduce false positives.

What changed (high level):
- New aggregation job: observability/aggregation/macroframe_job.py and Go worker shim for production ingestion. The job emits sparse lattice documents to the metrics index.
- Mini-trace anchors: tracing/mini_anchor contains an ID generator and a lightweight sampler that attaches an anchor id to a subset of invocations; anchors are joinable to lattice docs for quick debugging.
- Grafana dashboard: console/dashboards/invocation_lattice.jsonnet with cohort sparklines, heatmap, and an interactive drilldown panel linking to trace anchors.
- Alerting: pkg/alerts/cohort_cost_resolver.rs implements the cohorted-cost-drift alert gating and fallback suppression rules; runbook updated.
- Metrics surfaced: new time series for cohort_token_entropy, cohort_weighted_cost, cohort_latency_p50/p95.
- Docs and examples: docs/observability/macroframes.md and examples/quickstarts/invocation-lattice-demo.md.

Checklist:
- [x] Unit tests for lattice aggregation and resolver
- [x] Dashboard panels + template vars
- [x] Alert rule with sample-size-aware thresholding
- [x] Privacy review for mini-trace anchors (pseudonymous)
- [x] Oncall runbook stub and escalation notes

Commits (summary):
- feat(telemetry): add sparse invocation lattice aggregation job
- feat(tracing): add mini-trace anchor sampler and joiner
- feat(dashboard): add Invocation Lattice dashboard jsonnet
- feat(alerts): cohorted-cost-drift resolver and gating logic
- test: lattice job unit tests and end-to-end smoke test
- docs: macroframes overview and quickstart example

Files changed (high level):
- observability/aggregation/macroframe_job.py
- observability/aggregation/tests/test_macroframe.py
- tracing/mini_anchor/anchor_sampler.go
- console/dashboards/invocation_lattice.jsonnet
- pkg/alerts/cohort_cost_resolver.rs
- metrics/exports/cohort_metrics_exporter.sql
- docs/observability/macroframes.md
- examples/quickstarts/invocation-lattice-demo.md
- scripts/ci/telemetry_lint.sh

CI: Initial push had linter errors (jsonnet formatting + goimports). Fixed lint and a flaky telemetry test. Final CI status: all checks passed after second run.

Reviewer notes / conversation (summary):
- Anika Rao: "This looks promising. Two asks: (1) ensure the mini-trace anchors are pseudonymous and reversible only by KMS holders; (2) lower default sampling to 0.1% and expose a per-tenant override in Dedicated."
  - Author (Maya): "Addressed both: anchors are HMAC'd with KMS key id and sampling is 0.1% default; added feature flag and Dedicated-only override in config schema. See macroframe_job.py and anchor_sampler.go changes."
- Diego Alvarez: "Can we add a backfill check to validate older metrics map to cohorts?"
  - Maya: "Added a small backfill verifier script and tests under observability/aggregation/tests/backfill_verify.py — runs as part of the smoke suite."
- Liam O'Dell: "Alert tuning looks good; please add a comment explaining the sample-size gating math (Wilson interval) so oncall understands tradeoffs."
  - Maya: "Added doc comments and an alert-rule annotated comment in cohort_cost_resolver.rs and the runbook entry."

Merge: Squash merged to main after approvals from Anika and Liam. Linear items: ENG-3421, ENG-3500 marked done.

Rollout plan:
1) Deploy aggregation worker behind feature flag and enable anchor sampling at 0.01% internally for 48h.
2) Enable metric export and dashboard in staging for SRE review.
3) Gradually increase sampling to 0.1% on prod for internal tenants; Dedicated customers opt-in.
4) Watch cohort_token_entropy and cohort_weighted_cost for 72h, adjust alert thresholds per observed variance.

Notes/risks:
- Added cardinality controls to avoid cohort explosion; cohorts are computed with hashing + top-k caller fingerprint retention.
- Privacy: anchors are pseudonymous and joinable only with KMS-held secrets; product reviewed by compliance.
- Cost: additional aggregation job adds modest CPU and storage; exporter uses compact sparse document format to limit index growth.

Anika Rao: This looks promising. Two asks: (1) ensure the mini-trace anchors are pseudonymous and reversible only by KMS holders; (2) lower default sampling to 0.1% and expose a per-tenant override in Dedicated.
Maya Chen: Addressed both: anchors are HMAC'd with KMS key id and sampling is 0.1% default; added feature flag and Dedicated-only override in config schema. See macroframe_job.py and anchor_sampler.go changes.
Diego Alvarez: Can we add a backfill check to validate older metrics map to cohorts?
Maya Chen: Added a small backfill verifier script and tests under observability/aggregation/tests/backfill_verify.py — runs as part of the smoke suite.
Liam O'Dell: Alert tuning looks good; please add a comment explaining the sample-size gating math (Wilson interval) so oncall understands tradeoffs.
Maya Chen: Added doc comments and an alert-rule annotated comment in cohort_cost_resolver.rs and the runbook entry.