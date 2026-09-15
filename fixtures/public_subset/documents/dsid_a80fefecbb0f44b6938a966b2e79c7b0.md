Add intermittent-sampler sweep runner and opportunistic cache auditor

Summary: Adds a new benchmark runner (intermittent-sampler-sweep) and an Opportunistic Cache Auditor reporter to the benchmark-lab suite. The runner simulates bursty/intermittent traffic patterns with configurable sleep windows to exercise cold vs warm cache transitions; the auditor records KV cache hit/miss transitions and correlates them with per-request kernel samples to help diagnose transient thrash and warmup regressions.

Motivation: Several recent production incidents showed short-lived cache thrash during bursty ingress (p95 spikes for ~10s) that escaped our steady-state benchmarks. This PR provides a reproducible harness for those patterns and instrumentation to pinpoint whether regressions are scheduling-, batch-, or cache-related.

Key features:
1) Deterministic intermittent workload generator with phase scripting (pattern file format) and seeded RNG for reproducible runs.
2) Opportunistic Cache Auditor that attaches (non-invasively) to the benchmark runtime to capture KV cache evictions, prefix/hotpath counts, and short-lived miss bursts; outputs a delta-based audit stream suitable for our reporting pipeline.
3) Sampled kernel-stack annotator that emits a small stack capture per sampled request (configurable sampling rate) to correlate hot kernels with cache miss windows.
4) Phase-aware tail-latency heatmap reporter (per-phase p50/p95/p99 distributions and duration buckets).
5) CI benchmark job integration and a small local runner for dev iteration.

Description (detailed): The intermittent-sampler-sweep runner lives under benchmark-lab/runners/ and accepts a pattern.yaml that describes sequences of: {duration, qps, parallelism, sleep_ms, seed}. The default patterns include a short-burst-with-sleep scenario and a multi-burst staircase. The runner boots the workload, primes caches for a configurable window, then performs a sweep across phases while emitting standardized events to the bench-event bus. The Opportunistic Cache Auditor subscribes to that bus and tracks low-level KV events emitted by our test runtime (kvcache.eviction, kvcache.hit, kvcache.miss). When a transient miss burst is detected (miss-rate delta > configurable threshold within a sliding window), the auditor triggers a brief increased sampling rate for the kernel-stack annotator and tags the event with the active phase id and seed.

Implementation notes:
- Deterministic seeding is enforced across runner and reporters to make runs fully reproducible when run with the same pattern.yaml and seed.
- The auditor tracks per-model and global cache metrics and writes an audit stream in newline-delimited JSON (ndjson) under artifacts/audits/<run-id>.ndjson to be ingested by the reporting pipeline.
- The sampled-stack annotator uses a lightweight in-process signal-safe collector that collects at most one sample per request to minimize perturbation; sampling rate is adjustable (default 1%).
- The phase heatmap reporter produces both CSV and a condensed JSON summary for quick triage.

Commits:
- a1c3db2: runner: add intermittent-sampler-sweep implementation, pattern parsing and phase engine
- b4f7e90: reporter: add opportunistic cache auditor with ndjson audit stream
- c9d2f17: reporter: sampled kernel-stack annotator and per-phase heatmap exporter
- d2a9f41: tests: unit tests for pattern parsing and auditor detection thresholds
- e78b5c0: ci: add bench job and example pattern files

Files changed (high level):
- runners/intermittent_sweep.py (new)
- reporters/cache_auditor.py (new)
- reporters/stack_sampler.py (new)
- reporters/phase_heatmap.py (new)
- config/patterns/intermittent_default.yaml (new)
- tests/test_pattern_parser.py (new)
- ci/bench-workflow.yaml (modified)
- docs/bench/bench-guide.md (modified)

Small illustrative snippet (runner usage): ```yaml
pattern:
  - duration: 30s
    qps: 250
    parallelism: 16
    sleep_ms: 10000
    seed: 42
```

Integration notes:
- The new bench workflow (ci/bench-workflow.yaml) is gated behind a bench label to avoid running on every PR; it runs nightly for main and on-demand via workflow_dispatch.
- Local development: `python -m benchmark_lab.runners.intermittent_sweep --pattern config/patterns/intermittent_default.yaml --out artifacts/run-123`

Observability and outputs:
- artifacts/run-123/audits.ndjson (time-series of cache deltas and annotations)
- artifacts/run-123/heatmap.json (phase->quantile buckets)
- artifacts/run-123/stack-samples.ndjson (sampled kernel stacks)

Review conversation (high level, chronological):
- Ethan Cole (Reviewer) 2024-11-18: "Nice focused runner. Can we ensure the sampling collector doesn't skew latencies? Please include micro-benchmark numbers." Author (Priya) 2024-11-18: "Added a micro-benchmark in stack_sampler microbench and lowered default sampling to 1% — numbers added to docs."
- Marisa Flores (Reviewer) 2024-11-19: "Audit stream format looks good. Can you include model-id tag on each audit event so we can correlate multi-model Dedicated tests?" Author (Priya) 2024-11-19: "Done — auditor now emits model_id and route_id when available."
- Luca Romano (Reviewer) 2024-11-20: "CI bench job failed on first run due to missing workflow dispatch permission. Also saw a flake in test_pattern_parser on ~arm runners." Author (Priya) 2024-11-20: "Fixed CI YAML to use workflow_dispatch and relaxed timing in test to be tolerant on low-power runners. Re-ran CI."

CI and checks:
- build: pass
- lint: pass (minor style fixes applied)
- unit-tests: pass
- bench-smoke: fail on first run (missing permissions) -> fixed -> pass
- nightly-bench (manual run): pass

Merge: Squash and merge after approvals. Merge outcome: merged into main on 2024-11-21 via squash merge.

Post-merge follow-ups:
- ENG-4897: add a visualizer that consumes the heatmap JSON for quick on-call triage (tracked).
- ENG-4821: run the intermittent patterns against the Dedicated fleet in the staging cluster (scheduled).

Why this is different from existing suites: Existing bench-lab runners focus on steady-state sequences and phase-mixing; this PR intentionally targets short-lived intermittent bursts and transient cache behavior with tight correlation between cache deltas and kernel-level samples. That combination (opportunistic audit + phase-aware sampling) provides a new signal for diagnosing short-lived regressions that previously evaded our baseline benchmarks.

How to run locally:
1) pip install -e .[bench]
2) python -m benchmark_lab.runners.intermittent_sweep --pattern config/patterns/intermittent_default.yaml --out /tmp/bench-run
3) tail -n +1 /tmp/bench-run/audits.ndjson
4) python benchmark_lab.reporters.phase_heatmap --input /tmp/bench-run/audits.ndjson --out /tmp/heatmap.json

Notes:
- No behavioral changes to production code-paths; all instrumentation is opt-in and isolated to benchmark artifacts.
- Added tests to keep pattern parsing deterministic and to verify auditor detection thresholds.
- This PR is intentionally conservative on sampling defaults to avoid perturbing measurements.
New benchmark runner: intermittent-sampler-sweep. Adds opportunistic cache auditing to detect transient KV-cache thrash under bursty traffic patterns; new reporters for sampled kernel stacks and per-phase tail-latency heatmaps.