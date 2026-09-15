Phase-synced prober and traffic-decoupling rollout guard

Motivation: Canary detectors have been noisy for services with strong diurnal or periodic traffic phases (batch windows, analytics bursts). This PR introduces a phase-aware probing and a traffic-decoupling rollout guard that reduces false positives by aligning probe windows to observed traffic phases and by isolating probe signals from orthogonal workload spikes.

What changed (high level): added PhaseSyncedProber, a component that: 1) computes traffic phase boundaries using a lightweight seasonal decomposition (short-term STL) on recent tps/time-series; 2) schedules synthetic probes and aggregates per-phase latency fingerprints; 3) computes phase-weighted harmonic mean of tail percentiles to be robust to phase-specific heavy tails. Added TrafficDecoupler guard that splits telemetry by workload tag (route/service tag) and joins evidence only when regressions are coherent across decoupled groups.

Design details: phase detection uses a sliding window (configurable) with autocorrelation peak detection to identify periodicity. Probes are stretched to cover full phase cycles when period > configured threshold. Aggregation uses harmonic mean of p95/p99 across phase subwindows to avoid single-phase outlier domination. Guard logic requires: (a) phase-coherent delta > threshold AND (b) at least 2 independent workload slices showing same direction of change before triggering a hard rollback.

Files changed (high level):
- perf_canary/prober/phase_synced_prober.py: new prober implementation and phase detector
- perf_canary/guards/traffic_decoupler.py: new rollout guard logic and configuration schema
- perf_canary/config/schemas.yaml: added flags for phase-window, min-phase-period, workload-slices
- perf_canary/tests/test_phase_prober.py: unit and integration tests for synthetic sequences
- infra/dashboards/phase-prober.json: dashboard panels for phase boundaries and per-slice latency heatmap
- docs/perf_canary/phase-prober.md: user guide and rollout checklist
- perf_canary/prober/__init__.py: export bindings

Commits (summary):
- a1b2c3: initial implementation of phase detector, basic probe scheduler
- d4e5f6: add harmonic-mean aggregator and per-slice grouping
- f7g8h9: implement guard join logic, config schema, and error propagation
- i0j1k2: tests: add deterministic traffic generator and unit tests for edge cases
- l3m4n5: dashboards and docs; address review comments

Testing and validation: Unit tests cover periodic and aperiodic inputs; integration tests run against small canary harness with synthetic workload traces (batch, diurnal, bursty). Local perf runs show reduced false positive rate on historical traces: from 18% -> 4% on dataset with daily analytics windows. CI: all checks passed after two retries (flake in linter fixed).

Rollout plan: feature gated under runtime flag 'phase_synced_prober.enabled' (off by default). Stage rollout: (1) enable for internal perf-canary staging; (2) opt-in beta customers (10% of canaries) for 1 week with monitoring; (3) global enable if no elevated rollback signal. Metrics to monitor: phase_detection_rate, probe_coverage_ratio, phase_coherence_alerts, false_positive_count.

Backward compatibility and safety: The guard is additive and requires explicit opt-in. Existing detectors are unchanged unless the flag is enabled. We keep an escape hatch: immediate disable via config toggle and a kill-switch in the console.

Review convo (summary):
- Ethan Cole: "Nice approach — can you add unit tests for boundary conditions when period detection flips between nearby frequencies? Also prefer metric names to include service tag."
- Priya Nair: "Added deterministic generator test and renamed metrics to include {service}. Fixed in commit l3m4n5."
- Lin Zhao: "What about synthetic probe cost? Can we throttle during heavy tail phases?"
- Priya Nair: "Prober has adaptive probe-rate limiter — documented and tunable via config.min_probe_interval; added note in docs."
- Jules Park: "Reviewed and ran canary harness locally. LGTM after CI passes."

CI: build: pass; unit tests: pass; integration tests: pass; lint: pass (one initial flake fixed).

Merge: squash-merged to main after approvals from Jules and Lin. Merged on 2026-02-21.

Notes for on-call: if you see spike in phase_coherence_alerts after enabling, disable feature flag and notify perf infra. Emergency runbook path documented in docs/perf_canary/phase-prober.md.
Introduce PhaseSyncedProber and TrafficDecoupler guard to reduce false-positive perf regressions during periodic traffic swings. Adds new dashboards, config flags, and an opt-in rollout for beta customers.
Add phase-synced prober and traffic-decoupling rollout guard (opt-in). Improves canary stability for periodic workloads.
This PR is intentionally conservative: opt-in, observable, and includes a clear kill-switch. It targets reducing noisy canary rollbacks caused by periodic workload phases.