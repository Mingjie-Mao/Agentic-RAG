Connection snapshot + staged traffic wash orchestrator for regional soft-recoveries

Reduce client-visible 5xx spikes during regional failovers by persisting ephemeral connection state, restoring a consistent snapshot on rejoin, and applying a staged traffic wash/ramp with probe-driven holdoffs. No public API changes.
Context: In a recent multi-region failover test we observed brief client-facing error spikes (5xx) after nodes rejoined. Root cause analysis found two contributing factors: (1) ephemeral connection state (inflight KV and scheduler leases) was not consistently restored leading to transient backpressure and hot-path rejects, and (2) full traffic immediately flooded restored nodes before internal caches (KV/prefix) warmed, causing short-lived overloads.

Goal: Introduce a lightweight snapshot/restore for ephemeral connection metadata + an orchestrated, probe-driven traffic wash (staged-ramp) to give caches and rate controllers time to stabilize. This is a targeted reliability hotfix for regional soft recoveries; it is conservative and opt-in by feature-flag for initial rollout.

What changed (summary of commits):
- Add connection-snapshot store and restore glue (store ephemeral leases & inflight counts to local / tmp snapshot on drain) — small persisted snapshot lifecycle with TTL.
- Implement staged traffic-wash controller: probe-driven ramp with holdoff timers and per-tenant ramp factors.
- KV cache warm-checks and probe hook added to serving runtime; staged ramp consults warm-checks before advancing phases.
- Telemetry: new metrics (snapshot.restore.time, traffic_wash.phase, probe.warm_success_rate) and traces for restore paths.
- Feature flag + safe defaults: disabled by default for older clusters, enabled in canary via operator.

Files changed (high-level):
- runtime/connection/snapshot.go (new) — snapshot save/restore and TTL handling
- serving/wash/controller.go (new) — staged-ramp state machine and probe integration
- serving/wash/flags.go — feature flag + config knobs (ramp duration, probe window)
- telemetry/metrics.yaml — new metric definitions and dashboard hooks
- operator/canary-configs/redwood-canary.yaml — rollout toggle and default scaling policy updates
- tests/serving/wash_test.go — unit tests for ramp progression and failure modes
- docs/operational/traffic-wash.md — operator-facing runbook and recommended rollout steps

Review conversation (high-level excerpts):
- Miguel: "This looks sensible; can we ensure snapshot storage is local-only and never shipped to control plane? Also add a non-blocking path if snapshot fails."
- Author (Aisha): "Snapshot kept ephemeral and local; added non-blocking fallback — if snapshot write fails we fall back to cold-warm probe path and emit a metric (snapshot.write.failed)."
- Jordan: "Probe integration needs to avoid adding excess load; recommend exponential backoff on probe retries and cap probe concurrency per node."
- Author: "Added probe concurrency cap and backoff; unit tests updated."
- Sophie (approval): "Telemetry and runbook are clear. Approve subject to CI green and one more e2e validation in canary."

CI and validation: All checks passed on final push (unit tests, lint, integration smoke). We ran a canary in us-west-2 with 3% of traffic and observed no 5xx spikes during a simulated rejoin; metrics show snapshot.restore.time ~= 120-300ms and traffic_wash progressed across 3 phases with stable probe success rate > 95%.

Fallbacks and safety: If probe success rate remains low, controller stalls progression and moves nodes into a temporary warm-hold state with conservative rate limits. Snapshot failures are safe — the code falls back to existing slowpath recovery. Feature flag allows rollback without code revert.

Rollout plan: Enable as canary (3%) for 24h, monitor snapshot.restore.time and probe.warm_success_rate, then gradually increase to 25% and 100% in 48h windows. Operator runbook (docs/operational/traffic-wash.md) includes quick rollback steps and metrics to watch.

Motivation/Why unique: This PR stitches a short-lived persisted snapshot of ephemeral serving state with an orchestrated, probe-aware traffic ramp to avoid the classic "rejoin flood" problem where un-warmed caches + immediate load cause transient overloads. It targets production reliability without changing public APIs and keeps the change reversible and telemetry-rich for operational confidence.

Merge: Squash-merged after approvals and green CI.