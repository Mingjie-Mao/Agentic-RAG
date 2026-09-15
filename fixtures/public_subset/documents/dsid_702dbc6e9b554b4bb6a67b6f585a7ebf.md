Introduce tenant audit-sampling profiles and KMS residency cohorting

Motivation: customers with large-volume tenants and strict data residency requirements requested a way to reduce audit storage costs while preserving compliance signal and maintaining cryptographic provenance for audited operations. This PR introduces two coordinated features: (1) tenant-level audit-sampling profiles that let ops/security teams configure sampling rate, time-windowed sampling, and retention hooks; (2) KMS residency cohorting that pins envelope encryption to regional CMKs based on tenant residency profiles and cohort policies.

Design summary: sampling profiles are stored as a tenant-scoped policy object and enforced by the audit pipeline before long-term storage. Profiles support 'uniform', 'cohort-window', and 'triggered' sampling strategies and include a small deterministic hash salt to permit reproducible sampling for debugging. KMS cohorting maps tenant residency attributes to a prioritized list of CMK aliases (region-prefixed) and attaches a short provenance tag into the audit envelope that records the chosen key alias and residency decision.

RBAC: only roles with 'audit.policy.write' can mutate sampling profiles; RBAC v2 condition evaluator was extended to validate location claims and time-scoped grants.

Backward-compatibility: default sampling profile for tenants without configuration remains 'uniform:100%' (previous behaviour).

Implementation notes:
 - New tenant API: POST/PUT /v1/tenants/:id/audit-sampling (JSON schema included in files changed).
 - Audit pipeline change: auditd-worker applies sampling decision, tags metadata, encrypts envelope with selected CMK alias via kms-router, and forwards to blob-store.
 - kms-router change: added cohort resolution service and fallback order; resolves candidates then attempts envelope wrapping via region-aware key proxy.
 - Console: added UI to view and edit sampling profile with recommended presets and a preview estimator of storage impact.
 - Migrations: added background job audit_sampling_migration to annotate recent audits with provenance tag for retroactive analytics; job is opt-in during rollout.

Checklist:
 - [x] API schema and validation
 - [x] Server-side enforcement with tests
 - [x] End-to-end integration tests for both sampling and KMS routing
 - [x] Console components and e2e tests
 - [x] RBAC unit tests and policy docs
 - [x] Rollout plan and feature flag gating

Commits summary:
 - 9d3a7f2 Add tenant audit-sampling schema and API handlers
 - 2b6f4c1 Implement sampling decision in auditd-worker and provenance tagging
 - 7f1b8c0 Introduce kms-router cohort resolver and region fallback
 - c3a6d92 Add RBAC v2 mutation checks and unit tests
 - 4b2e18a Console: sampling UI and preset recommendations
 - 5e9f01b Add migration job and documentation updates

Files changed (high level): ["services/auditd/worker.go", "services/auditd/sampling.go", "pkg/kms/router.go", "pkg/kms/cohort_resolver.go", "api/tenants/sampling_handler.go", "api/tenants/schemas/audit_sampling.json", "webapp/src/pages/tenant/Sampling.tsx", "webapp/src/components/SamplingPresets.tsx", "infra/featureflags/ffdefs.yaml", "tests/e2e/audit_sampling.spec.ts", "scripts/audit_sampling_migration.go", "docs/operational/audit_sampling.md"].

Review conversation (high level):
 - Diego Alvarez: "Great direction. Please add explicit tests for deterministic sampling seed collisions and a note about how seed rotates on tenant re-creation."
 - Aisha Raman (author): "Added unit tests for seed collision and documented seed lifecycle behavior in audit_sampling.md (see migration section)."
 - Mei Chen: "We need assurance that provenance tags do not leak sensitive region identifiers in cases where tenants are multi-region; recommend obfuscation option."
 - Aisha Raman: "Added obfuscation toggle to sampling profile (obfuscation:true/false) and defaulted to true for multi-region tenants. Also added policy to redact clear region names in the audit ui unless viewer has 'audit.region.view' permission."
 - Samir Patel: "Can we avoid additional KMS API calls for every audit event? Suggest caching resolved CMK alias per tenant with TTL and ensure fail-open path uses server-side envelope caching."
 - Aisha Raman: "Implemented per-tenant CMK alias cache (configurable TTL) and added metrics for cache hits/misses. Failure path uses fallback server-held envelope-wrapping pool (documented)."

CI and checks:
 - unit-tests: pass
 - integration-tests (audit pipeline): pass
 - lint: pass
 - security-scan (static): pass (one informational note about console dependency upgraded)
 - e2e: pass

Rollout and migration plan (short):
 - Stage 1 (FF off): merged to main; API schema behind feature flag.
 - Stage 2 (pilot tenants): enable FF for a handful of internal/pilot tenants, run migration job to annotate recent audits, and validate storage impact with estimator.
 - Stage 3 (wider rollout): enable for production tenants gradually with monitoring on sampling rate, provenance integrity checks, and KMS errors.
 - Stage 4 (on by default): flip default FF and retire migration job after 30 days of stable metrics.

Observability: added metrics audit.sampling.decisions.{sampled,skipped}, audit.kms.resolve.{success,fallback,fail}, and audit.provenance.tag_rate. Alerts: P1 if kms.resolve.fail rate > 1% sustained for 10m.

Merge outcome: squashed and merged to main on 2025-11-02 via squash-merge.

Post-merge followups: create runbook for migration job, telemetry dashboards, and compliance review with legal to validate provenance tag compliance with tenant contracts.
Add tenant-configurable audit sampling profiles, default conservative sampling with retention hooks, and KMS residency cohort routing. Includes RBAC v2 checks for sampling policy mutation and a staged rollout plan behind feature flags.
Diego Alvarez: Please add explicit tests for deterministic sampling seed collisions.
Mei Chen: Consider obfuscation of region identifiers for multi-region tenants.
Samir Patel: Avoid per-event KMS calls by caching resolved CMK alias.