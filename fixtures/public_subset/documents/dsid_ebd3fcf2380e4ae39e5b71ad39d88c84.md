Implement residency-proofing admission webhook and KMS usage boundary

Motivation: Customers running Private deployments requested stronger, declarative enforcement of data residency assertions at the API ingress point so KMS selection and audit tagging cannot be bypassed by malformed requests. This PR introduces an admission-style residency-proofing webhook and a lightweight KMS usage boundary evaluator that enforces: 1) requests must carry a residency assertion header (X-Resident-Region) or a signed residency claim; 2) the request's effective KMS region must be compatible with the asserted residency; 3) mismatches are rejected or routed to a configurable fallback model based on policy. This is a production-grade change: it updates RBAC policy mappings to include a new operator role for residency-readers, augments audit events with residency_proof fields, and wires the webhook into Private installer manifests so it can be enabled per-cluster via feature flags.

What changed (high level): Added residency webhook controller and policy cache, a KMS boundary evaluator library, RBAC policy generator changes (templates), audit enrichment to include residency_proof and kms_affinity tags, unit and integration tests, and installer manifest updates for Private deployments.

Design notes: The webhook validates residency headers first; if a signed assertion is present we validate signature (delegated to the existing attestation verifier). The KMS boundary evaluator consults a small policy cache populated from the policy service (read-only) and applies a deterministic match: exact-region, geozone (continent), or global fallback. The webhook supports three outcomes (allow, allow-with-warning, deny) controlled by cluster policy. Allow-with-warning emits a high-severity audit event and attaches remediation hints.

Rollout plan: staged feature flag rollout: disabled by default for Hosted, enabled in Private alpha clusters behind feature flag residency-proofing.v1; contact SRE for canary. The webhook runs with read-only RBAC (residency:reader) and falls back to evaluations that do not require secret access.

Checklist: [x] Unit tests for evaluator and signature path; [x] Integration test with installer; [x] Audit fields added and sampled in test harness; [x] RBAC templates updated; [x] Documentation sketch (docs/security/residency-webhook.md) included as a short draft.

Commits: see commits list below.

Files changed (high level): see changed_files list below.

Notes on compatibility: This change is additive to mainline API behavior when disabled; when enabled it is enforcement-level and can cause request rejections — operators must enable it deliberately. We recommend enabling with allow-with-warning first for one week to capture signals.

Related: references ENG-4120 (policy cache stabilization) and SEC-889 (residency attestation requirements).
Arjun Patel (2026-02-19): Can you clarify why we need a separate residency:reader role instead of reusing the existing operator roles? Is the intent least-privilege for the webhook's service account?
Maya Chen (2026-02-19): Correct — residency:reader is a narrow role that only allows reading policy and attestation public keys. Existing operator roles grant broader secrets access; separating reduces blast radius in Private clusters.
Sofia Morales (2026-02-21): CI failed on first run due to a data race in the policy cache; please add a simple sync.Once around index initialization. Also add an explicit timeout on HTTP calls to policy service.
Maya Chen (2026-02-22): Added sync.Once and explicit http.Client timeout; updated tests to use a fake policy service with deterministic responses.
Diego Alvarez (2026-02-25): Small nit — audit field should be residency_proof.version to support future formats. Please make the field a nested string like 'v1:...' in the enricher.
Maya Chen (2026-02-26): Implemented versioned residency_proof as a single string prefix (v1:BASE64SIG). Updated docs and tests.
Add residency-proofing admission webhook and KMS boundary evaluator. Operators can enable via residency-proofing.v1 feature flag in Private installations. Adds residency_proof and kms_affinity to audit logs. See docs/security/residency-webhook.md for operator guidance.