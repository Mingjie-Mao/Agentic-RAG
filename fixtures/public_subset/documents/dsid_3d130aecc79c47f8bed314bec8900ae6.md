Introduce SessionFacade and idiomatic SDK adapters (Python/TS/Go)

Context: Each SDK implemented session and streaming primitives slightly differently over time (handoff, heartbeat, reauth). That drift made it hard to reason about cross-language behavior and complicated runtime code paths. This PR introduces a small core 'SessionFacade' abstraction in the redwood runtime and exposes lightweight, idiomatic adapters for Python, TypeScript, and Go that align on single-source semantics (lease-based session, deterministic retry windows, explicit lifecycle signals).

What changed (high level): - runtime: add SessionFacade interface + minimal lifecycle hooks (CreateLease, ExtendLease, Release, HeartbeatHint) and tests for deterministic retry windows. - python SDK: new redwood.session_facade.SessionFacade and an AsyncIterator wrapper (session_facade.AsyncStream) that emits typed chunks and surface ergonomic helpers (aiter_lines, aiter_json). - typescript SDK: export sessionFacade() factory; AsyncIterable wrapper emits chunk objects and exposes backpressureHint() to assist consumers. - go SDK: channel-based SessionAdapter and a StreamController that sends typed events to the consumer goroutine; simplified reconnect/retry API. - compatibility: implement per-SDK shims named LegacySessionCompat that map existing calls to the facade for a gradual migration. - docs/examples: migration guide snippets for each language and a codemod for the Python helper rename.

Motivation: unify behavior across languages so integrations that depend on token refresh, stream resume, and retry policy observe the same semantics without duplicating large swathes of logic in each SDK. This also simplifies the runtime surface area for session affinity and leased credentials.

Migration notes: - Current callers continue to work (LegacySessionCompat). - We recommend moving to the idiomatic adapters: Python: use 'async for chunk in session.async_iter()' TypeScript: 'for await (const chunk of session.asyncIterable())' Go: 'for ev := range session.Events() { ... }' - Small breaking edge-case: users relying on internal order of some low-level event enums should consult the migration guide; most apps unaffected.

Testing & CI: Added cross-language integration tests exercising token rotation, heartbeat, and session handoff under load. CI matrix includes Py 3.10/3.11, Node 18/20, Go 1.20/1.21. All checks green.

Rollout: staged rollout across hosted and dedicated infra with feature-flagged runtime behavior. SDK releases are coordinated; we published prerelease candidates internally and cut stable releases after this merge.
Adds a SessionFacade in the core runtime and language-specific adapters to provide idiomatic streaming/session APIs for Python (async iterator), TypeScript (AsyncIterable with backpressure hints) and Go (channel-based stream). SDKs bumped: python 2.3.0, typescript 1.11.0, go v1.5.4. Backwards-compatible shims provided for current callers; recommend migration for ergonomic improvements and clearer retry lifecycles.
Ava Patel: Can we avoid exposing heartbeat internals in the public TS surface? Suggest exposing only a hint rather than raw timestamps.
Author (Priya): Switched TS surface to backpressureHint() and added a comment explaining runtime-only timestamps remain private.
Marco Ruiz: Please add a unit test that simulates token refresh failure during handoff and validate fallback path.
Author (Priya): Added test in sdk/python/tests and expanded the Go integration to cover the token-refresh-failure fallback. CI updated.
Elena Chen: Rename CreateLease -> AcquireLease for clarity; small naming nit across all adapters.
Author (Priya): Renamed methods and updated migration docs. Left LegacySessionCompat alias for backwards compatibility.
Hotfix follow-up planned: reduce Python generator memory spike under extreme parallel streams (tracking ENG-1487)
SDK owners: coordinate minor release notes; callouts added to console deprecation page for legacy shim in 3 months