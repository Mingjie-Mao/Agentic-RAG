# Selection score backfill implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development and test-driven-development. Preserve registered v1 and all its observations; current directory is authorized. No Core, gold-derived runtime routing, new search/LLM call, default activation or commits.

**Goal:** Complete the three missing conditional selector observations with actual, separately registered BGE scores.

**Architecture:** The v1 zero-inference replay reproduced all47 rank_floor contexts and source/index identities, but guarded retained branches for which historical scores do not exist. Keep that failed run unchanged. Add explicit opt-in fresh ranking to fill exact missing query/row pairs from the same authorized candidate pool with the existing BGE reranker; share the observed scores across policies and meter actual inference separately from cache replay.

**Tech Stack:** Existing `app.rerank.rerank`, `_ranked` integrity checks, selector/replay/readonly source helpers.

## Task 1: Explicit measured cache fill

Files: `scripts/run_guarded_selection_dev.py`, `tests/test_guarded_selection_replay.py`.

- [x] Write RED tests: default missing pairs still fail; opt-in provider fills only missing exact query/source pairs; changed query or source requires fresh scoring; existing cached values are unchanged; provider input mutation, incomplete/duplicate/nonfinite output and budget exhaustion fail without fake scores.
- [x] Add an explicit `--fresh-ranking` opt-in; absence preserves strict zero-inference v1 behavior. Registration v2 declares the same three selection arms, original inputs, no search/embedding/policy/generation calls, at most8 actual reranker dispatches per task and at most8 times its unique candidate count new pairs. Register actual model revision/file hashes, batch/max-token and library identity before any new scores.
- [x] Reuse `_ranked` with the existing reranker over the exact missing rows and full window, retain known observations, record attempts/completed/failed calls, pairs and wall time even on interruption. Do not compare distinct query score scales. A fresh scorer receives only query/authorized rows, never task ID, required facts or references.
- [x] Current source/model/code/settings/input/private bytes identity is checked before and after; preserve readonly DB/index validation and full47 denominators/null behavior. New private scores, input keys, query and events live only under the new ignored run directory, while public costs contain identifiers/hashes/counts. Keep the raw frozen v1 sources at `.runtime/evidence-first-20261011/frozen-code-v1` and never rewrite v1 result/registration/private source files.
- [x] Before mixing cached and fresh scores, require the registered/current reranker to equal both original parent and repair reranker identities, and require original parent library versions to match. A newly frozen different model is not a compatible score scale. Reject mismatches before fresh inference, retain null full47 outcomes on execution failure.
- [x] Run focused runner/selector tests, Ruff/diff, fresh spec then quality review. Save new `selection-input-v2.json` before inference and `dev-guarded-selection-20261011-v2.json` afterward. Deployment still requires the registered no-native-loss and strict external-gain gate; a complete negative result is retained and reported.

2026-10-11 v2 code gate: fresh spec PASS, quality APPROVE after cached/active model compatibility fix. Runner143 passed; previous ten-file regression588 passed. New selection-input-v2 registered for47 tasks before inference; exact code preserved in ignored frozen-code-v2 (207 files). Complete actual Dev replay is in progress; no strategy/accuracy claim yet.

2026-10-11 actual v2 result: complete valid47, all9 drift flags false, all profiles0errors. Original rank_floor reproduces47 contexts. Revised native45/45 in all arms; accepted external baseline15/19, rank_floor14/19, guarded14/19 (candidates18/19). Guarded fails strict-gain adoption rule, chosen baseline; production unchanged. Missing-score backfill consumed3 actual dispatches/111pairs/15.803s wall; tokens and monetary costs unavailable/null. No new search, embedding, policy or generation calls. Result: artifacts/dev-guarded-selection-20261011-v2.json; private run85e55995fa0a42dea2c8a4c9fab88512.
