Heterogeneous-bandwidth-aware allreduce fanout with KV-shard coalescing and comm/compute overlap

Motivation: on mixed-cluster deployments we observed long tail latency and suboptimal throughput for tensor-parallel allreduce when nodes have heterogeneous NVLink/intra-node bandwidths or asymmetric cross-node links. Smaller microbatches and low-latency applications (streaming chat) disproportionately hit these tails.

This PR introduces a bandwidth-aware fanout planner and a KV-shard coalescing pass that together reduce cross-node stalls and improve overlap between communication and compute. It also adds a narrow-copy optimized kernel for strided small-chunk merges and a runtime hook that maps comm streams to physical links based on a link-rate profile discovered at startup.

What changed (high level):
- Fanout planner: builds a minimal spanning fanout for allreduce that avoids slow links and increases per-hop concurrency when it detects bandwidth heterogeneity. The planner prefers intra-node NVLink-RDMA pairs, then high-rate cross-node links, and uses low-rate links only as fallback.
- KV-shard coalescer: groups adjacent KV cache shards into aligned coalesced writes to reduce the number of small allgathers. This reduces per-step launch overhead and amortizes DMA for common prefix cache hits.
- Stride-optimized memcpy kernel: a small, vectorized kernel for <4KB strided copies used by the coalescer; reduces memcpy time by up to 3x in microbenchmarks.
- Stream mapper: startup probe collects link bandwidths (via existing topo_probe) and creates stream->link mappings to route collective streams away from congested links.
- Scheduler tweak: a light-weight scheduling hint that encourages launching compute kernels on the stream that finished the local reduce, improving KV reuse and overlap.

Benchmarks: (multi-node, 8xA100-80GB, TP-4)
- Throughput: +18% median for 512-token workloads with batch size 8; +11% median for batch size 2.
- P99 latency: reduced by 24% for streaming microbatch workloads (<=2 microbatches).
- Small allgather count: reduced by 38% for a realistic prefix-caching trace.
- Overhead: startup probe adds ~45ms to init time on first run; negligible once cached.

Compatibility and safety:
- The fanout planner is opt-in via config flag "runtime.bandwidth_fanout.enabled" (default: true for Dedicated and Private, false for Hosted until canary completes).
- Fallbacks: if topology probe fails or detects homogeneous links, planner falls back to existing balanced tree.
- Unit and integration tests added; new runtime flag is gated behind a feature flag at rollout.

Checklist: 
- [x] Perf microbenchmarks
- [x] Unit tests for planner and coalescer
- [x] E2E smoke on CI
- [x] Docs: added runtime flags and rollout notes
- [x] Bench dashboard entry (Linear ENG-4281)

Related: ENG-4219 (heterogeneous link investigation), ENG-4281 (bench dashboard and canary plan).

Notes: this is intentionally conservative in memory use for the coalescer (max 1.5x temporary buffer). We plan a follow-up to expose a memory/perf knob and to extend the stream mapper to NVSwitch-aware topologies.
Diego Alvarez: Nice work — planner looks solid. Two concerns: (1) how do we ensure the planner doesn't create hotspots on high-bandwidth links? (2) can the coalescer accidentally increase mem usage in pathological KV layouts?
Maya Chen (author): For (1) planner bounds per-link concurrent fanout degree and we prefer fanout breadth over depth when bandwidth headroom is available — added comments and a unit test that simulates a single high-capacity link. For (2) coalescer enforces a 1.5x temporary buffer cap; I added a regression test that mirrors EXTREME_KV layout and it passes. See tests/unit/test_kv_coalescer.cc
Priya Natarajan: CI showed a flaky on the e2e smoke (timeout). I reran and it's green. Please add a brief note in the canary doc about rollback steps if p99 increases in the first 24h.
Lucas Meyer: approved
(bench numbers look believable; we should run on a heterogeneous NVSwitch+NVLink cluster before broad rollout).
Improves TP allreduce performance on clusters with heterogeneous link bandwidth by introducing a bandwidth-aware fanout planner and KV-shard coalescing. Adds a small stride-optimized memcpy kernel and stream mapping to reduce comm/computation stalls. Opt-in flag and canary plan included.