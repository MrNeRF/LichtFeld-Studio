# Round-2 measured directives (Fable deep-dive #2, 2026-08-07)
Profiles: bench worktree perf_campaign/profiles/*; microbench: RNG kernel-shape A/B on this RTX 4080.

## Directive 1 — blend_backward_cu (40-56% of ALL GPU busy; 5-6x slower than forward on same tiles)
Measured: bonsai-late p50 1955us vs forward 400us; bicycle 797 vs 130. Sources (verified in
kernels_backward.cuh:439-704): (1) walks ALL tile instances though forward recorded per-pixel
last_contributor — dead tail splats ride the full 383-sync diagonal; (2) barrier chain: ~3
block syncs/splat vs forward ~1/256-batch; (3) __launch_bounds__(128,8) caps occupancy 67%.
Atomics are already right (register-accum, ~11/contributing splat); stored final-T + recompute
is correct — keep.
- **BWD-A (P0, EXACT math, ~15 lines)**: block-reduce max(last_contributor) after :504-519;
  clamp walk to T_eff. Bit-identical grads. Expect −25-50% in saturated scenes. => WO-BWD-A.
- BWD-B (P1): in-block survivor compaction over [0,T_eff) (spirulae surv[] pattern).
- BWD-C (P1): 8x8 micro-tile + __syncwarp diagonal (spirulae port; grid x4, batch 32).
- BWD-D (P1 alt): pixel-centric lockstep rewrite (gsplat-style), warp-reduced grads, 1 atomic/warp.
  Prototype C vs D behind flags, ship ONE. Target: bwd ~2-2.5x fwd (~900-1000us bonsai-late).
- BWD-E (P2): __expf (bwd only), launch_bounds sweep, batch thresholds, conic AoS float4.
Experiments (no ncu): batch-size 64-vs-128 sync probe; __expf A/B; atomic-to-scratch probe
(timing-only). All via profile.sh late window + bicycle window.

## Directive 2 — noise injection (measured 99.5% = curand_init XORWOW skip-ahead)
Microbench N=400k, 60% gated: current 1149us; Philox4_32_10 + curand_normal4: 6.5us (-99.4%);
XORWOW-minus-init isolation 5.0us (proves init is everything); __powf irrelevant.
Fix sites: mrnf_kernels.cu:73-74, mcmc_kernels.cu:241-242 (+ :964-965, :1083 refine-time).
Philox init is counter-setup (~free); statistically >= XORWOW. Demoted after measurement:
frequency gating, half noise, fusing into other kernels (~10us launch, P3 hygiene). => WO-NOISE-PHILOX.

## Directive 3 — host phase: decode THROUGHPUT, not overlap
Measured (bicycle 200-500): 1201ms of 1972ms window is GPU gaps; nvjpegDecodeJpegHost
(GPU_HYBRID = host Huffman) 300 calls / 1440ms / mean 4.8ms — ALL on one thread
(pipelined_image_loader.cpp:345). Double-buffering can't fix service_time > consumption.
steady_ms is BLIND to this (next() wait outside the timed span, perf_bench.cpp:96-135).
- **HP-1 (P0)**: decoded-GT device cache (bicycle 592MB, bonsai 356MB decoded u8 CHW) with
  VRAM budget gate + pinned-host middle tier + fallback; add dataloader_wait_ms metric FIRST.
  Expect bicycle wall −30-45%. => WO-HP1-gt-cache.
- HP-2 (P1): 2-4 decode workers / wire existing batch-decode API (nvcodec_image_loader.cpp:1156-1493).
- HP-3 (P2): event-based RGB handoff instead of stream-sync (nvcodec_image_loader.cpp:1127-1137).

## Revised Wave-5 ranking
1 NOISE-PHILOX (−1.3ms/iter large-N, trivial) | 2 BWD-A (exact, −0.5-1.0ms) | 3 HP-1 (+metric;
bicycle wall −30-45%) | 4 WO-G2 codec fix (late-window gate) | 5 BWD-B/C/D structural |
6 HP-2 parallel decode | 7 CUDA graphs (DEMOTED: <=0.2ms, blocked by n_instances event) |
8 stream/memset polish (noise-level).
