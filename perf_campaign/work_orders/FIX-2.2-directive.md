# Fix directive: joint Adam codec regression (from Fable deep analysis, 2026-08-06)
[Full analysis preserved. Root cause of +0.222 ms in preprocess_backward_cu at 63aa08c6:]
1. DOMINANT (0.08-0.13ms): register live-range explosion — unified Adam tail keeps
   shN_grads[15]x3 + 5 attr grad sets live across the covariance/EWA backward, plus
   us_u[48]/us_s[48] across a block reduction; no __launch_bounds__; block 128->256.
2. (0.04-0.07ms): lost early-exit for invisible/out-of-range prims + 30 block barriers
   (5 per Adam section x 6; alternating static-shared min/max reductions fragile).
3. (0.04-0.08ms): expm1f+log1pf+sqrtf+2 fdiv per cell x 62 cells x ~500k prims; encode
   re-divides block-uniform ranges per cell.
4. (~0.01-0.02ms): non-SH moment bytes 2->4 (priced into the memory win).

F1 (do first): close the visible-branch after convert_sh_to_color_backward_grads; run
   sh0+shN Adam THERE (kills the big live ranges before the geometry backward); reopen
   visible for covariance/EWA; tail keeps only means/rot/scale/opacity. Then sweep
   __launch_bounds__(256, 2..4). kernels_backward.cuh:105-410.
F2: fused block_reduce_min4 on float4{u_min,-u_max,s_min,-s_max} via __shfl_xor
   butterfly + one shared float4 round -> 2 barriers per section (30->12 total); fixes
   the fragile static-shared alternation in warp_reduce.cuh:118-165.
F3: hoist inv_u_range/inv_s_range (block-uniform) out of encode_us (2 fdiv->FMA per
   cell); guarded fast transcode in joint_adam_codec.cuh:28-38: __logf(1+x) when
   x>0.125 else log1pf; __expf-1 when log_s>0.118 else expm1f (0<->0 fixed point stays
   exact); keep host mirror joint_adam_codec.hpp in lockstep; JointAdamCodecTest 6/6.
F4 contingency: if ncu still shows local-mem in shN section after F1: drop us_u/us_s
   buffers, recompute decode in the encode loop from L1-resident joint_packed
   (legacy two-pass style). Keep block=256 (bounds layout requires blockIdx==prim/256).

MEASURE in order: nsys/ncu preprocess_backward_cu at 63aa08c6 vs parent (confirm
attribution); ncu registers/occupancy/local-mem/barrier-stall/XU-pipe to rank; after
each F-step re-measure. GATE: bonsai 3-run median steady_ms <= 4.065 (BICYCLE IS FLAT
FOR THIS KERNEL — bonsai gates), B/splat stays 409.4, codec tests 6/6, ledger green,
loss in Wave-2 band.

## Profiler availability note (2026-08-06)
- nsys (2026.1.3) works unprivileged NOW: use `nsys profile --stats=true -o /tmp/prof <bench cmd>`
  and compare `preprocess_backward_cu` kernel-duration sums per F-step. This suffices for
  attribution and step confirmation.
- ncu hardware counters are currently admin-locked (RmProfilingAdminOnly=1). Until the
  modprobe unlock lands (or run ncu under sudo), treat the ncu metrics in MEASURE as
  optional confirmation, not blockers. Do NOT skip the nsys measurements.

## MEASURED ATTRIBUTION (profiling harness, 2026-08-07) — READ FIRST
- Regression is SH-DEGREE-GATED: preprocess_backward_cu bonsai iters 200-500 (SH deg 0):
  parent 179.3us vs codec 140.3us (codec FASTER); iters 1600-1900 (SH>=1): 450.0us vs
  906.3us (+101%). Confirms root cause #1 (shN register live range).
- GATE PROTOCOL: measure with the harness LATE WINDOW:
  LFS_PROF_START=1600 LFS_PROF_STOP=1900 LFS_PROF_ITERS=1920 ./perf_campaign/profile.sh ...
  An early-window or whole-run-median check can FALSELY PASS. Harness + usage:
  perf_campaign/profile.sh (merged from lfs-elite-prof), details in
  perf_campaign/profiles/codec-pair-63aa08c6.md in the bench worktree.
- ncu deep metrics: admin-locked; exact sudo command documented in the harness docs.
