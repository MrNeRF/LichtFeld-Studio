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
