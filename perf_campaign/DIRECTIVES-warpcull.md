# Warp-level blend optimization directive (from ACM CGIT 9(4):54, DOI 10.1145/3820019 — local PDF 3820019.pdf)
Paper optimizes the Faster-GS blend_cu (our FastGS shares the lineage: identical stage
structure, 16x16 tiles, 256-batches, two-stage fetch/render with block fences). Their
measured bottlenecks match OUR Fable-directive findings independently: alpha-eval
instruction dominance + block-wide fence stalls + 96-bit color loads.

Three techniques (all bit-identical output, differentiable-pipeline compatible):
1. WARP SUB-TILE CULLING: subdivide 16x16 tile into 8x4 sub-tiles (32 px = 1 warp,
   1 px/thread). Warp iterates splat batches 32-at-a-time; each thread bbox-tests one
   splat vs the whole sub-tile; __ballot_sync builds a 32-bit intersect mask; threads
   skip non-intersecting splats entirely (no full data pull, no alpha eval). Warp
   lock-step => ballot "fence" costs nothing (unlike block fences).
2. BATCH SIZE: sweep 32..256 in steps of 32. Paper: 192 optimal on 4090/3090 (~+5-8%),
   256 on 2070S. We are RTX 4080 — sweep and pick per-arch.
3. 128-BIT LAYOUT: color float3 (96-bit, no optimized load) -> padded float4; pack
   bbox as 4x u16 with the 64-bit screen position into one 128-bit vector.
Paper results vs Faster-GS: frame time −17% geomean (−24% on weakest GPU), FLOPs −31..56%.
Training port explicitly left open by the paper — that is OUR edge: apply the same
warp-cull structure to blend_backward_cu (complementary to the landed T_eff clamp:
clamp bounds the RANGE, warp-cull skips WITHIN range; also replaces block fences with
warp-scoped sync = exactly Fable root-cause #2, 30 barriers).
Citation policy (maintainer-approved): commit messages and code comments MAY cite this
paper explicitly: "Warp-Level Culling for Efficient Blending in 3D Gaussian Splatting,
Yang, Drettakis, Bernstein — ACM CGIT 9(4):54, 2026, doi:10.1145/3820019". Cite it in the
kernel-head comment and in the landing commits for WO-WARP-FWD / WO-WARP-BWD.
(The no-external-names rule remains in force for everything else.)
