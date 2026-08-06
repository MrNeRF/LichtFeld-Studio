# Codec-pair kernel delta: 487d5c2b (parent) vs 63aa08c6 (joint Adam codec)

Feeds WO-G2 / FIX-2.2-directive. All numbers from `profile.sh` nsys captures on
this machine (RTX 4080, bonsai `images_4`, mrnf, max_cap 500k), 300-iteration
steady slices, identical hooks commit cherry-picked onto both commits.
Scene-size parity per slice was verified via N-proportional kernels
(`mrnf_noise_injection_kernel` within 0.2% between the two commits).

## preprocess_backward_cu per-launch time (300 launches per capture)

| slice | SH degree | N proxy (noise-kernel µs) | 487d5c2b avg µs | 63aa08c6 avg µs | delta |
|:--|:--|--:|--:|--:|--:|
| iters 200–500 | 0 | 442 | **179.3** (med 176.7) | **140.3** (med 136.0) | **−39.0 µs (−22%)** |
| iters 1600–1900 | ≥1 | 1324 | **450.0** (med 447.3) | **906.3** (med 891.7) | **+456.3 µs (+101%)** |

Profile dirs: `487d5c2b-bonsai`, `63aa08c6-bonsai`, `487d5c2b-bonsai-late`,
`63aa08c6-bonsai-late` (raw .nsys-rep on disk, gitignored).

## Findings

1. **The regression is SH-degree gated.** With the shN Adam section idle
   (active SH degree 0, iters < 1000), the joint (u,log_s) codec is actually
   ~22% FASTER per launch than the parent. Once shN coefficients are live
   (iter ≥ 1000), the kernel DOUBLES (450 → 906 µs). This is exactly
   consistent with FIX-2.2 root cause #1 — the register live-range explosion
   is in the unified shN Adam tail (shN_grads[15]x3 + us_u[48]/us_s[48] live
   across the reduction), not in the non-SH sections.
2. **Magnitude reconciles with the wall-clock regression.** Weighting the two
   slices over the bench steady window (iters 200–2000: ~800 degree-0 iters,
   ~1000 degree ≥1 iters) gives (800·(−39 µs) + 1000·(+456 µs)) / 1800 ≈
   **+0.24 ms/iter**, matching the +0.222 ms steady_ms regression the
   directive reports.
3. **Measurement-protocol implication for WO-G2:** gate/verify F-steps on a
   LATE window (e.g. `LFS_PROF_START=1600 LFS_PROF_STOP=1900 LFS_PROF_ITERS=1920
   ./perf_campaign/profile.sh timeline <label>`), not on early iterations —
   an early-window comparison will falsely show the codec as an improvement.
   F1 (running sh0+shN Adam inside the visible branch, before the geometry
   backward) attacks precisely the section this data implicates.
4. ncu per-kernel confirmation (registers/local-mem/barrier stalls) is
   blocked on admin-locked HW counters; `profile.sh ncu <label>
   'preprocess_backward'` prints the exact sudo command for the maintainer.
