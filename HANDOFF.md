# Campaign Handoff — Speed/VRAM + GUI-hardening (2026-08-06 → 2026-08-08, evening)

Branch **lfs-elite** (local). Published mirror: origin/lfs-elite @ **69efab99** (2026-08-08 ~13:00,
content-stripped via ./push-clean.sh — owner-run). INTERNAL doc, stripped at publish.
Evidence: perf_campaign/{PROGRESS,ISSUES,RULES}.md, receipts ~/lfs-campaign-out/ (logs kept,
binary payloads cleaned 08-08 evening; stale worktrees deleted, ~181 GB freed).

## 1. Results (measured, receipts in PROGRESS)

| Metric | Campaign start | Now (local tip) |
|---|---:|---:|
| Bonsai steady | 4.129 ms/iter | ~2.61 (−37%) |
| Bicycle-7k steady | 3.290 ms/iter | ~2.65 (wall −34%) |
| Training state | 429 B/splat | 307.4 |
| Quality 30k/5M (PSNR) | bonsai 33.05 / bicycle 24.92 | 32.67 (variance) / 25.00 (better) |
| GUI VRAM, bicycle images_4 mrnf | ~5.1 GiB (08-08 morning) | **2.2 GiB measured** (2-gen scale); 5M projection ~3.5 |
| GUI GT cache | 5.4 GiB device | 0 device + pinned/prefetch (dl_wait unchanged) |
| Suite | 14 chronic reds | 3440 PASS; 13 env-only reds (green in desktop env) |
| Env flags | 20 campaign flags | 0 (optimized paths only; --perf-bench/--profile-window CLI) |

GUI crash lineage fixed & gated: ISS-022 mask freeze, ISS-023 capacity hook, ISS-025 grow-rebind
wipe, ISS-026 pinned teardown, ISS-027 layer-1. Permanent forced-collision tests in suite.

## 2. OPEN CRITICAL PATH — the q16 always-commit fault (one bug left)

GUI + always-commit q16 SH faults ~iter 1001 (5th densify / first degree-up). Currently masked
on the branch by the float-densify window (safe but −490 MiB during refine + stop_refine
landmine ~15k, NEVER gate-crossed on the PUBLISHED tip — warn owner off long GUI runs there).

**Falsification ledger (all with receipts):** NOT stale-pointers-post-grow (generation-checked
fetches landed, 5e4fb453..), NOT layout-generation mismatch, NOT Scene-cache rebuild race
(one-lock complete c267d3ae/26489b80: ctor-bound mutex, fence-before-release, combined hold
across commit+trim, training cache-rebuild skip), NOT Vulkan import, NOT preview try-lock
(verified skip-and-retain), NOT headless/unit reachable.

**Hunter B static audit (hunt/static-audit.md, M1 ~0.45):** the lock is enforced per CALL SITE
and only on `refining` branches; unlocked live-model mutations exist — mrnf.cpp:735 degree bump
on non-refining steps; mrnf.cpp:744-749 UNLOCKED float→q16 shN swap at stop_refine. Fix
directive: RAII ShMutationGuard INSIDE the mutation helpers (exclusive + combined +
waitForModelReaders, depth-counted) + debug assert lock-held (names guilty sites). Runners-up
M2 (densify-barrier host sync vs in-flight viewer GPU work), M3 (render-thread storage
retirement vs epoch pinning). Fix-regardless: get_bucket_size unbraced-return bug (2-byte
bucket for >8 GiB reqs, size_bucketed_pool.hpp ~66-73).
**Hunter A dynamic capture: IN FLIGHT** (sanitizer trace to corroborate M1). Then: grok worker
implements per WO to be written from A+B convergence; gate = rock-solid bar (ISSUES addendum)
incl. stop_refine-crossing run + sanitizer reaching iter 1001+ + SH degrees 0-3.

## 3. Remaining to done
1. Hunter A trace → fix WO → grok implements → full gate.
2. Supervisor review vs rock-solid bar; then TRUE 5M acceptance (images_4, GUI mrnf, per-second
   peak; target <= 4.0 GiB; projection ~3.5). Nothing dataset/resolution-specific (verified by
   3-point scaling tables in gtzero/attrf16 receipts).
3. push-clean.sh <rev> (owner-run; NO content rewriting — that sed was removed after it broke
   the published build once). Update .100 worktree ~/projects/lfs-elite-test (env vars needed:
   VCPKG_ROOT=/home/paja/projects/vcpkg, CUDA 12.8 PATH; owner's main checkout there is on
   their own feature branch — do not touch).
4. Owner verification on .100: GUI mrnf session past densify AND past stop_refine.
5. Deferred backlog: HANDOFF-era §4 items (graphs/WDDM, HP-2, per-band SH bits, themes B-F),
   viewer q16 zero-copy f16 fallback polish, ISS-028 ledger dedupe (minor).

## 4. Ops runbook (hard-won; all traps hit at least once)
- Workers: grok CLI via systemd-run, lfs.slice, MemoryHigh=12G MemoryMax=20G, RuntimeMaxSec=8h.
  Fable forks: analysis/supervision only (owner directive; grok implements).
- BUILD FIRST via ./perf_campaign/build.sh before any bench/ctest (unbounded rebuild OOM-killed
  two workers). GPU timing under flock /tmp/lfs-bench.lock, quiet machine.
- TRAPS: (1) never `pkill -f LichtFeld-Studio` — self-kill via own cmdline; use PID or
  'build/LichtFeld-Studio'. (2) watchdog.sh pgrep matched unrelated WOs by filename suffix
  (WO-...L.md ≡ fleetL) — patched to skip .stalled; never name WOs ending in old fleet letters.
  (3) NEVER write run outputs to the session /tmp scratchpad — quota bricked tooling and forged
  a false PSNR; use ~/lfs-campaign-out/. (4) "killed" background tasks often completed — check
  logs before rerunning. (5) verify a "clean" run by ITERATION TRACE, not the plan line (one
  worker misread "1500 planned" as success). (6) GUI hang diagnosis without ptrace: `ps -o
  wchan=` — drm_syncobj_array_wait_timeout = Vulkan semaphore never signaled.
- GATES that exist because each one caught something real: GUI repro past densify at DEFAULT
  reserve (max-cap 500k skips grow!), multi-resolution scaling smokes, stop_refine crossing,
  sanitizer reaching the fault window, red-provenance (git-log proof), receipts-or-it-didn't-
  happen (one worker claimed an unrun sweep).
- Publication: push-clean strips internals (perf_campaign, HANDOFF, plan docs) and Claude
  trailers; owner deletes the remote branch during testing — just re-push, don't investigate.

## 5. Lessons that must survive (added to the 08-07 list)
6. Locks enforced at call sites rot; enforce INSIDE the mutation (RAII), assert-held in debug.
7. A gate is only as good as the parameters of its repro — under-specified repros passed five
   broken builds; write iteration counts and reserve sizes INTO the order.
8. Falsification ledgers beat confidence: six wrong mechanisms died to cheap discriminating
   experiments; every "obvious" fix would have shipped still-broken.
9. When two independent analyses converge (static + dynamic), act; when they diverge, the
   experiment is under-specified.

## 6. 2026-08-09 ~02:20 — ISS-029 CLOSED; branch published
The §2 critical path is DONE. Root cause was none of the §2 corridors: the backward kernel
decoded shN-rest with fused-Adam's enablement-gated sh_value_* (null through SH warmup 1000 ==
default degree interval) → q16 u16 codes read as fp32 float4-swizzle → ~3x overread off the
exportable block's committed pages → Warp MMU fault (GUI) / silent bad SH gradients (headless,
mapped arena). Fix 9806cd89 (explicit model-truth decode binds through backward_raw, mirroring
forward). Plus grok's WO-FIX-Q16-GUARD1 Parts A (LiveModelMutationGuard 6b95b121) and B
(bucket-127 fc088459). Full mechanism + falsification history: ISSUES.md ISS-029 addendum;
gate table: PROGRESS.md (deferred: debug-assert sweep, live-GUI crop MCP variant).
5M acceptance: 30k iters / 5.0M gaussians / 615.7s GUI mrnf images_4, zero errors; VRAM steady
3482 MiB flat (<4096 ✓), nine 1-second silent ~1.06 GiB transients peak 4546 (ISS-030 minor
follow-up). Published 9f84a117 -> origin/lfs-elite (remote had been deleted again — re-pushed
per runbook). .100 verification staged: ~/lfs-campaign-out/q16m1/verify-100.sh (supervisor
session lacks ssh permission; owner runs it or grants Bash(ssh:*)).
