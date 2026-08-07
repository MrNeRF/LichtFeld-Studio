Implementation engineer, LichtFeld-Studio. Dir: /home/gauss/projects/LichtFeld-Studio, branch lfs-elite (verify; NEVER checkout). Reads: perf_campaign/RULES.md, perf_campaign/DIRECTIVES-round2.md §Directive-3 (measured: single-threaded nvjpeg HOST Huffman decode 4.8ms/image = 73% of bicycle window; H2D itself is fine).
TASK (rank-3 Wave 5), two parts IN ORDER:
1. **Metric first**: add `dataloader_wait_ms` (time spent in dataloader->next() around trainer.cpp:~5984) to perf_bench.json + bench.sh table — steady_ms is blind to this cost. Record the CURRENT baseline numbers for both workloads with it.
2. **Decoded-GT device cache**: cache decoded u8 CHW GT tensors per image key on device at first decode (bicycle images_4 = 592MB, bonsai 356MB); `next()` returns resident tensor from epoch 2 (zero decode, zero H2D). VRAM-budget gate: enable only if n_images*bytes < min(free_vram - 2GB headroom, (purged-env:GT_CACHE_CAP) override); pinned-host cache + async H2D as middle tier; today's path as fallback. Implement at PipelinedImageLoader/GT-fetch level (pipelined_image_loader.cpp:345 area).
TDD: cache-hit path returns bit-identical tensors (hash test); budget-gate tests; eviction/fallback test.
GATE: dual-workload with the NEW metric — expect bicycle 7k wall −30-45% and dataloader_wait -> ~0 from epoch 2; steady_ms unchanged; VRAM within budget (report peak). Commit with numbers.

PRECONDITION — MAIN-CHECKOUT LOCK: before touching any file, acquire the main-checkout lock and hold it for your whole session: run all your work under `flock /home/gauss/lfs-campaign-out/main-checkout.lock <your commands>` or simply verify no other worker holds it (flock -n test; if busy, sleep 120 and retry). Only one worker may operate in /home/gauss/projects/LichtFeld-Studio at a time.

VRAM ACCOUNTING (mandatory): the GT cache spends VRAM to buy wall-time. It must appear as
its own line in the VRAM ledger/bench output (gt_cache_MiB) and count into peak_VRAM in
perf_bench.json. The budget gate must leave >= 2GB free VRAM headroom above training peak.
Report the ledger with cache on AND off in your final table. Memory axis is co-equal: this
order is acceptable ONLY because it is budget-gated, observable, and reversible.
