# POPSpa viewport freeze investigation

The report in [PR #2043](https://github.com/MrNeRF/LichtFeld-Studio/pull/2043#issuecomment-5632738397)
describes Windows marking the UI unresponsive during POPSpa scoring. Investigation
found two defects in the shared CUDA/Vulkan integration. Both are fixed here;
the reporter's exact freeze remains unconfirmed without their reproduction.

## Bounded arena acquisition could block on GPU work

The viewport requests an arena frame with a 15 ms timeout. Previously that
timeout covered only CPU ownership; the subsequent streamless GPU handoff
could call `cudaDeviceSynchronize` on the GUI thread. A controlled pending
producer delayed a 15 ms request for about 1,009 ms. An unrelated CUDA stream
could cause the same delay even after the arena producer had completed.

Finite streamless acquisitions now query the producer event when sufficient,
or complete the existing handoff on a worker under the original deadline.
The worker retains the arena reservation until GPU dependencies complete.
Timeouts do not publish a frame, reset scratch storage, or release that
reservation. Later acquisitions consume completion or propagate failure.
Reset, move, and destruction drain the worker before transferring or freeing
its resources. Existing external-semaphore ordering is preserved.

With a producer pending for 1,000 ms, the same request now defers in about
15.2 ms. With only an unrelated stream pending, acquisition takes about
0.002 ms. These probes use finite sleeping CUDA host callbacks, not long
kernels or modified driver timeout settings. This is not a hard real-time
guarantee for every CUDA API or every GUI operation.

## Scratch aliases could retain retired Vulkan handles

Legacy projection aliases sort indices to depth keys and index offsets to
tile counts. Private-scratch cleanup previously skipped non-owning views,
leaving handles and capacity pointing at retired owner buffers. A later
survivor projection could reuse those handles without rebinding them.

Cleanup now invalidates both owners and views. Only owned allocations enter
the existing timeline retirement path; external parent allocations remain
untouched. An unpatched validation run reported invalid buffer handles.
A separate earlier run lost the Vulkan device, but the connection between
that device loss and these aliases was not proven.

## Validation

Local environment: Windows, RTX 5090 (32 GB), driver 595.79, CUDA 13.3,
MSVC 19.44. The local checkout also includes the previously reported COLMAP
calibration correction, which is outside this freeze-fix commit.

- Release application and dependent DLLs/Python extension rebuilt together.
- 21 arena/shared-scratch tests and 8 buffer-retirement tests passed through
  focused local runners using repository test sources and Release libraries.
- 9 Python VkSplat output-lifetime regression tests passed.
- Both new alias tests failed with the original cleanup behavior and passed
  with the fix. Arena tests cover deferred ownership, unrelated streams,
  moving completed handoffs, and reset during a pending handoff.
- Garden GUI trials with FastGS and GUT each reduced 2.56 million Gaussians
  to 512,000, with two scoring passes over 161 cameras. Each used shortened
  phases of 10 optimization and 10 refinement steps, not a quality benchmark.
- With Khronos validation active and the camera moving, the final FastGS
  trial captured 92 frames (88 distinct), and GUT captured 100 (94 distinct).
  Neither recorded validation errors, device loss, capture errors, or
  window-message timeouts through five seconds after completion.
- Formatting, Windows source preflight, and error-debt checks passed.

The full repository test suite was not run. Earlier profiling found no
whole-pass scoring freeze in these local conditions; it did observe one
recovering stall after scoring. Progress still remains on one training step
during scoring, and peak memory pressure on affected hardware remains an
open investigation. These changes address concrete integration defects,
without changing POPSpa's optimization algorithm or establishing that every
reported freeze has the same cause.

Raw traces, dumps, test logs, and the original chronological notes remain
local under ignored `build/popspa-ui-investigation/`.
