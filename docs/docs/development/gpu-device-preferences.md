# GPU selection and independent rendering

This branch starts from `4ef50cbbc7f649cfed1d76d1d1770a3bff62d27f` on `dev`, including the merged image-cache changes. It keeps the existing cache implementation. Hardware discovery and device selection are separate from tensor context initialization.

## User behavior

- **Hardware & GPUs** lists detected hardware, supported roles and driver-reported dedicated or shared memory. Unsupported choices explain why they cannot be selected.
- Training chooses a tensor backend and its device. The Vulkan viewer has a separate device selector. CUDA device selection uses the adapter UUID rather than assuming ordinal zero.
- Saved choices take effect on the next application start. The active configuration remains visible in Preferences and in the R/T status tooltips.
- A persistent **Restart needed** action opens the hardware preferences. Selecting the actual running configuration again clears it. Narrow windows use a compact action with the same tooltip.
- An unavailable device or backend falls back for the current run and logs the reason in the CLI. The saved request is retained, so a temporarily disconnected card can be selected on a later launch. Device menus list only enumerated hardware; an absent saved identity displays Auto without creating a synthetic option or overwriting the saved request.
- Auto preserves the existing CUDA default. An inactive Vulkan tensor preference does not redirect the viewer when CUDA or Metal is active.

Names are presentation only. Selection uses normalized UUIDs, with index aliases resolved against the current inventory. Identically named adapters include index and UUID suffixes in their labels. Integrated memory is identified as shared; its reported heap is not described as dedicated VRAM or as the BIOS reservation.

## Transfer and synchronization rules

Matching physical devices retain the existing direct interop path when its required external-memory and semaphore capabilities are enabled. A different viewer device uses a host transfer adapter. Tensor views preserve their byte offset and each exported lease owns an immutable snapshot.

Foreign Vulkan buffers are padded for packed shader loads, while their logical byte range and readback remain the original tensor storage size. The tests include float32, float16, short uint8 masks and offset views.

Consumer queues know whether their own context supports external interop. A headless or secondary context does not change the primary application's interop policy. A foreign Vulkan consumer uses a host wait instead of importing a semaphore from another device. Host waits have a bounded timeout and report failures.

CUDA selection is activated on worker threads as well as the main thread. Device information, decoder selection, streams, allocation and DLPack metadata follow the selected adapter. Linux Vulkan-enabled builds use dynamic CUDA driver lookup so an absent CUDA driver does not by itself prevent the application from loading.

## Validation protocol

Functional checks cover CUDA and Vulkan training, both Windows adapters, independent viewer devices, saved aliases, missing devices, fallback logs, pending edits and restoration of the active settings. Pure policy tests cover unavailable, reordered and duplicate adapters without relying on the test machine's particular inventory.

Training comparisons use one common checkpoint (SHA-256 `4273c94a3c29daffc84c490651510d91ac71ba9cfb9eba8aa3abc0fced3f63f7`), the same 82-image dataset, max width 1920, CPU image cache, SH0, MCMC, a one-million Gaussian cap and 2,000 steps. The timing window is steps 201–1900; final evaluation is excluded. Memory counters and process RSS sampled once per second are reported separately. Evaluation uses the same evaluator backend for each platform's comparison.

Parallel floating-point accumulation can produce different training results between repeats of the same build. Compare the candidate against repeated controls and inspect paired rendered images; do not interpret nonzero numerical differences alone as proof of a regression.

The per-step trace measures `train_step`; the data loader wait happens outside that span. Report the collector's separate data loader wait and the full process wall time as well. Phase device timestamps and host intervals describe asynchronous work and waits, so their columns must not be summed together as independent costs. The full process time also contains startup, checkpoint saving and final evaluation.

GPU memory counters have backend-specific scopes: CUDA derives usage from device total/free memory, Vulkan uses VMA device-local heap budget usage, and Metal reports its backend allocation statistics. Compare each backend with its own control. They are not interchangeable measurements of physical residency. RSS sampled once per second can miss transient peaks; do not add RSS and GPU counters on unified-memory hardware as if they were disjoint allocations.

## Reproduction checklist

1. Build Windows/Linux with `LFS_ENABLE_CUDA=ON`, `LFS_TENSOR_VULKAN=ON` and `LFS_BUILD_TRAINER=ON`. This is a combined build; disabling CUDA is not the deployment strategy. On macOS build both tensor backends and validate the Metal and Vulkan graphics configurations separately.
2. Build `gpu_configuration_policy_tests` and `gpu_cross_device_interop_tests`, then run `ctest --test-dir <build> -R "GpuConfigurationPolicy|GpuCrossDeviceInterop" --output-on-failure` from the build's normal runtime environment.
3. Repeat the Vulkan source transfer test in a separate process for each UUID using `LFS_TEST_VULKAN_DEVICE` and the filter `GpuCrossDeviceInterop.VulkanSnapshotsRoundTripThroughEveryCompatibleVulkanAdapter`. The test records the actual source and consumers; do not count a run that silently used Auto instead.
4. Run the Python preference/localization tests and the visualizer suite, including `StatusBarFitTest.*`. Inspect the actual application at normal and narrow widths, with a pending edit and after restoring the active settings.
5. Train with every available tensor backend and both matching and different rendering devices. Repeat with an unavailable saved UUID, backend fallback and an inactive saved Vulkan selection. Confirm the active hardware, saved request and CLI reason independently.
6. Compare fresh control and candidate runs with the common training parameters above. Use an identical evaluator for each pair. Include repeated controls, per-image metrics and visual crops; inspect face detail, silhouettes, foliage and background rather than relying only on average PSNR.
7. Check the combined Linux application's complete ELF dependency closure for eager `libcuda` dependencies and test startup with driver loading disabled for that process. Hiding CUDA with `CUDA_VISIBLE_DEVICES=-1` tests an unavailable device, not an absent driver installation.

For independent-device viewer tests, open the same checkpoint on each configuration and capture the same dataset camera. Check that offscreen captures do not change the primary R/T hardware status. Keep benchmark outputs separate from projects used for further training so validation cannot overwrite a control checkpoint.

The local validation artifacts live under `.local/gpu-backend-preferences`. Detailed final measurements and platform limitations must be recorded there after all runs complete. One physical CUDA adapter is available on the Windows test machine; simulated policy tests cannot substitute for testing two physical CUDA adapters. WSL validation cannot establish native Radeon Linux performance.

The measured results and exact platform coverage are in [the validation report](gpu-device-validation-2026-10-18.md). Builds without Vulkan tensor support preserve an inactive saved Vulkan device choice; startup sanitization only validates devices for compiled backends.
