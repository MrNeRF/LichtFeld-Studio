# L0: Does NVML per-process compute memory include Vulkan allocations?

**Machine:** NVIDIA GeForce RTX 4090, driver 580.173.02, Linux, CUDA 13.0 toolkit headers / runtime as linked.
**Date:** 2026-08-10
**API under test:** `nvmlDeviceGetComputeRunningProcesses` (same entry the app uses via `nvmlDeviceGetComputeRunningProcesses_v3` in `gpu_memory_query.cpp`). Cross-checked against `nvmlDeviceGetGraphicsRunningProcesses`.

## Answer

**Yes. On this machine, NVML compute-process `usedGpuMemory` includes Vulkan `DEVICE_LOCAL` allocations.**

Therefore **root F (Vulkan VMA blocks / device-local `VkDeviceMemory`) BELONGS in the ledger sum** whose denominator is NVML per-PID `process_used`.

Host-visible (non-device-local) Vulkan memory is **not** included in NVML process VRAM and must stay out of root F.

## Method

Standalone probe `.codex_tmp/hud/l0_nvml_probe.cpp` (source + binary kept under that dir):

1. `cudaSetDevice(0)` + tiny `cudaMalloc` to force a CUDA primary context (so the PID appears in compute processes).
2. Sample NVML compute + graphics per-PID memory and `cudaMemGetInfo` used.
3. Create a Vulkan instance/device on the same NVIDIA GPU.
4. `vkAllocateMemory` **512 MiB** pure `DEVICE_LOCAL` (memory type 1, heap 0).
5. `vkAllocateMemory` **128 MiB** `HOST_VISIBLE|HOST_COHERENT` (memory type 2, heap 1, **not** device-local).
6. `cudaMalloc` **256 MiB** as positive control.
7. Resample after each step; compare deltas.

Build/run:

```bash
g++ -O2 -std=c++17 l0_nvml_probe.cpp -o l0_nvml_probe \
  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
  -lcudart -ldl -lvulkan -lnvidia-ml
./l0_nvml_probe
```

Full log: `.codex_tmp/hud/l0_nvml_probe_run.log`

## Measured deltas (MiB)

| step | d(nvml_compute) | d(nvml_graphics) | d(cudaMemGetInfo) |
|---|---:|---:|---:|
| +512 MiB Vulkan DEVICE_LOCAL | **+512.00** | **+512.00** | **+512.00** |
| +128 MiB Vulkan HOST_VISIBLE (heap 1) | **+0.00** | **+0.00** | **+0.00** |
| +256 MiB cudaMalloc (control) | **+256.00** | **+256.00** | **+256.00** |

Absolute samples (MiB):

| sample | nvml_compute | nvml_graphics | cudaMemGetInfo_used |
|---|---:|---:|---:|
| after CUDA context | 386.00 | 0.00 | 978.75 |
| after vkCreateDevice | 395.91 | 395.91 | 1006.00 |
| after +512 DEVICE_LOCAL | 907.91 | 907.91 | 1518.00 |
| after +128 HOST_VISIBLE | 907.91 | 907.91 | 1518.00 |
| after +256 cudaMalloc | 1163.91 | 1163.91 | 1774.00 |

Notes:

- Before any Vulkan device, the process is only in the compute list (graphics = 0). After `vkCreateDevice`, compute and graphics report the **same** unified footprint for this PID.
- The DEVICE_LOCAL allocation is tracked 1:1 by NVML compute, NVML graphics, and `cudaMemGetInfo` (device-wide free/total), matching the allocated size exactly on this driver.
- Host-visible heap memory (type 2/3 on heap 1) is not device-local and does not move NVML process VRAM or `cudaMemGetInfo` used.

## Spine decision (feeds L2)

| root | in NVML-denominator sum? |
|---|---|
| A CUDA async pool reserved | yes |
| B CUDA slab reserved | yes |
| C CUDA direct | yes |
| D Rasterizer arena | yes |
| E Exportable VMM | yes |
| **F Vulkan VMA blocks (device-local only)** | **yes** |
| G Vulkan raw `vkAllocateMemory` outside VMA (device-local) | yes (same evidence) |
| H CUDA context / driver-opaque | yes (already in NVML) |
| I Unattributed residual | derived |

**Root F must sum only DEVICE_LOCAL block bytes**, matching what `VulkanContext::queryVmaUsedBytes` already does (`memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT`). Do not add host-visible VMA blocks.

## Secondary: `vulkan.vksplat.readback_buffer/shared`

On this GPU the pure host-visible type is heap 1 (not device-local). A buffer created with `VMA_MEMORY_USAGE_AUTO_PREFER_HOST` will land on heap 1 and is **not** in NVML process VRAM. Excluding host-visible heaps (already done for the budget sum) keeps root F correct; no extra spine change.

## Platform note

Windows / DXGI was not measured on this lane (no Windows host available). Linux answer is locked: root F stays in the sum.
