# Profile summary — `fix22-f1-bonsai-late`

- commit: `1a647af7`  dataset: `/home/gauss/data/360_v2/bonsai`
- slice: iters [1600, 1900] of 1920  strategy: mrnf  images: images_4
- GPU: NVIDIA GeForce RTX 4080  date: 2026-08-07T00:06:28Z

## Top kernels by total GPU time

| # | time % | total ms | count | avg µs | kernel |
|--:|-------:|---------:|------:|-------:|:-------|
| 1 | 57.9 | 624.00 | 300 | 2080.0 | `void fast_lfs::rasterization::kernels::backward::blend_backward_cu<(D…` |
| 2 | 17.3 | 186.59 | 300 | 622.0 | `void fast_lfs::rasterization::kernels::backward::preprocess_backward_…` |
| 3 | 11.6 | 124.80 | 300 | 416.0 | `void fast_lfs::rasterization::kernels::forward::blend_cu<(bool)0>(con…` |
| 4 | 3.4 | 36.75 | 38 | 967.0 | `edge_compute::rasterization::kernels::forward::edge_blend_cu(const ui…` |
| 5 | 1.8 | 19.61 | 300 | 65.4 | `fast_lfs::rasterization::kernels::forward::preprocess_cu(const float3…` |
| 6 | 1.5 | 15.86 | 300 | 52.9 | `void <unnamed>::fusedL1SSIMBackwardCUDA<unsigned char, __half>(float,…` |
| 7 | 1.4 | 14.62 | 1352 | 10.8 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortOneswe…` |
| 8 | 1.1 | 12.22 | 300 | 40.7 | `void <unnamed>::fusedL1SSIMForwardCUDA<unsigned char>(int, int, int, …` |
| 9 | 0.9 | 9.74 | 300 | 32.5 | `fast_lfs::rasterization::kernels::forward::create_instances_cu(const …` |
| 10 | 0.4 | 4.45 | 300 | 14.8 | `lfs::training::mrnf_strategy::fold_densification_and_zero_kernel(floa…` |
| 11 | 0.3 | 2.94 | 300 | 9.8 | `void lfs::training::kernels::<unnamed>::fused_l1_ssim_sum_kernel<unsi…` |
| 12 | 0.2 | 2.16 | 432 | 5.0 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortMergeK…` |
| 13 | 0.2 | 1.98 | 78 | 25.3 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortBlockS…` |
| 14 | 0.2 | 1.92 | 300 | 6.4 | `lfs::training::mrnf_strategy::mrnf_noise_injection_kernel(float *, co…` |
| 15 | 0.1 | 1.57 | 338 | 4.7 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortHistog…` |
| 16 | 0.1 | 1.49 | 300 | 5.0 | `void cub::_V_300304_SM_890::detail::scan::DeviceScanKernel<cub::_V_30…` |
| 17 | 0.1 | 1.23 | 432 | 2.8 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortPartit…` |
| 18 | 0.1 | 1.21 | 38 | 31.9 | `edge_compute::rasterization::kernels::forward::create_instances_cu(co…` |
| 19 | 0.1 | 1.19 | 38 | 31.4 | `edge_compute::rasterization::kernels::forward::preprocess_cu(const fl…` |
| 20 | 0.1 | 1.12 | 2 | 559.5 | `lfs::core::tensor_ops::<unnamed>::fused_segmented_transform_reduce_ke…` |

## Memory operations

| op | total ms | count | avg µs | total MB |
|:---|--------:|------:|-------:|---------:|
| [CUDA memset] | 5.56 | 4680 | 1.2 | 4543.163 |
| [CUDA memcpy Device-to-Host] | 1.26 | 751 | 1.7 | 6.499 |
| [CUDA memcpy Device-to-Device] | 0.77 | 316 | 2.5 | 338.212 |
| [CUDA memcpy Host-to-Device] | 0.55 | 134 | 4.1 | 6.398 |

## Launch-gap analysis (CUDA-graphs opportunity)

- capture window: **1131.4 ms**, GPU busy 1085.0 ms, **idle 46.3 ms (4.1%)**
- ops: 11110 kernels, 1201 memcpy, 4680 memset; 16975 busy islands

| gap bucket | count | total ms |
|:---|--:|--:|
| lt_2us | 12792 | 8.53 |
| 2_10us | 2870 | 15.65 |
| 10_100us | 1305 | 21.21 |
| gt_100us | 7 | 0.94 |

Per-iteration medians (NVTX `train_step` attribution):

- kernel launches/iter: **29**
- span 3495 µs = busy 3397 µs + gap **95 µs (2.7% of span)**
- iterations attributed: 300

