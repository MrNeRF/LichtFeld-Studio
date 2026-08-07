# Profile summary — `fix22-f3-bonsai-late`

- commit: `60d97b26`  dataset: `/home/gauss/data/360_v2/bonsai`
- slice: iters [1600, 1900] of 1920  strategy: mrnf  images: images_4
- GPU: NVIDIA GeForce RTX 4080  date: 2026-08-07T00:13:29Z

## Top kernels by total GPU time

| # | time % | total ms | count | avg µs | kernel |
|--:|-------:|---------:|------:|-------:|:-------|
| 1 | 58.9 | 629.41 | 300 | 2098.0 | `void fast_lfs::rasterization::kernels::backward::blend_backward_cu<(D…` |
| 2 | 16.3 | 174.17 | 300 | 580.6 | `void fast_lfs::rasterization::kernels::backward::preprocess_backward_…` |
| 3 | 11.7 | 125.19 | 300 | 417.3 | `void fast_lfs::rasterization::kernels::forward::blend_cu<(bool)0>(con…` |
| 4 | 3.3 | 35.25 | 38 | 927.7 | `edge_compute::rasterization::kernels::forward::edge_blend_cu(const ui…` |
| 5 | 1.8 | 19.70 | 300 | 65.7 | `fast_lfs::rasterization::kernels::forward::preprocess_cu(const float3…` |
| 6 | 1.5 | 15.86 | 300 | 52.9 | `void <unnamed>::fusedL1SSIMBackwardCUDA<unsigned char, __half>(float,…` |
| 7 | 1.4 | 14.53 | 1352 | 10.7 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortOneswe…` |
| 8 | 1.1 | 12.21 | 300 | 40.7 | `void <unnamed>::fusedL1SSIMForwardCUDA<unsigned char>(int, int, int, …` |
| 9 | 0.9 | 9.56 | 300 | 31.9 | `fast_lfs::rasterization::kernels::forward::create_instances_cu(const …` |
| 10 | 0.4 | 4.80 | 300 | 16.0 | `lfs::training::mrnf_strategy::fold_densification_and_zero_kernel(floa…` |
| 11 | 0.3 | 2.94 | 300 | 9.8 | `void lfs::training::kernels::<unnamed>::fused_l1_ssim_sum_kernel<unsi…` |
| 12 | 0.2 | 2.16 | 432 | 5.0 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortMergeK…` |
| 13 | 0.2 | 1.98 | 78 | 25.4 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortBlockS…` |
| 14 | 0.2 | 1.94 | 300 | 6.5 | `lfs::training::mrnf_strategy::mrnf_noise_injection_kernel(float *, co…` |
| 15 | 0.1 | 1.59 | 338 | 4.7 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortHistog…` |
| 16 | 0.1 | 1.44 | 300 | 4.8 | `void cub::_V_300304_SM_890::detail::scan::DeviceScanKernel<cub::_V_30…` |
| 17 | 0.1 | 1.23 | 432 | 2.8 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortPartit…` |
| 18 | 0.1 | 1.19 | 38 | 31.2 | `edge_compute::rasterization::kernels::forward::preprocess_cu(const fl…` |
| 19 | 0.1 | 1.16 | 38 | 30.6 | `edge_compute::rasterization::kernels::forward::create_instances_cu(co…` |
| 20 | 0.1 | 1.12 | 2 | 559.7 | `lfs::core::tensor_ops::<unnamed>::fused_segmented_transform_reduce_ke…` |

## Memory operations

| op | total ms | count | avg µs | total MB |
|:---|--------:|------:|-------:|---------:|
| [CUDA memset] | 5.50 | 4680 | 1.2 | 4543.390 |
| [CUDA memcpy Device-to-Host] | 1.28 | 751 | 1.7 | 6.491 |
| [CUDA memcpy Device-to-Device] | 0.84 | 316 | 2.7 | 338.292 |
| [CUDA memcpy Host-to-Device] | 0.55 | 134 | 4.1 | 6.395 |

## Launch-gap analysis (CUDA-graphs opportunity)

- capture window: **1141.5 ms**, GPU busy 1077.0 ms, **idle 64.4 ms (5.64%)**
- ops: 11110 kernels, 1201 memcpy, 4680 memset; 16976 busy islands

| gap bucket | count | total ms |
|:---|--:|--:|
| lt_2us | 12077 | 7.72 |
| 2_10us | 3056 | 16.67 |
| 10_100us | 1830 | 38.10 |
| gt_100us | 12 | 1.93 |

Per-iteration medians (NVTX `train_step` attribution):

- kernel launches/iter: **29**
- span 3492 µs = busy 3378 µs + gap **113 µs (3.2% of span)**
- iterations attributed: 300

