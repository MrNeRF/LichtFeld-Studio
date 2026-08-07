# Profile summary — `fix22-f2-bonsai-late`

- commit: `a93cd668`  dataset: `/home/gauss/data/360_v2/bonsai`
- slice: iters [1600, 1900] of 1920  strategy: mrnf  images: images_4
- GPU: NVIDIA GeForce RTX 4080  date: 2026-08-07T00:09:37Z

## Top kernels by total GPU time

| # | time % | total ms | count | avg µs | kernel |
|--:|-------:|---------:|------:|-------:|:-------|
| 1 | 58.4 | 627.65 | 300 | 2092.2 | `void fast_lfs::rasterization::kernels::backward::blend_backward_cu<(D…` |
| 2 | 16.8 | 181.15 | 300 | 603.8 | `void fast_lfs::rasterization::kernels::backward::preprocess_backward_…` |
| 3 | 11.7 | 125.86 | 300 | 419.5 | `void fast_lfs::rasterization::kernels::forward::blend_cu<(bool)0>(con…` |
| 4 | 3.4 | 36.08 | 38 | 949.4 | `edge_compute::rasterization::kernels::forward::edge_blend_cu(const ui…` |
| 5 | 1.8 | 19.75 | 300 | 65.8 | `fast_lfs::rasterization::kernels::forward::preprocess_cu(const float3…` |
| 6 | 1.5 | 15.86 | 300 | 52.9 | `void <unnamed>::fusedL1SSIMBackwardCUDA<unsigned char, __half>(float,…` |
| 7 | 1.4 | 14.60 | 1352 | 10.8 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortOneswe…` |
| 8 | 1.1 | 12.20 | 300 | 40.7 | `void <unnamed>::fusedL1SSIMForwardCUDA<unsigned char>(int, int, int, …` |
| 9 | 0.9 | 9.56 | 300 | 31.9 | `fast_lfs::rasterization::kernels::forward::create_instances_cu(const …` |
| 10 | 0.4 | 4.44 | 300 | 14.8 | `lfs::training::mrnf_strategy::fold_densification_and_zero_kernel(floa…` |
| 11 | 0.3 | 2.95 | 300 | 9.8 | `void lfs::training::kernels::<unnamed>::fused_l1_ssim_sum_kernel<unsi…` |
| 12 | 0.2 | 2.16 | 432 | 5.0 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortMergeK…` |
| 13 | 0.2 | 1.98 | 78 | 25.4 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortBlockS…` |
| 14 | 0.2 | 1.93 | 300 | 6.4 | `lfs::training::mrnf_strategy::mrnf_noise_injection_kernel(float *, co…` |
| 15 | 0.1 | 1.54 | 338 | 4.6 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortHistog…` |
| 16 | 0.1 | 1.50 | 300 | 5.0 | `void cub::_V_300304_SM_890::detail::scan::DeviceScanKernel<cub::_V_30…` |
| 17 | 0.1 | 1.23 | 432 | 2.8 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortPartit…` |
| 18 | 0.1 | 1.19 | 38 | 31.2 | `edge_compute::rasterization::kernels::forward::preprocess_cu(const fl…` |
| 19 | 0.1 | 1.16 | 38 | 30.5 | `edge_compute::rasterization::kernels::forward::create_instances_cu(co…` |
| 20 | 0.1 | 1.12 | 2 | 559.1 | `lfs::core::tensor_ops::<unnamed>::fused_segmented_transform_reduce_ke…` |

## Memory operations

| op | total ms | count | avg µs | total MB |
|:---|--------:|------:|-------:|---------:|
| [CUDA memset] | 5.56 | 4680 | 1.2 | 4543.177 |
| [CUDA memcpy Device-to-Host] | 1.28 | 751 | 1.7 | 6.488 |
| [CUDA memcpy Device-to-Device] | 0.78 | 316 | 2.5 | 338.257 |
| [CUDA memcpy Host-to-Device] | 0.55 | 134 | 4.1 | 6.394 |

## Launch-gap analysis (CUDA-graphs opportunity)

- capture window: **1129.9 ms**, GPU busy 1083.5 ms, **idle 46.4 ms (4.11%)**
- ops: 11110 kernels, 1201 memcpy, 4680 memset; 16975 busy islands

| gap bucket | count | total ms |
|:---|--:|--:|
| lt_2us | 12798 | 8.51 |
| 2_10us | 2845 | 15.73 |
| 10_100us | 1326 | 21.64 |
| gt_100us | 5 | 0.53 |

Per-iteration medians (NVTX `train_step` attribution):

- kernel launches/iter: **29**
- span 3465 µs = busy 3368 µs + gap **96 µs (2.8% of span)**
- iterations attributed: 300

