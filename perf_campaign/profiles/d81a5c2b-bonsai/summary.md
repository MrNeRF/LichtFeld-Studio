# Profile summary — `d81a5c2b-bonsai`

- commit: `d81a5c2b`  dataset: `/home/gauss/data/360_v2/bonsai`
- slice: iters [200, 500] of 520  strategy: mrnf  images: images_4
- GPU: NVIDIA GeForce RTX 4080  date: 2026-08-06T21:53:00Z

## Top kernels by total GPU time

| # | time % | total ms | count | avg µs | kernel |
|--:|-------:|---------:|------:|-------:|:-------|
| 1 | 55.5 | 487.79 | 300 | 1626.0 | `void fast_lfs::rasterization::kernels::backward::blend_backward_cu<(D…` |
| 2 | 15.0 | 131.53 | 300 | 438.4 | `lfs::training::mrnf_strategy::mrnf_noise_injection_kernel(float *, co…` |
| 3 | 10.3 | 90.24 | 300 | 300.8 | `void fast_lfs::rasterization::kernels::forward::blend_cu<(bool)0>(con…` |
| 4 | 4.7 | 41.71 | 300 | 139.0 | `void fast_lfs::rasterization::kernels::backward::preprocess_backward_…` |
| 5 | 4.5 | 39.51 | 38 | 1039.7 | `edge_compute::rasterization::kernels::forward::edge_blend_cu(const ui…` |
| 6 | 1.8 | 15.80 | 300 | 52.7 | `void <unnamed>::fusedL1SSIMBackwardCUDA<unsigned char, __half>(float,…` |
| 7 | 1.4 | 12.71 | 1352 | 9.4 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortOneswe…` |
| 8 | 1.4 | 12.15 | 300 | 40.5 | `void <unnamed>::fusedL1SSIMForwardCUDA<unsigned char>(int, int, int, …` |
| 9 | 1.0 | 8.37 | 300 | 27.9 | `fast_lfs::rasterization::kernels::forward::create_instances_cu(const …` |
| 10 | 0.6 | 5.70 | 300 | 19.0 | `fast_lfs::rasterization::kernels::forward::preprocess_cu(const float3…` |
| 11 | 0.3 | 2.94 | 300 | 9.8 | `void lfs::training::kernels::<unnamed>::fused_l1_ssim_sum_kernel<unsi…` |
| 12 | 0.3 | 2.75 | 300 | 9.2 | `void nvjpeg::dctQuantInvJpegKernelMultiChannel<ushort2, (int)1, (int)…` |
| 13 | 0.3 | 2.32 | 300 | 7.7 | `lfs::io::cuda::uint8_hwc_to_uint8_chw_kernel(const unsigned char *, u…` |
| 14 | 0.2 | 2.03 | 300 | 6.8 | `lfs::training::mrnf_strategy::fold_densification_and_zero_kernel(floa…` |
| 15 | 0.2 | 1.97 | 409 | 4.8 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortMergeK…` |
| 16 | 0.2 | 1.96 | 78 | 25.1 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortBlockS…` |
| 17 | 0.2 | 1.68 | 300 | 5.6 | `void nvjpeg::ycbcr_to_format_kernel_roi<(nvjpegChromaSubsampling_t)0,…` |
| 18 | 0.1 | 1.17 | 300 | 3.9 | `void cub::_V_300304_SM_890::detail::scan::DeviceScanKernel<cub::_V_30…` |
| 19 | 0.1 | 1.14 | 409 | 2.8 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortPartit…` |
| 20 | 0.1 | 1.12 | 338 | 3.3 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortHistog…` |

## Memory operations

| op | total ms | count | avg µs | total MB |
|:---|--------:|------:|-------:|---------:|
| [CUDA memcpy Host-to-Device] | 58.27 | 394 | 147.9 | 735.415 |
| [CUDA memset] | 4.66 | 4648 | 1.0 | 2842.077 |
| [CUDA memcpy Device-to-Host] | 0.90 | 652 | 1.4 | 1.464 |
| [CUDA memcpy Device-to-Device] | 0.81 | 284 | 2.9 | 257.846 |

## Launch-gap analysis (CUDA-graphs opportunity)

- capture window: **1682.2 ms**, GPU busy 927.2 ms, **idle 755.1 ms (44.88%)**
- ops: 11849 kernels, 1330 memcpy, 4648 memset; 17348 busy islands

| gap bucket | count | total ms |
|:---|--:|--:|
| lt_2us | 8039 | 5.42 |
| 2_10us | 5070 | 27.84 |
| 10_100us | 3799 | 103.34 |
| gt_100us | 439 | 618.47 |

Per-iteration medians (NVTX `train_step` attribution):

- kernel launches/iter: **29**
- span 3874 µs = busy 2735 µs + gap **819 µs (21.1% of span)**
- iterations attributed: 300

