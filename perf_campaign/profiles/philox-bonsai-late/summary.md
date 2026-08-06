# Profile summary — `philox-bonsai-late`

- commit: `cd9f4c92`  dataset: `/home/gauss/data/360_v2/bonsai`
- slice: iters [1600, 1900] of 1920  strategy: mrnf  images: images_4
- GPU: NVIDIA GeForce RTX 4080  date: 2026-08-06T22:36:00Z

## Top kernels by total GPU time

| # | time % | total ms | count | avg µs | kernel |
|--:|-------:|---------:|------:|-------:|:-------|
| 1 | 53.2 | 629.39 | 300 | 2098.0 | `void fast_lfs::rasterization::kernels::backward::blend_backward_cu<(D…` |
| 2 | 23.9 | 282.59 | 300 | 942.0 | `void fast_lfs::rasterization::kernels::backward::preprocess_backward_…` |
| 3 | 10.5 | 124.68 | 300 | 415.6 | `void fast_lfs::rasterization::kernels::forward::blend_cu<(bool)0>(con…` |
| 4 | 2.9 | 34.78 | 38 | 915.2 | `edge_compute::rasterization::kernels::forward::edge_blend_cu(const ui…` |
| 5 | 1.5 | 18.25 | 300 | 60.8 | `fast_lfs::rasterization::kernels::forward::preprocess_cu(const float3…` |
| 6 | 1.4 | 16.03 | 300 | 53.4 | `void <unnamed>::fusedL1SSIMBackwardCUDA<unsigned char, __half>(float,…` |
| 7 | 1.2 | 14.61 | 1352 | 10.8 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortOneswe…` |
| 8 | 1.0 | 12.22 | 300 | 40.7 | `void <unnamed>::fusedL1SSIMForwardCUDA<unsigned char>(int, int, int, …` |
| 9 | 0.8 | 9.39 | 300 | 31.3 | `fast_lfs::rasterization::kernels::forward::create_instances_cu(const …` |
| 10 | 0.4 | 4.68 | 300 | 15.6 | `void nvjpeg::dctQuantInvJpegKernelMultiChannel<ushort2, (int)1, (int)…` |
| 11 | 0.4 | 4.52 | 300 | 15.1 | `lfs::training::mrnf_strategy::fold_densification_and_zero_kernel(floa…` |
| 12 | 0.2 | 2.95 | 300 | 9.8 | `void lfs::training::kernels::<unnamed>::fused_l1_ssim_sum_kernel<unsi…` |
| 13 | 0.2 | 2.68 | 300 | 8.9 | `lfs::io::cuda::uint8_hwc_to_uint8_chw_kernel(const unsigned char *, u…` |
| 14 | 0.2 | 2.16 | 432 | 5.0 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortMergeK…` |
| 15 | 0.2 | 1.97 | 78 | 25.3 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortBlockS…` |
| 16 | 0.2 | 1.78 | 300 | 5.9 | `lfs::training::mrnf_strategy::mrnf_noise_injection_kernel(float *, co…` |
| 17 | 0.1 | 1.56 | 300 | 5.2 | `void nvjpeg::ycbcr_to_format_kernel_roi<(nvjpegChromaSubsampling_t)0,…` |
| 18 | 0.1 | 1.56 | 338 | 4.6 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortHistog…` |
| 19 | 0.1 | 1.45 | 300 | 4.8 | `void cub::_V_300304_SM_890::detail::scan::DeviceScanKernel<cub::_V_30…` |
| 20 | 0.1 | 1.23 | 432 | 2.8 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortPartit…` |

## Memory operations

| op | total ms | count | avg µs | total MB |
|:---|--------:|------:|-------:|---------:|
| [CUDA memcpy Host-to-Device] | 58.87 | 434 | 135.7 | 740.374 |
| [CUDA memset] | 5.65 | 4680 | 1.2 | 4543.704 |
| [CUDA memcpy Device-to-Host] | 1.40 | 751 | 1.9 | 6.492 |
| [CUDA memcpy Device-to-Device] | 0.80 | 316 | 2.5 | 338.463 |

## Launch-gap analysis (CUDA-graphs opportunity)

- capture window: **1282.0 ms**, GPU busy 1185.2 ms, **idle 96.8 ms (7.55%)**
- ops: 12010 kernels, 1501 memcpy, 4680 memset; 16708 busy islands

| gap bucket | count | total ms |
|:---|--:|--:|
| lt_2us | 12907 | 8.39 |
| 2_10us | 2283 | 12.53 |
| 10_100us | 1014 | 18.73 |
| gt_100us | 503 | 57.15 |

Per-iteration medians (NVTX `train_step` attribution):

- kernel launches/iter: **29**
- span 3996 µs = busy 3772 µs + gap **107 µs (2.7% of span)**
- iterations attributed: 300

