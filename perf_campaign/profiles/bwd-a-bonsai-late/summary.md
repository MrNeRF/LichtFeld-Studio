# Profile summary — `bwd-a-bonsai-late`

- commit: `4ae56159`  dataset: `/home/gauss/data/360_v2/bonsai`
- slice: iters [1600, 1900] of 1920  strategy: mrnf  images: images_4
- GPU: NVIDIA GeForce RTX 4080  date: 2026-08-06T22:52:34Z

## Top kernels by total GPU time

| # | time % | total ms | count | avg µs | kernel |
|--:|-------:|---------:|------:|-------:|:-------|
| 1 | 53.6 | 633.81 | 300 | 2112.7 | `void fast_lfs::rasterization::kernels::backward::blend_backward_cu<(D…` |
| 2 | 23.0 | 271.76 | 300 | 905.9 | `void fast_lfs::rasterization::kernels::backward::preprocess_backward_…` |
| 3 | 10.6 | 124.90 | 300 | 416.3 | `void fast_lfs::rasterization::kernels::forward::blend_cu<(bool)0>(con…` |
| 4 | 3.1 | 37.20 | 38 | 978.9 | `edge_compute::rasterization::kernels::forward::edge_blend_cu(const ui…` |
| 5 | 1.6 | 18.63 | 300 | 62.1 | `fast_lfs::rasterization::kernels::forward::preprocess_cu(const float3…` |
| 6 | 1.3 | 15.86 | 300 | 52.9 | `void <unnamed>::fusedL1SSIMBackwardCUDA<unsigned char, __half>(float,…` |
| 7 | 1.2 | 14.78 | 1352 | 10.9 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortOneswe…` |
| 8 | 1.0 | 12.21 | 300 | 40.7 | `void <unnamed>::fusedL1SSIMForwardCUDA<unsigned char>(int, int, int, …` |
| 9 | 0.8 | 9.74 | 300 | 32.5 | `fast_lfs::rasterization::kernels::forward::create_instances_cu(const …` |
| 10 | 0.4 | 5.25 | 300 | 17.5 | `void nvjpeg::dctQuantInvJpegKernelMultiChannel<ushort2, (int)1, (int)…` |
| 11 | 0.4 | 4.60 | 300 | 15.3 | `lfs::training::mrnf_strategy::fold_densification_and_zero_kernel(floa…` |
| 12 | 0.3 | 3.90 | 300 | 13.0 | `lfs::io::cuda::uint8_hwc_to_uint8_chw_kernel(const unsigned char *, u…` |
| 13 | 0.2 | 2.95 | 300 | 9.8 | `void lfs::training::kernels::<unnamed>::fused_l1_ssim_sum_kernel<unsi…` |
| 14 | 0.2 | 2.16 | 300 | 7.2 | `void nvjpeg::ycbcr_to_format_kernel_roi<(nvjpegChromaSubsampling_t)0,…` |
| 15 | 0.2 | 2.15 | 432 | 5.0 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortMergeK…` |
| 16 | 0.2 | 1.98 | 78 | 25.3 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortBlockS…` |
| 17 | 0.2 | 1.81 | 300 | 6.0 | `lfs::training::mrnf_strategy::mrnf_noise_injection_kernel(float *, co…` |
| 18 | 0.1 | 1.59 | 338 | 4.7 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortHistog…` |
| 19 | 0.1 | 1.52 | 300 | 5.1 | `void cub::_V_300304_SM_890::detail::scan::DeviceScanKernel<cub::_V_30…` |
| 20 | 0.1 | 1.23 | 432 | 2.8 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortPartit…` |

## Memory operations

| op | total ms | count | avg µs | total MB |
|:---|--------:|------:|-------:|---------:|
| [CUDA memcpy Host-to-Device] | 58.43 | 434 | 134.6 | 740.372 |
| [CUDA memset] | 5.59 | 4680 | 1.2 | 4544.101 |
| [CUDA memcpy Device-to-Host] | 1.42 | 751 | 1.9 | 6.490 |
| [CUDA memcpy Device-to-Device] | 1.13 | 316 | 3.6 | 337.875 |

## Launch-gap analysis (CUDA-graphs opportunity)

- capture window: **1616.8 ms**, GPU busy 1207.1 ms, **idle 409.7 ms (25.34%)**
- ops: 12010 kernels, 1501 memcpy, 4680 memset; 17194 busy islands

| gap bucket | count | total ms |
|:---|--:|--:|
| lt_2us | 8431 | 5.73 |
| 2_10us | 4482 | 24.53 |
| 10_100us | 4066 | 115.02 |
| gt_100us | 214 | 264.43 |

Per-iteration medians (NVTX `train_step` attribution):

- kernel launches/iter: **29**
- span 4424 µs = busy 3723 µs + gap **370 µs (8.4% of span)**
- iterations attributed: 300

