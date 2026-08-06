# Profile summary — `487d5c2b-bonsai`

- commit: `1ffc4baa`  dataset: `/home/gauss/data/360_v2/bonsai`
- slice: iters [200, 500] of 520  strategy: mrnf  images: images_4
- GPU: NVIDIA GeForce RTX 4080  date: 2026-08-06T21:55:36Z

## Top kernels by total GPU time

| # | time % | total ms | count | avg µs | kernel |
|--:|-------:|---------:|------:|-------:|:-------|
| 1 | 54.3 | 487.94 | 300 | 1626.5 | `void fast_lfs::rasterization::kernels::backward::blend_backward_cu<(D…` |
| 2 | 14.8 | 132.53 | 300 | 441.8 | `lfs::training::mrnf_strategy::mrnf_noise_injection_kernel(float *, co…` |
| 3 | 10.1 | 90.53 | 300 | 301.8 | `void fast_lfs::rasterization::kernels::forward::blend_cu<(bool)0>(con…` |
| 4 | 6.0 | 53.80 | 300 | 179.3 | `void fast_lfs::rasterization::kernels::backward::preprocess_backward_…` |
| 5 | 4.5 | 40.34 | 38 | 1061.5 | `edge_compute::rasterization::kernels::forward::edge_blend_cu(const ui…` |
| 6 | 1.8 | 15.85 | 300 | 52.8 | `void <unnamed>::fusedL1SSIMBackwardCUDA<unsigned char, __half>(float,…` |
| 7 | 1.4 | 12.73 | 1352 | 9.4 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortOneswe…` |
| 8 | 1.4 | 12.19 | 300 | 40.6 | `void <unnamed>::fusedL1SSIMForwardCUDA<unsigned char>(int, int, int, …` |
| 9 | 0.9 | 8.36 | 300 | 27.9 | `fast_lfs::rasterization::kernels::forward::create_instances_cu(const …` |
| 10 | 0.7 | 5.85 | 300 | 19.5 | `fast_lfs::rasterization::kernels::forward::preprocess_cu(const float3…` |
| 11 | 0.6 | 5.08 | 300 | 16.9 | `void nvjpeg::dctQuantInvJpegKernelMultiChannel<ushort2, (int)1, (int)…` |
| 12 | 0.3 | 3.14 | 300 | 10.5 | `void nvjpeg::ycbcr_to_format_kernel_roi<(nvjpegChromaSubsampling_t)0,…` |
| 13 | 0.3 | 3.02 | 300 | 10.1 | `lfs::io::cuda::uint8_hwc_to_uint8_chw_kernel(const unsigned char *, u…` |
| 14 | 0.3 | 2.95 | 300 | 9.8 | `void lfs::training::kernels::<unnamed>::fused_l1_ssim_sum_kernel<unsi…` |
| 15 | 0.2 | 1.98 | 300 | 6.6 | `lfs::training::mrnf_strategy::fold_densification_and_zero_kernel(floa…` |
| 16 | 0.2 | 1.97 | 409 | 4.8 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortMergeK…` |
| 17 | 0.2 | 1.96 | 78 | 25.1 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortBlockS…` |
| 18 | 0.1 | 1.20 | 300 | 4.0 | `void cub::_V_300304_SM_890::detail::scan::DeviceScanKernel<cub::_V_30…` |
| 19 | 0.1 | 1.14 | 409 | 2.8 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortPartit…` |
| 20 | 0.1 | 1.13 | 338 | 3.3 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortHistog…` |

## Memory operations

| op | total ms | count | avg µs | total MB |
|:---|--------:|------:|-------:|---------:|
| [CUDA memcpy Host-to-Device] | 59.88 | 406 | 147.5 | 754.507 |
| [CUDA memset] | 5.03 | 4696 | 1.1 | 2840.283 |
| [CUDA memcpy Device-to-Host] | 2.95 | 736 | 4.0 | 27.741 |
| [CUDA memcpy Device-to-Device] | 0.68 | 296 | 2.3 | 278.180 |

## Launch-gap analysis (CUDA-graphs opportunity)

- capture window: **978.2 ms**, GPU busy 904.1 ms, **idle 74.2 ms (7.58%)**
- ops: 11837 kernels, 1438 memcpy, 4696 memset; 15951 busy islands

| gap bucket | count | total ms |
|:---|--:|--:|
| lt_2us | 10703 | 6.81 |
| 2_10us | 3360 | 18.25 |
| 10_100us | 1861 | 44.35 |
| gt_100us | 26 | 4.78 |

Per-iteration medians (NVTX `train_step` attribution):

- kernel launches/iter: **29**
- span 2887 µs = busy 2763 µs + gap **121 µs (4.2% of span)**
- iterations attributed: 300

