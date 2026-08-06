# Profile summary — `63aa08c6-bonsai-late`

- commit: `c692c782`  dataset: `/home/gauss/data/360_v2/bonsai`
- slice: iters [1600, 1900] of 1920  strategy: mrnf  images: images_4
- GPU: NVIDIA GeForce RTX 4080  date: 2026-08-06T21:57:57Z

## Top kernels by total GPU time

| # | time % | total ms | count | avg µs | kernel |
|--:|-------:|---------:|------:|-------:|:-------|
| 1 | 40.0 | 630.29 | 300 | 2101.0 | `void fast_lfs::rasterization::kernels::backward::blend_backward_cu<(D…` |
| 2 | 25.2 | 397.69 | 300 | 1325.6 | `lfs::training::mrnf_strategy::mrnf_noise_injection_kernel(float *, co…` |
| 3 | 17.3 | 271.90 | 300 | 906.3 | `void fast_lfs::rasterization::kernels::backward::preprocess_backward_…` |
| 4 | 8.0 | 126.03 | 300 | 420.1 | `void fast_lfs::rasterization::kernels::forward::blend_cu<(bool)0>(con…` |
| 5 | 2.1 | 33.32 | 38 | 876.8 | `edge_compute::rasterization::kernels::forward::edge_blend_cu(const ui…` |
| 6 | 1.2 | 19.30 | 300 | 64.3 | `fast_lfs::rasterization::kernels::forward::preprocess_cu(const float3…` |
| 7 | 1.0 | 15.87 | 300 | 52.9 | `void <unnamed>::fusedL1SSIMBackwardCUDA<unsigned char, __half>(float,…` |
| 8 | 0.9 | 14.58 | 1352 | 10.8 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortOneswe…` |
| 9 | 0.8 | 12.23 | 300 | 40.8 | `void <unnamed>::fusedL1SSIMForwardCUDA<unsigned char>(int, int, int, …` |
| 10 | 0.6 | 9.40 | 300 | 31.3 | `fast_lfs::rasterization::kernels::forward::create_instances_cu(const …` |
| 11 | 0.4 | 5.90 | 4 | 1476.2 | `lfs::training::mrnf_strategy::gumbel_key_for_indices_kernel(const flo…` |
| 12 | 0.3 | 4.58 | 300 | 15.3 | `lfs::training::mrnf_strategy::fold_densification_and_zero_kernel(floa…` |
| 13 | 0.3 | 4.22 | 300 | 14.1 | `void nvjpeg::dctQuantInvJpegKernelMultiChannel<ushort2, (int)1, (int)…` |
| 14 | 0.2 | 2.95 | 300 | 9.8 | `void lfs::training::kernels::<unnamed>::fused_l1_ssim_sum_kernel<unsi…` |
| 15 | 0.1 | 2.17 | 432 | 5.0 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortMergeK…` |
| 16 | 0.1 | 2.09 | 300 | 7.0 | `lfs::io::cuda::uint8_hwc_to_uint8_chw_kernel(const unsigned char *, u…` |
| 17 | 0.1 | 1.98 | 78 | 25.4 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortBlockS…` |
| 18 | 0.1 | 1.74 | 300 | 5.8 | `void nvjpeg::ycbcr_to_format_kernel_roi<(nvjpegChromaSubsampling_t)0,…` |
| 19 | 0.1 | 1.59 | 338 | 4.7 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortHistog…` |
| 20 | 0.1 | 1.46 | 300 | 4.9 | `void cub::_V_300304_SM_890::detail::scan::DeviceScanKernel<cub::_V_30…` |

## Memory operations

| op | total ms | count | avg µs | total MB |
|:---|--------:|------:|-------:|---------:|
| [CUDA memcpy Host-to-Device] | 58.73 | 434 | 135.3 | 740.377 |
| [CUDA memset] | 5.48 | 4680 | 1.2 | 4543.649 |
| [CUDA memcpy Device-to-Host] | 1.37 | 751 | 1.8 | 6.500 |
| [CUDA memcpy Device-to-Device] | 0.77 | 316 | 2.4 | 338.913 |

## Launch-gap analysis (CUDA-graphs opportunity)

- capture window: **1626.3 ms**, GPU busy 1576.0 ms, **idle 50.3 ms (3.1%)**
- ops: 12010 kernels, 1501 memcpy, 4680 memset; 16900 busy islands

| gap bucket | count | total ms |
|:---|--:|--:|
| lt_2us | 12557 | 8.32 |
| 2_10us | 2967 | 16.05 |
| 10_100us | 1363 | 23.27 |
| gt_100us | 12 | 2.69 |

Per-iteration medians (NVTX `train_step` attribution):

- kernel launches/iter: **29**
- span 5181 µs = busy 5080 µs + gap **98 µs (1.9% of span)**
- iterations attributed: 300

