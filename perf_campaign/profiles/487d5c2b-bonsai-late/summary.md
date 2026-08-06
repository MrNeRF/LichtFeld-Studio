# Profile summary — `487d5c2b-bonsai-late`

- commit: `1ffc4baa`  dataset: `/home/gauss/data/360_v2/bonsai`
- slice: iters [1600, 1900] of 1920  strategy: mrnf  images: images_4
- GPU: NVIDIA GeForce RTX 4080  date: 2026-08-06T21:58:30Z

## Top kernels by total GPU time

| # | time % | total ms | count | avg µs | kernel |
|--:|-------:|---------:|------:|-------:|:-------|
| 1 | 43.1 | 611.63 | 300 | 2038.8 | `void fast_lfs::rasterization::kernels::backward::blend_backward_cu<(D…` |
| 2 | 27.9 | 397.02 | 300 | 1323.4 | `lfs::training::mrnf_strategy::mrnf_noise_injection_kernel(float *, co…` |
| 3 | 9.5 | 134.99 | 300 | 450.0 | `void fast_lfs::rasterization::kernels::backward::preprocess_backward_…` |
| 4 | 8.6 | 122.87 | 300 | 409.6 | `void fast_lfs::rasterization::kernels::forward::blend_cu<(bool)0>(con…` |
| 5 | 2.5 | 34.98 | 38 | 920.6 | `edge_compute::rasterization::kernels::forward::edge_blend_cu(const ui…` |
| 6 | 1.3 | 18.85 | 300 | 62.8 | `fast_lfs::rasterization::kernels::forward::preprocess_cu(const float3…` |
| 7 | 1.1 | 15.87 | 300 | 52.9 | `void <unnamed>::fusedL1SSIMBackwardCUDA<unsigned char, __half>(float,…` |
| 8 | 1.0 | 14.63 | 1352 | 10.8 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortOneswe…` |
| 9 | 0.9 | 12.23 | 300 | 40.8 | `void <unnamed>::fusedL1SSIMForwardCUDA<unsigned char>(int, int, int, …` |
| 10 | 0.7 | 9.44 | 300 | 31.5 | `fast_lfs::rasterization::kernels::forward::create_instances_cu(const …` |
| 11 | 0.4 | 5.94 | 4 | 1484.7 | `lfs::training::mrnf_strategy::gumbel_key_for_indices_kernel(const flo…` |
| 12 | 0.4 | 5.40 | 300 | 18.0 | `void nvjpeg::dctQuantInvJpegKernelMultiChannel<ushort2, (int)1, (int)…` |
| 13 | 0.3 | 4.39 | 300 | 14.6 | `lfs::training::mrnf_strategy::fold_densification_and_zero_kernel(floa…` |
| 14 | 0.2 | 3.44 | 300 | 11.5 | `lfs::io::cuda::uint8_hwc_to_uint8_chw_kernel(const unsigned char *, u…` |
| 15 | 0.2 | 3.31 | 300 | 11.0 | `void nvjpeg::ycbcr_to_format_kernel_roi<(nvjpegChromaSubsampling_t)0,…` |
| 16 | 0.2 | 2.95 | 300 | 9.8 | `void lfs::training::kernels::<unnamed>::fused_l1_ssim_sum_kernel<unsi…` |
| 17 | 0.2 | 2.16 | 432 | 5.0 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortMergeK…` |
| 18 | 0.1 | 1.99 | 78 | 25.5 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortBlockS…` |
| 19 | 0.1 | 1.56 | 338 | 4.6 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortHistog…` |
| 20 | 0.1 | 1.45 | 300 | 4.8 | `void cub::_V_300304_SM_890::detail::scan::DeviceScanKernel<cub::_V_30…` |

## Memory operations

| op | total ms | count | avg µs | total MB |
|:---|--------:|------:|-------:|---------:|
| [CUDA memcpy Host-to-Device] | 66.21 | 470 | 140.9 | 836.437 |
| [CUDA memcpy Device-to-Host] | 9.59 | 1003 | 9.6 | 114.264 |
| [CUDA memset] | 5.67 | 4752 | 1.2 | 4541.782 |
| [CUDA memcpy Device-to-Device] | 1.45 | 376 | 3.9 | 436.215 |

## Launch-gap analysis (CUDA-graphs opportunity)

- capture window: **1568.5 ms**, GPU busy 1437.4 ms, **idle 131.1 ms (8.36%)**
- ops: 11974 kernels, 1849 memcpy, 4752 memset; 16808 busy islands

| gap bucket | count | total ms |
|:---|--:|--:|
| lt_2us | 8543 | 5.77 |
| 2_10us | 4457 | 24.66 |
| 10_100us | 3776 | 93.42 |
| gt_100us | 31 | 7.25 |

Per-iteration medians (NVTX `train_step` attribution):

- kernel launches/iter: **29**
- span 4730 µs = busy 4518 µs + gap **239 µs (5.1% of span)**
- iterations attributed: 300

