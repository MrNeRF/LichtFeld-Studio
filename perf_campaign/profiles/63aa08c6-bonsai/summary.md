# Profile summary — `63aa08c6-bonsai`

- commit: `c692c782`  dataset: `/home/gauss/data/360_v2/bonsai`
- slice: iters [200, 500] of 520  strategy: mrnf  images: images_4
- GPU: NVIDIA GeForce RTX 4080  date: 2026-08-06T21:56:39Z

## Top kernels by total GPU time

| # | time % | total ms | count | avg µs | kernel |
|--:|-------:|---------:|------:|-------:|:-------|
| 1 | 54.9 | 479.97 | 300 | 1599.9 | `void fast_lfs::rasterization::kernels::backward::blend_backward_cu<(D…` |
| 2 | 15.2 | 132.86 | 300 | 442.9 | `lfs::training::mrnf_strategy::mrnf_noise_injection_kernel(float *, co…` |
| 3 | 10.2 | 89.03 | 300 | 296.8 | `void fast_lfs::rasterization::kernels::forward::blend_cu<(bool)0>(con…` |
| 4 | 4.8 | 42.09 | 300 | 140.3 | `void fast_lfs::rasterization::kernels::backward::preprocess_backward_…` |
| 5 | 4.3 | 37.69 | 38 | 991.8 | `edge_compute::rasterization::kernels::forward::edge_blend_cu(const ui…` |
| 6 | 1.8 | 15.86 | 300 | 52.9 | `void <unnamed>::fusedL1SSIMBackwardCUDA<unsigned char, __half>(float,…` |
| 7 | 1.5 | 12.73 | 1352 | 9.4 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortOneswe…` |
| 8 | 1.4 | 12.19 | 300 | 40.6 | `void <unnamed>::fusedL1SSIMForwardCUDA<unsigned char>(int, int, int, …` |
| 9 | 0.9 | 8.26 | 300 | 27.5 | `fast_lfs::rasterization::kernels::forward::create_instances_cu(const …` |
| 10 | 0.7 | 5.80 | 300 | 19.3 | `void nvjpeg::dctQuantInvJpegKernelMultiChannel<ushort2, (int)1, (int)…` |
| 11 | 0.6 | 5.67 | 300 | 18.9 | `fast_lfs::rasterization::kernels::forward::preprocess_cu(const float3…` |
| 12 | 0.4 | 3.19 | 300 | 10.6 | `lfs::io::cuda::uint8_hwc_to_uint8_chw_kernel(const unsigned char *, u…` |
| 13 | 0.3 | 2.95 | 300 | 9.8 | `void lfs::training::kernels::<unnamed>::fused_l1_ssim_sum_kernel<unsi…` |
| 14 | 0.3 | 2.62 | 300 | 8.7 | `void nvjpeg::ycbcr_to_format_kernel_roi<(nvjpegChromaSubsampling_t)0,…` |
| 15 | 0.2 | 2.00 | 300 | 6.7 | `lfs::training::mrnf_strategy::fold_densification_and_zero_kernel(floa…` |
| 16 | 0.2 | 1.97 | 408 | 4.8 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortMergeK…` |
| 17 | 0.2 | 1.96 | 78 | 25.2 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortBlockS…` |
| 18 | 0.2 | 1.88 | 2 | 941.0 | `lfs::training::mrnf_strategy::gumbel_key_for_indices_kernel(const flo…` |
| 19 | 0.1 | 1.18 | 300 | 3.9 | `void cub::_V_300304_SM_890::detail::scan::DeviceScanKernel<cub::_V_30…` |
| 20 | 0.1 | 1.14 | 408 | 2.8 | `void cub::_V_300304_SM_890::detail::merge_sort::DeviceMergeSortPartit…` |

## Memory operations

| op | total ms | count | avg µs | total MB |
|:---|--------:|------:|-------:|---------:|
| [CUDA memcpy Host-to-Device] | 58.03 | 394 | 147.3 | 735.415 |
| [CUDA memset] | 4.99 | 4648 | 1.1 | 2841.914 |
| [CUDA memcpy Device-to-Host] | 0.97 | 653 | 1.5 | 1.464 |
| [CUDA memcpy Device-to-Device] | 0.65 | 284 | 2.3 | 257.829 |

## Launch-gap analysis (CUDA-graphs opportunity)

- capture window: **954.8 ms**, GPU busy 878.5 ms, **idle 76.3 ms (7.99%)**
- ops: 11848 kernels, 1331 memcpy, 4648 memset; 15754 busy islands

| gap bucket | count | total ms |
|:---|--:|--:|
| lt_2us | 10247 | 6.65 |
| 2_10us | 3441 | 18.01 |
| 10_100us | 2049 | 46.04 |
| gt_100us | 16 | 5.63 |

Per-iteration medians (NVTX `train_step` attribution):

- kernel launches/iter: **29**
- span 2856 µs = busy 2719 µs + gap **150 µs (5.3% of span)**
- iterations attributed: 300

