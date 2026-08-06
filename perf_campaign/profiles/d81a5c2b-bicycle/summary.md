# Profile summary — `d81a5c2b-bicycle`

- commit: `d81a5c2b`  dataset: `/home/gauss/data/360_v2/bicycle`
- slice: iters [200, 500] of 520  strategy: mrnf  images: images_4
- GPU: NVIDIA GeForce RTX 4080  date: 2026-08-06T21:53:39Z

## Top kernels by total GPU time

| # | time % | total ms | count | avg µs | kernel |
|--:|-------:|---------:|------:|-------:|:-------|
| 1 | 42.1 | 237.34 | 300 | 791.1 | `void fast_lfs::rasterization::kernels::backward::blend_backward_cu<(D…` |
| 2 | 11.7 | 65.81 | 25 | 2632.4 | `edge_compute::rasterization::kernels::forward::edge_blend_cu(const ui…` |
| 3 | 7.4 | 41.88 | 300 | 139.6 | `void <unnamed>::fusedL1SSIMBackwardCUDA<unsigned char, __half>(float,…` |
| 4 | 7.3 | 41.13 | 300 | 137.1 | `lfs::training::mrnf_strategy::mrnf_noise_injection_kernel(float *, co…` |
| 5 | 6.9 | 38.74 | 300 | 129.1 | `void fast_lfs::rasterization::kernels::forward::blend_cu<(bool)0>(con…` |
| 6 | 5.5 | 31.26 | 300 | 104.2 | `void <unnamed>::fusedL1SSIMForwardCUDA<unsigned char>(int, int, int, …` |
| 7 | 3.0 | 17.18 | 300 | 57.3 | `fast_lfs::rasterization::kernels::forward::create_instances_cu(const …` |
| 8 | 2.9 | 16.24 | 300 | 54.1 | `void fast_lfs::rasterization::kernels::backward::preprocess_backward_…` |
| 9 | 2.2 | 12.20 | 1300 | 9.4 | `void cub::_V_300304_SM_890::detail::radix_sort::DeviceRadixSortOneswe…` |
| 10 | 1.5 | 8.41 | 900 | 9.3 | `void nvjpeg::DecodeSingleGPU::dcAcDecodeKernel<nvjpeg::DecodeSingleGP…` |
| 11 | 1.2 | 6.89 | 300 | 23.0 | `void lfs::training::kernels::<unnamed>::fused_l1_ssim_sum_kernel<unsi…` |
| 12 | 0.8 | 4.34 | 900 | 4.8 | `nvjpeg::DecodeSingleGPU::transposeKernel(short *)` |
| 13 | 0.7 | 4.19 | 300 | 14.0 | `lfs::io::cuda::uint8_hwc_to_uint8_chw_kernel(const unsigned char *, u…` |
| 14 | 0.6 | 3.62 | 300 | 12.1 | `fast_lfs::rasterization::kernels::forward::preprocess_cu(const float3…` |
| 15 | 0.5 | 2.66 | 300 | 8.9 | `void nvjpeg::dctQuantInvJpegKernelMultiChannel<ushort2, (int)1, (int)…` |
| 16 | 0.4 | 2.47 | 300 | 8.2 | `nvjpeg::DecodeSingleGPU::destuffKernel(unsigned char *, int, unsigned…` |
| 17 | 0.4 | 2.41 | 300 | 8.0 | `lfs::core::tensor_ops::clamp_kernel_vectorized(float *, float, float,…` |
| 18 | 0.4 | 2.07 | 300 | 6.9 | `void nvjpeg::ycbcr_to_format_kernel_roi<(nvjpegChromaSubsampling_t)0,…` |
| 19 | 0.3 | 1.79 | 900 | 2.0 | `void nvjpeg::DecodeSingleGPU::dcPrefixSumUpUpDownDownKernel<short>(T1…` |
| 20 | 0.3 | 1.77 | 300 | 5.9 | `lfs::training::kernels::fused_grad_alpha_chw_kernel(const float *, co…` |

## Memory operations

| op | total ms | count | avg µs | total MB |
|:---|--------:|------:|-------:|---------:|
| [CUDA memcpy Host-to-Device] | 26.43 | 968 | 27.3 | 322.043 |
| [CUDA memset] | 5.65 | 5118 | 1.1 | 2619.828 |
| [CUDA memcpy Device-to-Host] | 0.74 | 560 | 1.3 | 0.396 |
| [CUDA memcpy Device-to-Device] | 0.67 | 206 | 3.2 | 254.121 |

## Launch-gap analysis (CUDA-graphs opportunity)

- capture window: **1972.1 ms**, GPU busy 594.2 ms, **idle 1377.9 ms (69.87%)**
- ops: 15866 kernels, 1734 memcpy, 5118 memset; 22324 busy islands

| gap bucket | count | total ms |
|:---|--:|--:|
| lt_2us | 12768 | 8.71 |
| 2_10us | 4481 | 24.63 |
| 10_100us | 4533 | 143.53 |
| gt_100us | 541 | 1201.08 |

Per-iteration medians (NVTX `train_step` attribution):

- kernel launches/iter: **29**
- span 1800 µs = busy 1554 µs + gap **246 µs (13.6% of span)**
- iterations attributed: 300

