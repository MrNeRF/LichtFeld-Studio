/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/tensor_generic_ops.cuh"
#include "lfs/cuda_scratch.hpp"
#include "mrnf_kernels.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "kernel_stream.hpp"

namespace lfs::training::mrnf_strategy {

    namespace {

        __device__ __forceinline__ float d_sigmoid(float x) {
            return 1.0f / (1.0f + expf(-x));
        }

        __device__ __forceinline__ float d_logit(float p) {
            if (isnan(p))
                return p;
            // The upper bound must be representable below 1 in float32.
            p = fminf(fmaxf(p, 1e-12f), nextafterf(1.0f, 0.0f));
            return logf(p / (1.0f - p));
        }

        struct positive_weight {
            __host__ __device__ bool operator()(const float w) const {
                return w > 0.0f;
            }
        };

    } // namespace

    // Matches normalizing a copy by the positive median, then mul_(weight)
    // and add_(1) as separate roundings.
    __device__ __forceinline__ float edge_guidance_factor(const EdgeGuidance& edge, const size_t i) {
        const float score = edge.scores[i];
        const float normalized = (edge.median > 0.0f && !isnan(score))
                                     ? score / fmaxf(edge.median, 1e-9f)
                                     : 0.0f;
        return __fadd_rn(__fmul_rn(normalized, edge.weight), 1.0f);
    }

    __global__ void replace_parent_weights_kernel(
        const float* __restrict__ raw_opacities,
        const float* __restrict__ visibility,
        const bool* __restrict__ active_mask,
        const bool* __restrict__ trainable_mask,
        const EdgeGuidance edge_guidance,
        float* __restrict__ output,
        size_t N) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= N)
            return;
        const float opacity = 1.0f / (1.0f + expf(-raw_opacities[i]));
        float weight = opacity * static_cast<float>(visibility[i] > 0.0f);
        if (active_mask)
            weight = weight * static_cast<float>(active_mask[i]);
        if (trainable_mask)
            weight = weight * static_cast<float>(trainable_mask[i]);
        if (edge_guidance.scores)
            weight = weight * edge_guidance_factor(edge_guidance, i);
        output[i] = weight;
    }

    void launch_replace_parent_weights(
        const float* raw_opacities,
        const float* visibility,
        const bool* active_mask,
        const bool* trainable_mask,
        const EdgeGuidance edge_guidance,
        float* output,
        size_t N,
        void* stream) {
        if (N == 0)
            return;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        replace_parent_weights_kernel<<<blocks, threads, 0, s>>>(
            raw_opacities, visibility, active_mask, trainable_mask, edge_guidance, output, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.replace_parent_weights");
    }

    __global__ void apply_edge_guidance_kernel(float* __restrict__ weights,
                                               const EdgeGuidance edge_guidance, const size_t N) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= N)
            return;
        weights[i] = weights[i] * edge_guidance_factor(edge_guidance, i);
    }

    void launch_apply_edge_guidance(float* weights, const EdgeGuidance edge_guidance, const size_t N,
                                    void* stream) {
        if (N == 0 || edge_guidance.scores == nullptr)
            return;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        apply_edge_guidance_kernel<<<blocks, threads, 0, s>>>(weights, edge_guidance, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.apply_edge_guidance");
    }

    // Ascending float order as unsigned radix keys, matching sort_less_op:
    // NaN after everything and -0 equal to +0.
    __device__ __forceinline__ uint32_t ascending_float_key(const float value) {
        const uint32_t bits = __float_as_uint(value);
        if ((bits & 0x7fffffffu) > 0x7f800000u)
            return 0xffffffffu;
        if (bits == 0x80000000u)
            return 0x80000000u;
        return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    }

    __global__ void exchange_importance_keys_kernel(
        const float* __restrict__ importance,
        const bool* __restrict__ eligible,
        uint32_t* __restrict__ keys,
        uint32_t* __restrict__ rows,
        const size_t N) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= N)
            return;
        keys[i] = ascending_float_key(eligible[i] ? importance[i] : INFINITY);
        rows[i] = static_cast<uint32_t>(i);
    }

    __global__ void exchange_positions_kernel(
        const uint32_t* __restrict__ sorted_rows,
        int32_t* __restrict__ ranks,
        const bool first,
        const size_t N) {
        const size_t position = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (position >= N)
            return;
        const uint32_t row = sorted_rows[position];
        const auto rank = static_cast<int32_t>(position);
        ranks[row] = first ? rank : max(ranks[row], rank);
    }

    __global__ void exchange_rank_keys_kernel(
        const int32_t* __restrict__ ranks,
        const bool* __restrict__ eligible,
        uint32_t* __restrict__ keys,
        uint32_t* __restrict__ rows,
        const size_t N) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= N)
            return;
        keys[i] = static_cast<uint32_t>(eligible[i] ? ranks[i] : INT_MAX);
        rows[i] = static_cast<uint32_t>(i);
    }

    __global__ void widen_rows_kernel(
        const uint32_t* __restrict__ rows,
        int64_t* __restrict__ out,
        const size_t count) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < count)
            out[i] = static_cast<int64_t>(rows[i]);
    }

    void launch_rank_exchange_rows(
        const float* primary,
        const float* secondary,
        const bool* eligible,
        const size_t N,
        const size_t count,
        int64_t* rows,
        void* stream) {
        if (N == 0 || count == 0)
            return;
        LFS_ASSERT_MSG(N <= static_cast<size_t>(INT_MAX) && count <= N,
                       "MRNF growth exchange exceeds int32 rank capacity");
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        const cudaStream_t s = resolve_stream(stream);
        const lfs::core::CUDAStreamGuard stream_guard(s);

        // Two key and two row buffers serve every sort; 20 bytes a row in all.
        const auto buffer = [N] {
            return lfs::core::Tensor::empty({N}, lfs::core::Device::CUDA, lfs::core::DataType::Int32);
        };
        auto keys_a = buffer(), keys_b = buffer(), rows_a = buffer(), rows_b = buffer(), ranks = buffer();
        const auto u32 = [](lfs::core::Tensor& t) { return reinterpret_cast<uint32_t*>(t.ptr<int32_t>()); };

        // Stable, so equal keys keep row order, as the comparator sorts did.
        const auto sort = [&](const int end_bit) {
            cub::DoubleBuffer<uint32_t> keys(u32(keys_a), u32(keys_b));
            cub::DoubleBuffer<uint32_t> values(u32(rows_a), u32(rows_b));
            const auto radix_sort = [&](void* workspace, size_t& workspace_bytes) {
                return cub::DeviceRadixSort::SortPairs(workspace, workspace_bytes, keys, values,
                                                       static_cast<int>(N), 0, end_bit, s);
            };
            cuda_scratch::CubWorkspace workspace("cub::DeviceRadixSort::SortPairs", s, radix_sort);
            workspace.run(radix_sort);
            return values.Current();
        };

        const auto rank_by = [&](const float* importance, const bool first) {
            exchange_importance_keys_kernel<<<blocks, threads, 0, s>>>(
                importance, eligible, u32(keys_a), u32(rows_a), N);
            LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.exchange_importance_keys");
            exchange_positions_kernel<<<blocks, threads, 0, s>>>(sort(32), ranks.ptr<int32_t>(), first, N);
            LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.exchange_positions");
        };
        rank_by(primary, true);
        if (secondary != nullptr)
            rank_by(secondary, false);

        exchange_rank_keys_kernel<<<blocks, threads, 0, s>>>(
            ranks.ptr<int32_t>(), eligible, u32(keys_a), u32(rows_a), N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.exchange_rank_keys");
        const uint32_t* ranked = sort(31);
        widen_rows_kernel<<<static_cast<int>((count + threads - 1) / threads), threads, 0, s>>>(
            ranked, rows, count);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.exchange_widen_rows");
    }

    __global__ void fill_index_mask_kernel(
        const int64_t* __restrict__ indices,
        const size_t count,
        bool* __restrict__ mask) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < count)
            mask[indices[i]] = true;
    }

    void launch_fill_index_mask(
        const int64_t* indices,
        const size_t count,
        bool* mask,
        void* stream) {
        if (count == 0)
            return;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        fill_index_mask_kernel<<<blocks, threads, 0, s>>>(indices, count, mask);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.fill_index_mask");
    }

    __global__ void mrnf_noise_injection_kernel(
        float* __restrict__ means,
        const float* __restrict__ raw_opacities,
        const float* __restrict__ vis_count,
        const bool* __restrict__ frozen_mask,
        size_t frozen_mask_size,
        float lr_mean,
        float noise_weight,
        float median_scale,
        size_t N,
        uint64_t seed) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;
        if (frozen_mask != nullptr && idx < frozen_mask_size && frozen_mask[idx])
            return;

        if (vis_count[idx] <= 0.0f)
            return;

        const float inv_opac = 1.0f - d_sigmoid(raw_opacities[idx]);
        float weight = powf(fmaxf(inv_opac, 0.0f), 150.0f);
        weight *= lr_mean * noise_weight;

        if (weight < 1e-12f)
            return;

        curandStatePhilox4_32_10_t rng;
        curand_init(seed, idx, 0, &rng);
        const float4 n = curand_normal4(&rng);
        const float noise_xyz[3] = {n.x, n.y, n.z};

        for (int d = 0; d < 3; ++d) {
            const float noise = noise_xyz[d] * weight;
            const float clamped_noise = fminf(fmaxf(noise, -median_scale), median_scale);
            means[idx * 3 + d] += clamped_noise;
        }
    }

    void launch_mrnf_noise_injection(
        float* means,
        const float* raw_opacities,
        const float* vis_count,
        const bool* frozen_mask,
        size_t frozen_mask_size,
        float lr_mean,
        float noise_weight,
        float median_scale,
        size_t N,
        uint64_t seed,
        void* stream) {

        if (N == 0)
            return;

        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);

        mrnf_noise_injection_kernel<<<blocks, threads, 0, s>>>(
            means, raw_opacities, vis_count,
            frozen_mask, frozen_mask_size,
            lr_mean, noise_weight, median_scale, N, seed);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.noise_injection");
    }

    template <bool RenderedOnly>
    __global__ void mrnf_decay_kernel(
        float* __restrict__ raw_opacities,
        float* __restrict__ log_scales,
        const bool* __restrict__ frozen_mask,
        size_t frozen_mask_size,
        float opacity_decay,
        float scale_decay,
        float train_t,
        size_t N,
        const float* __restrict__ rendered_count) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;
        if (frozen_mask != nullptr && idx < frozen_mask_size && frozen_mask[idx])
            return;

        const float t_shrink = 1.0f - train_t;

        const float opacity_delta = opacity_decay * t_shrink;
        // A sigmoid/logit round trip loses finite saturated logits even when
        // decay is disabled. Still repair infinities from older checkpoints.
        if ((!RenderedOnly || rendered_count[idx] > 0.0f) &&
            (opacity_delta != 0.0f || isinf(raw_opacities[idx]))) {
            raw_opacities[idx] = d_logit(d_sigmoid(raw_opacities[idx]) - opacity_delta);
        }

        const float decay_factor = 1.0f - scale_decay * t_shrink;
        for (int d = 0; d < 3; ++d) {
            const float scale = expf(log_scales[idx * 3 + d]) * decay_factor;
            log_scales[idx * 3 + d] = logf(fmaxf(scale, 1e-12f));
        }
    }

    void launch_mrnf_decay(
        float* raw_opacities,
        float* log_scales,
        const bool* frozen_mask,
        size_t frozen_mask_size,
        float opacity_decay,
        float scale_decay,
        float train_t,
        size_t N,
        void* stream,
        const float* rendered_count) {

        if (N == 0)
            return;

        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);

        if (rendered_count != nullptr) {
            mrnf_decay_kernel<true><<<blocks, threads, 0, s>>>(
                raw_opacities, log_scales, frozen_mask, frozen_mask_size,
                opacity_decay, scale_decay, train_t, N, rendered_count);
            LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.decay_with_rendered_count");
        } else {
            mrnf_decay_kernel<false><<<blocks, threads, 0, s>>>(
                raw_opacities, log_scales, frozen_mask, frozen_mask_size,
                opacity_decay, scale_decay, train_t, N, nullptr);
            LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.decay_without_rendered_count");
        }
    }

    __global__ void fold_densification_error_and_zero_kernel(
        float* __restrict__ refine_weight_max,
        float* __restrict__ densification_info,
        size_t N) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;

        refine_weight_max[idx] = fmaxf(refine_weight_max[idx], densification_info[N + idx]);
        densification_info[N + idx] = 0.0f;
    }

    void launch_fold_densification_error_and_zero(
        float* refine_weight_max,
        float* densification_info,
        size_t N,
        void* stream) {
        if (N == 0)
            return;
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        fold_densification_error_and_zero_kernel<<<blocks, threads, 0, s>>>(
            refine_weight_max, densification_info, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.fold_densification_error_and_zero");
    }

    __global__ void extract_axis_kernel(
        const float* __restrict__ means,
        float* __restrict__ output,
        int axis,
        size_t N) {
        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx < N)
            output[idx] = means[idx * 3 + axis];
    }

    void launch_percentile_bounds(
        const float* means,
        size_t N,
        float percentile,
        MRNFBounds* bounds,
        void* stream) {

        LFS_ASSERT(N > 0);
        LFS_ASSERT(bounds != nullptr);
        LFS_ASSERT_MSG(std::isfinite(percentile) && percentile >= 0.0f && percentile <= 1.0f,
                       "MRNF bounds percentile must be finite and within [0, 1]");
        LFS_ASSERT_MSG(N <= static_cast<size_t>(std::numeric_limits<int>::max()),
                       "MRNF percentile input exceeds CUB's int item-count limit");

        cudaStream_t s = resolve_stream(stream);

        const float low_pct = (1.0f - percentile) / 2.0f;
        const float high_pct = 1.0f - low_pct;
        const size_t low_idx = static_cast<size_t>(low_pct * static_cast<float>(N - 1));
        const size_t high_idx = static_cast<size_t>(high_pct * static_cast<float>(N - 1));

        const int n_int = static_cast<int>(N);

        const size_t values_bytes = cuda_scratch::checked_bytes(
            N, sizeof(float), "MRNF percentile values");
        cuda_scratch::DeviceBuffer input_buffer(values_bytes, s, "mrnf.percentile.input");
        cuda_scratch::DeviceBuffer sorted_buffer(values_bytes, s, "mrnf.percentile.sorted");
        auto* d_input = input_buffer.as<float>();
        auto* d_sorted = sorted_buffer.as<float>();

        cuda_scratch::CubWorkspace cub_workspace(
            "cub::DeviceRadixSort::SortKeys", s,
            [&](void* workspace, size_t& workspace_bytes) {
                return cub::DeviceRadixSort::SortKeys(
                    workspace, workspace_bytes, d_input, d_sorted, n_int, 0, 32, s);
            });

        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);

        float h_low, h_high;
        float extents[3], centers[3];

        for (int axis = 0; axis < 3; ++axis) {
            extract_axis_kernel<<<blocks, threads, 0, s>>>(means, d_input, axis, N);
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "MRNF percentile axis extraction");
            cub_workspace.run([&](void* workspace, size_t& workspace_bytes) {
                return cub::DeviceRadixSort::SortKeys(
                    workspace, workspace_bytes, d_input, d_sorted, n_int, 0, 32, s);
            });
            LFS_CUDA_CHECK_MSG(
                cudaMemcpyAsync(&h_low, d_sorted + low_idx, sizeof(float), cudaMemcpyDeviceToHost, s),
                "MRNF percentile low readback");
            LFS_CUDA_CHECK_MSG(
                cudaMemcpyAsync(&h_high, d_sorted + high_idx, sizeof(float), cudaMemcpyDeviceToHost, s),
                "MRNF percentile high readback");
            LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(s), "MRNF percentile stream sync");

            centers[axis] = (h_low + h_high) * 0.5f;
            extents[axis] = (h_high - h_low) * 0.5f;
        }

        for (int i = 0; i < 3; ++i) {
            bounds->center[i] = centers[i];
            bounds->extent[i] = extents[i];
        }

        float sorted_ext[3] = {extents[0], extents[1], extents[2]};
        std::sort(sorted_ext, sorted_ext + 3);
        bounds->median_size = sorted_ext[1] * 2.0f;
        bounds->max_extent = sorted_ext[2];
    }

    __global__ void geomean_extent_kernel(
        const float* __restrict__ scaling_raw,
        float* __restrict__ extents,
        size_t N) {
        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;
        const float s0 = scaling_raw[idx * 3 + 0];
        const float s1 = scaling_raw[idx * 3 + 1];
        const float s2 = scaling_raw[idx * 3 + 2];
        const float g = expf((s0 + s1 + s2) * (1.0f / 3.0f));
        extents[idx] = (isfinite(g) && g > 0.0f) ? g : 0.0f;
    }

    __global__ void fill_pos_inf_kernel(float* data, size_t n) {
        const size_t i = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (i < n)
            data[i] = INFINITY;
    }

    __global__ void store_positive_median_kernel(
        const float* __restrict__ sorted,
        const int* __restrict__ count,
        float* __restrict__ out) {
        const int c = *count;
        if (c <= 0) {
            *out = 0.0f;
            return;
        }
        const float m = sorted[c / 2];
        *out = (isfinite(m) && m > 0.0f) ? m : 0.0f;
    }

    void launch_median_geomean_extent(
        const float* scaling_raw,
        size_t N,
        float* out_median,
        bool* out_valid,
        void* stream) {

        LFS_ASSERT(out_median != nullptr);
        LFS_ASSERT(out_valid != nullptr);
        *out_median = 0.0f;
        *out_valid = false;
        if (N == 0 || scaling_raw == nullptr)
            return;
        LFS_ASSERT_MSG(N <= static_cast<size_t>(std::numeric_limits<int>::max()),
                       "MRNF median-extent input exceeds CUB's int item-count limit");

        cudaStream_t s = resolve_stream(stream);
        const int n_int = static_cast<int>(N);
        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);

        const size_t values_bytes = cuda_scratch::checked_bytes(
            N, sizeof(float), "MRNF median geomean extents");
        cuda_scratch::DeviceBuffer extents_buf(values_bytes, s, "mrnf.median_extent.input");
        cuda_scratch::DeviceBuffer selected_buf(values_bytes, s, "mrnf.median_extent.selected");
        cuda_scratch::DeviceBuffer sorted_buf(values_bytes, s, "mrnf.median_extent.sorted");
        cuda_scratch::DeviceBuffer count_buf(sizeof(int), s, "mrnf.median_extent.count");
        cuda_scratch::DeviceBuffer median_buf(sizeof(float), s, "mrnf.median_extent.scalar");

        auto* d_ext = extents_buf.as<float>();
        auto* d_sel = selected_buf.as<float>();
        auto* d_sorted = sorted_buf.as<float>();
        auto* d_count = count_buf.as<int>();
        auto* d_median = median_buf.as<float>();
        LFS_CUDA_CHECK_MSG(cudaMemsetAsync(d_count, 0, sizeof(int), s),
                           "MRNF median-extent count zero");
        LFS_CUDA_CHECK_MSG(cudaMemsetAsync(d_median, 0, sizeof(float), s),
                           "MRNF median-extent scalar zero");

        geomean_extent_kernel<<<blocks, threads, 0, s>>>(scaling_raw, d_ext, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.geomean_extent");
        fill_pos_inf_kernel<<<blocks, threads, 0, s>>>(d_sel, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.median_extent_fill_inf");

        // Named predicate, kept from the PR #1840 Windows fix: this CUB call sits inside a
        // host lambda, where nvcc+MSVC rejected a braced temporary. Do not inline it back.
        const positive_weight positive_pred;
        auto select_op = [&](void* workspace, size_t& workspace_bytes) {
            return cub::DeviceSelect::If(
                workspace, workspace_bytes, d_ext, d_sel, d_count,
                n_int, positive_pred, s);
        };
        size_t select_bytes = 0;
        LFS_CUDA_CHECK_MSG(select_op(nullptr, select_bytes), "MRNF median-extent select size");
        if (select_bytes > 0) {
            cuda_scratch::CubWorkspace select_ws(
                "cub::DeviceSelect::If", s, select_op);
            select_ws.run(select_op);
        } else {
            LFS_CUDA_CHECK_MSG(select_op(nullptr, select_bytes), "MRNF median-extent select");
        }

        auto sort_op = [&](void* workspace, size_t& workspace_bytes) {
            return cub::DeviceRadixSort::SortKeys(
                workspace, workspace_bytes, d_sel, d_sorted, n_int, 0, 32, s);
        };
        size_t sort_bytes = 0;
        LFS_CUDA_CHECK_MSG(sort_op(nullptr, sort_bytes), "MRNF median-extent sort size");
        if (sort_bytes > 0) {
            cuda_scratch::CubWorkspace sort_ws(
                "cub::DeviceRadixSort::SortKeys", s, sort_op);
            sort_ws.run(sort_op);
        } else {
            LFS_CUDA_CHECK_MSG(sort_op(nullptr, sort_bytes), "MRNF median-extent sort");
        }

        store_positive_median_kernel<<<1, 1, 0, s>>>(d_sorted, d_count, d_median);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.store_positive_median");
        LFS_CUDA_CHECK_MSG(
            cudaMemcpyAsync(out_median, d_median, sizeof(float), cudaMemcpyDeviceToHost, s),
            "MRNF median-extent readback");
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(s), "MRNF median-extent stream sync");
        *out_valid = std::isfinite(*out_median) && *out_median > 0.0f;
    }

    namespace {
        // Unsigned bits that order like the float: a larger key has larger bits.
        __host__ __device__ __forceinline__ uint32_t orderable_bits(const uint32_t bits) {
            return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        }

        __host__ __device__ __forceinline__ uint32_t float_bits_from_orderable(const uint32_t bits) {
            return (bits & 0x80000000u) ? (bits & 0x7fffffffu) : ~bits;
        }

        // Non-positive and NaN weights get 0, below every real key.
        __global__ void gumbel_key_bits_kernel(
            const float* __restrict__ weights,
            uint32_t* __restrict__ key_bits,
            const size_t N,
            const uint64_t seed) {
            const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
            if (idx >= N)
                return;

            const float w = weights[idx];
            if (!(w > 0.0f)) {
                key_bits[idx] = 0u;
                return;
            }

            curandStatePhilox4_32_10_t rng;
            curand_init(seed, idx, 0, &rng);
            float u = curand_uniform(&rng);
            u = fmaxf(u, 1e-10f);
            u = fminf(u, 1.0f - 1e-7f);

            key_bits[idx] = orderable_bits(__float_as_uint(-logf(-logf(u)) + logf(w)));
        }

        __global__ void orderable_value_bits_kernel(
            const float* __restrict__ values,
            const bool* __restrict__ include,
            uint32_t* __restrict__ key_bits,
            const size_t N) {
            const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
            if (idx < N) {
                key_bits[idx] = include && !include[idx] ? 0u : orderable_bits(__float_as_uint(values[idx]));
            }
        }

        // Radix select state: bits decided so far, how many keys rank above
        // them, and how many of the keys equal to them are still wanted.
        struct RadixSelectState {
            uint32_t prefix;
            uint32_t remaining;
            uint32_t greater;
            uint32_t equal;
        };

        constexpr int RADIX_BITS = 8;
        constexpr int RADIX_BINS = 1 << RADIX_BITS;

        __global__ void radix_histogram_kernel(
            const uint32_t* __restrict__ keys,
            const size_t n,
            const RadixSelectState* __restrict__ state,
            const int shift,
            uint32_t* __restrict__ histogram) {
            __shared__ uint32_t local[RADIX_BINS];
            for (int bin = threadIdx.x; bin < RADIX_BINS; bin += blockDim.x)
                local[bin] = 0;
            __syncthreads();

            const uint32_t prefix = state->prefix;
            const uint32_t decided = shift + RADIX_BITS >= 32 ? 0u : (0xffffffffu << (shift + RADIX_BITS));
            const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
            for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += stride) {
                const uint32_t key = keys[i];
                if (((key ^ prefix) & decided) == 0)
                    atomicAdd(&local[(key >> shift) & (RADIX_BINS - 1)], 1u);
            }
            __syncthreads();
            for (int bin = threadIdx.x; bin < RADIX_BINS; bin += blockDim.x) {
                if (local[bin] != 0)
                    atomicAdd(&histogram[bin], local[bin]);
            }
        }

        __global__ void radix_pick_kernel(
            RadixSelectState* __restrict__ state,
            uint32_t* __restrict__ histogram,
            const int shift) {
            uint32_t above = 0;
            for (int digit = RADIX_BINS - 1; digit >= 0; --digit) {
                const uint32_t count = histogram[digit];
                if (above + count >= state->remaining) {
                    state->prefix |= static_cast<uint32_t>(digit) << shift;
                    state->remaining -= above;
                    state->greater += above;
                    state->equal = count;
                    break;
                }
                above += count;
            }
            for (int digit = 0; digit < RADIX_BINS; ++digit)
                histogram[digit] = 0;
        }

        struct KeyAbove {
            const uint32_t* keys;
            uint32_t threshold;
            __host__ __device__ bool operator()(const uint32_t i) const { return keys[i] > threshold; }
        };

        struct KeyEquals {
            const uint32_t* keys;
            uint32_t threshold;
            __host__ __device__ bool operator()(const uint32_t i) const { return keys[i] == threshold; }
        };

        __global__ void gather_key_bits_kernel(
            const uint32_t* __restrict__ keys,
            const uint32_t* __restrict__ rows,
            uint32_t* __restrict__ out,
            const size_t K) {
            const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
            if (idx < K)
                out[idx] = keys[rows[idx]];
        }

        __global__ void widen_gumbel_indices_kernel(
            const uint32_t* __restrict__ input,
            int64_t* __restrict__ output,
            size_t N) {
            const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
            if (idx < N) {
                output[idx] = static_cast<int64_t>(input[idx]);
            }
        }

        struct RadixThreshold {
            uint32_t bits = 0;
            size_t greater = 0;
            size_t equal = 0;
            size_t remaining = 0;
        };

        // The K-th largest of `keys` by 8-bit radix passes over the data:
        // a 256-bin histogram per pass instead of sorting every key.
        RadixThreshold radix_select_largest(const uint32_t* keys, const size_t n, const size_t K, const cudaStream_t s) {
            cuda_scratch::DeviceBuffer state_buffer(sizeof(RadixSelectState), s, "mrnf.topk.state");
            cuda_scratch::DeviceBuffer histogram_buffer(RADIX_BINS * sizeof(uint32_t), s, "mrnf.topk.histogram");
            auto* state = state_buffer.as<RadixSelectState>();
            auto* histogram = histogram_buffer.as<uint32_t>();
            const RadixSelectState initial{0u, static_cast<uint32_t>(K), 0u, 0u};
            LFS_CUDA_CHECK_MSG(cudaMemcpyAsync(state, &initial, sizeof(initial), cudaMemcpyHostToDevice, s),
                               "MRNF top-k state upload");
            LFS_CUDA_CHECK_MSG(cudaMemsetAsync(histogram, 0, RADIX_BINS * sizeof(uint32_t), s),
                               "MRNF top-k histogram clear");

            constexpr int threads = 256;
            int device = 0;
            int sms = 0;
            LFS_CUDA_CHECK(cudaGetDevice(&device));
            LFS_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
            const int blocks = static_cast<int>(std::min<size_t>((n + threads - 1) / threads,
                                                                 static_cast<size_t>(std::max(sms, 1)) * 8));
            for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
                radix_histogram_kernel<<<blocks, threads, 0, s>>>(keys, n, state, shift, histogram);
                LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.topk_histogram");
                radix_pick_kernel<<<1, 1, 0, s>>>(state, histogram, shift);
                LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.topk_pick");
            }
            RadixSelectState result{};
            LFS_CUDA_CHECK_MSG(cudaMemcpyAsync(&result, state, sizeof(result), cudaMemcpyDeviceToHost, s),
                               "MRNF top-k state readback");
            LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(s), "MRNF top-k state sync");
            return {result.prefix, result.greater, result.equal, result.remaining};
        }

        template <typename Predicate>
        void select_rows(const Predicate predicate, const size_t n, uint32_t* out, const cudaStream_t s) {
            cuda_scratch::DeviceBuffer count_buffer(sizeof(int), s, "mrnf.topk.selected");
            const auto rows = thrust::counting_iterator<uint32_t>(0);
            auto select = [&](void* workspace, size_t& workspace_bytes) {
                return cub::DeviceSelect::If(workspace, workspace_bytes, rows, out, count_buffer.as<int>(),
                                             static_cast<int>(n), predicate, s);
            };
            cuda_scratch::CubWorkspace workspace("cub::DeviceSelect::If", s, select);
            workspace.run(select);
        }
    } // namespace

    float launch_kth_largest(
        const float* values,
        const bool* include,
        const size_t N,
        const size_t K,
        void* stream) {
        LFS_ASSERT(K > 0 && K <= N);
        LFS_ASSERT_MSG(N <= static_cast<size_t>(std::numeric_limits<int>::max()),
                       "k-th largest input exceeds CUB's int item-count limit");
        const cudaStream_t s = resolve_stream(stream);
        cuda_scratch::DeviceBuffer key_buffer(
            cuda_scratch::checked_bytes(N, sizeof(uint32_t), "k-th largest keys"), s, "mrnf.kth.keys");
        constexpr int threads = 256;
        orderable_value_bits_kernel<<<static_cast<int>((N + threads - 1) / threads), threads, 0, s>>>(
            values, include, key_buffer.as<uint32_t>(), N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.kth_keys");
        const auto threshold = radix_select_largest(key_buffer.as<uint32_t>(), N, K, s);
        const uint32_t bits = float_bits_from_orderable(threshold.bits);
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    void launch_gumbel_topk(
        const float* weights,
        size_t N,
        size_t K,
        uint64_t seed,
        int64_t* output_indices,
        void* stream) {

        LFS_ASSERT(K <= N);
        if (K == 0)
            return;

        LFS_ASSERT_MSG(N <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                       "MRNF Gumbel input exceeds the uint32 payload contract");
        LFS_ASSERT_MSG(N <= static_cast<size_t>(std::numeric_limits<int>::max()),
                       "MRNF Gumbel input exceeds CUB's int item-count limit");

        cudaStream_t s = resolve_stream(stream);

        if (K == N) {
            auto out_ptr = thrust::device_pointer_cast(output_indices);
            // Caller may consume output_indices via a Tensor whose home stream
            // is not `s` (scratch allocated earlier). Keep the host wait.
            thrust::sequence(thrust::cuda::par.on(s), out_ptr, out_ptr + N);
            return;
        }

        // Keys for every row (4 B each); the K winners are found by radix
        // select and only they are sorted, in descending key order with ties
        // by row as a stable sort of all keys would give.
        cuda_scratch::DeviceBuffer key_buffer(
            cuda_scratch::checked_bytes(N, sizeof(uint32_t), "MRNF Gumbel keys"), s, "mrnf.gumbel.keys");
        auto* keys = key_buffer.as<uint32_t>();
        constexpr int threads = 256;
        gumbel_key_bits_kernel<<<static_cast<int>((N + threads - 1) / threads), threads, 0, s>>>(
            weights, keys, N, seed);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.gumbel_keys");

        const auto threshold = radix_select_largest(keys, N, K, s);

        const size_t pick_bytes = cuda_scratch::checked_bytes(K, sizeof(uint32_t), "MRNF Gumbel picks");
        cuda_scratch::DeviceBuffer rows_buffer(pick_bytes, s, "mrnf.gumbel.rows");
        cuda_scratch::DeviceBuffer rows_sorted_buffer(pick_bytes, s, "mrnf.gumbel.rows_sorted");
        cuda_scratch::DeviceBuffer pick_keys_buffer(pick_bytes, s, "mrnf.gumbel.pick_keys");
        cuda_scratch::DeviceBuffer pick_keys_sorted_buffer(pick_bytes, s, "mrnf.gumbel.pick_keys_sorted");
        auto* rows = rows_buffer.as<uint32_t>();

        select_rows(KeyAbove{keys, threshold.bits}, N, rows, s);
        if (threshold.remaining > 0) {
            // Ties at the threshold go to the lowest rows.
            cuda_scratch::DeviceBuffer equal_buffer(
                cuda_scratch::checked_bytes(threshold.equal, sizeof(uint32_t), "MRNF Gumbel ties"), s,
                "mrnf.gumbel.ties");
            select_rows(KeyEquals{keys, threshold.bits}, N, equal_buffer.as<uint32_t>(), s);
            LFS_CUDA_CHECK_MSG(cudaMemcpyAsync(rows + threshold.greater, equal_buffer.as<uint32_t>(),
                                               threshold.remaining * sizeof(uint32_t),
                                               cudaMemcpyDeviceToDevice, s),
                               "MRNF Gumbel tie rows");
        }

        const int output_blocks = static_cast<int>((K + threads - 1) / threads);
        gather_key_bits_kernel<<<output_blocks, threads, 0, s>>>(keys, rows, pick_keys_buffer.as<uint32_t>(), K);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.gumbel_pick_keys");

        cub::DoubleBuffer<uint32_t> key_buffers(pick_keys_buffer.as<uint32_t>(), pick_keys_sorted_buffer.as<uint32_t>());
        cub::DoubleBuffer<uint32_t> row_buffers(rows, rows_sorted_buffer.as<uint32_t>());
        auto sort_pairs = [&](void* workspace, size_t& workspace_bytes) {
            return cub::DeviceRadixSort::SortPairsDescending(
                workspace, workspace_bytes, key_buffers, row_buffers, static_cast<int>(K), 0, 32, s);
        };
        cuda_scratch::CubWorkspace sort_workspace("cub::DeviceRadixSort::SortPairsDescending", s, sort_pairs);
        sort_workspace.run(sort_pairs);

        widen_gumbel_indices_kernel<<<output_blocks, threads, 0, s>>>(row_buffers.Current(), output_indices, K);
        LFS_CUDA_LAUNCH_CHECK(s, "MRNF Gumbel uint32 output widening");
    }

    __global__ void far_field_mask_kernel(
        const float* __restrict__ means,
        float centroid_x,
        float centroid_y,
        float centroid_z,
        float far_radius_sq,
        bool* __restrict__ far_out,
        size_t N) {

        const size_t idx = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
        if (idx >= N)
            return;

        const float dx = means[idx * 3 + 0] - centroid_x;
        const float dy = means[idx * 3 + 1] - centroid_y;
        const float dz = means[idx * 3 + 2] - centroid_z;
        far_out[idx] = (dx * dx + dy * dy + dz * dz) > far_radius_sq;
    }

    void launch_far_field_mask(
        const float* means,
        float centroid_x,
        float centroid_y,
        float centroid_z,
        float far_radius,
        bool* far_out,
        size_t N,
        void* stream) {

        if (N == 0)
            return;
        LFS_ASSERT(means != nullptr);
        LFS_ASSERT(far_out != nullptr);

        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);
        cudaStream_t s = resolve_stream(stream);
        const float far_radius_sq = far_radius * far_radius;
        far_field_mask_kernel<<<blocks, threads, 0, s>>>(
            means, centroid_x, centroid_y, centroid_z, far_radius_sq, far_out, N);
        LFS_CUDA_LAUNCH_CHECK(s, "training.mrnf.far_field_mask");
    }

} // namespace lfs::training::mrnf_strategy
