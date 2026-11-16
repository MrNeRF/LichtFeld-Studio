/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "core/tensor.hpp"
#include "lfs/cuda_scratch.hpp"
#include "match_kernels.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "kernel_stream.hpp"

namespace lfs::training::match {

    namespace {
        constexpr int kThreads = 256;
        constexpr float kDeadLogScale = -40.0f;

        int blocks_for(const size_t n) {
            return static_cast<int>((n + kThreads - 1) / kThreads);
        }

        __device__ __forceinline__ float sigmoid(const float x) {
            return 1.0f / (1.0f + expf(-x));
        }

        __device__ __forceinline__ float logit(float p) {
            p = fminf(fmaxf(p, 1e-12f), nextafterf(1.0f, 0.0f));
            return logf(p / (1.0f - p));
        }

        // Rotation matrix (row-major) of a raw (w, x, y, z) quaternion.
        __device__ __forceinline__ bool rotation_matrix(const float* q, float R[9]) {
            float w = q[0], x = q[1], y = q[2], z = q[3];
            const float norm2 = w * w + x * x + y * y + z * z;
            if (!(norm2 > 0.0f))
                return false;
            const float inv = rsqrtf(norm2);
            w *= inv;
            x *= inv;
            y *= inv;
            z *= inv;
            R[0] = 1.0f - 2.0f * (y * y + z * z);
            R[1] = 2.0f * (x * y - w * z);
            R[2] = 2.0f * (x * z + w * y);
            R[3] = 2.0f * (x * y + w * z);
            R[4] = 1.0f - 2.0f * (x * x + z * z);
            R[5] = 2.0f * (y * z - w * x);
            R[6] = 2.0f * (x * z - w * y);
            R[7] = 2.0f * (y * z + w * x);
            R[8] = 1.0f - 2.0f * (x * x + y * y);
            return true;
        }

        __device__ __forceinline__ float decayed_peak(const float peak, const float decay, const float step_peak) {
            return fmaxf(peak * decay, step_peak);
        }

        __global__ void peak_update_kernel(float* step_peak, float* peak, float* renders, const float peak_decay,
                                           const size_t n) {
            const size_t i = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
            if (i >= n)
                return;
            const float s = step_peak[i];
            step_peak[i] = 0.0f;
            if (renders != nullptr && s > 0.0f)
                renders[i] += 1.0f;
            peak[i] = decayed_peak(peak[i], peak_decay, s);
        }

        __global__ void step_update_kernel(const StepRows rows, const StepSettings settings, const size_t n) {
            const size_t i = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
            if (i >= n)
                return;
            const float step_peak = rows.step_peak[i];
            rows.step_peak[i] = 0.0f;
            const float error = rows.error[i];
            rows.error[i] = 0.0f;
            rows.streak[i] = step_peak > 0.0f ? 0.0f : rows.streak[i] + 1.0f;
            if (rows.renders != nullptr && step_peak > 0.0f)
                rows.renders[i] += 1.0f;
            if (!settings.scoring)
                return;
            const float radius = rows.radius != nullptr ? rows.radius[i] : 0.0f;
            if (radius > settings.screen_limit)
                rows.oversize[i] += log2f(radius / settings.screen_limit);
            const bool visible = rows.visible[i] > 0.0f;
            if (visible && error != 0.0f && isfinite(error)) {
                const float w = powf(fabsf(error), settings.score_power);
                const float count = rows.score_count[i];
                rows.score[i] = (rows.score[i] * count + w) / (count + 1.0f);
                rows.score_count[i] = count + 1.0f;
            }
            rows.peak[i] = decayed_peak(rows.peak[i], settings.peak_decay, step_peak);
        }

        __global__ void size_clip_kernel(float* log_scales, const float* radius, const float limit,
                                         const float max_factor, const size_t n) {
            const size_t i = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
            if (i >= n)
                return;
            const float f = fminf(radius[i] / limit, max_factor);
            if (!(f > 1.0f))
                return;
            const float shrink = logf(f);
            log_scales[i * 3 + 0] -= shrink;
            log_scales[i * 3 + 1] -= shrink;
            log_scales[i * 3 + 2] -= shrink;
        }

        __global__ void dead_mask_kernel(const float* means, const float* log_scales, const float* raw_rotations,
                                         const float* raw_opacities, const float* peak, const float* streak,
                                         const float* score, const DeadSettings settings, uint8_t* dead,
                                         uint8_t* eligible, const float* renders, const size_t n) {
            const size_t i = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
            if (i >= n)
                return;
            bool finite = isfinite(raw_opacities[i]);
            float s_max = -INFINITY;
            for (int c = 0; c < 3; ++c) {
                finite = finite && isfinite(means[i * 3 + c]) && isfinite(log_scales[i * 3 + c]);
                s_max = fmaxf(s_max, log_scales[i * 3 + c]);
            }
            for (int c = 0; c < 4; ++c)
                finite = finite && isfinite(raw_rotations[i * 4 + c]);
            const bool is_dead = !finite || sigmoid(raw_opacities[i]) <= settings.min_opacity ||
                                 (settings.dead_peak > 0.0f && peak[i] <= settings.dead_peak &&
                                  (renders == nullptr || renders[i] >= settings.peak_grace)) ||
                                 (settings.dead_streak > 0.0f && streak[i] >= settings.dead_streak) ||
                                 s_max <= kDeadLogScale;
            dead[i] = is_dead ? 1 : 0;
            eligible[i] = (!is_dead && score[i] > 0.0f) ? 1 : 0;
        }

        __device__ __forceinline__ float hash_uniform(const uint64_t seed, const uint64_t i) {
            // splitmix64 of the seed and the row, top 24 bits, in (0, 1].
            uint64_t h = seed ^ (i * 0x9E3779B97F4A7C15ull);
            h += 0x9E3779B97F4A7C15ull;
            h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
            h = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
            h ^= h >> 31;
            return (static_cast<float>(h >> 40) + 1.0f) * (1.0f / 16777216.0f);
        }

        __global__ void race_keys_kernel(const float* score, const float* oversize, const uint8_t* mask,
                                         const size_t n, const uint64_t seed, float* keys, int32_t* rows,
                                         unsigned int* positive) {
            const size_t i = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
            if (i >= n)
                return;
            float w = fmaxf(score[i], 0.0f);
            if (oversize != nullptr)
                w *= oversize[i] + 0.001f;
            if (mask != nullptr && mask[i] == 0)
                w = 0.0f;
            float key = INFINITY;
            if (w > 0.0f && isfinite(w)) {
                key = -logf(hash_uniform(seed, i)) / w;
                atomicAdd(positive, 1u);
            }
            keys[i] = key;
            rows[i] = static_cast<int32_t>(i);
        }

        __global__ void widen_rows_kernel(const int32_t* rows, const size_t count, int64_t* out) {
            const size_t j = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
            if (j < count)
                out[j] = rows[j];
        }

        __global__ void long_axis_split_kernel(const SplitRows rows, const int64_t* src, const int64_t* dst,
                                               const size_t count, const float opacity_share) {
            const size_t j = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
            if (j >= count)
                return;
            const int64_t s = src[j];
            const int64_t d = dst[j];
            float* qs = rows.raw_rotations + s * 4;
            float* ls = rows.log_scales + s * 3;
            int axis = 0;
            for (int c = 1; c < 3; ++c)
                if (ls[c] > ls[axis])
                    axis = c;
            float R[9];
            float delta[3] = {0.0f, 0.0f, 0.0f};
            if (rotation_matrix(qs, R)) {
                const float half = 0.5f * expf(ls[axis]);
                for (int c = 0; c < 3; ++c)
                    delta[c] = R[3 * c + axis] * half;
            }
            for (int c = 0; c < 3; ++c) {
                const float mean = rows.means[s * 3 + c];
                rows.means[s * 3 + c] = mean - delta[c];
                rows.means[d * 3 + c] = mean + delta[c];
            }
            const float long_shrink = logf(0.5f);
            const float other_shrink = logf(0.85f);
            for (int c = 0; c < 3; ++c) {
                const float scaled = ls[c] + (c == axis ? long_shrink : other_shrink);
                ls[c] = scaled;
                rows.log_scales[d * 3 + c] = scaled;
            }
            for (int c = 0; c < 4; ++c)
                rows.raw_rotations[d * 4 + c] = qs[c];
            for (int c = 0; c < 3; ++c)
                rows.sh0[d * 3 + c] = rows.sh0[s * 3 + c];
            const float raw = logit(opacity_share * sigmoid(rows.raw_opacities[s]));
            rows.raw_opacities[s] = raw;
            rows.raw_opacities[d] = raw;
            const float peak = opacity_share * rows.peak[s];
            rows.peak[s] = peak;
            rows.peak[d] = peak;
            rows.streak[s] = 0.0f;
            rows.streak[d] = 0.0f;
            if (rows.renders != nullptr)
                rows.renders[d] = 0.0f;
            rows.score[d] = rows.score[s];
            rows.score_count[d] = rows.score_count[s];
            rows.oversize[d] = rows.oversize[s];
        }

        __global__ void walk_kernel(float* means, const float* log_scales, const float* raw_rotations,
                                    const float* raw_opacities, float* visible, const float* peak,
                                    const float amplitude, const size_t n, const uint64_t seed) {
            const size_t i = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
            if (i >= n)
                return;
            const bool seen = visible[i] > 0.0f;
            visible[i] = 0.0f;
            if (!seen || !(amplitude > 0.0f))
                return;
            const float opacity = sigmoid(raw_opacities[i]);
            const float lifted = 255.0f * opacity;
            if (!(lifted > 1.0f))
                return;
            const float extent = sqrtf(2.0f * logf(lifted));
            const float gate = peak != nullptr ? fminf(opacity, peak[i]) : opacity;
            const float k = fminf(amplitude * extent * powf(fmaxf(1.0f - gate, 0.0f), 150.0f), 1.0f);
            if (!(k > 0.0f))
                return;
            float R[9];
            if (!rotation_matrix(raw_rotations + i * 4, R))
                return;
            curandStatePhilox4_32_10_t rng;
            curand_init(seed, i, 0, &rng);
            const float4 z = curand_normal4(&rng);
            const float local[3] = {k * expf(log_scales[i * 3 + 0]) * z.x, k * expf(log_scales[i * 3 + 1]) * z.y,
                                    k * expf(log_scales[i * 3 + 2]) * z.z};
            for (int d = 0; d < 3; ++d) {
                const float step = R[3 * d] * local[0] + R[3 * d + 1] * local[1] + R[3 * d + 2] * local[2];
                if (isfinite(step))
                    means[i * 3 + d] += step;
            }
        }

        __global__ void lowest_keys_kernel(const float* key, const float* renders, const float grace,
                                           const uint8_t* dead, const size_t n, float* keys, int32_t* rows,
                                           unsigned int* candidates) {
            const size_t i = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
            if (i >= n)
                return;
            const bool candidate = dead[i] == 0 && (renders == nullptr || renders[i] >= grace) && isfinite(key[i]);
            keys[i] = candidate ? key[i] : INFINITY;
            rows[i] = static_cast<int32_t>(i);
            if (candidate)
                atomicAdd(candidates, 1u);
        }

        __global__ void mark_rows_kernel(const int32_t* rows, const size_t count, uint8_t* dead, uint8_t* eligible) {
            const size_t j = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
            if (j >= count)
                return;
            dead[rows[j]] = 1;
            if (eligible != nullptr)
                eligible[rows[j]] = 0;
        }

        __global__ void zero_rows_kernel(float* a, float* b, float* c, float* d, const size_t n) {
            const size_t i = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
            if (i >= n)
                return;
            if (a != nullptr)
                a[i] = 0.0f;
            if (b != nullptr)
                b[i] = 0.0f;
            if (c != nullptr)
                c[i] = 0.0f;
            if (d != nullptr)
                d[i] = 0.0f;
        }
    } // namespace

    void launch_step_update(const StepRows& rows, const StepSettings& settings, const size_t n, void* stream) {
        if (n == 0)
            return;
        LFS_ASSERT_MSG(rows.error && rows.step_peak && rows.visible && rows.score && rows.score_count &&
                           rows.oversize && rows.peak && rows.streak,
                       "match step update needs every row");
        const cudaStream_t s = resolve_stream(stream);
        step_update_kernel<<<blocks_for(n), kThreads, 0, s>>>(rows, settings, n);
        LFS_CUDA_LAUNCH_CHECK(s, "training.match.step_update");
    }

    void launch_size_clip(float* log_scales, const float* radius, const float limit, const float max_factor,
                          const size_t n, void* stream) {
        if (n == 0 || !(max_factor > 1.0f))
            return;
        const cudaStream_t s = resolve_stream(stream);
        size_clip_kernel<<<blocks_for(n), kThreads, 0, s>>>(log_scales, radius, limit, max_factor, n);
        LFS_CUDA_LAUNCH_CHECK(s, "training.match.size_clip");
    }

    void launch_dead_mask(const float* means, const float* log_scales, const float* raw_rotations,
                          const float* raw_opacities, const float* peak, const float* streak, const float* score,
                          const DeadSettings& settings, uint8_t* dead, uint8_t* eligible, const size_t n,
                          const float* renders, void* stream) {
        if (n == 0)
            return;
        const cudaStream_t s = resolve_stream(stream);
        dead_mask_kernel<<<blocks_for(n), kThreads, 0, s>>>(means, log_scales, raw_rotations, raw_opacities, peak,
                                                            streak, score, settings, dead, eligible, renders, n);
        LFS_CUDA_LAUNCH_CHECK(s, "training.match.dead_mask");
    }

    size_t draw_without_replacement(const float* score, const float* oversize, const uint8_t* mask, const size_t n,
                                    const size_t count, const uint64_t seed, int64_t* out, void* stream) {
        if (n == 0 || count == 0)
            return 0;
        LFS_ASSERT_MSG(n <= static_cast<size_t>(INT_MAX), "match draw exceeds int32 rows");
        const cudaStream_t s = resolve_stream(stream);
        const lfs::core::CUDAStreamGuard stream_guard(s);
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;
        auto keys_a = Tensor::empty({n}, Device::CUDA, DataType::Float32);
        auto keys_b = Tensor::empty({n}, Device::CUDA, DataType::Float32);
        auto rows_a = Tensor::empty({n}, Device::CUDA, DataType::Int32);
        auto rows_b = Tensor::empty({n}, Device::CUDA, DataType::Int32);
        auto positive = Tensor::zeros({1}, Device::CUDA, DataType::Int32);
        auto* positive_ptr = reinterpret_cast<unsigned int*>(positive.ptr<int32_t>());
        race_keys_kernel<<<blocks_for(n), kThreads, 0, s>>>(score, oversize, mask, n, seed, keys_a.ptr<float>(),
                                                            rows_a.ptr<int32_t>(), positive_ptr);
        LFS_CUDA_LAUNCH_CHECK(s, "training.match.race_keys");
        unsigned int h_positive = 0;
        LFS_CUDA_CHECK_MSG(cudaMemcpyAsync(&h_positive, positive_ptr, sizeof(h_positive), cudaMemcpyDeviceToHost, s),
                           "match draw positive count");
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(s), "match draw sync");
        const size_t drawn = std::min(count, static_cast<size_t>(h_positive));
        if (drawn == 0)
            return 0;
        cub::DoubleBuffer<float> keys(keys_a.ptr<float>(), keys_b.ptr<float>());
        cub::DoubleBuffer<int32_t> values(rows_a.ptr<int32_t>(), rows_b.ptr<int32_t>());
        const auto radix_sort = [&](void* workspace, size_t& workspace_bytes) {
            return cub::DeviceRadixSort::SortPairs(workspace, workspace_bytes, keys, values, static_cast<int>(n), 0,
                                                   32, s);
        };
        cuda_scratch::CubWorkspace workspace("cub::DeviceRadixSort::SortPairs", s, radix_sort);
        workspace.run(radix_sort);
        widen_rows_kernel<<<blocks_for(drawn), kThreads, 0, s>>>(values.Current(), drawn, out);
        LFS_CUDA_LAUNCH_CHECK(s, "training.match.widen_rows");
        return drawn;
    }

    void launch_peak_update(float* step_peak, float* peak, float* renders, const float peak_decay, const size_t n,
                            void* stream) {
        if (n == 0)
            return;
        const cudaStream_t s = resolve_stream(stream);
        peak_update_kernel<<<blocks_for(n), kThreads, 0, s>>>(step_peak, peak, renders, peak_decay, n);
        LFS_CUDA_LAUNCH_CHECK(s, "training.match.peak_update");
    }

    size_t mark_lowest_dead(const float* key, const float* renders, const float grace, uint8_t* dead,
                            uint8_t* eligible, const size_t n, const size_t count, void* stream) {
        if (n == 0 || count == 0)
            return 0;
        LFS_ASSERT_MSG(n <= static_cast<size_t>(INT_MAX), "match death marking exceeds int32 rows");
        const cudaStream_t s = resolve_stream(stream);
        const lfs::core::CUDAStreamGuard stream_guard(s);
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;
        auto keys_a = Tensor::empty({n}, Device::CUDA, DataType::Float32);
        auto keys_b = Tensor::empty({n}, Device::CUDA, DataType::Float32);
        auto rows_a = Tensor::empty({n}, Device::CUDA, DataType::Int32);
        auto rows_b = Tensor::empty({n}, Device::CUDA, DataType::Int32);
        auto candidates = Tensor::zeros({1}, Device::CUDA, DataType::Int32);
        auto* candidates_ptr = reinterpret_cast<unsigned int*>(candidates.ptr<int32_t>());
        lowest_keys_kernel<<<blocks_for(n), kThreads, 0, s>>>(key, renders, grace, dead, n, keys_a.ptr<float>(),
                                                              rows_a.ptr<int32_t>(), candidates_ptr);
        LFS_CUDA_LAUNCH_CHECK(s, "training.match.lowest_keys");
        unsigned int h_candidates = 0;
        LFS_CUDA_CHECK_MSG(cudaMemcpyAsync(&h_candidates, candidates_ptr, sizeof(h_candidates),
                                           cudaMemcpyDeviceToHost, s),
                           "match death candidate count");
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(s), "match death sync");
        const size_t marked = std::min(count, static_cast<size_t>(h_candidates));
        if (marked == 0)
            return 0;
        cub::DoubleBuffer<float> keys(keys_a.ptr<float>(), keys_b.ptr<float>());
        cub::DoubleBuffer<int32_t> values(rows_a.ptr<int32_t>(), rows_b.ptr<int32_t>());
        const auto radix_sort = [&](void* workspace, size_t& workspace_bytes) {
            return cub::DeviceRadixSort::SortPairs(workspace, workspace_bytes, keys, values, static_cast<int>(n), 0,
                                                   32, s);
        };
        cuda_scratch::CubWorkspace workspace("cub::DeviceRadixSort::SortPairs", s, radix_sort);
        workspace.run(radix_sort);
        mark_rows_kernel<<<blocks_for(marked), kThreads, 0, s>>>(values.Current(), marked, dead, eligible);
        LFS_CUDA_LAUNCH_CHECK(s, "training.match.mark_dead");
        return marked;
    }

    void launch_long_axis_split(const SplitRows& rows, const int64_t* src, const int64_t* dst, const size_t count,
                                const float opacity_share, void* stream) {
        if (count == 0)
            return;
        const cudaStream_t s = resolve_stream(stream);
        long_axis_split_kernel<<<blocks_for(count), kThreads, 0, s>>>(rows, src, dst, count, opacity_share);
        LFS_CUDA_LAUNCH_CHECK(s, "training.match.long_axis_split");
    }

    void launch_walk(float* means, const float* log_scales, const float* raw_rotations, const float* raw_opacities,
                     float* visible, const float* peak, const float amplitude, const size_t n,
                     const uint64_t seed, void* stream) {
        if (n == 0)
            return;
        const cudaStream_t s = resolve_stream(stream);
        walk_kernel<<<blocks_for(n), kThreads, 0, s>>>(means, log_scales, raw_rotations, raw_opacities,
                                                       visible, peak, amplitude, n, seed);
        LFS_CUDA_LAUNCH_CHECK(s, "training.match.walk");
    }

    void launch_zero_rows(float* a, float* b, float* c, float* d, const size_t n, void* stream) {
        if (n == 0)
            return;
        const cudaStream_t s = resolve_stream(stream);
        zero_rows_kernel<<<blocks_for(n), kThreads, 0, s>>>(a, b, c, d, n);
        LFS_CUDA_LAUNCH_CHECK(s, "training.match.zero_rows");
    }

} // namespace lfs::training::match
