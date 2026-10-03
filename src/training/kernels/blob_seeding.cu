/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "blob_seeding.hpp"
#include "core/cuda_error.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <nvtx3/nvToolsExt.h>

namespace lfs::training::kernels::blob_seeding {
    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;
        using lfs::core::TensorShape;

        constexpr int BLOCK_SIZE = 256;
        constexpr int SMOOTH_RADIUS = 3;
        constexpr float SMOOTH_SIGMA = 0.7f;
        constexpr int MORPHOLOGY_RADIUS = 3;
        constexpr int PEAK_NMS_RADIUS = 2;
        constexpr int PIXELS_PER_WORD = 16;
        constexpr float BRIGHT_CONTRAST_MIN = 28.0f;
        constexpr float DARK_CONTRAST_MIN = 22.0f;

        constexpr int MAX_DEPTH_SAMPLES = 1024;
        constexpr float DEPTH_STEP_RATIO = 1.0085f;
        constexpr float MIN_VISIBLE_VIEWS = 8.0f;
        constexpr int CANDIDATES = 6;
        constexpr int CANDIDATE_NMS_RADIUS = 7;
        constexpr float CANDIDATE_MIN_SCORE = 3.0f;
        constexpr float ACCEPT_MIN_SUPPORT = 5.0f;
        constexpr float ACCEPT_MIN_HIT_RATE = 0.25f;
        constexpr float AMBIGUOUS_SCORE_RATIO = 0.7f;
        constexpr float AMBIGUOUS_DEPTH_RATIO = 1.15f;
        constexpr float MIN_CAMERA_DEPTH = 0.1f;

        struct SmoothWeights {
            float w[2 * SMOOTH_RADIUS + 1];
        };

        SmoothWeights gaussian_weights() {
            SmoothWeights g{};
            float sum = 0.0f;
            for (int k = -SMOOTH_RADIUS; k <= SMOOTH_RADIUS; ++k) {
                g.w[k + SMOOTH_RADIUS] = std::exp(-0.5f * k * k / (SMOOTH_SIGMA * SMOOTH_SIGMA));
                sum += g.w[k + SMOOTH_RADIUS];
            }
            for (float& w : g.w)
                w /= sum;
            return g;
        }

        __device__ int reflect_index(int index, const int length) {
            const int period = 2 * length;
            index %= period;
            if (index < 0)
                index += period;
            return index < length ? index : period - 1 - index;
        }

        __global__ void smooth_luminance_rows(const float* image, float* rows, const int height, const int width,
                                              const SmoothWeights g) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y;
            if (x >= width)
                return;
            const size_t plane = static_cast<size_t>(height) * width;
            float acc = 0.0f;
            for (int k = -SMOOTH_RADIUS; k <= SMOOTH_RADIUS; ++k) {
                const size_t i = static_cast<size_t>(y) * width + reflect_index(x + k, width);
                const float luminance = 255.0f * (0.299f * image[i] + 0.587f * image[plane + i] + 0.114f * image[2 * plane + i]);
                acc += g.w[k + SMOOTH_RADIUS] * luminance;
            }
            rows[static_cast<size_t>(y) * width + x] = acc;
        }

        __global__ void smooth_columns(const float* rows, float2* smooth, const int height, const int width,
                                       const SmoothWeights g) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y;
            if (x >= width)
                return;
            float acc = 0.0f;
            for (int k = -SMOOTH_RADIUS; k <= SMOOTH_RADIUS; ++k)
                acc += g.w[k + SMOOTH_RADIUS] * rows[static_cast<size_t>(reflect_index(y + k, height)) * width + x];
            smooth[static_cast<size_t>(y) * width + x] = make_float2(acc, acc);
        }

        // Min/max filters over a clamped window equal scipy's reflect mode: mirrored taps repeat in-window pixels.
        template <bool Horizontal>
        __global__ void extrema_pass(const float2* in, float2* out, const int height, const int width, const int radius,
                                     const bool x_max, const bool y_max) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y;
            if (x >= width)
                return;
            float2 acc = in[static_cast<size_t>(y) * width + x];
            for (int k = -radius; k <= radius; ++k) {
                const int sx = Horizontal ? min(max(x + k, 0), width - 1) : x;
                const int sy = Horizontal ? y : min(max(y + k, 0), height - 1);
                const float2 v = in[static_cast<size_t>(sy) * width + sx];
                acc.x = x_max ? fmaxf(acc.x, v.x) : fminf(acc.x, v.x);
                acc.y = y_max ? fmaxf(acc.y, v.y) : fminf(acc.y, v.y);
            }
            out[static_cast<size_t>(y) * width + x] = acc;
        }

        __global__ void peak_contrast(const float2* smooth, const float2* opening_closing, float2* contrast, const size_t n) {
            const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
            if (i >= n)
                return;
            const float s = smooth[i].x;
            contrast[i] = make_float2(s - opening_closing[i].x, opening_closing[i].y - s);
        }

        __global__ void emit_peaks(const float2* contrast, const float2* window_max, const float* image,
                                   uint8_t* mask, float* peaks, int* count, const int capacity,
                                   const int height, const int width, const float view) {
            const size_t n = static_cast<size_t>(height) * width;
            const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
            if (i >= n)
                return;
            const float2 c = contrast[i];
            const float2 m = window_max[i];
            const bool bright = c.x == m.x && c.x >= BRIGHT_CONTRAST_MIN;
            const bool dark = c.y == m.y && c.y >= DARK_CONTRAST_MIN;
            mask[i] = static_cast<uint8_t>(bright) | static_cast<uint8_t>(dark) << 1;
            for (int polarity = 0; polarity < 2; ++polarity) {
                if (!(polarity == 0 ? bright : dark))
                    continue;
                const int slot = atomicAdd(count, 1);
                if (slot >= capacity)
                    continue;
                float* row = peaks + static_cast<size_t>(slot) * PeakFieldCount;
                row[PeakX] = static_cast<float>(i % width);
                row[PeakY] = static_cast<float>(i / width);
                row[PeakView] = view;
                row[PeakPolarity] = static_cast<float>(polarity);
                row[PeakR] = image[i];
                row[PeakG] = image[n + i];
                row[PeakB] = image[2 * n + i];
            }
        }

        __global__ void pack_dilated_peaks(const uint8_t* mask, uint32_t* bitmap, unsigned int* set_bits,
                                           const int height, const int width, const size_t words) {
            const size_t word = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
            uint32_t bits = 0;
            if (word < words) {
                const size_t n = static_cast<size_t>(height) * width;
                for (int j = 0; j < PIXELS_PER_WORD; ++j) {
                    const size_t i = word * PIXELS_PER_WORD + j;
                    if (i >= n)
                        break;
                    const int x = static_cast<int>(i % width), y = static_cast<int>(i / width);
                    uint32_t dilated = 0;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int sx = x + dx, sy = y + dy;
                            if (sx >= 0 && sx < width && sy >= 0 && sy < height)
                                dilated |= mask[static_cast<size_t>(sy) * width + sx];
                        }
                    bits |= dilated << (2 * j);
                }
                bitmap[word] = bits;
            }
            constexpr uint32_t BRIGHT_BITS = 0x55555555u;
            unsigned int bright = __popc(bits & BRIGHT_BITS), dark = __popc(bits & (BRIGHT_BITS << 1));
            for (int offset = 16; offset > 0; offset /= 2) {
                bright += __shfl_down_sync(0xffffffffu, bright, offset);
                dark += __shfl_down_sync(0xffffffffu, dark, offset);
            }
            if ((threadIdx.x & 31) == 0) {
                atomicAdd(&set_bits[0], bright);
                atomicAdd(&set_bits[1], dark);
            }
        }

        __device__ bool bitmap_hit(const SweepView& view, const int x, const int y, const int polarity) {
            const size_t i = static_cast<size_t>(y) * view.width + x;
            return (view.bitmap[i / PIXELS_PER_WORD] >> (2 * (i % PIXELS_PER_WORD) + polarity)) & 1u;
        }

        __global__ void sweep_kernel(const float* peaks, const SweepView* views, const int* neighbors,
                                     const int neighbor_count, float* out) {
            __shared__ float score[MAX_DEPTH_SAMPLES];
            __shared__ float visible[MAX_DEPTH_SAMPLES];
            __shared__ float smoothed[MAX_DEPTH_SAMPLES];

            const float* peak = peaks + static_cast<size_t>(blockIdx.x) * PeakFieldCount;
            const int view_index = static_cast<int>(peak[PeakView]);
            const int polarity = static_cast<int>(peak[PeakPolarity]);
            const SweepView& v = views[view_index];
            float* result = out + static_cast<size_t>(blockIdx.x) * SweepFieldCount;

            const bool has_range = v.z_min > 0.0f && v.z_max > v.z_min;
            const int samples = has_range
                                    ? min(MAX_DEPTH_SAMPLES, static_cast<int>(ceilf(logf(v.z_max / v.z_min) / logf(DEPTH_STEP_RATIO))) + 1)
                                    : 0;
            if (samples < 2 * CANDIDATE_NMS_RADIUS + 1) {
                if (threadIdx.x == 0) {
                    result[SweepDepth] = 0.0f;
                    result[SweepSupport] = 0.0f;
                    result[SweepConfidence] = static_cast<float>(Confidence::None);
                }
                return;
            }
            const float log_step = logf(v.z_max / v.z_min) / static_cast<float>(samples - 1);

            const float dx = (peak[PeakX] + 0.5f - v.cx) / v.fx;
            const float dy = (peak[PeakY] + 0.5f - v.cy) / v.fy;
            const float* R = v.R;
            const float Cx = -(R[0] * v.t[0] + R[3] * v.t[1] + R[6] * v.t[2]);
            const float Cy = -(R[1] * v.t[0] + R[4] * v.t[1] + R[7] * v.t[2]);
            const float Cz = -(R[2] * v.t[0] + R[5] * v.t[1] + R[8] * v.t[2]);
            const float wx = R[0] * dx + R[3] * dy + R[6];
            const float wy = R[1] * dx + R[4] * dy + R[7];
            const float wz = R[2] * dx + R[5] * dy + R[8];
            const int* own_neighbors = neighbors + static_cast<size_t>(view_index) * neighbor_count;

            for (int j = threadIdx.x; j < samples; j += blockDim.x) {
                const float z = v.z_min * expf(log_step * j);
                const float px = Cx + z * wx, py = Cy + z * wy, pz = Cz + z * wz;
                float s = 0.0f, seen = 0.0f;
                for (int k = 0; k < neighbor_count; ++k) {
                    const int n = own_neighbors[k];
                    if (n < 0)
                        break;
                    const SweepView& o = views[n];
                    const float X = o.R[0] * px + o.R[1] * py + o.R[2] * pz + o.t[0];
                    const float Y = o.R[3] * px + o.R[4] * py + o.R[5] * pz + o.t[1];
                    const float Z = o.R[6] * px + o.R[7] * py + o.R[8] * pz + o.t[2];
                    if (Z <= MIN_CAMERA_DEPTH)
                        continue;
                    const int u = __float2int_rn(o.fx * X / Z + o.cx - 0.5f);
                    const int w = __float2int_rn(o.fy * Y / Z + o.cy - 0.5f);
                    if (u < 0 || u >= o.width || w < 0 || w >= o.height)
                        continue;
                    seen += 1.0f;
                    s += static_cast<float>(bitmap_hit(o, u, w, polarity)) - o.density[polarity];
                }
                score[j] = s;
                visible[j] = seen;
            }
            __syncthreads();
            for (int j = threadIdx.x; j < samples; j += blockDim.x) {
                const float a = score[j > 0 ? j - 1 : 0], b = score[j], c = score[j + 1 < samples ? j + 1 : samples - 1];
                smoothed[j] = visible[j] < MIN_VISIBLE_VIEWS ? -1e9f : (a + b + c) / 3.0f;
            }
            __syncthreads();
            if (threadIdx.x != 0)
                return;

            float cand_score[CANDIDATES], cand_rate[CANDIDATES], cand_depth[CANDIDATES];
            int found = 0;
            for (int j = 0; j < samples; ++j) {
                const float sj = smoothed[j];
                if (!(sj > CANDIDATE_MIN_SCORE))
                    continue;
                bool local_max = true;
                for (int o = -CANDIDATE_NMS_RADIUS; o <= CANDIDATE_NMS_RADIUS && local_max; ++o)
                    local_max = smoothed[reflect_index(j + o, samples)] <= sj;
                if (!local_max)
                    continue;
                int pos = found < CANDIDATES ? found++ : CANDIDATES;
                while (pos > 0 && cand_score[pos - 1] < sj) {
                    if (pos < CANDIDATES) {
                        cand_score[pos] = cand_score[pos - 1];
                        cand_rate[pos] = cand_rate[pos - 1];
                        cand_depth[pos] = cand_depth[pos - 1];
                    }
                    --pos;
                }
                if (pos < CANDIDATES) {
                    cand_score[pos] = sj;
                    cand_rate[pos] = sj / fmaxf(visible[j], 1.0f);
                    cand_depth[pos] = v.z_min * expf(log_step * j);
                }
            }

            int best = -1, second = -1;
            for (int i = 0; i < found; ++i) {
                if (cand_rate[i] < ACCEPT_MIN_HIT_RATE || cand_score[i] < ACCEPT_MIN_SUPPORT)
                    continue;
                if (best < 0)
                    best = i;
                else if (second < 0)
                    second = i;
            }
            Confidence confidence = Confidence::None;
            int pick = -1;
            if (best >= 0) {
                pick = best;
                confidence = Confidence::Unique;
                if (second >= 0 && cand_score[second] >= AMBIGUOUS_SCORE_RATIO * cand_score[best] &&
                    fabsf(logf(cand_depth[second] / cand_depth[best])) > logf(AMBIGUOUS_DEPTH_RATIO))
                    confidence = Confidence::Ambiguous;
            } else if (found > 0) {
                pick = 0;
                confidence = Confidence::Low;
            }
            result[SweepDepth] = pick >= 0 ? cand_depth[pick] : 0.0f;
            result[SweepSupport] = pick >= 0 ? cand_score[pick] : 0.0f;
            result[SweepConfidence] = static_cast<float>(confidence);
        }

        unsigned int blocks_for(const size_t n) {
            return static_cast<unsigned int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        }
    } // namespace

    ViewPeaks detect_peaks(const Tensor& image, const int view) {
        assert(image.device() == Device::CUDA && image.dtype() == DataType::Float32);
        assert(image.ndim() == 3 && image.shape()[0] >= 3);
        nvtxRangePush("blob_seeding.detect_peaks");
        const auto source = image.contiguous();
        const int height = static_cast<int>(source.shape()[1]);
        const int width = static_cast<int>(source.shape()[2]);
        const size_t n = static_cast<size_t>(height) * width;
        const size_t words = (n + PIXELS_PER_WORD - 1) / PIXELS_PER_WORD;
        const int capacity = static_cast<int>(n / 4 + 1);
        const cudaStream_t stream = source.stream();

        auto make = [stream](TensorShape shape, DataType dtype) {
            auto t = Tensor::empty(std::move(shape), Device::CUDA, dtype);
            t.set_stream(stream);
            return t;
        };
        auto rows = make({n}, DataType::Float32);
        auto a = make({n, 2}, DataType::Float32);
        auto b = make({n, 2}, DataType::Float32);
        auto c = make({n, 2}, DataType::Float32);
        auto mask = make({n}, DataType::UInt8);
        auto peaks = make({static_cast<size_t>(capacity), static_cast<size_t>(PeakFieldCount)}, DataType::Float32);
        auto counters = make({3}, DataType::Int32);
        counters.zero_();
        auto bitmap = make({words}, DataType::UInt32);

        auto* pa = reinterpret_cast<float2*>(a.ptr<float>());
        auto* pb = reinterpret_cast<float2*>(b.ptr<float>());
        auto* pc = reinterpret_cast<float2*>(c.ptr<float>());
        const dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, height);
        const auto g = gaussian_weights();

        smooth_luminance_rows<<<grid, BLOCK_SIZE, 0, stream>>>(source.ptr<float>(), rows.ptr<float>(), height, width, g);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.blob_seeding.smooth_rows");
        smooth_columns<<<grid, BLOCK_SIZE, 0, stream>>>(rows.ptr<float>(), pc, height, width, g);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.blob_seeding.smooth_columns");
        // c holds (smooth, smooth): erode and dilate, then dilate the erosion and erode the dilation.
        extrema_pass<true><<<grid, BLOCK_SIZE, 0, stream>>>(pc, pa, height, width, MORPHOLOGY_RADIUS, false, true);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.blob_seeding.morphology");
        extrema_pass<false><<<grid, BLOCK_SIZE, 0, stream>>>(pa, pb, height, width, MORPHOLOGY_RADIUS, false, true);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.blob_seeding.morphology");
        extrema_pass<true><<<grid, BLOCK_SIZE, 0, stream>>>(pb, pa, height, width, MORPHOLOGY_RADIUS, true, false);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.blob_seeding.morphology");
        extrema_pass<false><<<grid, BLOCK_SIZE, 0, stream>>>(pa, pb, height, width, MORPHOLOGY_RADIUS, true, false);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.blob_seeding.morphology");
        peak_contrast<<<blocks_for(n), BLOCK_SIZE, 0, stream>>>(pc, pb, pa, n);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.blob_seeding.contrast");
        extrema_pass<true><<<grid, BLOCK_SIZE, 0, stream>>>(pa, pb, height, width, PEAK_NMS_RADIUS, true, true);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.blob_seeding.peak_window");
        extrema_pass<false><<<grid, BLOCK_SIZE, 0, stream>>>(pb, pc, height, width, PEAK_NMS_RADIUS, true, true);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.blob_seeding.peak_window");
        emit_peaks<<<blocks_for(n), BLOCK_SIZE, 0, stream>>>(pa, pc, source.ptr<float>(), mask.ptr<uint8_t>(),
                                                             peaks.ptr<float>(), counters.ptr<int>(), capacity,
                                                             height, width, static_cast<float>(view));
        LFS_CUDA_LAUNCH_CHECK(stream, "training.blob_seeding.emit_peaks");
        pack_dilated_peaks<<<blocks_for(words), BLOCK_SIZE, 0, stream>>>(
            mask.ptr<uint8_t>(), bitmap.ptr<uint32_t>(), reinterpret_cast<unsigned int*>(counters.ptr<int>() + 1),
            height, width, words);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.blob_seeding.pack");

        const auto host = counters.cpu();
        const int* counts = host.ptr<int>();
        const size_t emitted = static_cast<size_t>(std::min(counts[0], capacity));
        ViewPeaks result;
        result.peaks = peaks.slice(0, 0, emitted).clone();
        result.bitmap = std::move(bitmap);
        result.density = {static_cast<float>(static_cast<unsigned int>(counts[1])) / static_cast<float>(n),
                          static_cast<float>(static_cast<unsigned int>(counts[2])) / static_cast<float>(n)};
        nvtxRangePop();
        return result;
    }

    Tensor sweep_peaks(const Tensor& peaks, const SweepView* views, const Tensor& neighbors) {
        assert(peaks.device() == Device::CUDA && peaks.ndim() == 2 && peaks.shape()[1] == PeakFieldCount);
        assert(neighbors.device() == Device::CUDA && neighbors.dtype() == DataType::Int32 && neighbors.ndim() == 2);
        nvtxRangePush("blob_seeding.sweep");
        const size_t count = peaks.shape()[0];
        const cudaStream_t stream = peaks.stream();
        auto out = Tensor::empty({count, static_cast<size_t>(SweepFieldCount)}, Device::CUDA, DataType::Float32);
        out.set_stream(stream);
        neighbors.sync_to_stream(stream);
        if (count > 0) {
            sweep_kernel<<<static_cast<unsigned int>(count), BLOCK_SIZE, 0, stream>>>(
                peaks.ptr<float>(), views, neighbors.ptr<int>(), static_cast<int>(neighbors.shape()[1]), out.ptr<float>());
            LFS_CUDA_LAUNCH_CHECK(stream, "training.blob_seeding.sweep");
        }
        nvtxRangePop();
        return out;
    }

} // namespace lfs::training::kernels::blob_seeding
