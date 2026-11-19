// SPDX-License-Identifier: GPL-3.0-or-later
#include "../components/calibration_correction.hpp"
#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"
#include "lfs/kernels/ssim.cuh"
#include "normal_loss.hpp"
#include <algorithm>

namespace lfs::training {
    using namespace core;
    namespace {
        using Tile = CalibrationCorrection::Tile;
        constexpr int threads = 128, chunk = 512;
        constexpr int partial_size = 4 * 5 + 1; // Four corner systems and an exact residual-presence bit.
        constexpr float epsilon = .03f;
        // One solver for training and the scoring alignment; only the share of each step they take differs.
        struct Solver {
            float smoothness, prior, damping, max_pixels;
            float reach = 0; // Largest displacement as a share of the image's short side; zero is unlimited.
        };
        // A photo the scene cannot explain (occluded, defocused) must not drift without limit.
        constexpr Solver constants{.05f, .01f, .1f, 1.5f, .05f};

        struct Location {
            int cell;
            float u, v, x, y;
            __device__ float weight(int k) const {
                return (k & 1 ? u : 1 - u) * (k & 2 ? v : 1 - v);
            }
        };
        __device__ Location locate(const Tile t, int x, int y, const float2* row) {
            const int side = t.side, cells = side - 1;
            // An exact cell boundary must agree with the integer partition used by the reduction.
            const float gx = __fdiv_rn((x + .5f) * cells, float(t.width));
            const float gy = __fdiv_rn((y + .5f) * cells, float(t.height));
            const int i = min(int(gx), cells - 1), j = min(int(gy), cells - 1);
            Location p{i + j * side, gx - i, gy - j, float(x), float(y)};
            float dx = 0, dy = 0;
            for (int k = 0; k < 4; ++k) {
                const auto d = row[p.cell + (k & 1) + (k >> 1) * side];
                dx += p.weight(k) * d.x;
                dy += p.weight(k) * d.y;
            }
            p.x += dx * t.sx;
            p.y += dy * t.sy;
            return p;
        }
        // A panorama continues across its left and right edge; rows, and the columns of other images, end there.
        __device__ int column(Tile t, int x) {
            return t.wrap ? (x % t.width + t.width) % t.width : min(max(x, 0), t.width - 1);
        }
        __device__ int line(Tile t, int y) {
            return min(max(y, 0), t.height - 1);
        }
        // The pixel left of and above a sample inside the photograph and the fractions towards the next ones.
        struct Source {
            int x, y;
            float u, v;
        };
        __device__ bool source(Location p, Tile t, Source& s) {
            if (!(p.y >= 0 && p.y <= t.height - 1))
                return false;
            float x = p.x;
            if (t.wrap) {
                if (!isfinite(x))
                    return false;
                x -= floorf(x / t.width) * t.width;
                if (x >= t.width)
                    x -= t.width;
            } else if (!(x >= 0 && x <= t.width - 1)) {
                return false;
            }
            s = {int(x), int(p.y), 0, 0};
            s.u = x - s.x;
            s.v = p.y - s.y;
            return true;
        }
        __device__ float cubic(float a, float b, float c, float d, float f) {
            return b + .5f * f * ((c - a) + f * ((2 * a - 5 * b + 4 * c - d) + f * (3 * (b - c) + d - a)));
        }
        struct Maps {
            const void* mask;
            void* out_mask;
            bool mask_float;
            const float* structure;
            float* out_structure;
            const float* edge;
            float* out_edge;
            const float *alpha, *background, *stable_background;
            bool background_image;
            float* stable;
        };
        __device__ float linear(const float* src, Tile t, Source s) {
            const int x1 = column(t, s.x + 1), y1 = line(t, s.y + 1);
            const float top = (1 - s.u) * src[s.y * t.width + s.x] + s.u * src[s.y * t.width + x1];
            const float bottom = (1 - s.u) * src[y1 * t.width + s.x] + s.u * src[y1 * t.width + x1];
            return (1 - s.v) * top + s.v * bottom;
        }
        // Catmull-Rom through the 4 x 4 taps, clamped to the range of the 2 x 2 cell around the sample: past a hard
        // edge the spline overshoots to values that no render can match.
        template <class Read>
        __device__ float sample_cubic(Source s, Tile t, Read read) {
            if (s.u == 0 && s.v == 0)
                return read(s.y * t.width + s.x);
            float taps[4][4], lines[4];
            for (int j = 0; j < 4; ++j) {
                const int row = line(t, s.y + j - 1) * t.width;
                for (int k = 0; k < 4; ++k)
                    taps[j][k] = read(row + column(t, s.x + k - 1));
                lines[j] = cubic(taps[j][0], taps[j][1], taps[j][2], taps[j][3], s.u);
            }
            const float value = cubic(lines[0], lines[1], lines[2], lines[3], s.v);
            const float low = fminf(fminf(taps[1][1], taps[1][2]), fminf(taps[2][1], taps[2][2]));
            const float high = fmaxf(fmaxf(taps[1][1], taps[1][2]), fmaxf(taps[2][1], taps[2][2]));
            return fminf(fmaxf(value, low), high);
        }
        template <class Pixel>
        __global__ void sample(Tile t, const Pixel* src, const float* render, const float2* row, float* output, Maps maps) {
            const int n = t.w * t.h, a = blockIdx.x * blockDim.x + threadIdx.x;
            if (a >= n)
                return;
            const auto p = locate(t, a % t.w + t.x, a / t.w + t.y, row);
            Source s{};
            const bool valid = source(p, t, s);
            const int pixels = t.width * t.height, dest = (a / t.w + t.y) * t.width + a % t.w + t.x;
            constexpr float scale = sizeof(Pixel) == 1 ? 1.f / 255.f : 1.f;
            for (int c = 0; c < 3; ++c) {
                const Pixel* plane = src + c * pixels;
                // Each tap is composited over the destination's background; the tap weights sum to one, so the
                // background enters once and unshifted.
                const auto over = [&](float background) {
                    return sample_cubic(s, t, [&](int i) {
                        const float alpha = fminf(fmaxf(maps.alpha[i], 0.f), 1.f);
                        return float(plane[i]) * scale * alpha + background * (1 - alpha);
                    });
                };
                float value = render[c * n + a];
                if (valid && maps.alpha) {
                    value = over(maps.background_image ? maps.background[c * pixels + dest] : maps.background[c]);
                } else if (valid) {
                    value = sample_cubic(s, t, [&](int i) { return float(plane[i]); });
                    if constexpr (sizeof(Pixel) == 1)
                        value *= 1.f / 255.f;
                }
                if (maps.stable)
                    maps.stable[c * pixels + dest] = valid ? over(maps.stable_background[c]) : value;
                output[c * n + a] = value;
            }
            if (maps.mask) {
                const int q = valid ? int(p.y + .5f) * t.width + column(t, int(s.x + s.u + .5f)) : 0;
                if (maps.mask_float)
                    static_cast<float*>(maps.out_mask)[a] = valid ? static_cast<const float*>(maps.mask)[q] : 0;
                else
                    static_cast<uint8_t*>(maps.out_mask)[a] = valid ? static_cast<const uint8_t*>(maps.mask)[q] : 0;
            }
            if (maps.structure)
                maps.out_structure[a] = valid ? linear(maps.structure, t, s) : 0;
            if (maps.edge)
                maps.out_edge[a] = valid ? linear(maps.edge, t, s) : 0;
        }

        __global__ void clear_outside_tile(Tile t, const float2* row, float* grad) {
            const int n = t.w * t.h, a = blockIdx.x * blockDim.x + threadIdx.x;
            Source s{};
            if (a >= n || source(locate(t, a % t.w + t.x, a / t.w + t.y, row), t, s))
                return;
            for (int c = 0; c < 3; ++c)
                grad[c * n + a] = 0;
        }

        // The prior's own grid is a tile covering the whole photograph at the prior's resolution.
        __global__ void follow_prior(Tile t, const float2* row, const float* src, float* output, int channels) {
            const int n = t.w * t.h, a = blockIdx.x * blockDim.x + threadIdx.x;
            if (a >= n)
                return;
            Source s{};
            const bool inside = source(locate(t, a % t.w, a / t.w, row), t, s);
            if (inside && s.u == 0 && s.v == 0) {
                for (int c = 0; c < channels; ++c)
                    output[c * n + a] = src[c * n + s.y * t.w + s.x];
                return;
            }
            float sum[3]{}, total = 0;
            for (int k = 0; inside && k < 4; ++k) {
                const float w = (k & 1 ? s.u : 1 - s.u) * (k & 2 ? s.v : 1 - s.v);
                const int i = line(t, s.y + (k >> 1)) * t.w + column(t, s.x + (k & 1));
                float tap[3]{}, length = 0;
                bool finite = true;
                for (int c = 0; c < channels; ++c) {
                    tap[c] = src[c * n + i];
                    finite &= isfinite(tap[c]);
                    length += tap[c] * tap[c];
                }
                const bool valid = finite && (channels == 1 ? tap[0] > 0 : sqrtf(length) >= kernels::kNormalLossMinPriorNorm);
                if (!(w > 0 && valid))
                    continue;
                for (int c = 0; c < channels; ++c)
                    sum[c] += w * tap[c];
                total += w;
            }
            float length = 0;
            for (int c = 0; c < channels; ++c) {
                sum[c] = total > 0 ? sum[c] / total : 0;
                length += sum[c] * sum[c];
            }
            const float norm = channels == 3 && length > 0 ? rsqrtf(length) : 1.f;
            for (int c = 0; c < channels; ++c)
                output[c * n + a] = sum[c] * norm;
        }

        // Each block owns four corners of a single cell. Only the update gathers across cells.
        __global__ void accumulate_tile(Tile t, const float2* row, const float* image, const float* rendered,
                                        const float* grad, const void* weight, bool float_weight, float values,
                                        int margin, float* partials, int chunks, bool first) {
            const int cells = t.side - 1;
            const int cell = blockIdx.y, col = cell % cells, r = cell / cells;
            const int x0 = (2 * col * t.width + cells - 1) / (2 * cells), x1 = (2 * (col + 1) * t.width + cells - 1) / (2 * cells);
            const int y0 = (2 * r * t.height + cells - 1) / (2 * cells), y1 = (2 * (r + 1) * t.height + cells - 1) / (2 * cells);
            // The loss's valid padding leaves the pixels this close to the tile's edge without a term of their own.
            const int mx = t.w > 2 * margin ? margin : 0, my = t.h > 2 * margin ? margin : 0;
            // A tile spanning a panorama's full width continues across the seam for the central differences.
            const bool seam = t.wrap && t.w == t.width;
            float sums[4][5]{};
            bool differs = false;
            for (int j = threadIdx.x; j < chunk; j += threads) {
                const int idx = chunk * blockIdx.x + j;
                if (idx >= (x1 - x0) * (y1 - y0))
                    break;
                const int x = x0 + idx % (x1 - x0), y = y0 + idx / (x1 - x0);
                const int xx = x - t.x, yy = y - t.y;
                if (xx < 0 || yy < 0 || xx >= t.w || yy >= t.h)
                    continue;
                const int a = yy * t.w + xx;
                const auto p = locate(t, x, y, row);
                Source s{};
                if (!source(p, t, s))
                    continue;
                if ((!seam && (xx == 0 || xx == t.w - 1)) || yy == 0 || yy == t.h - 1)
                    continue;
                float w = 1;
                if (weight) {
                    w = float_weight ? static_cast<const float*>(weight)[a] : float(static_cast<const uint8_t*>(weight)[a] > 0);
                    if (!(w > 0))
                        continue;
                }
                const int left = xx == 0 ? t.w - 1 : -1, right = xx == t.w - 1 ? 1 - t.w : 1;
                float q[5]{}, ss = 0;
                for (int c = 0; c < 3; ++c) {
                    const int i = c * t.w * t.h + a;
                    const float dx = .5f * t.sx * (image[i + right] - image[i + left]);
                    const float dy = .5f * t.sy * (image[i + t.w] - image[i - t.w]);
                    const float residual = image[i] - rendered[i];
                    ss += residual * residual;
                    const float g = grad ? -epsilon * values * grad[i] : residual;
                    q[0] += dx * g;
                    q[1] += dy * g;
                    q[2] += dx * dx;
                    q[3] += dx * dy;
                    q[4] += dy * dy;
                }
                if (!isfinite(ss))
                    continue;
                const float rho = epsilon * rsqrtf(ss / 3 + epsilon * epsilon);
                // Weighted least squares: a loss gradient already carries the pixel's weight, the curvature needs it too.
                const bool counted = xx >= mx && yy >= my && xx < t.w - mx && yy < t.h - my;
                const float curvature = counted ? rho * w : 0.f;
                for (int f = 2; f < 5; ++f)
                    q[f] *= curvature;
                if (!grad)
                    for (int f = 0; f < 2; ++f)
                        q[f] *= rho * w;
                if (!(isfinite(q[0]) && isfinite(q[1]) && isfinite(q[2]) && isfinite(q[3]) && isfinite(q[4])))
                    continue;
                differs |= ss > 0;
                for (int k = 0; k < 4; ++k)
                    for (int f = 0; f < 5; ++f)
                        sums[k][f] += p.weight(k) * q[f];
            }
            __shared__ float warps[threads / 32][20];
            for (int k = 0; k < 4; ++k)
                for (int f = 0; f < 5; ++f) {
                    float v = sums[k][f];
                    for (int offset = 16; offset; offset /= 2)
                        v += __shfl_down_sync(0xffffffff, v, offset);
                    if (threadIdx.x % 32 == 0)
                        warps[threadIdx.x / 32][k * 5 + f] = v;
                }
            const bool any_difference = __syncthreads_or(differs);
            auto* block = partials + (cell * chunks + blockIdx.x) * partial_size;
            if (threadIdx.x == 20)
                block[20] = float(any_difference || (!first && block[20]));
            if (threadIdx.x < 20) {
                float sum = 0;
                for (int w = 0; w < threads / 32; ++w)
                    sum += warps[w][threadIdx.x];
                float& dst = block[threadIdx.x];
                dst = first ? sum : dst + sum;
            }
        }
        __global__ void update_row(float2* row, const float* partials, int chunks, Tile t, Solver solver, float step) {
            const int side = t.side, cells = side - 1, points = side * side;
            const int i = threadIdx.x, x = i % side, y = i / side;
            extern __shared__ float2 shared[];
            float2* old = shared;
            float* trace = reinterpret_cast<float*>(old + points);
            float q[5]{};
            bool differs = false;
            for (int k = 0; k < 4; ++k) {
                int cx = x - (k & 1);
                const int cy = y - (k >> 1);
                // A panorama's first and last control column are one longitude: both gather the cells on either side.
                if (t.wrap && cx < 0)
                    cx = cells - 1;
                if (t.wrap && cx == cells)
                    cx = 0;
                if (cx < 0 || cx >= cells || cy < 0 || cy >= cells)
                    continue;
                for (int j = 0; j < chunks; ++j) {
                    const auto* block = partials + ((cy * cells + cx) * chunks + j) * partial_size;
                    differs |= block[20] != 0;
                    for (int f = 0; f < 5; ++f)
                        q[f] += block[k * 5 + f];
                }
            }
            old[i] = row[i];
            trace[i] = q[2] + q[4];
            if (!__syncthreads_or(differs)) {
                // Equal images can leave roundoff in a windowed loss derivative.
                q[0] = q[1] = 0;
            }
            float mean = 0;
            for (int j = 0; j < points; ++j)
                mean += trace[j];
            mean /= 2 * points;
            if (!(mean > 0 && isfinite(mean)))
                return;
            float2 lap{};
            int degree = 0;
            // A panorama's seam column has the column across the seam as its outer neighbour.
            const int seam = t.wrap ? cells - 1 : 0;
            const int offsets[4] = {x > 0 ? -1 : seam, x < cells ? 1 : -seam, y > 0 ? -side : 0, y < cells ? side : 0};
            for (int offset : offsets)
                if (offset) {
                    lap.x += old[i].x - old[i + offset].x;
                    lap.y += old[i].y - old[i + offset].y;
                    ++degree;
                }
            const float diagonal = mean * (2 * degree * solver.smoothness + solver.prior + solver.damping);
            const float a = q[2] + diagonal, b = q[3], c = q[4] + diagonal, det = a * c - b * b;
            const float gx = q[0] + mean * (solver.smoothness * lap.x + solver.prior * old[i].x);
            const float gy = q[1] + mean * (solver.smoothness * lap.y + solver.prior * old[i].y);
            if (!(det > 0))
                return;
            const float dx = (c * gx - b * gy) / det, dy = (a * gy - b * gx) / det;
            const float length = hypotf(dx * t.sx, dy * t.sy);
            if (isfinite(length)) {
                const float scale = step * (length > solver.max_pixels ? solver.max_pixels / length : 1.f);
                float2 moved = make_float2(old[i].x - scale * dx, old[i].y - scale * dy);
                const float reach = hypotf(moved.x * t.sx, moved.y * t.sy), limit = solver.reach * min(t.width, t.height);
                if (limit > 0 && reach > limit)
                    moved = make_float2(moved.x * (limit / reach), moved.y * (limit / reach));
                row[i] = moved;
            }
        }
        void reserve(Tensor& buffer, TensorShape shape, DataType dtype = DataType::Float32) {
            if (!buffer.is_valid() || buffer.shape() != shape || buffer.dtype() != dtype)
                buffer = Tensor::empty(shape, Device::CUDA, dtype);
            if (buffer.stream() != getCurrentCUDAStream())
                buffer.set_stream(getCurrentCUDAStream());
        }
        int chunks_for(Tile t) {
            const int cells = t.side - 1;
            return (((t.width + cells - 1) / cells) * ((t.height + cells - 1) / cells) + chunk - 1) / chunk;
        }
    } // namespace

    Tensor CalibrationCorrection::warp(int slot, float fx, float fy, int x, int y, const Tensor& photo, const Tensor& render,
                                       Tensor& mask, Tensor& structure, Tensor& edge, bool first_tile, const Composite* composite,
                                       bool wrap) {
        return sample_image(slot, fx * .001f, fy * .001f, x, y, photo, render, mask, structure, edge, first_tile, composite, wrap);
    }

    Tensor CalibrationCorrection::sample_image(int slot, float sx, float sy, int x, int y, const Tensor& photo, const Tensor& render,
                                               Tensor& mask, Tensor& structure, Tensor& edge, bool first_tile,
                                               const Composite* composite, bool wrap) {
        const auto stream = getCurrentCUDAStream();
        const int side = field_.shape()[1], points = side * side;
        const auto w = render.shape()[2], h = render.shape()[1];
        tile_ = {int(photo.shape()[2]), int(photo.shape()[1]), x, y, int(w), int(h), side, sx, sy, wrap};
        slot_ = slot;
        first_tile_ = first_tile;
        LFS_ASSERT(slot >= 0 && size_t(slot) < field_.shape()[0] && x >= 0 && y >= 0 &&
                   x + w <= photo.shape()[2] && y + h <= photo.shape()[1] && photo.is_contiguous() && render.is_contiguous());
        if (field_.stream() != stream)
            field_.set_stream(stream);
        photo.sync_to_stream(stream);
        render.sync_to_stream(stream);
        reserve(image_, {3, h, w});
        Maps maps{};
        if (composite) {
            LFS_ASSERT(composite->alpha->is_contiguous() && composite->alpha->numel() * 3 == photo.numel() &&
                       (composite->background->numel() == 3 || composite->background->numel() == photo.numel()));
            composite->alpha->sync_to_stream(stream);
            composite->background->sync_to_stream(stream);
            maps.alpha = composite->alpha->ptr<float>();
            maps.background = composite->background->ptr<float>();
            maps.background_image = composite->background->numel() != 3;
            if (composite->stable) {
                composite->stable_background->sync_to_stream(stream);
                if (composite->stable->stream() != stream)
                    composite->stable->set_stream(stream);
                maps.stable_background = composite->stable_background->ptr<float>();
                maps.stable = composite->stable->ptr<float>();
            }
        }
        if (mask.is_valid()) {
            mask.sync_to_stream(stream);
            reserve(mask_, {h, w}, mask.dtype());
            maps.mask = mask.data_ptr();
            maps.out_mask = mask_.data_ptr();
            maps.mask_float = mask.dtype() == DataType::Float32;
        }
        if (structure.is_valid()) {
            structure.sync_to_stream(stream);
            reserve(structure_, {h, w});
            maps.structure = structure.ptr<float>();
            maps.out_structure = structure_.ptr<float>();
        }
        if (edge.is_valid()) {
            edge.sync_to_stream(stream);
            reserve(edge_, {h, w});
            maps.edge = edge.ptr<float>();
            maps.out_edge = edge_.ptr<float>();
        }
        const float2* row = reinterpret_cast<const float2*>(field_.ptr<float>()) + slot * points;
        auto launch = [&](auto src) {
            sample<<<(w * h + 255) / 256, 256, 0, stream>>>(tile_, src, render.ptr<float>(), row, image_.ptr<float>(), maps);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.calibration.sample");
        };
        if (photo.dtype() == DataType::UInt8)
            launch(photo.ptr<uint8_t>());
        else
            launch(photo.ptr<float>());
        if (maps.mask)
            mask = mask_;
        if (maps.structure)
            structure = structure_;
        if (maps.edge)
            edge = edge_;
        return image_;
    }

    void CalibrationCorrection::accumulate(const Tensor& render, const Tensor& gradient, const Tensor& weight, float values,
                                           int margin, const kernels::DecoupledRawGradient* raw) {
        const auto stream = getCurrentCUDAStream();
        const int side = tile_.side, cells = side - 1, points = side * side;
        const int chunks = chunks_for(tile_);
        const Tensor* full = &gradient;
        if (raw && gradient.is_valid()) {
            // The target enters both parts of the decoupled loss, so its gradient is their sum.
            reserve(gradient_, gradient.shape());
            gradient_.copy_from(gradient);
            kernels::accumulate_decoupled_raw_gradient(*raw, gradient_);
            full = &gradient_;
        }
        reserve(partials_, {size_t(cells * cells), size_t(chunks), partial_size});
        const float2* row = reinterpret_cast<const float2*>(field_.ptr<float>()) + slot_ * points;
        accumulate_tile<<<dim3(chunks, cells * cells), threads, 0, stream>>>(tile_, row, image_.ptr<float>(), render.ptr<float>(),
                                                                             full->is_valid() ? full->ptr<float>() : nullptr, weight.is_valid() ? weight.data_ptr() : nullptr, weight.is_valid() && weight.dtype() == DataType::Float32,
                                                                             values, margin, partials_.ptr<float>(), chunks, first_tile_);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.calibration.accumulate");
    }

    void CalibrationCorrection::clear_outside(Tensor& gradient) const {
        const auto stream = getCurrentCUDAStream();
        const int points = tile_.side * tile_.side, n = tile_.w * tile_.h;
        LFS_ASSERT(gradient.is_contiguous() && gradient.dtype() == DataType::Float32 && gradient.numel() == size_t(3 * n));
        gradient.sync_to_stream(stream);
        const float2* row = reinterpret_cast<const float2*>(field_.ptr<float>()) + slot_ * points;
        clear_outside_tile<<<(n + 255) / 256, 256, 0, stream>>>(tile_, row, gradient.ptr<float>());
        LFS_CUDA_LAUNCH_CHECK(stream, "training.calibration.clear_outside");
    }

    Tensor CalibrationCorrection::follow(const Tensor& prior, bool normal) const {
        const auto stream = getCurrentCUDAStream();
        const int channels = normal ? 3 : 1, points = tile_.side * tile_.side;
        const auto& shape = prior.shape();
        LFS_ASSERT(prior.device() == Device::CUDA && prior.dtype() == DataType::Float32 && prior.is_contiguous() &&
                   shape.rank() >= 2 && prior.numel() == size_t(channels) * shape[shape.rank() - 1] * shape[shape.rank() - 2]);
        const int w = int(shape[shape.rank() - 1]), h = int(shape[shape.rank() - 2]);
        // The field is read at the prior's own pixel centres, its displacement scaled to the prior's pixels.
        const Tile t{w, h, 0, 0, w, h, tile_.side, tile_.sx * w / tile_.width, tile_.sy * h / tile_.height, tile_.wrap};
        prior.sync_to_stream(stream);
        auto output = Tensor::empty(shape, Device::CUDA);
        output.set_stream(stream);
        const float2* row = reinterpret_cast<const float2*>(field_.ptr<float>()) + slot_ * points;
        follow_prior<<<(w * h + 255) / 256, 256, 0, stream>>>(t, row, prior.ptr<float>(), output.ptr<float>(), channels);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.calibration.follow_prior");
        return output;
    }

    void CalibrationCorrection::update() {
        const auto stream = getCurrentCUDAStream();
        const int points = tile_.side * tile_.side;
        auto row = reinterpret_cast<float2*>(field_.ptr<float>()) + slot_ * points;
        update_row<<<1, points, 3 * points * sizeof(float), stream>>>(row, partials_.ptr<float>(), chunks_for(tile_), tile_, constants,
                                                                      step_);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.calibration.update");
    }
} // namespace lfs::training
