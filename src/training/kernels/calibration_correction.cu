// SPDX-License-Identifier: GPL-3.0-or-later
#include "../components/calibration_correction.hpp"
#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"
#include <algorithm>

namespace lfs::training {
    using namespace core;
    namespace {
        using Tile = CalibrationCorrection::Tile;
        constexpr int threads = 128, chunk = 512;
        constexpr int partial_size = 4 * 5 + 1; // Four corner systems and an exact residual-presence bit.
        constexpr float epsilon = .03f;
        struct Solver {
            float smoothness, prior, damping, max_pixels, fraction;
            float reach = 0; // Largest displacement as a share of the image's short side; zero is unlimited.
        };
        constexpr Solver training{.05f, .01f, .1f, 1.5f, .5f}, scoring{.02f, 0.f, .1f, 1.5f, 1.f, .05f};

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
        __device__ bool inside(Location p, Tile t) {
            return p.x >= 0 && p.y >= 0 && p.x <= t.width - 1 && p.y <= t.height - 1;
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
            const void* foreground;
            const float *alpha, *background;
            float* stable;
            bool byte_foreground;
        };
        __device__ float linear(const float* src, Tile t, int x, int y, float u, float v) {
            const int x1 = min(x + 1, t.width - 1), y1 = min(y + 1, t.height - 1);
            const float top = (1 - u) * src[y * t.width + x] + u * src[y * t.width + x1];
            const float bottom = (1 - u) * src[y1 * t.width + x] + u * src[y1 * t.width + x1];
            return (1 - v) * top + v * bottom;
        }
        template <class Read>
        __device__ float sample_cubic(int x, int y, float u, float v, Tile t, Read read) {
            if (u == 0 && v == 0)
                return read(y * t.width + x);
            float lines[4];
            for (int j = 0; j < 4; ++j) {
                const int line = min(max(y + j - 1, 0), t.height - 1) * t.width;
                lines[j] = cubic(read(line + max(0, x - 1)), read(line + x),
                                 read(line + min(x + 1, t.width - 1)), read(line + min(x + 2, t.width - 1)), u);
            }
            return cubic(lines[0], lines[1], lines[2], lines[3], v);
        }
        template <class Pixel>
        __global__ void sample(Tile t, const Pixel* src, const float* render, const float2* row, float* output, Maps maps) {
            const int n = t.w * t.h, a = blockIdx.x * blockDim.x + threadIdx.x;
            if (a >= n)
                return;
            const auto p = locate(t, a % t.w + t.x, a / t.w + t.y, row);
            const bool valid = inside(p, t);
            const int x = valid ? int(p.x) : 0, y = valid ? int(p.y) : 0;
            const float u = p.x - x, v = p.y - y;
            for (int c = 0; c < 3; ++c) {
                float value = render[c * n + a];
                if (valid) {
                    const Pixel* plane = src + c * t.width * t.height;
                    value = sample_cubic(x, y, u, v, t, [&](int i) { return float(plane[i]); });
                    if constexpr (sizeof(Pixel) == 1)
                        value *= 1.f / 255.f;
                }
                if (maps.stable) {
                    const int channel = c * t.width * t.height;
                    const int dest = (a / t.w + t.y) * t.width + a % t.w + t.x;
                    maps.stable[channel + dest] = valid ? sample_cubic(x, y, u, v, t, [&](int i) {
                        const float pixel = maps.byte_foreground ? static_cast<const uint8_t*>(maps.foreground)[channel + i] * (1.f / 255.f)
                                                                 : static_cast<const float*>(maps.foreground)[channel + i];
                        const float alpha = fminf(fmaxf(maps.alpha[i], 0.f), 1.f);
                        return pixel * alpha + maps.background[c] * (1 - alpha);
                    })
                                                        : value;
                }
                output[c * n + a] = value;
            }
            if (maps.mask) {
                const int q = valid ? int(p.y + .5f) * t.width + int(p.x + .5f) : 0;
                if (maps.mask_float)
                    static_cast<float*>(maps.out_mask)[a] = valid ? static_cast<const float*>(maps.mask)[q] : 0;
                else
                    static_cast<uint8_t*>(maps.out_mask)[a] = valid ? static_cast<const uint8_t*>(maps.mask)[q] : 0;
            }
            if (maps.structure)
                maps.out_structure[a] = valid ? linear(maps.structure, t, x, y, u, v) : 0;
            if (maps.edge)
                maps.out_edge[a] = valid ? linear(maps.edge, t, x, y, u, v) : 0;
        }

        // Each block owns four corners of a single cell. Only the update gathers across cells.
        __global__ void accumulate_tile(Tile t, const float2* row, const float* image, const float* rendered,
                                        float* grad, const void* weight, bool float_weight, float values,
                                        float* partials, int chunks, bool first) {
            const int cells = t.side - 1;
            const int cell = blockIdx.y, col = cell % cells, r = cell / cells;
            const int x0 = (2 * col * t.width + cells - 1) / (2 * cells), x1 = (2 * (col + 1) * t.width + cells - 1) / (2 * cells);
            const int y0 = (2 * r * t.height + cells - 1) / (2 * cells), y1 = (2 * (r + 1) * t.height + cells - 1) / (2 * cells);
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
                if (!inside(p, t)) {
                    // SSIM neighbours can send gradients to an otherwise identical border sample.
                    if (grad)
                        for (int c = 0; c < 3; ++c)
                            grad[c * t.w * t.h + a] = 0;
                    continue;
                }
                if (xx == 0 || yy == 0 || xx == t.w - 1 || yy == t.h - 1)
                    continue;
                if (weight && !(float_weight ? static_cast<const float*>(weight)[a] > 0 : static_cast<const uint8_t*>(weight)[a] > 0))
                    continue;
                float q[5]{}, ss = 0;
                for (int c = 0; c < 3; ++c) {
                    const int i = c * t.w * t.h + a;
                    const float dx = .5f * t.sx * (image[i + 1] - image[i - 1]);
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
                for (int f = grad ? 2 : 0; f < 5; ++f)
                    q[f] *= rho;
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
        __global__ void update_row(float2* row, const float* partials, int chunks, Tile t, Solver solver) {
            const int side = t.side, cells = side - 1, points = side * side;
            const int i = threadIdx.x, x = i % side, y = i / side;
            extern __shared__ float2 shared[];
            float2* old = shared;
            float* trace = reinterpret_cast<float*>(old + points);
            float q[5]{};
            bool differs = false;
            for (int k = 0; k < 4; ++k) {
                const int cx = x - (k & 1), cy = y - (k >> 1);
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
            const int offsets[4] = {x > 0 ? -1 : 0, x < cells ? 1 : 0, y > 0 ? -side : 0, y < cells ? side : 0};
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
                const float scale = solver.fraction * (length > solver.max_pixels ? solver.max_pixels / length : 1.f);
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
                                       Tensor& mask, Tensor& structure, Tensor& edge, bool first_tile, const Composite* stable) {
        return sample_image(slot, fx * .001f, fy * .001f, x, y, photo, render, mask, structure, edge, first_tile, stable);
    }

    Tensor CalibrationCorrection::sample_image(int slot, float sx, float sy, int x, int y, const Tensor& photo, const Tensor& render,
                                               Tensor& mask, Tensor& structure, Tensor& edge, bool first_tile, const Composite* stable) {
        const auto stream = getCurrentCUDAStream();
        const int side = field_.shape()[1], points = side * side;
        const auto w = render.shape()[2], h = render.shape()[1];
        tile_ = {int(photo.shape()[2]), int(photo.shape()[1]), x, y, int(w), int(h), side, sx, sy};
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
        if (stable) {
            stable->foreground->sync_to_stream(stream);
            stable->alpha->sync_to_stream(stream);
            stable->background->sync_to_stream(stream);
            if (stable->output->stream() != stream)
                stable->output->set_stream(stream);
            maps.foreground = stable->foreground->data_ptr();
            maps.alpha = stable->alpha->ptr<float>();
            maps.background = stable->background->ptr<float>();
            maps.stable = stable->output->ptr<float>();
            maps.byte_foreground = stable->foreground->dtype() == DataType::UInt8;
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

    void CalibrationCorrection::accumulate(const Tensor& render, Tensor& gradient, const Tensor& weight, float values) {
        const auto stream = getCurrentCUDAStream();
        const int side = tile_.side, cells = side - 1, points = side * side;
        const int chunks = chunks_for(tile_);
        reserve(partials_, {size_t(cells * cells), size_t(chunks), partial_size});
        const float2* row = reinterpret_cast<const float2*>(field_.ptr<float>()) + slot_ * points;
        accumulate_tile<<<dim3(chunks, cells * cells), threads, 0, stream>>>(tile_, row, image_.ptr<float>(), render.ptr<float>(),
                                                                             gradient.is_valid() ? gradient.ptr<float>() : nullptr, weight.is_valid() ? weight.data_ptr() : nullptr, weight.is_valid() && weight.dtype() == DataType::Float32,
                                                                             values, partials_.ptr<float>(), chunks, first_tile_);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.calibration.accumulate");
    }
    void CalibrationCorrection::update() {
        const auto stream = getCurrentCUDAStream();
        const int points = tile_.side * tile_.side;
        auto row = reinterpret_cast<float2*>(field_.ptr<float>()) + slot_ * points;
        update_row<<<1, points, 3 * points * sizeof(float), stream>>>(row, partials_.ptr<float>(), chunks_for(tile_), tile_,
                                                                      tile_.side == grid ? training : scoring);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.calibration.update");
    }
} // namespace lfs::training
