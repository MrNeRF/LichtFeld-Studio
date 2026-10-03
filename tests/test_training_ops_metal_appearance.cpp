/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal and Vulkan Bilateral, PPISP, Controller, Lpips and SharedImage ops against CPU
// transliterations of the CUDA kernels, on fixture-sized inputs. Integer and
// byte outputs are exact; floats use tolerances; the bilateral and PPISP
// gradients are checked against central differences of the CPU forward.

#include "core/shared_image_ops.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_image.hpp"
#include "lfs/training/ops/registry.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace {
    // The backend of the running parameterized test.
    lfs::core::GpuBackend backend_under_test() { return testing::TestWithParam<lfs::core::GpuBackend>::GetParam(); }

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;
    namespace ops = lfs::gpu_ops;
    using Doubles = std::vector<double>;

    std::vector<float> pattern(const size_t count, const float scale, const int seed, const float offset = 0.f) {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) {
            const auto k = static_cast<int>((i * 7919u + static_cast<size_t>(seed) * 104729u) % 2003u);
            values[i] = offset + scale * (static_cast<float>(k) / 1001.0f - 1.0f);
        }
        return values;
    }

    Tensor gpu(const std::vector<float>& values, const TensorShape& shape) {
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    Tensor gpu_bytes(const std::vector<uint8_t>& values, const TensorShape& shape) {
        auto cpu = Tensor::empty(shape, Device::CPU, DataType::UInt8);
        std::memcpy(cpu.data_ptr(), values.data(), values.size());
        return cpu.gpu();
    }

    std::vector<float> host(const Tensor& tensor) {
        const auto cpu = (tensor.dtype() == DataType::Float16 ? tensor.to(DataType::Float32) : tensor).cpu().contiguous();
        const auto* data = cpu.ptr<float>();
        return {data, data + cpu.numel()};
    }

    std::vector<uint8_t> host_bytes(const Tensor& tensor) {
        const auto cpu = tensor.cpu().contiguous();
        const auto* data = static_cast<const uint8_t*>(cpu.data_ptr());
        return {data, data + cpu.bytes()};
    }

    Doubles widen(const std::vector<float>& values) { return {values.begin(), values.end()}; }

    float half_round(const float value) { return static_cast<float>(static_cast<_Float16>(value)); }

    void expect_close(const std::vector<float>& actual, const Doubles& expected, const double abs_tol, const double rel_tol,
                      const std::string& what, const std::vector<bool>& skip = {}) {
        ASSERT_EQ(actual.size(), expected.size()) << what;
        int reported = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!skip.empty() && skip[i])
                continue;
            const double limit = abs_tol + rel_tol * std::abs(expected[i]);
            if (!(std::abs(actual[i] - expected[i]) <= limit) && reported++ < 8)
                ADD_FAILURE() << std::format("{}[{}]: actual {} vs reference {} (limit {})", what, i, actual[i], expected[i], limit);
        }
    }

    // Central difference of a scalar function over each entry of `values`.
    Doubles finite_difference(Doubles values, const std::function<double(const Doubles&)>& loss, const double step = 1e-4) {
        Doubles gradient(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            const double keep = values[i];
            values[i] = keep + step;
            const double up = loss(values);
            values[i] = keep - step;
            const double down = loss(values);
            values[i] = keep;
            gradient[i] = (up - down) / (2 * step);
        }
        return gradient;
    }

    // ---- PPISP reference (ppisp_math.cuh) in double precision -------------------

    using V3 = std::array<double, 3>;
    using M3 = std::array<double, 9>;

    M3 mul(const M3& a, const M3& b) {
        M3 c{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                c[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        return c;
    }

    V3 mul(const M3& a, const V3& v) {
        return {a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
                a[6] * v[0] + a[7] * v[1] + a[8] * v[2]};
    }

    V3 cross(const V3& a, const V3& b) {
        return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    }

    double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

    constexpr float kZca[16] = {0.0480542f, -0.0043631f, -0.0043631f, 0.0481283f, 0.0580570f, -0.0179872f,
                                -0.0179872f, 0.0431061f, 0.0433336f, -0.0180537f, -0.0180537f, 0.0580500f,
                                0.0128369f, -0.0034654f, -0.0034654f, 0.0128158f};

    M3 homography(const double* c) {
        std::array<std::array<double, 2>, 4> d{};
        for (int k = 0; k < 4; ++k) {
            const float* z = kZca + k * 4;
            d[k] = {z[0] * c[2 * k] + z[1] * c[2 * k + 1], z[2] * c[2 * k] + z[3] * c[2 * k + 1]};
        }
        const V3 tb{d[0][0], d[0][1], 1}, tr{1 + d[1][0], d[1][1], 1}, tg{d[2][0], 1 + d[2][1], 1};
        const V3 gray{1.0 / 3.0 + d[3][0], 1.0 / 3.0 + d[3][1], 1};
        const M3 t{tb[0], tr[0], tg[0], tb[1], tr[1], tg[1], tb[2], tr[2], tg[2]};
        const M3 skew{0, -gray[2], gray[1], gray[2], 0, -gray[0], -gray[1], gray[0], 0};
        const M3 m = mul(skew, t);
        const V3 r0{m[0], m[1], m[2]}, r1{m[3], m[4], m[5]}, r2{m[6], m[7], m[8]};
        V3 lambda = cross(r0, r1);
        if (dot(lambda, lambda) < 1e-20) {
            lambda = cross(r0, r2);
            if (dot(lambda, lambda) < 1e-20)
                lambda = cross(r1, r2);
        }
        const M3 s_inv{-1, -1, 1, 1, 0, 0, 0, 1, 0};
        M3 h = mul(mul(t, M3{lambda[0], 0, 0, 0, lambda[1], 0, 0, 0, lambda[2]}), s_inv);
        if (std::abs(h[8]) > 1e-20) {
            const double inv = 1.0 / h[8];
            for (double& v : h)
                v *= inv;
        }
        return h;
    }

    V3 exposure(const V3& rgb, const double ev) {
        const double f = std::exp2(std::clamp(ev, -16.0, 16.0));
        return {rgb[0] * f, rgb[1] * f, rgb[2] * f};
    }

    V3 color_correct(const V3& rgb_in, const double* c) {
        const M3 h = homography(c);
        const V3 rgb{std::max(rgb_in[0], 0.0), std::max(rgb_in[1], 0.0), std::max(rgb_in[2], 0.0)};
        const double intensity = rgb[0] + rgb[1] + rgb[2];
        V3 rgi = mul(h, V3{rgb[0], rgb[1], intensity});
        const double norm = intensity / (std::max(rgi[2], 0.0) + 1e-5);
        for (double& v : rgi)
            v *= norm;
        return {rgi[0], rgi[1], rgi[2] - rgi[0] - rgi[1]};
    }

    V3 vignette(const V3& rgb, const double* v, const double px, const double py, const double rx, const double ry) {
        const double max_res = std::max(rx, ry);
        const double u = (px - rx * 0.5) / max_res, w = (py - ry * 0.5) / max_res;
        V3 out = rgb;
        for (int i = 0; i < 3; ++i) {
            const double* p = v + i * 5;
            const double dx = u - p[0], dy = w - p[1];
            const double r2 = dx * dx + dy * dy;
            out[i] *= std::clamp(1 + p[2] * r2 + p[3] * r2 * r2 + p[4] * r2 * r2 * r2, 0.0, 1.0);
        }
        return out;
    }

    double bounded_positive(const double raw, const double min_value) {
        const double v = std::min(32.0, raw);
        return min_value + std::max(v, 0.0) + std::log(1 + std::exp(-std::abs(v)));
    }

    V3 crf(const V3& rgb, const double* c) {
        V3 out{};
        for (int i = 0; i < 3; ++i) {
            const double* p = c + i * 4;
            const double toe = bounded_positive(p[0], 0.3), shoulder = bounded_positive(p[1], 0.3);
            const double gamma = bounded_positive(p[2], 0.1);
            const double center = std::clamp(1 / (1 + std::exp(-p[3])), 1e-4, 1 - 1e-4);
            const double lerp = (shoulder - toe) * center + toe;
            const double a = shoulder * center / lerp, b = 1 - a;
            const double x = std::clamp(rgb[i], 0.0, 1.0);
            const double y = x <= center ? a * std::pow(x / center, toe)
                                         : 1 - b * std::pow((1 - x) / (1 - center), shoulder);
            out[i] = std::pow(std::max(0.0, y), gamma);
        }
        return out;
    }

    struct PpispParams {
        Doubles exposure, vignetting, color, crf;
    };

    // Forward of one CHW image band; rows are offset by y_offset inside full_height.
    Doubles ppisp_forward(const PpispParams& p, const Doubles& rgb, const int height, const int width, const int y_offset,
                          const int full_height, const int camera, const int frame) {
        const int pixels = height * width;
        Doubles out(rgb.size());
        for (int i = 0; i < pixels; ++i) {
            V3 v{rgb[i], rgb[pixels + i], rgb[2 * pixels + i]};
            const double px = i % width + 0.5, py = y_offset + i / width + 0.5;
            if (frame != -1)
                v = exposure(v, p.exposure[frame]);
            if (camera != -1)
                v = vignette(v, p.vignetting.data() + camera * 15, px, py, width, full_height);
            if (frame != -1)
                v = color_correct(v, p.color.data() + frame * 8);
            if (camera != -1)
                v = crf(v, p.crf.data() + camera * 12);
            for (int c = 0; c < 3; ++c)
                out[c * pixels + i] = v[c];
        }
        return out;
    }

    double weighted(const Doubles& values, const std::vector<float>& weights) {
        double sum = 0;
        for (size_t i = 0; i < values.size(); ++i)
            sum += values[i] * weights[i];
        return sum;
    }

    // ---- Bilateral reference (bilateral_grid_forward.cu) ------------------------

    struct Grid {
        int C, L, H, W;
        Doubles values;
    };

    struct Cell {
        int x0, y0, z0, x1, y1, z1;
        double fx, fy, fz, z;
    };

    Cell bilateral_cell(const Grid& g, const int wi, const int hi, const int h, const int w, const V3& rgb) {
        const double x = w > 1 ? double(wi) / (w - 1) * (g.W - 1) : 0.0;
        const double y = h > 1 ? double(hi) / (h - 1) * (g.H - 1) : 0.0;
        const double z = std::clamp(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2], 0.0, 1.0) * (g.L - 1);
        Cell c{};
        c.x0 = static_cast<int>(std::floor(x));
        c.y0 = static_cast<int>(std::floor(y));
        const int z0 = static_cast<int>(std::floor(z));
        c.x1 = std::min(c.x0 + 1, g.W - 1);
        c.y1 = std::min(c.y0 + 1, g.H - 1);
        c.z0 = std::clamp(z0, 0, g.L - 1);
        c.z1 = std::clamp(z0 + 1, 0, g.L - 1);
        c.fx = x - c.x0;
        c.fy = y - c.y0;
        c.fz = z - c.z0;
        c.z = z;
        return c;
    }

    double bilateral_sample(const Grid& g, const Cell& c, const int ci, const Doubles& offset) {
        const auto at = [&](const int z, const int y, const int x) {
            return g.values[((ci * g.L + z) * g.H + y) * g.W + x];
        };
        const double c00 = at(c.z0, c.y0, c.x0) * (1 - c.fx) + at(c.z0, c.y0, c.x1) * c.fx;
        const double c01 = at(c.z0, c.y1, c.x0) * (1 - c.fx) + at(c.z0, c.y1, c.x1) * c.fx;
        const double c10 = at(c.z1, c.y0, c.x0) * (1 - c.fx) + at(c.z1, c.y0, c.x1) * c.fx;
        const double c11 = at(c.z1, c.y1, c.x0) * (1 - c.fx) + at(c.z1, c.y1, c.x1) * c.fx;
        const double c0 = c00 * (1 - c.fy) + c01 * c.fy, c1 = c10 * (1 - c.fy) + c11 * c.fy;
        return c0 * (1 - c.fz) + c1 * c.fz + offset[ci];
    }

    // HWC or CHW slice of one image.
    Doubles bilateral_forward(const Grid& g, const Doubles& rgb, const Doubles& offset, const int h, const int w,
                              const bool chw, const bool exposure_chroma) {
        const int hw = h * w;
        const auto index = [&](const int pixel, const int c) { return chw ? c * hw + pixel : pixel * 3 + c; };
        Doubles out(rgb.size());
        for (int pixel = 0; pixel < hw; ++pixel) {
            const V3 v{rgb[index(pixel, 0)], rgb[index(pixel, 1)], rgb[index(pixel, 2)]};
            const Cell c = bilateral_cell(g, pixel % w, pixel / w, h, w, v);
            V3 o{};
            if (exposure_chroma) {
                double s[9];
                for (int ci = 0; ci < 9; ++ci)
                    s[ci] = bilateral_sample(g, c, ci, offset);
                o = color_correct(exposure(v, s[0]), s + 1);
            } else {
                for (int ci = 0; ci < 12; ++ci)
                    o[ci / 4] += bilateral_sample(g, c, ci, offset) * (ci % 4 == 3 ? 1.0 : v[ci % 4]);
            }
            for (int k = 0; k < 3; ++k)
                out[index(pixel, k)] = o[k];
        }
        return out;
    }

    class PortableAppearance : public ::testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(backend_under_test()))
                GTEST_SKIP() << lfs::core::gpu_backend_name(GetParam()) << " device unavailable";
            scope_.emplace(backend_under_test());
        }

        static const lfs::training::TrainingOps& table() { return lfs::training::training_ops(backend_under_test()); }

    private:
        std::optional<lfs::core::GpuBackendScope> scope_;
    };

    void check_bilateral_slice(const bool chw, const bool exposure_chroma) {
        const auto& table = *lfs::training::training_ops(backend_under_test()).bilateral;
        const std::string what = std::format("bilateral {} {}", chw ? "chw" : "hwc", exposure_chroma ? "exposure-chroma" : "affine");
        constexpr int h = 5, w = 7;
        const int channels = exposure_chroma ? 9 : 12;
        Grid grid{channels, 2, 3, 4, {}};
        const size_t cells = static_cast<size_t>(channels) * 2 * 3 * 4;
        auto grid_values = pattern(cells, exposure_chroma ? 0.3f : 0.6f, 3, exposure_chroma ? 0.f : 0.2f);
        auto rgb_values = pattern(h * w * 3, 0.35f, 11, 0.5f);
        auto offset_values = pattern(channels, 0.1f, 17);
        auto grad_values = pattern(h * w * 3, 0.5f, 19);
        grid.values = widen(grid_values);
        const TensorShape image = chw ? TensorShape{3, h, w} : TensorShape{h, w, 3};
        const auto grid_t = gpu(grid_values, {1, static_cast<size_t>(channels), 2, 3, 4});
        const auto rgb_t = gpu(rgb_values, image);
        const auto offset_t = gpu(offset_values, {static_cast<size_t>(channels)});
        const ops::GridSliceParams params{chw ? ops::Layout::CHW : ops::Layout::HWC,
                                          exposure_chroma ? ops::GridTransform::ExposureChroma : ops::GridTransform::Affine, true};

        auto output = Tensor::zeros(image, Device::GPU);
        table.slice_forward(grid_t, rgb_t, offset_t, output, params);
        const Doubles rgb = widen(rgb_values), offset = widen(offset_values);
        expect_close(host(output), bilateral_forward(grid, rgb, offset, h, w, chw, exposure_chroma), 2e-5, 2e-5,
                     what + " forward");

        // grad_grid accumulates onto its initial ones.
        auto grad_grid = Tensor::ones(grid_t.shape(), Device::GPU);
        auto grad_rgb = Tensor::zeros(image, Device::GPU);
        table.slice_backward(grid_t, rgb_t, gpu(grad_values, image), offset_t, grad_grid, grad_rgb, params);

        const auto loss_of_grid = [&](const Doubles& values) {
            Grid g = grid;
            g.values = values;
            return weighted(bilateral_forward(g, rgb, offset, h, w, chw, exposure_chroma), grad_values);
        };
        Doubles expected_grid = finite_difference(grid.values, loss_of_grid);
        for (double& v : expected_grid)
            v += 1.0;
        expect_close(host(grad_grid), expected_grid, 2e-4, 1e-3, what + " grid gradient");

        const auto loss_of_rgb = [&](const Doubles& values) {
            return weighted(bilateral_forward(grid, values, offset, h, w, chw, exposure_chroma), grad_values);
        };
        // Skip pixels whose guidance sits on a grid plane, where the gradient jumps.
        std::vector<bool> skip(rgb.size(), false);
        for (int pixel = 0; pixel < h * w; ++pixel) {
            const auto at = [&](const int c) { return rgb[chw ? c * h * w + pixel : pixel * 3 + c]; };
            const double z = bilateral_cell(grid, pixel % w, pixel / w, h, w, {at(0), at(1), at(2)}).z;
            if (std::abs(z - std::round(z)) < 1e-3)
                for (int c = 0; c < 3; ++c)
                    skip[chw ? c * h * w + pixel : pixel * 3 + c] = true;
        }
        expect_close(host(grad_rgb), finite_difference(rgb, loss_of_rgb), 2e-4, 1e-3, what + " rgb gradient", skip);
    }

    TEST_P(PortableAppearance, BilateralSliceMatchesReferenceAndFiniteDifferences) {
        check_bilateral_slice(false, false);
        check_bilateral_slice(true, false);
        check_bilateral_slice(false, true);
        check_bilateral_slice(true, true);
    }

    TEST_P(PortableAppearance, BilateralRegularizerAndOptimizer) {
        const auto& table = *this->table().bilateral;
        constexpr int N = 2, C = 9, L = 3, H = 4, W = 5, norm_n = 7;
        const size_t count = static_cast<size_t>(N) * C * L * H * W;
        const auto grid_values = pattern(count, 1.f, 5);
        const auto grids = gpu(grid_values, {N, C, L, H, W});

        // TV loss and gradient (bilateral_grid_tv.cu).
        double loss = 0;
        Doubles gradient = widen(pattern(count, 1.f, 11));
        const double s = 2.0 * 0.25 / (C * norm_n);
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int l = 0; l < L; ++l) {
                    for (int y = 0; y < H; ++y) {
                        for (int x = 0; x < W; ++x) {
                            const size_t i = ((((static_cast<size_t>(n) * C + c) * L + l) * H + y) * W + x);
                            const double v = grid_values[i];
                            const auto term = [&](const bool has, const size_t j, const double norm) {
                                if (has) {
                                    const double d = v - grid_values[j];
                                    loss += d * d / norm / (C * norm_n);
                                }
                            };
                            const auto grad = [&](const bool has, const size_t j, const double norm) {
                                if (has)
                                    gradient[i] += (v - grid_values[j]) * s / norm;
                            };
                            term(x > 0, i - 1, L * H * (W - 1));
                            term(y > 0, i - W, L * (H - 1) * W);
                            term(l > 0, i - W * H, (L - 1) * H * W);
                            grad(x > 0, i - 1, L * H * (W - 1));
                            grad(x < W - 1, i + 1, L * H * (W - 1));
                            grad(y > 0, i - W, L * (H - 1) * W);
                            grad(y < H - 1, i + W, L * (H - 1) * W);
                            grad(l > 0, i - W * H, (L - 1) * H * W);
                            grad(l < L - 1, i + W * H, (L - 1) * H * W);
                        }
                    }
                }
            }
        }
        auto loss_t = Tensor::zeros({1}, Device::GPU);
        auto temp = Tensor::empty({2048}, Device::GPU);
        table.tv_forward(grids, loss_t, temp, norm_n);
        expect_close(host(loss_t), {loss}, 1e-6, 1e-5, "tv loss");
        auto tv_grad = gpu(pattern(count, 1.f, 11), {N, C, L, H, W});
        table.tv_backward(grids, tv_grad, 0.25f, norm_n);
        expect_close(host(tv_grad), gradient, 1e-6, 1e-5, "tv gradient");

        // Mean projection, per image and shared.
        for (const int per_image : {0, 1}) {
            const auto values = pattern(2 * 12 * 24, 1.f, 2);
            const auto mean = pattern(per_image ? 24 : 12, 1.f, 11);
            const auto identity = pattern(12, 1.f, 19);
            auto grids_t = gpu(values, {2, 12, 2, 3, 4});
            table.project_mean(grids_t, gpu(mean, {mean.size()}), gpu(identity, {12}), per_image);
            Doubles expected(values.size());
            for (size_t i = 0; i < values.size(); ++i) {
                const size_t c = (i / 24) % 12, n = i / (24 * 12);
                expected[i] = values[i] + (identity[c] - (per_image ? mean[n * 12 + c] : mean[c]));
            }
            expect_close(host(grids_t), expected, 1e-6, 1e-6, std::format("project mean per_image={}", per_image));
        }

        {
            const auto sum = pattern(12, 1.f, 1), identity = pattern(12, 1.f, 19);
            const auto old_mean = pattern(12, 1.f, 23), new_mean = pattern(12, 1.f, 29);
            auto sum_t = gpu(sum, {12});
            auto shared_t = gpu(pattern(12, 1.f, 11), {12});
            table.update_offset(sum_t, shared_t, gpu(identity, {12}), gpu(old_mean, {12}), gpu(new_mean, {12}), 24.f, 1.f / 48.f);
            Doubles expected_sum(12), expected_shared(12);
            for (int c = 0; c < 12; ++c) {
                expected_sum[c] = sum[c] + (new_mean[c] - old_mean[c]) * 24.f;
                expected_shared[c] = identity[c] - expected_sum[c] / 48.0;
            }
            expect_close(host(sum_t), expected_sum, 1e-5, 1e-6, "offset channel sum");
            expect_close(host(shared_t), expected_shared, 1e-6, 1e-5, "shared offset");
        }

        {
            const size_t n = 96;
            const auto p = pattern(n, 1.f, 4), m = pattern(n, 1.f, 11), v = pattern(n, 1.f, 17, 1.1f), g = pattern(n, 1.f, 19);
            auto p_t = gpu(p, {n}), m_t = gpu(m, {n}), v_t = gpu(v, {n});
            const ops::AdamUpdateParams adam{0.002f, 0.9f, 0.999f, 10.f, 31.622776f, 1e-15f};
            table.adam(p_t, m_t, v_t, gpu(g, {n}), adam);
            table.scale_moments(m_t, v_t, 0.81f, 0.998001f);
            Doubles ep(n), em(n), ev(n);
            for (size_t i = 0; i < n; ++i) {
                const float mi = 0.9f * m[i] + 0.1f * g[i];
                const float vi = 0.999f * v[i] + 0.001f * g[i] * g[i];
                ep[i] = p[i] - 0.002f * (mi * 10.f) / (std::sqrt(vi * 31.622776f * 31.622776f) + 1e-15f);
                em[i] = mi * 0.81f;
                ev[i] = vi * 0.998001f;
            }
            expect_close(host(p_t), ep, 1e-6, 1e-5, "bilateral adam parameter");
            expect_close(host(m_t), em, 1e-7, 1e-6, "bilateral adam moment1");
            expect_close(host(v_t), ev, 1e-7, 1e-6, "bilateral adam moment2");
        }

        {
            const auto values = pattern(8, 0.5f, 7);
            auto host_t = Tensor::from_vector(values, {8}, Device::CPU);
            auto device = Tensor::zeros({12}, Device::GPU);
            table.upload_slice(host_t, device, 2, 5, 6);
            auto back = Tensor::zeros({10}, Device::CPU);
            table.download_slice(back, device, 1, 4, 7);
            const auto d = host(device), b = host(back);
            const std::vector<float> expect_device{0, 0, 0, 0, 0, values[2], values[3], values[4], values[5], values[6], values[7], 0};
            const std::vector<float> expect_back{0, 0, values[2], values[3], values[4], values[5], values[6], values[7], 0, 0};
            EXPECT_EQ(d, expect_device);
            EXPECT_EQ(b, expect_back);
        }
    }

    PpispParams ppisp_values() {
        return {widen(pattern(3, 0.3f, 1)), widen(pattern(30, 0.15f, 2)), widen(pattern(24, 0.6f, 3)),
                widen(pattern(24, 0.5f, 4))};
    }

    std::vector<float> narrow(const Doubles& values) { return {values.begin(), values.end()}; }

    TEST_P(PortableAppearance, PpispForwardMatchesReference) {
        const auto& table = *this->table().ppisp;
        const PpispParams p = ppisp_values();
        // Keep vignetting alphas nonpositive so the falloff stays inside [0, 1].
        auto vig = p.vignetting;
        for (int i = 0; i < 30; ++i)
            if (i % 5 >= 2)
                vig[i] = -std::abs(vig[i]);
        const PpispParams q{p.exposure, vig, p.color, p.crf};
        const auto exposure = gpu(narrow(q.exposure), {3}), vignetting = gpu(narrow(q.vignetting), {30});
        const auto color = gpu(narrow(q.color), {24}), crf = gpu(narrow(q.crf), {24});
        constexpr int H = 6, W = 5;
        const auto rgb_values = pattern(3 * H * W, 0.3f, 7, 0.5f);
        const auto rgb = gpu(rgb_values, {3, H, W});
        for (const auto [camera, frame] : {std::pair{1, 2}, std::pair{-1, 0}, std::pair{0, -1}, std::pair{-1, -1}}) {
            auto out = Tensor::empty({3, H, W}, Device::GPU);
            table.forward({exposure, vignetting, color, crf}, rgb, out, {0, H, 2, 3, camera, frame});
            expect_close(host(out), ppisp_forward(q, widen(rgb_values), H, W, 0, H, camera, frame), 2e-5, 5e-5,
                         std::format("ppisp forward camera={} frame={}", camera, frame));
        }
        // A band of rows 2..4 equals those rows of the full pass.
        auto full = Tensor::empty({3, H, W}, Device::GPU);
        table.forward({exposure, vignetting, color, crf}, rgb, full, {0, H, 2, 3, 1, 2});
        const auto band_in = rgb.slice(1, 2, 5).contiguous();
        auto band = Tensor::empty({3, 3, W}, Device::GPU);
        table.forward({exposure, vignetting, color, crf}, band_in, band, {2, H, 2, 3, 1, 2});
        EXPECT_EQ(host(band), host(full.slice(1, 2, 5).contiguous()));
    }

    TEST_P(PortableAppearance, PpispBackwardMatchesFiniteDifferences) {
        const auto& table = *this->table().ppisp;
        PpispParams p = ppisp_values();
        for (int i = 0; i < 30; ++i)
            if (i % 5 >= 2)
                p.vignetting[i] = -std::abs(p.vignetting[i]) * 0.5;
        constexpr int H = 3, W = 4, camera = 1, frame = 2;
        const auto rgb_values = pattern(3 * H * W, 0.25f, 7, 0.45f);
        const auto grad_values = pattern(3 * H * W, 1.f, 9);
        const Doubles rgb = widen(rgb_values);
        const auto exposure = gpu(narrow(p.exposure), {3}), vignetting = gpu(narrow(p.vignetting), {30});
        const auto color = gpu(narrow(p.color), {24}), crf = gpu(narrow(p.crf), {24});
        auto ge = Tensor::zeros({3}, Device::GPU), gv = Tensor::zeros({30}, Device::GPU);
        auto gc = Tensor::zeros({24}, Device::GPU), gk = Tensor::zeros({24}, Device::GPU);
        auto grad_rgb = Tensor::empty({3, H, W}, Device::GPU);
        table.backward({exposure, vignetting, color, crf}, gpu(rgb_values, {3, H, W}), gpu(grad_values, {3, H, W}),
                       {ge, gv, gc, gk}, grad_rgb, 2, 3, camera, frame);

        const auto loss = [&](const PpispParams& q, const Doubles& image) {
            return weighted(ppisp_forward(q, image, H, W, 0, H, camera, frame), grad_values);
        };
        const auto group = [&](Doubles PpispParams::* member) {
            return finite_difference(p.*member, [&](const Doubles& values) {
                PpispParams q = p;
                q.*member = values;
                return loss(q, rgb);
            });
        };
        expect_close(host(grad_rgb), finite_difference(rgb, [&](const Doubles& image) { return loss(p, image); }), 2e-4, 2e-3,
                     "ppisp rgb gradient");
        expect_close(host(ge), group(&PpispParams::exposure), 2e-4, 2e-3, "ppisp exposure gradient");
        expect_close(host(gv), group(&PpispParams::vignetting), 2e-4, 2e-3, "ppisp vignetting gradient");
        expect_close(host(gc), group(&PpispParams::color), 2e-4, 2e-3, "ppisp color gradient");
        expect_close(host(gk), group(&PpispParams::crf), 2e-4, 2e-3, "ppisp crf gradient");
    }

    // Parameter gradients summed over many workgroups: each reduces its pixels
    // before one atomic per parameter.
    TEST_P(PortableAppearance, PpispBackwardSumsAcrossWorkgroups) {
        const auto& table = *this->table().ppisp;
        PpispParams p = ppisp_values();
        for (int i = 0; i < 30; ++i)
            if (i % 5 >= 2)
                p.vignetting[i] = -std::abs(p.vignetting[i]) * 0.5;
        constexpr int H = 48, W = 96, camera = 1, frame = 2;
        const auto rgb_values = pattern(3 * H * W, 0.25f, 7, 0.45f);
        const auto grad_values = pattern(3 * H * W, 1.f, 9);
        const Doubles rgb = widen(rgb_values);
        const auto exposure = gpu(narrow(p.exposure), {3}), vignetting = gpu(narrow(p.vignetting), {30});
        const auto color = gpu(narrow(p.color), {24}), crf = gpu(narrow(p.crf), {24});
        auto ge = Tensor::zeros({3}, Device::GPU), gv = Tensor::zeros({30}, Device::GPU);
        auto gc = Tensor::zeros({24}, Device::GPU), gk = Tensor::zeros({24}, Device::GPU);
        auto grad_rgb = Tensor::empty({3, H, W}, Device::GPU);
        table.backward({exposure, vignetting, color, crf}, gpu(rgb_values, {3, H, W}), gpu(grad_values, {3, H, W}),
                       {ge, gv, gc, gk}, grad_rgb, 2, 3, camera, frame);

        const auto group = [&](Doubles PpispParams::* member) {
            return finite_difference(p.*member, [&](const Doubles& values) {
                PpispParams q = p;
                q.*member = values;
                return weighted(ppisp_forward(q, rgb, H, W, 0, H, camera, frame), grad_values);
            });
        };
        expect_close(host(ge), group(&PpispParams::exposure), 2e-4, 2e-3, "ppisp exposure gradient");
        expect_close(host(gv), group(&PpispParams::vignetting), 2e-4, 2e-3, "ppisp vignetting gradient");
        expect_close(host(gc), group(&PpispParams::color), 2e-4, 2e-3, "ppisp color gradient");
        expect_close(host(gk), group(&PpispParams::crf), 2e-4, 2e-3, "ppisp crf gradient");
    }

    TEST_P(PortableAppearance, PpispOptimizerAndRegularizers) {
        const auto& table = *this->table().ppisp;
        {
            auto e = gpu(pattern(3, 1.f, 1), {3}), v = gpu(pattern(30, 1.f, 2), {30});
            auto c = gpu(pattern(24, 1.f, 3), {24}), k = gpu(pattern(24, 1.f, 4), {24});
            table.initialize({e, v, c, k});
            const float toe = std::log(std::exp(0.7f) - 1.0f), gamma = std::log(std::exp(0.9f) - 1.0f);
            std::vector<float> expect_crf(24);
            for (int i = 0; i < 24; ++i)
                expect_crf[i] = i % 4 == 3 ? 0.f : i % 4 == 2 ? gamma
                                                              : toe;
            EXPECT_EQ(host(e), std::vector<float>(3, 0.f));
            EXPECT_EQ(host(v), std::vector<float>(30, 0.f));
            EXPECT_EQ(host(c), std::vector<float>(24, 0.f));
            // ppisp.cu computes the identity with __logf/__expf.
            const auto crf = host(k);
            ASSERT_EQ(crf.size(), expect_crf.size());
            for (size_t i = 0; i < crf.size(); ++i)
                EXPECT_NEAR(crf[i], expect_crf[i], 2e-6f * std::abs(expect_crf[i])) << "crf " << i;
        }

        const ops::PPISPAdamUpdateParams hyper{0.003f, 0.9f, 0.999f, 10.f, 31.622776f, 1e-8f};
        const auto adam_reference = [&](std::vector<float> p, std::vector<float> m, std::vector<float> v, const std::vector<float>& g) {
            for (size_t i = 0; i < p.size(); ++i) {
                const float mi = hyper.beta1 * m[i] + (1.0f - hyper.beta1) * g[i];
                const float vi = hyper.beta2 * v[i] + (1.0f - hyper.beta2) * g[i] * g[i];
                p[i] = p[i] - hyper.lr * (mi * hyper.bc1_rcp) / (std::sqrt(vi) * hyper.bc2_sqrt_rcp + hyper.eps);
            }
            return widen(p);
        };
        {
            const auto p = pattern(257, 1.f, 1), m = pattern(257, 1.f, 2), v = pattern(257, 0.5f, 3, 0.6f), g = pattern(257, 1.f, 4);
            auto pt = gpu(p, {257}), mt = gpu(m, {257}), vt = gpu(v, {257});
            table.adam({pt, mt, vt, gpu(g, {257})}, hyper);
            expect_close(host(pt), adam_reference(p, m, v, g), 1e-6, 1e-5, "ppisp adam");
        }
        {
            const size_t sizes[4] = {3, 30, 24, 24};
            std::array<Tensor, 4> pt, mt, vt, gt;
            std::array<std::vector<float>, 4> p, m, v, g;
            for (int i = 0; i < 4; ++i) {
                p[i] = pattern(sizes[i], 1.f, 1 + i);
                m[i] = pattern(sizes[i], 1.f, 5 + i);
                v[i] = pattern(sizes[i], 0.5f, 9 + i, 0.6f);
                g[i] = pattern(sizes[i], 1.f, 13 + i);
                pt[i] = gpu(p[i], {sizes[i]});
                mt[i] = gpu(m[i], {sizes[i]});
                vt[i] = gpu(v[i], {sizes[i]});
                gt[i] = gpu(g[i], {sizes[i]});
            }
            // An invalid parameter binding skips the vignetting group.
            Tensor absent;
            table.adam_batch({ops::PPISPAdamGroup{pt[0], mt[0], vt[0], gt[0]}, ops::PPISPAdamGroup{absent, mt[1], vt[1], gt[1]},
                              ops::PPISPAdamGroup{pt[2], mt[2], vt[2], gt[2]}, ops::PPISPAdamGroup{pt[3], mt[3], vt[3], gt[3]}},
                             hyper);
            for (int i = 0; i < 4; ++i) {
                const std::string what = std::format("ppisp adam batch group {}", i);
                if (i == 1)
                    EXPECT_EQ(host(mt[1]), m[1]) << what;
                else
                    expect_close(host(pt[i]), adam_reference(p[i], m[i], v[i], g[i]), 1e-6, 1e-5, what);
            }
        }
        {
            constexpr int cameras = 2;
            const auto vig = pattern(15 * cameras, 1.f, 8), grad = pattern(15 * cameras, 1.f, 3);
            const float center = 0.01f, channel = 0.02f, non_positive = 0.03f;
            auto grad_t = gpu(grad, {grad.size()});
            auto loss_t = Tensor::zeros({1}, Device::GPU);
            table.vignetting_regularization(gpu(vig, {vig.size()}), grad_t, loss_t, center, channel, non_positive);
            Doubles expect_grad = widen(grad);
            double loss = 0;
            const double inv_center = 1.0 / (cameras * 3), inv_non_pos = 1.0 / (cameras * 9), inv_channel = 1.0 / (cameras * 15);
            for (int cam = 0; cam < cameras; ++cam) {
                const float* v = vig.data() + cam * 15;
                double* g = expect_grad.data() + cam * 15;
                for (int ch = 0; ch < 3; ++ch) {
                    loss += center * (v[ch * 5] * v[ch * 5] + v[ch * 5 + 1] * v[ch * 5 + 1]) * inv_center;
                    g[ch * 5] += 2 * center * inv_center * v[ch * 5];
                    g[ch * 5 + 1] += 2 * center * inv_center * v[ch * 5 + 1];
                    for (int a = 0; a < 3; ++a) {
                        if (v[ch * 5 + 2 + a] > 0) {
                            loss += non_positive * v[ch * 5 + 2 + a] * inv_non_pos;
                            g[ch * 5 + 2 + a] += non_positive * inv_non_pos;
                        }
                    }
                }
                for (int k = 0; k < 5; ++k) {
                    const double mean = (double(v[k]) + v[5 + k] + v[10 + k]) / 3;
                    for (int ch = 0; ch < 3; ++ch) {
                        const double diff = v[ch * 5 + k] - mean;
                        loss += channel * diff * diff * inv_channel;
                        g[ch * 5 + k] += 2 * channel * inv_channel * diff;
                    }
                }
            }
            expect_close(host(grad_t), expect_grad, 1e-6, 1e-5, "vignetting regularizer gradient");
            expect_close(host(loss_t), {loss}, 1e-7, 1e-5, "vignetting regularizer loss");
        }
        {
            constexpr int frames = 300;
            const auto e = pattern(frames, 1.f, 1), c = pattern(frames * 8, 1.f, 3);
            auto et = gpu(e, {frames}), ct = gpu(c, {frames * 8});
            table.project_mean(et, ct);
            Doubles ee(frames), ec(frames * 8);
            double sum = 0;
            for (const float v : e)
                sum += v;
            for (int i = 0; i < frames; ++i)
                ee[i] = e[i] - sum / frames;
            for (int k = 0; k < 8; ++k) {
                double s = 0;
                for (int f = 0; f < frames; ++f)
                    s += c[f * 8 + k];
                for (int f = 0; f < frames; ++f)
                    ec[f * 8 + k] = c[f * 8 + k] - s / frames;
            }
            expect_close(host(et), ee, 1e-6, 0, "ppisp exposure projection");
            expect_close(host(ct), ec, 1e-6, 0, "ppisp color projection");
        }
    }

    TEST_P(PortableAppearance, ControllerOps) {
        const auto& table = *this->table().controller;
        const auto features = pattern(1600, 1.f, 2);
        auto fc = gpu(pattern(1601, 1.f, 5), {1, 1601});
        table.prepare_input({}, fc, 0.7f);
        auto expected = pattern(1601, 1.f, 5);
        expected[1600] = 0.7f;
        EXPECT_EQ(host(fc), expected);
        // A prior of 1 keeps the previous prior slot.
        table.prepare_input(gpu(features, {1, 1600}), fc, 1.0f);
        std::copy(features.begin(), features.end(), expected.begin());
        EXPECT_EQ(host(fc), expected);
        table.prepare_input(gpu(features, {1, 1600}), fc, 0.25f);
        expected[1600] = 0.25f;
        EXPECT_EQ(host(fc), expected);

        constexpr int m = 9, n = 128;
        const auto grad = pattern(m, 1.f, 3), activation = pattern(n, 1.f, 2), weight = pattern(m * n, 1.f, 3);
        const auto weight_grad = pattern(m * n, 1.f, 4), bias_grad = pattern(m, 1.f, 5);
        auto wg = gpu(weight_grad, {m, n}), bg = gpu(bias_grad, {m}), ig = Tensor::empty({1, n}, Device::GPU);
        table.backward_layer(gpu(grad, {1, m}), gpu(activation, {1, n}), gpu(weight, {m, n}), wg, bg, ig);
        Doubles ewg(m * n), ebg(m), eig(n);
        for (int i = 0; i < m; ++i) {
            ebg[i] = bias_grad[i] + grad[i];
            for (int j = 0; j < n; ++j)
                ewg[i * n + j] = weight_grad[i * n + j] + double(grad[i]) * activation[j];
        }
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            for (int i = 0; i < m; ++i)
                sum += double(grad[i]) * weight[i * n + j];
            eig[j] = activation[j] > 0 ? sum : 0;
        }
        expect_close(host(wg), ewg, 1e-6, 1e-6, "controller weight gradient");
        expect_close(host(bg), ebg, 1e-6, 1e-6, "controller bias gradient");
        expect_close(host(ig), eig, 1e-5, 1e-5, "controller input gradient");

        // Without an input gradient the first layer only accumulates.
        Tensor none;
        table.backward_layer(gpu(grad, {1, m}), gpu(activation, {1, n}), gpu(weight, {m, n}), wg, bg, none);
        for (int i = 0; i < m; ++i)
            ebg[i] += grad[i];
        expect_close(host(bg), ebg, 1e-6, 1e-6, "controller bias gradient, first layer");
    }

    TEST_P(PortableAppearance, LpipsOps) {
        const auto& table = *this->table().lpips;
        const auto half_values = [](std::vector<float> values) {
            for (float& v : values)
                v = half_round(v);
            return values;
        };
        {
            constexpr int cout = 16, cin = 8;
            const auto w = half_values(pattern(cout * cin * 9, 1.f, 7));
            auto taps = Tensor::empty({9, cout, cin}, Device::GPU, DataType::Float16);
            table.weight_taps(gpu(w, {cout, cin, 3, 3}).to(DataType::Float16), taps);
            Doubles expected(w.size());
            for (int t = 0; t < 9; ++t)
                for (int o = 0; o < cout; ++o)
                    for (int i = 0; i < cin; ++i)
                        expected[(t * cout + o) * cin + i] = w[(o * cin + i) * 9 + t];
            expect_close(host(taps), expected, 0, 0, "lpips weight taps");
        }
        constexpr int H = 7, W = 9;
        {
            const auto x = pattern(3 * H * W, 1.f, 1, 0.5f);
            const auto w = half_values(pattern(64 * 27, 1.f, 2)), b = half_values(pattern(64, 1.f, 3));
            const ops::RGBConvParams params{{-.030f, -.088f, -.188f}, {.458f, .448f, .450f}, true};
            auto out = Tensor::empty({1, 64, H, W}, Device::GPU, DataType::Float16);
            table.rgb_conv(gpu(x, {1, 3, H, W}), gpu(w, {64, 3, 3, 3}).to(DataType::Float16),
                           gpu(b, {64}).to(DataType::Float16), out, params);
            Doubles expected(64 * H * W);
            for (int oc = 0; oc < 64; ++oc) {
                for (int p = 0; p < H * W; ++p) {
                    float sum = 0;
                    for (int ic = 0; ic < 3; ++ic)
                        for (int kh = 0; kh < 3; ++kh)
                            for (int kw = 0; kw < 3; ++kw) {
                                const int ih = p / W + kh - 1, iw = p % W + kw - 1;
                                if (ih < 0 || iw < 0 || ih >= H || iw >= W)
                                    continue;
                                const float v = x[ic * H * W + ih * W + iw] * 2.f - 1.f;
                                sum += half_round((v - params.shift[ic]) / params.scale[ic]) * w[oc * 27 + ic * 9 + kh * 3 + kw];
                            }
                    expected[oc * H * W + p] = half_round(std::max(sum + b[oc], 0.f));
                }
            }
            expect_close(host(out), expected, 1e-3, 2e-3, "lpips rgb conv");
        }
        for (const bool half_path : {true, false}) {
            constexpr int cin = 16, cout = 8;
            auto x = pattern(cin * H * W, 1.f, 1), w = pattern(cout * cin * 9, 0.3f, 2), b = pattern(cout, 1.f, 3);
            if (half_path) {
                x = half_values(x);
                w = half_values(w);
                b = half_values(b);
            }
            const auto dtype = half_path ? DataType::Float16 : DataType::Float32;
            auto out = Tensor::empty({1, cout, H, W}, Device::GPU, dtype);
            Tensor scratch, taps;
            ops::ConvParams params;
            params.pad_h = params.pad_w = 1;
            params.activation = lfs::core::nn::Activation::Relu;
            table.convolution(gpu(x, {1, cin, H, W}).to(dtype), gpu(w, {cout, cin, 3, 3}).to(dtype), taps,
                              gpu(b, {cout}).to(dtype), out, scratch, params);
            Doubles expected(cout * H * W);
            for (int oc = 0; oc < cout; ++oc) {
                for (int p = 0; p < H * W; ++p) {
                    double sum = b[oc];
                    for (int ic = 0; ic < cin; ++ic)
                        for (int kh = 0; kh < 3; ++kh)
                            for (int kw = 0; kw < 3; ++kw) {
                                const int ih = p / W + kh - 1, iw = p % W + kw - 1;
                                if (ih >= 0 && iw >= 0 && ih < H && iw < W)
                                    sum += double(x[(ic * H + ih) * W + iw]) * w[((oc * cin + ic) * 3 + kh) * 3 + kw];
                            }
                    const float relu = static_cast<float>(std::max(sum, 0.0));
                    expected[oc * H * W + p] = half_path ? half_round(relu) : relu;
                }
            }
            expect_close(host(out), expected, half_path ? 1e-3 : 1e-5, half_path ? 2e-3 : 1e-5,
                         std::format("lpips conv {}", half_path ? "fp16" : "fp32"));
        }
        {
            constexpr int C = 16, h = 3, w = 8;
            const auto x = half_values(pattern(C * h * w, 1.f, 1)), y = half_values(pattern(C * h * w, 1.f, 2));
            const auto lin = half_values(pattern(C, 1.f, 3));
            auto score = Tensor::full({1}, 0.5f, Device::GPU);
            auto px = Tensor::empty({1, C, 1, 4}, Device::GPU, DataType::Float16);
            auto py = Tensor::empty({1, C, 1, 4}, Device::GPU, DataType::Float16);
            const ops::PoolReduceParams params{0, 2, 1, 7, 1.f / 12.f};
            table.pool_reduce(gpu(x, {1, C, h, w}).to(DataType::Float16), gpu(y, {1, C, h, w}).to(DataType::Float16),
                              gpu(lin, {1, C, 1, 1}).to(DataType::Float16), score, px, py, params);
            double total = 0;
            for (int row = 0; row < h; ++row) {
                for (int col = 0; col < w; ++col) {
                    double nx = 0, ny = 0;
                    for (int c = 0; c < C; ++c) {
                        nx += double(x[(c * h + row) * w + col]) * x[(c * h + row) * w + col];
                        ny += double(y[(c * h + row) * w + col]) * y[(c * h + row) * w + col];
                    }
                    if (row < params.y0 || row >= params.y1 || col < params.x0 || col >= params.x1)
                        continue;
                    for (int c = 0; c < C; ++c) {
                        const double d = x[(c * h + row) * w + col] / std::sqrt(nx + 1e-10) -
                                         y[(c * h + row) * w + col] / std::sqrt(ny + 1e-10);
                        total += lin[c] * d * d;
                    }
                }
            }
            expect_close(host(score), {0.5 + total * params.inverse_count}, 1e-5, 1e-4, "lpips pool score");
            Doubles pooled(C * 4);
            for (int c = 0; c < C; ++c)
                for (int j = 0; j < 4; ++j)
                    pooled[c * 4 + j] = std::max({x[c * h * w + 2 * j], x[c * h * w + 2 * j + 1], x[c * h * w + w + 2 * j],
                                                  x[c * h * w + w + 2 * j + 1]});
            expect_close(host(px), pooled, 0, 0, "lpips max pool");
        }
    }

    // ---- SharedImage ------------------------------------------------------------

    uint8_t sentinel(const uint32_t index, const uint32_t seed) {
        uint32_t value = index ^ seed;
        value ^= value >> 16u;
        value *= 0x7feb352du;
        value ^= value >> 15u;
        value *= 0x846ca68bu;
        value ^= value >> 16u;
        return static_cast<uint8_t>(value);
    }

    std::vector<uint8_t> image_bytes(const size_t count) {
        std::vector<uint8_t> bytes(count);
        for (size_t i = 0; i < count; ++i)
            bytes[i] = static_cast<uint8_t>((i * 71 + 149) % 256);
        return bytes;
    }

    std::vector<float> image_floats(const size_t count, const float offset = 0.f, const float scale = 1.f) {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i)
            values[i] = offset + scale * static_cast<float>((i * 7919 + 104729) % 2003) / 2002.f;
        return values;
    }

    TEST_P(PortableAppearance, SharedImageSentinelAndConversions) {
        const auto* table = this->table().shared_image;
        ASSERT_NE(table, nullptr);
        {
            constexpr uint32_t seed = 0x932abe71u;
            auto bytes = Tensor::empty({1031}, Device::GPU, DataType::UInt8);
            table->sentinel_fill(bytes, seed);
            auto filled = host_bytes(bytes);
            for (uint32_t i = 0; i < filled.size(); ++i)
                ASSERT_EQ(filled[i], sentinel(i, seed)) << i;
            auto flag = Tensor::full({1}, 1.f, Device::GPU, DataType::UInt32);
            table->sentinel_check(bytes, flag, seed);
            EXPECT_EQ(flag.cpu().ptr<uint32_t>()[0], 1u);
            filled[517] ^= 1u;
            table->sentinel_check(gpu_bytes(filled, {1031}), flag, seed);
            EXPECT_EQ(flag.cpu().ptr<uint32_t>()[0], 0u);
        }

        constexpr size_t H = 13, W = 17, P = H * W;
        const auto u8 = image_bytes(P * 6);
        std::vector<uint16_t> u16(P * 3);
        std::memcpy(u16.data(), u8.data(), P * 6);
        const auto f32 = image_floats(P * 3, -0.2f, 1.4f);
        const auto run = [&](const ops::ImageConversion kind, const Tensor& source, const TensorShape& shape,
                             const DataType dtype, const size_t channels, const ops::NormalPriorTransform& transform = {}) {
            auto destination = Tensor::zeros(shape, Device::GPU, dtype);
            table->convert(source, destination, kind, H, W, channels, transform);
            return destination;
        };
        const auto f32_t = gpu(f32, {H, W, 3});
        const auto u8_t = gpu_bytes({u8.begin(), u8.begin() + P * 3}, {H, W, 3});
        const auto u16_t = gpu_bytes(u8, {H, W, 6});

        Doubles u8_chw(P * 3), u16_chw(P * 3), u16_hwc(P * 3), j2k(P * 3), normal(P * 3);
        std::vector<uint8_t> u8_u8(P * 3), u16_u8(P * 3), f32_u8(P * 3), f32_u16(P * 6);
        for (size_t i = 0; i < P * 3; ++i) {
            const size_t c = i % 3, pixel = i / 3, chw = c * P + pixel;
            u8_chw[chw] = static_cast<float>(u8[i]) * (1.0f / 255.0f);
            u16_chw[chw] = static_cast<float>(u16[i]) * (1.0f / 65535.f);
            u16_hwc[i] = static_cast<float>(u16[i]) * (1.0f / 65535.f);
            u8_u8[chw] = u8[i];
            u16_u8[chw] = static_cast<uint8_t>((static_cast<uint32_t>(u16[i]) * 255u + 32767u) / 65535u);
            f32_u8[i] = static_cast<uint8_t>(std::clamp(f32[i], 0.0f, 1.0f) * 255.0f + 0.5f);
            const auto u16_value = static_cast<uint16_t>(std::min(std::max(f32[i] * 65535.0f, 0.0f), 65535.0f) + 0.5f);
            std::memcpy(f32_u16.data() + i * 2, &u16_value, 2);
            j2k[pixel * 3 + c] = std::clamp(f32[chw] * 0.5f + 0.5f, 0.0f, 1.0f);
            normal[chw] = f32[pixel * 3 + c] * 2.0f - 1.0f;
        }
        expect_close(host(run(ops::ImageConversion::U8HWCToF32CHW, u8_t, {3, H, W}, DataType::Float32, 3)), u8_chw, 0, 0, "u8 -> f32 chw");
        expect_close(host(run(ops::ImageConversion::U16HWCToF32CHW, u16_t, {3, H, W}, DataType::Float32, 3)), u16_chw, 0, 0, "u16 -> f32 chw");
        expect_close(host(run(ops::ImageConversion::U16HWCToF32HWC, u16_t, {H, W, 3}, DataType::Float32, 3)), u16_hwc, 0, 0, "u16 -> f32 hwc");
        expect_close(host(run(ops::ImageConversion::NormalCHWToJ2KHWC, f32_t, {H, W, 3}, DataType::Float32, 3)), j2k, 0, 0, "normal -> j2k");
        expect_close(host(run(ops::ImageConversion::J2KHWCToNormalCHW, f32_t, {3, H, W}, DataType::Float32, 3)), normal, 0, 0, "j2k -> normal");
        EXPECT_EQ(host_bytes(run(ops::ImageConversion::F32HWCToU16HWC, f32_t, {H, W, 6}, DataType::UInt8, 3)), f32_u16);
        EXPECT_EQ(host_bytes(run(ops::ImageConversion::U8HWCToU8CHW, u8_t, {3, H, W}, DataType::UInt8, 3)), u8_u8);
        EXPECT_EQ(host_bytes(run(ops::ImageConversion::U16HWCToU8CHW, u16_t, {3, H, W}, DataType::UInt8, 3)), u16_u8);
        EXPECT_EQ(host_bytes(run(ops::ImageConversion::F32CHWToU8CHW, gpu(f32, {3, H, W}), {3, H, W}, DataType::UInt8, 3)), f32_u8);
        {
            Doubles gray(P);
            for (size_t i = 0; i < P; ++i)
                gray[i] = static_cast<float>(u8[i]) * (1.0f / 255.0f);
            expect_close(host(run(ops::ImageConversion::U8HWToF32HW, gpu_bytes({u8.begin(), u8.begin() + P}, {H, W}), {H, W},
                                  DataType::Float32, 1)),
                         gray, 0, 0, "u8 -> f32 gray");
        }
        for (const bool wide : {false, true}) {
            const ops::NormalPriorTransform transform{.srgb = true, .flip_yz = true, .world_to_camera = true, .w2c = {0, -1, 0, 1, 0, 0, 0, 0, 1}};
            Doubles expected(P * 3);
            for (size_t pixel = 0; pixel < P; ++pixel) {
                double n[3];
                for (int c = 0; c < 3; ++c) {
                    const double v = wide ? u16[pixel * 3 + c] / 65535.0 : u8[pixel * 3 + c] / 255.0;
                    const double linear = v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
                    n[c] = linear * 2 - 1;
                }
                n[1] = -n[1];
                n[2] = -n[2];
                const double x = n[0], y = n[1], z = n[2];
                for (int r = 0; r < 3; ++r)
                    expected[r * P + pixel] = transform.w2c[r * 3] * x + transform.w2c[r * 3 + 1] * y + transform.w2c[r * 3 + 2] * z;
            }
            const auto kind = wide ? ops::ImageConversion::NormalPriorU16 : ops::ImageConversion::NormalPriorU8;
            expect_close(host(run(kind, wide ? u16_t : u8_t, {3, H, W}, DataType::Float32, 3, transform)), expected, 1e-6, 1e-6,
                         wide ? "normal prior u16" : "normal prior u8");
        }

        {
            const auto rgba = image_bytes(P * 4);
            auto rgb_f = Tensor::empty({3, H, W}, Device::GPU), alpha = Tensor::empty({H, W}, Device::GPU);
            auto rgb_u = Tensor::empty({3, H, W}, Device::GPU, DataType::UInt8);
            table->rgba_split(gpu_bytes(rgba, {H, W, 4}), rgb_f, alpha);
            auto alpha_u = Tensor::empty({H, W}, Device::GPU);
            table->rgba_split(gpu_bytes(rgba, {H, W, 4}), rgb_u, alpha_u);
            Doubles rgb(P * 3), a(P);
            std::vector<uint8_t> rgb_bytes(P * 3);
            for (size_t i = 0; i < P; ++i) {
                for (int c = 0; c < 3; ++c) {
                    rgb[c * P + i] = static_cast<float>(rgba[i * 4 + c]) * (1.0f / 255.0f);
                    rgb_bytes[c * P + i] = rgba[i * 4 + c];
                }
                a[i] = static_cast<float>(rgba[i * 4 + 3]) * (1.0f / 255.0f);
            }
            expect_close(host(rgb_f), rgb, 0, 0, "rgba split rgb");
            expect_close(host(alpha), a, 0, 0, "rgba split alpha");
            EXPECT_EQ(host_bytes(rgb_u), rgb_bytes);
            expect_close(host(alpha_u), a, 0, 0, "rgba split alpha (uint8 rgb)");
        }
        {
            const auto values = image_floats(P);
            auto thresholded = gpu(values, {H, W}), inverted = gpu(values, {H, W});
            table->mask(thresholded, ops::MaskTransform::Threshold, 0.61f);
            table->mask(inverted, ops::MaskTransform::Invert, 0.f);
            Doubles t(P), inv(P);
            for (size_t i = 0; i < P; ++i) {
                t[i] = values[i] >= 0.61f ? 1.0f : 0.0f;
                inv[i] = 1.0f - values[i];
            }
            expect_close(host(thresholded), t, 0, 0, "mask threshold");
            expect_close(host(inverted), inv, 0, 0, "mask invert");
        }
    }

    // lanczos_resize.cu in float: normalized tap weights, then the separable box sum.
    std::vector<float> lanczos_taps(const int in, const int out, const int k, const int index, int& lo) {
        const float scale = 1.0f * in / out, center = (index + 0.5f) * scale;
        lo = std::max(static_cast<int>(center - k * scale + 0.5f), 0);
        const int hi = std::min(static_cast<int>(center + k * scale + 0.5f), in);
        std::vector<float> taps;
        float norm = 0;
        const auto sinc = [](const float x) {
            return std::abs(x) < 1e-12f ? 1.0f : std::sin(static_cast<float>(M_PI) * x) / (static_cast<float>(M_PI) * x);
        };
        for (int i = lo; i < hi; ++i) {
            const float x = (i + 0.5f - center) / scale;
            const float value = x <= -k || x >= k ? 0.0f : sinc(x) * sinc(x / k);
            taps.push_back(value);
            norm += value;
        }
        for (float& t : taps)
            t /= norm;
        return taps;
    }

    // layout 0: HWC RGB, 1: HW gray, 2: CHW RGB, as SharedImageOps::resize.
    Doubles lanczos_reference(const std::function<float(int, int, int)>& at, const int in_h, const int in_w, const int out_h,
                              const int out_w, const int k, const int channels) {
        Doubles out(static_cast<size_t>(channels) * out_h * out_w);
        for (int py = 0; py < out_h; ++py) {
            int ly = 0;
            const auto ty = lanczos_taps(in_h, out_h, k, py, ly);
            for (int px = 0; px < out_w; ++px) {
                int lx = 0;
                const auto tx = lanczos_taps(in_w, out_w, k, px, lx);
                for (int c = 0; c < channels; ++c) {
                    double sum = 0;
                    for (size_t j = 0; j < ty.size(); ++j)
                        for (size_t i = 0; i < tx.size(); ++i)
                            sum += at(ly + static_cast<int>(j), lx + static_cast<int>(i), c) * ty[j] * tx[i];
                    out[(c * out_h + py) * out_w + px] = sum;
                }
            }
        }
        return out;
    }

    TEST_P(PortableAppearance, SharedImageResizeAndUndistort) {
        const auto* table = this->table().shared_image;
        ASSERT_NE(table, nullptr);
        constexpr int H = 13, W = 17;
        const auto bytes = image_bytes(H * W * 3);
        const auto floats = image_floats(H * W * 3);
        for (const auto [oh, ow, k] : {std::tuple{7, 9, 2}, std::tuple{20, 11, 3}}) {
            const std::string size = std::format("{}x{} k={}", oh, ow, k);
            expect_close(host(table->resize(gpu_bytes(bytes, {H, W, 3}), oh, ow, ops::Resample::LanczosRGB, k)),
                         lanczos_reference([&](int y, int x, int c) { return bytes[(y * W + x) * 3 + c] / 255.0f; }, H, W, oh, ow, k, 3),
                         1e-5, 1e-5, "lanczos rgb u8 " + size);
            expect_close(host(table->resize(gpu(floats, {H, W, 3}), oh, ow, ops::Resample::LanczosRGB, k)),
                         lanczos_reference([&](int y, int x, int c) { return floats[(y * W + x) * 3 + c]; }, H, W, oh, ow, k, 3),
                         1e-5, 1e-5, "lanczos rgb f32 " + size);
            expect_close(host(table->resize(gpu_bytes({bytes.begin(), bytes.begin() + H * W}, {H, W}), oh, ow, ops::Resample::LanczosGray, k)),
                         lanczos_reference([&](int y, int x, int) { return bytes[y * W + x] * (1.0f / 255.0f); }, H, W, oh, ow, k, 1),
                         1e-5, 1e-5, "lanczos gray u8 " + size);
            expect_close(host(table->resize(gpu(floats, {3, H, W}), oh, ow, ops::Resample::LanczosFloatCHW, k)),
                         lanczos_reference([&](int y, int x, int c) { return floats[(c * H + y) * W + x]; }, H, W, oh, ow, k, 3),
                         1e-5, 1e-5, "lanczos chw " + size);
        }

        // Prior resizes: bilinear over valid taps, gated by the nearest tap (resize_prior_kernel).
        for (const bool normal : {false, true}) {
            const int channels = normal ? 3 : 1, oh = 7, ow = 9, sp = H * W;
            const auto prior = image_floats(channels * H * W, normal ? -0.5f : -0.1f);
            const auto valid = [&](const int i) {
                if (!normal)
                    return prior[i] > 0;
                return prior[i] * prior[i] + prior[sp + i] * prior[sp + i] + prior[2 * sp + i] * prior[2 * sp + i] >= 0.25f;
            };
            Doubles expected(static_cast<size_t>(channels) * oh * ow);
            for (int y = 0; y < oh; ++y) {
                for (int x = 0; x < ow; ++x) {
                    const float sx = std::max(0.0f, std::min(W - 1.0f, (x + 0.5f) * W / ow - 0.5f));
                    const float sy = std::max(0.0f, std::min(H - 1.0f, (y + 0.5f) * H / oh - 0.5f));
                    const int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
                    double value[3] = {}, weight = 0;
                    if (valid(static_cast<int>(sy + 0.5f) * W + static_cast<int>(sx + 0.5f))) {
                        for (int j = 0; j < 2; ++j) {
                            for (int i = 0; i < 2; ++i) {
                                const int index = std::min(y0 + j, H - 1) * W + std::min(x0 + i, W - 1);
                                if (!valid(index))
                                    continue;
                                const double wt = (i ? sx - x0 : 1.0 - (sx - x0)) * (j ? sy - y0 : 1.0 - (sy - y0));
                                weight += wt;
                                for (int c = 0; c < channels; ++c)
                                    value[c] += wt * prior[c * sp + index];
                            }
                        }
                    }
                    if (normal)
                        weight = std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
                    for (int c = 0; c < channels; ++c)
                        expected[(c * oh + y) * ow + x] = weight > 1e-8 ? value[c] / weight : 0.0;
                }
            }
            const auto shape = normal ? TensorShape{3, H, W} : TensorShape{H, W};
            expect_close(host(table->resize(gpu(prior, shape), oh, ow, normal ? ops::Resample::NormalPrior : ops::Resample::DepthPrior, 2)),
                         expected, 1e-5, 1e-5, normal ? "normal prior resize" : "depth prior resize");
        }

        lfs::core::UndistortParams p{};
        p.src_fx = p.dst_fx = 15.f;
        p.src_fy = p.dst_fy = 14.f;
        p.src_cx = p.dst_cx = 8.f;
        p.src_cy = p.dst_cy = 6.f;
        p.src_width = W;
        p.src_height = H;
        p.dst_width = 15;
        p.dst_height = 11;
        p.model_type = lfs::core::CameraModelType::PINHOLE;
        p.num_distortion = 4;
        p.distortion[0] = 0.08f;
        p.distortion[1] = -0.03f;
        p.distortion[2] = 0.001f;
        p.distortion[3] = -0.002f;
        const auto source = image_floats(3 * H * W);
        // RGB undistortion integrates the pixel footprint with Lanczos-3, unlike
        // mask/prior resampling. Keep this double-precision oracle independent
        // of the GPU kernels, including normalization over in-bounds border taps.
        constexpr int quadrature = 8;
        const auto lanczos3 = [](const double x) {
            if (std::abs(x) >= 3.0)
                return 0.0;
            if (std::abs(x) < 1e-12)
                return 1.0;
            const double px = M_PI * x;
            return std::sin(px) / px * std::sin(px / 3.0) / (px / 3.0);
        };
        Doubles expected(3 * 15 * 11);
        for (int oy = 0; oy < p.dst_height; ++oy) {
            for (int ox = 0; ox < p.dst_width; ++ox) {
                for (int qy = 0; qy < quadrature; ++qy) {
                    for (int qx = 0; qx < quadrature; ++qx) {
                        const double x = (ox + (qx + 0.5) / quadrature - p.dst_cx) / p.dst_fx;
                        const double y = (oy + (qy + 0.5) / quadrature - p.dst_cy) / p.dst_fy;
                        const double r2 = x * x + y * y;
                        const double radial = 1 + p.distortion[0] * r2 + p.distortion[1] * r2 * r2 + p.distortion[2] * r2 * r2 * r2;
                        const double p1 = p.distortion[3]; // Packed as k1, k2, k3, p1; p2 is absent.
                        const double dx = x * radial + 2 * p1 * x * y, dy = y * radial + p1 * (r2 + 2 * y * y);
                        const double sx = dx * p.src_fx + p.src_cx - 0.5, sy = dy * p.src_fy + p.src_cy - 0.5;
                        const int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
                        std::array<double, 3> value{};
                        double weight_sum = 0, absolute_weight_sum = 0;
                        for (int yy = y0 - 2; yy <= y0 + 3; ++yy) {
                            for (int xx = x0 - 2; xx <= x0 + 3; ++xx) {
                                if (xx < 0 || yy < 0 || xx >= W || yy >= H)
                                    continue;
                                const double weight = lanczos3(sx - xx) * lanczos3(sy - yy);
                                weight_sum += weight;
                                absolute_weight_sum += std::abs(weight);
                                for (int c = 0; c < 3; ++c)
                                    value[c] += weight * source[(c * H + yy) * W + xx];
                            }
                        }
                        // This fixture stays on the Lanczos path, without the
                        // ill-conditioned-weight bilinear/nearest fallback.
                        ASSERT_GT(std::abs(weight_sum), 1e-4 * absolute_weight_sum);
                        for (int c = 0; c < 3; ++c)
                            expected[(c * p.dst_height + oy) * p.dst_width + ox] += value[c] / weight_sum / (quadrature * quadrature);
                    }
                }
            }
        }
        const auto actual = host(table->undistort(gpu(source, {3, H, W}), p, false));
        RecordProperty("undistort_0", std::format("{:.9f}", actual[0]));
        RecordProperty("undistort_5", std::format("{:.9f}", actual[5]));
        RecordProperty("reference_0", std::format("{:.9f}", expected[0]));
        RecordProperty("reference_5", std::format("{:.9f}", expected[5]));
        expect_close(actual, expected, 1e-4, 1e-4, "undistort");
        if (lfs::core::gpu_backend_available(GpuBackend::CUDA)) {
            const lfs::core::GpuBackendScope cuda_scope(GpuBackend::CUDA);
            const auto* cuda_table = lfs::core::shared_image_ops(GpuBackend::CUDA);
            ASSERT_NE(cuda_table, nullptr);
            const auto cuda = host(cuda_table->undistort(gpu(source, {3, H, W}), p, false));
            RecordProperty("cuda_0", std::format("{:.9f}", cuda[0]));
            RecordProperty("cuda_5", std::format("{:.9f}", cuda[5]));
            expect_close(actual, widen(cuda), 1e-4, 1e-4, "undistort vs CUDA");
            expect_close(cuda, expected, 1e-4, 1e-4, "CUDA undistort");
        }
    }
    INSTANTIATE_TEST_SUITE_P(Backends, PortableAppearance, testing::Values(GpuBackend::Metal, GpuBackend::Vulkan),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });

} // namespace
