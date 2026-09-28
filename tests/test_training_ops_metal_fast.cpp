/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal FastRasterOps against a CPU reference of the fastgs math (preprocess,
// depth-ordered blend, background, depth and normals). Gradients are read back
// from one fused Adam step configured so the parameter moves by exactly -grad
// (beta1 = 0, eps = 1, bc2_sqrt_rcp = 0, step 1) and are compared with
// central differences of the reference with its contribution lists frozen, so
// both sides differentiate the same piecewise-smooth function.

#include "core/gpu_backend_fwd.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace {
    namespace ops = lfs::gpu_ops;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;

    constexpr double kMinAlpha = 1.0 / 255.0;
    constexpr double kMaxAlpha = 0.999;
    constexpr double kTransmittanceMin = 1e-4;

    uint32_t rest_count(const int degree) { return static_cast<uint32_t>((degree + 1) * (degree + 1) - 1); }
    uint32_t float4_slots(const uint32_t rest) { return rest == 0 ? 0 : std::min(12u, (rest * 3 + 3) / 4); }
    size_t padded(const size_t n) { return (n + 31) / 32 * 32; }
    size_t swizzled_float_index(const uint32_t p, const uint32_t coeff, const uint32_t channel, const uint32_t rest) {
        const uint32_t linear = coeff * 3 + channel;
        return (size_t{p / 32} * (float4_slots(rest) * 32) + (linear / 4) * 32 + p % 32) * 4 + linear % 4;
    }
    size_t q16_index(const uint32_t p, const uint32_t cell, const uint32_t cells) {
        return size_t{p / 32} * (cells * 32) + cell * 32 + p % 32;
    }

    struct Scene {
        int count = 0, width = 0, height = 0, degree = 0;
        float fx = 0, fy = 0, cx = 0, cy = 0;
        std::array<float, 16> view{};
        std::array<float, 3> camera{};
        std::vector<float> means, scales, rotations, opacities, sh0, rest; // rest: [count][15][3]
        std::vector<float> bg_color{0.2f, 0.3f, 0.4f};
        std::vector<float> bg_image; // [3, H, W] or empty
    };

    Scene make_scene(const int count, const int degree, const int width, const int height, const unsigned seed,
                     const float scale_lo = -2.6f, const float scale_hi = -1.6f) {
        std::mt19937 rng(seed);
        auto uniform = [&](const float lo, const float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); };
        Scene s;
        s.count = count;
        s.width = width;
        s.height = height;
        s.degree = degree;
        s.fx = 0.9f * static_cast<float>(width);
        s.fy = 0.9f * static_cast<float>(width);
        s.cx = 0.5f * static_cast<float>(width) + 0.3f;
        s.cy = 0.5f * static_cast<float>(height) - 0.2f;
        const float angle = 0.1f, c = std::cos(angle), sn = std::sin(angle);
        const std::array<float, 9> r{c, 0, sn, 0, 1, 0, -sn, 0, c};
        const std::array<float, 3> t{0.1f, -0.05f, 4.0f};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col)
                s.view[row * 4 + col] = r[row * 3 + col];
            s.view[row * 4 + 3] = t[row];
        }
        s.view[15] = 1.0f;
        for (int col = 0; col < 3; ++col)
            s.camera[col] = -(r[col] * t[0] + r[3 + col] * t[1] + r[6 + col] * t[2]);
        const float aspect = static_cast<float>(height) / static_cast<float>(width);
        for (int i = 0; i < count; ++i) {
            s.means.insert(s.means.end(), {uniform(-1.9f, 1.9f), uniform(-1.9f, 1.9f) * aspect, uniform(-0.8f, 0.8f)});
            for (int k = 0; k < 3; ++k)
                s.scales.push_back(uniform(scale_lo, scale_hi));
            for (int k = 0; k < 4; ++k)
                s.rotations.push_back(uniform(-1.0f, 1.0f) + (k == 0 ? 1.0f : 0.0f));
            s.opacities.push_back(uniform(-1.0f, 1.5f));
            for (int k = 0; k < 3; ++k)
                s.sh0.push_back(uniform(0.0f, 1.2f));
            for (uint32_t k = 0; k < 15; ++k)
                for (int ch = 0; ch < 3; ++ch)
                    s.rest.push_back(k < rest_count(degree) ? uniform(-0.05f, 0.05f) : 0.0f);
        }
        return s;
    }

    std::vector<float> random_image(const int width, const int height, const unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        std::vector<float> image(size_t{3} * width * height);
        for (auto& v : image)
            v = uniform(rng);
        return image;
    }

    // Parameters in double so central differences can perturb them.
    struct Params {
        std::vector<double> means, scales, rotations, opacities, sh0, rest;
        std::vector<double> offset2d;     // blend-only mean2d offsets
        std::vector<double> compensation; // held mip opacity factors, as the backward treats them
        explicit Params(const Scene& s)
            : means(s.means.begin(), s.means.end()),
              scales(s.scales.begin(), s.scales.end()),
              rotations(s.rotations.begin(), s.rotations.end()),
              opacities(s.opacities.begin(), s.opacities.end()),
              sh0(s.sh0.begin(), s.sh0.end()),
              rest(s.rest.begin(), s.rest.end()),
              offset2d(size_t{2} * s.count, 0.0) {}
    };

    struct Splat {
        bool visible = false;
        double mean[2]{}, conic[3]{}, opacity = 0, color[3]{}, depth = 0, normal[3]{};
        uint32_t key = 0;
        double extent[2]{};
        double compensation = 1.0;
    };

    struct View {
        int width, height;
        double fx, fy, cx, cy;
        bool mip;
        uint32_t depth_bits, grid_w, grid_h;
    };

    View make_view(const Scene& s, const bool mip, const int tile_x = 0, const int tile_y = 0, const int tile_w = 0,
                   const int tile_h = 0) {
        View v{tile_w > 0 ? tile_w : s.width, tile_h > 0 ? tile_h : s.height, s.fx, s.fy,
               static_cast<double>(s.cx) - tile_x, static_cast<double>(s.cy) - tile_y, mip, 0, 0, 0};
        v.grid_w = static_cast<uint32_t>((v.width + 15) / 16);
        v.grid_h = static_cast<uint32_t>((v.height + 15) / 16);
        const uint32_t tiles = v.grid_w * v.grid_h;
        const int tile_bits = tiles <= 1 ? 0 : std::bit_width(tiles - 1);
        v.depth_bits = static_cast<uint32_t>(std::clamp(32 - tile_bits, 0, 23));
        return v;
    }

    uint32_t depth_key(const double depth, const uint32_t bits) {
        float normalized = (2.0f * static_cast<float>(depth) + 1.0f) / (static_cast<float>(depth) + 1.0f);
        normalized = std::min(std::max(normalized, 1.0f), std::bit_cast<float>(0x3fffffffu));
        return (std::bit_cast<uint32_t>(normalized) & 0x7fffffu) >> (23u - bits);
    }

    void sh_basis(const double x, const double y, const double z, double* b) {
        const double xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z;
        b[0] = -0.48860251190291987 * y;
        b[1] = 0.48860251190291987 * z;
        b[2] = -0.48860251190291987 * x;
        b[3] = 1.0925484305920792 * xy;
        b[4] = -1.0925484305920792 * yz;
        b[5] = 0.94617469575755997 * zz - 0.31539156525251999;
        b[6] = -1.0925484305920792 * xz;
        b[7] = 0.54627421529603959 * xx - 0.54627421529603959 * yy;
        b[8] = 0.59004358992664352 * y * (-3.0 * xx + yy);
        b[9] = 2.8906114426405538 * xy * z;
        b[10] = 0.45704579946446572 * y * (1.0 - 5.0 * zz);
        b[11] = 0.3731763325901154 * z * (5.0 * zz - 3.0);
        b[12] = 0.45704579946446572 * x * (1.0 - 5.0 * zz);
        b[13] = 1.4453057213202769 * z * (xx - yy);
        b[14] = 0.59004358992664352 * x * (-xx + 3.0 * yy);
    }

    // The fastgs preprocess in double; `cull` applies the visibility tests.
    Splat project(const Scene& s, const Params& p, const View& v, const int i, const bool cull) {
        Splat out;
        const auto& m = s.view;
        const double* mean = &p.means[size_t{3} * i];
        auto row = [&](const int r) { return m[r * 4] * mean[0] + m[r * 4 + 1] * mean[1] + m[r * 4 + 2] * mean[2] + m[r * 4 + 3]; };
        const double depth = row(2);
        const double opacity = 1.0 / (1.0 + std::exp(-p.opacities[i]));
        const double* q = &p.rotations[size_t{4} * i];
        const double qn = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        if (cull && (depth < 0.01 || depth > 1e10 || opacity < kMinAlpha || qn < 1e-8))
            return out;
        const double inv = 2.0 / std::max(qn, 1e-8);
        const double qr = q[0], qx = q[1], qy = q[2], qz = q[3];
        const double rot[3][3] = {
            {1 - (qy * qy + qz * qz) * inv, (qx * qy - qr * qz) * inv, (qr * qy + qx * qz) * inv},
            {(qr * qz + qx * qy) * inv, 1 - (qx * qx + qz * qz) * inv, (qy * qz - qr * qx) * inv},
            {(qx * qz - qr * qy) * inv, (qr * qx + qy * qz) * inv, 1 - (qx * qx + qy * qy) * inv}};
        double var[3];
        for (int k = 0; k < 3; ++k)
            var[k] = std::exp(2.0 * std::min(p.scales[size_t{3} * i + k], 20.0));
        double cov[3][3];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                cov[a][b] = rot[a][0] * var[0] * rot[b][0] + rot[a][1] * var[1] * rot[b][1] + rot[a][2] * var[2] * rot[b][2];
        const double x = row(0) / depth, y = row(1) / depth;
        const double clip_l = (-0.15 * v.width - v.cx) / v.fx, clip_r = (1.15 * v.width - v.cx) / v.fx;
        const double clip_t = (-0.15 * v.height - v.cy) / v.fy, clip_b = (1.15 * v.height - v.cy) / v.fy;
        const double tx = std::clamp(x, clip_l, clip_r), ty = std::clamp(y, clip_t, clip_b);
        double jw[2][3];
        for (int c = 0; c < 3; ++c) {
            jw[0][c] = v.fx / depth * m[c] - v.fx / depth * tx * m[8 + c];
            jw[1][c] = v.fy / depth * m[4 + c] - v.fy / depth * ty * m[8 + c];
        }
        double cov2d[3] = {0, 0, 0};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                cov2d[0] += jw[0][a] * cov[a][b] * jw[0][b];
                cov2d[1] += jw[0][a] * cov[a][b] * jw[1][b];
                cov2d[2] += jw[1][a] * cov[a][b] * jw[1][b];
            }
        const double det_raw = std::max(cov2d[0] * cov2d[2] - cov2d[1] * cov2d[1], 0.0);
        const double kernel = v.mip ? 0.1 : 0.3;
        cov2d[0] += kernel;
        cov2d[2] += kernel;
        const double det = cov2d[0] * cov2d[2] - cov2d[1] * cov2d[1];
        if (cull && det < 1e-6)
            return out;
        out.compensation = std::sqrt(det_raw / det);
        if (!p.compensation.empty())
            out.compensation = p.compensation[i];
        out.opacity = v.mip ? opacity * out.compensation : opacity;
        if (cull && out.opacity < kMinAlpha)
            return out;
        out.conic[0] = cov2d[2] / det;
        out.conic[1] = -cov2d[1] / det;
        out.conic[2] = cov2d[0] / det;
        out.mean[0] = x * v.fx + v.cx + p.offset2d[size_t{2} * i];
        out.mean[1] = y * v.fy + v.cy + p.offset2d[size_t{2} * i + 1];
        const double factor = std::sqrt(2.0 * std::log(out.opacity * 255.0));
        out.extent[0] = factor * std::sqrt(cov2d[0]);
        out.extent[1] = factor * std::sqrt(cov2d[2]);
        if (cull) {
            const double ex = std::max(out.extent[0] - 0.5, 0.0), ey = std::max(out.extent[1] - 0.5, 0.0);
            const int x0 = std::max(0, static_cast<int>(std::ceil((out.mean[0] - ex) / 16)) - 1);
            const int x1 = std::max(0, static_cast<int>(std::floor((out.mean[0] + ex) / 16)) + 1);
            const int y0 = std::max(0, static_cast<int>(std::ceil((out.mean[1] - ey) / 16)) - 1);
            const int y1 = std::max(0, static_cast<int>(std::floor((out.mean[1] + ey) / 16)) + 1);
            if (std::min<int>(x1, v.grid_w) <= std::min<int>(x0, v.grid_w) ||
                std::min<int>(y1, v.grid_h) <= std::min<int>(y0, v.grid_h))
                return out;
        }
        out.visible = true;
        out.depth = depth;
        out.key = depth_key(depth, v.depth_bits);
        const double dir_raw[3] = {mean[0] - s.camera[0], mean[1] - s.camera[1], mean[2] - s.camera[2]};
        const double norm = std::sqrt(dir_raw[0] * dir_raw[0] + dir_raw[1] * dir_raw[1] + dir_raw[2] * dir_raw[2]);
        double basis[15];
        sh_basis(dir_raw[0] / norm, dir_raw[1] / norm, dir_raw[2] / norm, basis);
        for (int c = 0; c < 3; ++c) {
            out.color[c] = 0.5 + 0.28209479177387814 * p.sh0[size_t{3} * i + c];
            for (uint32_t k = 0; k < rest_count(s.degree); ++k)
                out.color[c] += basis[k] * p.rest[(static_cast<size_t>(i) * 15 + k) * 3 + c];
        }
        const int axis = (var[0] <= var[1] && var[0] <= var[2]) ? 0 : (var[1] <= var[2] ? 1 : 2);
        double world[3] = {rot[0][axis], rot[1][axis], rot[2][axis]};
        if (world[0] * dir_raw[0] + world[1] * dir_raw[1] + world[2] * dir_raw[2] > 0)
            for (double& w : world)
                w = -w;
        for (int r = 0; r < 3; ++r)
            out.normal[r] = m[r * 4] * world[0] + m[r * 4 + 1] * world[1] + m[r * 4 + 2] * world[2];
        return out;
    }

    struct Rendered {
        std::vector<double> image, alpha, depth, normal;
        std::vector<std::vector<int>> lists;                   // per pixel, front to back
        std::vector<double> weight, weight_error, weight_edge; // per splat
    };

    // Front-to-back blend over splats ordered by (depth key, index). With
    // `lists`, only the listed splats blend, with no alpha threshold or early stop.
    Rendered render(const Scene& s, const Params& p, const View& v, const std::vector<std::vector<int>>* lists,
                    const std::vector<float>* error = nullptr, const std::vector<float>* edge = nullptr) {
        std::vector<Splat> splats(s.count);
        std::vector<int> order;
        for (int i = 0; i < s.count; ++i) {
            splats[i] = project(s, p, v, i, lists == nullptr);
            if (splats[i].visible)
                order.push_back(i);
        }
        std::stable_sort(order.begin(), order.end(), [&](const int a, const int b) { return splats[a].key < splats[b].key; });
        const size_t pixels = size_t{static_cast<size_t>(v.width)} * v.height;
        Rendered r;
        r.image.assign(3 * pixels, 0.0);
        r.alpha.assign(pixels, 0.0);
        r.depth.assign(pixels, 0.0);
        r.normal.assign(3 * pixels, 0.0);
        r.lists.resize(pixels);
        r.weight.assign(s.count, 0.0);
        r.weight_error.assign(s.count, 0.0);
        r.weight_edge.assign(s.count, 0.0);
        for (int py = 0; py < v.height; ++py) {
            for (int px = 0; px < v.width; ++px) {
                const size_t pixel = size_t{static_cast<size_t>(py)} * v.width + px;
                double color[3] = {0, 0, 0}, depth = 0, normal[3] = {0, 0, 0}, transmittance = 1.0;
                auto blend = [&](const int i, const bool gate) {
                    const Splat& g = splats[i];
                    const double dx = g.mean[0] - (px + 0.5), dy = g.mean[1] - (py + 0.5);
                    const double sigma = 0.5 * (g.conic[0] * dx * dx + g.conic[2] * dy * dy) + g.conic[1] * dx * dy;
                    if (sigma < 0)
                        return false;
                    const double alpha = std::min(g.opacity * std::exp(-sigma), kMaxAlpha);
                    if (gate && alpha < kMinAlpha)
                        return false;
                    const double w = transmittance * alpha;
                    for (int c = 0; c < 3; ++c) {
                        color[c] += w * std::clamp(g.color[c], 0.0, 4.0);
                        normal[c] += w * g.normal[c];
                    }
                    depth += w * g.depth;
                    r.weight[i] += w;
                    if (error != nullptr)
                        r.weight_error[i] += w * (*error)[pixel];
                    if (edge != nullptr)
                        r.weight_edge[i] += w * (*edge)[pixel];
                    transmittance *= 1.0 - alpha;
                    return true;
                };
                if (lists != nullptr) {
                    for (const int i : (*lists)[pixel])
                        blend(i, false);
                } else {
                    for (const int i : order) {
                        const Splat& g = splats[i];
                        if (std::abs(g.mean[0] - (px + 0.5)) > g.extent[0] + 1 ||
                            std::abs(g.mean[1] - (py + 0.5)) > g.extent[1] + 1)
                            continue;
                        if (blend(i, true)) {
                            r.lists[pixel].push_back(i);
                            if (transmittance < kTransmittanceMin)
                                break;
                        }
                    }
                }
                double bg[3];
                for (int c = 0; c < 3; ++c)
                    bg[c] = s.bg_image.empty() ? s.bg_color[c] : s.bg_image[c * pixels + pixel];
                for (int c = 0; c < 3; ++c) {
                    r.image[c * pixels + pixel] = color[c] + transmittance * bg[c];
                    r.normal[c * pixels + pixel] = normal[c];
                }
                r.alpha[pixel] = 1.0 - transmittance;
                r.depth[pixel] = depth;
            }
        }
        return r;
    }

    struct LossWeights {
        std::vector<float> image, alpha, depth, normal;
    };

    LossWeights random_weights(const int width, const int height, const unsigned seed, const bool extras) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
        const size_t pixels = size_t{static_cast<size_t>(width)} * height;
        LossWeights w;
        w.image.resize(3 * pixels);
        for (auto& v : w.image)
            v = uniform(rng);
        if (extras) {
            w.alpha.resize(pixels);
            w.depth.resize(pixels);
            w.normal.resize(3 * pixels);
            for (auto& v : w.alpha)
                v = 0.5f * uniform(rng);
            for (auto& v : w.depth)
                v = 0.1f * uniform(rng);
            for (auto& v : w.normal)
                v = 0.5f * uniform(rng);
        }
        return w;
    }

    double loss(const Rendered& r, const LossWeights& w) {
        double total = 0.0;
        for (size_t i = 0; i < r.image.size(); ++i)
            total += w.image[i] * r.image[i];
        for (size_t i = 0; i < w.alpha.size(); ++i)
            total += w.alpha[i] * r.alpha[i] + w.depth[i] * r.depth[i];
        for (size_t i = 0; i < w.normal.size(); ++i)
            total += w.normal[i] * r.normal[i];
        return total;
    }

    Tensor upload(const std::vector<float>& values, lfs::core::TensorShape shape) {
        return Tensor::from_blob(const_cast<float*>(values.data()), std::move(shape), Device::CPU, DataType::Float32)
            .to(Device::GPU);
    }

    std::vector<float> download(const Tensor& tensor) { return tensor.to(Device::CPU).to_vector(); }

    std::vector<uint8_t> download_bytes(const Tensor& tensor) {
        const Tensor host = tensor.to(Device::CPU);
        const auto* bytes = static_cast<const uint8_t*>(host.data_ptr());
        return {bytes, bytes + host.bytes()};
    }

    // Scene tensors on the Metal backend. Q16 storage replaces the scene's rest
    // coefficients with their decoded values so the reference sees the same inputs.
    struct Gpu {
        Tensor means, scales, rotations, opacities, sh0, shN, bounds, view, camera, bg_color, bg_image;
        ops::ShStorage storage = ops::ShStorage::Float32;
        uint32_t layout_rest = 0, cells = 0;
    };

    Gpu upload_scene(Scene& s, const ops::ShStorage storage) {
        const auto n = static_cast<size_t>(s.count);
        Gpu g;
        g.storage = storage;
        g.layout_rest = rest_count(s.degree);
        g.means = upload(s.means, {n, 3});
        g.scales = upload(s.scales, {n, 3});
        g.rotations = upload(s.rotations, {n, 4});
        g.opacities = upload(s.opacities, {n, 1});
        g.sh0 = upload(s.sh0, {n, 1, 3});
        g.view = upload(std::vector<float>(s.view.begin(), s.view.end()), {1, 4, 4});
        g.camera = upload(std::vector<float>(s.camera.begin(), s.camera.end()), {3});
        g.bg_color = upload(s.bg_color, {3});
        if (!s.bg_image.empty())
            g.bg_image = upload(s.bg_image, {3, static_cast<size_t>(s.height), static_cast<size_t>(s.width)});
        if (g.layout_rest == 0)
            return g;
        if (storage != ops::ShStorage::Q16) {
            std::vector<float> swizzled(padded(n) * float4_slots(g.layout_rest) * 4, 0.0f);
            for (uint32_t p = 0; p < n; ++p)
                for (uint32_t k = 0; k < g.layout_rest; ++k)
                    for (uint32_t c = 0; c < 3; ++c)
                        swizzled[swizzled_float_index(p, k, c, g.layout_rest)] = s.rest[(size_t{p} * 15 + k) * 3 + c];
            g.shN = upload(swizzled, {swizzled.size()});
            if (storage == ops::ShStorage::IeeeFloat16) {
                g.shN = g.shN.to(DataType::Float16);
                const std::vector<float> rounded = download(g.shN.to(DataType::Float32));
                for (uint32_t p = 0; p < n; ++p)
                    for (uint32_t k = 0; k < g.layout_rest; ++k)
                        for (uint32_t c = 0; c < 3; ++c)
                            s.rest[(size_t{p} * 15 + k) * 3 + c] = rounded[swizzled_float_index(p, k, c, g.layout_rest)];
            }
            return g;
        }
        g.cells = g.layout_rest * 3;
        const size_t blocks = (n + 255) / 256;
        std::vector<uint16_t> codes(padded(n) * g.cells, 0);
        std::vector<float> bounds(blocks * 2, 0.0f);
        for (size_t b = 0; b < blocks; ++b) {
            float lo = 1e30f, hi = -1e30f;
            for (size_t p = b * 256; p < std::min(n, (b + 1) * 256); ++p)
                for (uint32_t cell = 0; cell < g.cells; ++cell) {
                    lo = std::min(lo, s.rest[p * 45 + cell]);
                    hi = std::max(hi, s.rest[p * 45 + cell]);
                }
            bounds[b * 2] = lo;
            bounds[b * 2 + 1] = hi;
            for (size_t p = b * 256; p < std::min(n, (b + 1) * 256); ++p)
                for (uint32_t cell = 0; cell < g.cells; ++cell) {
                    float& value = s.rest[p * 45 + cell];
                    const float q = std::clamp(std::round(65535.0f * (value - lo) / std::max(hi - lo, 1e-20f)), 0.0f, 65535.0f);
                    codes[q16_index(static_cast<uint32_t>(p), cell, g.cells)] = static_cast<uint16_t>(q);
                    value = lo + (hi - lo) * (q * (1.0f / 65535.0f));
                }
        }
        g.shN = Tensor::from_blob(codes.data(), {codes.size()}, Device::CPU, DataType::Float16).to(Device::GPU);
        g.bounds = upload(bounds, {blocks, 2});
        return g;
    }

    struct Config {
        bool mip = false;
        bool normals = false;
        int tile_x = 0, tile_y = 0, tile_w = 0, tile_h = 0;
        uint32_t active_bases = 0; // 0: the scene's degree
    };

    struct Frame {
        ops::FastSaved saved;
        Tensor image, alpha, depth, normal, share;
        ops::RasterResult result;
    };

    const ops::FastRasterOps& fast_ops() { return *lfs::training::training_ops(GpuBackend::Metal).fast; }

    void forward(Frame& frame, const Scene& s, const Gpu& g, const Config& c) {
        const auto& table = fast_ops();
        if (!frame.saved.backend)
            frame.saved.backend = table.create();
        const uint32_t layout = g.layout_rest + 1;
        const ops::FastParams params{
            .full_image = {s.height, s.width},
            .intrinsics = {s.fx, s.fy, s.cx, s.cy},
            .tile_x = c.tile_x,
            .tile_y = c.tile_y,
            .tile_w = c.tile_w,
            .tile_h = c.tile_h,
            .sh = {.storage = g.storage, .active_bases = c.active_bases > 0 ? c.active_bases : layout, .layout_bases = layout},
            .mip_filter = c.mip,
            .render_normal = c.normals,
            .render_depth = true,
        };
        frame.share = Tensor::zeros({static_cast<size_t>(s.count)}, Device::GPU);
        frame.result = table.forward(frame.saved,
                                     {.means = g.means, .raw_scales = g.scales, .raw_rotations = g.rotations, .raw_opacities = g.opacities, .sh0 = g.sh0, .shN = g.shN, .sh_value_bounds = g.bounds},
                                     g.view, g.camera, g.bg_color, g.bg_image, params,
                                     {.image = frame.image, .alpha = frame.alpha, .depth = frame.depth, .normal = frame.normal},
                                     frame.share);
    }

    struct AdamSetup {
        float beta1 = 0.0f, beta2 = 0.999f, eps = 1.0f, step = 1.0f, bc2_sqrt_rcp = 0.0f;
    };

    // Zeroed moments and bounds for all six groups.
    // Zeroed moments and bounds for all six groups, plus the optional masks
    // and regularizer tensors (absent unless a test sets them).
    struct AdamState {
        std::array<Tensor, 6> moments, bounds;
        Tensor none, frozen, crop, screen_share, scale_loss, opacity_loss, sparsity_sigmoid, sparsity_z, sparsity_u,
            far_mask;
        float frozen_lr_scale = 0.0f, cropbox_lr_scale = 1.0f, screen_share_limit = 0.0f, screen_share_penalty = 0.0f;
    };

    ops::BackwardAdam make_adam(Gpu& g, AdamState& state, const int count, const AdamSetup& setup) {
        const auto n = static_cast<size_t>(count);
        const size_t blocks = (n + 255) / 256;
        std::array<Tensor*, 6> params{&g.means, &g.scales, &g.rotations, &g.opacities, &g.sh0, &g.shN};
        const std::array<int, 6> attributes{3, 3, 4, 1, 3, static_cast<int>(float4_slots(g.layout_rest) * 4)};
        std::array<size_t, 6> cells{};
        for (size_t i = 0; i < 5; ++i)
            cells[i] = n * attributes[i];
        cells[5] = padded(n) * float4_slots(g.layout_rest) * 4;
        for (size_t i = 0; i < 6; ++i) {
            const size_t bytes = cells[i] * (i == 5 ? 2 : 4);
            state.moments[i] = bytes > 0 ? Tensor::zeros({bytes}, Device::GPU, DataType::UInt8) : Tensor();
            state.bounds[i] = Tensor::zeros({blocks, 4}, Device::GPU);
        }
        auto group = [&](const size_t i) {
            const bool q16 = i == 5 && g.storage == ops::ShStorage::Q16;
            return ops::BackwardAdamParam{
                .parameter = *params[i],
                .packed_moments = state.moments[i],
                .joint_bounds = state.bounds[i],
                .sh_value_bounds = q16 ? g.bounds : state.none,
                .frozen_mask = state.frozen,
                .crop_damping_mask = state.crop,
                .screen_share = i == 1 ? state.screen_share : state.none,
                .joint_bits = i == 5 ? 8 : 16,
                .value_bits = i == 5 && g.storage != ops::ShStorage::Float32 ? 16 : 0,
                .value_cells = q16 ? static_cast<int>(g.cells) : 0,
                .primitives = count,
                .elements = static_cast<int>(params[i]->numel()),
                .attributes = attributes[i],
                .step_size = setup.step,
                .bc2_sqrt_rcp = setup.bc2_sqrt_rcp,
                .frozen_lr_scale = state.frozen_lr_scale,
                .cropbox_lr_scale = state.cropbox_lr_scale,
                .screen_share_limit = i == 1 ? state.screen_share_limit : 0.0f,
                .screen_share_penalty = i == 1 ? state.screen_share_penalty : 0.0f,
                .enabled = i != 5 || g.layout_rest > 0,
            };
        };
        return {.groups = {group(0), group(1), group(2), group(3), group(4), group(5)},
                .scale_reg_loss = state.scale_loss,
                .opacity_reg_loss = state.opacity_loss,
                .sparsity_sigmoid = state.sparsity_sigmoid,
                .sparsity_z = state.sparsity_z,
                .sparsity_u = state.sparsity_u,
                .far_mask = state.far_mask,
                .beta1 = setup.beta1,
                .beta2 = setup.beta2,
                .eps = setup.eps};
    }

    struct HostParams {
        std::vector<float> means, scales, rotations, opacities, sh0, shN;
    };

    HostParams read_params(const Gpu& g) {
        return {download(g.means), download(g.scales), download(g.rotations), download(g.opacities), download(g.sh0),
                g.shN.is_valid() && g.storage != ops::ShStorage::Q16 ? download(g.shN.to(DataType::Float32))
                                                                     : std::vector<float>{}};
    }

    Tensor plane(const std::vector<float>& values, const int channels, const int height, const int width) {
        return upload(values, {static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)});
    }

    // Every value within `loose`, and all but `outliers` of them within
    // `tolerance`: a splat whose alpha sits on the 1/255 cutoff may land on
    // either side in float and double.
    void expect_close(const std::vector<float>& actual, const std::vector<double>& expected, const double tolerance,
                      const std::string& name, const double outliers = 0.0, const double loose = 0.0) {
        ASSERT_EQ(actual.size(), expected.size()) << name;
        double worst = 0.0;
        size_t worst_index = 0, above = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const double diff = std::abs(actual[i] - expected[i]);
            above += diff > tolerance ? 1 : 0;
            if (!(diff <= worst)) {
                worst = diff;
                worst_index = i;
            }
        }
        EXPECT_LE(worst, std::max(tolerance, loose)) << name << " worst at " << worst_index << ": " << actual[worst_index]
                                                     << " vs " << expected[worst_index];
        EXPECT_LE(static_cast<double>(above), outliers * static_cast<double>(actual.size()))
            << name << ": " << above << " values beyond " << tolerance << ", worst at " << worst_index << ": "
            << actual[worst_index] << " vs " << expected[worst_index];
    }

    class MetalFastRaster : public ::testing::Test {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GpuBackend::Metal))
                GTEST_SKIP() << "Metal device unavailable";
            if (lfs::training::training_ops(GpuBackend::Metal).fast == nullptr)
                GTEST_SKIP() << "Metal Fast slot is empty";
            session_.emplace(GpuBackend::Metal);
            ASSERT_TRUE(session_->switched());
        }
        void TearDown() override { session_.reset(); }

    private:
        std::optional<lfs::test::DefaultGpuBackendForTesting> session_;
    };

    struct ForwardCase {
        const char* name;
        int count, degree, width, height;
        ops::ShStorage storage;
        bool bg_image;
        Config config;
    };

    void check_forward(const ForwardCase& fc) {
        SCOPED_TRACE(fc.name);
        Scene s = make_scene(fc.count, fc.degree, fc.width, fc.height, 11u + static_cast<unsigned>(fc.count));
        const Config& c = fc.config;
        // The background image covers the rendered tile.
        const int w = c.tile_w > 0 ? c.tile_w : s.width;
        const int h = c.tile_h > 0 ? c.tile_h : s.height;
        if (fc.bg_image)
            s.bg_image = random_image(w, h, 5u);
        Gpu g = upload_scene(s, fc.storage);
        if (fc.bg_image)
            g.bg_image = plane(s.bg_image, 3, h, w);
        Frame frame;
        forward(frame, s, g, c);
        ASSERT_EQ(frame.result.code, ops::RasterResult::Code::Success) << frame.result.message;
        EXPECT_TRUE(frame.result.has_work);

        Scene reference = s;
        if (c.active_bases > 0)
            reference.degree = static_cast<int>(std::lround(std::sqrt(static_cast<double>(c.active_bases)))) - 1;
        const Params p(reference);
        const View v = make_view(reference, c.mip, c.tile_x, c.tile_y, c.tile_w, c.tile_h);
        const Rendered r = render(reference, p, v, nullptr);
        constexpr double outliers = 1e-4;
        expect_close(download(frame.image), r.image, 2e-4, "image", outliers, 2e-2);
        expect_close(download(frame.alpha), r.alpha, 2e-4, "alpha", outliers, 2e-2);
        expect_close(download(frame.depth), r.depth, 1e-3, "depth", outliers, 1e-1);
        if (c.normals)
            expect_close(download(frame.normal), r.normal, 2e-4, "normal", outliers, 2e-2);
        else
            EXPECT_FALSE(frame.normal.is_valid());
        fast_ops().release(frame.saved);
    }

    TEST_F(MetalFastRaster, ForwardMatchesReference) {
        const ForwardCase cases[] = {
            {"sh0", 40, 0, 48, 40, ops::ShStorage::Float32, false, {}},
            {"sh3_float_bg_image_mip", 40, 3, 64, 48, ops::ShStorage::Float32, true, {.mip = true}},
            {"sh3_q16_normals", 40, 3, 64, 48, ops::ShStorage::Q16, false, {.normals = true}},
            {"sh1_of_layout_3", 32, 3, 32, 32, ops::ShStorage::Float32, false, {.active_bases = 4}},
            {"sh3_half", 40, 3, 48, 40, ops::ShStorage::IeeeFloat16, false, {}},
            {"sh2_q16_of_layout_3", 32, 3, 40, 36, ops::ShStorage::Q16, true, {.active_bases = 9}},
            {"tile", 40, 2, 64, 48, ops::ShStorage::Float32, true, {.tile_x = 16, .tile_y = 8, .tile_w = 40, .tile_h = 32}},
            {"many_blocks", 4000, 3, 320, 240, ops::ShStorage::Float32, false, {.normals = true}},
        };
        for (const auto& fc : cases)
            check_forward(fc);
    }

    // Zero moments and a zero image gradient leave every parameter bit-identical,
    // as the parity fixture expects.
    TEST_F(MetalFastRaster, ZeroGradientLeavesParametersUnchanged) {
        Scene s = make_scene(40, 3, 64, 48, 3u);
        Gpu g = upload_scene(s, ops::ShStorage::Q16);
        const HostParams before = read_params(g);
        const std::vector<uint8_t> codes_before = download_bytes(g.shN);
        Frame frame;
        forward(frame, s, g, {});
        ASSERT_EQ(frame.result.code, ops::RasterResult::Code::Success) << frame.result.message;
        AdamState state;
        const ops::BackwardAdam adam = make_adam(g, state, s.count, {.beta1 = 0.9f, .eps = 1e-15f, .step = 1e-2f, .bc2_sqrt_rcp = 1.0f / std::sqrt(0.001f)});
        const Tensor zero = Tensor::zeros_like(frame.image);
        Tensor none;
        fast_ops().backward(frame.saved, {.image = zero, .alpha = none, .depth = none, .normal = none}, none, none,
                            none, none, adam, DensificationType::None);
        const HostParams after = read_params(g);
        EXPECT_EQ(before.means, after.means);
        EXPECT_EQ(before.scales, after.scales);
        EXPECT_EQ(before.rotations, after.rotations);
        EXPECT_EQ(before.opacities, after.opacities);
        EXPECT_EQ(before.sh0, after.sh0);
        EXPECT_EQ(codes_before, download_bytes(g.shN));
    }

    // Reference gradient of every parameter by central differences.
    struct ReferenceGrads {
        std::vector<double> means, scales, rotations, opacities, sh0, rest, mean2d;
    };

    ReferenceGrads reference_grads(const Scene& s, const View& v, const LossWeights& w, const bool mean2d) {
        Params base(s);
        if (v.mip) {
            std::vector<double> compensation;
            for (int i = 0; i < s.count; ++i)
                compensation.push_back(project(s, base, v, i, false).compensation);
            base.compensation = std::move(compensation);
        }
        const Rendered frozen = render(s, base, v, nullptr);
        const auto lists = frozen.lists;
        auto derivative = [&](std::vector<double> Params::*member, const size_t index) {
            constexpr double h = 1e-5;
            Params p = base;
            (p.*member)[index] += h;
            const double up = loss(render(s, p, v, &lists), w);
            (p.*member)[index] -= 2 * h;
            const double down = loss(render(s, p, v, &lists), w);
            return (up - down) / (2 * h);
        };
        auto all = [&](std::vector<double> Params::*member, const size_t size) {
            std::vector<double> out(size);
            for (size_t i = 0; i < size; ++i)
                out[i] = derivative(member, i);
            return out;
        };
        const auto n = static_cast<size_t>(s.count);
        ReferenceGrads r;
        r.means = all(&Params::means, 3 * n);
        r.scales = all(&Params::scales, 3 * n);
        r.rotations = all(&Params::rotations, 4 * n);
        r.opacities = all(&Params::opacities, n);
        r.sh0 = all(&Params::sh0, 3 * n);
        r.rest.assign(n * 45, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t cell = 0; cell < rest_count(s.degree) * 3; ++cell)
                r.rest[i * 45 + cell] = derivative(&Params::rest, i * 45 + cell);
        if (mean2d)
            r.mean2d = all(&Params::offset2d, 2 * n);
        return r;
    }

    // Metal gradient (old - new under the unit step) against the reference,
    // relative to the largest reference gradient of the group.
    void expect_grads(const std::vector<float>& before, const std::vector<float>& after,
                      const std::vector<double>& expected, const double relative, const std::string& name) {
        ASSERT_EQ(before.size(), expected.size()) << name;
        double scale = 0.0;
        for (const double v : expected)
            scale = std::max(scale, std::abs(v));
        double worst = 0.0;
        size_t worst_index = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            const double metal = static_cast<double>(before[i]) - static_cast<double>(after[i]);
            const double diff = std::abs(metal - expected[i]);
            if (diff > worst) {
                worst = diff;
                worst_index = i;
            }
        }
        EXPECT_GT(scale, 0.0) << name;
        EXPECT_LE(worst, relative * scale + 1e-5)
            << name << " worst at " << worst_index << ": metal " << before[worst_index] - after[worst_index]
            << " reference " << expected[worst_index] << " (group scale " << scale << ")";
    }

    void check_gradients(const int degree, const bool extras, const ops::ShStorage storage = ops::ShStorage::Float32,
                         const bool mip = false) {
        SCOPED_TRACE(std::format("degree {} extras {} storage {} mip {}", degree, extras, static_cast<int>(storage), mip));
        Scene s = make_scene(24, degree, 48, 40, 21u + degree, -2.3f, -1.6f);
        const Config c{.mip = mip, .normals = extras};
        Gpu g = upload_scene(s, storage);
        const HostParams before = read_params(g);
        const LossWeights w = random_weights(s.width, s.height, 9u, extras);
        const ReferenceGrads expected = reference_grads(s, make_view(s, mip), w, false);

        Frame frame;
        forward(frame, s, g, c);
        ASSERT_EQ(frame.result.code, ops::RasterResult::Code::Success) << frame.result.message;
        AdamState state;
        const ops::BackwardAdam adam = make_adam(g, state, s.count, {});
        const Tensor grad_image = plane(w.image, 3, s.height, s.width);
        Tensor none;
        const Tensor grad_alpha = extras ? plane(w.alpha, 1, s.height, s.width) : none;
        const Tensor grad_depth = extras ? plane(w.depth, 1, s.height, s.width) : none;
        const Tensor grad_normal = extras ? plane(w.normal, 3, s.height, s.width) : none;
        fast_ops().backward(frame.saved, {.image = grad_image, .alpha = grad_alpha, .depth = grad_depth, .normal = grad_normal},
                            none, none, none, none, adam, DensificationType::None);
        const HostParams after = read_params(g);
        expect_grads(before.means, after.means, expected.means, 1e-4, "means");
        expect_grads(before.scales, after.scales, expected.scales, 1e-4, "scales");
        expect_grads(before.rotations, after.rotations, expected.rotations, 1e-4, "rotations");
        expect_grads(before.opacities, after.opacities, expected.opacities, 1e-4, "opacities");
        expect_grads(before.sh0, after.sh0, expected.sh0, 1e-4, "sh0");
        if (degree > 0) {
            std::vector<float> rest_before(s.count * 45), rest_after(s.count * 45);
            for (uint32_t p = 0; p < static_cast<uint32_t>(s.count); ++p)
                for (uint32_t cell = 0; cell < rest_count(degree) * 3; ++cell) {
                    const size_t at = swizzled_float_index(p, cell / 3, cell % 3, g.layout_rest);
                    rest_before[p * 45 + cell] = before.shN[at];
                    rest_after[p * 45 + cell] = after.shN[at];
                }
            // Half storage rounds the updated value to 11 significant bits.
            expect_grads(rest_before, rest_after, expected.rest,
                         storage == ops::ShStorage::IeeeFloat16 ? 2e-3 : 1e-4, "shN");
        }
    }

    TEST_F(MetalFastRaster, GradientsMatchFiniteDifferences) {
        check_gradients(0, false);
        check_gradients(3, false);
        check_gradients(2, true);
        check_gradients(3, false, ops::ShStorage::IeeeFloat16);
        check_gradients(1, true, ops::ShStorage::Float32, true);
    }

    // With the trainer's Adam settings a first step from zero moments moves every
    // gradient-carrying parameter by the learning rate against its gradient.
    TEST_F(MetalFastRaster, FirstAdamStepFollowsGradientSign) {
        Scene s = make_scene(24, 1, 48, 40, 33u, -2.3f, -1.6f);
        Gpu g = upload_scene(s, ops::ShStorage::Float32);
        const HostParams before = read_params(g);
        const LossWeights w = random_weights(s.width, s.height, 4u, false);
        const ReferenceGrads expected = reference_grads(s, make_view(s, false), w, false);
        Frame frame;
        forward(frame, s, g, {});
        ASSERT_EQ(frame.result.code, ops::RasterResult::Code::Success) << frame.result.message;
        constexpr float lr = 1e-3f;
        AdamState state;
        const ops::BackwardAdam adam = make_adam(
            g, state, s.count, {.beta1 = 0.9f, .eps = 1e-15f, .step = lr / 0.1f, .bc2_sqrt_rcp = 1.0f / std::sqrt(0.001f)});
        const Tensor grad_image = plane(w.image, 3, s.height, s.width);
        Tensor none;
        fast_ops().backward(frame.saved, {.image = grad_image, .alpha = none, .depth = none, .normal = none}, none,
                            none, none, none, adam, DensificationType::None);
        const HostParams after = read_params(g);
        auto check = [&](const std::vector<float>& b, const std::vector<float>& a, const std::vector<double>& grad,
                         const char* name) {
            double scale = 0.0;
            for (const double v : grad)
                scale = std::max(scale, std::abs(v));
            int moved = 0;
            for (size_t i = 0; i < grad.size(); ++i) {
                const double step = static_cast<double>(b[i]) - a[i];
                if (std::abs(grad[i]) > 1e-2 * scale) {
                    EXPECT_NEAR(step, grad[i] > 0 ? lr : -lr, 1e-5) << name << " " << i << " grad " << grad[i];
                    ++moved;
                } else if (grad[i] == 0.0) {
                    EXPECT_EQ(step, 0.0) << name << " " << i;
                }
            }
            EXPECT_GT(moved, 0) << name;
        };
        check(before.means, after.means, expected.means, "means");
        check(before.scales, after.scales, expected.scales, "scales");
        check(before.rotations, after.rotations, expected.rotations, "rotations");
        check(before.opacities, after.opacities, expected.opacities, "opacities");
        check(before.sh0, after.sh0, expected.sh0, "sh0");
    }

    TEST_F(MetalFastRaster, DensificationMatchesReference) {
        Scene s = make_scene(30, 1, 48, 40, 44u, -2.3f, -1.6f);
        // Keep every splat well inside the image so visibility is unambiguous.
        for (int i = 0; i < s.count; ++i) {
            s.means[i * 3] *= 0.6f;
            s.means[i * 3 + 1] *= 0.6f;
        }
        const size_t pixels = size_t{static_cast<size_t>(s.width)} * s.height;
        std::mt19937 rng(8u);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        std::vector<float> error(pixels), edge(pixels);
        for (auto& v : error)
            v = uniform(rng);
        for (auto& v : edge)
            v = uniform(rng);
        const LossWeights w = random_weights(s.width, s.height, 12u, false);
        const View view = make_view(s, false);
        const Params base(s);
        const Rendered r = render(s, base, view, nullptr, &error, &edge);
        const ReferenceGrads expected = reference_grads(s, view, w, true);
        const auto n = static_cast<size_t>(s.count);

        struct Mode {
            DensificationType type;
            bool error_map;
        };
        for (const Mode mode : {Mode{DensificationType::MRNF, true}, Mode{DensificationType::MRNF, false},
                                Mode{DensificationType::MCMC, true}, Mode{DensificationType::None, false}}) {
            SCOPED_TRACE(std::format("type {} error map {}", static_cast<int>(mode.type), mode.error_map));
            Gpu g = upload_scene(s, ops::ShStorage::Float32);
            Frame frame;
            forward(frame, s, g, {});
            ASSERT_EQ(frame.result.code, ops::RasterResult::Code::Success) << frame.result.message;
            AdamState state;
            const ops::BackwardAdam adam = make_adam(g, state, s.count, {});
            Tensor densification = Tensor::zeros({2, n}, Device::GPU);
            Tensor edge_scores = Tensor::zeros({n}, Device::GPU);
            const Tensor error_map = upload(error, {static_cast<size_t>(s.height), static_cast<size_t>(s.width)});
            const Tensor edge_map = upload(edge, {static_cast<size_t>(s.height), static_cast<size_t>(s.width)});
            const Tensor grad_image = plane(w.image, 3, s.height, s.width);
            Tensor none;
            fast_ops().backward(frame.saved, {.image = grad_image, .alpha = none, .depth = none, .normal = none},
                                densification, mode.error_map ? error_map : none, edge_map, edge_scores, adam, mode.type);
            const std::vector<float> dens = download(densification);
            std::vector<double> row0(n), row1(n);
            for (size_t i = 0; i < n; ++i) {
                if (mode.type == DensificationType::None) {
                    const bool visible = project(s, base, view, static_cast<int>(i), true).visible;
                    row0[i] = visible ? 1.0 : 0.0;
                    row1[i] = visible ? std::hypot(expected.mean2d[2 * i] * 0.5 * s.width,
                                                   expected.mean2d[2 * i + 1] * 0.5 * s.height)
                                      : 0.0;
                } else {
                    row0[i] = r.weight[i];
                    row1[i] = mode.error_map ? r.weight_error[i] : r.weight[i];
                }
            }
            expect_close(std::vector<float>(dens.begin(), dens.begin() + n), row0, 1e-4, "densification row 0");
            double scale = 1.0;
            for (const double v : row1)
                scale = std::max(scale, std::abs(v));
            expect_close(std::vector<float>(dens.begin() + n, dens.end()), row1, 2e-3 * scale, "densification row 1");
            expect_close(download(edge_scores), r.weight_edge, 1e-4, "edge scores");
        }
    }

    // Q16 shN: the fused step moves the decoded values by -grad and re-encodes them
    // under the new block bounds.
    TEST_F(MetalFastRaster, Q16ShRestFollowsReferenceGradient) {
        Scene s = make_scene(24, 2, 48, 40, 51u, -2.3f, -1.6f);
        Gpu g = upload_scene(s, ops::ShStorage::Q16);
        const LossWeights w = random_weights(s.width, s.height, 13u, false);
        const ReferenceGrads expected = reference_grads(s, make_view(s, false), w, false);
        Frame frame;
        forward(frame, s, g, {});
        ASSERT_EQ(frame.result.code, ops::RasterResult::Code::Success) << frame.result.message;
        AdamState state;
        const ops::BackwardAdam adam = make_adam(g, state, s.count, {});
        const Tensor grad_image = plane(w.image, 3, s.height, s.width);
        Tensor none;
        fast_ops().backward(frame.saved, {.image = grad_image, .alpha = none, .depth = none, .normal = none}, none,
                            none, none, none, adam, DensificationType::None);
        const std::vector<float> bounds = download(g.bounds);
        const std::vector<uint8_t> raw = download_bytes(g.shN);
        const auto* codes = reinterpret_cast<const uint16_t*>(raw.data());
        const float lo = bounds[0], hi = bounds[1];
        double worst = 0.0;
        for (uint32_t p = 0; p < static_cast<uint32_t>(s.count); ++p)
            for (uint32_t cell = 0; cell < g.cells; ++cell) {
                const double decoded = lo + (hi - lo) * (codes[q16_index(p, cell, g.cells)] / 65535.0);
                const double wanted = s.rest[p * 45 + cell] - expected.rest[p * 45 + cell];
                worst = std::max(worst, std::abs(decoded - wanted));
            }
        EXPECT_LE(worst, 1e-3 * (hi - lo) + 1e-4) << "bounds " << lo << ".." << hi;
    }

    Tensor upload_mask(const std::vector<uint8_t>& mask) {
        return Tensor::from_blob(const_cast<uint8_t*>(mask.data()), {mask.size()}, Device::CPU, DataType::Bool)
            .to(Device::GPU);
    }

    double sigmoid(const double x) { return 1.0 / (1.0 + std::exp(-x)); }

    // Masks, regularizers, sparsity, the per-splat mean step and the
    // screen-share hinge on top of the render gradient, under the unit step.
    TEST_F(MetalFastRaster, FusedAdamTermsMatchReference) {
        Scene s = make_scene(24, 1, 48, 40, 61u, -2.3f, -1.6f);
        const auto n = static_cast<size_t>(s.count);
        Gpu g = upload_scene(s, ops::ShStorage::Float32);
        const HostParams before = read_params(g);
        const LossWeights w = random_weights(s.width, s.height, 14u, false);
        const View view = make_view(s, false);
        const ReferenceGrads render = reference_grads(s, view, w, false);

        std::vector<uint8_t> frozen(n, 0), crop(n, 0), far(n, 0);
        std::vector<float> sparsity(n), z(n), u(n);
        std::mt19937 rng(3u);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        for (size_t i = 0; i < n; ++i) {
            frozen[i] = i % 7 == 3;
            crop[i] = i % 5 == 1;
            far[i] = i % 3 == 0;
            sparsity[i] = uniform(rng);
            z[i] = uniform(rng);
            u[i] = uniform(rng) - 0.5f;
        }
        constexpr float scale_w = 0.3f, flatten_w = 0.2f, opacity_w = 0.4f, rho = 0.7f, grad_loss = 0.5f;
        constexpr float median = 0.1f, r_min = 1.0f, r_max = 300.0f, limit = 0.01f, penalty = 0.5f;
        AdamState state;
        state.frozen = upload_mask(frozen);
        state.crop = upload_mask(crop);
        state.far_mask = upload_mask(far);
        state.frozen_lr_scale = 0.0f;
        state.cropbox_lr_scale = 0.5f;
        state.scale_loss = Tensor::zeros({1}, Device::GPU);
        state.opacity_loss = Tensor::zeros({1}, Device::GPU);
        state.sparsity_sigmoid = upload(sparsity, {n});
        state.sparsity_z = upload(z, {n});
        state.sparsity_u = upload(u, {n});
        state.screen_share_limit = limit;
        state.screen_share_penalty = penalty;

        Frame frame;
        forward(frame, s, g, {});
        ASSERT_EQ(frame.result.code, ops::RasterResult::Code::Success) << frame.result.message;
        const std::vector<float> share = download(frame.share);
        state.screen_share = frame.share;
        ops::BackwardAdam adam = make_adam(g, state, s.count, {});
        adam.scale_reg_weight = scale_w;
        adam.flatten_reg_weight = flatten_w;
        adam.opacity_reg_weight = opacity_w;
        adam.sparsity_rho = rho;
        adam.sparsity_grad_loss = grad_loss;
        adam.per_splat_mean_step = true;
        adam.median_extent = median;
        adam.r_min = r_min;
        adam.r_max = r_max;
        const Tensor grad_image = plane(w.image, 3, s.height, s.width);
        Tensor none;
        fast_ops().backward(frame.saved, {.image = grad_image, .alpha = none, .depth = none, .normal = none}, none,
                            none, none, none, adam, DensificationType::None);
        const HostParams after = read_params(g);

        const Params base(s);
        std::vector<double> means(3 * n), scales(3 * n), opacities(n);
        double scale_loss = 0.0, opacity_loss = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double row = frozen[i] ? 0.0 : (crop[i] ? 0.5 : 1.0);
            const double* sc = &base.scales[3 * i];
            const double ratio =
                far[i] ? std::clamp(std::exp((sc[0] + sc[1] + sc[2]) / 3.0) / median, double{r_min}, double{r_max}) : 1.0;
            const int axis = (sc[0] <= sc[1] && sc[0] <= sc[2]) ? 0 : (sc[1] <= sc[2] ? 1 : 2);
            const Splat splat = project(s, base, view, static_cast<int>(i), true);
            double expected_share = 0.0;
            if (splat.visible) {
                const double* m = &base.means[3 * i];
                const double r = std::exp(std::max({sc[0], sc[1], sc[2]})) *
                                 std::sqrt(2.0 * std::log(std::max(255.0 * sigmoid(base.opacities[i]), 1.0)));
                const double d = std::hypot(m[0] - s.camera[0], m[1] - s.camera[1], m[2] - s.camera[2]);
                expected_share = std::clamp(r / (std::max(d, r) + std::sqrt(std::max(d * d - r * r, 0.0))), 0.0, 1.0);
            }
            EXPECT_NEAR(share[i], expected_share, 1e-5) << "screen share " << i;
            const double hinge = share[i] > limit ? penalty * std::log2(share[i] / limit) : 0.0;
            for (int k = 0; k < 3; ++k) {
                means[3 * i + k] = row * ratio * render.means[3 * i + k];
                double grad = render.scales[3 * i + k] + scale_w * std::exp(sc[k]) / (3.0 * n) + hinge;
                if (k == axis)
                    grad += 3.0 * flatten_w * std::exp(sc[k]) / (3.0 * n);
                scales[3 * i + k] = row * grad;
                scale_loss += scale_w / (3.0 * n) * std::exp(sc[k]);
            }
            const double o = sigmoid(base.opacities[i]);
            opacities[i] = row * (render.opacities[i] + opacity_w * o * (1.0 - o) / n +
                                  rho * (sparsity[i] - z[i] + u[i]) * sparsity[i] * (1.0 - sparsity[i]) * grad_loss);
            opacity_loss += opacity_w * o / n;
        }
        expect_grads(before.means, after.means, means, 1e-4, "means");
        expect_grads(before.scales, after.scales, scales, 1e-4, "scales");
        expect_grads(before.opacities, after.opacities, opacities, 1e-4, "opacities");
        for (size_t i = 0; i < n; ++i)
            if (frozen[i])
                EXPECT_EQ(before.rotations[4 * i], after.rotations[4 * i]) << "frozen rotation " << i;
        EXPECT_NEAR(download(state.scale_loss)[0], scale_loss, 1e-5 * scale_loss);
        EXPECT_NEAR(download(state.opacity_loss)[0], opacity_loss, 1e-5 * opacity_loss);
    }

    // Timing on a synthetic scene: ./lichtfeld_tests --tensor-backend=metal
    // --gtest_also_run_disabled_tests --gtest_filter='*MetalFastRaster.DISABLED_Benchmark*'
    TEST_F(MetalFastRaster, DISABLED_Benchmark) {
        constexpr int count = 1'000'000;
        Scene s = make_scene(count, 3, 1000, 700, 77u, -4.5f, -3.0f);
        Gpu g = upload_scene(s, ops::ShStorage::Q16);
        const LossWeights w = random_weights(s.width, s.height, 1u, false);
        const Tensor grad_image = plane(w.image, 3, s.height, s.width);
        AdamState state;
        const ops::BackwardAdam adam = make_adam(g, state, count, {.beta1 = 0.9f, .eps = 1e-15f, .step = 1e-6f, .bc2_sqrt_rcp = 1.0f / std::sqrt(0.001f)});
        Frame frame;
        Tensor none;
        using clock = std::chrono::steady_clock;
        double forward_ms = 0.0, backward_ms = 0.0;
        constexpr int warm = 3, runs = 10;
        for (int i = 0; i < warm + runs; ++i) {
            const auto t0 = clock::now();
            forward(frame, s, g, {});
            (void)download(frame.alpha.flatten().slice(0, 0, 1));
            const auto t1 = clock::now();
            ASSERT_EQ(frame.result.code, ops::RasterResult::Code::Success) << frame.result.message;
            fast_ops().backward(frame.saved, {.image = grad_image, .alpha = none, .depth = none, .normal = none},
                                none, none, none, none, adam, DensificationType::None);
            (void)download(g.opacities.slice(0, 0, 1));
            const auto t2 = clock::now();
            if (i >= warm) {
                forward_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                backward_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
            }
        }
        std::printf("Metal fast raster, %d splats, %dx%d: forward %.2f ms, backward %.2f ms\n", count, s.width,
                    s.height, forward_ms / runs, backward_ms / runs);
    }
} // namespace
