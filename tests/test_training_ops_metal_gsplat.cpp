/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal GsplatRasterOps against a float64 CPU model of the CUDA 3DGUT math:
// unscented projection and tile culling, depth-ordered world-space ray
// blending, SH colour. Gradients are checked by central differences of that
// model; densification, edge scores and screen share by its forward sums.

#include "core/camera_types.h"
#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    namespace ops = lfs::gpu_ops;
    using D3 = std::array<double, 3>;
    using CameraModel = lfs::core::CameraModelType;

    constexpr int kW = 40, kH = 24, kTile = 16;
    constexpr size_t kRest = 15; // SH-rest layout of degree 3

    D3 operator+(const D3& a, const D3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
    D3 operator-(const D3& a, const D3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
    D3 operator*(const double s, const D3& a) { return {s * a[0], s * a[1], s * a[2]}; }
    double dot(const D3& a, const D3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
    D3 cross(const D3& a, const D3& b) {
        return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    }
    D3 normalized(const D3& a) { return (1.0 / std::sqrt(dot(a, a))) * a; }

    struct Scene {
        size_t n = 0;
        std::vector<double> means, scales, quats, opacities, sh0, rest; // rest canonical [n, 15, 3]
    };

    struct Cam {
        CameraModel model = CameraModel::PINHOLE;
        int W = kW, H = kH;
        double fx = 36, fy = 34, cx = 20.5, cy = 11.5;
        std::array<double, 9> R{}; // row-major world to camera
        D3 t{};
        std::vector<float> radial, tangential;
    };

    Scene make_scene(const size_t n, const double log_scale = -2.0) {
        std::mt19937 rng(7);
        auto uniform = [&](const double lo, const double hi) {
            return static_cast<double>(static_cast<float>(std::uniform_real_distribution<double>(lo, hi)(rng)));
        };
        Scene s;
        s.n = n;
        for (size_t i = 0; i < n; ++i) {
            s.means.insert(s.means.end(), {uniform(-0.9, 0.9), uniform(-0.55, 0.55), uniform(-0.6, 0.6)});
            for (int k = 0; k < 3; ++k)
                s.scales.push_back(uniform(log_scale - 0.4, log_scale + 0.4));
            s.quats.insert(s.quats.end(), {uniform(0.5, 1.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5)});
            s.opacities.push_back(uniform(-0.6, 0.9));
            for (int c = 0; c < 3; ++c)
                s.sh0.push_back(uniform(-1.0, 1.0));
            for (size_t k = 0; k < kRest * 3; ++k)
                s.rest.push_back(uniform(-0.3, 0.3));
        }
        return s;
    }

    Cam make_camera(const CameraModel model) {
        Cam cam;
        cam.model = model;
        // Rotation about y by 0.12 then x by -0.07, camera 3 units back.
        const double a = 0.12, b = -0.07;
        const std::array<double, 9> ry{std::cos(a), 0, std::sin(a), 0, 1, 0, -std::sin(a), 0, std::cos(a)};
        const std::array<double, 9> rx{1, 0, 0, 0, std::cos(b), -std::sin(b), 0, std::sin(b), std::cos(b)};
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                double v = 0;
                for (int k = 0; k < 3; ++k)
                    v += rx[r * 3 + k] * ry[k * 3 + c];
                cam.R[r * 3 + c] = static_cast<float>(v);
            }
        cam.t = {0.08, -0.05, 3.0};
        if (model == CameraModel::FISHEYE)
            cam.radial = {-0.03f, 0.004f, 0.001f, 0.f};
        return cam;
    }

    // --- float64 model of the CUDA camera models (Cameras.cuh) ---

    struct RefCamera {
        const Cam& cam;
        double max_angle = std::numeric_limits<double>::max();
        std::array<double, 6> k{};
        std::array<double, 2> p{};

        explicit RefCamera(const Cam& c) : cam(c) {
            for (size_t i = 0; i < c.radial.size() && i < 6; ++i)
                k[i] = c.radial[i];
            for (size_t i = 0; i < c.tangential.size() && i < 2; ++i)
                p[i] = c.tangential[i];
            if (c.model == CameraModel::FISHEYE) {
                // First positive root of the forward polynomial's derivative, if any.
                double root = std::numeric_limits<double>::max();
                for (double th = 1e-3; th < 4.0; th += 1e-3)
                    if (dforward(th) <= 0) {
                        root = th;
                        break;
                    }
                const double mx = std::max(c.W - c.cx, c.cx), my = std::max(c.H - c.cy, c.cy);
                const double radius = std::sqrt(mx * mx + my * my);
                max_angle = std::min(root, std::max(radius / c.fx, radius / c.fy));
            }
        }
        double forward(const double th) const {
            const double t2 = th * th;
            return th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
        }
        double dforward(const double th) const {
            const double t2 = th * th;
            return 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])));
        }
        bool distorted() const { return !cam.radial.empty() || !cam.tangential.empty(); }
        std::array<double, 2> distort(const double u, const double v, double& icd) const {
            const double r2 = u * u + v * v;
            icd = (1 + r2 * (k[0] + r2 * (k[1] + r2 * k[2]))) / (1 + r2 * (k[3] + r2 * (k[4] + r2 * k[5])));
            return {icd * u + p[0] * 2 * u * v + p[1] * (r2 + 2 * u * u), icd * v + p[0] * (r2 + 2 * v * v) + p[1] * 2 * u * v};
        }
        bool in_bounds(const double x, const double y) const {
            return -0.1 * cam.W <= x && x < cam.W * 1.1 && -0.1 * cam.H <= y && y < cam.H * 1.1;
        }
        bool to_image(const D3& q, double& x, double& y) const {
            x = y = 0;
            if (q[2] <= 0)
                return false;
            if (cam.model == CameraModel::PINHOLE) {
                double icd = 1;
                const auto d = distort(q[0] / q[2], q[1] / q[2], icd);
                x = d[0] * cam.fx + cam.cx;
                y = d[1] * cam.fy + cam.cy;
                return (!distorted() || icd > 0.8) && in_bounds(x, y);
            }
            const double norm = std::max(std::hypot(q[0], q[1]), static_cast<double>(FLT_EPSILON));
            const double th = std::min(std::atan2(norm, q[2]), max_angle);
            const double delta = forward(th) / norm;
            if (delta <= 0)
                return false;
            x = cam.fx * delta * q[0] + cam.cx;
            y = cam.fy * delta * q[1] + cam.cy;
            return in_bounds(x, y);
        }
        // World ray through a pixel position; false for an invalid ray.
        bool ray(const double px, const double py, D3& origin, D3& dir) const {
            double u = (px - cam.cx) / cam.fx, v = (py - cam.cy) / cam.fy;
            D3 d{u, v, 1};
            if (cam.model == CameraModel::PINHOLE && distorted()) {
                double x = u, y = v;
                for (int it = 0; it < 50; ++it) { // Newton with a numeric Jacobian
                    double icd;
                    const auto f = distort(x, y, icd);
                    const double h = 1e-7;
                    const auto fx = distort(x + h, y, icd), fy = distort(x, y + h, icd);
                    const double a = (fx[0] - f[0]) / h, b = (fy[0] - f[0]) / h, c = (fx[1] - f[1]) / h,
                                 e = (fy[1] - f[1]) / h;
                    const double det = a * e - b * c, rx = f[0] - u, ry = f[1] - v;
                    x -= (e * rx - b * ry) / det;
                    y -= (a * ry - c * rx) / det;
                }
                d = {x, y, 1};
            } else if (cam.model == CameraModel::FISHEYE) {
                const double delta = std::hypot(u, v);
                double th = delta;
                for (int it = 0; it < 50; ++it)
                    th -= (forward(th) - delta) / dforward(th);
                if (th < 0 || th >= max_angle)
                    return false;
                d = delta >= 1e-6 ? D3{std::sin(th) / delta * u, std::sin(th) / delta * v, std::cos(th)} : D3{0, 0, 1};
            }
            d = normalized(d);
            const auto& R = cam.R;
            origin = {-(R[0] * cam.t[0] + R[3] * cam.t[1] + R[6] * cam.t[2]),
                      -(R[1] * cam.t[0] + R[4] * cam.t[1] + R[7] * cam.t[2]),
                      -(R[2] * cam.t[0] + R[5] * cam.t[1] + R[8] * cam.t[2])};
            dir = {R[0] * d[0] + R[3] * d[1] + R[6] * d[2], R[1] * d[0] + R[4] * d[1] + R[7] * d[2],
                   R[2] * d[0] + R[5] * d[1] + R[8] * d[2]};
            return true;
        }
        D3 to_camera(const D3& w) const {
            const auto& R = cam.R;
            return {R[0] * w[0] + R[1] * w[1] + R[2] * w[2] + cam.t[0], R[3] * w[0] + R[4] * w[1] + R[5] * w[2] + cam.t[1],
                    R[6] * w[0] + R[7] * w[1] + R[8] * w[2] + cam.t[2]};
        }
    };

    std::array<D3, 3> rotation_columns(const double* raw) {
        const double len = std::sqrt(raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] + raw[3] * raw[3]);
        const double w = raw[0] / len, x = raw[1] / len, y = raw[2] / len, z = raw[3] / len;
        return {D3{1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)},
                D3{2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)},
                D3{2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)}};
    }

    double sigmoid(const double x) { return 1 / (1 + std::exp(-x)); }

    struct Reference {
        std::vector<double> image, alpha, dens_w, dens_e, edge, share;
    };

    struct Maps {
        std::vector<float> error, edge;
    };

    // `color_means` pins the SH view directions: gsplat's backward does not
    // differentiate colour through the view direction.
    Reference render(const Scene& s, const Cam& cam, const uint32_t bases, const D3& bg, const Maps* maps = nullptr,
                     const std::vector<double>* color_means = nullptr) {
        const RefCamera rc(cam);
        const int W = cam.W, H = cam.H;
        const size_t n = s.n;
        const int tiles_x = (W + kTile - 1) / kTile, tiles_y = (H + kTile - 1) / kTile;
        struct Proj {
            bool visible = false;
            uint32_t depth_bits = 0;
            int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
            D3 color{};
        };
        std::vector<Proj> proj(n);
        Reference out;
        out.share.assign(n, 0);
        const auto& R = cam.R;
        const D3 origin{-(R[0] * cam.t[0] + R[3] * cam.t[1] + R[6] * cam.t[2]),
                        -(R[1] * cam.t[0] + R[4] * cam.t[1] + R[7] * cam.t[2]),
                        -(R[2] * cam.t[0] + R[5] * cam.t[1] + R[8] * cam.t[2])};
        for (size_t g = 0; g < n; ++g) {
            const D3 mean{s.means[g * 3], s.means[g * 3 + 1], s.means[g * 3 + 2]};
            const D3 mc = rc.to_camera(mean);
            if (mc[2] < 0.01 || mc[2] > 1e4)
                continue;
            const auto rot = rotation_columns(&s.quats[g * 4]);
            const double lambda = 0.01 * 3 - 3, spread = std::sqrt(3 + lambda);
            const double wm0 = lambda / (3 + lambda), wc0 = wm0 + (1 - 0.01 + 2), wi = 1 / (2 * (3 + lambda));
            std::array<std::array<double, 2>, 7> pts{};
            double mx = 0, my = 0;
            bool valid = true;
            for (int i = 0; i < 7 && valid; ++i) {
                D3 point = mean;
                if (i > 0) {
                    const int axis = (i - 1) % 3;
                    const D3 delta = spread * std::exp(s.scales[g * 3 + axis]) * rot[axis];
                    point = i <= 3 ? mean + delta : mean - delta;
                }
                valid = rc.to_image(rc.to_camera(point), pts[i][0], pts[i][1]);
                mx += (i == 0 ? wm0 : wi) * pts[i][0];
                my += (i == 0 ? wm0 : wi) * pts[i][1];
            }
            if (!valid)
                continue;
            double c00 = 0, c01 = 0, c11 = 0;
            for (int i = 0; i < 7; ++i) {
                const double dx = pts[i][0] - mx, dy = pts[i][1] - my, w = i == 0 ? wc0 : wi;
                c00 += w * dx * dx;
                c01 += w * dx * dy;
                c11 += w * dy * dy;
            }
            const double det_orig = c00 * c11 - c01 * c01;
            c00 += 0.3;
            c11 += 0.3;
            const double det = c00 * c11 - c01 * c01;
            const double opacity = sigmoid(s.opacities[g]) * std::sqrt(std::max(0.0, det_orig / det));
            if (det <= 0 || opacity < 1.0 / 255)
                continue;
            const double extend = std::min(3.33, std::sqrt(2 * std::log(opacity * 255)));
            const double b = 0.5 * (c00 + c11);
            const double r1 = extend * std::sqrt(b + std::sqrt(std::max(0.01, b * b - det)));
            const double rx = std::ceil(std::min(extend * std::sqrt(c00), r1));
            const double ry = std::ceil(std::min(extend * std::sqrt(c11), r1));
            if (mx + rx <= 0 || mx - rx >= W || my + ry <= 0 || my - ry >= H)
                continue;
            auto& p = proj[g];
            p.visible = true;
            p.depth_bits = std::bit_cast<uint32_t>(static_cast<float>(mc[2]));
            p.x0 = std::clamp<int>(static_cast<int>(std::floor((mx - rx) / kTile)), 0, tiles_x);
            p.y0 = std::clamp<int>(static_cast<int>(std::floor((my - ry) / kTile)), 0, tiles_y);
            p.x1 = std::clamp<int>(static_cast<int>(std::ceil((mx + rx) / kTile)), 0, tiles_x);
            p.y1 = std::clamp<int>(static_cast<int>(std::ceil((my + ry) / kTile)), 0, tiles_y);
            const double dxs = std::max(0.0, std::min<double>(W, mx + rx) - std::max(0.0, mx - rx));
            const double dys = std::max(0.0, std::min<double>(H, my + ry) - std::max(0.0, my - ry));
            out.share[g] = (dxs / W) * (dys / H);
            // SH up to degree 2 in the canonical coefficient order.
            const auto& cm = color_means ? *color_means : s.means;
            const D3 d = normalized(D3{cm[g * 3], cm[g * 3 + 1], cm[g * 3 + 2]} - origin);
            const double x = d[0], y = d[1], z = d[2];
            const std::array<double, 9> basis{0.2820947917738781,
                                              -0.48860251190292 * y,
                                              0.48860251190292 * z,
                                              -0.48860251190292 * x,
                                              0.5462742152960395 * 2 * x * y,
                                              -1.092548430592079 * z * y,
                                              0.9461746957575601 * z * z - 0.3153915652525201,
                                              -1.092548430592079 * z * x,
                                              0.5462742152960395 * (x * x - y * y)};
            for (int c = 0; c < 3; ++c) {
                double color = basis[0] * s.sh0[g * 3 + c];
                for (uint32_t k = 1; k < bases; ++k)
                    color += basis[k] * s.rest[(g * kRest + k - 1) * 3 + c];
                p.color[c] = color + 0.5;
            }
        }

        std::vector<size_t> order(n);
        for (size_t i = 0; i < n; ++i)
            order[i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return proj[a].depth_bits < proj[b].depth_bits; });
        std::vector<std::vector<size_t>> tile_lists(static_cast<size_t>(tiles_x) * tiles_y);
        for (const size_t g : order) {
            const auto& p = proj[g];
            for (int ty = p.y0; p.visible && ty < p.y1; ++ty)
                for (int tx = p.x0; tx < p.x1; ++tx)
                    tile_lists[ty * tiles_x + tx].push_back(g);
        }
        out.image.assign(3 * W * H, 0);
        out.alpha.assign(W * H, 0);
        out.dens_w.assign(n, 0);
        out.dens_e.assign(n, 0);
        out.edge.assign(n, 0);
        for (int py = 0; py < H; ++py) {
            for (int px = 0; px < W; ++px) {
                const int pixel = py * W + px;
                const auto& list = tile_lists[(py / kTile) * tiles_x + px / kTile];
                D3 o, dir;
                double T = 1;
                D3 acc{};
                if (rc.ray(px + 0.5, py + 0.5, o, dir)) {
                    for (const size_t g : list) {
                        const auto rot = rotation_columns(&s.quats[g * 4]);
                        const D3 mean{s.means[g * 3], s.means[g * 3 + 1], s.means[g * 3 + 2]};
                        D3 gro, grd;
                        for (int a = 0; a < 3; ++a) {
                            const double inv = std::exp(-s.scales[g * 3 + a]);
                            gro[a] = inv * dot(rot[a], o - mean);
                            grd[a] = inv * dot(rot[a], dir);
                        }
                        const D3 gcrod = cross(normalized(grd), gro);
                        const double alpha = std::min(0.999, sigmoid(s.opacities[g]) * std::exp(-0.5 * dot(gcrod, gcrod)));
                        if (alpha < 1.0 / 255)
                            continue;
                        const double next_T = T * (1 - alpha);
                        if (next_T <= 1e-4)
                            break;
                        const double fac = alpha * T;
                        acc = acc + fac * proj[g].color;
                        if (maps) {
                            out.dens_w[g] += fac;
                            out.dens_e[g] += fac * maps->error[pixel];
                            if (maps->edge[pixel] > 0)
                                out.edge[g] += fac * maps->edge[pixel];
                        }
                        T = next_T;
                    }
                }
                out.alpha[pixel] = 1 - T;
                for (int c = 0; c < 3; ++c)
                    out.image[c * W * H + pixel] = acc[c] + T * bg[c];
            }
        }
        return out;
    }

    // --- device side ---

    size_t rest_index(const size_t p, const size_t coeff, const size_t channel) {
        const size_t offset = coeff * 3 + channel;
        return ((p / 32) * (12 * 32) + (offset / 4) * 32 + p % 32) * 4 + offset % 4;
    }

    Tensor gpu(const std::vector<double>& values, lfs::core::TensorShape shape) {
        return Tensor::from_vector(std::vector<float>(values.begin(), values.end()), shape, Device::GPU);
    }

    std::vector<float> host(const Tensor& t) {
        const auto cpu = t.cpu().contiguous();
        const auto* data = cpu.ptr<float>();
        return {data, data + cpu.numel()};
    }

    struct Rendered {
        Tensor image, alpha;
    };

    class MetalGsplat : public testing::Test {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GpuBackend::Metal))
                GTEST_SKIP() << "No Metal device";
            scope_.emplace(GpuBackend::Metal);
            table_ = lfs::training::training_ops(GpuBackend::Metal).gsplat;
            ASSERT_NE(table_, nullptr);
            saved_.backend = table_->create();
        }

        void TearDown() override {
            if (table_)
                table_->release(saved_);
        }

        Rendered forward(const Scene& s, const Cam& cam, const uint32_t bases, const D3& bg,
                         const bool q16 = false) {
            const size_t n = s.n;
            std::vector<double> rest(lfs::core::sh_swizzled_float_count(n, kRest), 0.0);
            for (size_t p = 0; p < n; ++p)
                for (size_t k = 0; k < kRest; ++k)
                    for (size_t c = 0; c < 3; ++c)
                        rest[rest_index(p, k, c)] = s.rest[(p * kRest + k) * 3 + c];
            inputs_ = {gpu(s.means, {n, 3}), gpu(s.scales, {n, 3}), gpu(s.quats, {n, 4}), gpu(s.opacities, {n, 1}),
                       gpu(s.sh0, {n, 1, 3}), gpu(rest, {rest.size()})};
            Tensor bounds;
            if (q16) {
                // One bounds block; codes in the pad-dropped swizzled cell layout.
                const auto [lo, hi] = std::minmax_element(s.rest.begin(), s.rest.end());
                const std::vector<double> range{*lo, *hi};
                bounds = gpu(range, {2});
                const size_t cells = kRest * 3;
                auto codes = Tensor::zeros({lfs::core::sh_value_quant::sh_value_u16_count(n, kRest)}, Device::CPU,
                                           DataType::Float16);
                auto* raw = static_cast<uint16_t*>(codes.data_ptr());
                for (size_t p = 0; p < n; ++p)
                    for (size_t cell = 0; cell < cells; ++cell)
                        raw[(p / 32) * (cells * 32) + cell * 32 + p % 32] = static_cast<uint16_t>(
                            std::lround(65535.0 * (s.rest[p * cells + cell] - *lo) / (*hi - *lo)));
                inputs_[5] = codes.to(Device::GPU);
            }
            std::vector<float> view(16, 0.f);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c)
                    view[r * 4 + c] = static_cast<float>(cam.R[r * 3 + c]);
                view[r * 4 + 3] = static_cast<float>(cam.t[r]);
            }
            view[15] = 1.f;
            view_ = Tensor::from_vector(view, {4, 4}, Device::GPU);
            radial_ = cam.radial.empty() ? Tensor() : Tensor::from_vector(cam.radial, {cam.radial.size()}, Device::CPU);
            tangential_ =
                cam.tangential.empty() ? Tensor() : Tensor::from_vector(cam.tangential, {cam.tangential.size()}, Device::CPU);
            bg_ = Tensor::from_vector(std::vector<float>{static_cast<float>(bg[0]), static_cast<float>(bg[1]),
                                                         static_cast<float>(bg[2])},
                                      {3}, Device::GPU);
            const ops::SplatInputs splats{inputs_[0], inputs_[1], inputs_[2], inputs_[3], inputs_[4], inputs_[5], bounds};
            const ops::GsplatParams params{
                .full_image = {cam.H, cam.W},
                .intrinsics = {static_cast<float>(cam.fx), static_cast<float>(cam.fy), static_cast<float>(cam.cx),
                               static_cast<float>(cam.cy)},
                .sh = {.storage = q16 ? ops::ShStorage::Q16 : ops::ShStorage::Float32,
                       .active_bases = bases,
                       .layout_bases = 16},
                .camera_model = cam.model,
            };
            Rendered out;
            Tensor depth, normal;
            const auto result = table_->forward(saved_, splats, view_, radial_, tangential_, bg_, Tensor(), params,
                                                {out.image, out.alpha, depth, normal});
            EXPECT_EQ(result.code, ops::RasterResult::Code::Success) << result.message;
            return out;
        }

        std::optional<lfs::core::GpuBackendScope> scope_;
        const ops::GsplatRasterOps* table_ = nullptr;
        ops::GsplatSaved saved_;
        std::vector<Tensor> inputs_;
        Tensor view_, radial_, tangential_, bg_;
    };

    void expect_near(const std::vector<float>& actual, const std::vector<double>& expected, const double tolerance,
                     const char* name) {
        ASSERT_EQ(actual.size(), expected.size()) << name;
        double worst = 0;
        size_t at = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const double diff = std::abs(actual[i] - expected[i]);
            if (diff > worst) {
                worst = diff;
                at = i;
            }
        }
        EXPECT_LE(worst, tolerance) << name << " index " << at << ": " << actual[at] << " vs " << expected[at];
    }

    constexpr D3 kBackground{0.2, 0.5, 0.1};

    TEST_F(MetalGsplat, ForwardMatchesReference) {
        const auto scene = make_scene(40);
        for (const auto model : {CameraModel::PINHOLE, CameraModel::FISHEYE}) {
            SCOPED_TRACE(static_cast<int>(model));
            auto cam = make_camera(model);
            const auto rendered = forward(scene, cam, 4, kBackground);
            const auto expected = render(scene, cam, 4, kBackground);
            EXPECT_EQ(rendered.image.shape(), lfs::core::TensorShape({3, kH, kW}));
            expect_near(host(rendered.image), expected.image, 2e-4, "image");
            expect_near(host(rendered.alpha), expected.alpha, 2e-4, "alpha");
            double coverage = 0;
            for (const double a : expected.alpha)
                coverage += a;
            EXPECT_GT(coverage, 0.2 * kW * kH) << "the scene should cover the image";
        }
        // OpenCV pinhole: radial and tangential distortion with Newton undistortion.
        auto distorted = make_camera(CameraModel::PINHOLE);
        distorted.radial = {-0.05f, 0.01f};
        distorted.tangential = {0.002f, -0.001f};
        const auto rendered = forward(scene, distorted, 9, kBackground);
        const auto expected = render(scene, distorted, 9, kBackground);
        expect_near(host(rendered.image), expected.image, 2e-4, "distorted image");
        expect_near(host(rendered.alpha), expected.alpha, 2e-4, "distorted alpha");
    }

    // Many radix blocks in both sorts and two passes over 260 tile ids. A
    // float alpha landing on the other side of a cutoff than its float64 value
    // changes a pixel by up to 1/255, so a few such pixels are allowed.
    TEST_F(MetalGsplat, ForwardMatchesReferenceAtScale) {
        const auto scene = make_scene(30000, -3.3);
        auto cam = make_camera(CameraModel::PINHOLE);
        cam.W = 320;
        cam.H = 200;
        cam.fx = cam.fy = 280;
        cam.cx = 160;
        cam.cy = 100;
        const auto image = host(forward(scene, cam, 4, kBackground).image);
        const auto expected = render(scene, cam, 4, kBackground).image;
        ASSERT_EQ(image.size(), expected.size());
        size_t outliers = 0;
        double worst = 0;
        for (size_t i = 0; i < image.size(); ++i) {
            const double diff = std::abs(image[i] - expected[i]);
            outliers += diff > 2e-4 ? 1 : 0;
            worst = std::max(worst, diff);
        }
        EXPECT_LE(outliers, image.size() / 1000) << "worst difference " << worst;
        EXPECT_LE(worst, 0.02);
    }

    TEST_F(MetalGsplat, Q16StorageMatchesDecodedFloats) {
        const auto scene = make_scene(40);
        const auto cam = make_camera(CameraModel::PINHOLE);
        const auto q16 = host(forward(scene, cam, 9, kBackground, true).image);
        // The same codes decoded on the host through the float path.
        auto decoded = scene;
        const auto [lo, hi] = std::minmax_element(scene.rest.begin(), scene.rest.end());
        const float flo = static_cast<float>(*lo), fhi = static_cast<float>(*hi);
        for (auto& v : decoded.rest) {
            const auto q = std::lround(65535.0 * (v - *lo) / (*hi - *lo));
            v = flo + (fhi - flo) * (static_cast<float>(q) * (1.0f / 65535.0f));
        }
        const auto floats = host(forward(decoded, cam, 9, kBackground).image);
        std::vector<double> expected(floats.begin(), floats.end());
        expect_near(q16, expected, 1e-5, "q16 image");
    }

    TEST_F(MetalGsplat, BackwardMatchesFiniteDifferences) {
        auto scene = make_scene(40);
        const auto cam = make_camera(CameraModel::PINHOLE);
        constexpr uint32_t bases = 4;
        const size_t n = scene.n;
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> weight(-1.f, 1.f);
        std::vector<float> w_image(3 * kW * kH), w_alpha(kW * kH);
        for (auto& w : w_image)
            w = weight(rng);
        for (auto& w : w_alpha)
            w = weight(rng);
        Maps maps{std::vector<float>(kW * kH), std::vector<float>(kW * kH)};
        for (size_t i = 0; i < maps.error.size(); ++i) {
            maps.error[i] = 0.5f + 0.5f * weight(rng);
            maps.edge[i] = std::max(0.f, weight(rng));
        }
        const auto color_means = scene.means;
        const auto loss = [&](const Scene& s) {
            const auto r = render(s, cam, bases, kBackground, nullptr, &color_means);
            double l = 0;
            for (size_t i = 0; i < w_image.size(); ++i)
                l += w_image[i] * r.image[i];
            for (size_t i = 0; i < w_alpha.size(); ++i)
                l += w_alpha[i] * r.alpha[i];
            return l;
        };

        forward(scene, cam, bases, kBackground);
        auto gm = Tensor::full({n, 3}, 1.f, Device::GPU); // gradients accumulate onto what is there
        auto gs = Tensor::zeros({n, 3}, Device::GPU), gr = Tensor::zeros({n, 4}, Device::GPU);
        auto go = Tensor::zeros({n, 1}, Device::GPU), g0 = Tensor::zeros({n, 1, 3}, Device::GPU);
        auto gn = Tensor::zeros({lfs::core::sh_swizzled_float_count(n, kRest)}, Device::GPU);
        std::array<Tensor*, 6> slots{&gm, &gs, &gr, &go, &g0, &gn};
        const ops::GsplatGradients gradients{&slots, [](void* owner, ops::AdamSlot slot) -> Tensor& {
                                                 return *(*static_cast<std::array<Tensor*, 6>*>(owner))[static_cast<size_t>(slot)];
                                             }};
        auto densification = Tensor::zeros({2, n}, Device::GPU);
        auto edge_scores = Tensor::zeros({n}, Device::GPU);
        auto share = Tensor::zeros({n}, Device::GPU);
        table_->backward(saved_, Tensor::from_vector(w_image, {3, kH, kW}, Device::GPU),
                         Tensor::from_vector(w_alpha, {1, kH, kW}, Device::GPU), gradients, densification,
                         Tensor::from_vector(maps.error, {kH, kW}, Device::GPU),
                         Tensor::from_vector(maps.edge, {kH, kW}, Device::GPU), edge_scores, share);

        const auto means = host(gm), scales = host(gs), quats = host(gr), opacity = host(go), sh0 = host(g0),
                   rest = host(gn);
        const auto reference = render(scene, cam, bases, kBackground, &maps);
        const auto dens = host(densification);
        expect_near({dens.begin(), dens.begin() + n}, reference.dens_w, 2e-4, "densification weight");
        expect_near({dens.begin() + n, dens.end()}, reference.dens_e, 2e-4, "densification error");
        expect_near(host(edge_scores), reference.edge, 2e-4, "edge scores");
        expect_near(host(share), reference.share, 1e-6, "screen share");

        // Central differences at the gaussians with the most blending weight.
        std::vector<size_t> ranked(n);
        for (size_t i = 0; i < n; ++i)
            ranked[i] = i;
        std::sort(ranked.begin(), ranked.end(), [&](size_t a, size_t b) { return reference.dens_w[a] > reference.dens_w[b]; });
        const double h = 1e-4;
        size_t checked = 0;
        for (size_t pick = 0; pick < 4; ++pick) {
            const size_t g = ranked[pick * 3];
            struct Param {
                std::vector<double>* values;
                size_t index;
                float analytic;
                const char* name;
            };
            std::vector<Param> params;
            for (size_t k = 0; k < 3; ++k) {
                params.push_back({&scene.means, g * 3 + k, means[g * 3 + k] - 1.f, "mean"});
                params.push_back({&scene.scales, g * 3 + k, scales[g * 3 + k], "scale"});
                params.push_back({&scene.sh0, g * 3 + k, sh0[g * 3 + k], "sh0"});
                params.push_back({&scene.rest, (g * kRest + k) * 3 + 1, rest[rest_index(g, k, 1)], "sh rest"});
            }
            for (size_t k = 0; k < 4; ++k)
                params.push_back({&scene.quats, g * 4 + k, quats[g * 4 + k], "quat"});
            params.push_back({&scene.opacities, g, opacity[g], "opacity"});
            for (const auto& p : params) {
                const double saved_value = (*p.values)[p.index];
                (*p.values)[p.index] = saved_value + h;
                const double up = loss(scene);
                (*p.values)[p.index] = saved_value - h;
                const double down = loss(scene);
                (*p.values)[p.index] = saved_value;
                const double numeric = (up - down) / (2 * h);
                EXPECT_NEAR(p.analytic, numeric, 1e-4 + 1e-4 * std::abs(numeric))
                    << p.name << " of gaussian " << g << " element " << p.index;
                ++checked;
            }
            // Inactive SH bands receive nothing.
            EXPECT_EQ(rest[rest_index(g, 3, 0)], 0.f);
        }
        EXPECT_EQ(checked, 4u * 17u);

        // Without a pixel error map, densification accumulates the mean-gradient norm.
        forward(scene, cam, bases, kBackground);
        auto norms = Tensor::zeros({2, n}, Device::GPU);
        std::array<Tensor, 6> scratch{Tensor::zeros({n, 3}, Device::GPU), Tensor::zeros({n, 3}, Device::GPU),
                                      Tensor::zeros({n, 4}, Device::GPU), Tensor::zeros({n, 1}, Device::GPU),
                                      Tensor::zeros({n, 1, 3}, Device::GPU),
                                      Tensor::zeros({lfs::core::sh_swizzled_float_count(n, kRest)}, Device::GPU)};
        std::array<Tensor*, 6> scratch_slots{&scratch[0], &scratch[1], &scratch[2], &scratch[3], &scratch[4], &scratch[5]};
        Tensor none;
        table_->backward(saved_, Tensor::from_vector(w_image, {3, kH, kW}, Device::GPU),
                         Tensor::from_vector(w_alpha, {1, kH, kW}, Device::GPU),
                         {&scratch_slots, gradients.get}, norms, none, none, none, none);
        const auto norm_rows = host(norms);
        std::vector<double> expected_norms(n);
        for (size_t g = 0; g < n; ++g)
            expected_norms[g] = std::hypot(means[g * 3] - 1.0, means[g * 3 + 1] - 1.0, means[g * 3 + 2] - 1.0);
        expect_near({norm_rows.begin(), norm_rows.begin() + n}, expected_norms, 1e-5, "gradient norms");
    }
} // namespace
