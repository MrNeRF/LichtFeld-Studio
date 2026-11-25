/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The Metal photometric, mask, extra-loss and geometry ops against CPU
// transliterations of their CUDA kernels (ssim.cu, ssim_reduction.cu,
// l1_loss.cu, mask_preprocess.cu, regularization.cu,
// depth_loss.cu, normal_loss.cu,
// normal_consistency_loss.cu), evaluated in double. SSIM partials are rounded
// to half as the CUDA workspace stores them. A finite-difference check covers
// the SSIM gradient itself.

#include "core/gpu_backend_fwd.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_execution.hpp"
#include "core/tensor_upload.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/ops/structure.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;
    namespace ops = lfs::gpu_ops;
    namespace k = lfs::training::kernels;
    using Field = std::vector<double>;

    // CPU references of the CUDA loss kernels, run on every portable backend.
    class PortableLosses : public testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << "No " << lfs::core::gpu_backend_name(GetParam()) << " device";
            scope_.emplace(GetParam());
            table_ = &lfs::training::training_ops(GetParam());
            if (!table_->photometric || !table_->masks || !table_->extra_loss || !table_->geometry)
                GTEST_SKIP() << "Loss families are not filled";
        }

        std::optional<lfs::core::GpuBackendScope> scope_;
        const lfs::training::TrainingOps* table_ = nullptr;
    };

    // Deterministic uniform values in [lo, hi).
    std::vector<float> uniform(const size_t count, const uint32_t seed, const float lo = 0.f, const float hi = 1.f) {
        std::vector<float> values(count);
        uint32_t state = seed * 747796405u + 2891336453u;
        for (float& value : values) {
            state = state * 1664525u + 1013904223u;
            value = lo + (hi - lo) * static_cast<float>(state >> 8) / 16777216.f;
        }
        return values;
    }

    Tensor upload(const std::vector<float>& values, const TensorShape& shape) {
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    Tensor upload_bytes(const std::vector<uint8_t>& values, const TensorShape& shape) {
        auto cpu = Tensor::empty(shape, Device::CPU, DataType::UInt8);
        std::copy(values.begin(), values.end(), cpu.ptr<uint8_t>());
        return cpu.to(Device::GPU);
    }

    std::vector<float> download(const Tensor& tensor) {
        Tensor view = tensor.dtype() == DataType::Float16 ? tensor.to(DataType::Float32) : tensor;
        const Tensor cpu = view.cpu().contiguous();
        const float* data = cpu.ptr<float>();
        return {data, data + cpu.numel()};
    }

    Field widen(const std::vector<float>& values) { return {values.begin(), values.end()}; }

    // |got - want| <= abs + rel * max|want| everywhere.
    void expect_field(const std::vector<float>& got, const Field& want, const double abs, const double rel,
                      const std::string& what) {
        ASSERT_EQ(got.size(), want.size()) << what;
        double scale = 0.0;
        for (const double value : want)
            scale = std::max(scale, std::abs(value));
        const double limit = abs + rel * scale;
        size_t worst = 0;
        double worst_error = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double error = std::abs(static_cast<double>(got[i]) - want[i]);
            if (!(error <= worst_error)) {
                worst_error = error;
                worst = i;
            }
        }
        EXPECT_LE(worst_error, limit) << what << ": index " << worst << " got " << got[worst] << " want "
                                      << want[worst] << " (max |want| " << scale << ")";
    }

    void expect_value(const double got, const double want, const double rel, const std::string& what) {
        EXPECT_NEAR(got, want, 1e-7 + rel * std::abs(want)) << what;
    }

    double sign(const double v) { return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0); }
    double to_half(const double v) { return static_cast<double>(static_cast<_Float16>(static_cast<float>(v))); }

    // ---- SSIM reference (ssim.cu) ---------------------------------------------

    constexpr std::array<float, 11> kGauss = {
        0.001028380123898387f, 0.0075987582094967365f, 0.036000773310661316f, 0.10936068743467331f,
        0.21300552785396576f, 0.26601171493530273f, 0.21300552785396576f, 0.10936068743467331f,
        0.036000773310661316f, 0.0075987582094967365f, 0.001028380123898387f};
    constexpr double kC1 = 0.01f * 0.01f;
    constexpr double kC2 = 0.03f * 0.03f;

    // Separable 11x11 gaussian with zero padding over one [H,W] plane.
    Field blur(const Field& f, const int H, const int W) {
        Field horizontal(f.size(), 0.0), out(f.size(), 0.0);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int d = -5; d <= 5; ++d)
                    if (x + d >= 0 && x + d < W)
                        horizontal[y * W + x] += kGauss[d + 5] * f[y * W + x + d];
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int d = -5; d <= 5; ++d)
                    if (y + d >= 0 && y + d < H)
                        out[y * W + x] += kGauss[d + 5] * horizontal[(y + d) * W + x];
        return out;
    }

    struct Shape4 {
        int n, c, h, w;
        [[nodiscard]] size_t plane() const { return static_cast<size_t>(h) * w; }
        [[nodiscard]] size_t numel() const { return static_cast<size_t>(n) * c * plane(); }
        [[nodiscard]] TensorShape tensor() const {
            return TensorShape({static_cast<size_t>(n), static_cast<size_t>(c), static_cast<size_t>(h), static_cast<size_t>(w)});
        }
    };

    Field plane_of(const Field& f, const Shape4& s, const int n, const int c) {
        const size_t base = (static_cast<size_t>(n) * s.c + c) * s.plane();
        return {f.begin() + static_cast<std::ptrdiff_t>(base), f.begin() + static_cast<std::ptrdiff_t>(base + s.plane())};
    }

    Field product(const Field& a, const Field& b) {
        Field out(a.size());
        for (size_t i = 0; i < a.size(); ++i)
            out[i] = a[i] * b[i];
        return out;
    }

    // Per element SSIM, CS and the half-rounded partials the backward reads.
    // Decoupled (raw non-empty): luminance from the corrected image, contrast
    // and structure from the raw render; partials as in decoupledFusedL1SSIMForwardCUDA.
    struct SsimForward {
        Field ssim, cs, dm_mu, dm_s1, dm_s12, raw_dm_mu;
    };

    SsimForward ssim_forward(const Field& x, const Field& raw, const Field& y, const Shape4& s, const double weight,
                             const Field& mask = {}) {
        SsimForward out;
        for (Field* f : {&out.ssim, &out.cs, &out.dm_mu, &out.dm_s1, &out.dm_s12, &out.raw_dm_mu})
            f->assign(s.numel(), 0.0);
        // Outside-image zero padding retains unit weight. Only excluded samples
        // inside the image reduce the Gaussian window's normalization mass.
        Field window_mass(s.plane(), 1.0);
        if (!mask.empty()) {
            Field excluded(mask.size());
            std::transform(mask.begin(), mask.end(), excluded.begin(), [](double m) { return 1.0 - m; });
            const Field missing = blur(excluded, s.h, s.w);
            for (size_t i = 0; i < window_mass.size(); ++i)
                window_mass[i] -= missing[i];
        }
        const auto moment = [&](const Field& values) {
            Field out = blur(mask.empty() ? values : product(values, mask), s.h, s.w);
            for (size_t i = 0; i < out.size(); ++i)
                out[i] = window_mass[i] >= 1e-30 ? out[i] / window_mass[i] : 0.0;
            return out;
        };
        for (int n = 0; n < s.n; ++n) {
            for (int c = 0; c < s.c; ++c) {
                const Field px = plane_of(x, s, n, c), py = plane_of(y, s, n, c);
                const Field pr = raw.empty() ? px : plane_of(raw, s, n, c);
                const Field mx = moment(px), my = moment(py), mr = moment(pr);
                const Field rr = moment(product(pr, pr)), yy = moment(product(py, py));
                const Field ry = moment(product(pr, py));
                const size_t base = (static_cast<size_t>(n) * s.c + c) * s.plane();
                for (size_t i = 0; i < s.plane(); ++i) {
                    // Weight before half rounding, as in the saved GPU partials.
                    const double derivative_weight = mask.empty() ? 1.0 : (window_mass[i] >= 1e-30 ? mask[i] / window_mass[i] : 0.0);
                    const auto partial = [&](double v) { return to_half(v * derivative_weight); };
                    const double s_r = rr[i] - mr[i] * mr[i], s_y = yy[i] - my[i] * my[i], s_ry = ry[i] - mr[i] * my[i];
                    const double B = s_r + s_y + kC2, D = 2 * s_ry + kC2;
                    const size_t e = base + i;
                    if (raw.empty()) {
                        const double A = mx[i] * mx[i] + my[i] * my[i] + kC1, C = 2 * mx[i] * my[i] + kC1;
                        out.ssim[e] = C * D / (A * B);
                        out.cs[e] = D / B;
                        const double mu1 = mx[i], mu2 = my[i];
                        out.dm_mu[e] = partial(mu2 * 2 * D / (A * B) - mu2 * 2 * C / (A * B) - mu1 * 2 * C * D / (A * A * B) +
                                               mu1 * 2 * C * D / (A * B * B));
                        out.dm_s1[e] = partial(-C * D / (A * B * B));
                        out.dm_s12[e] = partial(2 * C / (A * B));
                    } else {
                        const double A = mx[i] * mx[i] + my[i] * my[i] + kC1, C = 2 * mx[i] * my[i] + kC1;
                        const double lum = C / A;
                        out.cs[e] = D / B;
                        out.ssim[e] = lum * out.cs[e];
                        out.dm_mu[e] = partial(out.cs[e] * 2 * (my[i] * A - mx[i] * C) / (A * A));
                        const double ds1 = weight * (-(lum * D) / (B * B)), ds12 = weight * (2 * lum / B);
                        out.dm_s1[e] = partial(ds1);
                        out.dm_s12[e] = partial(ds12);
                        out.raw_dm_mu[e] = partial(ds1 * (-2 * mr[i]) + ds12 * (-my[i]));
                    }
                }
            }
        }
        return out;
    }

    // fusedL1SSIMBackwardCUDA / maskedFusedL1SSIMBackwardCUDA with chain(pixel).
    Field ssim_backward(const Field& x, const Field& y, const Field& dm_mu, const Field* dm_s1, const Field* dm_s12,
                        const Field& chain, const double weight, const Shape4& s, const Field& mask = {}) {
        Field grad(s.numel());
        double mask_sum = 0.0;
        for (const double m : mask)
            mask_sum += m;
        const double masked_chain = 1.0 / (static_cast<float>(mask_sum) * s.n * s.c + 1e-8);
        for (int n = 0; n < s.n; ++n) {
            for (int c = 0; c < s.c; ++c) {
                const size_t base = (static_cast<size_t>(n) * s.c + c) * s.plane();
                Field d0(s.plane()), d1(s.plane(), 0.0), d2(s.plane(), 0.0);
                for (size_t i = 0; i < s.plane(); ++i) {
                    const double derivative_chain = mask.empty() ? chain[i] : masked_chain;
                    d0[i] = -weight * dm_mu[base + i] * derivative_chain;
                    if (dm_s1) {
                        d1[i] = -weight * (*dm_s1)[base + i] * derivative_chain;
                        d2[i] = -weight * (*dm_s12)[base + i] * derivative_chain;
                    }
                }
                const Field s0 = blur(d0, s.h, s.w), s1 = blur(d1, s.h, s.w), s2 = blur(d2, s.h, s.w);
                for (size_t i = 0; i < s.plane(); ++i) {
                    const double p1 = x[base + i], p2 = y[base + i];
                    const double input_mask = mask.empty() ? 1.0 : mask[i];
                    grad[base + i] = input_mask * (s0[i] + 2 * p1 * s1[i] + p2 * s2[i]) + (1 - weight) * sign(p1 - p2) * chain[i];
                }
            }
        }
        return grad;
    }

    Field channel_mean(const Field& f, const Shape4& s) {
        Field out(static_cast<size_t>(s.n) * s.plane(), 0.0);
        for (int n = 0; n < s.n; ++n)
            for (int c = 0; c < s.c; ++c)
                for (size_t i = 0; i < s.plane(); ++i)
                    out[n * s.plane() + i] += f[(static_cast<size_t>(n) * s.c + c) * s.plane() + i] / s.c;
        return out;
    }

    // ssim_reduction.cu crops each axis on its own.
    bool in_mean_crop(const int x, const int y, const Shape4& s, const bool padding) {
        const bool crop_y = padding && s.h > 10, crop_x = padding && s.w > 10;
        return !((crop_y && (y < 5 || y >= s.h - 5)) || (crop_x && (x < 5 || x >= s.w - 5)));
    }

    // The backward crops each axis like the forward.
    Field valid_chain(const Shape4& s, const bool padding) {
        const bool crop_y = padding && s.h > 10, crop_x = padding && s.w > 10;
        const double count = static_cast<double>(s.n) * s.c * (crop_y ? s.h - 10 : s.h) * (crop_x ? s.w - 10 : s.w);
        const float per_pixel = 1.0f / static_cast<float>(count);
        Field chain(s.plane(), 0.0);
        for (int y = 0; y < s.h; ++y)
            for (int x = 0; x < s.w; ++x)
                if (in_mean_crop(x, y, s, padding))
                    chain[y * s.w + x] = per_pixel;
        return chain;
    }

    struct PhotoCase {
        ops::PhotoPath path;
        Shape4 shape;
        bool target_bytes = false;
        bool mask_bytes = false;
        float weight = 0.2f;
    };

    struct PhotoRef {
        double loss = 0.0;
        Field grad, grad_raw, ssim_map, cs_map;
    };

    PhotoRef photo_reference(const PhotoCase& pc, const Field& x, const Field& raw, const Field& y, const Field& mask) {
        const Shape4& s = pc.shape;
        const bool decoupled = pc.path == ops::PhotoPath::Decoupled || pc.path == ops::PhotoPath::MaskedDecoupled;
        const bool masked = !mask.empty();
        PhotoRef ref;
        if (pc.path == ops::PhotoPath::L1) {
            ref.grad.resize(s.numel());
            for (size_t i = 0; i < s.numel(); ++i) {
                ref.loss += std::abs(x[i] - y[i]) / static_cast<double>(s.numel());
                ref.grad[i] = sign(x[i] - y[i]) / static_cast<double>(s.numel());
            }
            return ref;
        }
        const bool pure = pc.path == ops::PhotoPath::SSIM;
        const double w = pure ? 1.0 : pc.weight;
        const SsimForward f = ssim_forward(x, decoupled ? raw : Field{}, y, s, w, mask);
        const Field ssim_mean = channel_mean(f.ssim, s);
        ref.ssim_map = pure ? f.ssim : ssim_mean;
        ref.cs_map = pure ? f.cs : channel_mean(f.cs, s);
        if (masked) {
            for (int n = 0; n < s.n; ++n)
                for (size_t i = 0; i < s.plane(); ++i)
                    if (mask[i] <= 0.0) {
                        ref.ssim_map[n * s.plane() + i] = 1.0;
                        ref.cs_map[n * s.plane() + i] = 1.0;
                    }
        }

        Field chain;
        if (masked) {
            double mask_sum = 0.0;
            for (const double m : mask)
                mask_sum += m;
            const double normalized = static_cast<float>(mask_sum) * s.n * s.c + 1e-8;
            chain.resize(s.plane());
            for (size_t i = 0; i < s.plane(); ++i)
                chain[i] = mask[i] / normalized;
            for (int n = 0; n < s.n; ++n)
                for (int c = 0; c < s.c; ++c)
                    for (size_t i = 0; i < s.plane(); ++i) {
                        const size_t e = (static_cast<size_t>(n) * s.c + c) * s.plane() + i;
                        ref.loss += chain[i] * ((1 - w) * std::abs(x[e] - y[e]) + w * (1 - ssim_mean[n * s.plane() + i]));
                    }
        } else {
            chain = valid_chain(s, true);
            double count = 0.0;
            for (int n = 0; n < s.n; ++n)
                for (int c = 0; c < s.c; ++c)
                    for (int py = 0; py < s.h; ++py)
                        for (int px = 0; px < s.w; ++px) {
                            if (!in_mean_crop(px, py, s, true))
                                continue;
                            const size_t i = static_cast<size_t>(py) * s.w + px;
                            const size_t e = (static_cast<size_t>(n) * s.c + c) * s.plane() + i;
                            count += 1.0;
                            ref.loss += pure ? f.ssim[e]
                                             : (1 - w) * std::abs(x[e] - y[e]) + w * (1 - ssim_mean[n * s.plane() + i]);
                        }
            ref.loss /= count;
            if (pure)
                ref.loss = 1.0 - ref.loss;
        }
        if (decoupled) {
            ref.grad = ssim_backward(x, y, f.dm_mu, nullptr, nullptr, chain, w, s, mask);
            ref.grad_raw = ssim_backward(raw, y, f.raw_dm_mu, &f.dm_s1, &f.dm_s12, chain, 1.0, s, mask);
        } else {
            ref.grad = ssim_backward(x, y, f.dm_mu, &f.dm_s1, &f.dm_s12, chain, w, s, mask);
        }
        return ref;
    }

    struct PhotoInputs {
        Tensor x, raw, y, mask;
        Field hx, hraw, hy, hmask;
    };

    PhotoInputs photo_inputs(const PhotoCase& pc, const uint32_t seed) {
        const Shape4& s = pc.shape;
        PhotoInputs in;
        const auto x = uniform(s.numel(), seed);
        auto noise = uniform(s.numel(), seed + 1, -0.15f, 0.15f);
        std::vector<float> y(s.numel());
        for (size_t i = 0; i < y.size(); ++i)
            y[i] = std::clamp(0.8f * x[i] + 0.1f + noise[i], 0.f, 1.f);
        in.x = upload(x, s.tensor());
        in.hx = widen(x);
        if (pc.target_bytes) {
            std::vector<uint8_t> bytes(y.size());
            for (size_t i = 0; i < y.size(); ++i) {
                bytes[i] = static_cast<uint8_t>(std::lround(y[i] * 255.f));
                y[i] = static_cast<float>(bytes[i]) * (1.0f / 255.0f);
            }
            in.y = upload_bytes(bytes, s.tensor());
        } else {
            in.y = upload(y, s.tensor());
        }
        in.hy = widen(y);
        const auto raw = uniform(s.numel(), seed + 2, 0.1f, 0.9f);
        in.raw = upload(raw, s.tensor());
        in.hraw = widen(raw);
        std::vector<float> mask(s.plane());
        std::vector<uint8_t> mask_bytes(s.plane());
        const auto soft = uniform(s.plane(), seed + 3);
        for (size_t i = 0; i < mask.size(); ++i) {
            mask_bytes[i] = soft[i] < 0.3f ? 0 : static_cast<uint8_t>(1 + (i % 200));
            mask[i] = pc.mask_bytes ? (mask_bytes[i] != 0 ? 1.f : 0.f) : (soft[i] < 0.3f ? 0.f : soft[i]);
        }
        const TensorShape plane({static_cast<size_t>(s.h), static_cast<size_t>(s.w)});
        in.mask = pc.mask_bytes ? upload_bytes(mask_bytes, plane) : upload(mask, plane);
        in.hmask = widen(mask);
        return in;
    }

    void check_photo(const lfs::training::TrainingOps& table, const PhotoCase& pc, const std::string& name,
                     const double map_tolerance = 2e-6) {
        SCOPED_TRACE(name);
        const PhotoInputs in = photo_inputs(pc, 11);
        const bool decoupled = pc.path == ops::PhotoPath::Decoupled || pc.path == ops::PhotoPath::MaskedDecoupled;
        const bool masked = pc.path == ops::PhotoPath::MaskedFused || pc.path == ops::PhotoPath::MaskedDecoupled;
        ops::PhotoSaved saved{.backend = table.photometric->create()};
        Tensor loss, grad, grad_raw;
        table.photometric->evaluate(saved, in.x, decoupled ? in.raw : Tensor{}, in.y, masked ? in.mask : Tensor{},
                                    {.path = pc.path, .ssim_weight = pc.weight, .valid_padding = true}, loss, grad,
                                    grad_raw);
        const PhotoRef ref = photo_reference(pc, in.hx, decoupled ? in.hraw : Field{}, in.hy, masked ? in.hmask : Field{});
        expect_value(download(loss)[0], ref.loss, 1e-5, name + " loss");
        expect_field(download(grad), ref.grad, 1e-9, 2e-4, name + " grad");
        if (decoupled)
            expect_field(download(grad_raw), ref.grad_raw, 1e-9, 2e-4, name + " grad_raw");
        else
            EXPECT_FALSE(grad_raw.is_valid());
        if (pc.path != ops::PhotoPath::L1) {
            expect_field(download(saved.ssim_map), ref.ssim_map, map_tolerance, 0.0, name + " ssim_map");
            expect_field(download(saved.cs_map), ref.cs_map, map_tolerance, 0.0, name + " cs_map");
        }
    }

    TEST_P(PortableLosses, PhotometricPathsMatchReference) {
        const Shape4 fixture{1, 3, 32, 40};
        const Shape4 odd{2, 3, 21, 27};
        check_photo(*table_, {ops::PhotoPath::L1, fixture}, "l1");
        check_photo(*table_, {ops::PhotoPath::L1, odd, true}, "l1_u8");
        check_photo(*table_, {ops::PhotoPath::SSIM, fixture}, "ssim");
        check_photo(*table_, {ops::PhotoPath::Fused, fixture}, "fused");
        check_photo(*table_, {ops::PhotoPath::Fused, odd, true}, "fused_batch_u8");
        check_photo(*table_, {ops::PhotoPath::Fused, {1, 3, 9, 30}}, "fused_thin");
        check_photo(*table_, {ops::PhotoPath::Decoupled, fixture}, "decoupled");
        check_photo(*table_, {ops::PhotoPath::MaskedFused, fixture}, "masked_fused");
        check_photo(*table_, {ops::PhotoPath::MaskedFused, odd, true, true}, "masked_fused_u8");
        check_photo(*table_, {ops::PhotoPath::MaskedDecoupled, fixture, false, true}, "masked_decoupled");
        // Soft normalized moments add float32 rounding before covariance
        // subtraction. Keep the original cases and all gradient/loss bounds;
        // this new map comparison allows 3 ppm against the double reference.
        check_photo(*table_, {ops::PhotoPath::MaskedDecoupled, fixture}, "masked_decoupled_soft", 3e-6);
        check_photo(*table_, {ops::PhotoPath::MaskedDecoupled, odd, true, true}, "masked_decoupled_batch_u8");
    }

    // Central differences of the GPU loss 1 - SSIM against the GPU gradient.
    TEST_P(PortableLosses, SsimGradientMatchesFiniteDifferences) {
        const Shape4 s{1, 3, 16, 20};
        const PhotoCase pc{ops::PhotoPath::SSIM, s};
        const PhotoInputs in = photo_inputs(pc, 5);
        const auto x = download(in.x);
        ops::PhotoSaved saved{.backend = table_->photometric->create()};
        const auto evaluate = [&](const std::vector<float>& image, Tensor& grad) {
            Tensor loss, grad_raw;
            table_->photometric->evaluate(saved, upload(image, s.tensor()), {}, in.y, {},
                                          {.path = ops::PhotoPath::SSIM, .ssim_weight = 1.f, .valid_padding = true}, loss,
                                          grad, grad_raw);
            return static_cast<double>(download(loss)[0]);
        };
        Tensor grad;
        evaluate(x, grad);
        const auto analytic = download(grad);
        constexpr float h = 1e-2f;
        const std::tuple<int, int, int> probes[] = {{0, 0, 0}, {0, 3, 4}, {1, 7, 9}, {2, 15, 19}, {1, 5, 5}, {2, 10, 14}, {0, 12, 2}, {1, 8, 17}};
        for (const auto& [c, py, px] : probes) {
            const size_t i = (static_cast<size_t>(c) * s.h + py) * s.w + px;
            auto plus = x, minus = x;
            plus[i] += h;
            minus[i] -= h;
            Tensor scratch;
            const double numeric = (evaluate(plus, scratch) - evaluate(minus, scratch)) / (2.0 * h);
            EXPECT_NEAR(analytic[i], numeric, 2e-6 + 3e-2 * std::abs(numeric))
                << "channel " << c << " pixel (" << px << ", " << py << ")";
        }
    }

    TEST_P(PortableLosses, PhotometricMetricAndErrorMaps) {
        const Shape4 s{1, 3, 32, 40};
        const PhotoCase pc{ops::PhotoPath::SSIM, s};
        const PhotoInputs in = photo_inputs(pc, 23);
        const SsimForward f = ssim_forward(in.hx, {}, in.hy, s, 1.0);
        ops::PhotoSaved saved{.backend = table_->photometric->create()};
        for (const bool padding : {true, false}) {
            double sum = 0.0, count = 0.0;
            for (int c = 0; c < s.c; ++c)
                for (int y = 0; y < s.h; ++y)
                    for (int x = 0; x < s.w; ++x)
                        if (in_mean_crop(x, y, s, padding)) {
                            sum += f.ssim[(static_cast<size_t>(c) * s.h + y) * s.w + x];
                            count += 1.0;
                        }
            for (const bool maps : {false, true}) {
                const Tensor value = table_->photometric->metric(saved, in.x, in.y, maps, padding);
                expect_value(download(value)[0], sum / count, 1e-5, std::format("metric maps={} padding={}", maps, padding));
            }
        }
        expect_field(download(saved.ssim_map), f.ssim, 2e-6, 0.0, "metric ssim_map");
        expect_field(download(saved.cs_map), f.cs, 2e-6, 0.0, "metric cs_map");

        for (const bool cs_only : {false, true}) {
            Tensor error;
            table_->photometric->error_map(saved, in.x, in.y, error, cs_only);
            const Field mean = channel_mean(cs_only ? f.cs : f.ssim, s);
            Field want(mean.size());
            for (size_t i = 0; i < mean.size(); ++i)
                want[i] = std::max(1.0 - mean[i], 0.0);
            expect_field(download(error), want, 2e-6, 0.0, std::format("error_map cs_only={}", cs_only));
        }
        Tensor from_map = Tensor::empty({static_cast<size_t>(s.h), static_cast<size_t>(s.w)}, Device::GPU);
        table_->photometric->map_to_error(saved.ssim_map, from_map);
        Field want(s.plane());
        const Field mean = channel_mean(f.ssim, s);
        for (size_t i = 0; i < want.size(); ++i)
            want[i] = std::max(1.0 - mean[i], 0.0);
        expect_field(download(from_map), want, 2e-6, 0.0, "map_to_error");
    }

    // LossWorkspaceArena accounting: exact on variant switches, a same-variant
    // high-water on shape changes, shrink to the active layout, reset to zero.
    TEST_P(PortableLosses, PhotometricWorkspaceBytesFollowTheArena) {
        const auto layout = [](const std::vector<size_t>& fields) {
            size_t total = 0;
            for (const size_t bytes : fields)
                total = ((total + 255) & ~size_t{255}) + bytes;
            return (total + 255) & ~size_t{255};
        };
        const auto fused_bytes = [&](const Shape4& s) {
            return layout({s.n * s.plane() * 4, s.numel() * 2, s.numel() * 2, s.numel() * 2, s.numel() * 4, 4096, 4});
        };
        // As LossWorkspaceArena::masked_decoupled_layout_bytes: the raw-render gradient is not an arena field.
        const auto masked_decoupled_bytes = [&](const Shape4& s) {
            return layout({s.n * s.plane() * 4, s.numel() * 2, s.numel() * 2, s.numel() * 2, s.numel() * 2,
                           s.numel() * 4, 8192, 4, 4});
        };
        ops::PhotoSaved saved{.backend = table_->photometric->create()};
        const auto run = [&](const ops::PhotoPath path, const Shape4& s) {
            const PhotoInputs in = photo_inputs({path, s}, 3);
            Tensor loss, grad, grad_raw;
            table_->photometric->evaluate(saved, in.x, in.raw, in.y, in.mask,
                                          {.path = path, .ssim_weight = 0.2f, .valid_padding = true}, loss, grad, grad_raw);
            download(loss);
        };
        const Shape4 big{1, 3, 32, 40}, small{1, 3, 16, 20};
        run(ops::PhotoPath::Fused, big);
        auto bytes = table_->photometric->workspace_bytes(saved);
        EXPECT_EQ(bytes.required, fused_bytes(big));
        EXPECT_EQ(bytes.allocated, fused_bytes(big));
        run(ops::PhotoPath::Fused, small);
        bytes = table_->photometric->workspace_bytes(saved);
        EXPECT_EQ(bytes.required, fused_bytes(small));
        EXPECT_EQ(bytes.allocated, fused_bytes(big));
        table_->photometric->shrink_to_required(saved);
        EXPECT_EQ(table_->photometric->workspace_bytes(saved).allocated, fused_bytes(small));
        run(ops::PhotoPath::MaskedDecoupled, small);
        bytes = table_->photometric->workspace_bytes(saved);
        EXPECT_EQ(bytes.required, masked_decoupled_bytes(small));
        EXPECT_EQ(bytes.allocated, masked_decoupled_bytes(small));
        EXPECT_EQ(bytes.error_map, 0u);
        Tensor error;
        table_->photometric->error_map(saved, photo_inputs({ops::PhotoPath::SSIM, big}, 1).x,
                                       photo_inputs({ops::PhotoPath::SSIM, big}, 2).x, error, false);
        EXPECT_EQ(table_->photometric->workspace_bytes(saved).error_map, big.numel() * 4);
        table_->photometric->reset(saved);
        bytes = table_->photometric->workspace_bytes(saved);
        EXPECT_EQ(bytes.required, 0u);
        EXPECT_EQ(bytes.allocated, 0u);
    }

    // SSIM forward + backward at training resolution. Run with
    // --gtest_also_run_disabled_tests.
    TEST_P(PortableLosses, DISABLED_FusedSsimTiming) {
        const Shape4 s{1, 3, 700, 1000};
        const PhotoInputs in = photo_inputs({ops::PhotoPath::Fused, s}, 9);
        ops::PhotoSaved saved{.backend = table_->photometric->create()};
        Tensor loss, grad, grad_raw;
        const auto step = [&] {
            table_->photometric->evaluate(saved, in.x, {}, in.y, {},
                                          {.path = ops::PhotoPath::Fused, .ssim_weight = 0.2f, .valid_padding = true}, loss,
                                          grad, grad_raw);
        };
        for (int i = 0; i < 10; ++i)
            step();
        download(loss);
        constexpr int kSteps = 200;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < kSteps; ++i)
            step();
        download(loss);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::printf("fused L1+SSIM forward+backward 1000x700x3: %.3f ms/step\n", ms / kSteps);
    }

    // ---- Masks (mask_preprocess.cu) --------------------------------------------

    TEST_P(PortableLosses, MasksMatchReference) {
        constexpr size_t h = 33, w = 35, n = h * w;
        const auto soft = uniform(n, 7);
        std::vector<uint8_t> bytes(n);
        std::vector<float> floats(n);
        for (size_t i = 0; i < n; ++i) {
            bytes[i] = static_cast<uint8_t>((i * 79 + 31) % 256);
            floats[i] = static_cast<float>(bytes[i]) / 255.f;
        }
        const auto alpha_h = uniform(n, 8);
        const auto roi_h = uniform(n, 9, 0.2f, 1.f);
        const TensorShape shape({h, w});
        const Tensor alpha = upload(alpha_h, shape), roi = upload(roi_h, shape);
        for (const bool use_bytes : {false, true}) {
            const Tensor mask = use_bytes ? upload_bytes(bytes, shape) : upload(floats, shape);
            for (const bool use_roi : {false, true}) {
                for (const int mode : {0, 1}) {
                    const std::string tag = std::format("bytes={} roi={} mode={}", use_bytes, use_roi, mode);
                    const auto normalized = [&](size_t i) { return use_bytes ? bytes[i] / 255.0 : double(floats[i]); };
                    const auto as_float = [&](size_t i) { return use_bytes ? (bytes[i] != 0 ? 1.0 : 0.0) : double(floats[i]); };
                    const auto roi_at = [&](size_t i) { return use_roi ? double(roi_h[i]) : 1.0; };

                    Tensor weight = Tensor::full({h, w}, -9.f, Device::GPU);
                    table_->masks->photometric_weight(mask, use_roi ? roi : Tensor{}, weight, static_cast<ops::MaskPhotoMode>(mode));
                    Field want(n);
                    for (size_t i = 0; i < n; ++i)
                        want[i] = (mode == 1 ? (normalized(i) > k::kMaskKeepMin ? 1.0 : 0.0) : as_float(i)) * roi_at(i);
                    expect_field(download(weight), want, 1e-7, 0.0, "photometric weight " + tag);

                    constexpr float power = 1.7f, scale = 0.13f;
                    Tensor grad = Tensor::full({h, w}, -9.f, Device::GPU), temp = Tensor::zeros({1024}, Device::GPU);
                    Tensor loss = Tensor::full({1}, -9.f, Device::GPU);
                    table_->masks->opacity_penalty(alpha, mask, use_roi ? roi : Tensor{}, grad, temp, loss,
                                                   static_cast<ops::MaskOpacityMode>(mode), power, scale);
                    double sum = 0.0;
                    for (size_t i = 0; i < n; ++i) {
                        const double bg = mode == 1 ? (normalized(i) >= k::kMaskSegmentMin && normalized(i) <= k::kMaskKeepMin ? 1.0 : 0.0)
                                                    : 1.0 - as_float(i);
                        const double pen = (bg <= 0 ? 0.0 : (bg >= 1 ? 1.0 : std::pow(bg, double(power)))) * roi_at(i);
                        want[i] = pen * (scale / static_cast<float>(n));
                        sum += alpha_h[i] * pen;
                    }
                    expect_field(download(grad), want, 1e-9, 1e-5, "opacity grad " + tag);
                    expect_value(download(loss)[0], scale * sum / n, 1e-5, "opacity loss " + tag);

                    table_->masks->alpha_consistency(alpha, mask, use_roi ? roi : Tensor{}, grad, temp, loss, 10.f);
                    sum = 0.0;
                    for (size_t i = 0; i < n; ++i) {
                        const double d = alpha_h[i] - as_float(i);
                        want[i] = sign(d) * roi_at(i) * (10.0f / static_cast<float>(n));
                        sum += std::abs(d) * roi_at(i);
                    }
                    expect_field(download(grad), want, 1e-9, 1e-6, "alpha grad " + tag);
                    expect_value(download(loss)[0], 10.0 * sum / n, 1e-5, "alpha loss " + tag);
                }
            }
        }
        Tensor grad = Tensor::full({h, w}, -9.f, Device::GPU), temp = Tensor::zeros({1024}, Device::GPU);
        Tensor loss = Tensor::full({1}, -9.f, Device::GPU);
        table_->masks->opacity_penalty(alpha, upload(floats, shape), roi, grad, temp, loss, ops::MaskOpacityMode::BinaryGt0,
                                       1.f, 0.f);
        expect_field(download(grad), Field(n, 0.0), 0.0, 0.0, "zero-scale grad");
        EXPECT_EQ(download(loss)[0], 0.f);
    }

    // ---- Extra loss (regularization.cu) ------------------------------------------

    TEST_P(PortableLosses, ExtraLossMatchesReference) {
        constexpr size_t n = 1027;
        for (const auto kind : {ops::Regularizer::Scale, ops::Regularizer::Opacity}) {
            const size_t attrs = kind == ops::Regularizer::Scale ? 3 : 1;
            const auto raw_h = uniform(n * attrs, 4, -2.f, 2.f);
            const auto grad_h = uniform(n * attrs, 5, -0.5f, 0.5f);
            const Tensor raw = upload(raw_h, {n, attrs});
            for (const bool with_grad : {true, false}) {
                Tensor gradient = with_grad ? upload(grad_h, {n, attrs}) : Tensor{};
                Tensor loss = Tensor::full({1}, -9.f, Device::GPU), temp = Tensor::zeros({1024}, Device::GPU);
                constexpr float weight = 0.013f;
                table_->extra_loss->regularize(raw, gradient, loss, temp, kind, weight);
                const size_t count = n * attrs;
                Field want(count);
                double sum = 0.0;
                for (size_t i = 0; i < count; ++i) {
                    const double value = kind == ops::Regularizer::Scale ? std::exp(double(raw_h[i])) : 1.0 / (1.0 + std::exp(-double(raw_h[i])));
                    const double derivative = kind == ops::Regularizer::Scale ? value : value * (1.0 - value);
                    sum += value;
                    want[i] = grad_h[i] + weight / static_cast<float>(count) * derivative;
                }
                const std::string tag = std::format("regularizer={} gradient={}", static_cast<int>(kind), with_grad);
                expect_value(download(loss)[0], weight * sum / count, 1e-5, "loss " + tag);
                if (with_grad)
                    expect_field(download(gradient), want, 1e-7, 1e-6, "grad " + tag);
            }
        }
        Tensor loss = Tensor::full({1}, -9.f, Device::GPU), temp = Tensor::zeros({1024}, Device::GPU), none;
        table_->extra_loss->regularize(upload(uniform(n, 1), {n, 1}), none, loss, temp, ops::Regularizer::Scale, 0.f);
        EXPECT_EQ(download(loss)[0], -9.f) << "zero weight leaves the loss untouched";
    }

    // ---- Geometry (depth_loss.cu, normal_loss.cu, normal_consistency_loss.cu) --

    bool finite(const double v) { return std::isfinite(v); }

    struct GeomInputs {
        int w = 0, h = 0;
        std::vector<float> depth, alpha, target, weight, normal, prior;
    };

    GeomInputs geom_inputs(const int w, const int h, const bool sparse_weight) {
        GeomInputs in{w, h};
        const size_t n = static_cast<size_t>(w) * h;
        in.alpha = uniform(n, 31, 0.55f, 1.f);
        in.depth.resize(n);
        in.target = uniform(n, 32, 0.2f, 0.7f);
        in.weight = uniform(n, 33, 0.f, 1.5f);
        in.normal.resize(3 * n);
        in.prior.resize(3 * n);
        const auto jitter = uniform(3 * n, 34, -0.3f, 0.3f);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const size_t i = static_cast<size_t>(y) * w + x;
                const float e = 2.f + 0.3f * std::sin(x / 7.f) + 0.2f * std::cos(y / 5.f) + 0.01f * jitter[i];
                in.depth[i] = e * in.alpha[i];
                if (sparse_weight && (x + y) % 7 == 0)
                    in.weight[i] = 0.f;
                in.normal[i] = 0.2f + jitter[i];
                in.normal[n + i] = -0.1f + jitter[n + i];
                in.normal[2 * n + i] = -0.9f + 0.5f * jitter[2 * n + i];
                in.prior[i] = 0.1f - jitter[2 * n + i];
                in.prior[n + i] = 0.3f + jitter[i];
                in.prior[2 * n + i] = -0.8f + jitter[n + i];
            }
        in.alpha[5] = 0.f; // inactive pixels
        in.target[9] = -1.f;
        return in;
    }

    struct DepthRef {
        std::array<double, 10> finals{};
        double loss = 0.0;
        Field grad_depth, grad_alpha;
    };

    DepthRef depth_reference(const GeomInputs& in, const bool weighted, const ops::DepthParams& p) {
        namespace slots = k::depth_loss_slots;
        const size_t n = static_cast<size_t>(in.w) * in.h;
        const auto active = [&](const size_t i) {
            const double t = in.target[i], d = in.depth[i], a = in.alpha[i];
            return t > 0 && a > k::kDepthLossMinAlpha && finite(t) && finite(d) && finite(a);
        };
        const auto pixel_weight = [&](const size_t i) {
            return !weighted ? 1.0 : (finite(in.weight[i]) && in.weight[i] > 0 ? double(in.weight[i]) : 0.0);
        };
        const auto expected = [&](const size_t i) { return std::max(double(in.depth[i]), 0.0) / in.alpha[i]; };
        DepthRef ref;
        double sum_alpha = 0, sum_e = 0, count = 0;
        for (size_t i = 0; i < n; ++i)
            if (active(i) && pixel_weight(i) != 0) {
                const double aw = in.alpha[i] * pixel_weight(i);
                sum_alpha += aw;
                sum_e += aw * expected(i);
                count += 1;
            }
        bool valid = sum_alpha > 0;
        const double mean_e = valid ? sum_e / sum_alpha : 0;
        ref.finals[slots::kFloor] = std::max(1e-8, k::kDepthLossFloorFraction * mean_e);
        const bool anchor = p.anchor && p.anchor->valid;
        const double floor_f = anchor && p.anchor->floor > 0 ? p.anchor->floor : ref.finals[slots::kFloor];
        double p1 = 0, p2 = 0;
        if (valid)
            for (size_t i = 0; i < n; ++i)
                if (active(i) && pixel_weight(i) != 0) {
                    const double inv = 1.0 / (expected(i) + floor_f), aw = in.alpha[i] * pixel_weight(i);
                    p1 += aw * inv;
                    p2 += aw * inv * inv;
                }
        const double a_floor = anchor ? p.anchor->floor : 0, a_scale = anchor ? p.anchor->scale : 0;
        const double a_shift = anchor ? p.anchor->shift : 0;
        valid = valid && a_floor > 0;
        double sigma = 0;
        if (valid) {
            const double mean = p1 / sum_alpha;
            sigma = std::sqrt(std::max(std::max(p2 / sum_alpha - mean * mean, 0.0), 1e-20));
            ref.finals[slots::kFloor] = a_floor;
        }
        ref.finals[slots::kValid] = valid ? 1 : 0;
        ref.finals[slots::kModel] = anchor ? p.anchor->model : 0;
        ref.finals[slots::kScale] = a_scale;
        ref.finals[slots::kShift] = a_shift;
        ref.finals[slots::kSigmaP] = sigma;
        ref.finals[slots::kInvNorm] = valid ? 1.0 / sum_alpha : 0;
        ref.finals[slots::kSumAlpha] = sum_alpha;
        ref.finals[slots::kCount] = count;
        ref.finals[slots::kMeanExpectedDepth] = mean_e;

        const double fl = ref.finals[slots::kFloor], inv_norm = ref.finals[slots::kInvNorm];
        const double inv_sigma = valid && sigma > 0 ? 1.0 / (k::kDepthLossResidualScale * sigma) : 0;
        const double p_max = 1.0 / fl, half_step = 0.5 * std::max(p.prior_quantization_step, 0.f) * std::abs(a_scale);
        const int model = anchor ? p.anchor->model : 0;
        struct S {
            bool ok = false;
            double alpha = 0, wa = 0, e = 0, p = 0, d = 0, delta = 0;
        };
        const auto sample = [&](const size_t i) {
            S s;
            if (!active(i) || pixel_weight(i) == 0)
                return s;
            s.alpha = in.alpha[i];
            s.wa = in.alpha[i] * pixel_weight(i);
            s.e = expected(i);
            s.p = 1.0 / (s.e + fl);
            const double fit = a_scale * in.target[i] + a_shift;
            if (!(fit > 0))
                return s;
            if (model == 0) {
                s.d = std::min(fit, p_max);
                s.delta = half_step;
            } else {
                s.d = 1.0 / (fit + fl);
                s.delta = half_step * s.d * s.d;
            }
            s.ok = true;
            return s;
        };
        const auto rho = [](const double x) { return 0.5 * x * x / (1 + x * x); };
        const auto psi = [](const double x) { return x / ((1 + x * x) * (1 + x * x)); };
        const auto deadband = [](const double r, const double delta) {
            const double excess = std::abs(r) - delta;
            return excess > 0 ? sign(r) * excess : 0.0;
        };
        ref.grad_depth.assign(n, 0.0);
        ref.grad_alpha.assign(n, 0.0);
        double l0 = 0, l1 = 0;
        const double lambda = p.gradient_weight;
        for (size_t i = 0; valid && i < n; ++i) {
            const int x = static_cast<int>(i % in.w), y = static_cast<int>(i / in.w);
            const S c = sample(i);
            if (!c.ok)
                continue;
            double gp = 0;
            const double xr = deadband(c.p - c.d, c.delta) * inv_sigma;
            if (xr != 0) {
                l0 += c.wa * rho(xr);
                gp += c.wa * psi(xr) * inv_sigma;
            }
            const auto forward = [&](const size_t j) {
                const S nb = sample(j);
                if (!nb.ok)
                    return;
                const double xh = deadband((nb.p - c.p) - (nb.d - c.d), c.delta + nb.delta) * inv_sigma;
                if (xh != 0) {
                    const double w2 = std::min(c.wa, nb.wa);
                    l1 += w2 * rho(xh);
                    gp -= lambda * w2 * psi(xh) * inv_sigma;
                }
            };
            const auto backward = [&](const size_t j) {
                const S nb = sample(j);
                if (!nb.ok)
                    return;
                const double xh = deadband((c.p - nb.p) - (c.d - nb.d), c.delta + nb.delta) * inv_sigma;
                if (xh != 0)
                    gp += lambda * std::min(nb.wa, c.wa) * psi(xh) * inv_sigma;
            };
            if (x + 1 < in.w)
                forward(i + 1);
            if (y + 1 < in.h)
                forward(i + in.w);
            if (x > 0)
                backward(i - 1);
            if (y > 0)
                backward(i - in.w);
            if (gp != 0) {
                const double g = p.weight * inv_norm * gp * (-c.p * c.p);
                ref.grad_depth[i] = g / c.alpha;
                ref.grad_alpha[i] = -g * c.e / c.alpha;
            }
        }
        ref.loss = valid ? p.weight * inv_norm * (l0 + lambda * l1) : 0;
        return ref;
    }

    void expect_finals(const std::vector<float>& partials, const double* want, const size_t count, const std::string& what) {
        for (size_t i = 0; i < count; ++i)
            EXPECT_NEAR(partials[i], want[i], 1e-6 + 2e-5 * std::abs(want[i])) << what << " slot " << i;
    }

    TEST_P(PortableLosses, DepthLossMatchesReference) {
        const GeomInputs in = geom_inputs(41, 37, true);
        const size_t n = static_cast<size_t>(in.w) * in.h;
        const TensorShape plane({static_cast<size_t>(in.h), static_cast<size_t>(in.w)});
        k::DepthAnchor anchor;
        anchor.valid = true;
        anchor.scale = 1.1f;
        anchor.shift = 0.05f;
        anchor.floor = 0.01f;
        const std::tuple<int, float, bool> cases[] = {{0, 0.f, true}, {0, 1.f / 255.f, false}, {1, 1.f / 255.f, true}};
        for (const auto& [model, step, weighted] : cases) {
            anchor.model = model;
            const ops::DepthParams params{0.5f, 0.25f, step, &anchor};
            Tensor gd = Tensor::full(plane, 3.f, Device::GPU), ga = Tensor::full(plane, 3.f, Device::GPU);
            Tensor loss = Tensor::zeros({1}, Device::GPU);
            Tensor partials = Tensor::zeros({k::depth_loss_partial_count(n)}, Device::GPU);
            table_->geometry->depth(upload(in.depth, plane), upload(in.alpha, plane), upload(in.target, plane),
                                    weighted ? upload(in.weight, plane) : Tensor{}, gd, ga, loss, partials, params);
            const DepthRef ref = depth_reference(in, weighted, params);
            const std::string tag = std::format("model={} step={} weighted={}", model, step, weighted);
            expect_finals(download(partials), ref.finals.data(), ref.finals.size(), "depth finals " + tag);
            expect_value(download(loss)[0], ref.loss, 2e-5, "depth loss " + tag);
            expect_field(download(gd), ref.grad_depth, 1e-12, 1e-4, "depth grad_depth " + tag);
            expect_field(download(ga), ref.grad_alpha, 1e-12, 1e-4, "depth grad_alpha " + tag);
        }
        // An invalid anchor zeroes the gradients and the loss.
        anchor.valid = false;
        Tensor gd = Tensor::full(plane, 3.f, Device::GPU), ga = Tensor::full(plane, 3.f, Device::GPU);
        Tensor loss = Tensor::full({1}, 3.f, Device::GPU);
        Tensor partials = Tensor::zeros({k::depth_loss_partial_count(n)}, Device::GPU);
        table_->geometry->depth(upload(in.depth, plane), upload(in.alpha, plane), upload(in.target, plane), {}, gd, ga,
                                loss, partials, {0.5f, 0.25f, 0.f, &anchor});
        expect_field(download(gd), Field(n, 0.0), 0.0, 0.0, "invalid anchor grad_depth");
        expect_field(download(ga), Field(n, 0.0), 0.0, 0.0, "invalid anchor grad_alpha");
        EXPECT_EQ(download(loss)[0], 0.f);
    }

    struct Vec3 {
        double x, y, z;
    };
    Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    Vec3 operator*(double s, Vec3 a) { return {s * a.x, s * a.y, s * a.z}; }
    double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    double norm(Vec3 a) { return std::sqrt(dot(a, a)); }

    Vec3 vec_at(const std::vector<float>& f, const size_t n, const size_t i) { return {f[i], f[n + i], f[2 * n + i]}; }

    struct NormalRef {
        std::array<double, 5> finals{};
        double loss = 0.0;
        Field grad_normal, grad_depth, grad_alpha;
    };

    // Shared finalize of normal_loss.cu and normal_consistency_loss.cu.
    void normal_finals(NormalRef& ref, const double sum_alpha, const double count, const double sum_cos,
                       const double weight, const double min_count, const double min_weight) {
        const bool valid = count >= min_count && sum_alpha >= min_weight;
        ref.finals = {valid ? 1.0 : 0.0, sum_alpha, count, sum_alpha > 0 ? sum_cos / sum_alpha : 0.0,
                      valid ? weight / std::max(sum_alpha, 1.0) : 0.0};
    }

    NormalRef normal_reference(const GeomInputs& in, const std::vector<float>& target, const bool weighted,
                               const double weight) {
        const size_t n = static_cast<size_t>(in.w) * in.h;
        struct S {
            bool active = false;
            double alpha = 0, cos = 0, norm = 0;
            Vec3 t{}, r{};
        };
        const auto sample = [&](const size_t i) {
            S s;
            const double a = in.alpha[i];
            const Vec3 t = vec_at(target, n, i), r = vec_at(in.normal, n, i);
            if (a <= k::kNormalLossMinAlpha)
                return s;
            double eff = a;
            if (weighted) {
                if (!(in.weight[i] > 0))
                    return s;
                eff *= in.weight[i];
            }
            if (norm(t) < k::kNormalLossMinPriorNorm || norm(r) < k::kNormalLossMinRenderNorm)
                return s;
            s.t = (1 / norm(t)) * t;
            s.r = (1 / norm(r)) * r;
            s.cos = dot(s.t, s.r);
            s.alpha = eff;
            s.norm = norm(r);
            s.active = true;
            return s;
        };
        NormalRef ref;
        double sa = 0, cnt = 0, sc = 0;
        for (size_t i = 0; i < n; ++i)
            if (const S s = sample(i); s.active) {
                sa += s.alpha;
                cnt += 1;
                sc += s.alpha * s.cos;
            }
        normal_finals(ref, sa, cnt, sc, weight, k::kNormalLossMinValidCount, k::kNormalLossMinValidWeight);
        ref.grad_normal.assign(3 * n, 0.0);
        const double inv = ref.finals[4];
        double sum = 0;
        for (size_t i = 0; i < n; ++i) {
            const S s = sample(i);
            if (!s.active || inv == 0)
                continue;
            sum += s.alpha * (1 - s.cos);
            const Vec3 g = (-inv * s.alpha / s.norm) * (s.t - s.cos * s.r);
            ref.grad_normal[i] = g.x;
            ref.grad_normal[n + i] = g.y;
            ref.grad_normal[2 * n + i] = g.z;
        }
        ref.loss = inv * sum;
        return ref;
    }

    // Consistency (prior=false) and prior depth (prior=true), added onto the
    // given starting gradients.
    NormalRef depth_normal_reference(const GeomInputs& in, const bool prior, const bool weighted, const ops::Intrinsics& kk,
                                     const double weight, const Field& g_normal0, const Field& g_depth0, const Field& g_alpha0) {
        const int W = in.w, H = in.h;
        const size_t n = static_cast<size_t>(W) * H;
        const auto ray = [&](const int x, const int y) {
            return Vec3{(x + 0.5 - kk.cx) / kk.fx, (y + 0.5 - kk.cy) / kk.fy, 1.0};
        };
        const auto expected = [&](const size_t i, double& e, double& a) {
            a = in.alpha[i];
            if (a < k::kNormalConsistencyMinAlpha)
                return false;
            e = std::max(double(in.depth[i]), 0.0) / a;
            return e >= 1e-6;
        };
        struct S {
            bool active = false;
            double alpha = 0, sign = 0, raw_norm = 0, cos = 0, aw = 0, render_norm = 0;
            Vec3 nd{}, tx{}, ty{}, hat{};
            std::array<double, 4> e{}, a{};
        };
        const auto sample = [&](const int x, const int y) {
            S s;
            if (x <= 0 || y <= 0 || x >= W - 1 || y >= H - 1)
                return s;
            const size_t i = static_cast<size_t>(y) * W + x;
            double ec, ac;
            if (!expected(i, ec, ac))
                return s;
            const size_t nb[4] = {i + 1, i - 1, i + W, i - W};
            for (int q = 0; q < 4; ++q)
                if (!expected(nb[q], s.e[q], s.a[q]) || std::abs(s.e[q] - ec) > k::kNormalConsistencyMaxRelDepthJump * ec)
                    return s;
            s.tx = s.e[0] * ray(x + 1, y) - s.e[1] * ray(x - 1, y);
            s.ty = s.e[2] * ray(x, y + 1) - s.e[3] * ray(x, y - 1);
            const Vec3 raw = cross(s.tx, s.ty);
            if (dot(raw, raw) < 1e-24)
                return s;
            s.sign = dot(raw, ray(x, y)) > 0 ? -1.0 : 1.0;
            s.raw_norm = norm(raw);
            s.nd = (s.sign / s.raw_norm) * raw;
            s.alpha = ac;
            const Vec3 other = vec_at(prior ? in.prior : in.normal, n, i);
            const double other_norm = norm(other);
            if (other_norm < (prior ? k::kNormalLossMinPriorNorm : k::kNormalLossMinRenderNorm))
                return s;
            s.hat = (1 / other_norm) * other;
            s.render_norm = other_norm;
            s.cos = dot(s.nd, s.hat);
            s.aw = s.alpha * (!weighted ? 1.0 : (in.weight[i] > 0 ? double(in.weight[i]) : 0.0));
            s.active = s.aw != 0;
            return s;
        };
        NormalRef ref;
        double sa = 0, cnt = 0, sc = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (const S s = sample(x, y); s.active) {
                    sa += s.aw;
                    cnt += 1;
                    sc += s.aw * s.cos;
                }
        normal_finals(ref, sa, cnt, sc, weight, k::kNormalConsistencyMinValidCount, k::kNormalConsistencyMinValidWeight);
        ref.grad_normal = g_normal0;
        ref.grad_depth = g_depth0;
        ref.grad_alpha = g_alpha0;
        const double inv = ref.finals[4];
        double sum = 0;
        for (int y = 0; y < H && inv != 0; ++y)
            for (int x = 0; x < W; ++x) {
                const S s = sample(x, y);
                if (!s.active)
                    continue;
                const size_t i = static_cast<size_t>(y) * W + x;
                sum += s.aw * (1 - s.cos);
                const double gw = inv * s.aw;
                if (!prior) {
                    const Vec3 g = (-gw / s.render_norm) * (s.nd - s.cos * s.hat);
                    ref.grad_normal[i] += g.x;
                    ref.grad_normal[n + i] += g.y;
                    ref.grad_normal[2 * n + i] += g.z;
                }
                const Vec3 g_raw = (-gw * s.sign / s.raw_norm) * (s.hat - s.cos * s.nd);
                const Vec3 g_tx = cross(s.ty, g_raw), g_ty = cross(g_raw, s.tx);
                const double g_e[4] = {dot(g_tx, ray(x + 1, y)), -dot(g_tx, ray(x - 1, y)), dot(g_ty, ray(x, y + 1)),
                                       -dot(g_ty, ray(x, y - 1))};
                const size_t nb[4] = {i + 1, i - 1, i + W, i - W};
                for (int q = 0; q < 4; ++q) {
                    ref.grad_depth[nb[q]] += g_e[q] / s.a[q];
                    ref.grad_alpha[nb[q]] += -g_e[q] * s.e[q] / s.a[q];
                }
            }
        ref.loss = inv * sum;
        return ref;
    }

    TEST_P(PortableLosses, NormalLossesMatchReference) {
        const GeomInputs in = geom_inputs(38, 29, true);
        const size_t n = static_cast<size_t>(in.w) * in.h;
        const TensorShape plane({static_cast<size_t>(in.h), static_cast<size_t>(in.w)});
        const TensorShape planes({3, static_cast<size_t>(in.h), static_cast<size_t>(in.w)});
        const ops::Intrinsics kk{45.f, 43.f, 19.5f, 14.f};
        for (const bool weighted : {true, false}) {
            const std::string tag = std::format("weighted={}", weighted);
            const Tensor weight = weighted ? upload(in.weight, plane) : Tensor{};
            {
                Tensor gn = Tensor::full(planes, 3.f, Device::GPU), loss = Tensor::zeros({1}, Device::GPU);
                Tensor partials = Tensor::zeros({k::normal_loss_partial_count(n)}, Device::GPU);
                table_->geometry->normal(upload(in.normal, planes), upload(in.alpha, plane), upload(in.prior, planes), weight,
                                         gn, loss, partials, 0.5f);
                const NormalRef ref = normal_reference(in, in.prior, weighted, 0.5);
                expect_finals(download(partials), ref.finals.data(), ref.finals.size(), "normal finals " + tag);
                expect_value(download(loss)[0], ref.loss, 2e-5, "normal loss " + tag);
                expect_field(download(gn), ref.grad_normal, 1e-12, 1e-5, "normal grad " + tag);
            }
            const auto g_normal0 = uniform(3 * n, 40, -0.01f, 0.01f), g_depth0 = uniform(n, 41, -0.01f, 0.01f);
            const auto g_alpha0 = uniform(n, 42, -0.01f, 0.01f);
            for (const bool prior : {false, true}) {
                const std::string ptag = std::format("{} {}", prior ? "prior" : "consistency", tag);
                Tensor gn = upload(g_normal0, planes), gd = upload(g_depth0, plane), ga = upload(g_alpha0, plane);
                Tensor loss = Tensor::zeros({1}, Device::GPU);
                Tensor partials = Tensor::zeros({k::normal_consistency_partial_count(n)}, Device::GPU);
                if (prior)
                    table_->geometry->prior_depth(upload(in.prior, planes), upload(in.depth, plane), upload(in.alpha, plane),
                                                  weight, gd, ga, loss, partials, kk, 0.5f);
                else
                    table_->geometry->consistency(upload(in.normal, planes), upload(in.depth, plane), upload(in.alpha, plane),
                                                  weight, gn, gd, ga, loss, partials, kk, 0.5f);
                const NormalRef ref = depth_normal_reference(in, prior, weighted, kk, 0.5, widen(g_normal0), widen(g_depth0),
                                                             widen(g_alpha0));
                expect_finals(download(partials), ref.finals.data(), ref.finals.size(), ptag + " finals");
                EXPECT_GT(ref.finals[0], 0.5) << ptag << " fixture must be valid";
                expect_value(download(loss)[0], ref.loss, 2e-5, ptag + " loss");
                expect_field(download(gn), ref.grad_normal, 1e-12, 1e-4, ptag + " grad_normal");
                expect_field(download(gd), ref.grad_depth, 1e-12, 1e-4, ptag + " grad_depth");
                expect_field(download(ga), ref.grad_alpha, 1e-12, 1e-4, ptag + " grad_alpha");
            }
        }
    }

    TEST_P(PortableLosses, AnchorSamplesMatchProjection) {
        constexpr int W = 40, H = 30;
        const auto prior_h = uniform(static_cast<size_t>(W) * H, 50, 0.1f, 2.f);
        std::vector<float> xyz;
        for (int i = 0; i < 900; ++i) {
            // Quarter-pixel offsets keep every projection off a pixel boundary.
            const float u = static_cast<float>(i % 45) - 2.25f, v = static_cast<float>(i / 45) * 1.5f - 1.75f;
            const float z = 1.5f + 0.01f * static_cast<float>(i % 13);
            xyz.insert(xyz.end(), {(u - 20.f) * z / 40.f, (v - 15.f) * z / 40.f, z - 0.5f});
        }
        xyz[0] = std::numeric_limits<float>::quiet_NaN();
        const std::vector<float> view{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.5f, 0, 0, 0, 1};
        const ops::AnchorParams params{{40, 40, 20, 15}, 0.01f, {-10, -10, -10}, {10, 10, 1.9f}};
        auto got = table_->geometry->collect_anchor_samples(upload(xyz, {xyz.size() / 3, 3}), upload(view, {4, 4}),
                                                            upload(prior_h, {H, W}), params);
        std::vector<std::pair<float, float>> want;
        for (size_t i = 0; i < xyz.size() / 3; ++i) {
            const double x = xyz[3 * i], y = xyz[3 * i + 1], zw = xyz[3 * i + 2];
            if (!finite(x) || zw > params.aabb_hi[2])
                continue;
            const double z = zw + 0.5;
            const int u = static_cast<int>(std::floor(40 * x / z + 20)), v = static_cast<int>(std::floor(40 * y / z + 15));
            if (u < 0 || u >= W || v < 0 || v >= H)
                continue;
            want.emplace_back(prior_h[static_cast<size_t>(v) * W + u], static_cast<float>(z));
        }
        ASSERT_GE(want.size(), static_cast<size_t>(k::kMinAnchorSamples));
        ASSERT_EQ(got.size(), want.size());
        std::sort(got.begin(), got.end(), [](auto a, auto b) { return std::tie(a.x, a.y) < std::tie(b.x, b.y); });
        std::sort(want.begin(), want.end());
        for (size_t i = 0; i < want.size(); ++i) {
            EXPECT_EQ(got[i].x, want[i].first) << i;
            EXPECT_NEAR(got[i].y, want[i].second, 1e-6) << i;
        }
    }

    // ---- Blur-tolerant SSIM reference (ssim.cu blurTolerantForwardCUDA / blurTolerantBackwardCUDA) ----------
    // MRNF's loss for photos with a short side of 1,200 pixels or more, in double: the sharp and the blurred
    // decoupled SSIM branch of each window from the same photo statistics, their soft maximum, the blurred branch's
    // backward and the gradient with the adjoint blur. Partials and the branch pass's (v, u) are rounded to half
    // where the GPU stores them.

    // The host blur taps of a sigma (centre first), rounded to float as the kernels read them.
    std::vector<double> blur_taps_reference(const float sigma) {
        const int radius = static_cast<int>(std::ceil(3.0 * sigma));
        std::vector<double> raw(radius + 1), taps(2 * radius + 1);
        double sum = 0.0;
        for (int d = 0; d <= radius; ++d) {
            raw[d] = std::exp(-0.5 * d * d / (static_cast<double>(sigma) * sigma));
            sum += d == 0 ? raw[d] : 2.0 * raw[d];
        }
        for (int d = -radius; d <= radius; ++d)
            taps[d + radius] = static_cast<float>(raw[std::abs(d)] / sum);
        return taps;
    }

    // Separable filter with symmetric taps (2R+1) over one [H,W] plane; samples outside the image read `outside`.
    Field filter(const Field& f, const int H, const int W, const std::vector<double>& taps, const double outside = 0.0) {
        const int R = static_cast<int>(taps.size() / 2);
        Field horizontal(static_cast<size_t>(H + 2 * R) * W, 0.0), out(static_cast<size_t>(H) * W, 0.0);
        for (int y = -R; y < H + R; ++y)
            for (int x = 0; x < W; ++x) {
                double sum = 0.0;
                for (int d = -R; d <= R; ++d) {
                    const bool in = y >= 0 && y < H && x + d >= 0 && x + d < W;
                    sum += taps[d + R] * (in ? f[static_cast<size_t>(y) * W + x + d] : outside);
                }
                horizontal[static_cast<size_t>(y + R) * W + x] = sum;
            }
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int d = -R; d <= R; ++d)
                    out[static_cast<size_t>(y) * W + x] += taps[d + R] * horizontal[static_cast<size_t>(y + d + R) * W + x];
        return out;
    }

    std::vector<double> window_taps() { return {kGauss.begin(), kGauss.end()}; }

    // One decoupled branch (bt_branch): returns S; cs and the partials {raw mean, raw sigma^2, raw sigma12,
    // corrected mean}.
    double blur_branch_reference(const double mc, const double mr, const double mrr, const double my, const double myy,
                                 const double mry, const double weight, double& cs, std::array<double, 4>& partial) {
        const double A = mc * mc + my * my + kC1, C = 2 * mc * my + kC1, lum = C / A;
        const double B = (mrr - mr * mr) + (myy - my * my) + kC2, D = 2 * (mry - mr * my) + kC2;
        cs = D / B;
        const double ds1 = weight * (-(lum * D) / (B * B)), ds12 = weight * (2 * lum / B);
        partial = {ds1 * (-2 * mr) + ds12 * (-my), ds1, ds12, cs * (2 * (my * A - mc * C) / (A * A))};
        return lum * cs;
    }

    struct BlurCase {
        Shape4 shape;
        bool decoupled = false;
        int mask = 0; // 0 none, 1 soft float weights (some above one), 2 8-bit
        bool target_bytes = false;
        float sigma = 1.5f;
        float weight = 0.22f;
        float denominator = 0.f;
        int photo = 0; // 0 noisy copy of the render, 1 a blur of the render, 2 noisy copy with a constant block
    };

    struct BlurRef : PhotoRef {
        // [N,H,W]: in every channel the soft maximum is decided (|d| >= 5, the smaller weight below 0.7%) and the
        // photo window is not within 1e-4 of the flat threshold.
        std::vector<uint8_t> decided;
    };

    BlurRef blur_reference(const BlurCase& bc, const PhotoInputs& in) {
        const Shape4& s = bc.shape;
        const bool masked = bc.mask != 0;
        const double sw = bc.weight;
        const std::vector<double> T = blur_taps_reference(bc.sigma), Wn = window_taps();
        const size_t P = s.plane();
        const Field& x = in.hx;
        const Field& raw = bc.decoupled ? in.hraw : in.hx;
        const Field& y = in.hy;
        const auto inside_valid = [&](const int py, const int px) {
            return masked || ((s.h <= 10 || (py >= 5 && py < s.h - 5)) && (s.w <= 10 || (px >= 5 && px < s.w - 5)));
        };
        Field weight(P, 1.0), keep(P, 1.0);
        if (masked)
            for (size_t i = 0; i < P; ++i) {
                weight[i] = in.hmask[i];
                keep[i] = in.hmask[i] > 0.0 ? 1.0 : 0.0;
            }
        // Window mass with unit weight outside the image (masked); unmasked windows are not normalised.
        Field inv_window(P, 1.0);
        if (masked) {
            const Field mass = filter(weight, s.h, s.w, Wn, 1.0);
            for (size_t i = 0; i < P; ++i)
                inv_window[i] = mass[i] >= 1e-30 ? 1.0 / mass[i] : 0.0;
        }
        const Field keep_sum = filter(keep, s.h, s.w, T);
        Field support(P);
        for (size_t i = 0; i < P; ++i)
            support[i] = keep_sum[i] > 1e-30 ? 1.0 / keep_sum[i] : 0.0;
        const auto moment = [&](const Field& v) {
            Field out = filter(masked ? product(v, weight) : v, s.h, s.w, Wn);
            for (size_t i = 0; i < P; ++i)
                out[i] *= inv_window[i];
            return out;
        };
        const auto blurred = [&](const Field& v) {
            Field out = filter(product(v, keep), s.h, s.w, T);
            for (size_t i = 0; i < P; ++i)
                out[i] *= support[i];
            return out;
        };

        BlurRef ref;
        ref.grad.assign(s.numel(), 0.0);
        ref.grad_raw.assign(bc.decoupled ? s.numel() : 0, 0.0);
        ref.ssim_map.assign(static_cast<size_t>(s.n) * P, 0.0);
        ref.cs_map.assign(static_cast<size_t>(s.n) * P, 0.0);
        Field blend_map(static_cast<size_t>(s.n) * P, 0.0);
        ref.decided.assign(static_cast<size_t>(s.n) * P, 1);
        std::vector<std::array<Field, 4>> sharp(static_cast<size_t>(s.n) * s.c), soft(static_cast<size_t>(s.n) * s.c);
        for (int n = 0; n < s.n; ++n)
            for (int c = 0; c < s.c; ++c) {
                const Field px = product(plane_of(x, s, n, c), keep), pr = product(plane_of(raw, s, n, c), keep);
                const Field py = plane_of(y, s, n, c);
                const Field gx = blurred(plane_of(x, s, n, c)), gr = blurred(plane_of(raw, s, n, c));
                const Field mx = moment(px), mr = moment(pr), mrr = moment(product(pr, pr)), mry = moment(product(pr, py));
                const Field my = moment(py), myy = moment(product(py, py));
                const Field bx = moment(gx), br = moment(gr), brr = moment(product(gr, gr)), bry = moment(product(gr, py));
                auto& sp = sharp[static_cast<size_t>(n) * s.c + c];
                auto& bp = soft[static_cast<size_t>(n) * s.c + c];
                for (auto& f : sp)
                    f.assign(P, 0.0);
                for (auto& f : bp)
                    f.assign(P, 0.0);
                for (size_t i = 0; i < P; ++i) {
                    double cs0, cs1;
                    std::array<double, 4> p0, p1;
                    const double s0 = blur_branch_reference(mx[i], mr[i], mrr[i], my[i], myy[i], mry[i], sw, cs0, p0);
                    const double s1 = blur_branch_reference(bx[i], br[i], brr[i], my[i], myy[i], bry[i], sw, cs1, p1);
                    const double d = 50.0 * (s0 - s1), e = std::exp(-std::abs(d));
                    const double high = 1.0 / (1.0 + e), low = e * high;
                    const bool flat = myy[i] - my[i] * my[i] < 0.0005;
                    const double w0 = flat ? 1.0 : (d >= 0 ? high : low), w1 = flat ? 0.0 : (d >= 0 ? low : high);
                    if (std::abs(d) < 5.0 || std::abs(myy[i] - my[i] * my[i] - 0.0005) < 1e-4)
                        ref.decided[n * P + i] = 0;
                    ref.ssim_map[n * P + i] += s0 / s.c;
                    ref.cs_map[n * P + i] += (w0 * cs0 + w1 * cs1) / s.c;
                    blend_map[n * P + i] += (flat ? s0 : std::max(s0, s1) + (std::log1p(e) - std::log(2.0)) / 50.0) / s.c;
                    const double dw = (masked ? weight[i] : 1.0) * inv_window[i];
                    for (int k = 0; k < 4; ++k) {
                        const double scale = masked && k == 0 ? dw / 16.0 : dw;
                        sp[k][i] = to_half(p0[k] * (w0 * scale));
                        bp[k][i] = to_half(p1[k] * (w1 * scale) * (k == 3 ? sw : 1.0));
                    }
                }
            }
        if (masked)
            for (int n = 0; n < s.n; ++n)
                for (size_t i = 0; i < P; ++i)
                    if (weight[i] <= 0.0)
                        ref.ssim_map[n * P + i] = ref.cs_map[n * P + i] = blend_map[n * P + i] = 1.0;

        // Loss and chain.
        double chain = 0.0;
        if (masked) {
            double mask_sum = 0.0;
            for (const double m : weight)
                mask_sum += m;
            const double normalized = bc.denominator > 0.f ? bc.denominator : static_cast<float>(mask_sum) * s.n * s.c + 1e-8;
            chain = 1.0 / normalized;
        } else {
            const double count = static_cast<double>(s.n) * s.c * (s.h > 10 ? s.h - 10 : s.h) * (s.w > 10 ? s.w - 10 : s.w);
            chain = 1.0f / static_cast<float>(count);
        }
        for (int n = 0; n < s.n; ++n)
            for (int c = 0; c < s.c; ++c)
                for (int py = 0; py < s.h; ++py)
                    for (int px = 0; px < s.w; ++px) {
                        const size_t i = static_cast<size_t>(py) * s.w + px;
                        const size_t e = (static_cast<size_t>(n) * s.c + c) * P + i;
                        const double term = (1 - sw) * std::abs(x[e] - y[e]) + sw * (1 - blend_map[n * P + i]);
                        if (masked)
                            ref.loss += term * weight[i] * chain;
                        else if (inside_valid(py, px))
                            ref.loss += term * chain;
                    }

        // Branch pass (v, u), then both gradients.
        const auto adjoint = [&](const Field& partial) {
            Field valid_only(P, 0.0);
            for (int py = 0; py < s.h; ++py)
                for (int px = 0; px < s.w; ++px)
                    if (inside_valid(py, px))
                        valid_only[static_cast<size_t>(py) * s.w + px] = partial[static_cast<size_t>(py) * s.w + px];
            return filter(valid_only, s.h, s.w, Wn);
        };
        for (int n = 0; n < s.n; ++n)
            for (int c = 0; c < s.c; ++c) {
                const auto& sp = sharp[static_cast<size_t>(n) * s.c + c];
                const auto& bp = soft[static_cast<size_t>(n) * s.c + c];
                const Field pr = plane_of(raw, s, n, c), py = plane_of(y, s, n, c), pc = plane_of(x, s, n, c);
                const Field t_raw = filter(product(pr, keep), s.h, s.w, T);
                std::array<Field, 4> mb, ms;
                for (int k = 0; k < 4; ++k) {
                    mb[k] = adjoint(bp[k]);
                    ms[k] = adjoint(sp[k]);
                }
                Field v(P), u(P);
                for (size_t i = 0; i < P; ++i) {
                    const double nn = keep_sum[i] > 1e-30 ? 1.0 / keep_sum[i] : 0.0, kn = weight[i] * nn;
                    const double s0 = (masked ? -16.0 : -1.0) * mb[0][i];
                    v[i] = to_half(-kn * mb[3][i]);
                    u[i] = to_half(kn * (s0 - 2.0 * (t_raw[i] * nn) * mb[1][i] - py[i] * mb[2][i]));
                }
                const Field tv = filter(v, s.h, s.w, T), tu = filter(u, s.h, s.w, T);
                for (int yy = 0; yy < s.h; ++yy)
                    for (int xx = 0; xx < s.w; ++xx) {
                        const size_t i = static_cast<size_t>(yy) * s.w + xx;
                        const size_t e = (static_cast<size_t>(n) * s.c + c) * P + i;
                        const double center = weight[i];
                        const double l1 = (1 - sw) * sign(pc[i] - py[i]) * (masked ? center : (inside_valid(yy, xx) ? 1.0 : 0.0));
                        const double gc = chain * (center * -sw * ms[3][i] + l1 + keep[i] * tv[i]);
                        const double gr = chain * (center * ((masked ? -16.0 : -1.0) * ms[0][i] - 2.0 * pr[i] * ms[1][i] -
                                                             py[i] * ms[2][i]) +
                                                   keep[i] * tu[i]);
                        if (bc.decoupled) {
                            ref.grad[e] = gc;
                            ref.grad_raw[e] = gr;
                        } else {
                            ref.grad[e] = gc + gr;
                        }
                    }
            }
        return ref;
    }

    // The render, a raw render near it (as before exposure correction), and a photo: a noisy copy of the render
    // (the sharp branch leads), a blur of it with noise (the blurred branch leads) or a smooth photo whose windows
    // are all flat (plain SSIM).
    PhotoInputs blur_inputs(const BlurCase& bc, const uint32_t seed) {
        PhotoCase pc{bc.decoupled ? ops::PhotoPath::MaskedDecoupled : ops::PhotoPath::MaskedFused, bc.shape,
                     bc.target_bytes, bc.mask == 2};
        PhotoInputs in = photo_inputs(pc, seed);
        const Shape4& s = bc.shape;
        const auto raw_noise = uniform(s.numel(), seed + 5, -0.05f, 0.05f);
        std::vector<float> raw(s.numel());
        for (size_t i = 0; i < raw.size(); ++i)
            raw[i] = std::clamp(0.9f * static_cast<float>(in.hx[i]) + 0.05f + raw_noise[i], 0.f, 1.f);
        in.raw = upload(raw, s.tensor());
        in.hraw = widen(raw);
        if (bc.photo != 0) {
            const std::vector<double> taps = blur_taps_reference(1.0f);
            const auto noise = uniform(s.numel(), seed + 6, -0.05f, 0.05f);
            std::vector<float> y(s.numel());
            for (int n = 0; n < s.n; ++n)
                for (int c = 0; c < s.c; ++c) {
                    const Field blurred = filter(plane_of(in.hx, s, n, c), s.h, s.w, taps);
                    const Field mass = filter(Field(s.plane(), 1.0), s.h, s.w, taps);
                    for (int py = 0; py < s.h; ++py)
                        for (int px = 0; px < s.w; ++px) {
                            const size_t i = static_cast<size_t>(py) * s.w + px;
                            const size_t e = (static_cast<size_t>(n) * s.c + c) * s.plane() + i;
                            y[e] = bc.photo == 1 ? std::clamp(static_cast<float>(blurred[i] / mass[i]) + noise[e], 0.f, 1.f)
                                                 : 0.5f + 0.02f * std::sin(0.31f * px + 0.7f * c) * std::cos(0.43f * py);
                        }
                }
            if (bc.target_bytes) {
                std::vector<uint8_t> bytes(y.size());
                for (size_t i = 0; i < y.size(); ++i) {
                    bytes[i] = static_cast<uint8_t>(std::lround(std::clamp(y[i], 0.f, 1.f) * 255.f));
                    y[i] = static_cast<float>(bytes[i]) * (1.0f / 255.0f);
                }
                in.y = upload_bytes(bytes, s.tensor());
            } else {
                in.y = upload(y, s.tensor());
            }
            in.hy = widen(y);
        }
        if (bc.mask == 1) {
            // Soft weights in [0, 1.5]: zeros, values below one and above it (the tile scale).
            const auto soft = uniform(s.plane(), seed + 7);
            std::vector<float> mask(s.plane());
            for (size_t i = 0; i < mask.size(); ++i)
                mask[i] = soft[i] < 0.25f ? 0.f : 1.5f * soft[i];
            in.mask = upload(mask, TensorShape({static_cast<size_t>(s.h), static_cast<size_t>(s.w)}));
            in.hmask = widen(mask);
        }
        return in;
    }

    // A blur-tolerant gradient against the reference: every element within 3e-3 of the largest (moving where the
    // fp16 partials round moves elements by up to 2.8e-3 of the largest: the CUDA test's own measurement, between
    // its four- and three-pass versions), the L1 norm within 1e-4 (BlurTolerantSsim.MatchesReferenceValues'
    // budget) and the projection onto the reference within 1e-4 of it.
    void expect_blur_gradient(const std::vector<float>& got, const Field& want, const std::string& what) {
        expect_field(got, want, 0.0, 3e-3, what);
        double l1_got = 0.0, l1_want = 0.0, dot = 0.0, square = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            l1_got += std::abs(got[i]);
            l1_want += std::abs(want[i]);
            dot += got[i] * want[i];
            square += want[i] * want[i];
        }
        EXPECT_NEAR(l1_got, l1_want, 1e-4 * l1_want) << what << " L1 norm";
        EXPECT_NEAR(dot / square, 1.0, 1e-4) << what << " projection";
    }

    void check_blur(const lfs::training::TrainingOps& table, const BlurCase& bc, const std::string& name) {
        SCOPED_TRACE(name);
        const PhotoInputs in = blur_inputs(bc, 17);
        const bool masked = bc.mask != 0;
        const ops::PhotoPath path = masked ? (bc.decoupled ? ops::PhotoPath::MaskedDecoupled : ops::PhotoPath::MaskedFused)
                                           : (bc.decoupled ? ops::PhotoPath::Decoupled : ops::PhotoPath::Fused);
        ops::PhotoSaved saved{.backend = table.photometric->create()};
        Tensor loss, grad, grad_raw;
        table.photometric->evaluate(saved, in.x, bc.decoupled ? in.raw : Tensor{}, in.y, masked ? in.mask : Tensor{},
                                    {.path = path, .ssim_weight = bc.weight, .valid_padding = true, .denominator = bc.denominator, .blur_sigma = bc.sigma},
                                    loss, grad, grad_raw);
        const BlurRef ref = blur_reference(bc, in);
        expect_value(download(loss)[0], ref.loss, 1e-5, name + " loss");
        expect_blur_gradient(download(grad), ref.grad, name + " grad");
        if (bc.decoupled)
            expect_blur_gradient(download(grad_raw), ref.grad_raw, name + " grad_raw");
        else
            EXPECT_FALSE(grad_raw.is_valid());
        // Both maps within the plain maps' budget plus 1e-4 of each value: a nearly flat photo window or a blurred
        // window's small variance cancels its covariances against the window means, which keeps that much less of
        // the float precision. The blend's contrast-structure where the soft maximum is decided: near its balance
        // point the temperature (50) multiplies the float rounding of s0 - s1 in the branch weights by up to 12.5.
        const double map_tolerance = bc.mask == 1 ? 3e-6 : 2e-6;
        const auto expect_map = [&](const std::vector<float>& got, const Field& want, const bool decided_only,
                                    const std::string& what) {
            ASSERT_EQ(got.size(), want.size()) << what;
            size_t compared = 0, worst = 0;
            double worst_ratio = 0.0;
            for (size_t i = 0; i < got.size(); ++i) {
                if (decided_only && !ref.decided[i])
                    continue;
                ++compared;
                const double ratio = std::abs(got[i] - want[i]) / (map_tolerance + 1e-4 * std::abs(want[i]));
                if (!(ratio <= worst_ratio)) {
                    worst_ratio = ratio;
                    worst = i;
                }
            }
            EXPECT_LE(worst_ratio, 1.0) << what << ": index " << worst << " got " << got[worst] << " want " << want[worst]
                                        << " (" << compared << " of " << got.size() << " compared)";
        };
        expect_map(download(saved.ssim_map), ref.ssim_map, false, name + " ssim_map");
        expect_map(download(saved.cs_map), ref.cs_map, true, name + " cs_map");
    }

    // Catches a blur-tolerant SSIM that leaves CUDA's three kernels: the loss, both gradients and both maps against
    // the double reference for every blur radius, plain and decoupled, unweighted, soft-weighted and 8-bit masks,
    // float and 8-bit photos, batches, sizes that leave partial tiles or are smaller than a tile, a fixed
    // denominator, the sharp branch leading, the blurred one leading and flat photo windows.
    TEST_P(PortableLosses, BlurTolerantSsimMatchesReference) {
        const Shape4 fixture{1, 3, 32, 40}, odd{2, 3, 21, 27}, thin{1, 3, 9, 30}, tiny{1, 3, 7, 5}, gray{1, 1, 20, 24};
        check_blur(*table_, {fixture}, "plain");
        check_blur(*table_, {fixture, true}, "decoupled");
        check_blur(*table_, {fixture, false, 1}, "soft_weights");
        check_blur(*table_, {fixture, true, 1}, "decoupled_soft_weights");
        check_blur(*table_, {fixture, true, 2, true}, "decoupled_u8_mask_u8_photo");
        check_blur(*table_, {fixture, true, 1, false, 1.5f, 0.22f, 5000.f}, "decoupled_fixed_denominator");
        check_blur(*table_, {fixture, true, 0, false, 0.8215f}, "radius_3");
        check_blur(*table_, {fixture, false, 1, false, 0.6f}, "radius_2_soft_weights");
        check_blur(*table_, {fixture, true, 0, true, 0.3f}, "radius_1_u8_photo");
        check_blur(*table_, {fixture, false, 0, false, 1.5f, 0.22f, 0.f, 1}, "blurred_photo");
        check_blur(*table_, {fixture, true, 1, false, 1.5f, 0.22f, 0.f, 1}, "decoupled_blurred_photo_soft_weights");
        check_blur(*table_, {fixture, true, 0, false, 1.5f, 1.0f, 0.f, 2}, "flat_photo_ssim_only");
        check_blur(*table_, {fixture, false, 1, false, 1.5f, 0.22f, 0.f, 2}, "flat_photo_soft_weights");
        check_blur(*table_, {odd, true, 0, true}, "batch_decoupled_u8_photo");
        check_blur(*table_, {odd, false, 2}, "batch_u8_mask");
        check_blur(*table_, {thin, true}, "thin");
        check_blur(*table_, {tiny, true, 1}, "tiny_soft_weights");
        check_blur(*table_, {tiny}, "tiny");
        check_blur(*table_, {gray, true, 0, false, 1.0f}, "one_channel");
    }

    // Catches portable blur-tolerant SSIM values that leave CUDA's: the inputs and the values CUDA records in
    // BlurTolerantSsim.MatchesReferenceValues (tests/test_blur_tolerant_ssim.cpp, in the CUDA build's list), with its
    // budgets: the loss within 1e-6 and both gradients' L1 norms within 1e-4.
    TEST_P(PortableLosses, BlurTolerantSsimMatchesCudaValues) {
        constexpr size_t H = 301, W = 437;
        // That test's fixed_values and box_blurred.
        const auto fixed_values = [&](const size_t channels, const unsigned seed, const float lo, const float hi,
                                      const bool holes = false) {
            std::mt19937 rng(seed);
            std::vector<float> values(std::max<size_t>(channels, 1) * H * W);
            for (size_t i = 0; i < values.size(); ++i) {
                const size_t y = i / W % H, x = i % W;
                const bool hole = holes && ((y % 37 >= 10 && y % 37 < 17 && x % 41 >= 5 && x % 41 < 30) || y % 53 == 3);
                values[i] = hole ? 0.0f : lo + (hi - lo) * static_cast<float>(rng() >> 8) * (1.0f / 16777216.0f);
            }
            return values;
        };
        const auto box_blurred = [&](const std::vector<float>& in) {
            std::vector<float> out(in.size());
            for (size_t c = 0; c < 3; ++c)
                for (size_t y = 0; y < H; ++y)
                    for (size_t x = 0; x < W; ++x) {
                        float sum = 0.0f;
                        int count = 0;
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx) {
                                const auto yy = static_cast<long>(y) + dy, xx = static_cast<long>(x) + dx;
                                if (yy < 0 || xx < 0 || yy >= static_cast<long>(H) || xx >= static_cast<long>(W))
                                    continue;
                                sum += in[(c * H + yy) * W + xx];
                                ++count;
                            }
                        out[(c * H + y) * W + x] = sum / static_cast<float>(count);
                    }
            return out;
        };
        const TensorShape rgb({3, H, W});
        const auto image = fixed_values(3, 1, 0.02f, 0.98f), shade = fixed_values(3, 2, 0.0f, 0.3f);
        const auto noise = fixed_values(3, 5, 0.0f, 1.0f);
        std::vector<float> mixed(image.size()), raw(image.size());
        for (size_t i = 0; i < image.size(); ++i) {
            mixed[i] = image[i] * 0.7f + shade[i];
            raw[i] = noise[i] * 0.2f + image[i] * 0.8f;
        }
        const auto target = box_blurred(mixed);
        std::vector<uint8_t> target_bytes(target.size());
        for (size_t i = 0; i < target.size(); ++i)
            target_bytes[i] = static_cast<uint8_t>(std::round(std::clamp(target[i], 0.0f, 1.0f) * 255.0f));
        const Tensor x = upload(image, rgb), r = upload(raw, rgb), y = upload(target, rgb), y8 = upload_bytes(target_bytes, rgb);
        const Tensor weight = upload(fixed_values(0, 9, 0.5f, 1.5f, true), TensorShape({H, W}));
        struct Case {
            float sigma;
            bool decoupled;
            int weighting; // 0 none, 1 weights, 2 weights with the 8-bit photo and denominator 50,000
            float loss;
            double corrected_l1, raw_l1;
        };
        const Case cases[] = {
            {0.6f, false, 0, 0.304602087f, 1.040200202, 0},
            {0.6f, false, 1, 0.3034347f, 1.03684406, 0},
            {0.6f, false, 2, 2.07304907f, 7.083815804, 0},
            {0.6f, true, 0, 0.296424866f, 0.7809249387, 0.6007441256},
            {0.6f, true, 1, 0.295226306f, 0.7811272925, 0.589753502},
            {0.6f, true, 2, 2.01698184f, 5.336627963, 4.029545959},
            {1.5f, false, 0, 0.247425988f, 0.819398238, 0},
            {1.5f, false, 1, 0.24595879f, 0.8363325159, 0},
            {1.5f, false, 2, 1.6805985f, 5.712948791, 0},
            {1.5f, true, 0, 0.261338472f, 0.7804109969, 0.7261689108},
            {1.5f, true, 1, 0.25874871f, 0.7804907897, 0.6983592078},
            {1.5f, true, 2, 1.76796758f, 5.332270833, 4.77068885},
        };
        const auto l1_norm = [](const Tensor& t) {
            double sum = 0.0;
            for (const float v : download(t))
                sum += std::abs(v);
            return sum;
        };
        ops::PhotoSaved saved{.backend = table_->photometric->create()};
        for (const Case& c : cases) {
            SCOPED_TRACE(testing::Message() << "sigma " << c.sigma << ", decoupled " << c.decoupled << ", weighting "
                                            << c.weighting);
            const bool masked = c.weighting != 0;
            const ops::PhotoPath path = masked ? (c.decoupled ? ops::PhotoPath::MaskedDecoupled : ops::PhotoPath::MaskedFused)
                                               : (c.decoupled ? ops::PhotoPath::Decoupled : ops::PhotoPath::Fused);
            Tensor loss, grad, grad_raw;
            table_->photometric->evaluate(saved, x, c.decoupled ? r : Tensor{}, c.weighting == 2 ? y8 : y,
                                          masked ? weight : Tensor{},
                                          {.path = path, .ssim_weight = 0.22f, .valid_padding = true, .denominator = c.weighting == 2 ? 50000.0f : 0.0f, .blur_sigma = c.sigma},
                                          loss, grad, grad_raw);
            EXPECT_NEAR(download(loss)[0], c.loss, 1e-6f);
            EXPECT_NEAR(l1_norm(grad), c.corrected_l1, 1e-4 * c.corrected_l1);
            EXPECT_EQ(grad_raw.is_valid(), c.decoupled);
            if (c.decoupled)
                EXPECT_NEAR(l1_norm(grad_raw), c.raw_l1, 1e-4 * c.raw_l1);
        }
    }

    // Catches a blur-tolerant SSIM that allocates per step or leaks into the plain paths' arena: Metal keeps it in
    // its own arena, sized by its layout, reused at the same shape and kept when a plain evaluate runs in between
    // (photo sets that straddle the 1,200-pixel rule); shrink and reset cover it.
    TEST_P(PortableLosses, BlurTolerantSsimWorkspaceFollowsItsArena) {
        if (GetParam() != GpuBackend::Metal)
            GTEST_SKIP() << "Metal's workspace layout";
        const auto align = [](const size_t bytes) { return (bytes + 255) & ~size_t{255}; };
        const auto layout = [&](const std::vector<size_t>& fields) {
            size_t total = 0;
            for (const size_t bytes : fields)
                total = align(total) + bytes;
            return align(total);
        };
        // SSIM map, sharp partials (half4), the blurred partials' region that the two float gradients reuse,
        // v u (half2), partial sums, loss and mask sum.
        const auto blur_bytes = [&](const Shape4& s) {
            return layout({s.n * s.plane() * 4, s.numel() * 8, align(s.numel() * 4) + s.numel() * 4, s.numel() * 4, 8192, 4, 4});
        };
        const auto masked_decoupled_bytes = [&](const Shape4& s) {
            return layout({s.n * s.plane() * 4, s.numel() * 2, s.numel() * 2, s.numel() * 2, s.numel() * 2,
                           s.numel() * 4, 8192, 4, 4});
        };
        ops::PhotoSaved saved{.backend = table_->photometric->create()};
        const auto run = [&](const Shape4& s, const float sigma) {
            const PhotoInputs in = photo_inputs({ops::PhotoPath::MaskedDecoupled, s}, 3);
            Tensor loss, grad, grad_raw;
            table_->photometric->evaluate(saved, in.x, in.raw, in.y, in.mask,
                                          {.path = ops::PhotoPath::MaskedDecoupled, .ssim_weight = 0.2f, .valid_padding = true, .blur_sigma = sigma},
                                          loss, grad, grad_raw);
            download(loss);
        };
        const Shape4 big{1, 3, 33, 41}, small{1, 3, 16, 20};
        run(big, 1.5f);
        auto bytes = table_->photometric->workspace_bytes(saved);
        EXPECT_EQ(bytes.required, blur_bytes(big));
        EXPECT_EQ(bytes.allocated, blur_bytes(big));
        run(big, 0.8f);
        EXPECT_EQ(table_->photometric->workspace_bytes(saved).allocated, blur_bytes(big)) << "same shape";
        run(big, 0.f);
        bytes = table_->photometric->workspace_bytes(saved);
        EXPECT_EQ(bytes.required, blur_bytes(big) + masked_decoupled_bytes(big));
        EXPECT_EQ(bytes.allocated, blur_bytes(big) + masked_decoupled_bytes(big)) << "both arenas kept";
        run(small, 1.5f);
        bytes = table_->photometric->workspace_bytes(saved);
        EXPECT_EQ(bytes.required, blur_bytes(small) + masked_decoupled_bytes(big));
        EXPECT_EQ(bytes.allocated, blur_bytes(big) + masked_decoupled_bytes(big)) << "a smaller shape keeps the high-water";
        table_->photometric->shrink_to_required(saved);
        EXPECT_EQ(table_->photometric->workspace_bytes(saved).allocated, blur_bytes(small) + masked_decoupled_bytes(big));
        table_->photometric->reset(saved);
        bytes = table_->photometric->workspace_bytes(saved);
        EXPECT_EQ(bytes.required, 0u);
        EXPECT_EQ(bytes.allocated, 0u);
    }

    // The thin-structure denominator is read back once per camera and kept while what its weight was made from
    // stays the same; new inputs, another size or a cleared cache read the weight again.
    TEST_P(PortableLosses, StructureDenominatorCacheReadsEachCameraOnce) {
        const Tensor half = Tensor::full({8, 8}, 0.5f, Device::GPU);
        const Tensor ones = Tensor::ones({8, 8}, Device::GPU);
        const float half_sum = k::structure_base_denominator(half, 8, 8, false);
        const float ones_sum = k::structure_base_denominator(ones, 8, 8, false);
        ASSERT_NE(half_sum, ones_sum);
        k::StructureDenominatorCache<int> cache;
        EXPECT_EQ(cache.get(3, 7, half, 8, 8, false), half_sum);
        EXPECT_EQ(cache.get(3, 7, ones, 8, 8, false), half_sum) << "same camera and inputs: not read again";
        EXPECT_EQ(cache.get(4, 7, ones, 8, 8, false), ones_sum) << "another camera";
        EXPECT_EQ(cache.get(3, 8, ones, 8, 8, false), ones_sum) << "new inputs";
        EXPECT_EQ(cache.get(3, 8, half, 8, 8, true), half_sum) << "another padding";
        EXPECT_EQ(cache.get(3, 8, ones, 8, 8, true), half_sum);
        cache.clear();
        EXPECT_EQ(cache.get(3, 8, ones, 8, 8, true), ones_sum) << "cleared";
    }

    // Catches a calibration fit handed another denominator than the masked loss divides by: the caller's
    // denominator when it passes one, else the mask sum the reduction formed; the loss times it is the same sum
    // either way.
    TEST_P(PortableLosses, CalibrationFitGetsTheLossDenominator) {
        const Shape4 s{1, 3, 32, 40};
        for (const ops::PhotoPath path : {ops::PhotoPath::MaskedFused, ops::PhotoPath::MaskedDecoupled}) {
            const bool decoupled = path == ops::PhotoPath::MaskedDecoupled;
            const PhotoInputs in = photo_inputs({path, s}, 7);
            double mask_sum = 0.0;
            for (const double m : in.hmask)
                mask_sum += m;
            const double computed = static_cast<float>(mask_sum) * s.n * s.c + 1e-8;
            double weighted_sum[2] = {};
            for (const int fixed : {0, 1}) {
                SCOPED_TRACE(std::format("decoupled {} fixed denominator {}", decoupled, fixed));
                const float denominator = fixed ? 4321.5f : 0.0f;
                ops::PhotoSaved saved{.backend = table_->photometric->create()};
                Tensor loss, grad, grad_raw;
                table_->photometric->evaluate(saved, in.x, decoupled ? in.raw : Tensor{}, in.y, in.mask,
                                              {.path = path, .ssim_weight = 0.2f, .valid_padding = true, .denominator = denominator, .calibration_fit = true},
                                              loss, grad, grad_raw);
                if (fixed)
                    EXPECT_EQ(saved.mask_denominator, denominator);
                else
                    EXPECT_NEAR(saved.mask_denominator, computed, 1e-6 * computed);
                weighted_sum[fixed] = static_cast<double>(download(loss)[0]) * saved.mask_denominator;
            }
            EXPECT_NEAR(weighted_sum[1], weighted_sum[0], 1e-5 * std::abs(weighted_sum[0]));
        }
    }

    // Catches a masked loss that waits for the GPU to hand the calibration fit a denominator its caller passed: with
    // 40 chained 2048x2048 products queued ahead of it, evaluate returns while they still run.
    TEST_P(PortableLosses, CalibrationFitDoesNotWaitForTheGpu) {
        if (GetParam() != GpuBackend::Metal)
            GTEST_SKIP() << "Metal's queue";
        const Shape4 s{1, 3, 32, 40};
        const PhotoInputs in = photo_inputs({ops::PhotoPath::MaskedDecoupled, s}, 7);
        const ops::PhotoParams params{.path = ops::PhotoPath::MaskedDecoupled, .ssim_weight = 0.2f, .valid_padding = true, .denominator = 4321.5f, .calibration_fit = true};
        ops::PhotoSaved saved{.backend = table_->photometric->create()};
        Tensor loss, grad, grad_raw;
        // Pipelines and workspace first, so the timed call only encodes.
        table_->photometric->evaluate(saved, in.x, in.raw, in.y, in.mask, params, loss, grad, grad_raw);
        download(loss);
        constexpr size_t n = 2048;
        Tensor product = Tensor::ones({n, n}, Device::GPU);
        const Tensor identity = Tensor::eye(n, Device::GPU);
        for (int i = 0; i < 40; ++i)
            product = product.mm(identity);
        lfs::core::TensorFence queued(GpuBackend::Metal);
        queued.record(lfs::core::TensorExecutionTarget::current());
        if (queued.ready())
            GTEST_SKIP() << "the GPU finished the queued work before the evaluate";
        table_->photometric->evaluate(saved, in.x, in.raw, in.y, in.mask, params, loss, grad, grad_raw);
        EXPECT_FALSE(queued.ready()) << "the evaluate waited for the GPU";
        EXPECT_EQ(saved.mask_denominator, params.denominator);
        download(loss);
    }

    INSTANTIATE_TEST_SUITE_P(Backends, PortableLosses, testing::Values(GpuBackend::Metal, GpuBackend::Vulkan),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });

} // namespace
