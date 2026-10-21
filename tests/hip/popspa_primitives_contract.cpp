/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/popspa_hip.hpp"
#include <core/tensor_backend.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace lfs;
using namespace lfs::core;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, double absolute = 2e-6, double relative = 2e-5) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > absolute + relative * std::abs(expected)) {
        std::fprintf(stderr, "actual %.12g expected %.12g\n", actual, expected);
        throw std::runtime_error("HIP POPSpa differs from independent reference");
    }
}
Tensor floats(const std::vector<float>& values) {
    return Tensor::from_vector(values, {values.size()}, Device::GPU);
}
Tensor mask(const std::vector<float>& values) { return floats(values).to(DataType::Bool); }
std::vector<int64_t> selected(const std::vector<double>& values, const std::vector<float>& active,
                              const std::vector<float>& frozen, size_t k) {
    std::vector<int64_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), int64_t{0});
    std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
        if (active[a] != active[b]) return active[a] > active[b];
        if (frozen[a] != frozen[b]) return frozen[a] > frozen[b];
        if (values[a] != values[b]) return values[a] > values[b];
        return a < b;
    });
    indices.resize(k);
    std::sort(indices.begin(), indices.end());
    return indices;
}

void selection_and_projection(const gpu_ops::PopspaOps& ops, size_t n) {
    std::vector<double> scores(n);
    std::vector<float> active(n), frozen(n), raw(n, 0.f), dual(n);
    for (size_t i = 0; i < n; ++i) {
        scores[i] = i % 11 == 0 ? -0.0 : double(i % 13) / 16.;
        active[i] = i % 7 != 0;
        frozen[i] = active[i] && i % 17 == 0;
        dual[i] = float(int(i % 32) - 16) / 16.f;
    }
    auto score_host = Tensor::zeros({n, size_t{8}}, Device::CPU, DataType::UInt8);
    for (size_t i = 0; i < n; ++i)
        gpu_ops::encode_pop_score(scores[i], score_host.ptr<uint8_t>() + i * 8, ops.score_encoding);
    auto score_gpu = score_host.to(Device::GPU), a = mask(active), f = mask(frozen), r = floats(raw);
    for (size_t k : {size_t{0}, n / 2, n}) {
        Tensor keep;
        ops.keep_indices(score_gpu, a, f, k, keep);
        require(keep.numel() == k, "POPSpa selection count changed");
        if (k) {
            require(gpu_backend_of(keep) == GpuBackend::HIP, "POPSpa selection left HIP");
            require(keep.cpu().to_vector_int64() == selected(scores, active, frozen, k),
                    "POPSpa selection ordering, masks or numeric ties differ");
        }
        for (bool update : {false, true}) {
            auto z = Tensor::full({n}, -.7f, Device::GPU), u = floats(dual);
            ops.project(r, z, u, a, f, k, update);
            if (!n) continue;
            std::vector<double> shifted(n);
            for (size_t i = 0; i < n; ++i) shifted[i] = double(.5f + dual[i]);
            const auto ids = selected(shifted, active, frozen, k);
            const auto got_z = z.cpu().to_vector(), got_u = u.cpu().to_vector();
            for (size_t i = 0; i < n; ++i) {
                const float expected_z = std::binary_search(ids.begin(), ids.end(), int64_t(i)) ? float(shifted[i]) : 0.f;
                const float expected_u = update && active[i] && !frozen[i] ? dual[i] + .5f - expected_z : dual[i];
                near(got_z[i], expected_z, 0, 0); near(got_u[i], expected_u, 0, 0);
            }
        }
    }
    std::printf("PASS POPSpa masks/ties/selection/project n=%zu\n", n);
}

void penalty(const gpu_ops::PopspaOps& ops) {
    constexpr size_t n = 1025;
    std::vector<float> raw(n), z(n), u(n), active(n), frozen(n);
    for (size_t i = 0; i < n; ++i) {
        raw[i] = float(int(i % 37) - 18) / 7.f;
        z[i] = float(i % 11) / 13.f; u[i] = float(int(i % 9) - 4) / 17.f;
        active[i] = i % 7 != 0; frozen[i] = active[i] && i % 13 == 0;
    }
    auto r = floats(raw), proxy = floats(z), dual = floats(u), a = mask(active), f = mask(frozen);
    auto gradient = Tensor::full({n}, .125f, Device::GPU), loss = Tensor::full({1}, .25f, Device::GPU);
    constexpr float rho = .037f;
    ops.penalty(r, proxy, dual, a, f, gradient, loss, rho);
    const auto result = gradient.cpu().to_vector();
    double expected_loss = .25;
    for (size_t i = 0; i < n; ++i) {
        double expected = .125;
        if (active[i] && !frozen[i]) {
            const double sigmoid = 1. / (1. + std::exp(-double(raw[i]))), delta = sigmoid - z[i] + u[i];
            expected += rho * delta * sigmoid * (1. - sigmoid);
            expected_loss += .5 * rho * delta * delta;
        }
        near(result[i], expected);
    }
    near(loss.cpu().to_vector()[0], expected_loss, 2e-6, 2e-6);
    require(r.cpu().to_vector() == raw && proxy.cpu().to_vector() == z && dual.cpu().to_vector() == u,
            "POPSpa penalty mutated input parameters or proxy state");
    const auto before = gradient.cpu().to_vector();
    bool rejected = false;
    try { ops.penalty(r, proxy, dual, a, f, gradient, loss, -1.f); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && gradient.cpu().to_vector() == before, "Invalid POPSpa penalty changed gradients");
    std::puts("PASS POPSpa penalty accumulation/frozen masks/validation");
}

double rank_loss(const std::vector<double>& raw, size_t row, double rw, double tw, double epsilon) {
    double eigen[3], sum = 0, minimum = std::exp(raw[row * 3]);
    for (size_t d = 0; d < 3; ++d) {
        const double scale = std::exp(raw[row * 3 + d]);
        eigen[d] = scale * scale; sum += eigen[d]; minimum = std::min(minimum, scale);
    }
    double entropy = 0;
    for (double value : eigen) {
        const double p = value / std::max(sum, epsilon);
        entropy -= p * std::log(std::max(p, epsilon));
    }
    return rw * std::max(-std::log(std::exp(entropy) - 1. + epsilon), 0.) + tw * minimum;
}

void effective_rank(const gpu_ops::PopspaOps& ops) {
    constexpr size_t n = 257;
    constexpr float rw = .3f, tw = .1f, epsilon = 1e-8f;
    std::vector<float> raw(n * 3), active(n), frozen(n);
    size_t active_count = 0;
    for (size_t i = 0; i < n; ++i) {
        raw[3 * i] = -7.f + float(i % 13) * .03f;
        raw[3 * i + 1] = -1.f + float(i % 7) * .01f;
        raw[3 * i + 2] = .5f + float(i % 11) * .02f;
        active[i] = i % 7 != 0; frozen[i] = active[i] && i % 17 == 0;
        active_count += bool(active[i]);
    }
    auto scales = floats(raw), a = mask(active), f = mask(frozen);
    auto gradient = Tensor::full({n * 3}, .125f, Device::GPU), loss = Tensor::full({1}, .25f, Device::GPU);
    ops.effective_rank(scales, gradient, a, f, loss, active_count, rw, tw, epsilon);
    const auto actual = gradient.cpu().to_vector();
    std::vector<double> values(raw.begin(), raw.end());
    double expected_loss = .25;
    for (size_t i = 0; i < n; ++i) {
        if (active[i]) expected_loss += rank_loss(values, i, rw, tw, epsilon) / active_count;
        for (size_t d = 0; d < 3; ++d) {
            double expected = .125;
            if (active[i] && !frozen[i]) {
                const size_t index = 3 * i + d;
                constexpr double step = 1e-3;
                values[index] += step; const double upper = rank_loss(values, i, rw, tw, epsilon);
                values[index] -= 2 * step; const double lower = rank_loss(values, i, rw, tw, epsilon);
                values[index] = raw[index];
                expected += (upper - lower) / (2 * step * active_count);
            }
            near(actual[3 * i + d], expected, 2e-7, 2e-6);
        }
    }
    near(loss.cpu().to_vector()[0], expected_loss, 2e-6, 2e-6);
    Tensor none;
    auto value_only = Tensor::full({1}, .25f, Device::GPU);
    ops.effective_rank(scales, none, a, f, value_only, active_count, rw, tw, epsilon);
    near(value_only.cpu().to_vector()[0], expected_loss, 2e-6, 2e-6);
    auto zero_loss = Tensor::full({1}, .25f, Device::GPU), zero_gradient = gradient.clone();
    ops.effective_rank(scales, zero_gradient, a, f, zero_loss, 0, rw, tw, epsilon);
    require(zero_loss.cpu().to_vector()[0] == .25f && zero_gradient.cpu().to_vector() == actual,
            "Zero-active POPSpa rank regularization changed outputs");
    std::puts("PASS POPSpa effective-rank independent loss/finite differences/frozen/value-only/zero-active");
}

int main(int argc, char** argv) try {
    require(argc == 2 && std::filesystem::path(argv[1]).is_absolute(), "Pass the absolute native HIP module");
#ifdef _WIN32
    require(_putenv_s("LFS_HIP_TRAINING_MODULE", argv[1]) == 0, "Cannot select HIP module");
#else
    require(setenv("LFS_HIP_TRAINING_MODULE", argv[1], 1) == 0, "Cannot select HIP module");
#endif
    require(bool(set_default_gpu_backend(GpuBackend::HIP)), "Cannot select explicit HIP backend");
    {
        GpuBackendScope scope(GpuBackend::HIP);
        const auto& ops = training::hip_popspa_ops();
        for (size_t n : {size_t{0}, size_t{1}, size_t{33}, size_t{257}, size_t{1025}})
            selection_and_projection(ops, n);
        penalty(ops); effective_rank(ops);
    }
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)), "POPSpa retained GPU storage");
    std::puts("PASS native HIP POPSpa primitives; full family admission is still gated on raster score callbacks");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "HIP POPSPA FAILED: %s\n", error.what());
    return 1;
}
