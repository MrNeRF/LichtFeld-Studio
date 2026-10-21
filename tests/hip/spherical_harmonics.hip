/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "SphericalHarmonics.h"
#include "sh_oracle.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
    using hip_sh_oracle::basis;
    void check(hipError_t value) {
        if (value != hipSuccess)
            throw std::runtime_error(hipGetErrorString(value));
    }
    void require(bool value, const char* message) {
        if (!value)
            throw std::runtime_error(message);
    }
    struct Buffer {
        void* data = nullptr;
        explicit Buffer(size_t bytes) { check(hipMalloc(&data, std::max<size_t>(bytes, 4))); }
        ~Buffer() { (void)hipFree(data); }
        template <class T = float>
        T* as() { return static_cast<T*>(data); }
    };
    size_t swizzled(size_t gaussian, size_t attribute, size_t slots) {
        return ((gaussian / 32 * slots + attribute / 4) * 32 + gaussian % 32) * 4 + attribute % 4;
    }
} // namespace

int main() try {
    unsigned cases = 0;
    double forward_error = 0, coefficient_error = 0, direction_error = 0;
    for (size_t count : {size_t{1}, size_t{31}, size_t{32}, size_t{33}, size_t{263}})
        for (unsigned layout = 0; layout <= 3; ++layout) {
            const size_t bases = (layout + 1) * (layout + 1), rest = bases - 1, slots = (rest * 3 + 3) / 4;
            std::vector<float> coefficients(count * bases * 3), sh0(count * 3), dirs(count * 3), incoming(count * 3);
            std::vector<float> storage((count + 31) / 32 * slots * 32 * 4, -12345);
            std::vector<unsigned char> masks(count, 1);
            for (size_t g = 0; g < count; ++g) {
                masks[g] = g % 7 != 6;
                dirs[g * 3] = 0.3f + 0.01f * (g % 13);
                dirs[g * 3 + 1] = -0.7f + 0.025f * (g % 11);
                dirs[g * 3 + 2] = 1.1f + 0.02f * (g % 17);
                for (size_t k = 0; k < bases; ++k)
                    for (size_t c = 0; c < 3; ++c) {
                        const float value = 0.15f * std::sin(float(g * 17 + k * 7 + c * 3));
                        coefficients[(g * bases + k) * 3 + c] = value;
                        if (k == 0)
                            sh0[g * 3 + c] = value;
                        else
                            storage[swizzled(g, (k - 1) * 3 + c, slots)] = value;
                    }
                for (size_t c = 0; c < 3; ++c)
                    incoming[g * 3 + c] = 0.2f * std::cos(float(g * 11 + c * 7));
            }
            Buffer d_dirs(dirs.size() * 4), d_sh0(sh0.size() * 4), d_rest(storage.size() * 4), d_mask(count),
                d_in(incoming.size() * 4), d_colors((count * 3 + 16) * 4),
                d_coefficients((coefficients.size() + 16) * 4), d_direction((dirs.size() + 16) * 4);
            check(hipMemcpy(d_dirs.data, dirs.data(), dirs.size() * 4, hipMemcpyHostToDevice));
            check(hipMemcpy(d_sh0.data, sh0.data(), sh0.size() * 4, hipMemcpyHostToDevice));
            if (!storage.empty())
                check(hipMemcpy(d_rest.data, storage.data(), storage.size() * 4, hipMemcpyHostToDevice));
            check(hipMemcpy(d_mask.data, masks.data(), count, hipMemcpyHostToDevice));
            check(hipMemcpy(d_in.data, incoming.data(), incoming.size() * 4, hipMemcpyHostToDevice));
            for (unsigned mode = 0; mode < 2 * (layout + 1); ++mode) {
                const unsigned degree = mode % (layout + 1);
                const bool optional = mode >= layout + 1;
                std::vector<float> colors(count * 3 + 16, 12345), gradient(coefficients.size() + 16, 12345), v_dirs(dirs.size() + 16, 12345);
                std::fill(gradient.begin(), gradient.begin() + coefficients.size(), 0);
                std::fill(v_dirs.begin(), v_dirs.begin() + dirs.size(), 0);
                check(hipMemcpy(d_colors.data, colors.data(), colors.size() * 4, hipMemcpyHostToDevice));
                check(hipMemcpy(d_coefficients.data, gradient.data(), gradient.size() * 4, hipMemcpyHostToDevice));
                check(hipMemcpy(d_direction.data, v_dirs.data(), v_dirs.size() * 4, hipMemcpyHostToDevice));
                gsplat_lfs::launch_spherical_harmonics_swizzled_fwd_kernel(degree, layout, 3, optional ? nullptr : d_dirs.as(), d_sh0.as(),
                                                                           optional || !rest ? nullptr : d_rest.as(), optional ? nullptr : d_mask.as<bool>(), count, d_colors.as(), nullptr);
                gsplat_lfs::launch_spherical_harmonics_swizzled_bwd_kernel(degree, layout, 3, optional ? nullptr : d_dirs.as(), d_sh0.as(),
                                                                           optional || !rest ? nullptr : d_rest.as(), optional ? nullptr : d_mask.as<bool>(), d_in.as(), count, bases, !optional, d_coefficients.as(), d_direction.as(), nullptr);
                check(hipDeviceSynchronize());
                check(hipMemcpy(colors.data(), d_colors.data, colors.size() * 4, hipMemcpyDeviceToHost));
                check(hipMemcpy(gradient.data(), d_coefficients.data, gradient.size() * 4, hipMemcpyDeviceToHost));
                check(hipMemcpy(v_dirs.data(), d_direction.data, v_dirs.size() * 4, hipMemcpyDeviceToHost));
                const size_t active = optional ? 1 : (degree + 1) * (degree + 1);
                for (size_t g = 0; g < count; ++g) {
                    const auto b = basis(dirs[g * 3], dirs[g * 3 + 1], dirs[g * 3 + 2]);
                    for (size_t c = 0; c < 3; ++c) {
                        const bool enabled = optional || masks[g];
                        double expected = enabled ? 0.5 : 12345;
                        for (size_t k = 0; k < bases; ++k) {
                            if (enabled && k < active)
                                expected += b[k] * coefficients[(g * bases + k) * 3 + c];
                            const double v = enabled && k < active ? b[k] * incoming[g * 3 + c] : 0;
                            const double actual = gradient[(g * bases + k) * 3 + c];
                            require(std::isfinite(actual), "Nonfinite SH coefficient gradient");
                            coefficient_error = std::max(coefficient_error, std::abs(actual - v));
                            require(std::abs(actual - v) < 2e-6, "SH coefficient gradient CPU oracle mismatch");
                        }
                        require(std::isfinite(colors[g * 3 + c]), "Nonfinite SH color");
                        forward_error = std::max(forward_error, std::abs(colors[g * 3 + c] - expected));
                        require(std::abs(colors[g * 3 + c] - expected) < 2e-6, "SH color CPU oracle mismatch");
                    }
                    for (size_t axis = 0; axis < 3; ++axis) {
                        std::array<double, 3> direction{dirs[g * 3], dirs[g * 3 + 1], dirs[g * 3 + 2]};
                        direction[axis] += 1e-5;
                        const auto plus = basis(direction[0], direction[1], direction[2]);
                        direction[axis] -= 2e-5;
                        const auto minus = basis(direction[0], direction[1], direction[2]);
                        double expected = 0;
                        if (!optional && masks[g])
                            for (size_t k = 0; k < active; ++k)
                                for (size_t c = 0; c < 3; ++c)
                                    expected += (plus[k] - minus[k]) / 2e-5 * coefficients[(g * bases + k) * 3 + c] * incoming[g * 3 + c];
                        const double actual = v_dirs[g * 3 + axis];
                        require(std::isfinite(actual), "Nonfinite SH direction gradient");
                        direction_error = std::max(direction_error, std::abs(actual - expected));
                        require(std::abs(actual - expected) < 3e-6, "SH direction finite-difference mismatch");
                    }
                }
                for (size_t i = count * 3; i < colors.size(); ++i)
                    require(colors[i] == 12345 && v_dirs[i] == 12345, "SH color/direction tail overwrite");
                for (size_t i = coefficients.size(); i < gradient.size(); ++i)
                    require(gradient[i] == 12345, "SH coefficient tail overwrite");
                ++cases;
            }
        }
    std::printf("PASS %u SH degree/layout/tail cases; max color %.9g, coefficient gradient %.9g, direction %.9g\n",
                cases, forward_error, coefficient_error, direction_error);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
