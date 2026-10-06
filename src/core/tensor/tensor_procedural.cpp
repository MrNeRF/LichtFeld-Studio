/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_procedural.hpp"
#include "core/tensor_fused.hpp"
#include "internal/tensor_impl.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <format>
#include <mutex>
#include <numbers>
#include <tbb/parallel_for.h>

// Match the expression backends' explicit Float32 operations, not host FMA contraction.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace lfs::core {
    namespace {
        using fused::Expr;
        using fused::where;
        float where(bool condition, float yes, float no) { return condition ? yes : no; }
        using fused::floor;
        float floor(float x) { return std::floor(x); }
        using fused::clamp;
        float clamp(float x, float lo, float hi) { return std::clamp(x, lo, hi); }
        using fused::isfinite;
        bool isfinite(float x) { return std::isfinite(x); }
        Expr uint_bits(const Expr& x) { return x.cast(DataType::UInt32); }
        uint32_t uint_bits(float x) { return static_cast<uint32_t>(x); }

        void check_field(const Tensor& value, const Tensor& reference, size_t columns = 0) {
            const size_t rank = columns ? 2 : 1;
            LFS_ASSERT_MSG(value.is_valid() && value.ndim() == rank && value.dtype() == DataType::Float32 &&
                               value.size(0) == reference.size(0) && (!columns || value.size(1) == columns) &&
                               value.device() == reference.device(),
                           std::format("procedural field requires Float32 [N{}] on the same device (rank={}, count={}, expected={}, dtype={}, device={}, expected_device={})",
                                       columns ? ",3" : "", value.ndim(), value.size(0), reference.size(0),
                                       int(value.dtype()), int(value.device()), int(reference.device())));
            internal::require_same_gpu_backend(value, reference, "procedural field");
        }

        // Fixed gradients avoid backend-dependent normalization/transcendentals.
        // Periodic reduction is exact (a power of two) and keeps integer casts in range,
        // including huge finite coordinates. Only the lattice hash is periodic.
        template <class F>
        F noise(F px, F py, F pz) {
            px = where(isfinite(px), px, 0.0f);
            py = where(isfinite(py), py, 0.0f);
            pz = where(isfinite(pz), pz, 0.0f);
            const auto cx = floor(px), cy = floor(py), cz = floor(pz);
            const auto x = px - cx, y = py - cy, z = pz - cz;
            const auto ix = uint_bits(cx - floor(cx * 0x1p-16f) * 65536.0f);
            const auto iy = uint_bits(cy - floor(cy * 0x1p-16f) * 65536.0f);
            const auto iz = uint_bits(cz - floor(cz * 0x1p-16f) * 65536.0f);
            const auto fade = [](F t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); };
            const auto u = fade(x), v = fade(y), w = fade(z);
            const auto corner = [&](uint32_t dx, uint32_t dy, uint32_t dz) {
                auto hash = ((ix + dx) & 65535u) * 0x8da6b343u ^ ((iy + dy) & 65535u) * 0xd8163841u ^ ((iz + dz) & 65535u) * 0xcb1ab31fu;
                hash = (hash ^ (hash >> 16u)) * 0x7feb352du;
                hash = (hash ^ (hash >> 15u)) * 0x846ca68bu;
                hash = (hash ^ (hash >> 16u)) & 15u;
                const auto a = x - float(dx), b = y - float(dy), c = z - float(dz);
                const auto first = where(hash < 8u, a, b);
                const auto second = where(hash < 4u, b, where((hash == 12u) || (hash == 14u), a, c));
                return where((hash & 1u) == 0u, first, -first) + where((hash & 2u) == 0u, second, -second);
            };
            const auto lerp = [](F a, F b, F t) { return a + (b - a) * t; };
            const auto low = lerp(lerp(corner(0, 0, 0), corner(1, 0, 0), u), lerp(corner(0, 1, 0), corner(1, 1, 0), u), v);
            const auto high = lerp(lerp(corner(0, 0, 1), corner(1, 0, 1), u), lerp(corner(0, 1, 1), corner(1, 1, 1), u), v);
            return lerp(low, high, w) * 0.5f + 0.5f;
        }

        const fused::Kernel& noise_kernel() {
            static const auto kernel = [] {
                fused::Builder b(2); // [element, octave], folded to [element]
                const auto p = b.input(DataType::Float32, 2);
                const auto scale = b.input(DataType::Float32, 1).load({0});
                const auto detail = clamp(b.input(DataType::Float32, 1).load({0}), 0.0f, 15.0f);
                const auto roughness = clamp(b.input(DataType::Float32, 1).load({0}), 0.0f, 1.0f);
                const auto distortion = b.input(DataType::Float32, 1).load({0});
                const auto seed = b.input(DataType::Float32, 1).load({0});
                const auto coordinate = [&](int axis) { return p.gather({b.iota(0), b.constant(axis)}) * scale + seed * 19.19f; };
                const auto x = coordinate(0), y = coordinate(1), z = coordinate(2);
                const auto warp = [&](float offset) {
                    return where(distortion != 0.0f, (noise(x + offset, y + offset, z + offset) * 2.0f - 1.0f) * distortion, 0.0f);
                };
                // Invariant expressions are emitted before the octave loop, once per element.
                const auto wx = x + warp(11.7f), wy = y + warp(37.1f), wz = z + warp(73.9f);
                const auto octave = b.iota(1);
                auto amplitude = b.constant(1.0f);
                for (int j = 1; j < 16; ++j)
                    amplitude = amplitude * where(octave >= j, roughness, 1.0f);
                const auto weight = clamp(detail - octave.cast(DataType::Float32), 0.0f, 1.0f);
                const auto frequency = (b.constant(1u) << octave.cast(DataType::UInt32)).cast(DataType::Float32);
                const auto value = where(weight > 0.0f, noise(wx * frequency, wy * frequency, wz * frequency) * amplitude * weight, 0.0f);
                const auto sum = b.fold(fused::Fold::Sum, value, 1);
                const auto normalization = b.fold(fused::Fold::Sum, amplitude * weight, 1);
                b.output(clamp(where(normalization == 0.0f, 0.0f, fused::precise_divide(sum, normalization)), 0.0f, 1.0f), DataType::Float32);
                return fused::Kernel(b);
            }();
            return kernel;
        }

        void compensated_add(float value, float& sum, float& compensation) {
            const float y = value - compensation, next = sum + y;
            compensation = std::isfinite(next) ? (next - sum) - y : 0.0f;
            sum = next;
        }

        // One cached program per search depth, never per set of control values.
        template <class Build>
        const fused::Kernel& lookup_kernel(size_t count, Build build) {
            static std::array<std::once_flag, 32> once;
            static std::array<std::unique_ptr<fused::Kernel>, 32> kernels;
            const auto depth = std::bit_width(count - 1);
            LFS_ASSERT_MSG(depth < kernels.size(), std::format("procedural control count exceeds int32 (count={})", count));
            std::call_once(once[depth], [&] { kernels[depth] = std::make_unique<fused::Kernel>(build(depth)); });
            return *kernels[depth];
        }

        Tensor bind_controls(Tensor host, size_t parameter_words, const Tensor& reference) {
            // Small blocks go directly in the launch arguments. Larger tables retain
            // the field's backend even when it differs from the current thread's scope.
            if (host.numel() + parameter_words <= 64 || reference.device() == Device::CPU)
                return host;
            auto table = internal::allocate_like(reference, host.shape(), DataType::Float32);
            table.copy_from(host);
            return table;
        }

        Tensor controls(const float* values, size_t rows, size_t columns) {
            auto host = Tensor::empty({rows, columns}, Device::CPU);
            std::copy_n(values, rows * columns, host.ptr<float>());
            return host;
        }

        Expr segment(fused::Builder& b, const fused::Input& table, const Expr& x,
                     const Expr& offset, const Expr& count, size_t depth) {
            auto index = b.constant(0);
            for (size_t d = depth; d > 0; --d) {
                const auto candidate = fused::min(index + int32_t(1u << (d - 1)), count - 1);
                index = where(x >= table.gather({offset + candidate, b.constant(0)}), candidate, index);
            }
            return fused::min(index, count - 2);
        }

        template <class F>
        F hermite(F x, F ax, F ay, F at, F bx, F by, F bt) {
            const auto h = bx - ax, t = (x - ax) / h, t2 = t * t, t3 = t2 * t;
            return (t3 * 2.0f - t2 * 3.0f + 1.0f) * ay + (t3 - t2 * 2.0f + t) * h * at +
                   (t2 * 3.0f - t3 * 2.0f) * by + (t3 - t2) * h * bt;
        }

        Expr curve_expression(fused::Builder& b, const fused::Input& table, Expr x, const Expr& offset,
                              const Expr& count, size_t depth, const Expr& clamped) {
            const auto get = [&](const Expr& row, int column) { return table.gather({offset + row, b.constant(column)}); };
            x = where(clamped, clamp(x, get(b.constant(0), 0), get(count - 1, 0)), x);
            const auto a = segment(b, table, x, offset, count, depth), c = a + 1;
            return hermite(x, get(a, 0), get(a, 1), get(a, 2), get(c, 0), get(c, 1), get(c, 2));
        }

        float curve_scalar(float x, std::span<const CurveKnot> knots, bool clamped) {
            if (clamped)
                x = std::clamp(x, knots.front()[0], knots.back()[0]);
            const auto upper = std::upper_bound(knots.begin(), knots.end(), x, [](float value, const auto& knot) { return value < knot[0]; });
            const auto index = std::clamp<ptrdiff_t>(upper - knots.begin() - 1, 0, knots.size() - 2);
            const auto& a = knots[index];
            const auto& c = knots[index + 1];
            return hermite(x, a[0], a[1], a[2], c[0], c[1], c[2]);
        }

        void check_knots(std::span<const CurveKnot> knots) {
            LFS_ASSERT_MSG(knots.size() >= 2 && knots.size() <= INT32_MAX, std::format("curve requires 2..INT32_MAX knots (count={})", knots.size()));
            for (size_t i = 0; i < knots.size(); ++i) {
                const auto& k = knots[i];
                LFS_ASSERT_MSG(std::isfinite(k[0]) && std::isfinite(k[1]) && std::isfinite(k[2]) && (!i || k[0] > knots[i - 1][0]),
                               std::format("invalid curve knot (index={}, x={}, y={}, tangent={})", i, k[0], k[1], k[2]));
            }
        }
    } // namespace

    Tensor procedural_noise(const Tensor& positions, const Tensor& scale, const Tensor& detail,
                            const Tensor& roughness, const Tensor& distortion, const Tensor& seed) {
        check_field(positions, positions, 3);
        for (const auto* field : {&scale, &detail, &roughness, &distortion, &seed})
            check_field(*field, positions);
        pin_operands({&positions, &scale, &detail, &roughness, &distortion, &seed});
        const auto n = positions.size(0);
        if (n && positions.device() == Device::GPU)
            return noise_kernel()({n, 16}, {positions, scale, detail, roughness, distortion, seed})[0];
        auto result = internal::allocate_like(positions, {n}, DataType::Float32);
        if (!n)
            return result;
        const auto p = positions.contiguous(), s = scale.contiguous(), d = detail.contiguous(), r = roughness.contiguous(), w = distortion.contiguous(), k = seed.contiguous();
        tbb::parallel_for(size_t{0}, n, [&](size_t i) {
            float x = p.ptr<float>()[3 * i] * s.ptr<float>()[i] + k.ptr<float>()[i] * 19.19f;
            float y = p.ptr<float>()[3 * i + 1] * s.ptr<float>()[i] + k.ptr<float>()[i] * 19.19f;
            float z = p.ptr<float>()[3 * i + 2] * s.ptr<float>()[i] + k.ptr<float>()[i] * 19.19f;
            const float distortion_value = w.ptr<float>()[i];
            if (distortion_value != 0.0f) {
                const float wx = (noise(x + 11.7f, y + 11.7f, z + 11.7f) * 2.0f - 1.0f) * distortion_value;
                const float wy = (noise(x + 37.1f, y + 37.1f, z + 37.1f) * 2.0f - 1.0f) * distortion_value;
                const float wz = (noise(x + 73.9f, y + 73.9f, z + 73.9f) * 2.0f - 1.0f) * distortion_value;
                x += wx;
                y += wy;
                z += wz;
            }
            const float detail_value = clamp(d.ptr<float>()[i], 0.0f, 15.0f), roughness_value = clamp(r.ptr<float>()[i], 0.0f, 1.0f);
            float sum = 0, norm = 0, sum_comp = 0, norm_comp = 0, amplitude = 1, frequency = 1;
            for (int octave = 0; octave < 16; ++octave) {
                const float weight = clamp(detail_value - float(octave), 0.0f, 1.0f);
                const float value = weight > 0 ? noise(x * frequency, y * frequency, z * frequency) * amplitude * weight : 0.0f;
                compensated_add(value, sum, sum_comp);
                compensated_add(amplitude * weight, norm, norm_comp);
                amplitude *= roughness_value;
                frequency *= 2;
            }
            result.ptr<float>()[i] = clamp(norm == 0 ? 0.0f : sum / norm, 0.0f, 1.0f);
        });
        return result;
    }

    Tensor procedural_gradient(const Tensor& positions, GradientType type) {
        check_field(positions, positions, 3);
        const auto n = positions.size(0);
        if (n && positions.device() == Device::GPU) {
            static const auto kernel = [] {
                fused::Builder b(1);
                const auto p = b.input(DataType::Float32, 2), parameter = b.input(DataType::Int32, 1);
                const auto x = p.gather({b.iota(0), b.constant(0)}), y = p.gather({b.iota(0), b.constant(1)}), z = p.gather({b.iota(0), b.constant(2)});
                const auto mode = parameter.at({0}), t = clamp(x, 0.0f, 1.0f), radius = fused::sqrt(x * x + y * y);
                auto angle = fused::acos(clamp(where(radius == 0.0f, 0.0f, x / radius), -1.0f, 1.0f));
                angle = where(y < 0.0f, -angle + 2 * std::numbers::pi_v<float>, angle) / (2 * std::numbers::pi_v<float>);
                const auto value = where(mode == int(GradientType::Quadratic), x * x,
                                         where(mode == int(GradientType::Easing), t * t * (-t * 2.0f + 3.0f),
                                               where(mode == int(GradientType::Diagonal), (x + y) * 0.5f,
                                                     where(mode == int(GradientType::Spherical), fused::sqrt((x * x + y * y) + z * z),
                                                           where(mode == int(GradientType::Radial), angle, x)))));
                b.output(value, DataType::Float32);
                return fused::Kernel(b);
            }();
            auto parameter = Tensor::empty({1}, Device::CPU, DataType::Int32);
            parameter.ptr<int>()[0] = int(type);
            return kernel({n}, {positions, parameter})[0];
        }
        auto result = internal::allocate_like(positions, {n}, DataType::Float32);
        if (!n)
            return result;
        const auto p = positions.contiguous();
        tbb::parallel_for(size_t{0}, n, [&](size_t i) {
            const float x = p.ptr<float>()[3 * i], y = p.ptr<float>()[3 * i + 1], z = p.ptr<float>()[3 * i + 2];
            float value = x;
            switch (type) {
            case GradientType::Linear: break;
            case GradientType::Quadratic: value = x * x; break;
            case GradientType::Easing: {
                const float t = clamp(x, 0.0f, 1.0f);
                value = t * t * (-t * 2.0f + 3.0f);
                break;
            }
            case GradientType::Diagonal: value = (x + y) * 0.5f; break;
            case GradientType::Spherical: value = std::sqrt((x * x + y * y) + z * z); break;
            case GradientType::Radial: {
                const float radius = std::sqrt(x * x + y * y);
                const float angle = std::acos(clamp(radius == 0 ? 0.0f : x / radius, -1.0f, 1.0f));
                value = (y < 0 ? -angle + 2 * std::numbers::pi_v<float> : angle) / (2 * std::numbers::pi_v<float>);
                break;
            }
            }
            result.ptr<float>()[i] = value;
        });
        return result;
    }

    Tensor interpolate_curve(const Tensor& values, std::span<const CurveKnot> knots, bool clamped) {
        check_field(values, values);
        check_knots(knots);
        const auto n = values.size(0);
        if (n && values.device() == Device::GPU) {
            const auto& kernel = lookup_kernel(knots.size(), [](size_t depth) {
                fused::Builder b(1);
                const auto x = b.input(DataType::Float32, 1).load();
                const auto table = b.input(DataType::Float32, 2), parameters = b.input(DataType::Int32, 1);
                b.output(curve_expression(b, table, x, b.constant(0), parameters.at({0}), depth, parameters.at({1}) != 0), DataType::Float32);
                return fused::Kernel(b);
            });
            auto parameters = Tensor::empty({2}, Device::CPU, DataType::Int32);
            parameters.ptr<int>()[0] = int(knots.size());
            parameters.ptr<int>()[1] = clamped;
            const auto table = bind_controls(controls(knots.front().data(), knots.size(), 3), parameters.numel(), values);
            return kernel({n}, {values, table, parameters})[0];
        }
        auto result = internal::allocate_like(values, {n}, DataType::Float32);
        if (n) {
            const auto source = values.contiguous();
            tbb::parallel_for(size_t{0}, n, [&](size_t i) { result.ptr<float>()[i] = curve_scalar(source.ptr<float>()[i], knots, clamped); });
        }
        return result;
    }

    Tensor interpolate_colour_ramp(const Tensor& values, std::span<const ColourStop> stops, RampInterpolation interpolation, bool alpha) {
        check_field(values, values);
        LFS_ASSERT_MSG(!stops.empty() && stops.size() <= INT32_MAX, std::format("ramp requires 1..INT32_MAX stops (count={})", stops.size()));
        for (size_t i = 0; i < stops.size(); ++i)
            LFS_ASSERT_MSG(std::ranges::all_of(stops[i], [](float x) { return std::isfinite(x); }) && (!i || stops[i][0] >= stops[i - 1][0]),
                           std::format("invalid ramp stop (index={}, position={})", i, stops[i][0]));
        const size_t n = values.size(0), channels = alpha ? 1 : 3;
        if (n && values.device() == Device::GPU) {
            const auto& kernel = lookup_kernel(stops.size(), [](size_t depth) {
                fused::Builder b(2);
                const auto x = b.input(DataType::Float32, 1).load({0});
                const auto table = b.input(DataType::Float32, 2), parameters = b.input(DataType::Int32, 1);
                const auto count = parameters.at({0}), mode = parameters.at({1}), channel = b.iota(1) + parameters.at({2}) + 1;
                const auto index = fused::max(segment(b, table, x, b.constant(0), count, depth), 0);
                const auto get = [&](Expr row, Expr column) { return table.gather({row, column}, fused::Bounds::Clamp); };
                auto t = clamp((x - get(index, b.constant(0))) / fused::max(get(index + 1, b.constant(0)) - get(index, b.constant(0)), 1e-12f), 0.0f, 1.0f);
                t = where(mode == int(RampInterpolation::Constant), (t >= 1.0f).cast(DataType::Float32),
                          where(mode == int(RampInterpolation::Ease), t * t * (-t * 2.0f + 3.0f), t));
                const auto a = get(index, channel), c = get(index + 1, channel);
                b.output(where(x >= get(count - 1, b.constant(0)), get(count - 1, channel), a + t * (c - a)), DataType::Float32);
                return fused::Kernel(b);
            });
            auto parameters = Tensor::empty({3}, Device::CPU, DataType::Int32);
            parameters.ptr<int>()[0] = int(stops.size());
            parameters.ptr<int>()[1] = int(interpolation);
            parameters.ptr<int>()[2] = alpha ? 3 : 0;
            const auto table = bind_controls(controls(stops.front().data(), stops.size(), 5), parameters.numel(), values);
            auto result = kernel({n, channels}, {values, table, parameters})[0];
            return alpha ? result.squeeze(1) : result;
        }
        auto result = internal::allocate_like(values, alpha ? TensorShape{n} : TensorShape{n, channels}, DataType::Float32);
        if (!n)
            return result;
        const auto source = values.contiguous();
        tbb::parallel_for(size_t{0}, n, [&](size_t i) {
            const float x = source.ptr<float>()[i];
            const auto upper = std::upper_bound(stops.begin(), stops.end(), x, [](float v, const auto& s) { return v < s[0]; });
            const size_t index = size_t(std::max<ptrdiff_t>(0, upper - stops.begin() - 1));
            const auto& a = stops[index];
            const auto& c = stops[std::min(index + 1, stops.size() - 1)];
            float t = clamp((x - a[0]) / std::max(c[0] - a[0], 1e-12f), 0.0f, 1.0f);
            if (interpolation == RampInterpolation::Constant)
                t = t >= 1 ? 1 : 0;
            else if (interpolation == RampInterpolation::Ease)
                t = t * t * (-t * 2 + 3);
            for (size_t j = 0; j < channels; ++j) {
                const size_t channel = alpha ? 4 : j + 1;
                result.ptr<float>()[i * channels + j] = x >= stops.back()[0] ? stops.back()[channel] : a[channel] + t * (c[channel] - a[channel]);
            }
        });
        return result;
    }

    Tensor interpolate_rgb_curves(const Tensor& colours, const Tensor& selection, const Tensor& factor,
                                  const std::array<std::span<const CurveKnot>, 4>& curves, bool sh_dc) {
        check_field(colours, colours, 3);
        check_field(selection, colours);
        check_field(factor, colours);
        pin_operands({&colours, &selection, &factor});
        size_t maximum = 0, count = 0;
        for (const auto& curve : curves) {
            check_knots(curve);
            maximum = std::max(maximum, curve.size());
            count += curve.size();
        }
        const size_t n = colours.size(0);
        constexpr float c0 = 0.28209479177387814f;
        if (n && colours.device() == Device::GPU) {
            const auto& kernel = lookup_kernel(maximum, [](size_t depth) {
                fused::Builder b(2);
                const auto original = b.input(DataType::Float32, 2).load();
                const auto weight = b.input(DataType::Float32, 1).load({0}) * clamp(b.input(DataType::Float32, 1).load({0}), 0.0f, 1.0f);
                const auto table = b.input(DataType::Float32, 2), parameters = b.input(DataType::Int32, 1);
                const auto sh = parameters.at({8}) != 0;
                const auto rgb = where(sh, original * c0 + 0.5f, original);
                const auto combined = curve_expression(b, table, rgb, b.constant(0), parameters.at({1}), depth, b.constant(true));
                const auto channel = (b.iota(1) + 1) * 2;
                const auto mapped = curve_expression(b, table, combined, parameters.gather({channel}), parameters.gather({channel + 1}), depth, b.constant(true));
                const auto blended = where(weight == 0.0f, rgb, where(weight == 1.0f, mapped, rgb * (-weight + 1.0f) + mapped * weight));
                b.output(where(sh, (blended - 0.5f) / c0, blended), DataType::Float32);
                return fused::Kernel(b);
            });
            auto table = Tensor::empty({count, 3}, Device::CPU);
            auto parameters = Tensor::empty({9}, Device::CPU, DataType::Int32);
            size_t offset = 0;
            for (size_t j = 0; j < 4; ++j) {
                parameters.ptr<int>()[2 * j] = int(offset);
                parameters.ptr<int>()[2 * j + 1] = int(curves[j].size());
                std::copy_n(curves[j].front().data(), curves[j].size() * 3, table.ptr<float>() + offset * 3);
                offset += curves[j].size();
            }
            parameters.ptr<int>()[8] = sh_dc;
            table = bind_controls(std::move(table), parameters.numel(), colours);
            return kernel({n, 3}, {colours, selection, factor, table, parameters})[0];
        }
        auto result = internal::allocate_like(colours, {n, 3}, DataType::Float32);
        if (!n)
            return result;
        const auto rgb = colours.contiguous(), s = selection.contiguous(), f = factor.contiguous();
        tbb::parallel_for(size_t{0}, n, [&](size_t i) {
            const float weight = s.ptr<float>()[i] * clamp(f.ptr<float>()[i], 0.0f, 1.0f);
            for (size_t j = 0; j < 3; ++j) {
                const float original = rgb.ptr<float>()[i * 3 + j], value = sh_dc ? original * c0 + 0.5f : original;
                const float mapped = curve_scalar(curve_scalar(value, curves[0], true), curves[j + 1], true);
                const float blended = weight == 0 ? value : weight == 1 ? mapped
                                                                        : value * (-weight + 1.0f) + mapped * weight;
                result.ptr<float>()[i * 3 + j] = sh_dc ? (blended - 0.5f) / c0 : blended;
            }
        });
        return result;
    }
} // namespace lfs::core
