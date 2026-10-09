/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The Metal and Vulkan Adam, Sh and Morton families against CPU transliterations of the
// CUDA kernels, on the inputs of the parity fixtures (capture_adam, capture_sh,
// capture_morton) plus larger, Q16 and skipped-row cases.
//
// Bookkeeping, copies and bounds are exact. Linear codec re-encoding may
// differ by one code within one float ULP of a rounding boundary. An Adam step's code may differ by one
// where the value it rounds lies within kAdamTie of a half step: the GPU's
// fast log and exp differ from the CPU's in the last bits.

#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_execution.hpp"
#include "lfs/training/idle_arena_scratch.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "optimizer/adam_optimizer.hpp"

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
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
    namespace ops = lfs::gpu_ops;
    namespace joint = lfs::training::joint_adam;
    namespace quant = lfs::core::sh_value_quant;

    constexpr uint32_t kR = 32;
    constexpr float kEps = 1e-15f;
    constexpr double kAdamTie = 0.05;

    class PortableAdamShMorton : public ::testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(backend_under_test()))
                GTEST_SKIP() << lfs::core::gpu_backend_name(GetParam()) << " device unavailable";
            session_.emplace(backend_under_test());
            ASSERT_TRUE(session_->switched());
            const auto& table = lfs::training::training_ops(backend_under_test());
            if (table.adam == nullptr || table.sh == nullptr || table.morton == nullptr)
                GTEST_SKIP() << "Adam, Sh or Morton slot is empty";
            adam = table.adam;
            sh = table.sh;
            morton = table.morton;
        }

        std::optional<lfs::test::DefaultGpuBackendForTesting> session_;
        const ops::AdamOps* adam = nullptr;
        const ops::ShOps* sh = nullptr;
        const ops::MortonOps* morton = nullptr;
    };

    // Fixture generators, as in test_training_ops_parity.cpp.
    std::vector<float> pattern(size_t count, float scale, int seed) {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) {
            const auto k = static_cast<int>((i * 7919u + static_cast<size_t>(seed) * 104729u) % 2003u);
            values[i] = scale * (static_cast<float>(k) / 1001.0f - 1.0f);
        }
        return values;
    }

    std::vector<uint8_t> period_mask(size_t count, size_t period) {
        std::vector<uint8_t> values(count);
        for (size_t i = 0; i < count; ++i)
            values[i] = i % period == 0 ? 1 : 0;
        return values;
    }

    std::vector<int64_t> shuffled(size_t n, unsigned seed) {
        std::vector<int64_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(perm.begin(), perm.end(), rng);
        return perm;
    }

    template <class T>
    Tensor upload(const std::vector<T>& values, lfs::core::TensorShape shape, DataType dtype) {
        Tensor cpu = Tensor::empty(shape, Device::CPU, dtype);
        EXPECT_EQ(cpu.bytes(), values.size() * sizeof(T));
        std::memcpy(cpu.data_ptr(), values.data(), cpu.bytes());
        return cpu.gpu();
    }

    Tensor gpu_f(const std::vector<float>& values) { return upload(values, {values.size()}, DataType::Float32); }

    template <class T>
    std::vector<T> host(const Tensor& tensor) {
        const Tensor cpu = tensor.cpu().contiguous();
        std::vector<T> values(cpu.bytes() / sizeof(T));
        std::memcpy(values.data(), cpu.data_ptr(), cpu.bytes());
        return values;
    }

    uint32_t slot_index(uint32_t prim, uint32_t k, uint32_t slots) {
        return (prim / kR) * (slots * kR) + k * kR + prim % kR;
    }

    template <class T>
    void expect_equal(const std::vector<T>& actual, const std::vector<T>& expected, const std::string& name) {
        ASSERT_EQ(actual.size(), expected.size()) << name;
        size_t mismatches = 0;
        size_t first = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (std::memcmp(&actual[i], &expected[i], sizeof(T)) != 0 && mismatches++ == 0)
                first = i;
        }
        EXPECT_EQ(mismatches, 0u) << name << ": first mismatch at " << first << " of " << actual.size() << ": "
                                  << +actual[first] << " vs " << +expected[first];
    }

    void expect_near(const std::vector<float>& actual, const std::vector<float>& expected, double abs, double rel,
                     const std::string& name) {
        ASSERT_EQ(actual.size(), expected.size()) << name;
        for (size_t i = 0; i < actual.size(); ++i) {
            const double limit = abs + rel * std::abs(static_cast<double>(expected[i]));
            ASSERT_LE(std::abs(static_cast<double>(actual[i]) - expected[i]), limit)
                << name << " index " << i << ": " << actual[i] << " vs " << expected[i];
        }
    }

    // Quantized codes of `bits` (8 or 16) with the value each was rounded from;
    // NaN marks a code the reference did not write.
    struct Codes {
        int bits = 16;
        std::vector<uint8_t> bytes;
        std::vector<float> unrounded;

        Codes(int code_bits, std::vector<uint8_t> raw)
            : bits(code_bits), bytes(std::move(raw)), unrounded(bytes.size() * 8 / bits, std::numeric_limits<float>::quiet_NaN()) {}

        [[nodiscard]] float q_max() const { return static_cast<float>((1 << bits) - 1); }

        [[nodiscard]] uint32_t get(size_t i) const {
            if (bits == 8)
                return bytes[i];
            uint16_t code = 0;
            std::memcpy(&code, &bytes[2 * i], 2);
            return code;
        }

        // Rounds `x` into code i, as the CUDA codecs do.
        void put(size_t i, float x) {
            const auto q = static_cast<uint32_t>(std::min(std::max(std::round(x), 0.0f), q_max()));
            if (bits == 8) {
                bytes[i] = static_cast<uint8_t>(q);
            } else {
                const auto code = static_cast<uint16_t>(q);
                std::memcpy(&bytes[2 * i], &code, 2);
            }
            unrounded[i] = x;
        }

        // nvcc contracts the CUDA decode into an fma.
        [[nodiscard]] float decode(size_t i, float lo, float hi) const {
            return std::fma(hi - lo, static_cast<float>(get(i)) * (1.0f / q_max()), lo);
        }
    };

    void expect_codes(const std::vector<uint8_t>& actual_bytes, const Codes& expected, std::optional<double> window,
                      const std::string& name) {
        const Codes actual(expected.bits, actual_bytes);
        ASSERT_EQ(actual.bytes.size(), expected.bytes.size()) << name;
        for (size_t i = 0; i < expected.unrounded.size(); ++i) {
            const uint32_t a = actual.get(i);
            const uint32_t e = expected.get(i);
            if (a == e)
                continue;
            const float x = expected.unrounded[i];
            const double tolerance = window.value_or(std::nextafter(x, std::numeric_limits<float>::infinity()) - x);
            const bool tie = !std::isnan(x) && std::abs(x - std::floor(x) - 0.5f) <= tolerance &&
                             (a == e + 1 || e == a + 1);
            ASSERT_TRUE(tie) << name << " code " << i << ": " << a << " vs " << e << " from " << x;
        }
    }

    // ---- joint (u, log_s) codec, joint_adam_codec.cuh ----

    std::array<float, 2> joint_decode_us(const Codes& c, size_t cell, const float* mm) {
        return {c.decode(2 * cell, mm[0], mm[1]), c.decode(2 * cell + 1, mm[2], mm[3])};
    }

    // The CUDA codec's __expf and __logf: base-2 functions of a scaled argument.
    float fast_exp(float x) { return std::exp2(x * 1.44269504f); }
    float fast_log(float x) { return std::log2(x) * 0.693147181f; }

    std::array<float, 2> joint_g1g2(const std::array<float, 2> us) {
        const float e1 = us[1] > 0.118f ? fast_exp(us[1]) - 1.0f : std::expm1(us[1]);
        const float sqrt_g2 = kEps * e1;
        const float g2 = sqrt_g2 * sqrt_g2;
        return {g2 == 0.0f ? 0.0f : us[0] * (sqrt_g2 + kEps), g2};
    }

    std::array<float, 2> joint_us(float g1, float g2) {
        const float sqrt_g2 = std::sqrt(std::max(g2, 0.0f));
        const float x = sqrt_g2 * (1.0f / kEps);
        return {g1 / (sqrt_g2 + kEps), x > 0.125f ? fast_log(1.0f + x) : std::log1p(x)};
    }

    void joint_encode(Codes& c, size_t cell, const std::array<float, 2> us, const float* mm) {
        const float inv_u = 1.0f / std::max(mm[1] - mm[0], kEps);
        const float inv_s = 1.0f / std::max(mm[3] - mm[2], kEps);
        c.put(2 * cell, c.q_max() * (us[0] - mm[0]) * inv_u);
        c.put(2 * cell + 1, c.q_max() * (us[1] - mm[2]) * inv_s);
    }

    std::array<float, 4> joint_bounds(float u_min, float u_max, float s_min, float s_max) {
        if (u_min > u_max)
            return {0.f, 0.f, 0.f, 0.f};
        return {u_min, u_max, s_min, s_max};
    }

    // ---- Q16 value codec, sh_value_codec.cuh ----

    void q16_encode(Codes& c, size_t i, float v, float lo, float hi) {
        c.put(i, 65535.0f * (v - lo) / std::max(hi - lo, 1e-20f));
    }

    // ---- CPU Adam: adam_step_joint_contiguous_batched_cu, apply_shN_grads_packed_joint ----

    struct HostMasks {
        std::vector<uint8_t> frozen, crop;
        std::vector<float> share;
    };

    // Rate after the frozen and crop-box factors; false when the step is skipped.
    bool host_rate(const HostMasks& m, const ops::AdamModifiers& mod, size_t prim, float& rate) {
        bool apply = true;
        if (prim < m.frozen.size() && m.frozen[prim]) {
            if (mod.frozen_lr_scale == 0.f)
                apply = false;
            else
                rate *= mod.frozen_lr_scale;
        }
        if (prim < m.crop.size() && m.crop[prim]) {
            if (mod.cropbox_lr_scale == 0.f)
                apply = false;
            else
                rate *= mod.cropbox_lr_scale;
        }
        return apply;
    }

    struct HostStep {
        std::vector<float> parameter, bounds, gradient;
        Codes packed{16, {}};
        int primitives = 0, attributes = 0;
        float lr = 0.f;
        bool screen_share = false;
    };

    void host_step_batch(HostStep& s, const HostMasks& m, const ops::AdamHyper& h, const ops::AdamModifiers& mod,
                         float bc1_rcp, float bc2_sqrt_rcp) {
        const int blocks = (s.primitives + 255) / 256;
        for (int b = 0; b < blocks; ++b) {
            const std::array<float, 4> old{s.bounds[4 * b], s.bounds[4 * b + 1], s.bounds[4 * b + 2],
                                           s.bounds[4 * b + 3]};
            std::vector<std::array<float, 2>> us(256 * s.attributes);
            float u_min = 1e30f, u_max = -1e30f, s_min = 1e30f, s_max = -1e30f;
            const int count = std::min(256, s.primitives - b * 256);
            for (int lane = 0; lane < count; ++lane) {
                const int prim = b * 256 + lane;
                float rate = s.lr;
                const bool apply = host_rate(m, mod, prim, rate);
                const float step = rate * bc1_rcp;
                for (int i = 0; i < s.attributes; ++i) {
                    const size_t cell = static_cast<size_t>(prim) * s.attributes + i;
                    const auto mv = joint_g1g2(joint_decode_us(s.packed, cell, old.data()));
                    float mn = mv[0], vn = mv[1];
                    if (apply) {
                        const float grad = s.gradient[cell];
                        const float limit = mod.screen_share_limit;
                        float hinge = 0.f;
                        if (s.screen_share && static_cast<size_t>(prim) < m.share.size() && limit > 0.f &&
                            limit < 1.f && m.share[prim] > limit && mod.screen_share_penalty > 0.f)
                            hinge = mod.screen_share_penalty * std::log2(m.share[prim] / limit) *
                                    (std::sqrt(mv[1]) * bc2_sqrt_rcp + h.eps);
                        mn = h.beta1 * mv[0] + (1.f - h.beta1) * (grad + hinge);
                        vn = h.beta2 * mv[1] + (1.f - h.beta2) * grad * grad;
                        s.parameter[cell] -= step * mn / (std::sqrt(vn) * bc2_sqrt_rcp + h.eps);
                    }
                    const auto next = joint_us(mn, vn);
                    us[lane * s.attributes + i] = next;
                    u_min = std::min(u_min, next[0]);
                    u_max = std::max(u_max, next[0]);
                    s_min = std::min(s_min, next[1]);
                    s_max = std::max(s_max, next[1]);
                }
            }
            const auto nb = joint_bounds(u_min, u_max, s_min, s_max);
            std::copy(nb.begin(), nb.end(), s.bounds.begin() + 4 * b);
            for (int lane = 0; lane < count; ++lane)
                for (int i = 0; i < s.attributes; ++i)
                    joint_encode(s.packed, static_cast<size_t>(b * 256 + lane) * s.attributes + i,
                                 us[lane * s.attributes + i], nb.data());
        }
    }

    struct HostSh {
        std::vector<float> parameter, bounds, gradient, value_bounds;
        Codes packed{8, {}};
        Codes values{16, {}};
    };

    void host_step_sh(HostSh& s, const HostMasks& m, const ops::AdamHyper& h, const ops::AdamModifiers& mod,
                      const ops::ShStepParams& p) {
        const bool q16 = p.value_bits == 16;
        const auto slots = static_cast<uint32_t>(p.layout_slots);
        const uint32_t active = p.active_bases > 9 ? 12u : p.active_bases > 4 ? 6u
                                                                              : 3u;
        const auto cells = static_cast<uint32_t>(p.value_cells);
        for (int b = 0; b < (p.primitives + 255) / 256; ++b) {
            const std::array<float, 4> old{s.bounds[4 * b], s.bounds[4 * b + 1], s.bounds[4 * b + 2],
                                           s.bounds[4 * b + 3]};
            const float v_old_lo = q16 ? s.value_bounds[2 * b] : 0.f;
            const float v_old_hi = q16 ? s.value_bounds[2 * b + 1] : 0.f;
            float u_min = 1e30f, u_max = -1e30f, s_min = 1e30f, s_max = -1e30f, v_lo = 1e30f, v_hi = -1e30f;
            struct Cell {
                size_t cell, value_code;
                std::array<float, 2> us;
                float value;
                bool has_value;
            };
            std::vector<Cell> out;
            const auto count = static_cast<uint32_t>(std::min(256, p.primitives - b * 256));
            for (uint32_t prim = b * 256; prim < b * 256 + count; ++prim) {
                float rate = p.step_size;
                const bool apply = host_rate(m, mod, prim, rate);
                for (uint32_t k = 0; k < std::min(slots, 12u); ++k) {
                    const uint32_t slot = slot_index(prim, k, slots);
                    const bool is_active = k < active;
                    std::array<float, 4> values{};
                    for (uint32_t c = 0; c < 4; ++c) {
                        if (!q16)
                            values[c] = s.parameter[slot * 4 + c];
                        else if (k * 4 + c < cells)
                            values[c] = s.values.decode(slot_index(prim, k * 4 + c, cells), v_old_lo, v_old_hi);
                    }
                    for (uint32_t c = 0; c < 4; ++c) {
                        const size_t cell = static_cast<size_t>(slot) * 4 + c;
                        const float grad = is_active ? s.gradient[cell] : 0.f;
                        const auto mv = joint_g1g2(joint_decode_us(s.packed, cell, old.data()));
                        float mn = mv[0], vn = mv[1];
                        if (apply) {
                            mn = h.beta1 * mv[0] + (1.f - h.beta1) * grad;
                            vn = h.beta2 * mv[1] + (1.f - h.beta2) * grad * grad;
                            if (is_active)
                                values[c] -= rate * mn / (std::sqrt(vn) * p.bc2_sqrt_rcp + h.eps);
                        }
                        const auto us = joint_us(mn, vn);
                        u_min = std::min(u_min, us[0]);
                        u_max = std::max(u_max, us[0]);
                        s_min = std::min(s_min, us[1]);
                        s_max = std::max(s_max, us[1]);
                        const bool has_value = q16 && k * 4 + c < cells;
                        if (has_value) {
                            v_lo = std::min(v_lo, values[c]);
                            v_hi = std::max(v_hi, values[c]);
                        }
                        out.push_back({cell, has_value ? slot_index(prim, k * 4 + c, cells) : 0, us, values[c],
                                       has_value});
                    }
                    if (apply && is_active && !q16)
                        std::copy(values.begin(), values.end(), s.parameter.begin() + slot * 4);
                }
            }
            const auto nb = joint_bounds(u_min, u_max, s_min, s_max);
            std::copy(nb.begin(), nb.end(), s.bounds.begin() + 4 * b);
            if (q16) {
                s.value_bounds[2 * b] = v_lo <= v_hi ? v_lo : 0.f;
                s.value_bounds[2 * b + 1] = v_lo <= v_hi ? v_hi : 0.f;
            }
            for (const Cell& cell : out) {
                joint_encode(s.packed, cell.cell, cell.us, nb.data());
                if (cell.has_value)
                    q16_encode(s.values, cell.value_code, cell.value, s.value_bounds[2 * b], s.value_bounds[2 * b + 1]);
            }
        }
    }

    // Two steps, each from the GPU's state, so the second decodes nonzero
    // moments and a tie rounded differently does not carry over.
    TEST_P(PortableAdamShMorton, AdamStepBatchMatchesCpu) {
        constexpr size_t n = 700;
        constexpr ops::AdamModifiers modifiers{.frozen_lr_scale = 0.25f,
                                               .cropbox_lr_scale = 0.5f,
                                               .screen_share_limit = 0.3f,
                                               .screen_share_penalty = 0.05f};
        constexpr ops::AdamHyper hyper{};
        HostMasks hm{period_mask(n, 5), period_mask(n, 7), pattern(n, 0.4f, 23)};
        for (float& share : hm.share)
            share = std::abs(share);
        const Tensor frozen = upload(hm.frozen, {n}, DataType::Bool);
        const Tensor crop = upload(hm.crop, {n}, DataType::Bool);
        const Tensor share = gpu_f(hm.share);
        const ops::AdamMasks masks{frozen, crop, share};

        constexpr std::array<int, 5> attrs{3, 3, 3, 4, 1};
        std::vector<HostStep> expected(attrs.size());
        std::vector<std::array<Tensor, 4>> tensors;
        std::vector<ops::JointStep> steps;
        for (size_t i = 0; i < attrs.size(); ++i) {
            const size_t count = n - 40 * i;
            HostStep& s = expected[i];
            s.primitives = static_cast<int>(count);
            s.attributes = attrs[i];
            s.gradient = pattern(count * attrs[i], 1e-3f, static_cast<int>(10 * i + 1));
            s.lr = 0.01f * static_cast<float>(i + 1);
            s.screen_share = i == 2;
            tensors.push_back({gpu_f(pattern(count * attrs[i], 0.5f, static_cast<int>(10 * i))),
                               Tensor::zeros({count, size_t(attrs[i]) * 4}, Device::GPU, DataType::UInt8),
                               Tensor::zeros({joint::n_bounds_for_prims(count), 4}, Device::GPU),
                               gpu_f(s.gradient)});
        }
        for (size_t i = 0; i < attrs.size(); ++i) {
            auto& t = tensors[i];
            steps.push_back({.parameter = t[0], .packed = t[1], .bounds = t[2], .gradient = t[3], .primitives = expected[i].primitives, .attributes = expected[i].attributes, .bits = 16, .lr = expected[i].lr, .bc1_rcp = 10.f, .bc2_sqrt_rcp = static_cast<float>(1.0 / std::sqrt(0.001)), .apply_screen_share = expected[i].screen_share});
        }
        for (int iteration = 0; iteration < 2; ++iteration) {
            for (size_t i = 0; i < attrs.size(); ++i) {
                expected[i].parameter = host<float>(tensors[i][0]);
                expected[i].packed = Codes(16, host<uint8_t>(tensors[i][1]));
                expected[i].bounds = host<float>(tensors[i][2]);
            }
            adam->step_batch(steps, masks, hyper, modifiers);
            for (size_t i = 0; i < attrs.size(); ++i) {
                HostStep& s = expected[i];
                host_step_batch(s, hm, hyper, modifiers, steps[i].bc1_rcp, steps[i].bc2_sqrt_rcp);
                const std::string name = "iteration " + std::to_string(iteration) + " step " + std::to_string(i);
                expect_near(host<float>(tensors[i][0]), s.parameter, 1e-6, 1e-5, name + " parameter");
                expect_near(host<float>(tensors[i][2]), s.bounds, 1e-6, 1e-5, name + " bounds");
                expect_codes(host<uint8_t>(tensors[i][1]), s.packed, kAdamTie, name + " moments");
            }
        }
    }

    void run_step_sh(const ops::AdamOps& adam, bool q16, float frozen_lr_scale) {
        constexpr size_t n = 700;
        constexpr uint32_t rest = 15;
        const uint32_t slots = lfs::core::sh_float4_slots_for_rest(rest);
        const size_t floats = lfs::core::sh_swizzled_float_count(n, rest);
        const uint32_t cells = quant::n_value_cells_per_prim(rest);
        const ops::AdamModifiers modifiers{.frozen_lr_scale = frozen_lr_scale, .cropbox_lr_scale = 0.5f};
        constexpr ops::AdamHyper hyper{};
        const HostMasks hm{period_mask(n, 5), period_mask(n, 7), {}};
        const Tensor frozen = upload(hm.frozen, {n}, DataType::Bool);
        const Tensor crop = upload(hm.crop, {n}, DataType::Bool);
        const ops::AdamMasks masks{frozen, crop, {}};
        const ops::ShStepParams params{.primitives = static_cast<int>(n),
                                       .layout_slots = static_cast<int>(slots),
                                       .active_bases = q16 ? 9 : 4,
                                       .value_bits = q16 ? 16 : 0,
                                       .value_cells = q16 ? static_cast<int>(cells) : 0,
                                       .step_size = 0.005f * 10.f,
                                       .bc2_sqrt_rcp = static_cast<float>(1.0 / std::sqrt(0.001))};

        Tensor parameter = gpu_f(pattern(floats, 0.2f, 43));
        Tensor value_bounds;
        if (q16) {
            Tensor codes = Tensor::zeros({quant::sh_value_u16_count(n, rest)}, Device::GPU, DataType::Float16);
            value_bounds = Tensor::zeros({quant::n_bounds_for_prims(n) * 2}, Device::GPU);
            lfs::training::training_ops(backend_under_test()).sh->encode_q16(parameter, codes, value_bounds, n, rest, 0, 0);
            parameter = codes;
        }
        Tensor packed = Tensor::zeros({floats * 2}, Device::GPU, DataType::UInt8);
        Tensor bounds = Tensor::zeros({joint::n_bounds_for_prims(n), 4}, Device::GPU);
        HostSh s;
        s.gradient = pattern(floats, 1e-3f, 41);
        const Tensor gradient = gpu_f(s.gradient);
        for (int iteration = 0; iteration < 2; ++iteration) {
            s.bounds = host<float>(bounds);
            s.packed = Codes(8, host<uint8_t>(packed));
            if (q16) {
                s.values = Codes(16, host<uint8_t>(parameter));
                s.value_bounds = host<float>(value_bounds);
            } else {
                s.parameter = host<float>(parameter);
            }
            adam.step_sh(parameter, packed, bounds, value_bounds, gradient, masks, hyper, modifiers, params);
            host_step_sh(s, hm, hyper, modifiers, params);
            const std::string name = "iteration " + std::to_string(iteration);
            expect_near(host<float>(bounds), s.bounds, 1e-6, 1e-5, name + " sh bounds");
            expect_codes(host<uint8_t>(packed), s.packed, kAdamTie, name + " sh moments");
            if (q16) {
                expect_near(host<float>(value_bounds), s.value_bounds, 1e-6, 1e-5, name + " sh value bounds");
                expect_codes(host<uint8_t>(parameter), s.values, kAdamTie, name + " sh value codes");
            } else {
                expect_near(host<float>(parameter), s.parameter, 1e-6, 1e-5, name + " sh parameter");
            }
        }
    }

    TEST_P(PortableAdamShMorton, AdamStepShFloatMatchesCpu) { run_step_sh(*adam, false, 0.25f); }

    TEST_P(PortableAdamShMorton, AdamStepShQ16WithSkippedRowsMatchesCpu) { run_step_sh(*adam, true, 0.f); }

    // joint_encode_zero_{rows,shN}_cu on random codes and bounds.
    void run_encode_zero(const ops::AdamOps& adam, ops::JointLayout layout, int bits) {
        constexpr size_t n = 700;
        const bool swizzled = layout == ops::JointLayout::SwizzledSH;
        const uint32_t width = swizzled ? 3 : 4;
        const size_t cells = swizzled ? lfs::core::sh_swizzled_padded_n(n) * width * 4 : n * width;
        std::mt19937 rng(bits * 7 + width);
        std::vector<uint8_t> packed(cells * joint::bytes_per_cell(bits));
        for (auto& byte : packed)
            byte = static_cast<uint8_t>(rng() % 101);
        std::vector<float> bounds = pattern(joint::n_bounds_for_prims(n) * 4, 0.5f, 64);
        const std::vector<int64_t> indices{0, 5, 5, 255, 256, 300, 511, 512, 699, 3, -1, 700};
        Tensor packed_gpu = upload(packed, {packed.size()}, DataType::UInt8);
        Tensor bounds_gpu = upload(bounds, {bounds.size() / 4, 4}, DataType::Float32);
        adam.encode_zero(packed_gpu, bounds_gpu, upload(indices, {indices.size()}, DataType::Int64),
                         {.layout = layout, .primitives = int(n), .attributes_or_slots = int(width), .bits = bits});

        Codes expected(bits, packed);
        std::vector<uint8_t> flags(n, 0);
        std::vector<uint8_t> touched(joint::n_bounds_for_prims(n), 0);
        for (const int64_t index : indices) {
            if (index >= 0 && index < static_cast<int64_t>(n)) {
                flags[index] = 1;
                touched[index / 256] = 1;
            }
        }
        const uint32_t row_cells = swizzled ? width * 4 : width;
        for (size_t b = 0; b < touched.size(); ++b) {
            if (!touched[b])
                continue;
            float* mm = &bounds[4 * b];
            const std::array<float, 4> old{mm[0], mm[1], mm[2], mm[3]};
            const std::array<float, 4> nb{std::min(old[0], 0.f), std::max(old[1], 0.f), std::min(old[2], 0.f),
                                          std::max(old[3], 0.f)};
            for (uint32_t prim = b * 256; prim < std::min<size_t>(n, (b + 1) * 256); ++prim) {
                for (uint32_t i = 0; i < row_cells; ++i) {
                    const size_t cell =
                        swizzled ? size_t{slot_index(prim, i / 4, width)} * 4 + i % 4 : size_t{prim} * width + i;
                    if (old != nb)
                        joint_encode(expected, cell, joint_decode_us(expected, cell, old.data()), nb.data());
                    if (flags[prim])
                        joint_encode(expected, cell, {0.f, 0.f}, nb.data());
                }
            }
            std::copy(nb.begin(), nb.end(), mm);
        }
        expect_equal(host<uint8_t>(packed_gpu), expected.bytes, "encode_zero packed");
        expect_equal(host<float>(bounds_gpu), bounds, "encode_zero bounds");
    }

    TEST_P(PortableAdamShMorton, EncodeZeroRows16MatchesCpu) { run_encode_zero(*adam, ops::JointLayout::Rows, 16); }

    TEST_P(PortableAdamShMorton, EncodeZeroSwizzled8MatchesCpu) {
        run_encode_zero(*adam, ops::JointLayout::SwizzledSH, 8);
    }

    // ---- Sh ----

    struct HostQ16 {
        Codes codes{16, {}};
        std::vector<float> bounds;
    };

    // encode_float4_to_u16_block_kernel
    HostQ16 host_encode_q16(const std::vector<float>& swizzled, size_t n, uint32_t rest) {
        const uint32_t slots = lfs::core::sh_float4_slots_for_rest(rest);
        const uint32_t cells = quant::n_value_cells_per_prim(rest);
        HostQ16 out{Codes(16, std::vector<uint8_t>(quant::sh_value_u16_count(n, rest) * 2, 0)),
                    std::vector<float>(quant::n_bounds_for_prims(n) * 2, 0.f)};
        const auto value = [&](uint32_t p, uint32_t c) { return swizzled[slot_index(p, c / 4, slots) * 4 + c % 4]; };
        for (size_t b = 0; b < quant::n_bounds_for_prims(n); ++b) {
            float lo = 1e30f, hi = -1e30f;
            const size_t end = std::min(n, (b + 1) * 256);
            for (uint32_t p = b * 256; p < end; ++p) {
                for (uint32_t c = 0; c < cells; ++c) {
                    lo = std::min(lo, value(p, c));
                    hi = std::max(hi, value(p, c));
                }
            }
            if (lo > hi)
                lo = hi = 0.f;
            out.bounds[2 * b] = lo;
            out.bounds[2 * b + 1] = hi;
            for (uint32_t p = b * 256; p < end; ++p)
                for (uint32_t c = 0; c < cells; ++c)
                    q16_encode(out.codes, slot_index(p, c, cells), value(p, c), lo, hi);
        }
        return out;
    }

    TEST_P(PortableAdamShMorton, Q16EncodeDecodeMatchesCpu) {
        for (const auto [n, rest] : {std::pair<size_t, uint32_t>{8, 3}, {600, 15}, {300, 8}}) {
            const std::string name = std::to_string(n) + "x" + std::to_string(rest);
            const auto src = pattern(lfs::core::sh_swizzled_float_count(n, rest), 0.25f, 3);
            Tensor codes = Tensor::zeros({quant::sh_value_u16_count(n, rest)}, Device::GPU, DataType::Float16);
            Tensor bounds = Tensor::zeros({quant::n_bounds_for_prims(n) * 2}, Device::GPU);
            sh->encode_q16(gpu_f(src), codes, bounds, n, rest, 0, 0);
            const HostQ16 expected = host_encode_q16(src, n, rest);
            expect_equal(host<uint8_t>(codes), expected.codes.bytes, name + " codes");
            expect_equal(host<float>(bounds), expected.bounds, name + " bounds");

            // Decode the GPU's codes: slots of padding primitives stay zero.
            const Codes gpu_codes(16, host<uint8_t>(codes));
            Tensor decoded = Tensor::zeros({src.size()}, Device::GPU);
            sh->decode_q16(codes, bounds, decoded, n, rest);
            const uint32_t slots = lfs::core::sh_float4_slots_for_rest(rest);
            const uint32_t cells = quant::n_value_cells_per_prim(rest);
            std::vector<float> round_trip(src.size(), 0.f);
            std::vector<float> source(src.size(), 0.f);
            for (uint32_t p = 0; p < n; ++p) {
                const float lo = expected.bounds[2 * (p / 256)], hi = expected.bounds[2 * (p / 256) + 1];
                for (uint32_t c = 0; c < cells; ++c) {
                    const size_t at = slot_index(p, c / 4, slots) * 4 + c % 4;
                    round_trip[at] = gpu_codes.decode(slot_index(p, c, cells), lo, hi);
                    source[at] = src[at];
                }
            }
            expect_equal(host<float>(decoded), round_trip, name + " decode");
            // Within half a step of the block's [-0.25, 0.25] range.
            expect_near(round_trip, source, 0.25 / 65535.0 + 1e-7, 0, name + " round trip");
        }
    }

    TEST_P(PortableAdamShMorton, Q16ChunkEncodeWritesAtOffsets) {
        constexpr size_t n = 256;
        constexpr uint32_t rest = 3;
        const auto src = pattern(lfs::core::sh_swizzled_float_count(n, rest), 0.25f, 5);
        const size_t count = quant::sh_value_u16_count(n, rest);
        Tensor codes = Tensor::zeros({3 * count}, Device::GPU, DataType::Float16);
        Tensor bounds = Tensor::zeros({6}, Device::GPU);
        sh->encode_q16(gpu_f(src), codes, bounds, n, rest, count, 2);
        const HostQ16 chunk = host_encode_q16(src, n, rest);
        std::vector<uint8_t> expected(6 * count, 0);
        std::copy(chunk.codes.bytes.begin(), chunk.codes.bytes.end(), expected.begin() + 2 * count);
        expect_equal(host<uint8_t>(codes), expected, "chunk codes");
        expect_equal(host<float>(bounds), std::vector<float>{0.f, 0.f, chunk.bounds[0], chunk.bounds[1], 0.f, 0.f},
                     "chunk bounds");
    }

    // block_ids and block_runs on the fixture's shape, then reencode_touched
    // with a duplicate destination, a canonical order and rows past decode_rows.
    TEST_P(PortableAdamShMorton, TouchedBlockReencodeMatchesCpu) {
        const std::vector<int64_t> dest{0, 256, 257, 512, 256, -3};
        Tensor ids = Tensor::zeros({dest.size()}, Device::GPU);
        sh->block_ids(upload(dest, {dest.size()}, DataType::Int64), ids);
        expect_equal(host<float>(ids), std::vector<float>{0.f, 1.f, 1.f, 2.f, 1.f, -1.f}, "block ids");

        const std::vector<float> sorted_keys{-1.f, 0.f, 0.f, 1.f, 1.f, 1.f, 4.f, 9.f, 9.f};
        Tensor unique = Tensor::zeros({sorted_keys.size()}, Device::GPU, DataType::Int32);
        Tensor offsets = Tensor::zeros({sorted_keys.size()}, Device::GPU, DataType::Int32);
        Tensor runs = Tensor::zeros({1}, Device::GPU, DataType::Int32);
        auto state = sh->create_run_scratch();
        ASSERT_TRUE(state);
        sh->block_runs(*state, gpu_f(sorted_keys), unique, offsets, runs);
        expect_equal(host<int32_t>(runs), std::vector<int32_t>{5}, "run count");
        expect_equal(host<int32_t>(unique), std::vector<int32_t>{-1, 0, 1, 4, 9, 0, 0, 0, 0}, "run ids");
        expect_equal(host<int32_t>(offsets), std::vector<int32_t>{0, 1, 3, 6, 7, 0, 0, 0, 0}, "run offsets");

        constexpr size_t n = 600;
        constexpr uint32_t rest = 8;
        constexpr size_t decode_rows = 590;
        const uint32_t cells = quant::n_value_cells_per_prim(rest);
        const auto src = pattern(lfs::core::sh_swizzled_float_count(n, rest), 0.25f, 9);
        Tensor codes = Tensor::zeros({quant::sh_value_u16_count(n, rest)}, Device::GPU, DataType::Float16);
        Tensor bounds = Tensor::zeros({quant::n_bounds_for_prims(n) * 2}, Device::GPU);
        sh->encode_q16(gpu_f(src), codes, bounds, n, rest, 0, 0);
        HostQ16 expected{Codes(16, host<uint8_t>(codes)), host<float>(bounds)};

        // Touches sorted by block; canonical_order maps them to canonical rows.
        const std::vector<int64_t> sorted_dest{3, 7, 513, 599, 513};
        const std::vector<float> sorted_ids{0.f, 0.f, 2.f, 2.f, 2.f};
        const std::vector<int64_t> order{4, 0, 2, 1, 3};
        const auto canonical = pattern(sorted_dest.size() * cells, 0.8f, 21);
        Tensor t_unique = Tensor::zeros({sorted_dest.size()}, Device::GPU, DataType::Int32);
        Tensor t_offsets = Tensor::zeros({sorted_dest.size()}, Device::GPU, DataType::Int32);
        Tensor t_runs = Tensor::zeros({1}, Device::GPU, DataType::Int32);
        sh->block_runs(*state, gpu_f(sorted_ids), t_unique, t_offsets, t_runs);
        const ops::Q16TouchParams params{.sorted_count = sorted_dest.size(), .primitives = n, .decode_source_rows = decode_rows, .rest = rest};
        sh->reencode_touched(codes, bounds, upload(canonical, {sorted_dest.size(), rest, 3}, DataType::Float32),
                             upload(sorted_dest, {sorted_dest.size()}, DataType::Int64), t_unique, t_offsets, t_runs,
                             upload(order, {order.size()}, DataType::Int64), params);
        expect_equal(host<int32_t>(t_runs), std::vector<int32_t>{2}, "touch run count");

        // reencode_touched_q16_block_kernel for runs {0: rows 0-1}, {2: rows 2-4}.
        for (const auto [block, first, last] : {std::tuple{0u, 0, 2}, std::tuple{2u, 2, 5}}) {
            const float lo_old = expected.bounds[2 * block], hi_old = expected.bounds[2 * block + 1];
            const uint32_t start = block * 256, count = std::min<size_t>(256, n - start);
            std::vector<float> values(count * cells);
            for (uint32_t lane = 0; lane < count; ++lane) {
                const uint32_t p = start + lane;
                for (uint32_t c = 0; c < cells; ++c)
                    values[lane * cells + c] =
                        p < decode_rows ? expected.codes.decode(slot_index(p, c, cells), lo_old, hi_old) : 0.f;
                for (int i = first; i < last; ++i)
                    if (sorted_dest[i] == p)
                        std::copy_n(canonical.begin() + order[i] * cells, cells, values.begin() + lane * cells);
            }
            const float lo = *std::min_element(values.begin(), values.end());
            const float hi = *std::max_element(values.begin(), values.end());
            expected.bounds[2 * block] = lo;
            expected.bounds[2 * block + 1] = hi;
            for (uint32_t lane = 0; lane < count; ++lane)
                for (uint32_t c = 0; c < cells; ++c)
                    q16_encode(expected.codes, slot_index(start + lane, c, cells), values[lane * cells + c], lo, hi);
        }
        if (GetParam() == GpuBackend::Vulkan) {
            // Vulkan follows CUDA's fast-math division (a reciprocal multiply),
            // so a code on a rounding boundary may move by one.
            const auto got = host<uint16_t>(codes);
            std::vector<uint16_t> want(got.size());
            ASSERT_EQ(expected.codes.bytes.size(), want.size() * sizeof(uint16_t));
            std::memcpy(want.data(), expected.codes.bytes.data(), expected.codes.bytes.size());
            for (size_t i = 0; i < got.size(); ++i)
                EXPECT_LE(std::abs(int{got[i]} - int{want[i]}), 1) << "reencoded code " << i;
        } else {
            expect_equal(host<uint8_t>(codes), expected.codes.bytes, "reencoded codes");
        }
        expect_equal(host<float>(bounds), expected.bounds, "reencoded bounds");
    }

    TEST_P(PortableAdamShMorton, RangeDecodePreservesBlockBoundariesAndTail) {
        constexpr size_t n = 513;
        constexpr uint32_t rest = 15, cells = rest * 3;
        const uint32_t slots = lfs::core::sh_float4_slots_for_rest(rest);
        const auto source = pattern(lfs::core::sh_swizzled_float_count(n, rest), .25f, 29);
        const auto f32 = gpu_f(source);
        const auto f16 = f32.to(DataType::Float16);
        auto codes = Tensor::zeros({quant::sh_value_u16_count(n, rest)}, Device::GPU, DataType::Float16);
        auto bounds = Tensor::zeros({quant::n_bounds_for_prims(n) * 2}, Device::GPU);
        sh->encode_q16(f32, codes, bounds, n, rest, 0, 0);
        const Codes q16(16, host<uint8_t>(codes));
        const auto q_bounds = host<float>(bounds);
        // Unaligned rows, float4 padding, 32-row swizzle blocks, 256-row
        // Q16 bounds and the final partial storage block. The CPU reference
        // reads the original layout, independently of the range kernel.
        constexpr std::array<std::pair<uint64_t, size_t>, 6> ranges{{
            {0, 1},
            {42, 91},
            {32 * cells - 1, cells + 2},
            {256 * cells - 2, 92},
            {512 * cells - 1, cells + 1},
            {n * cells - 1, 1},
        }};
        for (const auto storage : {ops::ShStorage::Float32, ops::ShStorage::IeeeFloat16, ops::ShStorage::Q16}) {
            for (const auto [offset, count] : ranges) {
                SCOPED_TRACE(std::to_string(offset) + "+" + std::to_string(count));
                std::vector<float> expected(count + 4, -7.25f);
                auto output = gpu_f(expected);
                const ops::ShRangeParams range{
                    .canonical_float_offset = offset,
                    .float_count = count,
                    .primitives = n,
                    .destination_rest = rest,
                    .layout_rest = rest,
                    .storage = storage,
                };
                const auto& values = storage == ops::ShStorage::Q16 ? codes : storage == ops::ShStorage::IeeeFloat16 ? f16
                                                                                                                     : f32;
                sh->decode_range(values, storage == ops::ShStorage::Q16 ? bounds : Tensor{}, output, range);
                for (size_t i = 0; i < count; ++i) {
                    const uint32_t prim = (offset + i) / cells, cell = (offset + i) % cells;
                    float value = source[slot_index(prim, cell / 4, slots) * 4 + cell % 4];
                    if (storage == ops::ShStorage::IeeeFloat16)
                        value = static_cast<float>(static_cast<_Float16>(value));
                    if (storage == ops::ShStorage::Q16)
                        value = q16.decode(slot_index(prim, cell, cells), q_bounds[2 * (prim / 256)], q_bounds[2 * (prim / 256) + 1]);
                    expected[i] = value;
                }
                expect_equal(host<float>(output), expected, "range and untouched tail");
            }
        }
    }

    TEST_P(PortableAdamShMorton, RowOpsMatchCpu) {
        constexpr size_t n = 70;
        constexpr uint32_t rest = 8;
        const uint32_t slots = lfs::core::sh_float4_slots_for_rest(rest);
        const uint32_t floats_per_row = rest * 3;
        const auto src = pattern(lfs::core::sh_swizzled_float_count(n, rest), 0.25f, 3);
        const Tensor src_gpu = gpu_f(src);
        const std::vector<int32_t> idx32{1, 65, 2, 40};
        const std::vector<int64_t> idx64{1, 65, 2, 40};
        const Tensor i32 = upload(idx32, {idx32.size()}, DataType::Int32);
        const Tensor i64 = upload(idx64, {idx64.size()}, DataType::Int64);
        const size_t k = idx32.size();
        std::vector<float> canonical_all(n * floats_per_row);
        for (uint32_t p = 0; p < n; ++p)
            for (uint32_t f = 0; f < floats_per_row; ++f)
                canonical_all[p * floats_per_row + f] = src[slot_index(p, f / 4, slots) * 4 + f % 4];

        // decode_range of canonical floats [5, 55) from float, half and Q16 storage.
        const ops::ShRangeParams range{.canonical_float_offset = 5, .float_count = 50, .primitives = n, .destination_rest = rest, .layout_rest = rest, .storage = ops::ShStorage::Float32};
        const std::vector<float> expected_range(canonical_all.begin() + 5, canonical_all.begin() + 55);
        Tensor out = Tensor::zeros({50}, Device::GPU);
        sh->decode_range(src_gpu, Tensor{}, out, range);
        expect_equal(host<float>(out), expected_range, "range f32");
        auto half_range = range;
        half_range.storage = ops::ShStorage::IeeeFloat16;
        sh->decode_range(src_gpu.to(DataType::Float16), Tensor{}, out, half_range);
        std::vector<float> expected_half(expected_range);
        for (float& v : expected_half)
            v = static_cast<float>(static_cast<_Float16>(v));
        expect_equal(host<float>(out), expected_half, "range f16");
        Tensor codes = Tensor::zeros({quant::sh_value_u16_count(n, rest)}, Device::GPU, DataType::Float16);
        Tensor bounds = Tensor::zeros({quant::n_bounds_for_prims(n) * 2}, Device::GPU);
        sh->encode_q16(src_gpu, codes, bounds, n, rest, 0, 0);
        const Codes q(16, host<uint8_t>(codes));
        const auto q_bounds = host<float>(bounds);
        std::vector<float> q_range(50);
        for (size_t i = 0; i < 50; ++i) {
            const uint32_t p = (5 + i) / floats_per_row, cell = (5 + i) % floats_per_row;
            q_range[i] = q.decode(slot_index(p, cell, floats_per_row), q_bounds[2 * (p / 256)],
                                  q_bounds[2 * (p / 256) + 1]);
        }
        auto q_params = range;
        q_params.storage = ops::ShStorage::Q16;
        sh->decode_range(codes, bounds, out, q_params);
        expect_equal(host<float>(out), q_range, "range q16");
        auto too_long = range;
        too_long.float_count = n * floats_per_row;
        EXPECT_THROW(sh->decode_range(src_gpu, Tensor{}, out, too_long), std::out_of_range);

        Tensor zeroed = src_gpu.clone();
        sh->zero_rows(zeroed, i32, rest);
        auto expected = src;
        for (const int p : idx32)
            for (uint32_t s = 0; s < slots; ++s)
                std::fill_n(expected.begin() + slot_index(p, s, slots) * 4, 4, 0.f);
        expect_equal(host<float>(zeroed), expected, "zero_rows");

        // gather_swizzled into rows [3, 3 + k): floats with both index widths, uchar4 cells.
        const ops::ShRowsParams rows{.source_rows = n, .count = k, .destination_offset = 3, .source_rest = rest, .destination_rest = rest};
        std::vector<uint8_t> src_bytes(src.size());
        for (size_t i = 0; i < src_bytes.size(); ++i)
            src_bytes[i] = static_cast<uint8_t>(i * 13 + 7);
        std::vector<float> gathered(lfs::core::sh_swizzled_float_count(k + 3, rest), 0.f);
        std::vector<uint8_t> gathered_bytes(gathered.size(), 0);
        for (size_t i = 0; i < k; ++i)
            for (uint32_t s = 0; s < slots; ++s)
                for (uint32_t c = 0; c < 4; ++c) {
                    const size_t from = slot_index(idx32[i], s, slots) * 4 + c;
                    const size_t to = slot_index(3 + i, s, slots) * 4 + c;
                    gathered[to] = src[from];
                    gathered_bytes[to] = src_bytes[from];
                }
        for (const Tensor* indices : {&i32, &i64}) {
            Tensor destination = Tensor::zeros({gathered.size()}, Device::GPU);
            sh->gather_swizzled(src_gpu, *indices, destination, rows);
            expect_equal(host<float>(destination), gathered, "gather_swizzled");
        }
        Tensor bytes_destination = Tensor::zeros({gathered.size()}, Device::GPU, DataType::UInt8);
        sh->gather_swizzled(upload(src_bytes, {src_bytes.size()}, DataType::UInt8), i32, bytes_destination, rows);
        expect_equal(host<uint8_t>(bytes_destination), gathered_bytes, "gather_swizzled bytes");

        // gather_canonical at rest 3 from the rest-8 layout
        const ops::ShRowsParams to_canonical{.source_rows = n, .count = k, .source_rest = rest, .destination_rest = 3};
        Tensor canonical = Tensor::zeros({k, 3, 3}, Device::GPU);
        sh->gather_canonical(src_gpu, i64, canonical, to_canonical);
        std::vector<float> expected_canonical(k * 9);
        for (size_t i = 0; i < k; ++i)
            std::copy_n(canonical_all.begin() + idx32[i] * floats_per_row, 9, expected_canonical.begin() + i * 9);
        expect_equal(host<float>(canonical), expected_canonical, "gather_canonical");

        // append and scatter rest-3 rows into the rest-8 layout; the tail writes zero.
        const ops::ShRowsParams append{.source_rows = k, .count = k, .destination_offset = 30, .source_rest = 3, .destination_rest = rest};
        Tensor appended = src_gpu.clone();
        sh->append_canonical(canonical, appended, append);
        Tensor scattered = src_gpu.clone();
        sh->scatter_canonical(canonical, i32, scattered, append);
        auto expected_append = src;
        auto expected_scatter = src;
        for (size_t i = 0; i < k; ++i)
            for (uint32_t f = 0; f < slots * 4; ++f) {
                const float v = f < 9 ? expected_canonical[i * 9 + f] : 0.f;
                expected_append[slot_index(30 + i, f / 4, slots) * 4 + f % 4] = v;
                expected_scatter[slot_index(idx32[i], f / 4, slots) * 4 + f % 4] = v;
            }
        expect_equal(host<float>(appended), expected_append, "append_canonical");
        expect_equal(host<float>(scattered), expected_scatter, "scatter_canonical");

        Tensor storage = Tensor::zeros({11}, Device::GPU, DataType::UInt8);
        sh->fill_bytes(storage, 9, 0xA5);
        std::vector<uint8_t> filled(11, 0);
        std::fill_n(filled.begin(), 9, 0xA5);
        expect_equal(host<uint8_t>(storage), filled, "fill_bytes");

        const std::vector<float> a = pattern(4, 0.2f, 1), b = pattern(3, 0.3f, 2);
        const std::array<Tensor, 2> parts{gpu_f(a), gpu_f(b)};
        Tensor arena = Tensor::zeros({9}, Device::GPU);
        const Tensor result =
            sh->concatenate_into_arena(parts, static_cast<char*>(arena.data_ptr()) + 4, {7}, DataType::Float32,
                                       lfs::core::TensorExecutionTarget::current());
        std::vector<float> joined(a);
        joined.insert(joined.end(), b.begin(), b.end());
        expect_equal(host<float>(result), joined, "concatenated");
        joined.insert(joined.begin(), 0.f);
        joined.push_back(0.f);
        expect_equal(host<float>(arena), joined, "arena");
    }

    // ---- Morton ----

    uint32_t part1by2(uint32_t x) {
        x &= 0x3ffu;
        x = (x ^ (x << 16)) & 0xff0000ffu;
        x = (x ^ (x << 8)) & 0x0300f00fu;
        x = (x ^ (x << 4)) & 0x030c30c3u;
        x = (x ^ (x << 2)) & 0x09249249u;
        return x;
    }

    // launch_morton_permutation: bbox, 10-bit axes, stable sort by code.
    std::vector<int64_t> host_morton(const std::vector<float>& xyz) {
        const size_t n = xyz.size() / 3;
        std::array<float, 3> lo{INFINITY, INFINITY, INFINITY}, hi{-INFINITY, -INFINITY, -INFINITY};
        for (size_t i = 0; i < n; ++i)
            for (int a = 0; a < 3; ++a) {
                lo[a] = std::min(lo[a], xyz[i * 3 + a]);
                hi[a] = std::max(hi[a], xyz[i * 3 + a]);
            }
        std::vector<uint32_t> codes(n);
        for (size_t i = 0; i < n; ++i) {
            std::array<uint32_t, 3> q{};
            for (int a = 0; a < 3; ++a) {
                const float length = hi[a] - lo[a];
                const float mul = length == 0.f ? 0.f : 1024.f / length;
                const float t = (xyz[i * 3 + a] - lo[a]) * mul;
                q[a] = t > 0.f ? std::min(1023u, static_cast<uint32_t>(t)) : 0u;
            }
            codes[i] = (part1by2(q[2]) << 2) + (part1by2(q[1]) << 1) + part1by2(q[0]);
        }
        std::vector<int64_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return codes[a] < codes[b]; });
        return order;
    }

    TEST_P(PortableAdamShMorton, MortonPermutationMatchesCpuStableSort) {
        // The parity fixture (many equal codes, a flat axis) and a cloud longer
        // than one sort threadgroup.
        std::mt19937 rng(12345);
        std::vector<float> fixture(513 * 3);
        for (size_t i = 0; i < 513; ++i) {
            fixture[i * 3] = static_cast<float>(rng() % 17) - 8.f;
            fixture[i * 3 + 1] = static_cast<float>(rng() % 17) - 8.f;
            fixture[i * 3 + 2] = 2.f;
        }
        std::uniform_real_distribution<float> uniform(-3.f, 5.f);
        std::vector<float> cloud(20000 * 3);
        for (float& v : cloud)
            v = uniform(rng);
        for (const auto* xyz : {&fixture, &cloud}) {
            const size_t n = xyz->size() / 3;
            const Tensor permutation = morton->permutation(Tensor::from_vector(*xyz, {n, 3}, Device::GPU));
            ASSERT_EQ(permutation.dtype(), DataType::Int64);
            expect_equal(host<int64_t>(permutation), host_morton(*xyz), "morton permutation " + std::to_string(n));
        }
    }

    // joint_permute_{contiguous,shN}_{bounds,encode}_cu
    void host_permute_joint(const Codes& src, const std::vector<float>& src_bounds, const std::vector<int64_t>& perm,
                            Codes& dst, std::vector<float>& dst_bounds, int n, uint32_t width, bool swizzled) {
        const uint32_t cells = swizzled ? width * 4 : width;
        const auto cell_of = [&](int64_t prim, uint32_t i) -> size_t {
            return swizzled ? size_t{slot_index(static_cast<uint32_t>(prim), i / 4, width)} * 4 + i % 4
                            : static_cast<size_t>(prim) * width + i;
        };
        const auto decode = [&](int64_t source, uint32_t i) {
            return joint_decode_us(src, cell_of(source, i), &src_bounds[4 * (source / 256)]);
        };
        for (int b = 0; b < (n + 255) / 256; ++b) {
            float u_min = 1e30f, u_max = -1e30f, s_min = 1e30f, s_max = -1e30f;
            for (int p = b * 256; p < std::min(n, (b + 1) * 256); ++p) {
                for (uint32_t i = 0; i < cells; ++i) {
                    const auto us = decode(perm[p], i);
                    u_min = std::min(u_min, us[0]);
                    u_max = std::max(u_max, us[0]);
                    s_min = std::min(s_min, us[1]);
                    s_max = std::max(s_max, us[1]);
                }
            }
            const auto nb = joint_bounds(u_min, u_max, s_min, s_max);
            std::copy(nb.begin(), nb.end(), dst_bounds.begin() + 4 * b);
        }
        for (int p = 0; p < n; ++p)
            for (uint32_t i = 0; i < cells; ++i)
                joint_encode(dst, cell_of(p, i), decode(perm[p], i), &dst_bounds[4 * (p / 256)]);
    }

    TEST_P(PortableAdamShMorton, JointPermutationMatchesCpu) {
        constexpr int n = 513;
        const auto perm = shuffled(n, 731);
        const Tensor perm_gpu = upload(perm, {perm.size()}, DataType::Int64);
        std::vector<float> bounds(joint::n_bounds_for_prims(n) * 4);
        for (size_t b = 0; b < bounds.size() / 4; ++b) {
            bounds[b * 4] = -0.3f - b * 0.1f;
            bounds[b * 4 + 1] = 0.5f + b * 0.1f;
            bounds[b * 4 + 2] = 0.01f;
            bounds[b * 4 + 3] = 0.1f + b * 0.01f;
        }
        const Tensor bounds_gpu = gpu_f(bounds);
        std::mt19937 bytes_rng(991);
        for (const auto& [layout, width, bits] :
             {std::tuple{ops::JointLayout::SwizzledSH, 5u, 16}, std::tuple{ops::JointLayout::SwizzledSH, 3u, 8},
              std::tuple{ops::JointLayout::Rows, 4u, 16}}) {
            const bool swizzled = layout == ops::JointLayout::SwizzledSH;
            const size_t cells = swizzled ? lfs::core::sh_swizzled_padded_n(n) * width * 4 : size_t{n} * width;
            std::vector<uint8_t> packed(cells * joint::bytes_per_cell(bits));
            for (auto& byte : packed)
                byte = static_cast<uint8_t>(bytes_rng());
            const Tensor packed_gpu = upload(packed, {packed.size()}, DataType::UInt8);
            Codes expected(bits, std::vector<uint8_t>(packed.size(), 0));
            std::vector<float> expected_bounds(bounds.size(), 0.f);
            host_permute_joint(Codes(bits, packed), bounds, perm, expected, expected_bounds, n, width, swizzled);
            const ops::JointCodecParams codec{layout, n, static_cast<int>(width), bits};
            const std::string name = std::string(swizzled ? "swizzled" : "rows") + std::to_string(bits);

            Tensor permuted = Tensor::zeros({packed.size()}, Device::GPU, DataType::UInt8);
            Tensor permuted_bounds = Tensor::zeros({bounds.size()}, Device::GPU);
            morton->permute_joint(packed_gpu, bounds_gpu, perm_gpu, permuted, permuted_bounds, codec);
            const auto one_shot = host<uint8_t>(permuted);
            // CUDA and the portable backend must produce identical codes. The CPU
            // oracle can straddle a half-code boundary by one float ULP.
            if (lfs::core::gpu_backend_available(GpuBackend::CUDA)) {
                lfs::test::DefaultGpuBackendForTesting cuda_backend(GpuBackend::CUDA);
                auto cuda_packed = upload(packed, {packed.size()}, DataType::UInt8);
                auto cuda_bounds = gpu_f(bounds);
                auto cuda_perm = upload(perm, {perm.size()}, DataType::Int64);
                auto cuda_out = Tensor::zeros({packed.size()}, Device::GPU, DataType::UInt8);
                auto cuda_out_bounds = Tensor::zeros({bounds.size()}, Device::GPU);
                lfs::training::training_ops(GpuBackend::CUDA).morton->permute_joint(cuda_packed, cuda_bounds, cuda_perm, cuda_out, cuda_out_bounds, codec);
                expect_equal(one_shot, host<uint8_t>(cuda_out), name + " CUDA packed");
                expect_equal(host<float>(cuda_out_bounds), expected_bounds, name + " CUDA bounds");
            }
            expect_codes(one_shot, expected, std::nullopt, name + " packed");
            expect_equal(host<float>(permuted_bounds), expected_bounds, name + " bounds");
            if (!swizzled)
                continue;
            // Scratch of two slots and a bit: groups {0, 1}, {2, 3}, {4}. Same
            // bytes as the one-shot permutation.
            Tensor grouped = packed_gpu.clone();
            Tensor grouped_bounds = Tensor::zeros({bounds.size()}, Device::GPU);
            const size_t slot_bytes = cells / width * joint::bytes_per_cell(bits);
            Tensor scratch = Tensor::empty({slot_bytes * 2 + 7}, Device::GPU, DataType::UInt8);
            morton->permute_joint_grouped(grouped, bounds_gpu, perm_gpu, grouped_bounds, scratch, codec);
            expect_equal(host<uint8_t>(grouped), one_shot, name + " grouped packed");
            expect_equal(host<float>(grouped_bounds), expected_bounds, name + " grouped bounds");
        }
    }

    lfs::core::SplatData make_degree1_splat(size_t count) {
        auto make = [](lfs::core::TensorShape shape, float scale, int seed) {
            return Tensor::from_vector(pattern(shape.elements(), scale, seed), shape, Device::GPU);
        };
        return lfs::core::SplatData(1, make({count, 3}, 1.f, 1), make({count, 1, 3}, 0.5f, 2),
                                    make({count, 3, 3}, 0.2f, 3), make({count, 3}, 1.f, 4), make({count, 4}, 1.f, 5),
                                    make({count, 1}, 1.f, 6), 1.f);
    }

    TEST_P(PortableAdamShMorton, ShPermutationsMatchCpu) {
        constexpr size_t n = 300;
        constexpr uint32_t rest = 3;
        const uint32_t slots = lfs::core::sh_float4_slots_for_rest(rest);
        const uint32_t cells = quant::n_value_cells_per_prim(rest);
        const auto perm = shuffled(n, 17);
        const Tensor perm_gpu = upload(perm, {n}, DataType::Int64);
        const auto target = lfs::core::TensorExecutionTarget::current();
        const lfs::training::IdleArenaScratch arena(8u << 20, 0, target);
        const auto gather_slots = [&](const std::vector<float>& source) {
            std::vector<float> out(source.size(), 0.f);
            for (uint32_t p = 0; p < n; ++p)
                for (uint32_t k = 0; k < slots; ++k)
                    std::copy_n(source.begin() + slot_index(perm[p], k, slots) * 4, 4,
                                out.begin() + slot_index(p, k, slots) * 4);
            return out;
        };

        auto fp32 = make_degree1_splat(n);
        const auto before = host<float>(fp32.shN());
        morton->permute_sh_fp32(fp32, perm_gpu, target, arena);
        expect_equal(host<float>(fp32.shN()), gather_slots(before), "sh fp32");

        // encode_u16_gathered_block_kernel
        lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(true);
        auto q16 = make_degree1_splat(n);
        const bool quantized = lfs::training::sh_value::apply_shN_value_quant(q16);
        lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
        ASSERT_TRUE(quantized);
        const Codes codes(16, host<uint8_t>(q16.shN()));
        const auto bounds = host<float>(q16.shN_value_bounds());
        morton->permute_sh_q16(q16, perm_gpu, target, arena);
        std::vector<float> values(n * cells);
        for (uint32_t p = 0; p < n; ++p) {
            const auto s = static_cast<uint32_t>(perm[p]);
            for (uint32_t c = 0; c < cells; ++c)
                values[p * cells + c] =
                    codes.decode(slot_index(s, c, cells), bounds[2 * (s / 256)], bounds[2 * (s / 256) + 1]);
        }
        Codes expected_codes = codes;
        auto expected_bounds = bounds;
        for (size_t b = 0; b < quant::n_bounds_for_prims(n); ++b) {
            const auto first = values.begin() + b * 256 * cells;
            const auto last = values.begin() + std::min(n, (b + 1) * 256) * cells;
            const float lo = *std::min_element(first, last), hi = *std::max_element(first, last);
            expected_bounds[2 * b] = lo;
            expected_bounds[2 * b + 1] = hi;
            for (uint32_t p = b * 256; p < std::min(n, (b + 1) * 256); ++p)
                for (uint32_t c = 0; c < cells; ++c)
                    q16_encode(expected_codes, slot_index(p, c, cells), values[p * cells + c], lo, hi);
        }
        expect_equal(host<uint8_t>(q16.shN()), expected_codes.bytes, "sh q16 codes");
        expect_equal(host<float>(q16.shN_value_bounds()), expected_bounds, "sh q16 bounds");

        const auto gradient = pattern(lfs::core::sh_swizzled_float_count(n, rest), 0.2f, 3);
        Tensor gathered = Tensor::zeros({gradient.size()}, Device::GPU);
        morton->gather_gradient(gpu_f(gradient), perm_gpu, gathered, rest, target);
        expect_equal(host<float>(gathered), gather_slots(gradient), "gather_gradient");

        Tensor live = gpu_f(pattern(16, 0.1f, 1));
        const Tensor source = gpu_f(pattern(16, 0.2f, 2));
        morton->copy_back(live, source.data_ptr(), 10 * sizeof(float), target);
        auto expected_live = pattern(16, 0.1f, 1);
        const auto source_values = pattern(16, 0.2f, 2);
        std::copy_n(source_values.begin(), 10, expected_live.begin());
        expect_equal(host<float>(live), expected_live, "copy_back");
    }

    // 3DGUT materializes the shN gradient only once SH bands activate; MCMC growth after
    // that must extend every materialized gradient with its moments, whether the growth
    // fits the reserved capacity or not, or zero_grad slices past the gradient's end.
    TEST_P(PortableAdamShMorton, GatherGrowthExtendsMaterializedGradients) {
        using lfs::training::ParamType;
        constexpr size_t n = 300;
        constexpr uint32_t rest = 3;
        for (const size_t capacity : {size_t{512}, n}) {
            SCOPED_TRACE(capacity);
            auto splat = make_degree1_splat(n);
            for (auto* param : {&splat.means(), &splat.sh0(), &splat.scaling_raw(),
                                &splat.rotation_raw(), &splat.opacity_raw()})
                param->reserve(2 * n);
            lfs::training::AdamConfig cfg;
            cfg.initial_capacity = capacity;
            lfs::training::AdamOptimizer optimizer(splat, cfg);
            optimizer.allocate_gradients(capacity);
            const std::array types{ParamType::Means, ParamType::Sh0, ParamType::ShN,
                                   ParamType::Rotation, ParamType::Opacity, ParamType::Scaling};
            for (const auto type : types)
                optimizer.get_grad(type);
            ASSERT_EQ(optimizer.get_grad(ParamType::ShN).numel(), lfs::core::sh_swizzled_float_count(n, rest));

            constexpr size_t n_new = 40;
            std::vector<int64_t> sampled(n_new);
            std::iota(sampled.begin(), sampled.end(), int64_t{7});
            const Tensor indices = upload(sampled, {n_new}, DataType::Int64);
            for (const auto type : types)
                optimizer.add_new_params_gather(type, indices);

            EXPECT_EQ(optimizer.get_grad(ParamType::ShN).numel(),
                      lfs::core::sh_swizzled_float_count(n + n_new, rest));
            EXPECT_EQ(optimizer.get_grad(ParamType::Means).shape()[0], n + n_new);
            EXPECT_NO_THROW(optimizer.zero_grad(0));
        }
    }

    TEST(MetalTrainingLifetime, CachedKernelsSurviveBackendRestart) {
        if (!lfs::core::gpu_backend_available(GpuBackend::Metal))
            GTEST_SKIP() << "Metal device unavailable";
        const lfs::core::GpuBackendScope scope(GpuBackend::Metal);
        const auto* sh = lfs::training::training_ops(GpuBackend::Metal).sh;
        ASSERT_NE(sh, nullptr);
        for (const uint8_t value : {uint8_t{0xA5}, uint8_t{0x3C}, uint8_t{0x71}}) {
            SCOPED_TRACE(value);
            {
                auto output = Tensor::zeros({257}, Device::GPU, DataType::UInt8);
                sh->fill_bytes(output, 253, value);
                std::vector<uint8_t> expected(257, 0);
                std::fill_n(expected.begin(), 253, value);
                expect_equal(host<uint8_t>(output), expected, "fill after backend restart");
            }
            ASSERT_TRUE(lfs::core::shutdown_gpu_backend(GpuBackend::Metal));
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, PortableAdamShMorton, testing::Values(GpuBackend::Metal, GpuBackend::Vulkan),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });

} // namespace
