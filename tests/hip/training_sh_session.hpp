/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "../../src/training/hip/session.hpp"
#include "sh_oracle.hpp"
#include "training_fixture.hpp"
#include <cstdio>
#include <limits>

namespace hip_sh_test {
    template <class Fixture>
    void session(const std::filesystem::path& module, const Fixture& fixture) {
        using lfs::training::hip::Session;
        using lfs::training::hip::TrainingError;
        using namespace hip_fixture;
        auto require = [](bool value, const char* message) {
            if (!value)
                throw std::runtime_error(message);
        };
        const LfsHipTrainConfig config{sizeof(config), 0, n, 64 * 1024 * 1024};
        std::vector<float> rest(n * 45), image(channels), baseline(channels);
        for (size_t i = 0; i < rest.size(); ++i)
            rest[i] = 0.08f * std::sin(float(i * 13 + 7));
        LfsHipTrainSh sh{sizeof(sh), 3, 0, 0.005f, rest.data(), rest.size()};
        auto colors = [&](const Fixture& model, const std::vector<float>& coefficients, unsigned degree) {
            std::vector<float> result(n * 3);
            for (size_t g = 0; g < n; ++g) {
                const auto b = hip_sh_oracle::basis(model.values[0][g * 3], model.values[0][g * 3 + 1], model.values[0][g * 3 + 2]);
                for (unsigned c = 0; c < 3; ++c) {
                    double value = 0.5 + b[0] * model.values[3][g * 3 + c];
                    for (unsigned k = 1; k < (degree + 1) * (degree + 1); ++k)
                        value += b[k] * coefficients[g * 45 + (k - 1) * 3 + c];
                    result[g * 3 + c] = float(std::max(0.0, value));
                }
            }
            return result;
        };
        auto read = [&](Session& object) {
            std::vector<float> value(rest.size());
            object.read_parameter(5, value);
            return value;
        };
        auto reject = [&](auto&& operation, int expected) {
            bool rejected = false;
            try {
                operation();
            } catch (const TrainingError& error) {
                rejected = error.status == expected && error.what()[0];
            }
            require(rejected, "Wrong SH configuration failure or missing diagnostic");
        };
        Session object(module, config, fixture.model());
        object.set_frame(fixture.frame());
        object.render(baseline);
        reject([&] { object.set_sh_degree(1, 0.005f); }, LFS_HIP_TRAIN_INVALID);
        for (unsigned bad = 0; bad < 6; ++bad) {
            auto invalid = sh;
            switch (bad) {
            case 0: invalid.max_degree = 4; break;
            case 1: invalid.active_degree = 4; break;
            case 2: --invalid.float_count; break;
            case 3: invalid.rest_learning_rate = -1; break;
            case 4: invalid.rest_learning_rate = std::numeric_limits<float>::quiet_NaN(); break;
            case 5: invalid.struct_bytes = 0; break;
            }
            reject([&] { object.set_sh(invalid); }, LFS_HIP_TRAIN_INVALID);
        }
        auto invalid_rest = rest;
        invalid_rest[5] = std::numeric_limits<float>::quiet_NaN();
        auto invalid_sh = sh;
        invalid_sh.canonical_rest = invalid_rest.data();
        reject([&] { object.set_sh(invalid_sh); }, LFS_HIP_TRAIN_INVALID);
        const auto live = object.info().allocated_bytes;
        object.set_budget(live + ((n + 31) / 32 * 32) * 48 * 4);
        reject([&] { object.set_sh(sh); }, LFS_HIP_TRAIN_RESOURCE_EXHAUSTED);
        require(object.info().allocated_bytes == live && object.info().completed_steps == 0 && !object.info().poisoned,
                "Partial SH allocation leaked or poisoned session");
        object.set_budget(config.allocation_budget_bytes);
        object.set_sh(sh);
        require(read(object) == rest, "Canonical SH-rest swizzle round trip failed");
        object.render(image);
        require(image == baseline, "Configured but inactive SH changed SH0 render");
        const LfsHipTrainSh disabled{sizeof(disabled), 0, 0, 0, nullptr, 0};
        object.set_sh(disabled);
        require(object.info().allocated_bytes == live, "Clearing SH did not release its storage");
        reject([&] { (void)read(object); }, LFS_HIP_TRAIN_INVALID);
        object.set_sh(sh);
        require(read(object) == rest, "SH reconfiguration before training failed");
        double worst_render = 0;
        for (unsigned degree = 1; degree <= 3; ++degree) {
            object.set_sh_degree(degree, 0.005f);
            object.render(image);
            const auto expected = cpu_render(fixture.values[0], fixture.values[1], fixture.values[2], colors(fixture, rest, degree), fixture.values[4]);
            for (size_t i = 0; i < image.size(); ++i) {
                require(std::isfinite(image[i]), "Nonfinite SH session render");
                worst_render = std::max(worst_render, std::abs(double(image[i]) - expected[i]));
            }
        }
        require(worst_render < 3e-6, "SH session render differs from CPU oracle");
        auto translated = fixture;
        const float translation[3]{0.125f, -0.25f, 0.375f};
        for (size_t g = 0; g < n; ++g)
            for (unsigned c = 0; c < 3; ++c)
                translated.values[0][g * 3 + c] += translation[c];
        auto shifted_frame = fixture.frame();
        for (unsigned c = 0; c < 3; ++c)
            shifted_frame.world_to_camera[c * 4 + 3] = translation[c];
        object.set_frame(shifted_frame);
        object.render(image);
        const auto shifted_expected = cpu_render(translated.values[0], translated.values[1], translated.values[2], colors(translated, rest, 3), translated.values[4]);
        for (size_t i = 0; i < image.size(); ++i)
            require(std::abs(image[i] - shifted_expected[i]) < 3e-6, "SH camera-center transform differs from CPU oracle");
        object.set_frame(fixture.frame());
        const LfsHipTrainRates frozen{0, 0, 0, 0, 0};
        object.set_sh_degree(1, 0.005f);
        object.step(frozen);
        const auto changed = read(object);
        bool updated = false;
        for (size_t g = 0; g < n; ++g)
            for (size_t i = 0; i < 45; ++i) {
                if (i >= 9)
                    require(changed[g * 45 + i] == rest[g * 45 + i], "Inactive SH bases changed");
                else
                    updated |= changed[g * 45 + i] != rest[g * 45 + i];
            }
        require(updated, "Active SH-rest optimizer did not update");
        object.set_sh_degree(0, 0.005f);
        object.step(frozen);
        require(read(object) == changed, "Inactive SH-rest drifted with live moments");
        object.set_sh_degree(1, 0);
        object.step(frozen);
        require(read(object) == changed, "Zero SH-rest rate changed values");
        reject([&] { object.set_sh(sh); }, LFS_HIP_TRAIN_INVALID);
        object.close();
        std::printf("PASS SH session: configuration, budget cleanup, swizzle, active/inactive states, CPU render max %.9g\n", worst_render);

        // Zero SH-rest must preserve the existing first SH0 optimizer update.
        auto perturbed = fixture;
        perturbed.perturb();
        Session old(module, config, perturbed.model()), extended(module, config, perturbed.model());
        old.set_frame(perturbed.frame());
        extended.set_frame(perturbed.frame());
        std::vector<float> zero(rest.size());
        sh = {sizeof(sh), 3, 3, 0, zero.data(), zero.size()};
        extended.set_sh(sh);
        const LfsHipTrainRates rates{0.002f, 0.002f, 0.003f, 0.015f / sh_c0, 0.01f};
        old.step(rates);
        extended.step(rates);
        double first_delta = 0;
        for (unsigned family = 0; family < 5; ++family) {
            std::vector<float> a(fixture.values[family].size()), b(a.size());
            old.read_parameter(family, a);
            extended.read_parameter(family, b);
            for (size_t i = 0; i < a.size(); ++i)
                first_delta = std::max(first_delta, std::abs(double(a[i]) - b[i]));
        }
        require(first_delta < 1e-5 && read(extended) == zero, "SH0 first update changed or SH-rest zero rate failed");
        const auto late_base = old.info().allocated_bytes;
        old.set_sh(sh); // Initial SH allocation after an existing SH0 update.
        require(old.info().completed_steps == 1 && old.info().allocated_bytes > late_base && read(old) == zero,
                "Deferred initial SH allocation failed or reset the optimizer step");
        old.step(frozen);
        require(old.info().completed_steps == 2 && !old.info().poisoned, "Deferred SH activation failed");
        old.close();
        extended.close();
        std::printf("PASS SH0 first-update consistency max %.9g\n", first_delta);

        for (unsigned variant = 1; variant <= 4; ++variant) {
            const unsigned degree = std::min(variant, 3u);
            const bool photometric = variant == 4;
            const auto target = cpu_render(fixture.values[0], fixture.values[1], fixture.values[2], colors(fixture, rest, degree), fixture.values[4]);
            auto frame = fixture.frame();
            frame.target_chw = target.data();
            Session trained(module, config, fixture.model());
            trained.set_frame(frame);
            sh = {sizeof(sh), 3, degree, 0.005f, zero.data(), zero.size()};
            trained.set_sh(sh);
            if (photometric) {
                const LfsHipTrainLoss loss{sizeof(loss), LFS_HIP_LOSS_L1_SSIM, 0.2f, 1};
                trained.set_loss(loss);
            }
            const float initial = trained.loss();
            for (unsigned i = 0; i < 200; ++i) {
                if (photometric)
                    trained.set_sh_degree(degree, 0.002f * std::max(0.05f, std::pow(0.98f, float(i))));
                trained.step(frozen);
            }
            const float final = trained.loss();
            require(std::isfinite(final) && final < initial * 0.1f && !trained.info().poisoned,
                    "SH-rest-only training did not converge");
            const auto parameters = read(trained);
            for (size_t g = 0; g < n; ++g)
                for (size_t i = ((degree + 1) * (degree + 1) - 1) * 3; i < 45; ++i)
                    require(parameters[g * 45 + i] == 0, "Inactive higher SH coefficient changed during training");
            std::printf("PASS SH%u%s: 200 steps loss %.9g -> %.9g\n", degree, photometric ? " L1/SSIM" : " MSE", initial, final);
            if (degree == 3) {
                trained.set_sh_degree(1, 0.005f);
                trained.step(frozen);
                const auto paused = read(trained);
                for (size_t g = 0; g < n; ++g)
                    for (size_t i = 9; i < 45; ++i)
                        require(paused[g * 45 + i] == parameters[g * 45 + i], "Deactivated higher SH drifted with existing moments");
                trained.set_sh_degree(3, 0.005f);
                trained.step(frozen);
                require(!trained.info().poisoned, "SH degree reactivation failed");
            }
            trained.close();
        }
        auto clamped = fixture;
        std::fill(clamped.values[3].begin(), clamped.values[3].end(), -4.f);
        Session black(module, config, clamped.model());
        black.set_frame(clamped.frame());
        sh = {sizeof(sh), 3, 3, 0.005f, zero.data(), zero.size()};
        black.set_sh(sh);
        black.step(rates);
        for (unsigned family = 0; family < 5; ++family) {
            std::vector<float> value(clamped.values[family].size());
            black.read_parameter(family, value);
            require(value == clamped.values[family], "Clamped SH colors propagated a model gradient");
        }
        require(read(black) == zero, "Clamped SH colors propagated a rest gradient");
        black.close();
        auto at_camera = fixture;
        at_camera.values[0][0] = at_camera.values[0][1] = at_camera.values[0][2] = 0;
        at_camera.values[4][0] = -100;
        Session coincident(module, config, at_camera.model());
        coincident.set_frame(at_camera.frame());
        coincident.set_sh(sh);
        coincident.step(frozen);
        coincident.render(image);
        for (float value : image)
            require(std::isfinite(value), "Culled Gaussian at camera caused nonfinite SH rendering");
        const auto skipped = read(coincident);
        for (size_t i = 0; i < 45; ++i)
            require(skipped[i] == 0, "Culled zero-direction Gaussian received SH gradient");
        coincident.close();
        std::puts("PASS SH camera transform, clamp derivative and culled zero direction");
    }

    template <class Fixture>
    void large_session(const std::filesystem::path& module, const Fixture& fixture) {
        using lfs::training::hip::Session;
        using lfs::training::hip::TrainingError;
        auto require = [](bool value, const char* message) {
            if (!value)
                throw std::runtime_error(message);
        };
        constexpr size_t count = 1000000;
        std::array<std::vector<float>, 5> values{std::vector<float>(count * 3), std::vector<float>(count * 4),
                                                 std::vector<float>(count * 3, -1.5f), std::vector<float>(count * 3), std::vector<float>(count, -4)};
        for (size_t g = 0; g < count; ++g) {
            values[0][g * 3 + 2] = 2000;
            values[1][g * 4] = 1;
        }
        for (unsigned family = 0; family < 5; ++family)
            std::copy(fixture.values[family].begin(), fixture.values[family].end(), values[family].begin());
        const LfsHipTrainModel model{values[0].data(), values[1].data(), values[2].data(), values[3].data(), values[4].data()};
        const LfsHipTrainConfig config{sizeof(config), 0, count, 512 * 1024 * 1024};
        Session object(module, config, model);
        const auto base = object.info().allocated_bytes;
        {
            std::vector<float> rest(count * 45);
            const LfsHipTrainSh sh{sizeof(sh), 3, 0, 0.001f, rest.data(), rest.size()};
            bool rejected = false;
            try {
                object.set_sh(sh);
            } catch (const TrainingError& error) {
                rejected = error.status == LFS_HIP_TRAIN_RESOURCE_EXHAUSTED;
            }
            require(rejected && object.info().allocated_bytes == base && !object.info().poisoned,
                    "Large SH3 budget rejection leaked or poisoned the base model");
        }
        std::vector<float> rest(count * 9);
        const LfsHipTrainSh sh{sizeof(sh), 1, 1, 0.001f, rest.data(), rest.size()};
        object.set_sh(sh);
        object.set_frame(fixture.frame());
        const LfsHipTrainRates frozen{0, 0, 0, 0, 0};
        object.step(frozen);
        const auto info = object.info();
        require(info.completed_steps == 1 && !info.poisoned && info.peak_allocated_bytes <= config.allocation_budget_bytes,
                "Large SH1 model exceeded its allocation budget or failed");
        std::printf("PASS 1M SH model: SH3 rejected at 512 MiB; SH1 step live %llu, peak %llu bytes\n",
                    static_cast<unsigned long long>(info.allocated_bytes), static_cast<unsigned long long>(info.peak_allocated_bytes));
        std::puts("Only 263 visible entries at 35x29; memory/budget validation, not dataset throughput.");
        object.close();
    }
} // namespace hip_sh_test
