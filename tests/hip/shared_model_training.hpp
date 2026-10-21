/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "../../src/training/hip/session.hpp"
#include "../../src/training/hip/splat_storage_layout.hpp"
#include "training_fixture.hpp"
#include <cstring>
#include <functional>

// Called only while Vulkan has released ownership to EXTERNAL and completed.
// inspect performs a Vulkan transfer readback, returning ownership to EXTERNAL.
inline void shared_model_training(const std::filesystem::path& module, const LfsHipTrainSharedModel& shared,
                                  const std::function<void(const LfsHipTrainModelLayout&, lfs::training::hip::Session&)>& inspect,
                                  const LfsHipTrainSharedModel* other_adapter = nullptr, bool fused_photo = false, bool application_layout = false,
                                  const std::function<uint32_t(uint64_t, uint32_t)>& exchange = {}, void* resident_pointer = nullptr) {
    using lfs::training::hip::Session;
    using lfs::training::hip::TrainingError;
    using namespace hip_fixture;
    auto require = [](bool value, const char* message) {
        if (!value)
            throw std::runtime_error(message);
    };
    constexpr size_t count = 1000000;
    const auto private_layout = Session::model_layout(module, count);
    auto layout = private_layout;
    if (application_layout) {
        using Storage = lfs::core::SplatExportableStorage;
        Storage::Control storage;
        storage.block = std::make_shared<lfs::core::ExportableBlock>();
        storage.block->device_ptr = resident_pointer;
        storage.block->reserved_bytes = shared.allocation_bytes;
        storage.capacity = count + 50000;
        storage.generation = 1;
        const size_t regions[] = {Storage::Means, Storage::Scaling, Storage::Rotation, Storage::Opacity, Storage::Sh0};
        const size_t sizes[] = {3, 3, 4, 1, 3};
        uint64_t cursor = 65536;
        for (size_t i = 0; i < 5; ++i) {
            storage.region_offsets[regions[i]] = cursor;
            storage.region_bytes[regions[i]] = storage.capacity * sizes[i] * sizeof(float);
            cursor += storage.region_bytes[regions[i]];
            cursor = (cursor + 65535) / 65536 * 65536;
        }
        // Use the actual core Control adapter, not a second HIP layout formula.
        layout = lfs::training::hip::splat_storage_layout(storage, count);
    }
    require(layout.required_bytes <= shared.allocation_bytes, "Shared model does not fit allocation");
    for (unsigned i = 0; i < 5; ++i)
        require(layout.offsets_bytes[i] % 256 == 0 &&
                    layout.offsets_bytes[i] + layout.float_counts[i] * 4 <= layout.required_bytes,
                "Invalid shared model layout");
    std::array<std::vector<float>, 5> values{std::vector<float>(count * 3), std::vector<float>(count * 4),
                                             std::vector<float>(count * 3, -1.5f), std::vector<float>(count * 3), std::vector<float>(count, -4)};
    std::vector<float> colors(n * 3);
    for (size_t g = 0; g < count; ++g) {
        values[0][g * 3 + 2] = 2000;
        values[1][g * 4] = 1;
    }
    for (unsigned g = 0; g < n; ++g) {
        values[0][g * 3] = (int(g % 3) - 1) * 0.65f;
        values[0][g * 3 + 1] = (int((g / 3) % 2) * 2 - 1) * 0.45f;
        values[0][g * 3 + 2] = 3 + 0.005f * g;
        values[1][g * 4 + 1] = 0.08f * (g % 6);
        values[1][g * 4 + 2] = 0.03f * (g % 6);
        for (unsigned c = 0; c < 3; ++c) {
            values[2][g * 3 + c] = std::log(0.22f + 0.06f * c + 0.01f * (g % 6));
            colors[g * 3 + c] = 0.2f + float((g * 3 + c * 5) % 7) * 0.1f;
            values[3][g * 3 + c] = (colors[g * 3 + c] - 0.5f) / sh_c0;
        }
        values[4][g] = -3.5f + 0.1f * (g % 6);
    }
    const auto target = cpu_render(values[0], values[1], values[2], colors, values[4]);
    for (unsigned g = 0; g < n; ++g) {
        values[0][g * 3] += 0.07f;
        values[1][g * 4 + 2] += 0.04f;
        values[2][g * 3 + 1] += 0.08f;
        values[4][g] -= 0.2f;
    }
    const LfsHipTrainModel model{values[0].data(), values[1].data(), values[2].data(), values[3].data(), values[4].data()};
    LfsHipTrainConfig config{sizeof(config), 0, count, 128 * 1024 * 1024};
    auto reject = [&](const LfsHipTrainSharedModel& descriptor, int expected) {
        bool rejected = false;
        try {
            if (application_layout)
                Session invalid(module, config, model, descriptor, layout);
            else
                Session invalid(module, config, model, descriptor);
        } catch (const TrainingError& error) {
            rejected = error.status == expected && error.what()[0];
        }
        require(rejected, "Incorrect shared model failure or silent private allocation fallback");
    };
    auto invalid = shared;
    invalid.allocation_bytes = layout.required_bytes - 1;
    reject(invalid, LFS_HIP_TRAIN_INVALID);
    if (other_adapter)
        reject(*other_adapter, LFS_HIP_TRAIN_INVALID);
    invalid = shared;
    std::memset(invalid.device_luid, 0, sizeof(invalid.device_luid));
    reject(invalid, LFS_HIP_TRAIN_INVALID);
    reject(shared, LFS_HIP_TRAIN_RESOURCE_EXHAUSTED);
    config.allocation_budget_bytes = 512 * 1024 * 1024;
    if (application_layout) {
        auto reject_layout = [&](LfsHipTrainModelLayout bad) {
            bool rejected = false;
            try {
                Session invalid(module, config, model, shared, bad);
            } catch (const TrainingError& error) {
                rejected = error.status == LFS_HIP_TRAIN_INVALID;
            }
            require(rejected, "Malformed external layout was accepted");
        };
        auto bad = layout;
        bad.struct_bytes = 0;
        reject_layout(bad);
        bad = layout;
        bad.offsets_bytes[1] = layout.offsets_bytes[0];
        reject_layout(bad);
        bad = layout;
        bad.offsets_bytes[2] += 4;
        reject_layout(bad);
        bad = layout;
        bad.float_counts[3] += 1;
        reject_layout(bad);
        bad = layout;
        bad.offsets_bytes[4] = UINT64_MAX - 255;
        reject_layout(bad);
        bad = layout;
        bad.required_bytes = shared.allocation_bytes + 4;
        reject_layout(bad);
        bad = layout;
        bad.required_bytes = layout.offsets_bytes[3];
        reject_layout(bad);
    }
    if (application_layout) {
        Session initialize(module, config, model, shared, layout);
        inspect(layout, initialize);
        initialize.close();
        require(bool(exchange), "Resident model mutation callback missing");
        auto reject_resident = [&](unsigned family, uint32_t bits, unsigned components = 1) {
            std::array<uint32_t, 4> old{};
            for (unsigned c = 0; c < components; ++c)
                old[c] = exchange(layout.offsets_bytes[family] + c * 4, bits);
            bool rejected = false;
            try {
                Session invalid(module, config, shared, layout);
            } catch (const TrainingError& error) {
                rejected = error.status == LFS_HIP_TRAIN_INVALID;
            }
            for (unsigned c = 0; c < components; ++c)
                require(exchange(layout.offsets_bytes[family] + c * 4, old[c]) == bits,
                        "Failed attach changed resident parameters");
            require(rejected, "Invalid GPU-resident parameters were accepted");
        };
        for (unsigned family = 0; family < 5; ++family)
            reject_resident(family, 0x7fc00000u); // NaN in every family.
        reject_resident(1, 0, 4);                 // Zero quaternion norm.
        reject_resident(2, 0x447a0000u);          // exp(1000) overflow.
        reject_resident(2, 0xc47a0000u);          // exp(-1000) underflow.
        reject_resident(4, 0x7f800000u);          // Infinite raw opacity.
        std::puts("PASS: nine invalid resident models rejected without modifying parameters");
    }
    auto owner = application_layout ? std::make_unique<Session>(module, config, shared, layout)
                                    : std::make_unique<Session>(module, config, model, shared);
    auto& session = *owner;
    Session control(module, config, model);
    Session repeat(module, config, model);
    const auto imported_live = session.info().allocated_bytes;
    const auto private_live = control.info().allocated_bytes;
    require(imported_live == private_live - private_layout.required_bytes + shared.allocation_bytes,
            "Shared allocation is not charged exactly once in the session budget");
    LfsHipTrainFrame frame{};
    frame.struct_bytes = sizeof(frame);
    frame.width = width;
    frame.height = height;
    for (unsigned i = 0; i < 4; ++i)
        frame.world_to_camera[i * 5] = 1;
    frame.fx = frame.fy = 32;
    frame.cx = width / 2.0f;
    frame.cy = height / 2.0f;
    frame.near_plane = 0.01f;
    frame.far_plane = 1000;
    frame.target_chw = target.data();
    session.set_frame(frame);
    control.set_frame(frame);
    repeat.set_frame(frame);
    if (fused_photo) {
        const LfsHipTrainLoss loss{sizeof(loss), LFS_HIP_LOSS_L1_SSIM, 0.2f, 1};
        session.set_loss(loss);
        control.set_loss(loss);
        repeat.set_loss(loss);
    }
    std::printf("Shared model loss mode: %s\n", fused_photo ? "L1 + 0.2 SSIM, valid padding" : "MSE");
    const LfsHipTrainRates rates{0.002f, 0.002f, 0.003f, 0.015f / sh_c0, 0.01f};
    inspect(layout, session);
    std::vector<float> image(channels), reference(channels);
    std::vector<float> repeated(channels);
    auto loss = [&] {
        double sum = 0;
        for (unsigned i = 0; i < channels; ++i) {
            require(std::isfinite(image[i]), "Nonfinite shared model render");
            const double delta = double(image[i]) - target[i];
            sum += delta * delta;
        }
        return sum / channels;
    };
    session.render(image);
    const double initial = loss();
    for (unsigned i = 0; i < 100; ++i) {
        auto step_rates = rates;
        if (fused_photo) {
            // The fixed rates of the historical MSE fixture oscillate around
            // L1's cusp. Exercise the per-step scheduler contract and finish at
            // a small rate, identically for shared and both private controls.
            const float scale = std::max(0.02f, std::pow(0.95f, float(i)));
            step_rates = {rates.means * scale, rates.quaternions * scale, rates.log_scales * scale,
                          rates.sh0 * scale, rates.raw_opacities * scale};
        }
        session.step(step_rates);
        control.step(step_rates);
        repeat.step(step_rates);
        if (i == 0) {
            float worst = 0;
            for (unsigned family = 0; family < 5; ++family) {
                std::vector<float> a(layout.float_counts[family]), b(a.size());
                session.read_parameter(family, a);
                control.read_parameter(family, b);
                for (size_t j = 0; j < a.size(); ++j) {
                    require(std::isfinite(a[j]) && std::isfinite(b[j]), "Nonfinite first optimizer update");
                    worst = std::max(worst, std::abs(a[j] - b[j]));
                }
            }
            std::printf("First shared/private optimizer update max parameter delta %.9g\n", worst);
            require(worst < 1e-5f, "Shared storage changed the first optimizer update");
        }
        if ((i + 1) % 20 == 0)
            inspect(layout, session);
    }
    session.render(image);
    control.render(reference);
    repeat.render(repeated);
    float worst_parameter = 0, worst_image = 0;
    float repeat_parameter = 0, repeat_image = 0;
    for (unsigned family = 0; family < 5; ++family) {
        std::vector<float> a(layout.float_counts[family]), b(a.size()), c(a.size());
        session.read_parameter(family, a);
        control.read_parameter(family, b);
        repeat.read_parameter(family, c);
        for (size_t i = 0; i < a.size(); ++i) {
            require(std::isfinite(a[i]) && std::isfinite(b[i]) && std::isfinite(c[i]), "Nonfinite shared/private parameter");
            worst_parameter = std::max(worst_parameter, std::abs(a[i] - b[i]));
            repeat_parameter = std::max(repeat_parameter, std::abs(b[i] - c[i]));
        }
    }
    double reference_loss = 0, repeated_loss = 0, image_mse = 0, repeated_mse = 0;
    for (unsigned i = 0; i < channels; ++i) {
        require(std::isfinite(reference[i]) && std::isfinite(repeated[i]), "Nonfinite private control render");
        worst_image = std::max(worst_image, std::abs(image[i] - reference[i]));
        repeat_image = std::max(repeat_image, std::abs(reference[i] - repeated[i]));
        const double a = double(reference[i]) - target[i], b = double(repeated[i]) - target[i];
        const double c = double(image[i]) - reference[i], d = double(reference[i]) - repeated[i];
        reference_loss += a * a / channels;
        repeated_loss += b * b / channels;
        image_mse += c * c / channels;
        repeated_mse += d * d / channels;
    }
    const auto info = session.info();
    std::printf("Shared/private 1M model: 100 steps, parameter max delta %.9g, image delta %.9g, MSE %.9g -> %.9g\n",
                worst_parameter, worst_image, initial, loss());
    std::printf("Private/private control: parameter delta %.9g, image delta %.9g; target MSE %.9g / %.9g; inter-image MSE shared/private %.9g, private/private %.9g\n",
                repeat_parameter, repeat_image, reference_loss, repeated_loss, image_mse, repeated_mse);
    // Atomic gradient accumulation is not bitwise deterministic, including
    // private/private controls. Keep explicit hard caps: max RGB < one 8-bit
    // level, RGB RMSE < 1e-4, raw parameter max < 1e-3, target MSE within 1% of
    // both independent controls. Vulkan vs HIP alias readback remains byte exact.
    const double minimum_loss = std::min(reference_loss, repeated_loss);
    const double maximum_loss = std::max(reference_loss, repeated_loss);
    bool matching_objective = true;
    if (fused_photo) {
        const double a = session.loss(), b = control.loss(), c = repeat.loss();
        std::printf("L1/SSIM objective shared %.9g, private %.9g / %.9g\n", a, b, c);
        matching_objective = a >= std::min(b, c) * 0.99 - 1e-6 && a <= std::max(b, c) * 1.01 + 1e-6;
    }
    // L1/SSIM trajectories include a nonsmooth sign gradient and are compared
    // through their objective and rendered image, not raw-parameter identity.
    // The existing MSE bounds are unchanged. Both modes keep the strict first
    // update and byte-exact Vulkan/HIP alias checks above.
    const bool matching_image = fused_photo ? worst_image < 4.0f / 255 && image_mse < 1.0 / (255 * 255)
                                            : worst_parameter < 1e-3f && worst_image < 1.0f / 255 && image_mse < 1e-8;
    require(matching_image && matching_objective &&
                loss() >= minimum_loss * 0.99 - 1e-9 && loss() <= maximum_loss * 1.01 + 1e-9 && loss() < initial * 0.1,
            "Shared allocation changed training quality compared with private HIP allocation");
    std::printf("Shared session live %llu, peak %llu, imported allocation %llu bytes (charged once)\n",
                static_cast<unsigned long long>(info.allocated_bytes), static_cast<unsigned long long>(info.peak_allocated_bytes),
                static_cast<unsigned long long>(shared.allocation_bytes));
    control.close();
    repeat.close();
    session.close();
    std::puts("Only 263 visible Gaussians at 35x29: storage/quality comparison, not dataset throughput.");
}
