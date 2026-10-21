/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../src/core/tensor/internal/undistort_resample.hpp"
#include "../../src/training/hip/camera_frame.hpp"
#include "../../src/training/hip/prepared_image.hpp"
#include "../../src/training/hip/session.hpp"
#include "../../src/training/hip/splat_storage_layout.hpp"
#include "../../src/training/hip/tensor_model.hpp"
#include "training_session_fixture.hpp"
#include <core/splat_data.hpp>
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
using namespace lfs::core;
using lfs::training::hip::Session;
using lfs::training::hip::TrainingError;
using namespace hip_fixture;
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
int main(int argc, char** argv) try {
    require(argc == 2, "Expected absolute HIP training module path");
    const auto module = std::filesystem::absolute(argv[1]);
    const auto process_backend = configured_gpu_backend();
    {
        GpuBackendScope scope(GpuBackend::HIP);
        // Actual application Camera and prepared-image boundary. Deliberately
        // leave stale Camera image dimensions to catch incorrect resize scaling.
        auto rotation = Tensor::eye(3, Device::CPU);
        auto translation = Tensor::from_vector(std::vector<float>{0.1f, -0.2f, 0.3f}, TensorShape({3}), Device::CPU);
        Camera camera(rotation, translation, 80, 90, 40, 30, {}, {}, CameraModelType::PINHOLE,
                      "test", {}, {}, 80, 60, 0);
        camera.set_image_dimensions(13, 17);
        auto prepared = Tensor::full({3, 30, 40}, 0.25f, Device::CPU);
        lfs::training::hip::CameraFrame camera_frame(camera, prepared);
        auto descriptor = camera_frame.descriptor();
        require(descriptor.width == 40 && descriptor.height == 30 && descriptor.fx == 40 && descriptor.fy == 45 &&
                    descriptor.cx == 20 && descriptor.cy == 15,
                "Camera frame used stale dimensions or incorrect resize intrinsics");
        require(descriptor.world_to_camera[3] == 0.1f && descriptor.world_to_camera[7] == -0.2f &&
                    descriptor.world_to_camera[11] == 0.3f && descriptor.world_to_camera[15] == 1,
                "Camera frame changed world-to-camera convention");
        prepared.fill_(0.75f);
        auto moved_frame = std::move(camera_frame);
        require(moved_frame.descriptor().target_chw[0] == 0.25f, "Loader reuse changed prepared HIP target");
        bool bad_image = false;
        try {
            lfs::training::hip::CameraFrame invalid(camera, Tensor::zeros({30, 40, 3}, Device::CPU));
        } catch (const std::invalid_argument&) { bad_image = true; }
        require(bad_image, "HIP camera frame accepted HWC input");
        auto reject_frame = [&](const Camera& candidate, float near_plane = 0.01f, float far_plane = 1000.f) {
            bool rejected = false;
            try {
                lfs::training::hip::CameraFrame invalid(candidate, prepared, near_plane, far_plane);
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Unsupported camera frame was silently accepted");
        };
        Camera distorted(rotation, translation, 80, 90, 40, 30,
                         Tensor::from_vector(std::vector<float>{0.1f}, {1}, Device::CPU), {},
                         CameraModelType::PINHOLE, "distorted", {}, {}, 80, 60, 2);
        reject_frame(distorted);
        bool unprepared_rejected = false;
        try {
            lfs::training::hip::CameraFrame invalid(distorted, prepared, 0.01f, 1000.f, true);
        } catch (const std::invalid_argument&) { unprepared_rejected = true; }
        require(unprepared_rejected, "Undistorted target accepted without prepared calibration");
        distorted.prepare_undistortion();
        lfs::training::hip::CameraFrame rectified(distorted, prepared, 0.01f, 1000.f, true);
        require(rectified.descriptor().width == 40 && rectified.descriptor().height == 30,
                "Prepared pinhole calibration rejected its undistorted target");
        reject_frame(distorted);
        Camera ortho(rotation, translation, 80, 90, 40, 30, {}, {}, CameraModelType::ORTHO,
                     "ortho", {}, {}, 80, 60, 3);
        reject_frame(ortho);
        reject_frame(camera, 1, 1);
        reject_frame(camera, std::numeric_limits<float>::quiet_NaN());
        for (int lens_case = 0; lens_case < 4; ++lens_case) {
            UndistortParams lens{};
            lens.src_width = 263;
            lens.src_height = 31;
            lens.src_fx = lens.src_fy = 70;
            lens.src_cx = 131.5f;
            lens.src_cy = 15.5f;
            lens.model_type = CameraModelType::PINHOLE;
            lens.num_distortion = lens_case == 0 ? 0 : 5;
            lens.distortion[0] = lens_case == 1 ? 0.1f : lens_case == 2 ? -0.2f
                                                                        : 0.07f;
            if (lens_case == 3) {
                lens.distortion[1] = -0.002f;
                lens.distortion[3] = 0.003f;
                lens.distortion[4] = -0.004f;
            }
            const auto rays = inverse_distortion_sample_map(lens, nullptr).cpu().to_vector();
            size_t finite = 0, missing = 0;
            const float bound = 1 / std::sqrt(0.6f);
            const float maximum = bound * (1 - 0.2f * bound * bound);
            for (int y = 0; y < lens.src_height; ++y)
                for (int x = 0; x < lens.src_width; ++x) {
                    const size_t offset = (size_t(y) * lens.src_width + x) * 2;
                    const float ux = rays[offset], uy = rays[offset + 1];
                    const float xd = (x + 0.5f - lens.src_cx) / lens.src_fx;
                    const float yd = (y + 0.5f - lens.src_cy) / lens.src_fy;
                    if (lens_case == 2 && std::hypot(xd, yd) > maximum + 1e-4f) {
                        require(std::isnan(ux) && std::isnan(uy), "HIP inverse accepted a ray beyond the first radial fold");
                        ++missing;
                        continue;
                    }
                    require(std::isfinite(ux) && std::isfinite(uy), "HIP inverse rejected a valid pinhole ray");
                    float dx, dy;
                    lfs::core::internal::warp_math::apply_distortion_pinhole(ux, uy, lens.distortion, lens.num_distortion, dx, dy);
                    require(std::hypot((dx - xd) * lens.src_fx, (dy - yd) * lens.src_fy) < 0.011f,
                            "HIP inverse lens residual exceeds camera pixel tolerance");
                    if (lens_case == 0)
                        require(std::abs(ux - xd) < 1e-6f && std::abs(uy - yd) < 1e-6f, "Identity lens changed rays");
                    ++finite;
                }
            require(finite && (lens_case != 2 || missing), "Lens inverse fixture did not cover its branches");
        }
        std::puts("PASS native pinhole inverse rays: identity, positive radial, first-fold invalid rays, coupled tangential residuals");
        {
            UndistortParams bad{};
            bad.src_width = bad.src_height = 8;
            bad.src_fx = bad.src_fy = 10;
            auto reject_lens = [&](const UndistortParams& lens) {
                bool rejected = false;
                try {
                    (void)inverse_distortion_sample_map(lens, nullptr);
                } catch (const std::exception&) { rejected = true; }
                require(rejected, "Invalid inverse calibration was not rejected");
            };
            bad.src_width = 0;
            reject_lens(bad);
            bad.src_width = 8;
            bad.src_fx = std::numeric_limits<float>::quiet_NaN();
            reject_lens(bad);
            bad.src_fx = 10;
            bad.num_distortion = 13;
            reject_lens(bad);
            bad.num_distortion = 0;
            bad.model_type = CameraModelType::FISHEYE;
            reject_lens(bad);
        }
        {
            UndistortParams lens{};
            lens.src_width = lens.dst_width = 17;
            lens.src_height = lens.dst_height = 129;
            lens.src_fx = lens.dst_fx = 60;
            lens.src_fy = lens.dst_fy = 80;
            lens.src_cx = lens.dst_cx = 8.5f;
            lens.src_cy = lens.dst_cy = 64.51723f;
            lens.model_type = CameraModelType::PINHOLE;
            lens.num_distortion = 1;
            lens.distortion[0] = 0.1f;
            std::vector<float> pixels(3 * 17 * 129);
            for (size_t i = 0; i < pixels.size(); ++i)
                pixels[i] = float(i % 263) / 262;
            const auto source = Tensor::from_vector(pixels, {3, 129, 17}, Device::CPU);
            const auto whole = undistort_image(source, lens, nullptr).to_vector();
            const auto tiled = lfs::training::hip::undistort_cpu_target(source, lens).to_vector();
            require(whole == tiled, "Parallel CPU target preparation changed pixels at tile edges");
            // Exercise a native byte source and a different destination grid:
            // this is the combined resize/warp used by Media Ingest.
            auto bytes = Tensor::empty({3, 129, 17}, Device::CPU, DataType::UInt8);
            auto* samples = bytes.ptr<uint8_t>();
            for (size_t i = 0; i < bytes.numel(); ++i)
                samples[i] = uint8_t(i % 256);
            lens.dst_width = 11;
            lens.dst_height = 83;
            lens.dst_fx *= 11.f / 17;
            lens.dst_fy *= 83.f / 129;
            lens.dst_cx *= 11.f / 17;
            lens.dst_cy *= 83.f / 129;
            const auto byte_reference = undistort_image(bytes, lens, nullptr).to_vector();
            const auto byte_parallel = lfs::training::hip::undistort_cpu_target(bytes, lens).to_vector();
            require(byte_reference == byte_parallel, "Native-byte CPU target preparation changed combined resize/warp");
            for (const auto& cpu_source : {source, bytes}) {
                const auto reference = undistort_image(cpu_source, lens, nullptr).to_vector();
                const auto gpu_source = cpu_source.to(Device::GPU);
                const auto actual = undistort_image(gpu_source, lens, nullptr).cpu().to_vector();
                float max_error = 0;
                double square = 0;
                require(actual.size() == reference.size(), "HIP RGB warp changed output extent");
                for (size_t i = 0; i < actual.size(); ++i) {
                    const float error = std::abs(actual[i] - reference[i]);
                    require(std::isfinite(actual[i]), "HIP RGB warp produced a nonfinite pixel");
                    max_error = std::max(max_error, error);
                    square += double(error) * error;
                }
                require(max_error < 4e-5f && square / actual.size() < 1e-10,
                        "HIP RGB warp differs from the independent CPU resampler");
                std::printf("HIP RGB warp dtype %u: CPU max %.9g MSE %.9g\n", unsigned(cpu_source.dtype()), max_error, square / actual.size());
                for (unsigned kind = 0; kind < 4; ++kind) {
                    auto bad_warp = lens;
                    if (kind == 0)
                        bad_warp.dst_fx = 0;
                    if (kind == 1)
                        bad_warp.src_fx = std::numeric_limits<float>::quiet_NaN();
                    if (kind == 2)
                        bad_warp.num_distortion = 13;
                    if (kind == 3)
                        bad_warp.dst_fx = bad_warp.src_fx / 128.f;
                    bool rejected = false;
                    try {
                        (void)undistort_image(gpu_source, bad_warp, nullptr).cpu();
                    } catch (const std::exception&) { rejected = true; }
                    require(rejected, "HIP RGB warp accepted invalid calibration or unbounded work");
                }
                require(undistort_image(gpu_source, lens, nullptr).cpu().numel() == reference.size(),
                        "HIP RGB warp did not recover from a rejected request");
            }
            auto invalid = lens;
            invalid.dst_fy = 0;
            bool rejected = false;
            try {
                (void)lfs::training::hip::undistort_cpu_target(source, invalid);
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "CPU target preparation accepted invalid focal length");
            std::puts("PASS parallel CPU target preparation: every float matches existing full-frame resampler");
        }
        Fixture fixture;
        // This is a storage/ownership integration test, not a dense-scene
        // optimization benchmark. Use six identifiable active Gaussians and
        // retain 263 logical rows to cover tails and culled-gradient handling.
        for (size_t g = 6; g < n; ++g) {
            fixture.values[0][g * 3 + 2] = 2000;
            fixture.values[4][g] = -100;
        }
        std::vector<float> target_colors(n * 3);
        for (size_t i = 0; i < target_colors.size(); ++i)
            target_colors[i] = sh_c0 * fixture.values[3][i] + 0.5f;
        fixture.target = cpu_render(fixture.values[0], fixture.values[1], fixture.values[2], target_colors, fixture.values[4]);
        fixture.perturb();
        std::puts("Core storage fixture: 6 active of 263 fixed SH0 Gaussians, 35x29, 400 steps");
        constexpr size_t allocation_bytes = 65536, capacity = 320;
        auto owner = std::make_shared<Tensor>(Tensor::empty({allocation_bytes}, Device::GPU, DataType::UInt8));
        std::vector<uint8_t> sentinel(allocation_bytes, 0x37);
        owner->copy_from(Tensor::from_blob(sentinel.data(), {allocation_bytes}, Device::CPU, DataType::UInt8));
        using Storage = SplatExportableStorage;
        Storage::Control storage;
        storage.block = std::make_shared<ExportableBlock>();
        storage.block->device_ptr = const_cast<void*>(std::as_const(*owner).data_ptr());
        storage.block->reserved_bytes = allocation_bytes;
        storage.capacity = capacity;
        storage.generation = 1;
        const size_t order[] = {Storage::Means, Storage::Scaling, Storage::Rotation, Storage::Opacity, Storage::Sh0};
        const size_t attributes[] = {3, 3, 4, 1, 3};
        size_t cursor = 256;
        for (size_t i = 0; i < 5; ++i) {
            storage.region_offsets[order[i]] = cursor;
            storage.region_bytes[order[i]] = capacity * attributes[i] * 4;
            cursor = (cursor + storage.region_bytes[order[i]] + 255) / 256 * 256;
        }
        const auto layout = lfs::training::hip::splat_storage_layout(storage, n);
        std::array<Tensor, 5> views;
        const size_t family_attributes[] = {3, 4, 3, 3, 1};
        for (size_t i = 0; i < 5; ++i) {
            views[i] = Tensor::view_sharing_storage(*owner, layout.offsets_bytes[i], {n, family_attributes[i]}, capacity, DataType::Float32, "hip.training.model");
            views[i].copy_from(Tensor::from_vector(fixture.values[i], {n, family_attributes[i]}, Device::CPU));
        }
        LfsHipTrainDeviceModel device{sizeof(device), 0, reinterpret_cast<uintptr_t>(std::as_const(*owner).data_ptr()), allocation_bytes};
        LfsHipTrainConfig config{sizeof(config), 0, n, 64 * 1024 * 1024};
        auto reject = [&](LfsHipTrainDeviceModel bad, int expected = LFS_HIP_TRAIN_INVALID) {
            bool rejected = false;
            try {
                Session invalid(module, config, bad, layout, owner);
            } catch (const TrainingError& error) { rejected = error.status == expected; }
            require(rejected, "Native Tensor model contract was not rejected");
        };
        auto bad = device;
        bad.struct_bytes = 0;
        reject(bad);
        bad = device;
        bad.device_ordinal = 1;
        reject(bad);
        bad = device;
        bad.address = 0;
        reject(bad);
        bad = device;
        bad.address += 256;
        reject(bad);
        bad = device;
        bad.allocation_bytes = UINT64_MAX;
        reject(bad);
        config.allocation_budget_bytes = allocation_bytes;
        reject(device, LFS_HIP_TRAIN_RESOURCE_EXHAUSTED);
        config.allocation_budget_bytes = 64 * 1024 * 1024;
        {
            // Explicit FP32 swizzled SH3 view in the same real Tensor allocation.
            const size_t sh_offset = cursor;
            constexpr size_t sh_bytes = ((n + 31) / 32 * 32) * 12 * 16;
            const size_t total = sh_offset + sh_bytes + 256;
            auto shared = std::make_shared<Tensor>(Tensor::zeros({total}, Device::GPU, DataType::UInt8));
            auto current = std::make_shared<Storage::Control>(storage);
            current->block = std::make_shared<ExportableBlock>();
            current->block->device_ptr = const_cast<void*>(std::as_const(*shared).data_ptr());
            current->block->reserved_bytes = total;
            current->region_offsets[Storage::ShN] = sh_offset;
            current->region_bytes[Storage::ShN] = sh_bytes;
            for (size_t i = 0; i < 5; ++i) {
                auto view = Tensor::view_sharing_storage(*shared, layout.offsets_bytes[i], {n, family_attributes[i]}, capacity, DataType::Float32, "hip.training.base");
                view.copy_from(Tensor::from_vector(fixture.values[i], {n, family_attributes[i]}, Device::CPU));
            }
            auto sh = std::make_shared<Tensor>(Tensor::view_sharing_storage(*shared, sh_offset,
                                                                            {sh_bytes / sizeof(float)}, sh_bytes / sizeof(float), DataType::Float32, "hip.training.sh"));
            lfs::training::hip::TensorModel linked(shared, current, n, 0, sh, 3, 3);
            auto attached = linked.open(module, config);
            attached->set_frame(fixture.frame());
            attached->step({0.001f, 0.001f, 0.001f, 0.002f, 0.001f});
            std::vector<float> canonical(n * 45);
            attached->read_parameter(5, canonical);
            const auto swizzled = sh->cpu().to_vector();
            for (size_t g = 0; g < n; ++g)
                for (size_t c = 0; c < 45; ++c)
                    require(swizzled[((g / 32 * 12 + c / 4) * 32 + g % 32) * 4 + c % 4] == canonical[g * 45 + c],
                            "Core Tensor SH alias differs from native training");
            *sh = Tensor::zeros({sh_bytes / sizeof(float)}, Device::GPU);
            bool rejected = false;
            try {
                attached->step({0.001f, 0.001f, 0.001f, 0.002f, 0.001f});
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Reassigned SH Tensor was not rejected");
            attached->close();
        }
        for (size_t i = 0; i < 5; ++i)
            require(views[i].cpu().to_vector() == fixture.values[i], "Failed attachment modified model");
        {
            auto current = std::make_shared<Storage::Control>(storage);
            lfs::training::hip::TensorModel linked(owner, current, n, 0);
            auto attached = linked.open(module, config);
            attached->set_frame(fixture.frame());
            const auto unchanged = views[0].cpu().to_vector();
            auto reject_stale = [&] {
                bool rejected = false;
                try {
                    attached->step({0.002f, 0.002f, 0.003f, 0.01f, 0.01f});
                } catch (const std::invalid_argument&) { rejected = true; }
                require(rejected && views[0].cpu().to_vector() == unchanged, "Stale Tensor model was updated");
            };
            ++current->generation;
            reject_stale();
            --current->generation;
            current->poisoned = true;
            reject_stale();
            current->poisoned = false;
            ++current->region_offsets[Storage::Opacity];
            reject_stale();
            --current->region_offsets[Storage::Opacity];
            require(attached->info().completed_steps == 0, "Stale model advanced optimizer step");
            attached->close();
            auto replaced = std::make_shared<Tensor>(*owner);
            lfs::training::hip::TensorModel replacement(replaced, current, n, 0);
            auto old = replacement.open(module, config);
            old->set_frame(fixture.frame());
            *replaced = Tensor::empty({allocation_bytes}, Device::GPU, DataType::UInt8);
            bool rejected = false;
            try {
                old->step({0.002f, 0.002f, 0.003f, 0.01f, 0.01f});
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Reassigned Tensor allocation was not rejected");
            old->close();
            auto poisoned = linked.open(module, config);
            poisoned->set_frame(fixture.frame());
            auto invalid_sh = fixture.values[3];
            invalid_sh[0] = std::numeric_limits<float>::quiet_NaN();
            views[3].copy_from(Tensor::from_vector(invalid_sh, {n, size_t(3)}, Device::CPU));
            bool invalidated = false;
            try {
                poisoned->step({0.002f, 0.002f, 0.003f, 0.01f, 0.01f});
            } catch (const TrainingError& error) { invalidated = error.status == LFS_HIP_TRAIN_POISONED && current->poisoned; }
            require(invalidated, "Poisoned optimizer did not invalidate the application's model control");
            poisoned->close();
            current->poisoned = false;
            for (size_t family = 0; family < 5; ++family)
                views[family].copy_from(Tensor::from_vector(fixture.values[family], {n, family_attributes[family]}, Device::CPU));
            std::puts("PASS Tensor model generation, poison, offsets and allocation replacement preflight; stale sessions close safely");
        }
        std::weak_ptr<Tensor> lifetime = owner;
        // Leave native Tensor producers pending; attachment must bridge them.
        views[3].mul_(2.f);
        views[3].mul_(0.5f);
        auto attached_session = [&] {
            lfs::training::hip::TensorModel linked(owner, std::make_shared<Storage::Control>(storage), n, 0);
            return linked.open(module, config);
        }();
        Session& session = *attached_session;
        Session control(module, config, fixture.model());
        Session repeat(module, config, fixture.model());
        owner.reset();
        require(!lifetime.expired(), "Session did not retain Tensor model owner");
        const bool vulkan_live_before = gpu_backend_live(GpuBackend::Vulkan);
        {
            auto cpu = [&](size_t f) { return Tensor::from_vector(fixture.values[f], {n, family_attributes[f]}, Device::CPU); };
            SplatData host_model(0, cpu(0), cpu(3).reshape({int(n), 1, 3}), Tensor::empty({n, 0, 3}, Device::CPU), cpu(2), cpu(1), cpu(4), 1.f);
            require(gpu_backend_of(host_model.shN()) == GpuBackend::HIP, "CPU-backed HIP import/export model created foreign SH storage");
        }
        require(gpu_backend_live(GpuBackend::Vulkan) == vulkan_live_before, "HIP import/export initialized a Vulkan context");
        const auto frame = fixture.frame();
        Camera training_camera(rotation, Tensor::zeros({3}, Device::CPU), 64, 64, float(width), float(height), {}, {},
                               CameraModelType::PINHOLE, "training", {}, {}, width * 2, height * 2, 1);
        training_camera.set_image_dimensions(1, 1);
        const auto training_target = Tensor::from_vector(fixture.target, {3, height, width}, Device::CPU);
        lfs::training::hip::CameraFrame training_frame(training_camera, training_target);
        training_frame.bind(session);
        control.set_frame(frame);
        repeat.set_frame(frame);
        const LfsHipTrainRates rates{0.002f, 0.002f, 0.003f, 0.015f / sh_c0, 0.01f};
        auto inspect = [&] {
            for (size_t family = 0; family < 5; ++family) {
                std::vector<float> module_values(layout.float_counts[family]);
                session.read_parameter(family, module_values);
                const auto tensor_values = views[family].cpu().to_vector();
                require(std::memcmp(tensor_values.data(), module_values.data(), module_values.size() * sizeof(float)) == 0, "Training and core Tensor do not alias the same parameters");
            }
        };
        const float initial = session.loss();
        inspect();
        for (unsigned i = 0; i < 400; ++i) {
            // Keep full updates initially, then reduce all five learning rates
            // to measure final convergence rather than fixed-rate oscillation.
            // The exact same scheduler is passed to all three sessions.
            const float scale = std::max(0.02f, std::pow(0.97f, float(std::max(0, int(i) - 200))));
            const LfsHipTrainRates scheduled{rates.means * scale, rates.quaternions * scale, rates.log_scales * scale, rates.sh0 * scale, rates.raw_opacities * scale};
            session.step(scheduled);
            control.step(scheduled);
            repeat.step(scheduled);
            if (i == 0) {
                float worst = 0;
                for (size_t family = 0; family < 5; ++family) {
                    std::vector<float> a(layout.float_counts[family]), b(a.size());
                    session.read_parameter(family, a);
                    control.read_parameter(family, b);
                    for (size_t k = 0; k < a.size(); ++k)
                        worst = std::max(worst, std::abs(a[k] - b[k]));
                }
                std::printf("First core/private parameter update max %.9g\n", worst);
                require(worst < 1e-5f, "Native core Tensor layout changed first optimizer update");
            }
            if ((i + 1) % 50 == 0)
                inspect();
        }
        const float final = session.loss(), reference = control.loss(), repeated = repeat.loss();
        std::printf("Scheduled convergence: %.9g -> %.9g; controls %.9g / %.9g\n", initial, final, reference, repeated);
        require(final < initial * 0.1f && final >= std::min(reference, repeated) * 0.99f - 1e-9f && final <= std::max(reference, repeated) * 1.01f + 1e-9f,
                "Training with core Tensor model changed convergence");
        std::vector<float> image(channels), reference_image(channels);
        session.render(image);
        control.render(reference_image);
        double image_mse = 0;
        float maximum = 0;
        for (size_t i = 0; i < channels; ++i) {
            const double delta = double(image[i]) - reference_image[i];
            image_mse += delta * delta / channels;
            maximum = std::max(maximum, float(std::abs(delta)));
        }
        std::vector<float> repeat_image(channels);
        repeat.render(repeat_image);
        double repeat_mse = 0;
        float repeat_max = 0;
        for (size_t i = 0; i < channels; ++i) {
            const double d = double(reference_image[i]) - repeat_image[i];
            repeat_mse += d * d / channels;
            repeat_max = std::max(repeat_max, float(std::abs(d)));
        }
        std::printf("Core/private: initial %.9g final %.9g, controls %.9g / %.9g; image max %.9g MSE %.9g; private/private max %.9g MSE %.9g\n", initial, final, reference, repeated, maximum, image_mse, repeat_max, repeat_mse);
        require(maximum < 1.f / 255 && image_mse < 1e-8, "Core Tensor model changed rendered quality");
        auto pinned = lifetime.lock();
        require(bool(pinned), "Session released owner during training");
        const auto all = pinned->cpu();
        std::vector<uint8_t> bytes(allocation_bytes);
        std::memcpy(bytes.data(), all.data_ptr(), bytes.size());
        std::array<std::pair<size_t, size_t>, 5> intervals{};
        for (size_t i = 0; i < 5; ++i)
            intervals[i] = {layout.offsets_bytes[i], layout.offsets_bytes[i] + layout.float_counts[i] * 4};
        std::sort(intervals.begin(), intervals.end());
        size_t end = 0;
        for (auto [begin, next] : intervals) {
            for (size_t i = end; i < begin; ++i)
                require(bytes[i] == 0x37, "Training wrote into reserved model rows");
            end = next;
        }
        for (size_t i = end; i < bytes.size(); ++i)
            require(bytes[i] == 0x37, "Training wrote beyond model regions");
        pinned.reset();
        session.close();
        require(lifetime.expired(), "Closed session retained the Tensor owner object");
        require(views[0].cpu().to_vector() != fixture.values[0], "Training did not update core Tensor parameters");
        std::printf("PASS actual core Tensor training: 400 steps MSE %.9g -> %.9g; controls %.9g / %.9g; image max %.9g, MSE %.9g; all five aliases byte-exact, reserved rows intact, owner retained until close\n", initial, final, reference, repeated, maximum, image_mse);
        control.close();
        repeat.close();
    }
    require(configured_gpu_backend() == process_backend, "Training changed process Tensor backend default");
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)), "Core HIP context did not shut down after releasing model views");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "CORE HIP TRAINING FAILED: %s\n", error.what());
    return 1;
}
