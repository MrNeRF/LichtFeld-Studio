/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../src/training/hip/session.hpp"
#include "training_session_fixture.hpp"
#include <cstdio>
#include <cstring>
#include <hip/hip_runtime.h>
#include <limits>
#include <string_view>

using lfs::training::hip::Session;
using lfs::training::hip::TrainingError;
static void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
static void check(hipError_t error) {
    if (error != hipSuccess)
        throw std::runtime_error(hipGetErrorString(error));
}

int main(int argc, char** argv) {
    try {
        require(argc == 2 || (argc == 3 && std::string_view(argv[2]) == "--fused"), "Pass an absolute training module path and optional --fused");
        using namespace hip_fixture;
        Fixture fixture;
        for (size_t g = 6; g < n; ++g) {
            fixture.values[0][g * 3 + 2] = 2000;
            fixture.values[4][g] = -100;
        }
        const std::filesystem::path module(argv[1]);
        auto layout = Session::model_layout(module, n);
        const size_t sh_offset = (layout.required_bytes + 255) / 256 * 256;
        constexpr size_t slots = 12, rest = 45;
        const size_t sh_bytes = ((n + 31) / 32 * 32) * slots * 16;
        layout.required_bytes = sh_offset + sh_bytes + 256;
        std::vector<uint8_t> original(layout.required_bytes, 0x37);
        for (unsigned f = 0; f < 5; ++f)
            std::memcpy(original.data() + layout.offsets_bytes[f], fixture.values[f].data(), fixture.values[f].size() * sizeof(float));
        std::vector<float> coefficients(n * rest);
        for (size_t g = 0; g < n; ++g)
            for (size_t c = 0; c < rest; ++c) {
                const float value = 0.02f * std::sin(float(g * 13 + c));
                coefficients[g * rest + c] = value;
                const size_t address = ((g / 32 * slots + c / 4) * 32 + g % 32) * 4 + c % 4;
                std::memcpy(original.data() + sh_offset + address * sizeof(float), &value, sizeof(float));
            }
        void* pointer = nullptr;
        check(hipMalloc(&pointer, original.size()));
        auto owner = std::shared_ptr<void>(pointer, [](void* data) { (void)hipFree(data); });
        std::weak_ptr<void> lifetime = owner;
        check(hipMemcpy(pointer, original.data(), original.size(), hipMemcpyHostToDevice));
        const LfsHipTrainConfig config{sizeof(config), 0, n, 64 * 1024 * 1024};
        const LfsHipTrainDeviceModel device{sizeof(device), 0, reinterpret_cast<uintptr_t>(pointer), original.size()};
        Session resident(module, config, device, layout, owner);
        Session control(module, config, fixture.model());
        resident.set_rasterizer(1);
        control.set_rasterizer(1);
        if (argc == 3) {
            resident.set_optimizer(1);
            control.set_optimizer(1);
        }
        const auto resident_before = resident.info().allocated_bytes;
        const auto control_before = control.info().allocated_bytes;
        const LfsHipTrainResidentSh descriptor{sizeof(descriptor), 3, 3, 0.002f, sh_offset};
        resident.attach_resident_sh(descriptor);
        control.set_sh({sizeof(LfsHipTrainSh), 3, 3, 0.002f, coefficients.data(), coefficients.size()});
        const auto resident_delta = resident.info().allocated_bytes - resident_before;
        const auto control_delta = control.info().allocated_bytes - control_before;
        require(control_delta == resident_delta + sh_bytes, "Resident SH was allocated or charged twice");
        for (unsigned invalid = 0; invalid < 5; ++invalid) {
            auto bad = descriptor;
            if (invalid == 0)
                bad.offset_bytes += 4;
            if (invalid == 1)
                bad.offset_bytes = layout.offsets_bytes[0];
            if (invalid == 2)
                bad.offset_bytes = UINT64_MAX - 255;
            if (invalid == 3)
                bad.active_degree = 4;
            if (invalid == 4)
                bad.struct_bytes = 0;
            const auto live = resident.info().allocated_bytes;
            bool rejected = false;
            try {
                resident.attach_resident_sh(bad);
            } catch (const TrainingError& error) { rejected = error.status == LFS_HIP_TRAIN_INVALID; }
            require(rejected && resident.info().allocated_bytes == live, "Invalid resident SH changed session storage");
        }
        // A failed replacement must leave the attached SH state and budget intact.
        const float nan = std::numeric_limits<float>::quiet_NaN();
        check(hipMemcpy(static_cast<uint8_t*>(pointer) + sh_offset, &nan, sizeof(nan), hipMemcpyHostToDevice));
        const auto live_before_invalid = resident.info().allocated_bytes;
        bool finite_rejected = false;
        try {
            resident.attach_resident_sh(descriptor);
        } catch (const TrainingError& error) { finite_rejected = error.status == LFS_HIP_TRAIN_INVALID; }
        require(finite_rejected && resident.info().allocated_bytes == live_before_invalid,
                "Nonfinite resident SH replacement changed storage");
        check(hipMemcpy(static_cast<uint8_t*>(pointer) + sh_offset, coefficients.data(), sizeof(float), hipMemcpyHostToDevice));
        resident.set_frame(fixture.frame());
        control.set_frame(fixture.frame());
        std::vector<float> image(channels), reference(channels), values(coefficients.size());
        resident.render(image);
        control.render(reference);
        for (size_t i = 0; i < image.size(); ++i)
            require(std::abs(image[i] - reference[i]) <= 3e-6f, "Resident SH rendering differs from uploaded values");
        resident.read_parameter(5, values);
        require(values == coefficients, "Resident SH canonical readback changed coefficient values");
        const float before = resident.loss();
        for (unsigned step = 0; step < 50; ++step)
            resident.step({0.001f, 0.001f, 0.001f, 0.002f, 0.001f});
        const float after = resident.loss();
        require(std::isfinite(after) && after < before, "Resident SH training did not improve loss");
        resident.read_parameter(5, values);
        require(values != coefficients, "Resident SH coefficients were not updated");
        std::vector<uint8_t> actual(original.size());
        check(hipMemcpy(actual.data(), pointer, actual.size(), hipMemcpyDeviceToHost));
        std::vector<bool> live(actual.size());
        for (size_t f = 0; f < 5; ++f)
            for (size_t i = 0; i < layout.float_counts[f] * 4; ++i)
                live[layout.offsets_bytes[f] + i] = true;
        for (size_t g = 0; g < n; ++g)
            for (size_t c = 0; c < rest; ++c) {
                const size_t address = ((g / 32 * slots + c / 4) * 32 + g % 32) * 4 + c % 4;
                float value;
                std::memcpy(&value, actual.data() + sh_offset + address * sizeof(float), sizeof(float));
                require(value == values[g * rest + c], "Resident SH GPU alias differs from training readback");
                for (unsigned b = 0; b < 4; ++b)
                    live[sh_offset + address * 4 + b] = true;
            }
        for (size_t i = 0; i < actual.size(); ++i)
            if (!live[i])
                require(actual[i] == original[i], "Resident SH modified padding or another model region");
        owner.reset();
        require(!lifetime.expired(), "Resident SH model owner released during training");
        resident.close();
        require(lifetime.expired(), "Resident SH retained owner after close");
        std::printf("Resident HIP SH3: aliases exact, padding intact, %zu value bytes shared once; loss %.9g -> %.9g\n", sh_bytes, before, after);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
