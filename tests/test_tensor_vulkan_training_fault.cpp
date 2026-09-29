/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/error.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <vector>
#if defined(LFS_TEST_TENSOR_VULKAN)
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "lfs/training/ops/fast_vulkan.hpp"
#include "training/vulkan/fast_state.hpp"

namespace {
    TEST(TensorVulkanTrainingFault, InvalidTileDataSurfacesAtSynchronization) {
        using namespace lfs;
        using core::DataType;
        using core::Device;
        using core::Tensor;
        namespace vk = training::vulkan;
        if (!core::gpu_backend_available(core::GpuBackend::Vulkan))
            GTEST_SKIP();
        core::GpuBackendScope scope(core::GpuBackend::Vulkan);
        const auto context = core::internal::acquire_vulkan_context();
        for (const uint32_t kind : {0u, 1u, 2u}) {
            SCOPED_TRACE(kind);
            auto keys = Tensor::full({1}, kind == 0 ? 1 : 0, Device::GPU, DataType::Int32);
            auto values = Tensor::full({1}, kind == 1 ? 1 : 0, Device::GPU, DataType::Int32);
            auto ranges = Tensor::zeros({2}, Device::GPU, DataType::Int32);
            auto offsets = Tensor::ones({1}, Device::GPU, DataType::Int64);
            auto projected = Tensor::zeros({sizeof(vk::Projected) / 4}, Device::GPU);
            vk::FastPush p{};
            p.keys = vk::address(keys);
            p.values = vk::address(values);
            p.ranges = vk::address(ranges);
            p.offsets = vk::address(offsets);
            p.projected = vk::address(projected);
            p.status = context->fault_address();
            p.instances = p.visible = p.grid_width = p.grid_height = 1;
            p.stage = kind == 2 ? 3 : 4;
            const std::array reads{vk::ref(keys), vk::ref(values), vk::ref(offsets), vk::ref(projected)};
            const std::array writes{vk::ref(ranges)};
            // A zero-size projected tile with a nonzero offset tests emission
            // mismatch; the other cases test invalid tile and primitive ids.
            ASSERT_NO_THROW(vk::dispatch("fast_forward", p, reads, writes, 1, vk::specialization(p, p.stage)));
            try {
                (void)ranges.cpu();
                FAIL() << "Expected a deferred rasterizer fault";
            } catch (const lfs::Exception& error) {
                EXPECT_EQ(error.error().code(), ErrorCode::BoundsViolation);
                EXPECT_NE(lfs::format_for_developer(error.error()).find("invalid tile data"), std::string::npos);
            }
            EXPECT_NO_THROW((void)ranges.cpu());
            EXPECT_EQ(ranges.to_vector_int(), (std::vector<int>{0, 0}));
        }
    }
    TEST(TensorVulkanTrainingFault, IndirectForwardPreservesBitsThroughOverflowAndEmptyFrames) {
        using namespace lfs;
        using core::DataType;
        using core::Device;
        using core::Tensor;
        if (!core::gpu_backend_available(core::GpuBackend::Vulkan))
            GTEST_SKIP();
        core::GpuBackendScope scope(core::GpuBackend::Vulkan);
        const auto& table = training::vulkan_fast_ops();
        constexpr size_t count = 257;
        std::vector<float> positions(count * 3), rotations(count * 4, 0);
        for (size_t i = 0; i < count; ++i) {
            positions[i * 3] = 0.013f * float(int(i % 11) - 5);
            positions[i * 3 + 1] = 0.017f * float(int(i % 7) - 3);
            positions[i * 3 + 2] = 2.f + 0.02f * float(i % 3);
            rotations[i * 4] = 1.f;
        }
        auto means = Tensor::from_vector(positions, {count, 3}, Device::GPU);
        auto rotation = Tensor::from_vector(rotations, {count, 4}, Device::GPU);
        auto scales = Tensor::full({count, 3}, -2.f, Device::GPU);
        auto opacity = Tensor::full({count}, 2.f, Device::GPU);
        auto dc = Tensor::full({count, 3}, 0.3f, Device::GPU);
        auto view = Tensor::eye(4, Device::GPU), camera = Tensor::zeros({3}, Device::GPU);
        auto background = Tensor::from_vector({0.12f, 0.07f, 0.18f}, {3}, Device::GPU);
        Tensor absent;
        gpu_ops::FastSaved reference{.backend = table.create()}, indirect{.backend = table.create()};
        auto& exact_state = static_cast<training::vulkan::FastState&>(*reference.backend);
        auto& indirect_state = static_cast<training::vulkan::FastState&>(*indirect.backend);
        exact_state.indirect = false;
        for (size_t slot = 8; slot < 12; ++slot)
            indirect_state.scratch[slot] = Tensor::empty({1}, Device::GPU, DataType::UInt32);
        for (const auto [width, height, empty] : {std::tuple{35, 27, false}, std::tuple{65, 49, false}, std::tuple{35, 27, true}, std::tuple{35, 27, false}}) {
            SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height) + (empty ? " empty" : " visible"));
            opacity.fill_(empty ? -100.f : 2.f);
            Tensor ai, aa, ad, an, bi, ba, bd, bn;
            const gpu_ops::FastParams params{.full_image = {height, width}, .intrinsics = {100, 100, width * 0.5f, height * 0.5f}, .mip_filter = true, .render_normal = true, .render_depth = true};
            const gpu_ops::SplatInputs inputs{means, scales, rotation, opacity, dc, absent, absent};
            const auto ra = table.forward(reference, inputs, view, camera, background, absent, params, {ai, aa, ad, an}, absent);
            const auto rb = table.forward(indirect, inputs, view, camera, background, absent, params, {bi, ba, bd, bn}, absent);
            ASSERT_EQ(ra.code, gpu_ops::RasterResult::Code::Success);
            ASSERT_EQ(rb.code, ra.code);
            ASSERT_EQ(rb.has_work, ra.has_work);
            EXPECT_EQ(rb.has_work, !empty);
            EXPECT_GT(indirect_state.scratch[8].numel(), 256);
            const auto check = [](const char* name, const Tensor& x, const Tensor& y) {
                SCOPED_TRACE(name);
                const auto a = x.cpu().contiguous(), b = y.cpu().contiguous();
                ASSERT_EQ(a.bytes(), b.bytes());
                EXPECT_EQ(std::memcmp(a.data_ptr(), b.data_ptr(), a.bytes()), 0);
            };
            check("image", bi, ai);
            check("alpha", ba, aa);
            check("depth", bd, ad);
            check("normal", bn, an);
            check("ranges", indirect_state.ranges, exact_state.ranges);
            check("transmittance", indirect_state.transmittance, exact_state.transmittance);
            check("last", indirect_state.last, exact_state.last);
            check("offsets", indirect_state.offsets, exact_state.offsets);
            const size_t instances = exact_state.push.instances;
            for (auto [x, y] : {std::pair{&indirect_state.keys_a, &exact_state.keys_a}, std::pair{&indirect_state.keys_b, &exact_state.keys_b},
                                std::pair{&indirect_state.values_a, &exact_state.values_a}, std::pair{&indirect_state.values_b, &exact_state.values_b}})
                if (instances)
                    check("sorted pairs", x->slice(0, 0, instances), *y);
            if (ra.has_work) {
                // A single contributing pixel avoids the rasterizer's existing
                // cross-warp floating atomic-order variability in this bitwise check.
                std::vector<float> pixels(size_t(width) * height * 3, 0.f);
                pixels[size_t(height / 2) * width + width / 2] = 0.3f;
                auto gradient = Tensor::from_vector(pixels, {3, size_t(height), size_t(width)}, Device::GPU);
                auto ma = means.clone(), mb = means.clone();
                auto pa = Tensor::zeros({count * 3 * 4}, Device::GPU, DataType::UInt8), pb = pa.clone();
                auto qa = Tensor::zeros({(count + 255) / 256, 4}, Device::GPU), qb = qa.clone();
                gpu_ops::BackwardAdamParam ga{ma, pa, qa, absent, absent, absent, absent};
                gpu_ops::BackwardAdamParam gb{mb, pb, qb, absent, absent, absent, absent};
                for (auto* group : {&ga, &gb}) {
                    group->enabled = true;
                    group->joint_bits = 16;
                    group->primitives = count;
                    group->elements = count * 3;
                    group->attributes = 3;
                    group->step_size = 0.01f;
                }
                const gpu_ops::BackwardAdamParam disabled{absent, absent, absent, absent, absent, absent, absent};
                const gpu_ops::BackwardAdam aa{{ga, disabled, disabled, disabled, disabled, disabled}, absent, absent, absent, absent, absent, absent};
                const gpu_ops::BackwardAdam ab{{gb, disabled, disabled, disabled, disabled, disabled}, absent, absent, absent, absent, absent, absent};
                table.backward(reference, {gradient, absent, absent, absent}, absent, absent, absent, absent, aa, DensificationType::None);
                table.backward(indirect, {gradient, absent, absent, absent}, absent, absent, absent, absent, ab, DensificationType::None);
                check("updated means", mb, ma);
                check("packed moments", pb, pa);
                check("moment bounds", qb, qa);
            }
            table.release(reference);
            table.release(indirect);
        }
    }
} // namespace
#endif
