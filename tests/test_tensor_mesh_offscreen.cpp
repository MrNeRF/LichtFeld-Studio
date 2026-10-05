/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/mesh_data.hpp"
#include "core/tensor_backend.hpp"
#include "rendering/mesh_offscreen_renderer.hpp"
#include "rendering/viewport_tensor_meshes.hpp"
#include "window/graphics_context.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::GpuBackendScope;
    using lfs::core::MeshData;
    using lfs::core::Tensor;
    using lfs::rendering::MeshLayer;
    using lfs::vis::TensorMeshPass;
    using lfs::vis::ViewportMeshDrawItem;
    using lfs::vis::ViewportMeshPassDesc;

    constexpr int kWidth = 160;
    constexpr int kHeight = 120;
    constexpr float kPlaneDepth = 3.0f;

    MeshData makeTexturedPlane() {
        const std::vector<float> vertices{
            -1.0f, -0.75f, -kPlaneDepth,
            1.0f, -0.75f, -kPlaneDepth,
            1.0f, 0.75f, -kPlaneDepth,
            -1.0f, 0.75f, -kPlaneDepth,
        };
        const std::vector<std::int32_t> indices{0, 1, 2, 0, 2, 3};
        MeshData mesh(
            Tensor::from_vector(vertices, {4, 3}, Device::CPU),
            Tensor::from_vector(indices, {2, 3}, Device::CPU));
        mesh.normals = Tensor::from_vector(
            std::vector<float>{0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1},
            {4, 3}, Device::CPU);
        mesh.texcoords = Tensor::from_vector(
            std::vector<float>{0, 1, 1, 1, 1, 0, 0, 0},
            {4, 2}, Device::CPU);
        lfs::core::Material material;
        material.albedo_tex = 1;
        material.roughness = 0.7f;
        mesh.materials.push_back(material);
        mesh.texture_images.push_back({
            .pixels = {255, 32, 16, 255, 32, 255, 16, 255,
                       16, 32, 255, 255, 240, 220, 48, 255},
            .width = 2,
            .height = 2,
            .channels = 4,
        });
        return mesh;
    }

    ViewportMeshPassDesc makeDescription(const MeshData& mesh, const glm::mat4& projection,
                                         const bool wireframe) {
        return {
            .view_projection = projection,
            .camera_position = {0, 0, 0},
            .items = {ViewportMeshDrawItem{
                .mesh = &mesh,
                .light_dir = {0, 0, 1},
                .light_intensity = 0.8f,
                .ambient = 0.35f,
                .backface_culling = false,
                .wireframe_overlay = wireframe,
                .wireframe_color = {0.05f, 0.05f, 0.05f},
                .wireframe_width = 1.5f,
            }},
        };
    }

    void checkLayerContract(const MeshLayer& layer) {
        ASSERT_TRUE(layer.rgba.is_valid());
        ASSERT_TRUE(layer.view_depth.is_valid());
        EXPECT_EQ(layer.rgba.dtype(), DataType::Float32);
        EXPECT_EQ(layer.view_depth.dtype(), DataType::Float32);
        EXPECT_EQ(layer.rgba.shape(), (lfs::core::TensorShape{4, kHeight, kWidth}));
        EXPECT_EQ(layer.view_depth.shape(), (lfs::core::TensorShape{kHeight, kWidth}));

        const std::size_t pixels = std::size_t(kWidth) * kHeight;
        const float* rgba = layer.rgba.ptr<float>();
        const float* depth = layer.view_depth.ptr<float>();
        std::size_t covered = 0;
        std::size_t background = 0;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            if (rgba[3 * pixels + pixel] == 1.0f) {
                ++covered;
                EXPECT_NEAR(depth[pixel], kPlaneDepth, 3e-4f) << pixel;
            } else {
                ++background;
                EXPECT_EQ(rgba[3 * pixels + pixel], 0.0f) << pixel;
                EXPECT_TRUE(std::isinf(depth[pixel])) << pixel;
            }
        }
        EXPECT_GT(covered, pixels / 8);
        EXPECT_GT(background, pixels / 2);
    }

    struct Difference {
        float color_max = 0.0f;
        double color_mean = 0.0;
        float depth_relative_max = 0.0f;
        std::size_t coverage_mismatch = 0;
    };

    Difference difference(const MeshLayer& a, const MeshLayer& b) {
        const std::size_t pixels = std::size_t(kWidth) * kHeight;
        const float* ac = a.rgba.ptr<float>();
        const float* bc = b.rgba.ptr<float>();
        const float* ad = a.view_depth.ptr<float>();
        const float* bd = b.view_depth.ptr<float>();
        Difference result;
        std::size_t color_samples = 0;
        double color_sum = 0.0;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const bool a_covered = ac[3 * pixels + pixel] == 1.0f;
            const bool b_covered = bc[3 * pixels + pixel] == 1.0f;
            result.coverage_mismatch += a_covered != b_covered;
            if (!a_covered || !b_covered)
                continue;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const float value = std::abs(ac[channel * pixels + pixel] -
                                             bc[channel * pixels + pixel]);
                result.color_max = std::max(result.color_max, value);
                color_sum += value;
                ++color_samples;
            }
            result.depth_relative_max = std::max(
                result.depth_relative_max,
                std::abs(ad[pixel] - bd[pixel]) /
                    std::max(std::abs(ad[pixel]), 1e-6f));
        }
        result.color_mean = color_samples ? color_sum / color_samples : 0.0;
        return result;
    }

    TEST(TensorMeshOffscreenContracts, PlaneCoverageDepthWireframeAndBackendParity) {
        const glm::mat4 projection = glm::perspective(
            glm::radians(55.0f), float(kWidth) / kHeight, 0.1f, 100.0f);
        auto mesh = makeTexturedPlane();
        for (const bool wireframe : {false, true}) {
            std::vector<std::pair<GpuBackend, MeshLayer>> outputs;
            for (const GpuBackend backend : lfs::core::kCompiledGpuBackends) {
                if (!lfs::core::gpu_backend_available(backend))
                    continue;
                GpuBackendScope scope(backend);
                TensorMeshPass pass;
                auto rendered = pass.renderOffscreen(
                    makeDescription(mesh, projection, wireframe), projection,
                    kWidth, kHeight);
                ASSERT_TRUE(rendered) << lfs::core::gpu_backend_name(backend) << ": "
                                      << rendered.error().detail();
                checkLayerContract(*rendered);
                outputs.emplace_back(backend, std::move(*rendered));
            }
            ASSERT_FALSE(outputs.empty());
            for (std::size_t i = 1; i < outputs.size(); ++i) {
                const Difference diff = difference(outputs.front().second, outputs[i].second);
                std::cout << "tensor mesh parity wire=" << wireframe << " "
                          << lfs::core::gpu_backend_name(outputs.front().first) << " vs "
                          << lfs::core::gpu_backend_name(outputs[i].first)
                          << ": color max=" << diff.color_max
                          << " mean=" << diff.color_mean
                          << " depth rel max=" << diff.depth_relative_max
                          << " coverage mismatch=" << diff.coverage_mismatch << '\n';
                EXPECT_EQ(diff.coverage_mismatch, 0u);
                EXPECT_LE(diff.color_max, 2e-4f);
                EXPECT_LE(diff.depth_relative_max, 2e-5f);
            }
        }
    }

#if LFS_TEST_VULKAN_MESH_REFERENCE
    TEST(TensorMeshOffscreenContracts, MatchesHeadlessVulkanReferenceCoverageAndDepth) {
        if (!lfs::core::gpu_backend_available(GpuBackend::Metal))
            GTEST_SKIP() << "Metal tensor backend unavailable";
        const glm::mat4 projection = glm::perspective(
            glm::radians(55.0f), float(kWidth) / kHeight, 0.1f, 100.0f);
        auto mesh = makeTexturedPlane();
        const auto desc = makeDescription(mesh, projection, false);
        MeshLayer tensor_layer;
        {
            GpuBackendScope scope(GpuBackend::Metal);
            TensorMeshPass pass;
            auto rendered = pass.renderOffscreen(desc, projection, kWidth, kHeight);
            ASSERT_TRUE(rendered) << rendered.error().detail();
            tensor_layer = std::move(*rendered);
        }

        auto graphics = lfs::vis::createGraphicsContext();
        ASSERT_TRUE(graphics->initializeHeadless()) << graphics->lastError();
        lfs::vis::MeshOffscreenRenderer renderer;
        auto reference = renderer.render(*graphics, desc, projection, kWidth, kHeight);
        ASSERT_TRUE(reference) << reference.error().detail();
        const Difference diff = difference(tensor_layer, *reference);
        std::cout << "tensor Metal vs Vulkan MeshOffscreenRenderer: color max="
                  << diff.color_max << " mean=" << diff.color_mean
                  << " depth rel max=" << diff.depth_relative_max
                  << " coverage mismatch=" << diff.coverage_mismatch << '\n';
        const std::size_t pixels = std::size_t(kWidth) * kHeight;
        EXPECT_LE(diff.coverage_mismatch, pixels / 200);
        EXPECT_LE(diff.depth_relative_max, 5e-4f);
        EXPECT_LE(diff.color_mean, 0.08);
        renderer.shutdown();
        graphics->shutdown();
    }
#endif
} // namespace
