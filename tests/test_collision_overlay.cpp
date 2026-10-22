/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include "core/mesh_data.hpp"
#include "rendering/mesh_offscreen_renderer.hpp"
#include "rendering/passes/vulkan_mesh_pass.hpp"
#include "window/vulkan_context.hpp"
#include <SDL3/SDL.h>
#include <array>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

namespace {
    class CollisionOverlay : public ::testing::Test {
    protected:
        SDL_Window* window = nullptr;
        std::unique_ptr<lfs::vis::VulkanContext> context;
        lfs::vis::MeshOffscreenRenderer renderer;
        lfs::core::CollisionVolume volume;
        std::shared_ptr<lfs::core::MeshData> geometry;

        void SetUp() override {
            if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
                GTEST_SKIP() << SDL_GetError();
            window = SDL_CreateWindow("Collision overlay test", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
            if (!window)
                GTEST_SKIP() << SDL_GetError();
            context = std::make_unique<lfs::vis::VulkanContext>();
            if (!context->init(window, 64, 64))
                GTEST_SKIP() << context->lastError();
            volume.settings.voxel_size = 1;
            volume.origin = glm::vec3(-2);
            volume.dimensions = {4, 4, 4};
            volume.solid.assign(64, 1);
            auto surface = volume.mesh(lfs::core::CollisionShape::Faces);
            geometry = std::make_shared<lfs::core::MeshData>();
            geometry->vertices = lfs::core::Tensor::from_vector(
                surface.vertices, {surface.vertices.size() / 3, 3}, lfs::core::Device::CPU);
            geometry->indices = lfs::core::Tensor::from_vector(
                surface.indices, {surface.indices.size() / 3, 3}, lfs::core::Device::CPU);
            geometry->compute_normals();
        }

        void TearDown() override {
            renderer.shutdown();
            context.reset();
            if (window)
                SDL_DestroyWindow(window);
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }

        lfs::vis::VulkanMeshDrawItem item() const {
            return {.mesh = geometry.get(), .collision_voxel_boxes = true, .collision_volume = &volume};
        }

        glm::vec3 render(const glm::vec3 eye, const glm::vec3 direction,
                         std::vector<lfs::vis::VulkanMeshDrawItem> items) {
            const auto up = std::abs(direction.y) > 0.9f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
            const auto projection = glm::perspective(glm::radians(60.0f), 1.0f, 0.01f, 100.0f);
            lfs::vis::VulkanMeshPassParams params;
            params.camera_position = eye;
            params.view_projection = projection * glm::lookAt(eye, eye + direction, up);
            params.items = std::move(items);
            const auto image = renderer.render(*context, params, projection, 64, 64);
            EXPECT_TRUE(image) << (image ? "" : image.error());
            if (!image)
                return glm::vec3(0);
            const auto* pixels = image->rgba.ptr<float>();
            constexpr size_t center = 32 * 64 + 32;
            return {pixels[center], pixels[4096 + center], pixels[8192 + center]};
        }
    };

    TEST_F(CollisionOverlay, CompletelySolidVolumeShowsOrangeBoundingExitInEveryDirection) {
        for (const auto direction : std::array{
                 glm::vec3(1, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(0, 1, 0),
                 glm::vec3(0, -1, 0), glm::vec3(0, 0, 1), glm::vec3(0, 0, -1)}) {
            const auto color = render(glm::vec3(0.25f), direction, {item()});
            EXPECT_GT(color.r, 0.1f);
            EXPECT_NEAR(color.g, color.r * 0.35f, 0.001f);
            EXPECT_NEAR(color.b, color.r * 0.08f, 0.001f);
        }
    }

    TEST_F(CollisionOverlay, OutsideSurfaceIsGrayAndSelectedSurfaceIsBlue) {
        const auto gray = render({0.25f, 0.25f, 5}, {0, 0, -1}, {item()});
        EXPECT_GT(gray.r, 0.1f);
        EXPECT_NEAR(gray.r, gray.g, 0.001f);
        EXPECT_NEAR(gray.r, gray.b, 0.001f);
        auto selected = item();
        selected.is_emphasized = true;
        const auto blue = render({0.25f, 0.25f, 5}, {0, 0, -1}, {selected});
        EXPECT_GT(blue.b, blue.r);
        EXPECT_NEAR(blue.r, blue.b * 0.65f, 0.001f);
        EXPECT_NEAR(blue.g, blue.b * 0.85f, 0.001f);
    }

    TEST_F(CollisionOverlay, IndependentOverlaysCurrentlyCompositeInDrawOrder) {
        auto near = item();
        auto far = item();
        far.model = glm::translate(glm::mat4(1), glm::vec3(0, 0, -5));
        far.is_emphasized = true;
        const auto a = render({0.25f, 0.25f, 5}, {0, 0, -1}, {near, far});
        const auto b = render({0.25f, 0.25f, 5}, {0, 0, -1}, {far, near});
        EXPECT_GT(glm::length(a - b), 0.005f);
    }
    TEST_F(CollisionOverlay, SparseTraversalSkipsEmptyBoundsAndFindsBothDistantComponents) {
        lfs::core::CollisionSettings settings;
        settings.applyPreset(lfs::core::CollisionPreset::Object);
        settings.voxel_size = 0.25f;
        const std::array source{lfs::core::CollisionGaussian{{0, 0, 0}, glm::mat3(1.0f), 1},
                                lfs::core::CollisionGaussian{{1000, 0, 0}, glm::mat3(1.0f), 1}};
        volume = lfs::core::generateCollision(source, settings);
        ASSERT_TRUE(volume.solid.empty());
        for (float x : {0.25f, 1000.25f}) {
            const auto color = render({x, 0.25f, 5}, {0, 0, -1}, {item()});
            EXPECT_GT(color.r, 0.1f);
            EXPECT_NEAR(color.r, color.g, 0.001f);
            EXPECT_NEAR(color.r, color.b, 0.001f);
        }
        const auto gap = render({500.25f, 0.25f, 5}, {0, 0, -1}, {item()});
        EXPECT_LT(glm::length(gap), 0.001f);
    }
    TEST_F(CollisionOverlay, SparseBrickBoundaryAndTransformedInsideExit) {
        lfs::core::CollisionSettings settings;
        settings.applyPreset(lfs::core::CollisionPreset::Object);
        settings.voxel_size = 0.1f;
        const lfs::core::CollisionGaussian source{{0, 0, 0}, glm::mat3(0.5f), 1};
        volume = lfs::core::generateCollision(std::span(&source, 1), settings);
        auto transformed = item();
        transformed.model = glm::translate(glm::mat4(1), glm::vec3(10, 3, 4)) *
                            glm::scale(glm::mat4(1), glm::vec3(2));
        const auto color = render({10.05f, 3.05f, 4.05f}, {0, 0, -1}, {transformed});
        EXPECT_GT(color.r, 0.1f);
        EXPECT_NEAR(color.g, color.r * 0.35f, 0.001f);
        EXPECT_NEAR(color.b, color.r * 0.08f, 0.001f);
    }
} // namespace
