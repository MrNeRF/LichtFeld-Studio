/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include "core/mesh_data.hpp"
#include "core/scene.hpp"
#include "io/project_document.hpp"
#include "io/scene_chapter_adapter.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/mesh_offscreen_renderer.hpp"
#include "rendering/passes/vulkan_mesh_pass.hpp"
#include "window/vulkan_context.hpp"
#include <SDL3/SDL.h>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>

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

    // Opt-in production-volume benchmark; regular regressions remain tiny.
    TEST_F(CollisionOverlay, ProductionVolumeBenchmark) {
        const char* path = std::getenv("LFS_COLLISION_BENCHMARK_PROJECT");
        if (!path)
            GTEST_SKIP() << "Set LFS_COLLISION_BENCHMARK_PROJECT for a production-volume benchmark";
        renderer.setGpuTimingEnabled(true);
        auto document = lfs::io::project::ProjectDocument::open(path, {.defer_geometry_payloads = true});
        ASSERT_TRUE(document);
        lfs::core::Scene scene;
        auto shell = lfs::io::project::stage_scene_shell(document->scene_graph(), scene);
        ASSERT_TRUE(shell);
        auto retired = scene.commitRestoreStage(std::move(*shell));
        const lfs::core::SceneNode* collider = nullptr;
        for (const auto* node : scene.getNodes())
            if (node->collision) {
                collider = node;
                break;
            }
        ASSERT_NE(collider, nullptr);
        auto bounds = collider->collision->boundsMesh();
        geometry->vertices = lfs::core::Tensor::from_vector(bounds.vertices, {bounds.vertices.size() / 3, 3}, lfs::core::Device::CPU);
        geometry->indices = lfs::core::Tensor::from_vector(bounds.indices, {bounds.indices.size() / 3, 3}, lfs::core::Device::CPU);
        auto draw = item();
        draw.collision_volume = collider->collision.get();
        draw.collision_sharp = std::getenv("LFS_COLLISION_BENCHMARK_FAST") == nullptr;
        draw.model = lfs::rendering::dataWorldTransformToVisualizerWorld(scene.getWorldTransform(collider->id));
        const auto projection = glm::perspective(glm::radians(60.0f), 1920.f / 1080, .01f, 1000.f);
        lfs::vis::VulkanMeshPassParams params;
        params.camera_position = {-5, 3, -5};
        if (const char* view = std::getenv("LFS_COLLISION_BENCHMARK_VIEW")) {
            if (std::string_view(view) == "walk")
                params.camera_position = {0, 1, -3};
            else if (std::string_view(view) == "closeup")
                params.camera_position = {-1, 1, -1};
        }
        params.view_projection = projection * glm::lookAt(params.camera_position, glm::vec3(0), glm::vec3(0, 1, 0));
        params.items = {draw};
        if (const char* prefix = std::getenv("LFS_COLLISION_BENCHMARK_EXPORT")) {
            const auto& v = *draw.collision_volume;
            auto write = [&](const char* suffix, const auto& values) {
                std::ofstream file(std::string(prefix) + suffix, std::ios::binary);
                file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(values[0]));
            };
            write(".gpu.bin", v.gpuOccupancy());
            write(".tree.bin", v.tree);
            const auto inverse = glm::inverse(params.view_projection * draw.model);
            const auto camera = glm::vec3(glm::inverse(draw.model) * glm::vec4(params.camera_position, 1));
            nlohmann::json metadata{
                {"dimensions", {v.dimensions.x, v.dimensions.y, v.dimensions.z}},
                {"origin", {v.origin.x, v.origin.y, v.origin.z}},
                {"voxel_size", v.settings.voxel_size},
                {"root_size", v.tree_size},
                {"inverse_vp", std::vector<float>(&inverse[0][0], &inverse[0][0] + 16)},
                {"camera", {camera.x, camera.y, camera.z}},
                {"inside", v.occupied(v.cell(camera))},
                {"width", 1920},
                {"height", 1080}};
            std::ofstream(std::string(prefix) + ".json") << metadata.dump(2);
        }
        for (int i = 0; i < 10; ++i)
            ASSERT_TRUE(renderer.render(*context, params, projection, 1920, 1080));
        ASSERT_GT(renderer.lastGpuMilliseconds(), 0) << "GPU timestamps are unavailable";
        std::vector<double> times, gpu_times;
        for (int i = 0; i < 30; ++i) {
            const auto start = std::chrono::steady_clock::now();
            ASSERT_TRUE(renderer.render(*context, params, projection, 1920, 1080));
            times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            gpu_times.push_back(renderer.lastGpuMilliseconds());
        }
        if (const char* output = std::getenv("LFS_COLLISION_BENCHMARK_IMAGE")) {
            auto frame = renderer.render(*context, params, projection, 1920, 1080);
            ASSERT_TRUE(frame);
            std::ofstream file(output, std::ios::binary);
            const auto rgba = frame->rgba.to_vector();
            file.write(reinterpret_cast<const char*>(rgba.data()), rgba.size() * sizeof(float));
        }
        std::sort(times.begin(), times.end());
        std::sort(gpu_times.begin(), gpu_times.end());
        std::cout << "COLLISION_BENCHMARK project=" << path << " eye=" << params.camera_position.x << "," << params.camera_position.y << "," << params.camera_position.z << " resolution=1920x1080 median_ms=" << times[times.size() / 2] << " p90_ms=" << times[27] << " fast_requested=" << !draw.collision_sharp << " vrs_supported=" << context->hasFragmentShadingRate() << " gpu_median_ms=" << gpu_times[gpu_times.size() / 2] << " gpu_p90_ms=" << gpu_times[27] << std::endl;
    }

    TEST_F(CollisionOverlay, CornerCrossingsMatchIndependentRayBoxCoverage) {
        volume.settings.voxel_size = .05f;
        volume.origin = {-.371f, -.413f, -.397f};
        volume.dimensions = {16, 16, 16};
        volume.solid.assign(4096, 0);
        std::mt19937 random(181);
        for (auto& cell : volume.solid)
            cell = random() % 5 == 0;
        for (const auto eye : {glm::vec3(1.3f, .7f, 1.3f), glm::vec3(-1.1f, .3f, -1.1f)}) {
            const auto projection = glm::perspective(glm::radians(60.0f), 1.f, .01f, 100.f);
            lfs::vis::VulkanMeshPassParams params;
            params.camera_position = eye;
            params.view_projection = projection * glm::lookAt(eye, glm::vec3(0), glm::vec3(0, 1, 0));
            params.items = {item()};
            const auto frame = renderer.render(*context, params, projection, 64, 64);
            ASSERT_TRUE(frame);
            const auto inverse = glm::inverse(params.view_projection);
            const auto* pixels = frame->rgba.ptr<float>();
            for (int y = 0; y < 64; ++y)
                for (int x = 0; x < 64; ++x) {
                    const float nx = (x + .5f) / 32 - 1, ny = 1 - (y + .5f) / 32;
                    const auto near = inverse * glm::vec4(nx, ny, 0, 1), far = inverse * glm::vec4(nx, ny, 1, 1);
                    const glm::dvec3 ro = glm::vec3(near) / near.w;
                    const auto rd = glm::normalize(glm::dvec3(glm::vec3(far) / far.w) - ro);
                    bool expected = false;
                    for (int z = 0; z < 16 && !expected; ++z)
                        for (int b = 0; b < 16 && !expected; ++b)
                            for (int a = 0; a < 16 && !expected; ++a) {
                                if (!volume.occupied({a, b, z}))
                                    continue;
                                const glm::dvec3 lo = volume.origin + glm::vec3(a, b, z) * volume.settings.voxel_size;
                                const glm::dvec3 hi = volume.origin + glm::vec3(a + 1, b + 1, z + 1) * volume.settings.voxel_size;
                                const auto first = (lo - ro) / rd, last = (hi - ro) / rd;
                                const auto enter = glm::min(first, last), exit = glm::max(first, last);
                                expected = std::max({0.0, enter.x, enter.y, enter.z}) < std::min({exit.x, exit.y, exit.z});
                            }
                    EXPECT_EQ(pixels[y * 64 + x] > 0, expected) << x << "," << y;
                }
        }
    }

    TEST_F(CollisionOverlay, OffscreenAndEyePlaneBoundsAreConservative) {
        auto behind = item();
        behind.model = glm::translate(glm::mat4(1), {0, 0, 20});
        EXPECT_LT(glm::length(render({.25f, .25f, 5}, {0, 0, -1}, {behind})), .001f);
        auto lateral = item();
        lateral.model = glm::translate(glm::mat4(1), {100, 0, 0});
        EXPECT_LT(glm::length(render({.25f, .25f, 5}, {0, 0, -1}, {lateral})), .001f);
        auto fast = item();
        fast.collision_sharp = false;
        const auto orange = render({.25f, .25f, .25f}, {0, 0, -1}, {fast});
        EXPECT_GT(orange.r, .1f);
        EXPECT_NEAR(orange.g, orange.r * .35f, .001f);
    }

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

    TEST_F(CollisionOverlay, ObliqueSolidExitUsesHitCellInsteadOfSkippedLeafEntry) {
        volume.origin = glm::vec3(-4);
        volume.dimensions = {8, 8, 8};
        volume.solid.assign(512, 1);
        // Both rays exit the interior of the same voxel face. The long ray
        // crosses multiple cells inside a single compressed solid leaf.
        const glm::vec3 direction(0.3f, 0, -1);
        const auto long_ray = render({0.25f, 0.25f, 0.25f}, direction, {item()});
        const auto short_ray = render({1.45f, 0.25f, -3.75f}, direction, {item()});
        EXPECT_NEAR(short_ray.r, 0.18f, 0.005f);
        EXPECT_NEAR(long_ray.r, short_ray.r, 0.005f);
        EXPECT_NEAR(long_ray.g, long_ray.r * 0.35f, 0.001f);
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

    TEST_F(CollisionOverlay, OpacityAndCutawayAreLiveDisplayControls) {
        const glm::vec3 eye(0.25f, 3, 0.25f), down(0, -1, 0);
        const auto full = render(eye, down, {item()});
        auto faded = item();
        faded.collision_opacity = 0.25f;
        const auto quarter = render(eye, down, {faded});
        EXPECT_NEAR(quarter.r, full.r * 0.25f, 0.002f);
        faded.collision_opacity = 0;
        EXPECT_LT(glm::length(render(eye, down, {faded})), 0.001f);
        auto cut = item();
        cut.collision_slice = 0;
        EXPECT_LT(glm::length(render(eye, down, {cut})), 0.001f);
        cut.collision_slice = 0.5f;
        const auto free_space = render({0.25f, 1, 0.25f}, down, {cut});
        EXPECT_GT(free_space.r, 0.01f);
        EXPECT_NEAR(free_space.g, free_space.r, 0.002f);
        EXPECT_NEAR(free_space.b, free_space.r, 0.002f);
    }
} // namespace
