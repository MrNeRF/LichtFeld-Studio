/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_backend.hpp"
#include "metal_viewport_renderer.hpp"
#include "preferences.hpp"
#include <Python.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <random>
#include <stdexcept>
#include <unistd.h>

namespace {
    using namespace lfs;
    using core::Device;
    using core::Tensor;
    using Adapter = vis::VksplatViewportRenderer;
    using Shape = Adapter::SelectionMaskShape;
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    void precise_small_splats(vis::VulkanContext& context, vis::MetalViewportRenderer& renderer) {
        // Subpixel splats near real gesture boundaries exposed cancellation
        // that broad random selections did not exercise. Keep only geometry.
        const std::array<std::array<float, 10>, 8> geometry{{{.375725746f, -.741980493f, 2.271957159f, -4.215744972f, -4.617194176f, -5.498218536f, -.024384813f, .001928580f, .006719906f, .999678195f},
                                                             {-.608863533f, -.740784824f, 2.025210142f, -6.730025291f, -6.838163376f, -7.678555489f, .463676602f, -.004799385f, .006771366f, .885965645f},
                                                             {.539509296f, -.219786569f, 1.198791027f, -6.554687023f, -7.342196941f, -7.711044788f, .451674461f, -.006747358f, .014449934f, .892040253f},
                                                             {.543651283f, .204180837f, 1.060629249f, -6.813228130f, -7.043622017f, -7.382377625f, -.044447560f, .002716169f, .012134803f, .998934329f},
                                                             {.527110100f, .174101308f, 1.085454822f, -7.214101315f, -7.603221893f, -8.221177101f, -.686177135f, .017221944f, -.001918331f, .727228105f},
                                                             {.510118663f, .120262504f, 1.122224450f, -7.172321796f, -7.373991489f, -7.414416790f, -.689201117f, -.004335938f, .103279151f, .717158556f},
                                                             {.114320897f, .339419514f, 1.054803491f, -7.387386799f, -8.014912605f, -8.145614624f, -.704070985f, -.010196945f, .027925953f, .709506989f},
                                                             {.225137383f, .339107156f, 1.044668078f, -5.300498486f, -5.594229221f, -6.695940495f, -.032454044f, .002216667f, .006841690f, .999447346f}}};
        std::vector<float> means, scales, rotation;
        for (const auto& row : geometry) {
            means.insert(means.end(), row.begin(), row.begin() + 3);
            scales.insert(scales.end(), row.begin() + 3, row.begin() + 6);
            rotation.insert(rotation.end(), row.begin() + 6, row.end());
        }
        const size_t n = geometry.size();
        core::SplatData model(0, Tensor::from_vector(means, {n, 3}, Device::GPU), Tensor::zeros({n, 1, 3}, Device::GPU), {},
                              Tensor::from_vector(scales, {n, 3}, Device::GPU), Tensor::from_vector(rotation, {n, 4}, Device::GPU), Tensor::zeros({n, 1}, Device::GPU), 1.f);
        Adapter::SelectionMaskRequest request;
        request.frame_view.size = {600, 668};
        request.frame_view.translation = {-5.657000065f, 3.f, -5.657000065f};
        const float rows[]{-.707106769f, .248276189f, -.662086785f,
                           7.3059776e-9f, .936332166f, .351115495f, .707106829f, .248276159f, -.662086785f};
        request.frame_view.rotation = rendering::mat3FromRowMajor3x3(rows);
        request.gut = true;
        std::vector<glm::mat4> transforms{rendering::DATA_TO_VISUALIZER_WORLD_AXES_4};
        request.scene.model_transforms = &transforms;
        auto view = glm::mat4(glm::transpose(rendering::dataCameraToWorldFromVisualizerRotation(request.frame_view.rotation)));
        view[3] = glm::vec4(-glm::mat3(view) * request.frame_view.translation, 1);
        const glm::dmat4 matrix = glm::dmat4(view) * glm::dmat4(transforms[0]);
        const auto k = request.frame_view.getCameraIntrinsics();
        const auto project = [&](glm::dvec3 source) {
            const auto point = matrix * glm::dvec4(source, 1);
            return glm::dvec2(double(k.focal_x) * point.x / point.z + k.center_x, double(k.focal_y) * point.y / point.z + k.center_y);
        };
        for (auto shape : {Shape::Rectangle, Shape::Brush}) {
            request.shape = shape;
            request.primitives = shape == Shape::Rectangle ? std::vector<glm::vec4>{{40, 50, 390, 570}} : std::vector<glm::vec4>{{240, 320, 8100, 0}};
            auto output = renderer.buildSelectionMask(context, model, request);
            if (!output)
                throw std::runtime_error(format_for_developer(output.error()));
            const auto mask = output->cpu();
            for (size_t id = 0; id < n; ++id) {
                const auto& row = geometry[id];
                const glm::dvec3 source(row[0], row[1], row[2]);
                const auto basis = glm::mat3_cast(glm::normalize(glm::dquat(row[6], row[7], row[8], row[9])));
                const auto origin = project(source);
                glm::dvec2 delta(0);
                for (size_t axis = 0; axis < 3; ++axis) {
                    const auto offset = basis[axis] * (std::sqrt(.03) * std::exp(double(row[3 + axis])));
                    delta += project(source + offset) - origin;
                    delta += project(source - offset) - origin;
                }
                const auto center = origin + delta / .06 - .5;
                const bool expected = shape == Shape::Rectangle ? center.x >= 40 && center.x <= 390 && center.y >= 50 && center.y <= 570
                                                                : glm::length(center - glm::dvec2(240, 320)) <= 90;
                if (bool(mask.ptr<uint8_t>()[id]) != expected) {
                    std::fprintf(stderr, "Tiny GUT source=%zu shape=%u double center=%.9f,%.9f native=%u expected=%d\n", id, uint32_t(shape), center.x, center.y, mask.ptr<uint8_t>()[id], expected);
                    throw std::runtime_error("Subpixel GUT selection differs from independent double projection");
                }
            }
        }
    }
    void run(bool native_only) {
        core::GpuBackendScope scope(core::GpuBackend::Metal);
        vis::VulkanContext context;
        require(context.initHeadless(), context.lastError().c_str());
        constexpr size_t count = 1029;
        std::mt19937 rng(74549);
        std::uniform_real_distribution<float> random(-1.f, 1.f);
        std::vector<float> means(count * 3), scales(count * 3), rotations(count * 4), opacity(count);
        std::vector<int32_t> indices(count);
        std::vector<float> deleted(count);
        for (size_t n = 0; n < count; ++n) {
            means[n * 3] = random(rng) * 2.f;
            means[n * 3 + 1] = random(rng) * 1.3f;
            means[n * 3 + 2] = -3.8f + random(rng) * 1.2f;
            if (n % 29 == 0)
                means[n * 3 + 2] *= -1.f;
            for (size_t c = 0; c < 3; ++c)
                scales[n * 3 + c] = -3.2f + random(rng) * .7f;
            const float angle = random(rng) * .7f;
            rotations[n * 4] = std::cos(angle);
            rotations[n * 4 + 1] = std::sin(angle) * .6f;
            rotations[n * 4 + 2] = std::sin(angle) * .8f;
            opacity[n] = n % 13 ? random(rng) * 4.f : -100.f;
            indices[n] = int32_t(n % 3);
            deleted[n] = n % 37 == 0;
        }
        // The legacy editor clamps transform indices without a visibility
        // table, and rejects invalid indices when that table is present.
        indices[count - 1] = -1;
        indices[count - 2] = 6;
        core::SplatData model(0,
                              Tensor::from_vector(means, {count, 3}, Device::GPU),
                              Tensor::zeros({count, 1, 3}, Device::GPU), {},
                              Tensor::from_vector(scales, {count, 3}, Device::GPU),
                              Tensor::from_vector(rotations, {count, 4}, Device::GPU),
                              Tensor::from_vector(opacity, {count, 1}, Device::GPU), 1.f);
        model.deleted() = Tensor::from_vector(deleted, {count}, Device::GPU).to(core::DataType::Bool);
        model.notify_deleted_mask_changed();
        Adapter native_adapter, reference_adapter;
        vis::MetalViewportRenderer native;
        precise_small_splats(context, native);
        size_t cases = 0, hits = 0, ring_hits = 0;
        for (bool half : {false, true}) {
            if (half) {
                model.scaling_raw() = model.scaling_raw().to(core::DataType::Float16);
                model.rotation_raw() = model.rotation_raw().to(core::DataType::Float16);
                model.opacity_raw() = model.opacity_raw().to(core::DataType::Float16);
            }
            // The legacy selection binding requires expanded Float32 geometry
            // even when the viewer renders packed half attributes. Compare the
            // exact decoded half values without altering that production path.
            auto expanded = model.clone();
            expanded.scaling_raw() = expanded.scaling_raw().to(core::DataType::Float32);
            expanded.rotation_raw() = expanded.rotation_raw().to(core::DataType::Float32);
            expanded.opacity_raw() = expanded.opacity_raw().to(core::DataType::Float32);
            for (int camera = 0; camera < 3; ++camera)
                for (bool gut : {false, true}) {
                    if (camera == 2 && !gut)
                        continue;
                    for (bool affine : {false, true})
                        for (bool mip : {false, true})
                            for (const auto shape : {Shape::Brush, Shape::Rectangle, Shape::Polygon, Shape::Ring}) {
                                Adapter::SelectionMaskRequest request;
                                request.frame_view.size = {160, 112};
                                request.frame_view.translation = {.07f, -.13f, .19f};
                                request.frame_view.rotation = glm::mat3(glm::rotate(glm::mat4(1), .13f, glm::vec3(0, 1, 0)));
                                request.frame_view.orthographic = camera == 1;
                                request.frame_view.ortho_scale = 35.f;
                                request.gut = gut;
                                request.equirectangular = camera == 2;
                                request.mip_filter = mip;
                                request.shape = shape;
                                request.ring_width = .08f;
                                request.scene.transform_indices = std::make_shared<Tensor>(Tensor::from_vector(indices, {count}, Device::GPU));
                                std::vector<glm::mat4> transforms(3, glm::mat4(1));
                                if (affine) {
                                    transforms[0] = glm::translate(glm::mat4(1), glm::vec3(.17f, .08f, -.2f));
                                    transforms[1] = glm::scale(glm::rotate(glm::mat4(1), .24f, glm::vec3(1, 0, 0)), glm::vec3(.8f, 1.2f, 1));
                                    transforms[2][1][0] = .2f;
                                    transforms[2][3][0] = -.21f;
                                    request.scene.node_visibility_mask = {true, false, true};
                                }
                                request.scene.model_transforms = &transforms;
                                request.primitives = shape == Shape::Rectangle
                                                         ? std::vector<glm::vec4>{{29.25f, 24.25f, 103.25f, 86.25f}, {4.75f, 46.25f, 23.75f, 69.25f}}
                                                         : std::vector<glm::vec4>{{64.25f, 47.75f, shape == Shape::Ring ? 1.7f : 225.f, 0},
                                                                                  {103.75f, 63.25f, shape == Shape::Ring ? 2.3f : 121.f, 0}};
                                request.polygon_vertices = {{25.25f, 19.75f}, {121.75f, 29.25f}, {73.25f, 54.75f}, {129.25f, 93.25f}, {27.75f, 84.25f}};
                                uint32_t direct_id = ~0u, adapter_id = ~0u, reference_id = ~0u;
                                request.picked_ring_id_out = &direct_id;
                                require(vis::MetalViewportRenderer::supportsSelection(model, request), "Native query unexpectedly unsupported");
                                auto result = native.buildSelectionMask(context, model, request);
                                if (!result)
                                    throw std::runtime_error(format_for_developer(result.error()));
                                // Consume the native output through the shared editor's
                                // Metal tensor kernels before any host/GPU wait.
                                const auto editor_result = result->to(core::DataType::UInt8).cpu();
                                vis::UserPreferences::instance().setViewerBackend(rendering::ViewerBackend::Metal);
                                request.picked_ring_id_out = &adapter_id;
                                auto via_adapter = native_adapter.buildSelectionMask(context, model, request, true);
                                if (!via_adapter)
                                    throw std::runtime_error(via_adapter.error());
                                const auto adapter_cpu = via_adapter->cpu();
                                auto reference_cpu = editor_result;
                                reference_id = direct_id;
                                if (!native_only) {
                                    vis::UserPreferences::instance().setViewerBackend(rendering::ViewerBackend::Vulkan);
                                    request.picked_ring_id_out = &reference_id;
                                    auto reference = reference_adapter.buildSelectionMask(context, half ? expanded : model, request, true);
                                    if (!reference)
                                        throw std::runtime_error(reference.error());
                                    reference_cpu = reference->cpu();
                                }
                                for (size_t n = 0; n < count; ++n) {
                                    const auto actual = editor_result.ptr<uint8_t>()[n];
                                    if (actual != reference_cpu.ptr<uint8_t>()[n] || actual != adapter_cpu.ptr<uint8_t>()[n]) {
                                        std::fprintf(stderr, "Selection mismatch source=%zu shape=%u camera=%d gut=%d half=%d affine=%d mip=%d native=%u Vulkan=%u adapter=%u picks=%u/%u\n", n, uint32_t(shape), camera, gut, half, affine, mip, actual, reference_cpu.ptr<uint8_t>()[n], adapter_cpu.ptr<uint8_t>()[n], direct_id, reference_id);
                                        throw std::runtime_error("Native editor selection differs from Vulkan");
                                    }
                                    hits += actual != 0;
                                }
                                require(direct_id == reference_id && direct_id == adapter_id, "Ring picking identity differs from Vulkan");
                                ring_hits += shape == Shape::Ring && direct_id != ~0u;
                                ++cases;
                            }
                }
        }
        require(hits > 1000 && ring_hits > 8, "Selection fixtures did not exercise visible coverage and ring picks");
        // Selection output is handed directly to existing deletion/undo kernels;
        // no repacking, native wait or CPU mask staging belongs in that path.
        model.clear_deleted();
        Adapter::SelectionMaskRequest deletion;
        deletion.frame_view.size = {160, 112};
        deletion.shape = Shape::Rectangle;
        deletion.primitives = {{0, 0, 160, 112}};
        auto mask = native.buildSelectionMask(context, model, deletion);
        if (!mask)
            throw std::runtime_error(format_for_developer(mask.error()));
        auto undo = model.soft_delete(*mask);
        auto deleted_cpu = model.deleted().cpu();
        auto mask_cpu = mask->cpu();
        for (size_t n = 0; n < count; ++n)
            require(deleted_cpu.ptr<uint8_t>()[n] == mask_cpu.ptr<uint8_t>()[n], "Native selection was not ordered before deletion");
        model.undelete(undo);
        deleted_cpu = model.deleted().cpu();
        for (size_t n = 0; n < count; ++n)
            require(!deleted_cpu.ptr<uint8_t>()[n], "Native selection undo did not restore deletion state");
        std::printf("%zu %s selection adapter cases, %zu selected IDs, %zu ring picks; GPU deletion/undo ordering passed.\n", cases, native_only ? "native" : "native/Vulkan", hits, ring_hits);
    }
} // namespace
int main(int argc, char** argv) {
    @autoreleasepool {
        if (!core::gpu_backend_available(core::GpuBackend::Metal))
            return 77;
        const auto home = std::filesystem::temp_directory_path() / ("lichtfeld-metal-selection-contracts-" + std::to_string(getpid()));
        setenv("LFS_HOME", home.c_str(), 1);
        unsetenv("LFS_SAFE_MODE");
        Py_Initialize();
        try {
            run(argc == 2 && std::string_view(argv[1]) == "--native-only");
            return 0;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "%s\n", error.what());
            return 1;
        }
    }
}
