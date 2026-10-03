/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "device_requirements.hpp"
#include "core/tensor_backend.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"
#include "metal_viewport_renderer.hpp"
#include "point_cloud_vulkan_renderer.hpp"
#include "preferences.hpp"
#include "vksplat_viewport_renderer.hpp"
#include "vulkan_scene_output.hpp"
#import <Metal/Metal.h>
#include <Python.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unistd.h>

namespace {
    using namespace lfs;
    using Clock = std::chrono::steady_clock;
    using Json = nlohmann::json;
    constexpr vis::RenderTargetId kTarget{1};
    struct Options {
        size_t count = 100000, fixture_count = 512;
        int width = 1280, height = 720, warmup = 12, samples = 40, tone = 0;
        float exposure = 1.f;
        std::string output, images, overlay, input, camera, fixture_format;
        core::GpuBackend tensor_backend = core::GpuBackend::Metal;
        bool scalar_depth_reference = false, depth_boundary = false, depth_diagnostics = false, profile_gpu = false, deterministic_reference = false, frustum = false, verify_parity = false, saturation = false, input_fixture = false, transparent = false, depth_gray = false;
        bool mip = false, ortho = false, depth = false, export_scale = false, gut = false, equirect = false, subregion = false, near = false, portal = false, portal_tone = false, lod = false, lod_logical = false, lod_weights = false, lod_debug = false, spark = false, gpu_lod = false, gpu_lod_budget = false;
    };
    Options options(int argc, char** argv) {
        Options o;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--smoke") {
                o.count = 512;
                o.width = 128;
                o.height = 96;
                o.warmup = 6;
                o.samples = 4;
                o.verify_parity = true;
                continue;
            }
            if (arg == "--portal" || arg == "--portal_tone") {
                o.portal = true;
                o.portal_tone = arg == "--portal_tone";
                if (o.portal_tone) {
                    o.tone = 4;
                    o.exposure = 1.6f;
                }
                continue;
            }
            if (arg == "--lod" || arg == "--lod_logical" || arg == "--lod_weights" || arg == "--lod_debug") {
                o.lod = true;
                o.lod_logical = arg == "--lod_logical";
                o.lod_weights = arg == "--lod_weights";
                o.lod_debug = arg == "--lod_debug";
                continue;
            }
            if (arg == "--gpu_lod" || arg == "--gpu_lod_budget") {
                o.gpu_lod = true;
                o.gpu_lod_budget = arg == "--gpu_lod_budget";
                continue;
            }
            if (arg == "--spark") {
                o.spark = o.lod = true;
                continue;
            }
            if (arg == "--equirect") {
                o.equirect = o.gut = true;
                continue;
            }
            if (arg == "--subregion") {
                o.subregion = true;
                continue;
            }
            if (arg == "--near") {
                o.near = true;
                continue;
            }
            if (arg == "--gut") {
                o.gut = true;
                continue;
            }
            if (arg == "--export_scale") {
                o.export_scale = true;
                continue;
            }
            if (arg == "--profile") {
                o.profile_gpu = true;
                continue;
            }
            if (arg == "--deterministic-reference") {
                o.deterministic_reference = true;
                continue;
            }
            if (arg == "--verify-parity") {
                o.verify_parity = true;
                continue;
            }
            if (arg == "--frustum") {
                o.frustum = true;
                continue;
            }
            if (arg == "--saturation") {
                o.saturation = true;
                continue;
            }
            if (arg == "--mip") {
                o.mip = true;
                continue;
            }
            if (arg == "--ortho") {
                o.ortho = true;
                continue;
            }
            if (arg == "--transparent") {
                o.transparent = true;
                continue;
            }
            if (arg == "--depth-gray") {
                o.depth = o.depth_gray = true;
                continue;
            }
            if (arg == "--scalar-depth-reference") {
                o.depth = o.scalar_depth_reference = true;
                continue;
            }
            if (arg == "--depth-boundary") {
                o.depth = o.depth_boundary = true;
                o.count = 1025;
                continue;
            }
            if (arg == "--depth-diagnostics") {
                o.depth = o.depth_diagnostics = true;
                continue;
            }
            if (arg == "--depth") {
                o.depth = true;
                continue;
            }
            if (++i == argc)
                throw std::runtime_error("Missing argument for " + arg);
            const std::string value = argv[i];
            if (arg == "--tensor-backend") {
                if (value != "metal" && value != "vulkan")
                    throw std::invalid_argument("Benchmark tensor backend must be metal or vulkan");
                o.tensor_backend = value == "metal" ? core::GpuBackend::Metal : core::GpuBackend::Vulkan;
                continue;
            }
            if (arg == "--output") {
                o.output = value;
                continue;
            }
            if (arg == "--input" || arg == "--input-fixture") {
                if (!o.input.empty())
                    throw std::runtime_error("Only one benchmark input is allowed");
                o.input = value;
                o.input_fixture = arg == "--input-fixture";
                continue;
            }
            if (arg == "--fixture-format") {
                if (value != "ply" && value != "spz3" && value != "spz4" && value != "glb" && value != "sog" && value != "ssog" && value != "rad")
                    throw std::invalid_argument("Unknown benchmark fixture format");
                o.fixture_format = value;
                continue;
            }
            if (arg == "--camera") {
                o.camera = value;
                continue;
            }
            if (arg == "--images") {
                o.images = value;
                continue;
            }
            if (arg == "--overlay") {
                if (value != "selection" && value != "preview" && value != "crop" && value != "ellipsoid" && value != "window" && value != "markers" && value != "flash" && value != "affine" && value != "rings" && value != "selected-rings")
                    throw std::runtime_error("Unknown overlay fixture");
                o.overlay = value;
                continue;
            }
            size_t used = 0;
            if (arg == "--exposure") {
                o.exposure = std::stof(value, &used);
                if (used != value.size() || !std::isfinite(o.exposure) || o.exposure <= 0 || o.exposure > 32)
                    throw std::runtime_error("Invalid benchmark exposure");
                continue;
            }
            if (arg == "--tone") {
                const auto tone = std::stoll(value, &used);
                if (used != value.size() || tone < 0 || tone > 6)
                    throw std::runtime_error("Invalid benchmark tone operator");
                o.tone = int(tone);
                continue;
            }
            const auto number = std::stoll(value, &used);
            if (used != value.size() || number < 1 || number > 10000000)
                throw std::runtime_error("Invalid value for " + arg);
            if (arg == "--count")
                o.count = number;
            else if (arg == "--fixture-count")
                o.fixture_count = number;
            else if (arg == "--width")
                o.width = number;
            else if (arg == "--height")
                o.height = number;
            else if (arg == "--warmup")
                o.warmup = number;
            else if (arg == "--samples")
                o.samples = number;
            else
                throw std::runtime_error("Unknown option " + arg);
        }
        if (o.width > 4096 || o.height > 4096 || o.count > 1000000 || o.fixture_count > 1000000 || o.samples > 10000 || o.warmup > 1000)
            throw std::runtime_error("Benchmark reservation limit exceeded");
        if (!o.input_fixture && o.fixture_count != 512)
            throw std::runtime_error("--fixture-count requires --input-fixture");
        if (!o.camera.empty() && o.input.empty())
            throw std::runtime_error("--camera requires --input");
        if (!o.input.empty() && (o.lod || o.gpu_lod || o.spark || o.near || o.frustum || o.saturation || !o.overlay.empty()))
            throw std::runtime_error("Synthetic hierarchy/geometry/overlay fixtures cannot be combined with --input");
        return o;
    }
    core::SplatData scene(size_t count, int degree, float rest_amplitude = .1f, bool panorama = false, bool near = false, const Options* reference_cut = nullptr, bool spark = false, bool gpu_lod = false, bool saturation = false, const Options* frustum = nullptr, const Options* depth_boundary = nullptr) {
        std::mt19937 random(1939);
        std::uniform_real_distribution<float> unit(0.f, 1.f);
        std::vector<float> means(count * 3), sh0(count * 3), scales(count * 3), rotation(count * 4, 0), opacity(count);
        std::vector<float> rest(degree ? count * 45 : 0);
        for (size_t i = 0; i < count; ++i) {
            means[3 * i] = (unit(random) - .5f) * 3.6f;
            means[3 * i + 1] = (unit(random) - .5f) * 2.f;
            means[3 * i + 2] = -4.f - unit(random) * 4.f;
            for (int c = 0; c < 3; ++c) {
                sh0[3 * i + c] = (unit(random) - .5f) * 2.f;
                scales[3 * i + c] = -4.8f + unit(random) * .4f;
            }
            if (panorama) {
                const float azimuth = (unit(random) - .5f) * float(2 * M_PI);
                const float elevation = std::asin(2.f * unit(random) - 1.f);
                const float radius = 3.f + unit(random);
                means[3 * i] = radius * std::sin(azimuth) * std::cos(elevation);
                means[3 * i + 1] = radius * std::sin(elevation);
                means[3 * i + 2] = -radius * std::cos(azimuth) * std::cos(elevation);
                // Both sides of the longitude seam, including negative view Z.
                if (i < 16) {
                    means[3 * i] = (i % 2 ? 1.f : -1.f) * (.02f + .01f * float(i / 2));
                    means[3 * i + 1] = (float(i / 2) - 3.5f) * .15f;
                    means[3 * i + 2] = 3.f;
                }
                for (int c = 0; c < 3; ++c)
                    scales[3 * i + c] = -2.8f + unit(random) * .4f;
            }
            if (near) {
                means[3 * i] *= .006f;
                means[3 * i + 1] *= .006f;
                means[3 * i + 2] = -.03f - .05f * unit(random);
                for (int c = 0; c < 3; ++c)
                    scales[3 * i + c] -= 2.5f;
            }
            rotation[4 * i] = 1.f;
            opacity[i] = 1.f + unit(random);
            if (spark)
                opacity[i] = .2f + .4f * float(i % 5);
        }
        for (auto& value : rest)
            value = (unit(random) - .5f) * rest_amplitude;
        if (saturation) {
            if (count < 2)
                throw std::runtime_error("Saturation fixture requires two splats");
            std::fill(opacity.begin(), opacity.end(), -100.f);
            std::fill(rest.begin(), rest.end(), 0.f);
            // The second splat leaves T=0.05*0.001 < 1e-4 at the center.
            // Vulkan GUT's legacy chain omits that saturating color; the GS
            // macro chain retains it. Distinct colors expose that difference.
            for (size_t n = 0; n < 2; ++n) {
                means[3 * n] = means[3 * n + 1] = 0;
                means[3 * n + 2] = -4.f - float(n);
                opacity[n] = n ? 12.f : std::log(19.f);
                for (size_t c = 0; c < 3; ++c) {
                    scales[3 * n + c] = std::log(3.f);
                    const float color = n ? .8f : c == 0 ? .7f
                                                         : .1f;
                    sh0[3 * n + c] = (color - .5f) / .2820947917738781f;
                }
            }
        }
        if (frustum) {
            if (count < 2)
                throw std::invalid_argument("Frustum fixture requires at least two splats");
            std::fill(opacity.begin(), opacity.end(), -100.f);
            std::fill(rest.begin(), rest.end(), 0.f);
            rendering::FrameView camera;
            camera.size = {frustum->width, frustum->height};
            camera.orthographic = frustum->ortho;
            camera.ortho_scale = 32;
            const auto k = camera.getCameraIntrinsics();
            // A near red Gaussian has an off-frustum mean but enough support
            // to cover the image. The farther green Gaussian remains admitted.
            for (size_t i = 0; i < 2; ++i) {
                const float z = i ? 50.f : 3.f;
                means[3 * i] = i ? 0.f : ((1.2f * frustum->width + 3.f - k.center_x) / k.focal_x) * (frustum->ortho ? 1.f : z);
                means[3 * i + 1] = 0;
                means[3 * i + 2] = -z;
                opacity[i] = 3;
                const float colors[2][3] = {{.9f, .05f, .05f}, {.05f, .7f, .1f}};
                for (size_t c = 0; c < 3; ++c) {
                    scales[3 * i + c] = std::log(40.f);
                    sh0[3 * i + c] = (colors[i][c] - .5f) / .2820947917738781f;
                }
            }
        }
        if (depth_boundary) {
            // HiGS uses 1024-source macro batches. Batch one rounds T to .5
            // in FP16 while exact T is still above .5. The last Gaussian in
            // the next batch must establish the median, not leave FAR_DEPTH.
            if (count != 1025)
                throw std::invalid_argument("Depth boundary fixture requires 1025 splats");
            rendering::FrameView camera;
            camera.size = {depth_boundary->width, depth_boundary->height};
            const auto k = camera.getCameraIntrinsics();
            std::fill(rest.begin(), rest.end(), 0.f);
            for (size_t n = 0; n < count; ++n) {
                const bool filler = n > 0 && n + 1 < count;
                const float z = n == 0 ? 4.f : filler ? 4.1f
                                                      : 6.f;
                // Projection samples integer pixels after its half-pixel shift.
                means[3 * n] = (filler ? 16.5f : .5f) / k.focal_x * z;
                means[3 * n + 1] = -.5f / k.focal_y * z;
                means[3 * n + 2] = -z;
                const float alpha = n == 0 ? .49995f : .01f;
                opacity[n] = std::log(alpha / (1 - alpha));
                for (size_t c = 0; c < 3; ++c) {
                    scales[3 * n + c] = std::log(.005f);
                    sh0[3 * n + c] = 0;
                }
            }
        }
        if (reference_cut) {
            // The legacy GUT gather reads compact slots as source IDs and has
            // no LOD indirection bindings. Keep that production path untouched;
            // construct the same resident cut in a full source-layout reference.
            // Zero opacity hides unselected nodes; no attributes are repacked.
            for (size_t source = 0; source < count; ++source) {
                const size_t ordinal = count - 1 - source;
                if (ordinal % 2) {
                    opacity[source] = -100.f;
                    continue;
                }
                if (reference_cut->lod_weights) {
                    const float weight = ordinal % 6 == 0 ? 0.f : ordinal % 6 == 2 ? .3f
                                                                                   : 1.f;
                    const float alpha = weight / (1 + std::exp(-opacity[source]));
                    opacity[source] = alpha > 0 ? std::log(alpha / (1 - alpha)) : -100.f;
                }
                if (reference_cut->lod_debug) {
                    constexpr float palette[5][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1}};
                    for (size_t c = 0; c < 3; ++c)
                        if (palette[(ordinal / 2) % 5][c] == 0) {
                            sh0[source * 3 + c] = -.5f / .2820947917738781f;
                            if (degree)
                                for (size_t k = 0; k < 15; ++k)
                                    rest[(source * 15 + k) * 3 + c] = 0;
                        }
                }
            }
        }
        using core::Device;
        using core::Tensor;
        auto model = core::SplatData(degree,
                                     Tensor::from_vector(means, {count, 3}, Device::GPU),
                                     Tensor::from_vector(sh0, {count, 1, 3}, Device::GPU),
                                     degree ? Tensor::from_vector(rest, {count, 15, 3}, Device::GPU) : Tensor{},
                                     Tensor::from_vector(scales, {count, 3}, Device::GPU),
                                     Tensor::from_vector(rotation, {count, 4}, Device::GPU),
                                     Tensor::from_vector(opacity, {count, 1}, Device::GPU), 1.f);
        if (degree) {
            (void)model.apply_shN_value_quant();
            if (!model.shN_value_quantized())
                throw std::runtime_error("SH3 fixture must use production Q16 storage");
        }
        if (spark || gpu_lod) {
            model.lod_tree = std::make_unique<core::SplatLodTree>();
            auto& tree = *model.lod_tree;
            tree.lod_opacity_encoded = spark;
            tree.child_count.assign(count, 0);
            tree.child_start.assign(count, 0);
            tree.lod_level.assign(count, 0);
            tree.centers.resize(count);
            tree.sizes.resize(count);
            for (size_t n = 0; n < count; ++n) {
                tree.centers[n] = {means[n * 3], means[n * 3 + 1], means[n * 3 + 2]};
                tree.sizes[n] = 2.f * std::exp(std::max({scales[n * 3], scales[n * 3 + 1], scales[n * 3 + 2]}));
            }
            if (gpu_lod) {
                const size_t groups = (count + 59999) / 60000;
                if (count <= groups + 1)
                    throw std::runtime_error("GPU LOD fixture requires leaves");
                tree.child_start[0] = 1;
                tree.child_count[0] = uint16_t(groups);
                tree.lod_level[0] = 2;
                tree.sizes[0] = 8;
                const size_t leaves = count - groups - 1;
                size_t first = groups + 1;
                for (size_t g = 0; g < groups; ++g) {
                    const size_t length = leaves / groups + (g < leaves % groups);
                    tree.child_start[g + 1] = uint32_t(first);
                    tree.child_count[g + 1] = uint16_t(length);
                    tree.lod_level[g + 1] = 1;
                    tree.sizes[g + 1] = 4;
                    first += length;
                }
            }
        }
        return model;
    }
    void wait(vis::VulkanContext& context, const vis::VksplatViewportRenderer::RenderResult& frame) {
        if (!frame.image || !frame.completion_semaphore || !frame.completion_value)
            throw std::runtime_error("Missing real GPU frame completion");
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        const VkSemaphore completion = vis::vulkanSceneTimeline(frame.completion_semaphore);
        info.pSemaphores = &completion;
        info.pValues = &frame.completion_value;
        if (vkWaitSemaphores(context.device(), &info, 30'000'000'000ull) != VK_SUCCESS)
            throw std::runtime_error("GPU frame completion timed out or failed");
    }
    Json statistics(const std::vector<double>& raw) {
        auto sorted = raw;
        std::sort(sorted.begin(), sorted.end());
        const auto percentile = [&](double p) {
            const double index = p * (sorted.size() - 1);
            const size_t lo = size_t(index), hi = std::min(lo + 1, sorted.size() - 1);
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo);
        };
        return {{"samples_ms", raw}, {"median_ms", percentile(.5)}, {"p95_ms", percentile(.95)}};
    }
    Json quality(const core::Tensor& native, const core::Tensor& reference, const glm::vec3 background, bool depth_view) {
        if (native.numel() != reference.numel())
            throw std::runtime_error("Image shapes differ");
        double sum = 0, squares = 0, maximum = 0, native_signal = 0, reference_signal = 0;
        const auto a = native.ptr<float>(), b = reference.ptr<float>();
        for (size_t i = 0; i < native.numel(); ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
                throw std::runtime_error("Non-finite benchmark image");
            const double error = std::abs(double(a[i]) - b[i]);
            sum += error;
            squares += error * error;
            maximum = std::max(maximum, error);
            native_signal += std::abs(a[i] - background[i % 3]);
            reference_signal += std::abs(b[i] - background[i % 3]);
        }
        // Detect missing/background-only frames without imposing a hardware-dependent speed gate.
        if (native_signal < .1 || reference_signal < .1)
            throw std::runtime_error("Benchmark published an empty image");
        double valid_depth_max = 0, valid_depth_squares = 0;
        size_t valid_depth_channels = 0;
        size_t depth_coverage_disagreements = 0;
        if (depth_view) {
            for (size_t i = 0; i < native.numel(); i += 3) {
                bool native_empty = true, reference_empty = true;
                double pixel_error = 0;
                for (size_t c = 0; c < 3; ++c) {
                    const float bg = std::round(background[c] * 255) / 255;
                    native_empty = native_empty && std::abs(a[i + c] - bg) < 1e-6f;
                    reference_empty = reference_empty && std::abs(b[i + c] - bg) < 1e-6f;
                    pixel_error = std::max(pixel_error, std::abs(double(a[i + c]) - b[i + c]));
                }
                if (native_empty != reference_empty)
                    ++depth_coverage_disagreements;
                else {
                    valid_depth_max = std::max(valid_depth_max, pixel_error);
                    for (size_t c = 0; c < 3; ++c) {
                        const double delta = double(a[i + c]) - b[i + c];
                        valid_depth_squares += delta * delta;
                    }
                    valid_depth_channels += 3;
                }
            }
        }
        const double mse = squares / native.numel();
        return {{"mae", sum / native.numel()}, {"rmse", std::sqrt(mse)}, {"max_error", maximum}, {"psnr_db", mse == 0 ? Json(nullptr) : Json(-10 * std::log10(mse))}, {"identical", mse == 0}, {"depth_valid_max_error", depth_view ? Json(valid_depth_max) : Json(nullptr)}, {"depth_valid_rmse", depth_view ? Json(std::sqrt(valid_depth_squares / std::max<size_t>(1, valid_depth_channels))) : Json(nullptr)}, {"depth_coverage_disagreement_pixels", depth_coverage_disagreements}, {"depth_coverage_disagreement_fraction", 3. * depth_coverage_disagreements / native.numel()}};
    }
    Json run(Options o) {
        core::GpuBackendScope scope(o.tensor_backend);
        vis::VulkanContext context;
        if (!context.initHeadless())
            throw std::runtime_error(context.lastError());
        std::shared_ptr<core::SplatData> imported;
        rendering::FrameView imported_view;
        std::vector<glm::mat4> imported_transforms;
        std::vector<int> degrees{0, 3};
        std::string loader_name;
        if (!o.input.empty()) {
            if (o.input_fixture) {
                // Exercise production codecs and importer routing, including SH3 ->
                // Q16 preparation. No download, CUDA codec or user file is needed.
                // The corpus writer has one fixed tensor backend independent of the
                // import/storage backend under test. SSOG's CPU hierarchy builder
                // currently allocates intermediate device gathers on Metal.
                core::GpuBackendScope fixture_scope(core::GpuBackend::Metal);
                auto fixture = scene(o.fixture_count, 3);
                const auto format = o.fixture_format.empty() ? "ply" : o.fixture_format;
                const auto saved = [&]() -> io::Result<void> {
                    if (format == "ply")
                        return io::save_ply(fixture, {.output_path = o.input});
                    if (format == "spz3" || format == "spz4" || format == "glb")
                        return io::save_spz(fixture, {.output_path = o.input, .version = format == "spz4" ? 4 : 3, .glb = format == "glb"});
                    if (format == "sog")
                        return io::save_sog(fixture, {.output_path = o.input, .kmeans_iterations = 1, .use_gpu = false});
                    if (format == "ssog")
                        return io::save_ssog(fixture, {.output_path = o.input, .lod_levels = 2, .kmeans_iterations = 1, .use_gpu = false});
                    return io::save_rad(fixture, {.output_path = o.input});
                }();
                if (!saved)
                    throw std::runtime_error(saved.error().format());
            }
            // Loading, codec preparation and camera fitting precede all measured
            // frames. Both adapters consume this same resident model unchanged.
            auto loader = io::Loader::create();
            // Use the same shared migration step as desktop import. SPZ starts
            // in CPU decoder storage; direct loader calls without an allocator
            // would test an unprepared model that the application never renders.
            io::LoadOptions load_options;
            load_options.splat_tensor_allocator = context.tensorInterop().splat_allocator();
            auto loaded = loader->load(o.input, load_options);
            if (!loaded)
                throw std::runtime_error(loaded.error().format());
            auto splats = std::get_if<std::shared_ptr<core::SplatData>>(&loaded->data);
            if (!splats || !*splats || !(*splats)->means_raw().size(0))
                throw std::runtime_error("Benchmark input must contain a nonempty splat model");
            imported = *splats;
            loader_name = loaded->loader_used;
            o.count = imported->means_raw().size(0);
            if (o.input_fixture && o.fixture_format != "rad" && o.fixture_format != "ssog" && o.count != o.fixture_count)
                throw std::runtime_error("Flat fixture import changed its requested source count");
            if (o.count > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error("Real scene exceeds the renderer's uint32 source extent");
            degrees = {0};
            const int degree = std::min(3, imported->get_max_sh_degree());
            if (degree > 0) {
                (void)imported->apply_shN_value_quant();

                degrees.push_back(degree);
            }
            imported_transforms = {rendering::DATA_TO_VISUALIZER_WORLD_AXES_4};
            if (o.camera.empty()) {
                const auto low = imported->means_raw().min(0).cpu();
                const auto high = imported->means_raw().max(0).cpu();
                const glm::vec3 a(low.ptr<float>()[0], low.ptr<float>()[1], low.ptr<float>()[2]);
                const glm::vec3 b(high.ptr<float>()[0], high.ptr<float>()[1], high.ptr<float>()[2]);
                for (int c = 0; c < 3; ++c)
                    if (!std::isfinite(a[c]) || !std::isfinite(b[c]))
                        throw std::runtime_error("Cannot fit a camera to non-finite scene bounds");
                const auto center = rendering::visualizerWorldPointFromDataWorld(a * .5f + b * .5f);
                const float radius = std::max(.01f, glm::length(b - a) * .5f);
                if (!std::isfinite(radius))
                    throw std::runtime_error("Scene bounds exceed finite camera-fit precision");
                const float aspect = float(o.width) / o.height;
                const float half_fov = std::atan(std::tan(rendering::focalLengthToVFovRad(imported_view.focal_length_mm) * .5f) * std::min(1.f, aspect));
                imported_view.translation = center + glm::vec3(0, 0, 1.2f * radius / std::sin(half_fov));
            } else {
                std::ifstream stream(o.camera);
                if (!stream)
                    throw std::runtime_error("Cannot read benchmark camera JSON");
                Json camera;
                stream >> camera;
                if (camera.contains("structuredContent"))
                    camera = camera.at("structuredContent");
                if (camera.contains("camera"))
                    camera = camera.at("camera");
                for (int row = 0; row < 3; ++row) {
                    imported_view.translation[row] = camera.at("eye").at(row).get<float>();
                    for (int col = 0; col < 3; ++col)
                        imported_view.rotation[col][row] = camera.at("rotation_matrix").at(row).at(col).get<float>();
                }
                const float fov = camera.at("fov_degrees").get<float>();
                if (!(fov > 0 && fov < 179) || !std::isfinite(fov))
                    throw std::runtime_error("Invalid benchmark camera FOV");
                imported_view.focal_length_mm = rendering::vFovToFocalLength(fov);
                const auto orthogonal = glm::transpose(imported_view.rotation) * imported_view.rotation;
                for (int row = 0; row < 3; ++row) {
                    if (!std::isfinite(imported_view.translation[row]))
                        throw std::runtime_error("Non-finite benchmark camera position");
                    for (int col = 0; col < 3; ++col)
                        if (!std::isfinite(imported_view.rotation[col][row]) || std::abs(orthogonal[col][row] - (row == col ? 1.f : 0.f)) > .001f)
                            throw std::runtime_error("Invalid benchmark camera rotation");
                }
                if (std::abs(glm::determinant(imported_view.rotation) - 1.f) > .001f)
                    throw std::runtime_error("Benchmark camera rotation must preserve handedness");
            }
        }
        Json cases = Json::array();
        for (int degree : degrees) {
            auto generated = imported ? nullptr : std::make_unique<core::SplatData>(scene(o.count, degree, o.overlay == "affine" ? 1.f : .1f, o.equirect, o.near, nullptr, o.spark, o.gpu_lod, o.saturation, o.frustum ? &o : nullptr, o.depth_boundary ? &o : nullptr));
            auto& model = imported ? *imported : *generated;
            model.set_active_sh_degree(degree);
            vis::MetalViewportRenderer metal;
            metal.setProfilingEnabled(o.profile_gpu);
            vis::VksplatViewportRenderer vulkan;
            vulkan.setDepthCaptureMode(o.scalar_depth_reference);
            rendering::ViewportRenderRequest request;
            std::vector<uint32_t> lod_indices, lod_logical, lod_levels;
            std::vector<float> lod_weights;
            if (o.lod && !o.gpu_lod) {
                // Reverse, sparse source cut: never a contiguous prefix. Q16
                // reads must retain original cell swizzle and block bounds.
                for (size_t n = 0; n < o.count; n += 2) {
                    const uint32_t source = uint32_t(o.count - 1 - n);
                    lod_indices.push_back(source);
                    lod_logical.push_back(uint32_t(n));
                    lod_levels.push_back(uint32_t(n / 2) % 5u);
                    lod_weights.push_back(n % 6 == 0 ? 0.f : n % 6 == 2 ? .3f
                                                                        : 1.f);
                }
                request.lod_indices = lod_indices.data();
                request.lod_count = lod_indices.size();
                request.lod_logical_indices = o.lod_logical ? lod_logical.data() : nullptr;
                request.lod_levels = lod_levels.data();
                request.lod_debug_mode = o.lod_debug;
                request.lod_weights = o.lod_weights ? lod_weights.data() : nullptr;
            }
            if (o.gpu_lod) {
                auto& lod = request.lod_gpu_traversal;
                lod.enabled = true;
                lod.node_count = o.count;
                lod.output_capacity = o.gpu_lod_budget ? 1 : o.count;
                lod.pixel_scale_limit = .001f;
                lod.object_scale = 1;
                lod.behind_camera_penalty = lod.cone_foveation = 1;
                lod.viewport_foveation = false;
            }
            if (imported) {
                request.frame_view = imported_view;
                request.scene.model_transforms = &imported_transforms;
            }
            request.frame_view.size = {o.width, o.height};
            if (o.near)
                request.frame_view.far_plane = .06f;
            request.frame_view.rasterization_scale = o.export_scale ? 2.f : 1.f;
            request.sh_degree = degree;
            request.gut = o.gut;
            request.splat_render_profile = o.portal ? 1 : 0;
            request.color_tonemapping = o.tone;
            request.color_exposure = o.exposure;
            request.transparent_background = o.transparent;
            request.depth_visualization_mode = o.depth_gray ? rendering::DepthVisualizationMode::Grayscale : rendering::DepthVisualizationMode::Palette;
            request.equirectangular = o.equirect;
            if (o.subregion) {
                request.frame_view.subregion_full_size = {o.width * 2, o.height * 2};
                request.frame_view.subregion_origin = {o.width, o.height / 2};
            }
            request.raster_backend = o.gut ? rendering::GaussianRasterBackend::ThreeDgut : rendering::GaussianRasterBackend::ThreeDgs;
            request.frame_view.background_color = {.02f, .03f, .04f};
            request.mip_filter = o.mip;
            request.depth_view = o.depth;
            request.frame_view.orthographic = o.ortho;
            request.frame_view.ortho_scale = 32;
            if (o.overlay == "selection" || o.overlay == "preview" || o.overlay == "selected-rings") {
                std::vector<float> mask(o.count);
                for (size_t n = 0; n < o.count; ++n)
                    mask[n] = float(n % 3);
                auto tensor = std::make_shared<core::Tensor>(core::Tensor::from_vector(mask, {o.count}, core::Device::GPU).to(core::DataType::UInt8));
                if (o.overlay == "selection" || o.overlay == "selected-rings") {
                    request.overlay.has_selection = true;
                    request.overlay.emphasis.mask = tensor;
                } else {
                    request.overlay.emphasis.transient_mask.owned_mask = tensor;
                    request.overlay.emphasis.transient_mask.mask = tensor.get();
                }
            }
            if (o.overlay == "crop") {
                rendering::GaussianScopedBoxFilter crop;
                crop.bounds.min = {-.6f, -1, -9};
                crop.bounds.max = {.6f, 1, -3};
                crop.desaturate = true;
                request.filters.crop_region = crop;
            }
            if (o.overlay == "ellipsoid") {
                rendering::GaussianScopedEllipsoidFilter ellipsoid;
                ellipsoid.bounds.radii = {1, 1, 2};
                ellipsoid.bounds.transform = glm::mat4(1);
                ellipsoid.bounds.transform[3].z = 6;
                ellipsoid.desaturate = true;
                request.filters.ellipsoid_region = ellipsoid;
            }
            if (o.overlay == "window") {
                rendering::BoundingBox volume;
                volume.min = {-2, -2, -9};
                volume.max = {2, 2, -3};
                request.filters.view_volume = volume;
                request.filters.screen_window = rendering::SelectionScreenWindow{};
                request.filters.dim_outside_view_volume = true;
            }
            if (o.overlay == "rings" || o.overlay == "selected-rings") {
                request.overlay.markers.show_rings = true;
                request.overlay.markers.ring_width = .02f;
            }
            if (o.overlay == "markers")
                request.overlay.markers.show_center_markers = true;
            if (o.overlay == "affine") {
                static const std::vector<glm::mat4> transforms = {
                    glm::scale(glm::rotate(glm::mat4(1), .2f, glm::vec3(0, 1, 0)), glm::vec3(1.3f, .7f, 1.1f))};
                request.scene.model_transforms = &transforms;
            }
            if (o.overlay == "flash") {
                static const std::vector<glm::mat4> transforms = {glm::mat4(1)};
                request.scene.model_transforms = &transforms;
                request.scene.transform_indices = std::make_shared<core::Tensor>(core::Tensor::zeros({o.count}, core::Device::GPU, core::DataType::Int32));
                request.overlay.emphasis.emphasized_node_mask = {true};
                request.overlay.emphasis.flash_intensity = .8f;
            }
            if (!vis::MetalViewportRenderer::supports(model, request))
                throw std::runtime_error("Unsupported native benchmark frame: means device=" + std::to_string(int(model.means_raw().device())) +
                                         " dtype=" + std::to_string(int(model.means_raw().dtype())) +
                                         " sh0=" + std::to_string(int(model.sh0_raw().dtype())) +
                                         " packed_nonSH=" + std::to_string(model.non_sh_attrs_f16()) +
                                         " rad=" + std::to_string(bool(model.lod_tree && model.lod_tree->rad_source.valid())) +
                                         " gpu_lod=" + std::to_string(request.lod_gpu_traversal.enabled));
            // Compare tiled export against the full reference camera. The
            // legacy Vulkan panorama subregion wraps its local tile grid;
            // comparing that output would bless a clipped reference seam.
            auto reference_request = request;
            std::unique_ptr<core::SplatData> reference_cut;
            if (o.gut && o.lod) {
                reference_cut = std::make_unique<core::SplatData>(scene(o.count, degree, o.overlay == "affine" ? 1.f : .1f, o.equirect, o.near, &o));
                reference_request.lod_indices = reference_request.lod_logical_indices = reference_request.lod_levels = nullptr;
                reference_request.lod_weights = nullptr;
                reference_request.lod_count = 0;
                reference_request.lod_debug_mode = false;
                if (o.overlay == "selection" || o.overlay == "selected-rings") {
                    std::vector<float> mask(o.count);
                    for (size_t n = 0; n < lod_indices.size(); ++n)
                        mask[lod_indices[n]] = float((o.lod_logical ? lod_logical[n] : lod_indices[n]) % 3);
                    reference_request.overlay.emphasis.mask = std::make_shared<core::Tensor>(core::Tensor::from_vector(mask, {o.count}, core::Device::GPU).to(core::DataType::UInt8));
                }
            }
            if (o.equirect && o.subregion) {
                reference_request.frame_view.size = request.frame_view.cameraSize();
                reference_request.frame_view.subregion_full_size = reference_request.frame_view.subregion_origin = {0, 0};
            }
            auto frame = [&](bool native) {
                auto result = native ? vis::legacyMetalResult(metal.render(context, model, request, kTarget))
                                     : vulkan.render(context, reference_cut ? *reference_cut : model, reference_request, false, kTarget, false, o.deterministic_reference);
                if (!result)
                    throw std::runtime_error(result.error());
                const auto expected_backend = native ? rendering::ViewerBackend::Metal : rendering::ViewerBackend::Vulkan;
                if (result->viewer_backend != expected_backend || bool(result->generation >> 63) != native)
                    throw std::runtime_error("Benchmark frame did not use the requested renderer API");
                wait(context, *result);
            };
            auto complete = [&] {
                const auto status = metal.outputComplete(kTarget);
                if (!status)
                    throw std::runtime_error(lfs::format_for_developer(status.error()));
                return *status;
            };
            for (int n = 0; n < o.warmup; ++n) {
                frame(n % 2 == 0);
                frame(n % 2 != 0);
            }
            if (!complete())
                throw std::runtime_error("Native reservation did not converge during warmup");
            if (o.gpu_lod) {
                const auto status = metal.gpuLodSelectionStatus(kTarget);
                const size_t expected = o.gpu_lod_budget ? 1 : o.count - 1 - (o.count + 59999) / 60000;
                if (!status.active || status.selected != expected || status.overflow || status.resident_chunks != (o.count + core::SplatLodTree::kChunkSplats - 1) / core::SplatLodTree::kChunkSplats)
                    throw std::runtime_error("Native LOD diagnostics differ from the completed GPU cut");
            }
            std::vector<double> native_times, vulkan_times;
            std::array<std::vector<double>, 6> gpu_times;
            for (int n = 0; n < o.samples; ++n) {
                // AB/BA pairs reduce order, thermal and drift bias; keep every raw sample.
                for (int j = 0; j < 2; ++j) {
                    const bool native = (n + j) % 2 == 0;
                    const auto start = Clock::now();
                    frame(native);
                    const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                    if (!(ms > 0) || !std::isfinite(ms))
                        throw std::runtime_error("Invalid timing sample");
                    if (native && !complete())
                        throw std::runtime_error("Partial native frame in measured sample");
                    (native ? native_times : vulkan_times).push_back(ms);
                    if (native && o.profile_gpu) {
                        const auto diagnostics = metal.frameDiagnostics(kTarget);
                        if (!diagnostics)
                            throw std::runtime_error(lfs::format_for_developer(diagnostics.error()));
                        gpu_times[0].push_back(diagnostics->gpu_command_ms);
                        if (diagnostics->counter_timestamps_available)
                            for (size_t stage = 0; stage < 5; ++stage)
                                gpu_times[stage + 1].push_back(diagnostics->gpu_stage_ms[stage]);
                    }
                }
            }
            // GPU synchronization is measured; CPU image transfers are deliberately separate.
            auto pixels = core::Tensor::empty({size_t(o.height), size_t(o.width), 3}, core::Device::CPU, core::DataType::Float32);
            const auto read = metal.readColor(kTarget, pixels, 0, 0);
            if (!read)
                throw std::runtime_error(lfs::format_for_developer(read.error()));
            auto reference = vulkan.readOutputImage(context, kTarget);
            if (!reference)
                throw std::runtime_error(reference.error());
            if (o.equirect && o.subregion) {
                auto cropped = std::make_shared<core::Tensor>(core::Tensor::empty({size_t(o.height), size_t(o.width), 3}, core::Device::CPU, core::DataType::Float32));
                const auto origin = request.frame_view.subregion_origin;
                const size_t stride = reference_request.frame_view.size.x;
                for (size_t y = 0; y < size_t(o.height); ++y)
                    std::copy_n((*reference)->ptr<float>() + ((y + origin.y) * stride + origin.x) * 3,
                                size_t(o.width) * 3, cropped->ptr<float>() + y * o.width * 3);
                *reference = std::move(cropped);
            }
            if (o.depth_boundary) {
                const auto native_depth = metal.readDepth({.pixel = {o.width / 2, o.height / 2}, .source_size = {o.width, o.height}, .target = kTarget});
                const auto reference_depth = vulkan.readPreviewDepth(context, kTarget);
                if (!native_depth || !reference_depth)
                    throw std::runtime_error("Boundary depth readback failed");
                const auto center = size_t(o.height / 2) * o.width + o.width / 2;
                if (std::abs(*native_depth - 6.f) > 1e-5f || std::abs((*reference_depth)->ptr<float>()[center] - 6.f) > 1e-5f)
                    throw std::runtime_error("Exact median lost at FP16 batch boundary: Metal=" + std::to_string(*native_depth) + " Vulkan=" + std::to_string((*reference_depth)->ptr<float>()[center]) + " expected=6");
            }
            if (!o.images.empty()) {
                const auto save = [&](const core::Tensor& image, const char* backend) {
                    std::ofstream stream(o.images + "-sh" + std::to_string(degree) + "-" + backend + ".ppm", std::ios::binary);
                    stream << "P6\n"
                           << o.width << ' ' << o.height << "\n255\n";
                    for (size_t i = 0; i < image.numel(); ++i) {
                        const auto value = static_cast<unsigned char>(std::lround(std::clamp(image.ptr<float>()[i], 0.f, 1.f) * 255));
                        stream.write(reinterpret_cast<const char*>(&value), 1);
                    }
                    if (!stream)
                        throw std::runtime_error("Cannot write diagnostic image");
                };
                save(pixels, "metal");
                save(**reference, "vulkan");
            }
            auto difference = quality(pixels, **reference, request.frame_view.background_color, o.depth);
            if (o.depth_diagnostics) {
                if (o.subregion || o.gut || o.lod || o.gpu_lod)
                    throw std::runtime_error("Depth diagnostic currently requires ordinary full-frame GS");
                auto native_depth = core::Tensor::empty({size_t(o.height), size_t(o.width)}, core::Device::CPU, core::DataType::Float32);
                auto ticket = metal.submitReadback(kTarget, native_depth, 0, 0, true);
                if (!ticket)
                    throw std::runtime_error(lfs::format_for_developer(ticket.error()));
                auto ready = metal.pollReadback(*ticket, true);
                if (!ready || *ready != vis::VksplatViewportRenderer::ReadbackTicketStatus::Ready)
                    throw std::runtime_error("Depth diagnostic readback failed");
                auto reference_depth = vulkan.readPreviewDepth(context, kTarget);
                if (!reference_depth)
                    throw std::runtime_error(reference_depth.error());
                // An independent production FP32 chain checks whether the
                // discrepancy originates in projection or HiGS batch arithmetic.
                vulkan.setDepthCaptureMode(true);
                frame(false);
                auto scalar_pixels = vulkan.readOutputImage(context, kTarget);
                auto scalar_depth = vulkan.readPreviewDepth(context, kTarget);
                vulkan.setDepthCaptureMode(false);
                if (!scalar_pixels || !scalar_depth)
                    throw std::runtime_error("FP32 diagnostic readback failed");
                Json diagnostic{{"metal_vs_macro", difference},
                                {"metal_vs_scalar", quality(pixels, **scalar_pixels, request.frame_view.background_color, true)},
                                {"macro_vs_scalar", quality(**reference, **scalar_pixels, request.frame_view.background_color, true)}};
                std::vector<std::pair<double, size_t>> errors;
                for (size_t at = 0; at < native_depth.numel(); ++at) {
                    double error = 0;
                    for (size_t c = 0; c < 3; ++c)
                        error = std::max(error, std::abs(double(pixels.ptr<float>()[3 * at + c]) - (*reference)->ptr<float>()[3 * at + c]));
                    if (error > 1. / 255.)
                        errors.emplace_back(error, at);
                }
                std::sort(errors.rbegin(), errors.rend());
                diagnostic["pixels"] = Json::array();
                for (size_t j = 0; j < std::min<size_t>(100, errors.size()); ++j) {
                    const auto [error, at] = errors[j];
                    diagnostic["pixels"].push_back({{"x", at % o.width}, {"y", at / o.width}, {"rgb_error", error}, {"metal_depth", native_depth.ptr<float>()[at]}, {"macro_depth", (*reference_depth)->ptr<float>()[at]}, {"scalar_depth", (*scalar_depth)->ptr<float>()[at]}});
                }
                const auto prefix = o.output + "-sh" + std::to_string(degree);
                std::ofstream(prefix + "-depth-diagnostic.json") << diagnostic.dump(2);
                const auto save_depth = [&](const core::Tensor& tensor, const char* label) {
                    std::ofstream file(prefix + "-" + label + ".f32", std::ios::binary);
                    file.write(reinterpret_cast<const char*>(tensor.ptr<float>()), tensor.numel() * sizeof(float));
                };
                save_depth(native_depth, "metal-depth");
                save_depth(**reference_depth, "macro-depth");
                save_depth(**scalar_depth, "scalar-depth");
            }

            if (o.equirect && o.overlay == "markers") {
                const auto a = pixels.ptr<float>(), b = (*reference)->ptr<float>();
                // Center markers have two flat colors separated by a hard
                // 1.5-pixel boundary. Subpixel FP32 UT rounding can flip a
                // boundary pixel; require the exact flat-color pair and keep
                // its disagreement count explicit, as for median depth.
                glm::vec3 marker(0);
                for (size_t i = 0; i < pixels.numel(); i += 3)
                    if (glm::dot(glm::vec3(b[i], b[i + 1], b[i + 2]), glm::vec3(1)) > glm::dot(marker, glm::vec3(1)))
                        marker = {b[i], b[i + 1], b[i + 2]};
                const auto matches = [&](const float* rgb, glm::vec3 expected) {
                    for (int c = 0; c < 3; ++c)
                        if (std::abs(rgb[c] - expected[c]) > 2.f / 255)
                            return false;
                    return true;
                };
                size_t boundary = 0;
                double stable_max = 0;
                for (size_t i = 0; i < pixels.numel(); i += 3) {
                    const bool edge = (matches(a + i, marker) && matches(b + i, marker * .4f)) ||
                                      (matches(b + i, marker) && matches(a + i, marker * .4f));
                    if (edge)
                        ++boundary;
                    else
                        for (size_t c = 0; c < 3; ++c)
                            stable_max = std::max(stable_max, std::abs(double(a[i + c]) - b[i + c]));
                }
                difference["marker_boundary_disagreement_pixels"] = boundary;
                difference["marker_boundary_disagreement_fraction"] = 3. * boundary / pixels.numel();
                difference["marker_stable_max_error"] = stable_max;
            }
            if (o.transparent) {
                auto rgba = core::Tensor::empty({size_t(o.height), size_t(o.width), 4}, core::Device::CPU, core::DataType::Float32);
                const auto native = metal.readColor(kTarget, rgba, 0, 0);
                if (!native)
                    throw std::runtime_error(lfs::format_for_developer(native.error()));
                auto ref = vulkan.readOutputImageRgba(context, kTarget);
                if (!ref)
                    throw std::runtime_error(ref.error());
                double maximum = 0, squares = 0, native_sum = 0, reference_sum = 0, worst_rgb = -1;
                std::array<double, 2> composite_max{}, composite_squares{};
                double common_max = 0, common_squares = 0;
                size_t common_channels = 0, coverage_disagreements = 0;
                const auto origin = request.frame_view.subregion_origin;
                const size_t ref_width = reference_request.frame_view.size.x;
                for (size_t y = 0; y < size_t(o.height); ++y)
                    for (size_t x = 0; x < size_t(o.width); ++x) {
                        const size_t pixel = y * o.width + x;
                        const size_t ref_pixel = o.equirect && o.subregion ? (y + origin.y) * ref_width + x + origin.x : pixel;
                        const float a = rgba.ptr<float>()[pixel * 4 + 3], b = (*ref)->ptr<float>()[ref_pixel * 4 + 3];
                        if (!std::isfinite(a) || !std::isfinite(b) || a < 0 || a > 1 || b < 0 || b > 1)
                            throw std::runtime_error("Invalid transparent alpha");
                        double color_error = 0;
                        if ((a > 0) != (b > 0))
                            ++coverage_disagreements;
                        for (size_t c = 0; c < 3; ++c) {
                            const double native_rgb = rgba.ptr<float>()[pixel * 4 + c];
                            const double reference_rgb = (*ref)->ptr<float>()[ref_pixel * 4 + c];
                            const double straight_delta = native_rgb - reference_rgb;
                            color_error = std::max(color_error, std::abs(straight_delta));
                            if (a > 0 && b > 0) {
                                common_max = std::max(common_max, std::abs(straight_delta));
                                common_squares += straight_delta * straight_delta;
                                ++common_channels;
                            }
                            // Straight RGB is undefined at zero alpha. Keep its
                            // raw errors in the report, and check visible color
                            // on both black and white across ALL pixels instead.
                            // These endpoints bound every background in [0,1].
                            const double black_delta = native_rgb * a - reference_rgb * b;
                            const std::array<double, 2> delta{black_delta, black_delta + double(b) - a};
                            for (size_t background = 0; background < 2; ++background) {
                                composite_max[background] = std::max(composite_max[background], std::abs(delta[background]));
                                composite_squares[background] += delta[background] * delta[background];
                            }
                        }
                        if (color_error > worst_rgb) {
                            worst_rgb = color_error;
                            difference["transparent_worst_pixel"] = {
                                {"x", x},
                                {"y", y},
                                {"native_alpha", a},
                                {"reference_alpha", b},
                                {"native_rgb", {rgba.ptr<float>()[pixel * 4], rgba.ptr<float>()[pixel * 4 + 1], rgba.ptr<float>()[pixel * 4 + 2]}},
                                {"reference_rgb", {(*ref)->ptr<float>()[ref_pixel * 4], (*ref)->ptr<float>()[ref_pixel * 4 + 1], (*ref)->ptr<float>()[ref_pixel * 4 + 2]}}};
                        }
                        const double delta = std::abs(double(a) - b);
                        maximum = std::max(maximum, delta);
                        squares += delta * delta;
                        native_sum += a;
                        reference_sum += b;
                    }
                const double rms = std::sqrt(squares / (size_t(o.width) * o.height));
                if (native_sum < .1 || reference_sum < .1)
                    throw std::runtime_error("Transparent benchmark published empty coverage");
                difference["alpha_max_error"] = maximum;
                difference["alpha_rmse"] = rms;
                difference["transparent_valid_max_error"] = common_max;
                difference["transparent_valid_rmse"] = std::sqrt(common_squares / std::max<size_t>(1, common_channels));
                difference["transparent_coverage_disagreement_pixels"] = coverage_disagreements;
                difference["transparent_coverage_disagreement_fraction"] = double(coverage_disagreements) / (size_t(o.width) * o.height);
                double visible_max = 0, visible_rmse = 0;
                for (size_t background = 0; background < 2; ++background) {
                    const double error = std::sqrt(composite_squares[background] / (size_t(o.width) * o.height * 3));
                    difference[background ? "transparent_white_max_error" : "transparent_black_max_error"] = composite_max[background];
                    difference[background ? "transparent_white_rmse" : "transparent_black_rmse"] = error;
                    visible_max = std::max(visible_max, composite_max[background]);
                    visible_rmse = std::max(visible_rmse, error);
                }
                difference["transparent_visible_max_error"] = visible_max;
                difference["transparent_visible_rmse"] = visible_rmse;
                if (o.verify_parity && (maximum > 4. / 255 + 1e-7 || rms > 1. / 255))
                    throw std::runtime_error("Transparent alpha exceeds parity bounds: " + difference.dump());
                if (o.verify_parity && (visible_max > 4. / 255 + 1e-7 || visible_rmse > 1. / 255 ||
                                        double(difference["transparent_coverage_disagreement_fraction"]) > .001))
                    throw std::runtime_error("Transparent composite/coverage exceeds parity bounds: " + difference.dump());
            }
            // The Vulkan reference uses compressed FP16 footprints/colors. Bound
            // local and RMS error without claiming bit-identical arithmetic.
            // Median depth has a hard coverage boundary at 0.5. FP16 reference and
            // FP32 native can disagree on boundary pixels; record the full image
            // error and enforce a separate bound, never silently drop those pixels.
            // Unpremultiplied RGB has no visible meaning where either alpha
            // is zero. Its full error remains reported; common coverage keeps
            // the original straight-RGB max gate, with full-image black/white
            // composite and alpha gates enforced above (no pixels omitted).
            const double local_error = o.depth                                ? double(difference["depth_valid_max_error"])
                                       : o.transparent                        ? double(difference["transparent_valid_max_error"])
                                       : o.equirect && o.overlay == "markers" ? double(difference["marker_stable_max_error"])
                                                                              : double(difference["max_error"]);
            const double rms_error = double(difference[o.depth ? "depth_valid_rmse" : o.transparent ? "transparent_visible_rmse"
                                                                                                    : "rmse"]);
            if (o.verify_parity && (local_error > 4. / 255 + 1e-7 ||
                                    double(difference["depth_coverage_disagreement_fraction"]) > .001 ||
                                    (difference.contains("marker_boundary_disagreement_fraction") && double(difference["marker_boundary_disagreement_fraction"]) > .001) ||
                                    rms_error > 1. / 255))
                throw std::runtime_error("Native image exceeds FP16-reference parity bounds (valid max 4/255, RMS 1/255, depth coverage 0.1%): SH" + std::to_string(degree) + " " + difference.dump());
            const auto native_stats = statistics(native_times), vulkan_stats = statistics(vulkan_times);
            const auto diagnostics = metal.frameDiagnostics(kTarget);
            if (!diagnostics)
                throw std::runtime_error(lfs::format_for_developer(diagnostics.error()));
            Json gpu_diagnostics = {{"required_instances", diagnostics->required_instances},
                                    {"reserved_instances", diagnostics->reserved_instances},
                                    {"input_splats", diagnostics->input_splats},
                                    {"blend_threads_per_group", diagnostics->blend_threads},
                                    {"maximum_tile_instances", diagnostics->maximum_tile_instances},
                                    {"counter_timestamps_available", diagnostics->counter_timestamps_available}};
            if (o.profile_gpu) {
                const std::array<const char*, 6> stages{"command", "projection", "instances", "sort", "blend", "present"};
                for (size_t stage = 0; stage < stages.size(); ++stage)
                    gpu_diagnostics[stages[stage]] = gpu_times[stage].empty() ? Json(nullptr) : statistics(gpu_times[stage]);
            }
            cases.push_back({{"sh_degree", degree}, {"storage", degree ? (model.shN_value_quantized() ? "q16" : model.shN_ieee_f16() ? "f16"
                                                                                                                                     : "f32")
                                                                       : "sh0"},
                             {"metal", native_stats},
                             {"metal_gpu_diagnostics", gpu_diagnostics},
                             {"vulkan", vulkan_stats},
                             {"speedup_vulkan_over_metal", (o.subregion || (o.gut && o.lod)) ? Json(nullptr) : Json(double(vulkan_stats["median_ms"]) / double(native_stats["median_ms"]))},
                             {"image_difference", difference}});
        }
        Json camera_pose = Json::object();
        if (imported) {
            camera_pose["eye"] = {imported_view.translation.x, imported_view.translation.y, imported_view.translation.z};
            camera_pose["focal_length_mm"] = imported_view.focal_length_mm;
            camera_pose["rotation_matrix"] = Json::array();
            for (int row = 0; row < 3; ++row)
                camera_pose["rotation_matrix"].push_back({imported_view.rotation[0][row], imported_view.rotation[1][row], imported_view.rotation[2][row]});
        }
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        return {{"schema_version", 1}, {"renderer_identity_verified", true}, {"preferences_isolated", true}, {"metal_renderer", "metal"}, {"vulkan_reference_renderer", "vulkan"}, {"gpu_profiling", o.profile_gpu}, {"deterministic_reference", o.deterministic_reference}, {"tensor_backend", core::gpu_backend_name(o.tensor_backend)}, {"metric", "completed_frame_wall_latency_ms"}, {"includes", "host encode, submission, GPU raster, output conversion, completion wait"}, {"excludes", "warmup, CPU image readback, desktop UI/compositor, frame pipelining"}, {"device", MTLCreateSystemDefaultDevice().name.UTF8String}, {"os", NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String}, {"compiler", __clang_version__}, {"scene_seed", imported ? Json(nullptr) : Json(1939)}, {"scene_source", imported ? o.input : "synthetic"}, {"loader", loader_name}, {"generated_input_fixture", o.input_fixture}, {"generated_source_count", o.input_fixture ? Json(o.fixture_count) : Json(nullptr)}, {"fixture_format", o.fixture_format}, {"camera_source", imported ? (o.camera.empty() ? "fitted_bounds" : o.camera) : "synthetic"}, {"camera_pose", camera_pose}, {"metal_debug_layer", std::getenv("MTL_DEBUG_LAYER") ? std::getenv("MTL_DEBUG_LAYER") : "unset"}, {"metal_shader_validation", std::getenv("MTL_SHADER_VALIDATION") ? std::getenv("MTL_SHADER_VALIDATION") : "unset"}, {"count", o.count}, {"width", o.width}, {"height", o.height}, {"warmup_pairs", o.warmup}, {"profile", o.portal ? "portal" : "studio"}, {"tone_fixture", o.portal_tone}, {"tone_operator", o.tone}, {"exposure", o.exposure}, {"transparent", o.transparent}, {"depth_grayscale", o.depth_gray}, {"reference_resident_cut", o.gut && o.lod}, {"gpu_lod", o.gpu_lod}, {"gpu_lod_budget", o.gpu_lod_budget}, {"spark_opacity", o.spark}, {"lod", o.lod}, {"lod_logical", o.lod_logical}, {"lod_weights", o.lod_weights}, {"lod_debug", o.lod_debug}, {"gut", o.gut}, {"equirectangular", o.equirect}, {"near_fixture", o.near}, {"saturation_fixture", o.saturation}, {"frustum_fixture", o.frustum}, {"subregion", o.subregion}, {"reference_full_frame_crop", o.equirect && o.subregion}, {"mip", o.mip}, {"orthographic", o.ortho}, {"depth_view", o.depth}, {"scalar_depth_reference", o.scalar_depth_reference}, {"overlay_fixture", o.overlay}, {"rasterization_scale", o.export_scale ? 2.f : 1.f}, {"samples_per_backend", o.samples}, {"process_peak_rss_bytes", usage.ru_maxrss}, {"cases", cases}};
    }
} // namespace
int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            const auto o = options(argc, argv);
            // Safe mode always returns defaults and cannot retain an explicit
            // renderer choice. Use an isolated home instead of user settings.
            auto preference_home = (std::filesystem::temp_directory_path() / "lichtfeld-viewer-benchmark-XXXXXX").string();
            if (!mkdtemp(preference_home.data()) || setenv("LFS_HOME", preference_home.c_str(), 1) != 0 ||
                unsetenv("LFS_SAFE_MODE") != 0)
                throw std::runtime_error("Cannot isolate benchmark preferences");
            if (!core::gpu_backend_available(core::GpuBackend::Metal)) {
                return lfs::metal_test::unavailableMetal4();
            }
            Py_Initialize();
            const auto report = run(o).dump(2);
            if (!o.output.empty()) {
                std::ofstream output(o.output);
                output << report << '\n';
                output.flush();
                if (!output)
                    throw std::runtime_error("Cannot write benchmark report");
            }
            std::puts(report.c_str());
            Py_Finalize();
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }
}
