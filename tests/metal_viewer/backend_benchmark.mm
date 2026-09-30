/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_backend.hpp"
#include "metal_viewport_renderer.hpp"
#include "preferences.hpp"
#import <Metal/Metal.h>
#include <Python.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

namespace {
    using namespace lfs;
    using Clock = std::chrono::steady_clock;
    using Json = nlohmann::json;
    using Slot = vis::VksplatViewportRenderer::OutputSlot;
    struct Options {
        size_t count = 100000;
        int width = 1280, height = 720, warmup = 12, samples = 40;
        std::string output, images, overlay;
        bool verify_parity = false;
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
            if (arg == "--verify-parity") {
                o.verify_parity = true;
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
            if (arg == "--depth") {
                o.depth = true;
                continue;
            }
            if (++i == argc)
                throw std::runtime_error("Missing argument for " + arg);
            const std::string value = argv[i];
            if (arg == "--output") {
                o.output = value;
                continue;
            }
            if (arg == "--images") {
                o.images = value;
                continue;
            }
            if (arg == "--overlay") {
                if (value != "selection" && value != "preview" && value != "crop" && value != "ellipsoid" && value != "window" && value != "markers" && value != "flash" && value != "affine")
                    throw std::runtime_error("Unknown overlay fixture");
                o.overlay = value;
                continue;
            }
            size_t used = 0;
            const auto number = std::stoll(value, &used);
            if (used != value.size() || number < 1 || number > 10000000)
                throw std::runtime_error("Invalid value for " + arg);
            if (arg == "--count")
                o.count = number;
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
        if (o.width > 4096 || o.height > 4096 || o.count > 1000000 || o.samples > 10000 || o.warmup > 1000)
            throw std::runtime_error("Benchmark reservation limit exceeded");
        return o;
    }
    core::SplatData scene(size_t count, int degree, float rest_amplitude = .1f, bool panorama = false, bool near = false, const Options* reference_cut = nullptr, bool spark = false, bool gpu_lod = false) {
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
        info.pSemaphores = &frame.completion_semaphore;
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
    Json run(const Options& o) {
        core::GpuBackendScope scope(core::GpuBackend::Metal);
        vis::VulkanContext context;
        if (!context.initHeadless())
            throw std::runtime_error(context.lastError());
        // Safe mode prevents reading/writing user preferences. Automatic keeps the
        // production Vulkan path while the native adapter is called directly.
        if (vis::UserPreferences::instance().viewerBackend() != rendering::ViewerBackend::Automatic)
            throw std::runtime_error("Benchmark preferences must be isolated");
        Json cases = Json::array();
        for (int degree : {0, 3}) {
            auto model = scene(o.count, degree, o.overlay == "affine" ? 1.f : .1f, o.equirect, o.near, nullptr, o.spark, o.gpu_lod);
            vis::MetalViewportRenderer metal;
            vis::VksplatViewportRenderer vulkan;
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
            request.frame_view.size = {o.width, o.height};
            if (o.near)
                request.frame_view.far_plane = .06f;
            request.frame_view.rasterization_scale = o.export_scale ? 2.f : 1.f;
            request.sh_degree = degree;
            request.gut = o.gut;
            request.splat_render_profile = o.portal ? 1 : 0;
            request.color_tonemapping = o.portal_tone ? 4 : 0;
            request.color_exposure = o.portal_tone ? 1.6f : 1.f;
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
            if (o.overlay == "selection" || o.overlay == "preview") {
                std::vector<float> mask(o.count);
                for (size_t n = 0; n < o.count; ++n)
                    mask[n] = float(n % 3);
                auto tensor = std::make_shared<core::Tensor>(core::Tensor::from_vector(mask, {o.count}, core::Device::GPU).to(core::DataType::UInt8));
                if (o.overlay == "selection") {
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
                throw std::runtime_error("Unsupported native benchmark frame");
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
                if (o.overlay == "selection") {
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
                auto result = native ? vis::legacyMetalResult(metal.render(context, model, request, Slot::Main))
                                     : vulkan.render(context, reference_cut ? *reference_cut : model, reference_request, false, Slot::Main);
                if (!result)
                    throw std::runtime_error(result.error());
                wait(context, *result);
            };
            auto complete = [&] {
                const auto status = metal.outputComplete(Slot::Main);
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
                const auto status = metal.gpuLodSelectionStatus(Slot::Main);
                const size_t expected = o.gpu_lod_budget ? 1 : o.count - 1 - (o.count + 59999) / 60000;
                if (!status.active || status.selected != expected || status.overflow || status.resident_chunks != (o.count + core::SplatLodTree::kChunkSplats - 1) / core::SplatLodTree::kChunkSplats)
                    throw std::runtime_error("Native LOD diagnostics differ from the completed GPU cut");
            }
            std::vector<double> native_times, vulkan_times;
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
                }
            }
            // GPU synchronization is measured; CPU image transfers are deliberately separate.
            auto pixels = core::Tensor::empty({size_t(o.height), size_t(o.width), 3}, core::Device::CPU, core::DataType::Float32);
            const auto read = metal.readColor(Slot::Main, pixels, 0, 0);
            if (!read)
                throw std::runtime_error(lfs::format_for_developer(read.error()));
            auto reference = vulkan.readOutputImage(context, Slot::Main);
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
            // The production Vulkan reference blends in FP16; bit equality with the
            // native FP32 blend is not its contract. Bound both local and RMS error.
            // Median depth has a hard coverage boundary at 0.5. FP16 reference and
            // FP32 native can disagree on boundary pixels; record the full image
            // error and enforce a separate bound, never silently drop those pixels.
            const double local_error = o.depth ? double(difference["depth_valid_max_error"]) : o.equirect && o.overlay == "markers" ? double(difference["marker_stable_max_error"])
                                                                                                                                    : double(difference["max_error"]);
            if (o.verify_parity && (local_error > 4. / 255 + 1e-7 ||
                                    double(difference["depth_coverage_disagreement_fraction"]) > .001 ||
                                    (difference.contains("marker_boundary_disagreement_fraction") && double(difference["marker_boundary_disagreement_fraction"]) > .001) ||
                                    double(difference[o.depth ? "depth_valid_rmse" : "rmse"]) > 1. / 255))
                throw std::runtime_error("Native image exceeds FP16-reference parity bounds (valid max 4/255, RMS 1/255, depth coverage 0.1%): SH" + std::to_string(degree) + " " + difference.dump());
            const auto native_stats = statistics(native_times), vulkan_stats = statistics(vulkan_times);
            cases.push_back({{"sh_degree", degree}, {"storage", degree ? "q16" : "sh0"}, {"metal", native_stats}, {"vulkan", vulkan_stats}, {"speedup_vulkan_over_metal", (o.subregion || (o.gut && o.lod)) ? Json(nullptr) : Json(double(vulkan_stats["median_ms"]) / double(native_stats["median_ms"]))}, {"image_difference", difference}});
        }
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        return {{"schema_version", 1}, {"metric", "completed_frame_wall_latency_ms"}, {"includes", "host encode, submission, GPU raster, output conversion, completion wait"}, {"excludes", "warmup, CPU image readback, desktop UI/compositor, frame pipelining"}, {"device", MTLCreateSystemDefaultDevice().name.UTF8String}, {"os", NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String}, {"compiler", __clang_version__}, {"scene_seed", 1939}, {"metal_debug_layer", std::getenv("MTL_DEBUG_LAYER") ? std::getenv("MTL_DEBUG_LAYER") : "unset"}, {"metal_shader_validation", std::getenv("MTL_SHADER_VALIDATION") ? std::getenv("MTL_SHADER_VALIDATION") : "unset"}, {"count", o.count}, {"width", o.width}, {"height", o.height}, {"warmup_pairs", o.warmup}, {"profile", o.portal ? "portal" : "studio"}, {"tone_fixture", o.portal_tone}, {"reference_resident_cut", o.gut && o.lod}, {"gpu_lod", o.gpu_lod}, {"gpu_lod_budget", o.gpu_lod_budget}, {"spark_opacity", o.spark}, {"lod", o.lod}, {"lod_logical", o.lod_logical}, {"lod_weights", o.lod_weights}, {"lod_debug", o.lod_debug}, {"gut", o.gut}, {"equirectangular", o.equirect}, {"near_fixture", o.near}, {"subregion", o.subregion}, {"reference_full_frame_crop", o.equirect && o.subregion}, {"mip", o.mip}, {"orthographic", o.ortho}, {"depth_view", o.depth}, {"overlay_fixture", o.overlay}, {"rasterization_scale", o.export_scale ? 2.f : 1.f}, {"samples_per_backend", o.samples}, {"process_peak_rss_bytes", usage.ru_maxrss}, {"cases", cases}};
    }
} // namespace
int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            const auto o = options(argc, argv);
            if (setenv("LFS_SAFE_MODE", "1", 1) != 0)
                throw std::runtime_error("Cannot isolate benchmark preferences");
            if (!core::gpu_backend_available(core::GpuBackend::Metal)) {
                std::puts("SKIP: resident Metal tensors require a compatible macOS/Metal device");
                return 77;
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
