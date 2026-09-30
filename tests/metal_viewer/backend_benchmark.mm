/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "metal_viewport_renderer.hpp"
#include "core/tensor_backend.hpp"
#include "preferences.hpp"
#include <Python.h>
#include <nlohmann/json.hpp>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
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
    bool mip = false, ortho = false, depth = false;
};
Options options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--smoke") {
            o.count = 512; o.width = 128; o.height = 96; o.warmup = 6; o.samples = 4;
            o.verify_parity = true;
            continue;
        }
        if (arg == "--verify-parity") { o.verify_parity = true; continue; }
        if (arg == "--mip") { o.mip = true; continue; }
        if (arg == "--ortho") { o.ortho = true; continue; }
        if (arg == "--depth") { o.depth = true; continue; }
        if (++i == argc) throw std::runtime_error("Missing argument for " + arg);
        const std::string value = argv[i];
        if (arg == "--output") { o.output = value; continue; }
        if (arg == "--images") { o.images = value; continue; }
        if(arg=="--overlay"){
            if(value!="selection"&&value!="preview"&&value!="crop"&&value!="ellipsoid"&&value!="window"&&value!="markers"&&value!="flash")
                throw std::runtime_error("Unknown overlay fixture");
            o.overlay=value;continue;
        }
        size_t used = 0;
        const auto number = std::stoll(value, &used);
        if (used != value.size() || number < 1 || number > 10000000)
            throw std::runtime_error("Invalid value for " + arg);
        if (arg == "--count") o.count = number;
        else if (arg == "--width") o.width = number;
        else if (arg == "--height") o.height = number;
        else if (arg == "--warmup") o.warmup = number;
        else if (arg == "--samples") o.samples = number;
        else throw std::runtime_error("Unknown option " + arg);
    }
    if (o.width > 4096 || o.height > 4096 || o.count > 1000000 || o.samples > 10000 || o.warmup > 1000)
        throw std::runtime_error("Benchmark reservation limit exceeded");
    return o;
}
core::SplatData scene(size_t count, int degree) {
    std::mt19937 random(1939);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::vector<float> means(count * 3), sh0(count * 3), scales(count * 3), rotation(count * 4, 0), opacity(count);
    std::vector<float> rest(degree ? count * 45 : 0);
    for (size_t i = 0; i < count; ++i) {
        means[3*i] = (unit(random) - .5f) * 3.6f;
        means[3*i+1] = (unit(random) - .5f) * 2.f;
        means[3*i+2] = -4.f - unit(random) * 4.f;
        for (int c = 0; c < 3; ++c) {
            sh0[3*i+c] = (unit(random) - .5f) * 2.f;
            scales[3*i+c] = -4.8f + unit(random) * .4f;
        }
        rotation[4*i] = 1.f;
        opacity[i] = 1.f + unit(random);
    }
    for (auto& value : rest) value = (unit(random) - .5f) * .1f;
    using core::Tensor; using core::Device;
    auto model = core::SplatData(degree,
        Tensor::from_vector(means, {count, 3}, Device::GPU),
        Tensor::from_vector(sh0, {count, 1, 3}, Device::GPU),
        degree ? Tensor::from_vector(rest, {count, 15, 3}, Device::GPU) : Tensor{},
        Tensor::from_vector(scales, {count, 3}, Device::GPU),
        Tensor::from_vector(rotation, {count, 4}, Device::GPU),
        Tensor::from_vector(opacity, {count, 1}, Device::GPU), 1.f);
    if (degree) {
        (void)model.apply_shN_value_quant();
        if (!model.shN_value_quantized()) throw std::runtime_error("SH3 fixture must use production Q16 storage");
    }
    return model;
}
void wait(vis::VulkanContext& context, const vis::VksplatViewportRenderer::RenderResult& frame) {
    if (!frame.image || !frame.completion_semaphore || !frame.completion_value)
        throw std::runtime_error("Missing real GPU frame completion");
    VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    info.semaphoreCount = 1; info.pSemaphores = &frame.completion_semaphore;
    info.pValues = &frame.completion_value;
    if (vkWaitSemaphores(context.device(), &info, 30'000'000'000ull) != VK_SUCCESS)
        throw std::runtime_error("GPU frame completion timed out or failed");
}
Json statistics(const std::vector<double>& raw) {
    auto sorted = raw;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double p) {
        const double index = p * (sorted.size() - 1);
        const size_t lo = size_t(index), hi = std::min(lo+1, sorted.size()-1);
        return sorted[lo] + (sorted[hi]-sorted[lo]) * (index-lo);
    };
    return {{"samples_ms", raw}, {"median_ms", percentile(.5)}, {"p95_ms", percentile(.95)}};
}
Json quality(const core::Tensor& native, const core::Tensor& reference, const glm::vec3 background, bool depth_view) {
    if (native.numel() != reference.numel()) throw std::runtime_error("Image shapes differ");
    double sum = 0, squares = 0, maximum = 0, native_signal = 0, reference_signal = 0;
    const auto a = native.ptr<float>(), b = reference.ptr<float>();
    for (size_t i = 0; i < native.numel(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("Non-finite benchmark image");
        const double error = std::abs(double(a[i])-b[i]);
        sum += error; squares += error*error; maximum = std::max(maximum, error);
        native_signal += std::abs(a[i]-background[i%3]);
        reference_signal += std::abs(b[i]-background[i%3]);
    }
    // Detect missing/background-only frames without imposing a hardware-dependent speed gate.
    if (native_signal < .1 || reference_signal < .1) throw std::runtime_error("Benchmark published an empty image");
    double valid_depth_max = 0, valid_depth_squares = 0;
    size_t valid_depth_channels = 0;
    size_t depth_coverage_disagreements = 0;
    if (depth_view) {
        for (size_t i=0; i<native.numel(); i+=3) {
            bool native_empty=true, reference_empty=true;
            double pixel_error=0;
            for (size_t c=0; c<3; ++c) {
                const float bg=std::round(background[c]*255)/255;
                native_empty = native_empty && std::abs(a[i+c]-bg)<1e-6f;
                reference_empty = reference_empty && std::abs(b[i+c]-bg)<1e-6f;
                pixel_error=std::max(pixel_error,std::abs(double(a[i+c])-b[i+c]));
            }
            if (native_empty != reference_empty) ++depth_coverage_disagreements;
            else {
                valid_depth_max=std::max(valid_depth_max,pixel_error);
                for(size_t c=0;c<3;++c){const double delta=double(a[i+c])-b[i+c]; valid_depth_squares+=delta*delta;}
                valid_depth_channels+=3;
            }
        }
    }
    const double mse = squares / native.numel();
    return {{"mae", sum/native.numel()}, {"rmse", std::sqrt(mse)}, {"max_error", maximum},
            {"psnr_db", mse == 0 ? Json(nullptr) : Json(-10*std::log10(mse))}, {"identical", mse == 0},
            {"depth_valid_max_error", depth_view ? Json(valid_depth_max) : Json(nullptr)},
            {"depth_valid_rmse", depth_view ? Json(std::sqrt(valid_depth_squares/std::max<size_t>(1,valid_depth_channels))) : Json(nullptr)},
            {"depth_coverage_disagreement_pixels", depth_coverage_disagreements},
            {"depth_coverage_disagreement_fraction", 3.*depth_coverage_disagreements/native.numel()}};
}
Json run(const Options& o) {
    core::GpuBackendScope scope(core::GpuBackend::Metal);
    vis::VulkanContext context;
    if (!context.initHeadless()) throw std::runtime_error(context.lastError());
    // Safe mode prevents reading/writing user preferences. Automatic keeps the
    // production Vulkan path while the native adapter is called directly.
    if (vis::UserPreferences::instance().viewerBackend() != rendering::ViewerBackend::Automatic)
        throw std::runtime_error("Benchmark preferences must be isolated");
    Json cases = Json::array();
    for (int degree : {0, 3}) {
        auto model = scene(o.count, degree);
        vis::MetalViewportRenderer metal;
        vis::VksplatViewportRenderer vulkan;
        rendering::ViewportRenderRequest request;
        request.frame_view.size = {o.width, o.height}; request.sh_degree = degree;
        request.frame_view.background_color = {.02f, .03f, .04f};
        request.mip_filter = o.mip;
        request.depth_view = o.depth;
        request.frame_view.orthographic = o.ortho;
        request.frame_view.ortho_scale = 32;
        if(o.overlay=="selection" || o.overlay=="preview"){
            std::vector<float> mask(o.count);
            for(size_t n=0;n<o.count;++n)mask[n]=float(n%3);
            auto tensor=std::make_shared<core::Tensor>(core::Tensor::from_vector(mask,{o.count},core::Device::GPU).to(core::DataType::UInt8));
            if(o.overlay=="selection"){request.overlay.has_selection=true;request.overlay.emphasis.mask=tensor;}
            else {request.overlay.emphasis.transient_mask.owned_mask=tensor;request.overlay.emphasis.transient_mask.mask=tensor.get();}
        }
        if(o.overlay=="crop"){
            rendering::GaussianScopedBoxFilter crop;
            crop.bounds.min={-.6f,-1,-9};crop.bounds.max={.6f,1,-3};crop.desaturate=true;
            request.filters.crop_region=crop;
        }
        if(o.overlay=="ellipsoid"){
            rendering::GaussianScopedEllipsoidFilter ellipsoid;
            ellipsoid.bounds.radii={1,1,2};ellipsoid.bounds.transform=glm::mat4(1);ellipsoid.bounds.transform[3].z=6;
            ellipsoid.desaturate=true;request.filters.ellipsoid_region=ellipsoid;
        }
        if(o.overlay=="window"){
            rendering::BoundingBox volume;volume.min={-2,-2,-9};volume.max={2,2,-3};
            request.filters.view_volume=volume;request.filters.screen_window=rendering::SelectionScreenWindow{};
            request.filters.dim_outside_view_volume=true;
        }
        if(o.overlay=="markers")request.overlay.markers.show_center_markers=true;
        if(o.overlay=="flash"){
            static const std::vector<glm::mat4> transforms={glm::mat4(1)};
            request.scene.model_transforms=&transforms;
            request.scene.transform_indices=std::make_shared<core::Tensor>(core::Tensor::zeros({o.count},core::Device::GPU,core::DataType::Int32));
            request.overlay.emphasis.emphasized_node_mask={true};request.overlay.emphasis.flash_intensity=.8f;
        }
        if (!vis::MetalViewportRenderer::supports(model, request)) throw std::runtime_error("Unsupported native benchmark frame");
        auto frame = [&](bool native) {
            auto result = native ? metal.render(context, model, request, Slot::Main)
                                 : vulkan.render(context, model, request, false, Slot::Main);
            if (!result) throw std::runtime_error(result.error());
            wait(context, *result);
        };
        auto complete = [&] {
            const auto status = metal.outputComplete(Slot::Main);
            if (!status) throw std::runtime_error(status.error());
            return *status;
        };
        for (int n = 0; n < o.warmup; ++n) { frame(n%2 == 0); frame(n%2 != 0); }
        if (!complete()) throw std::runtime_error("Native reservation did not converge during warmup");
        std::vector<double> native_times, vulkan_times;
        for (int n = 0; n < o.samples; ++n) {
            // AB/BA pairs reduce order, thermal and drift bias; keep every raw sample.
            for (int j = 0; j < 2; ++j) {
                const bool native = (n+j)%2 == 0;
                const auto start = Clock::now(); frame(native);
                const auto ms = std::chrono::duration<double, std::milli>(Clock::now()-start).count();
                if (!(ms > 0) || !std::isfinite(ms)) throw std::runtime_error("Invalid timing sample");
                if (native && !complete()) throw std::runtime_error("Partial native frame in measured sample");
                (native ? native_times : vulkan_times).push_back(ms);
            }
        }
        // GPU synchronization is measured; CPU image transfers are deliberately separate.
        auto pixels = core::Tensor::empty({size_t(o.height), size_t(o.width), 3}, core::Device::CPU, core::DataType::Float32);
        const auto read = metal.readColor(Slot::Main, pixels, 0, 0);
        if (!read) throw std::runtime_error(read.error());
        const auto reference = vulkan.readOutputImage(context, Slot::Main);
        if (!reference) throw std::runtime_error(reference.error());
        if (!o.images.empty()) {
            const auto save = [&](const core::Tensor& image, const char* backend) {
                std::ofstream stream(o.images + "-sh" + std::to_string(degree) + "-" + backend + ".ppm", std::ios::binary);
                stream << "P6\n" << o.width << ' ' << o.height << "\n255\n";
                for (size_t i=0; i<image.numel(); ++i) {
                    const auto value=static_cast<unsigned char>(std::lround(std::clamp(image.ptr<float>()[i],0.f,1.f)*255));
                    stream.write(reinterpret_cast<const char*>(&value),1);
                }
                if (!stream) throw std::runtime_error("Cannot write diagnostic image");
            };
            save(pixels,"metal"); save(**reference,"vulkan");
        }
        const auto difference = quality(pixels, **reference, request.frame_view.background_color, o.depth);
        // The production Vulkan reference blends in FP16; bit equality with the
        // native FP32 blend is not its contract. Bound both local and RMS error.
        // Median depth has a hard coverage boundary at 0.5. FP16 reference and
        // FP32 native can disagree on boundary pixels; record the full image
        // error and enforce a separate bound, never silently drop those pixels.
        const double local_error = o.depth ? double(difference["depth_valid_max_error"]) : double(difference["max_error"]);
        if (o.verify_parity && (local_error > 4./255 + 1e-7 ||
                               double(difference["depth_coverage_disagreement_fraction"]) > .001 ||
                               double(difference[o.depth ? "depth_valid_rmse" : "rmse"]) > 1./255))
            throw std::runtime_error("Native image exceeds FP16-reference parity bounds (valid max 4/255, RMS 1/255, depth coverage 0.1%)");
        const auto native_stats = statistics(native_times), vulkan_stats = statistics(vulkan_times);
        cases.push_back({{"sh_degree", degree}, {"storage", degree ? "q16" : "sh0"},
            {"metal", native_stats}, {"vulkan", vulkan_stats},
            {"speedup_vulkan_over_metal", double(vulkan_stats["median_ms"])/double(native_stats["median_ms"])},
            {"image_difference", difference}});
    }
    rusage usage{}; getrusage(RUSAGE_SELF, &usage);
    return {{"schema_version", 1}, {"metric", "completed_frame_wall_latency_ms"},
        {"includes", "host encode, submission, GPU raster, output conversion, completion wait"},
        {"excludes", "warmup, CPU image readback, desktop UI/compositor, frame pipelining"},
        {"device", MTLCreateSystemDefaultDevice().name.UTF8String},
        {"os", NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String},
        {"compiler", __clang_version__}, {"scene_seed", 1939},
        {"metal_debug_layer", std::getenv("MTL_DEBUG_LAYER") ? std::getenv("MTL_DEBUG_LAYER") : "unset"},
        {"metal_shader_validation", std::getenv("MTL_SHADER_VALIDATION") ? std::getenv("MTL_SHADER_VALIDATION") : "unset"},
        {"count", o.count}, {"width", o.width}, {"height", o.height}, {"warmup_pairs", o.warmup},
        {"mip", o.mip}, {"orthographic", o.ortho}, {"depth_view", o.depth}, {"overlay_fixture",o.overlay},
        {"samples_per_backend", o.samples}, {"process_peak_rss_bytes", usage.ru_maxrss}, {"cases", cases}};
}
} // namespace
int main(int argc, char** argv) { @autoreleasepool {
    try {
        const auto o = options(argc, argv);
        if (setenv("LFS_SAFE_MODE", "1", 1) != 0)
            throw std::runtime_error("Cannot isolate benchmark preferences");
        if (!core::gpu_backend_available(core::GpuBackend::Metal)) {
            std::puts("SKIP: resident Metal tensors require a compatible macOS/Metal device"); return 77;
        }
        Py_Initialize();
        const auto report = run(o).dump(2);
        if (!o.output.empty()) {
            std::ofstream output(o.output); output << report << '\n'; output.flush();
            if (!output) throw std::runtime_error("Cannot write benchmark report");
        }
        std::puts(report.c_str());
        Py_Finalize();
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
} }
