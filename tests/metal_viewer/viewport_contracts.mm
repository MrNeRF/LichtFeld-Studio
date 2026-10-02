/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_backend.hpp"
#include "metal_viewport_renderer.hpp"
#include "point_cloud_vulkan_renderer.hpp"
#include "preferences.hpp"
#include "vksplat_viewport_renderer.hpp"
#include <Python.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <unistd.h>

using namespace lfs;
static void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
static void transparent_threshold_contract(vis::VulkanContext& context) {
    using core::Device;
    using core::Tensor;
    rendering::ViewportRenderRequest request;
    request.frame_view.size = {97, 97};
    request.gut = true;
    request.sh_degree = 0;
    request.transparent_background = true;
    const auto k = request.frame_view.getCameraIntrinsics();
    constexpr size_t x = 52, y = 48;
    const double dx = (double(x) + .5 - k.center_x) / k.focal_x;
    const double dy = (double(y) + .5 - k.center_y) / k.focal_y;
    constexpr float scale = .08f;
    const double power = .5 * 9 * (dx * dx + dy * dy) / (1 + dx * dx + dy * dy) / (double(scale) * scale);
    core::SplatData model(0,
                          Tensor::from_vector(std::vector<float>{0, 0, -3}, {1, 3}, Device::GPU),
                          Tensor::from_vector(std::vector<float>{1, 0, 0}, {1, 1, 3}, Device::GPU), {},
                          Tensor::from_vector(std::vector<float>{std::log(scale), std::log(scale), std::log(scale)}, {1, 3}, Device::GPU),
                          Tensor::from_vector(std::vector<float>{1, 0, 0, 0}, {1, 4}, Device::GPU),
                          Tensor::from_vector(std::vector<float>{0}, {1, 1}, Device::GPU), 1.f);
    vis::MetalViewportRenderer renderer;
    for (bool visible : {false, true}) {
        // The positive case remains above the real FP32 threshold but rounds
        // below it when alpha is stored in the half color texture. It failed
        // when presentation used color.a instead of the retained depth alpha.
        const double pixel_alpha = .5 / 255 + (visible ? 2e-7 : -2e-7);
        const double opacity = pixel_alpha * std::exp(power);
        require(opacity > 0 && opacity < 1, "Invalid transparent threshold oracle");
        model.opacity_raw() = Tensor::from_vector(std::vector<float>{float(std::log(opacity / (1 - opacity)))}, {1, 1}, Device::GPU);
        const auto frame = renderer.render(context, model, request, vis::RenderTargetId{1});
        require(bool(frame), "Transparent threshold render failed");
        auto pixels = Tensor::empty({97, 97, 4}, Device::CPU);
        require(bool(renderer.readColor(vis::RenderTargetId{1}, pixels, 0, 0)), "Transparent threshold readback failed");
        const auto pixel = pixels.ptr<float>() + (y * 97 + x) * 4;
        if (visible)
            require(pixel[3] > 0 && pixel[0] > .7f, "FP16 storage erased valid FP32 threshold coverage");
        else
            require(pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0 && pixel[3] == 0, "Empty transparent tail retained RGB/alpha");
    }
}
static void partial_selection_mask_contract(vis::VulkanContext& context) {
    using core::Device;
    using core::Tensor;
    // A short mask has an implicit unselected suffix. Exercise admission and
    // the production adapter, not only the lower-level shader buffer contract.
    for (const auto backend : {core::GpuBackend::Metal, core::GpuBackend::Vulkan}) {
        core::GpuBackendScope scope(backend);
        core::SplatData model(0,
                              Tensor::from_vector(std::vector<float>{-.45f, 0, -3, 0, 0, -3, .45f, 0, -3}, {3, 3}, Device::GPU),
                              Tensor::full({3, 1, 3}, .5f, Device::GPU), {},
                              Tensor::full({3, 3}, -3.f, Device::GPU),
                              Tensor::from_vector(std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, {3, 4}, Device::GPU),
                              Tensor::full({3, 1}, 4.f, Device::GPU), 1.f);
        auto adapter = vis::createSceneRenderer();
        rendering::ViewportRenderRequest request;
        request.frame_view.size = {96, 64};
        request.sh_degree = 0;
        const auto capture = [&] {
            require(vis::MetalViewportRenderer::supports(model, request), "Partial native selection mask unnecessarily rejected Metal");
            const auto frame = adapter->render(context, model, request, true, vis::RenderTargetId{1}, false, true);
            require(frame.has_value() && (frame->generation >> 63) != 0, "Partial selection mask fell back to Vulkan");
            auto rgb = Tensor::empty({64, 96, 3}, Device::CPU);
            const auto ticket = adapter->submitReadOutputImageIntoCpuHwcTicket(context, vis::RenderTargetId{1}, rgb, 0, 0);
            require(ticket.has_value() && adapter->waitReadbackTicket(*ticket).has_value(), "Partial selection readback failed");
            return rgb;
        };
        const std::array<uint32_t, 3> reordered_cut{2, 0, 1};
        for (bool gut : {false, true})
            for (bool reordered : {false, true}) {
                request.gut = gut;
                request.lod_indices = reordered ? reordered_cut.data() : nullptr;
                request.lod_count = reordered ? reordered_cut.size() : 0;
                const auto plain = capture();
                for (const auto dtype : {core::DataType::UInt8, core::DataType::Bool}) {
                    const auto full = std::make_shared<Tensor>(Tensor::from_vector(std::vector<float>{1, 0, 0}, {3}, Device::GPU).to(dtype));
                    const auto short_mask = std::make_shared<Tensor>(Tensor::from_vector(std::vector<float>{1}, {1}, Device::GPU).to(dtype));
                    for (bool preview : {false, true}) {
                        const auto assign = [&](const std::shared_ptr<Tensor>& mask) {
                            request.overlay = {};
                            if (preview) {
                                request.overlay.emphasis.transient_mask.owned_mask = mask;
                                request.overlay.emphasis.transient_mask.mask = mask.get();
                            } else {
                                request.overlay.has_selection = true;
                                request.overlay.emphasis.mask = mask;
                            }
                        };
                        assign(full);
                        const auto expected = capture();
                        require(std::memcmp(plain.ptr<float>(), expected.ptr<float>(), plain.bytes()) != 0, "Selection oracle never changed the image");
                        assign(short_mask);
                        const auto actual = capture();
                        require(std::memcmp(expected.ptr<float>(), actual.ptr<float>(), expected.bytes()) == 0, "Short selection mask changed its implicit unselected suffix");
                    }
                }
                request.overlay = {};
            }
        request.overlay.has_selection = true;
        request.overlay.emphasis.mask = std::make_shared<Tensor>(Tensor::full({1}, 1.f, Device::GPU));
        require(!vis::MetalViewportRenderer::supports(model, request), "Non-byte selection mask accepted");
        request.overlay.emphasis.mask = std::make_shared<Tensor>(Tensor::full({1}, 1.f, Device::CPU).to(core::DataType::UInt8));
        require(!vis::MetalViewportRenderer::supports(model, request), "Non-resident selection mask accepted");
        request.overlay.emphasis.mask = std::make_shared<Tensor>(Tensor::full({3, 2}, 1.f, Device::GPU).to(core::DataType::UInt8).slice(1, 0, 1));
        require(!vis::MetalViewportRenderer::supports(model, request), "Strided selection mask accepted");
    }
}
static void multi_target_auto_contract(vis::VulkanContext& context, bool compare_vulkan) {
    using core::Device;
    using core::Tensor;
    constexpr std::array<vis::RenderTargetId, 8> ids{{{5}, {37}, {1024}, {90001}, {17}, {700}, {0xFFFFFFFEu}, {23}}};
    for (const auto backend : {core::GpuBackend::Metal, core::GpuBackend::Vulkan}) {
        core::GpuBackendScope scope(backend);
        core::SplatData model(0,
                              Tensor::from_vector(std::vector<float>{0, 0, -3}, {1, 3}, Device::GPU),
                              Tensor::from_vector(std::vector<float>{1, 0, 0}, {1, 1, 3}, Device::GPU), {},
                              Tensor::full({1, 3}, -2.f, Device::GPU),
                              Tensor::from_vector(std::vector<float>{1, 0, 0, 0}, {1, 4}, Device::GPU),
                              Tensor::full({1, 1}, 4.f, Device::GPU), 1.f);
        auto adapter = vis::createSceneRenderer();
        std::array<Tensor, ids.size()> snapshots;
        const auto request_for = [](size_t index) {
            rendering::ViewportRenderRequest request;
            request.frame_view.size = {int(49 + index * 2), int(33 + index * 2)};
            request.sh_degree = 0;
            request.gut = (index & 1) != 0;
            request.frame_view.translation.x = float(index) * .01f;
            return request;
        };
        const auto capture = [&](size_t index) {
            const auto request = request_for(index);
            const auto size = request.frame_view.size;
            auto rgb = Tensor::empty({size_t(size.y), size_t(size.x), 3}, Device::CPU);
            const auto ticket = adapter->submitReadOutputImageIntoCpuHwcTicket(context, ids[index], rgb, 0, 0);
            require(ticket && vis::MetalViewportRenderer::nativeTicket(*ticket), "Dynamic view capture lost native identity");
            require(adapter->waitReadbackTicket(*ticket).has_value(), "Dynamic view capture failed");
            require(rgb.ptr<float>()[((size.y / 2) * size.x + size.x / 2) * 3] > .5f, "Dynamic view produced no visible splat");
            return rgb;
        };
        for (size_t index = 0; index < ids.size(); ++index) {
            const auto request = request_for(index);
            const auto frame = adapter->render(context, model, request, true, ids[index], false, true);
            require(frame && frame->viewer_backend == rendering::ViewerBackend::Metal && (frame->generation >> 63),
                    "Auto did not select native Metal for a compatible dynamic target");
            require(frame->size == request.frame_view.size && adapter->hasRenderTarget(ids[index]), "Dynamic target extent/ownership differs");
            snapshots[index] = capture(index);
        }
        for (size_t index = 0; index < ids.size(); ++index) {
            const auto current = capture(index);
            require(std::memcmp(current.data_ptr(), snapshots[index].data_ptr(), current.bytes()) == 0,
                    "Rendering another target changed a cached native view");
        }
        require(!adapter->render(context, model, request_for(0), true, {}).has_value(), "Invalid target zero was accepted");
        auto cancelled = Tensor::full({33, 49, 3}, -99.f, Device::CPU);
        auto survivor = Tensor::empty({35, 51, 3}, Device::CPU);
        const auto cancelled_ticket = adapter->submitReadOutputImageIntoCpuHwcTicket(context, ids[0], cancelled, 0, 0);
        const auto survivor_ticket = adapter->submitReadOutputImageIntoCpuHwcTicket(context, ids[1], survivor, 0, 0);
        require(cancelled_ticket && survivor_ticket, "Pending target tickets failed");
        require(adapter->releaseRenderTarget(ids[0]) && !adapter->hasRenderTarget(ids[0]), "Closed native target retained ownership");
        const auto cancelled_status = adapter->pollReadbackTicket(*cancelled_ticket);
        require(cancelled_status && *cancelled_status == vis::VksplatViewportRenderer::ReadbackTicketStatus::Failed,
                "Closed target did not fail its pending readback");
        require(!adapter->render(context, model, request_for(0), true, ids[0]).has_value(), "A retired target ID was reused");
        require(adapter->waitReadbackTicket(*survivor_ticket).has_value(), "Closing one target failed a different target's ticket");
        require(std::memcmp(survivor.data_ptr(), snapshots[1].data_ptr(), survivor.bytes()) == 0,
                "A ticket delivered pixels from a different view");
        for (size_t i = 0; i < cancelled.numel(); ++i)
            require(cancelled.ptr<float>()[i] == -99.f, "Closed target wrote to its abandoned host destination");
        for (size_t index = 1; index < ids.size(); ++index) {
            const auto current = capture(index);
            require(std::memcmp(current.data_ptr(), snapshots[index].data_ptr(), current.bytes()) == 0,
                    "Closing another target changed this native view");
        }
        if (!compare_vulkan)
            continue;
        // Unsupported native inputs must report an error, never switch backend.
        auto unsupported = request_for(1);
        unsupported.overlay.has_selection = true;
        unsupported.overlay.emphasis.mask = std::make_shared<Tensor>(Tensor::full({1}, 1.f, Device::CPU).to(core::DataType::UInt8));
        require(!adapter->render(context, model, unsupported, true, ids[1], false, true), "Unsupported native input was silently accepted");
        require(adapter->render(context, model, request_for(1), true, ids[1], false, true).has_value(), "Native renderer did not recover after an invalid request");
    }
}
static void run(bool compare_vulkan) {
    core::GpuBackendScope scope(core::GpuBackend::Metal);
    vis::VulkanContext context;
    require(context.initHeadless(), context.lastError().c_str());
    vis::MetalViewportRenderer renderer;
    using core::Device;
    using core::Tensor;
    core::SplatData model(0,
                          Tensor::from_vector(std::vector<float>{0, 0, -3}, {1, 3}, Device::GPU),
                          Tensor::from_vector(std::vector<float>{1, 0, 0}, {1, 1, 3}, Device::GPU), {},
                          Tensor::from_vector(std::vector<float>{-2, -2, -2}, {1, 3}, Device::GPU),
                          Tensor::from_vector(std::vector<float>{1, 0, 0, 0}, {1, 4}, Device::GPU),
                          Tensor::from_vector(std::vector<float>{4}, {1, 1}, Device::GPU), 1.f);
    rendering::ViewportRenderRequest request;
    request.frame_view.size = {96, 64};
    request.sh_degree = 0;
    std::vector<glm::mat4> transforms(2, glm::mat4(1));
    request.scene.model_transforms = &transforms;
    require(!vis::MetalViewportRenderer::supports(model, request), "Multiple objects accepted without indices");
    const auto malformed = renderer.render(context, model, request, vis::RenderTargetId{1});
    require(!malformed && malformed.error().code() == lfs::ErrorCode::InvalidArgument &&
                malformed.error().domain() == lfs::ErrorDomain::Rendering,
            "Malformed scene lost its typed argument error");
    const auto unknown = renderer.pollReadback(uint64_t{1} << 63, false);
    require(!unknown && unknown.error().code() == lfs::ErrorCode::NotFound,
            "Unknown ticket lost its typed lookup error");
    auto empty_destination = Tensor::empty({64, 96, 3}, Device::CPU);
    const auto empty_read = renderer.readColor(vis::RenderTargetId{1},
                                               empty_destination, 0, 0);
    require(!empty_read && empty_read.error().code() == lfs::ErrorCode::FailedPrecondition,
            "Empty output lost its typed precondition error");
    // Desktop single-node scenes omit the per-primitive index table.
    transforms.resize(1);
    require(vis::MetalViewportRenderer::supports(model, request), "Implicit single-object frame rejected");
    {
        auto lod_request = request;
        lod_request.transparent_background = true;
        uint32_t index = 0xffffffffu;
        lod_request.lod_indices = &index;
        lod_request.lod_count = 1;
        require(vis::MetalViewportRenderer::supports(model, lod_request), "Resident LOD cut incorrectly fell back");
        for (size_t count : {size_t(1), size_t(0)}) {
            lod_request.lod_count = count;
            const auto result = renderer.render(context, model, lod_request, vis::RenderTargetId{1});
            require(bool(result), "Invalid/empty LOD cut failed instead of publishing empty coverage");
            auto pixels = Tensor::empty({64, 96, 4}, Device::CPU);
            require(bool(renderer.readColor(vis::RenderTargetId{1}, pixels, 0, 0)), "Empty LOD readback failed");
            require(pixels.ptr<float>()[((32 * 96 + 48) * 4) + 3] == 0, "Invalid/empty LOD cut retained old visible coverage");
        }
    }
    for (int frame = 0; frame < 12; ++frame) {
        const auto output = renderer.render(context, model, request, vis::RenderTargetId{1});
        if (!output)
            throw std::runtime_error(lfs::format_for_developer(output.error()));
        require(output->image && output->image_view && output->completion_semaphore, "Missing native presentation handles");
        const auto size = request.frame_view.size;
        auto pixels = Tensor::empty({size_t(size.y), size_t(size.x), 4}, Device::CPU, core::DataType::Float32);
        auto read = renderer.readColor(vis::RenderTargetId{1}, pixels, 0, 0);
        if (!read)
            throw std::runtime_error(lfs::format_for_developer(read.error()));
        if (frame == 0) {
            const auto invalid = renderer.readColor(vis::RenderTargetId{1},
                                                    pixels, -1, 0);
            require(!invalid && invalid.error().code() == lfs::ErrorCode::InvalidArgument,
                    "Invalid readback destination lost its typed argument error");
        }
        const size_t center = ((size.y / 2) * size.x + size.x / 2) * 4;
        require(pixels.ptr<float>()[center] > .5f, "Native camera or color transfer differs");
        require(pixels.ptr<float>()[center] > pixels.ptr<float>()[center + 1], "SH0 channels differ");
        const auto depth = renderer.readDepth({.pixel = size / 2, .source_size = size, .target = vis::RenderTargetId{1}});
        if (!depth)
            throw std::runtime_error(lfs::format_for_developer(depth.error()));
        require(std::abs(*depth - 3.f) < 1e-4f, "Native desktop depth differs");
        auto asynchronous = Tensor::full({size_t(size.y) + 2, size_t(size.x) + 2, 3}, -1.f, Device::CPU);
        const auto ticket = renderer.submitReadback(vis::RenderTargetId{1}, asynchronous, 1, 1, false);
        require(ticket.has_value(), "Native asynchronous color submit failed");
        require(renderer.outstandingReadbacks() == 1, "Native ticket not tracked");
        const auto ready = renderer.pollReadback(*ticket, true);
        require(ready.has_value() && *ready == vis::VksplatViewportRenderer::ReadbackTicketStatus::Ready,
                "Native asynchronous color delivery failed");
        require(asynchronous.ptr<float>()[0] == -1.f, "Readback overwrote destination border");
        const size_t async_center = (((size.y / 2) + 1) * (size.x + 2) + (size.x / 2) + 1) * 3;
        require(asynchronous.ptr<float>()[async_center] == pixels.ptr<float>()[center], "Asynchronous color differs");
        auto plane = Tensor::full({size_t(size.y), size_t(size.x)}, -1.f, Device::CPU);
        const auto depth_ticket = renderer.submitReadback(vis::RenderTargetId{1}, plane, 0, 0, true);
        require(depth_ticket.has_value() && renderer.pollReadback(*depth_ticket, true).has_value(), "Depth plane ticket failed");
        require(std::abs(plane.ptr<float>()[(size.y / 2) * size.x + size.x / 2] - 3.f) < 1e-4f, "Asynchronous depth differs");
        const auto abandoned = renderer.submitReadback(vis::RenderTargetId{1}, plane, 0, 0, true);
        require(abandoned.has_value(), "Abandoned ticket submit failed");
        renderer.abandonReadback(*abandoned);
        plane.fill_(-7.f);
        require(!renderer.pollReadback(*abandoned, true).has_value(), "Abandoned ticket delivered");
        require(plane.ptr<float>()[0] == -7.f, "Abandoned ticket wrote to host destination");
        if (frame == 5)
            request.frame_view.size = {80, 48};
        if (frame == 6) {
            request.frame_view.size = {96, 64};
            model.scaling_raw() = model.scaling_raw().to(core::DataType::Float16);
            model.rotation_raw() = model.rotation_raw().to(core::DataType::Float16);
            model.opacity_raw() = model.opacity_raw().to(core::DataType::Float16);
            require(vis::MetalViewportRenderer::supports(model, request), "Compact geometry rejected");
        }
        if (frame == 9) {
            request.frame_view.orthographic = true;
            request.frame_view.ortho_scale = 32;
        }
    }
    request.overlay.markers.show_rings = true;
    require(vis::MetalViewportRenderer::supports(model, request), "Native rings rejected");
    request.overlay.markers.show_rings = false;
    request.frame_view.orthographic = false;
    const auto snapshot = [&](rendering::ViewportRenderRequest& r) {
        const auto frame = renderer.render(context, model, r, vis::RenderTargetId{1});
        if (!frame)
            throw std::runtime_error(lfs::format_for_developer(frame.error()));
        auto pixels = Tensor::empty({64, 96, 3}, Device::CPU, core::DataType::Float32);
        const auto read = renderer.readColor(vis::RenderTargetId{1}, pixels, 0, 0);
        if (!read)
            throw std::runtime_error(lfs::format_for_developer(read.error()));
        return pixels;
    };
    rendering::GaussianScopedBoxFilter crop;
    crop.bounds.min = {-1, -1, -4};
    crop.bounds.max = {1, 1, -2};
    request.filters.crop_region = crop;
    require(snapshot(request).ptr<float>()[((64 / 2) * 96 + 96 / 2) * 3] > .5f, "Inside crop lost splat");
    request.filters.crop_region->inverse = true;
    require(snapshot(request).ptr<float>()[((64 / 2) * 96 + 96 / 2) * 3] < .1f, "Inverse crop did not cull splat");
    request.filters.crop_region->desaturate = true;
    const auto dim = snapshot(request);
    const auto center = ((64 / 2) * 96 + 96 / 2) * 3;
    require(std::abs(dim.ptr<float>()[center] - dim.ptr<float>()[center + 1]) < .01f, "Crop desaturation differs");
    request.filters = {};
    request.overlay.has_selection = true;
    request.overlay.emphasis.mask = std::make_shared<Tensor>(Tensor::from_vector(std::vector<float>{1}, {1}, Device::GPU).to(core::DataType::UInt8));
    const auto selected = snapshot(request);
    require(selected.ptr<float>()[center + 1] > .2f, "Committed selection tint missing");
    request.overlay.has_selection = false;
    request.overlay.emphasis.mask.reset();
    request.overlay.markers.show_center_markers = true;
    const auto marker = snapshot(request);
    require(marker.ptr<float>()[center + 1] > .5f && marker.ptr<float>()[center] < .1f, "Native center marker missing");
    request.gut = true;
    require(vis::MetalViewportRenderer::supports(model, request), "Native 3DGUT frame rejected");
    request.overlay.markers.show_center_markers = false;
    const auto gut_pixels = snapshot(request);
    require(gut_pixels.ptr<float>()[center] > .5f, "Native 3DGUT ray contribution missing");
    const auto gut_depth = renderer.readDepth({.pixel = {48, 32}, .source_size = {96, 64}, .target = vis::RenderTargetId{1}});
    const auto k = request.frame_view.getCameraIntrinsics();
    const float x = (48.5f - k.center_x) / k.focal_x, y = (32.5f - k.center_y) / k.focal_y;
    const float analytic_depth = 3.f / (1.f + x * x + y * y);
    require(gut_depth.has_value() && std::abs(*gut_depth - analytic_depth) < 1e-4f,
            "3DGUT depth did not use the closest point on the pixel ray");
    request.equirectangular = true;
    require(vis::MetalViewportRenderer::supports(model, request), "Native panorama rejected");
    require(snapshot(request).ptr<float>()[center] > .3f, "Panorama lost the forward hemisphere");
    const float azimuth = float(2 * M_PI) * (48.5f / 96.f - .5f);
    const float elevation = float(M_PI) * (32.5f / 64.f - .5f);
    const float ray_z = std::cos(azimuth) * std::cos(elevation);
    const auto panorama_depth = renderer.readDepth({.pixel = {48, 32}, .source_size = {96, 64}, .target = vis::RenderTargetId{1}});
    require(panorama_depth.has_value() && std::abs(*panorama_depth - 3.f * ray_z * ray_z) < 1e-4f,
            "Panorama depth did not use its spherical pixel ray");
    model.means_raw() = Tensor::from_vector(std::vector<float>{0, 0, 3}, {1, 3}, Device::GPU);
    const auto seam = snapshot(request);
    const size_t seam_row = size_t(32) * 96 * 3;
    require(seam.ptr<float>()[seam_row] > .3f && seam.ptr<float>()[seam_row + 95 * 3] > .3f,
            "Panorama clipped the rear hemisphere or lost a longitude seam");
    // Exporting a tile must keep full-camera rays and produce the same pixels.
    request.frame_view.size = {31, 33};
    request.frame_view.subregion_full_size = {96, 64};
    request.frame_view.subregion_origin = {65, 15};
    auto tile = Tensor::empty({33, 31, 3}, Device::CPU, core::DataType::Float32);
    require(renderer.render(context, model, request, vis::RenderTargetId{1}).has_value(), "Panorama subregion failed");
    require(renderer.readColor(vis::RenderTargetId{1}, tile, 0, 0).has_value(), "Panorama subregion readback failed");
    for (size_t y = 0; y < 33; ++y)
        for (size_t x = 0; x < 31; ++x)
            for (size_t c = 0; c < 3; ++c)
                require(std::abs(tile.ptr<float>()[(y * 31 + x) * 3 + c] - seam.ptr<float>()[((y + 15) * 96 + x + 65) * 3 + c]) < 1.f / 255,
                        "Panorama subregion changed the full-camera image");
    request.frame_view.size = {96, 64};
    request.frame_view.subregion_full_size = request.frame_view.subregion_origin = {0, 0};
    model.means_raw() = Tensor::from_vector(std::vector<float>{0, 0, -3}, {1, 3}, Device::GPU);
    request.gut = false;
    require(vis::MetalViewportRenderer::supports(model, request), "3DGS panorama was rejected");
    const auto gaussian_panorama = snapshot(request);
    require(gaussian_panorama.ptr<float>()[center] > .3f, "3DGS panorama lost the forward hemisphere");
    request.gut = true;
    request.equirectangular = false;
    require(renderer.releaseAll().has_value(), "Native scene release failed");
    require(renderer.size(vis::RenderTargetId{1}) == glm::ivec2(0), "Released slot retained output");
    request.gut = false;
    request.overlay.markers.show_center_markers = false;
    // Jitter changes the projection, but must not perpetually request a
    // new scene frame and reset temporal reconstruction convergence.
    const auto calibrated = request.frame_view.getCameraIntrinsics();
    request.frame_view.containment_intrinsics = calibrated;
    request.frame_view.intrinsics_override = calibrated;
    const auto stable = renderer.render(context, model, request, vis::RenderTargetId{1});
    require(stable.has_value() && renderer.outputComplete(vis::RenderTargetId{1}).has_value(), "Stable jitter fixture failed");
    request.frame_view.intrinsics_override->center_x += .25f;
    request.frame_view.intrinsics_override->center_y -= .125f;
    const auto jittered = renderer.render(context, model, request, vis::RenderTargetId{1});
    require(jittered.has_value() && !jittered->lod_streaming_active, "Temporal jitter requested perpetual refinement");
    request.frame_view.containment_intrinsics.reset();
    request.frame_view.intrinsics_override.reset();
    {
        auto adapter = vis::createSceneRenderer();
        uint64_t first_ticket = 0;
        for (const auto slot : {vis::RenderTargetId{1},
                                vis::RenderTargetId{2},
                                vis::RenderTargetId{3},
                                vis::RenderTargetId{4}}) {
            const auto frame = adapter->render(context, model, request, true, slot, false, true);
            if (!frame)
                throw std::runtime_error(frame.error());
            require((frame->generation >> 63) != 0, "Requested native export used Vulkan");
            const auto rgba = adapter->readOutputImageRgba8(context, slot);
            require(rgba.has_value(), "Native adapter capture failed");
            auto rgb = Tensor::empty({64, 96, 3}, Device::CPU, core::DataType::Float32);
            const auto ticket = adapter->submitReadOutputImageIntoCpuHwcTicket(context, slot, rgb, 0, 0);
            require(ticket.has_value() && vis::MetalViewportRenderer::nativeTicket(*ticket), "Adapter ticket lost backend identity");
            if (!first_ticket)
                first_ticket = *ticket;
            require(adapter->waitReadbackTicket(*ticket).has_value(), "Adapter ticket delivery failed");
            require(rgb.ptr<float>()[center] > .5f, "Native adapter export lost splat");
        }
        model.opacity_raw() = Tensor::full({1, 1}, -4.f, Device::GPU).to(core::DataType::Float16);
        adapter->setDepthCaptureMode(true, true);
        const auto expected_frame = adapter->render(context, model, request, true, vis::RenderTargetId{4}, false, true);
        require(expected_frame.has_value() && (expected_frame->generation >> 63) != 0, "Expected-depth capture lost native backend");
        const auto expected = adapter->readPreviewDepth(context, vis::RenderTargetId{4});
        require(expected.has_value() && std::abs((*expected)->ptr<float>()[32 * 96 + 48] - 3.f) < 1e-3f, "Expected-depth capture differs");
        require((*expected)->ptr<float>()[0] >= 1e9f, "Empty expected depth lost sentinel");
        adapter->setDepthCaptureMode(true, false);
        require(adapter->render(context, model, request, true, vis::RenderTargetId{4}, false, true).has_value(), "Median capture failed");
        const auto median = adapter->readPreviewDepth(context, vis::RenderTargetId{4});
        require(median.has_value() && (*median)->ptr<float>()[32 * 96 + 48] >= 1e9f, "Low-opacity median differs");
        require(adapter->releaseRenderTarget(vis::RenderTargetId{4}), "Preview release failed");
        require(adapter->releaseRenderTarget(vis::RenderTargetId{2}), "Left view release failed");
        require(adapter->releaseRenderTarget(vis::RenderTargetId{3}), "Right view release failed");
        adapter->reset();
        model.opacity_raw() = Tensor::full({1, 1}, 4.f, Device::GPU).to(core::DataType::Float16);
        adapter->setDepthCaptureMode(false);
        require(adapter->render(context, model, request, true, vis::RenderTargetId{1}).has_value(), "Native adapter restart failed");
        auto restart_rgb = Tensor::empty({64, 96, 3}, Device::CPU, core::DataType::Float32);
        const auto restart_ticket = adapter->submitReadOutputImageIntoCpuHwcTicket(context,
                                                                                   vis::RenderTargetId{1}, restart_rgb, 0, 0);
        require(restart_ticket.has_value() && *restart_ticket != first_ticket, "Reset reused a stale native ticket identity");
        require(!adapter->pollReadbackTicket(first_ticket).has_value(), "Stale native ticket aliased a new destination");
        require(adapter->waitReadbackTicket(*restart_ticket).has_value(), "Restart ticket delivery failed");
    }
    auto positions = Tensor::from_vector(std::vector<float>{0, 0, -3, 0, 0, -6}, {2, 3}, Device::GPU);
    auto colors = Tensor::from_vector(std::vector<float>{0, 1, 0, 1, 0, 0}, {2, 3}, Device::GPU);
    vis::PointCloudVulkanRenderer::RenderRequest points;
    points.positions = &positions;
    points.colors = &colors;
    points.size = {96, 64};
    points.focal_y = 64;
    points.voxel_size = .2f;
    points.view = glm::mat4(1);
    // Explicit OpenGL-Z/Vulkan-Y projection supplied by the desktop contract.
    points.view_projection = glm::mat4(0);
    points.view_projection[0][0] = 1;
    points.view_projection[1][1] = -1.5f;
    points.view_projection[2][2] = -1.002002f;
    points.view_projection[2][3] = -1;
    points.view_projection[3][2] = -.2002002f;
    const auto point_frame = renderer.renderPoints(context, points, vis::RenderTargetId{1});
    require(point_frame.has_value(), "Native point raster failed");
    auto point_pixels = Tensor::empty({64, 96, 3}, Device::CPU, core::DataType::Float32);
    require(renderer.readColor(vis::RenderTargetId{1}, point_pixels, 0, 0).has_value(), "Point readback failed");
    require(point_pixels.ptr<float>()[center + 1] > .9f && point_pixels.ptr<float>()[center] < .1f,
            "Point depth test did not keep nearest color");
    const auto point_depth = renderer.readDepth({.pixel = {48, 32}, .source_size = {96, 64}, .target = vis::RenderTargetId{1}});
    require(point_depth.has_value() && std::abs(*point_depth - 3.f) < 1e-4f, "Point linear depth differs");
    if (compare_vulkan) {
        vis::PointCloudVulkanRenderer point_reference;
        const auto reference_frame = point_reference.render(context, points, vis::RenderTargetId{1});
        require(reference_frame.has_value(), "Point Vulkan reference failed");
        const auto reference_pixels = point_reference.readOutputImage(context, vis::RenderTargetId{1});
        require(reference_pixels.has_value(), "Point Vulkan reference readback failed");
        size_t coverage_difference = 0;
        for (size_t pixel = 0; pixel < 64 * 96; ++pixel) {
            const bool native_visible = point_pixels.ptr<float>()[pixel * 3 + 1] > .5f;
            const bool reference_visible = (*reference_pixels)->ptr<float>()[pixel * 3 + 1] > .5f;
            if (native_visible != reference_visible)
                ++coverage_difference;
        }
        require(coverage_difference == 0, "Native point coverage differs from desktop Vulkan");
    }
    auto point_auto = vis::createPointSceneRenderer();
    for (const auto target : {vis::RenderTargetId{19}, vis::RenderTargetId{45}, vis::RenderTargetId{903},
                              vis::RenderTargetId{701}, vis::RenderTargetId{300003}}) {
        const auto output = point_auto->render(context, points, target);
        require(output && output->viewer_backend == rendering::ViewerBackend::Metal && point_auto->hasRenderTarget(target),
                "Auto point view lost native target ownership");
        const auto pixels = point_auto->readOutputImage(context, target);
        require(pixels && (*pixels)->ptr<float>()[center + 1] > .9f, "Auto point view lost its nearest point");
    }
    // COLMAP imports keep RGB as bytes on the CPU. The platform adapter must
    // preserve their normalized colors while staging native storage.
    auto imported_positions = positions.to(Device::CPU);
    auto imported_colors = Tensor::from_vector(std::vector<float>{0, 255, 0, 255, 0, 0}, {2, 3}, Device::CPU).to(core::DataType::UInt8);
    auto imported_points = points;
    imported_points.positions = &imported_positions;
    imported_points.colors = &imported_colors;
    const auto imported_frame = point_auto->render(context, imported_points, vis::RenderTargetId{902});
    require(imported_frame.has_value(), "COLMAP CPU byte-color point rendering failed");
    const auto imported_pixels = point_auto->readOutputImage(context, vis::RenderTargetId{902});
    require(imported_pixels && std::memcmp((*imported_pixels)->data_ptr(), point_pixels.data_ptr(), point_pixels.bytes()) == 0,
            "COLMAP byte colors differ from normalized native point colors");
    imported_colors.copy_(Tensor::from_vector(std::vector<float>{255, 0, 0, 255, 0, 0}, {2, 3}, Device::CPU).to(core::DataType::UInt8));
    ++imported_points.colors_revision;
    require(point_auto->render(context, imported_points, vis::RenderTargetId{902}).has_value(),
            "COLMAP color revision rendering failed");
    const auto changed_pixels = point_auto->readOutputImage(context, vis::RenderTargetId{902});
    require(changed_pixels && (*changed_pixels)->ptr<float>()[center] > .9f && (*changed_pixels)->ptr<float>()[center + 1] < .1f,
            "COLMAP upload cache ignored an in-place color revision");
    require(point_auto->releaseRenderTarget(vis::RenderTargetId{19}), "Auto point target release failed");
    require(!point_auto->render(context, points, vis::RenderTargetId{19}).has_value(), "Closed point target ID was reused");
    require(point_auto->readOutputImage(context, vis::RenderTargetId{300003}).has_value(), "Closing one point view damaged another");
    multi_target_auto_contract(context, compare_vulkan);
    transparent_threshold_contract(context);
    partial_selection_mask_contract(context);
    std::puts("Native viewport texture, resident storage, camera, depth, resize and slot reuse contracts passed.");
}
int main(int argc, char** argv) {
    @autoreleasepool {
        if (!core::gpu_backend_available(core::GpuBackend::Metal))
            return 77;
        // Keep preference mutations local to this test, including direct runs.
        const auto home = std::filesystem::temp_directory_path() /
                          ("lichtfeld-metal-viewport-contracts-" + std::to_string(getpid()));
        setenv("LFS_HOME", home.c_str(), 1);
        unsetenv("LFS_SAFE_MODE");
        Py_Initialize();
        try {
            const bool native_only = argc == 2 && std::string_view(argv[1]) == "--native-only";
            run(!native_only);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }
}
