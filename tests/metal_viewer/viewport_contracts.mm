/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_backend.hpp"
#include "metal_viewport_renderer.hpp"
#include "preferences.hpp"
#include <Python.h>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <unistd.h>

using namespace lfs;
static void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
static void run() {
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
    const auto malformed = renderer.render(context, model, request, vis::VksplatViewportRenderer::OutputSlot::Main);
    require(!malformed && malformed.error().code() == lfs::ErrorCode::InvalidArgument &&
                malformed.error().domain() == lfs::ErrorDomain::Rendering,
            "Malformed scene lost its typed argument error");
    const auto unknown = renderer.pollReadback(uint64_t{1} << 63, false);
    require(!unknown && unknown.error().code() == lfs::ErrorCode::NotFound,
            "Unknown ticket lost its typed lookup error");
    auto empty_destination = Tensor::empty({64, 96, 3}, Device::CPU);
    const auto empty_read = renderer.readColor(vis::VksplatViewportRenderer::OutputSlot::Main,
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
            const auto result = renderer.render(context, model, lod_request, vis::VksplatViewportRenderer::OutputSlot::Main);
            require(bool(result), "Invalid/empty LOD cut failed instead of publishing empty coverage");
            auto pixels = Tensor::empty({64, 96, 4}, Device::CPU);
            require(bool(renderer.readColor(vis::VksplatViewportRenderer::OutputSlot::Main, pixels, 0, 0)), "Empty LOD readback failed");
            require(pixels.ptr<float>()[((32 * 96 + 48) * 4) + 3] == 0, "Invalid/empty LOD cut retained old visible coverage");
        }
    }
    for (int frame = 0; frame < 12; ++frame) {
        const auto output = renderer.render(context, model, request, vis::VksplatViewportRenderer::OutputSlot::Main);
        if (!output)
            throw std::runtime_error(lfs::format_for_developer(output.error()));
        require(output->image && output->image_view && output->completion_semaphore, "Missing native presentation handles");
        const auto size = request.frame_view.size;
        auto pixels = Tensor::empty({size_t(size.y), size_t(size.x), 4}, Device::CPU, core::DataType::Float32);
        auto read = renderer.readColor(vis::VksplatViewportRenderer::OutputSlot::Main, pixels, 0, 0);
        if (!read)
            throw std::runtime_error(lfs::format_for_developer(read.error()));
        if (frame == 0) {
            const auto invalid = renderer.readColor(vis::VksplatViewportRenderer::OutputSlot::Main,
                                                    pixels, -1, 0);
            require(!invalid && invalid.error().code() == lfs::ErrorCode::InvalidArgument,
                    "Invalid readback destination lost its typed argument error");
        }
        const size_t center = ((size.y / 2) * size.x + size.x / 2) * 4;
        require(pixels.ptr<float>()[center] > .5f, "Native camera or color transfer differs");
        require(pixels.ptr<float>()[center] > pixels.ptr<float>()[center + 1], "SH0 channels differ");
        const auto depth = renderer.readDepth({.pixel = size / 2, .source_size = size});
        if (!depth)
            throw std::runtime_error(lfs::format_for_developer(depth.error()));
        require(std::abs(*depth - 3.f) < 1e-4f, "Native desktop depth differs");
        auto asynchronous = Tensor::full({size_t(size.y) + 2, size_t(size.x) + 2, 3}, -1.f, Device::CPU);
        const auto ticket = renderer.submitReadback(vis::VksplatViewportRenderer::OutputSlot::Main, asynchronous, 1, 1, false);
        require(ticket.has_value(), "Native asynchronous color submit failed");
        require(renderer.outstandingReadbacks() == 1, "Native ticket not tracked");
        const auto ready = renderer.pollReadback(*ticket, true);
        require(ready.has_value() && *ready == vis::VksplatViewportRenderer::ReadbackTicketStatus::Ready,
                "Native asynchronous color delivery failed");
        require(asynchronous.ptr<float>()[0] == -1.f, "Readback overwrote destination border");
        const size_t async_center = (((size.y / 2) + 1) * (size.x + 2) + (size.x / 2) + 1) * 3;
        require(asynchronous.ptr<float>()[async_center] == pixels.ptr<float>()[center], "Asynchronous color differs");
        auto plane = Tensor::full({size_t(size.y), size_t(size.x)}, -1.f, Device::CPU);
        const auto depth_ticket = renderer.submitReadback(vis::VksplatViewportRenderer::OutputSlot::Main, plane, 0, 0, true);
        require(depth_ticket.has_value() && renderer.pollReadback(*depth_ticket, true).has_value(), "Depth plane ticket failed");
        require(std::abs(plane.ptr<float>()[(size.y / 2) * size.x + size.x / 2] - 3.f) < 1e-4f, "Asynchronous depth differs");
        const auto abandoned = renderer.submitReadback(vis::VksplatViewportRenderer::OutputSlot::Main, plane, 0, 0, true);
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
        const auto frame = renderer.render(context, model, r, vis::VksplatViewportRenderer::OutputSlot::Main);
        if (!frame)
            throw std::runtime_error(lfs::format_for_developer(frame.error()));
        auto pixels = Tensor::empty({64, 96, 3}, Device::CPU, core::DataType::Float32);
        const auto read = renderer.readColor(vis::VksplatViewportRenderer::OutputSlot::Main, pixels, 0, 0);
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
    const auto gut_depth = renderer.readDepth({.pixel = {48, 32}, .source_size = {96, 64}});
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
    const auto panorama_depth = renderer.readDepth({.pixel = {48, 32}, .source_size = {96, 64}});
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
    require(renderer.render(context, model, request, vis::VksplatViewportRenderer::OutputSlot::Main).has_value(), "Panorama subregion failed");
    require(renderer.readColor(vis::VksplatViewportRenderer::OutputSlot::Main, tile, 0, 0).has_value(), "Panorama subregion readback failed");
    for (size_t y = 0; y < 33; ++y)
        for (size_t x = 0; x < 31; ++x)
            for (size_t c = 0; c < 3; ++c)
                require(std::abs(tile.ptr<float>()[(y * 31 + x) * 3 + c] - seam.ptr<float>()[((y + 15) * 96 + x + 65) * 3 + c]) < 1.f / 255,
                        "Panorama subregion changed the full-camera image");
    request.frame_view.size = {96, 64};
    request.frame_view.subregion_full_size = request.frame_view.subregion_origin = {0, 0};
    model.means_raw() = Tensor::from_vector(std::vector<float>{0, 0, -3}, {1, 3}, Device::GPU);
    request.gut = false;
    require(!vis::MetalViewportRenderer::supports(model, request), "Panorama accepted a rasterizer without spherical rays");
    request.gut = true;
    request.equirectangular = false;
    require(renderer.release(vis::VksplatViewportRenderer::OutputSlot::Main).has_value(), "Native release failed");
    require(renderer.size(vis::VksplatViewportRenderer::OutputSlot::Main) == glm::ivec2(0), "Released slot retained output");
    request.gut = false;
    request.overlay.markers.show_center_markers = false;
    // Jitter changes the projection, but must not perpetually request a
    // new scene frame and reset temporal reconstruction convergence.
    const auto calibrated = request.frame_view.getCameraIntrinsics();
    request.frame_view.containment_intrinsics = calibrated;
    request.frame_view.intrinsics_override = calibrated;
    const auto stable = renderer.render(context, model, request, vis::VksplatViewportRenderer::OutputSlot::Main);
    require(stable.has_value() && renderer.outputComplete(vis::VksplatViewportRenderer::OutputSlot::Main).has_value(), "Stable jitter fixture failed");
    request.frame_view.intrinsics_override->center_x += .25f;
    request.frame_view.intrinsics_override->center_y -= .125f;
    const auto jittered = renderer.render(context, model, request, vis::VksplatViewportRenderer::OutputSlot::Main);
    require(jittered.has_value() && !jittered->lod_streaming_active, "Temporal jitter requested perpetual refinement");
    request.frame_view.containment_intrinsics.reset();
    request.frame_view.intrinsics_override.reset();
    vis::UserPreferences::instance().setViewerBackend(rendering::ViewerBackend::Metal);
    {
        vis::VksplatViewportRenderer adapter;
        uint64_t first_ticket = 0;
        for (const auto slot : {vis::VksplatViewportRenderer::OutputSlot::Main,
                                vis::VksplatViewportRenderer::OutputSlot::SplitLeft,
                                vis::VksplatViewportRenderer::OutputSlot::SplitRight,
                                vis::VksplatViewportRenderer::OutputSlot::Preview}) {
            const auto frame = adapter.render(context, model, request, true, slot, false, true);
            if (!frame)
                throw std::runtime_error(frame.error());
            require((frame->generation >> 63) != 0, "Requested native export used Vulkan");
            const auto rgba = adapter.readOutputImageRgba8(context, slot);
            require(rgba.has_value(), "Native adapter capture failed");
            auto rgb = Tensor::empty({64, 96, 3}, Device::CPU, core::DataType::Float32);
            const auto ticket = adapter.submitReadOutputImageIntoCpuHwcTicket(context, slot, rgb, 0, 0);
            require(ticket.has_value() && vis::MetalViewportRenderer::nativeTicket(*ticket), "Adapter ticket lost backend identity");
            if (!first_ticket)
                first_ticket = *ticket;
            require(adapter.waitReadbackTicket(*ticket).has_value(), "Adapter ticket delivery failed");
            require(rgb.ptr<float>()[center] > .5f, "Native adapter export lost splat");
        }
        model.opacity_raw() = Tensor::full({1, 1}, -4.f, Device::GPU).to(core::DataType::Float16);
        adapter.setDepthCaptureMode(true, true);
        const auto expected_frame = adapter.render(context, model, request, true, vis::VksplatViewportRenderer::OutputSlot::Preview, false, true);
        require(expected_frame.has_value() && (expected_frame->generation >> 63) != 0, "Expected-depth capture lost native backend");
        const auto expected = adapter.readPreviewDepth(context);
        require(expected.has_value() && std::abs((*expected)->ptr<float>()[32 * 96 + 48] - 3.f) < 1e-3f, "Expected-depth capture differs");
        require((*expected)->ptr<float>()[0] >= 1e9f, "Empty expected depth lost sentinel");
        adapter.setDepthCaptureMode(true, false);
        require(adapter.render(context, model, request, true, vis::VksplatViewportRenderer::OutputSlot::Preview, false, true).has_value(), "Median capture failed");
        const auto median = adapter.readPreviewDepth(context);
        require(median.has_value() && (*median)->ptr<float>()[32 * 96 + 48] >= 1e9f, "Low-opacity median differs");
        adapter.releasePreviewResources();
        adapter.releaseSplitOutputResources();
        adapter.reset();
        model.opacity_raw() = Tensor::full({1, 1}, 4.f, Device::GPU).to(core::DataType::Float16);
        adapter.setDepthCaptureMode(false);
        require(adapter.render(context, model, request, true).has_value(), "Native adapter restart failed");
        auto restart_rgb = Tensor::empty({64, 96, 3}, Device::CPU, core::DataType::Float32);
        const auto restart_ticket = adapter.submitReadOutputImageIntoCpuHwcTicket(context,
                                                                                  vis::VksplatViewportRenderer::OutputSlot::Main, restart_rgb, 0, 0);
        require(restart_ticket.has_value() && *restart_ticket != first_ticket, "Reset reused a stale native ticket identity");
        require(!adapter.pollReadbackTicket(first_ticket).has_value(), "Stale native ticket aliased a new destination");
        require(adapter.waitReadbackTicket(*restart_ticket).has_value(), "Restart ticket delivery failed");
    }
    vis::UserPreferences::instance().setViewerBackend(rendering::ViewerBackend::Vulkan);
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
    const auto point_frame = renderer.renderPoints(context, points, vis::PointCloudVulkanRenderer::OutputSlot::Main);
    require(point_frame.has_value(), "Native point raster failed");
    auto point_pixels = Tensor::empty({64, 96, 3}, Device::CPU, core::DataType::Float32);
    require(renderer.readColor(vis::VksplatViewportRenderer::OutputSlot::Main, point_pixels, 0, 0).has_value(), "Point readback failed");
    require(point_pixels.ptr<float>()[center + 1] > .9f && point_pixels.ptr<float>()[center] < .1f,
            "Point depth test did not keep nearest color");
    const auto point_depth = renderer.readDepth({.pixel = {48, 32}, .source_size = {96, 64}});
    require(point_depth.has_value() && std::abs(*point_depth - 3.f) < 1e-4f, "Point linear depth differs");
    vis::PointCloudVulkanRenderer point_reference;
    const auto reference_frame = point_reference.render(context, points);
    require(reference_frame.has_value(), "Point Vulkan reference failed");
    const auto reference_pixels = point_reference.readOutputImage(context);
    require(reference_pixels.has_value(), "Point Vulkan reference readback failed");
    size_t coverage_difference = 0;
    for (size_t pixel = 0; pixel < 64 * 96; ++pixel) {
        const bool native_visible = point_pixels.ptr<float>()[pixel * 3 + 1] > .5f;
        const bool reference_visible = (*reference_pixels)->ptr<float>()[pixel * 3 + 1] > .5f;
        if (native_visible != reference_visible)
            ++coverage_difference;
    }
    require(coverage_difference == 0, "Native point coverage differs from desktop Vulkan");
    std::puts("Native viewport texture, resident storage, camera, depth, resize and slot reuse contracts passed.");
}
int main() {
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
            run();
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }
}
