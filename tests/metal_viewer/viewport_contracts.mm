/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_backend.hpp"
#include "metal_viewport_renderer.hpp"
#include <Python.h>
#include <cstdio>
#include <stdexcept>

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
    require(vis::MetalViewportRenderer::supports(model, request), "Native frame rejected");
    for (int frame = 0; frame < 12; ++frame) {
        const auto output = renderer.render(context, model, request, vis::VksplatViewportRenderer::OutputSlot::Main);
        if (!output)
            throw std::runtime_error(output.error());
        require(output->image && output->image_view && output->completion_semaphore, "Missing native presentation handles");
        const auto size = request.frame_view.size;
        auto pixels = Tensor::empty({size_t(size.y), size_t(size.x), 4}, Device::CPU, core::DataType::Float32);
        auto read = renderer.readColor(vis::VksplatViewportRenderer::OutputSlot::Main, pixels, 0, 0);
        if (!read)
            throw std::runtime_error(read.error());
        const size_t center = ((size.y / 2) * size.x + size.x / 2) * 4;
        require(pixels.ptr<float>()[center] > .5f, "Native camera or color transfer differs");
        require(pixels.ptr<float>()[center] > pixels.ptr<float>()[center + 1], "SH0 channels differ");
        const auto depth = renderer.readDepth({.pixel = size / 2, .source_size = size});
        if (!depth)
            throw std::runtime_error(depth.error());
        require(std::abs(*depth - 3.f) < 1e-4f, "Native desktop depth differs");
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
    require(!vis::MetalViewportRenderer::supports(model, request), "Unsupported overlay silently dropped");
    std::puts("Native viewport texture, resident storage, camera, depth, resize and slot reuse contracts passed.");
}
int main() {
    @autoreleasepool {
        if (!core::gpu_backend_available(core::GpuBackend::Metal))
            return 77;
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
