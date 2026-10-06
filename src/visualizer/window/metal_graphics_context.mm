/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "metal_graphics_context.hpp"

#include "core/gpu_backend_fwd.hpp"
#include "core/gpu_device_runtime.hpp"
#include "core/logger.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_metal_reader.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_metal.h>

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dispatch/dispatch.h>
#include <format>
#include <future>
#include <limits>
#include <thread>

namespace lfs::vis {
    namespace {
        constexpr std::uint32_t kFramesInFlight = 3;
        constexpr std::uint32_t kMaximumConsecutiveAcquireFailures = 8;
        constexpr auto kFrameWait = std::chrono::seconds(5);

        lfs::Error metalGraphicsError(const lfs::ErrorCode code, std::string detail) {
            return lfs::make_error({
                .code = code,
                .domain = lfs::ErrorDomain::Tensor,
                .user_message = "Metal presentation failed",
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        std::uint8_t unorm(const float value) {
            return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
        }

        void publishCompleted(std::atomic<std::uint64_t>& completed, const std::uint64_t value) {
            auto observed = completed.load(std::memory_order_relaxed);
            while (observed < value &&
                   !completed.compare_exchange_weak(observed, value,
                                                    std::memory_order_release,
                                                    std::memory_order_relaxed)) {
            }
        }
    } // namespace

    struct MetalGraphicsContext::Impl {
        struct SharedState {
            std::atomic<std::uint64_t> completed{0};
            std::atomic_bool failed{false};
            std::mutex error_mutex;
            std::string error;
        };

        SDL_Window* window = nullptr;
        SDL_MetalView view = nullptr;
        CAMetalLayer* layer = nil;
        std::unique_ptr<lfs::core::MetalTensorReader> reader;
        id<MTLRenderPipelineState> rgba8_present = nil;
        id<MTLComputePipelineState> rgba8_clear = nil;
        dispatch_semaphore_t frame_slots = dispatch_semaphore_create(kFramesInFlight);
        std::shared_ptr<SharedState> shared = std::make_shared<SharedState>();
        lfs::core::Tensor final_image;
        std::array<float, 4> clear_color{0, 0, 0, 1};
        int width = 1;
        int height = 1;
        std::uint64_t generation = 0;
        std::uint64_t submitted = 0;
        std::uint32_t consecutive_acquire_failures = 0;
        bool active = false;
        bool frame_slot_held = false;
        bool headless = false;
        bool shutdown = false;

        void setError(std::string message, const bool terminal = false) {
            {
                std::lock_guard lock(shared->error_mutex);
                shared->error = std::move(message);
            }
            if (terminal)
                shared->failed.store(true, std::memory_order_release);
        }

        std::string error() const {
            std::lock_guard lock(shared->error_mutex);
            return shared->error;
        }

        void releaseFrameSlot() {
            if (!frame_slot_held)
                return;
            frame_slot_held = false;
            dispatch_semaphore_signal(frame_slots);
        }

        void resizeLayer() {
            if (!layer)
                return;
            layer.drawableSize = CGSizeMake(std::max(width, 1), std::max(height, 1));
            if (window) {
                int logical_width = 0;
                int logical_height = 0;
                SDL_GetWindowSize(window, &logical_width, &logical_height);
                if (logical_width > 0 && logical_height > 0)
                    layer.contentsScale = std::max(double(width) / logical_width,
                                                   double(height) / logical_height);
            }
        }

        void ensureImage() {
            const lfs::core::TensorShape shape{
                static_cast<std::size_t>(std::max(height, 1)),
                static_cast<std::size_t>(std::max(width, 1)), 4};
            if (final_image.is_valid() && final_image.shape() == shape &&
                final_image.dtype() == lfs::core::DataType::UInt8 &&
                lfs::core::gpu_backend_of(final_image) == lfs::core::GpuBackend::Metal)
                return;
            lfs::core::GpuBackendScope scope(lfs::core::GpuBackend::Metal);
            final_image = lfs::core::Tensor::zeros(shape, lfs::core::Device::GPU,
                                                    lfs::core::DataType::UInt8);
            final_image.set_name("render.presentation");
        }

        // Every frame starts from the clear color, like a Vulkan render pass.
        void clearImage() {
            if (!rgba8_clear)
                return;
            const std::array<std::uint8_t, 4> color{
                unorm(clear_color[0]), unorm(clear_color[1]),
                unorm(clear_color[2]), unorm(clear_color[3])};
            const auto count = static_cast<std::uint32_t>(final_image.numel() / 4);
            const auto pipeline = rgba8_clear;
            const std::array<lfs::core::Tensor*, 1> outputs{&final_image};
            id<MTLCommandBuffer> command = reader->submitWrites(
                {}, outputs, [&](id<MTLCommandBuffer> buffer, std::span<const lfs::core::MetalTensorView>,
                                 std::span<const lfs::core::MetalTensorView> written) {
                    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:written[0].buffer offset:written[0].offset atIndex:0];
                    [encoder setBytes:color.data() length:color.size() atIndex:1];
                    [encoder setBytes:&count length:sizeof(count) atIndex:2];
                    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [encoder endEncoding];
                });
            if (!command)
                throw std::runtime_error(std::format("Metal frame clear was not submitted ({} pixels)", count));
        }

    };

    MetalGraphicsContext::MetalGraphicsContext() : impl_(std::make_unique<Impl>()) {}

    MetalGraphicsContext::~MetalGraphicsContext() {
        shutdown();
    }

    bool MetalGraphicsContext::initialize(SDL_Window* const window,
                                          const int framebuffer_width,
                                          const int framebuffer_height) {
        if (!window) {
            impl_->setError("Metal presentation requires an SDL window", true);
            return false;
        }
        if (@available(macOS 26.0, *)) {
            try {
                if (!lfs::core::gpu_backend_available(lfs::core::GpuBackend::Metal)) {
                    impl_->setError("Metal 4 is unavailable. LichtFeld Studio requires macOS 26 and a Metal 4 GPU.", true);
                    return false;
                }
                impl_->window = window;
                impl_->width = std::max(framebuffer_width, 1);
                impl_->height = std::max(framebuffer_height, 1);
                impl_->reader = std::make_unique<lfs::core::MetalTensorReader>();
                impl_->view = SDL_Metal_CreateView(window);
                if (!impl_->view)
                    throw std::runtime_error(std::format("SDL_Metal_CreateView failed: {}", SDL_GetError()));
                impl_->layer = (__bridge CAMetalLayer*)SDL_Metal_GetLayer(impl_->view);
                if (!impl_->layer)
                    throw std::runtime_error("SDL_Metal_GetLayer returned no CAMetalLayer");
                impl_->layer.device = impl_->reader->device();
                impl_->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
                // The drawable is only a render attachment; tensor compute stays in buffers.
                // Apple restricts framebuffer-only textures to render-pass attachments:
                // https://developer.apple.com/documentation/metal/mtltexture/isframebufferonly
                // https://developer.apple.com/documentation/quartzcore/cametallayer/framebufferonly
                impl_->layer.framebufferOnly = YES;
                impl_->layer.maximumDrawableCount = kFramesInFlight;
                impl_->layer.allowsNextDrawableTimeout = YES;
                // Like the Vulkan swapchain's MAILBOX mode: present without
                // waiting for vblank so navigation stays low-latency and the GUI
                // never blocks in present while it takes arena turns with
                // training. The frame demand ledger paces idle frames.
                impl_->layer.displaySyncEnabled = NO;
                impl_->layer.presentsWithTransaction = NO;
                impl_->resizeLayer();

                static constexpr char source[] = R"metal(
#include <metal_stdlib>
using namespace metal;
vertex float4 present_vertex(uint vertex_id [[vertex_id]]) {
    const float2 corners[] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
    return float4(corners[vertex_id], 0.0f, 1.0f);
}
fragment float4 present_rgba8(float4 position [[position]],
                             device const uchar4* source [[buffer(0)]],
                             constant uint2& dimensions [[buffer(1)]]) {
    const uint2 pixel = uint2(position.xy);
    if (pixel.x >= dimensions.x || pixel.y >= dimensions.y) return float4(0.0f, 0.0f, 0.0f, 1.0f);
    return float4(source[pixel.y * dimensions.x + pixel.x]) / 255.0f;
}
kernel void clear_rgba8(device uchar4* destination [[buffer(0)]],
                        constant uchar4& color [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < count) destination[gid] = color;
}
)metal";
                NSError* error = nil;
                MTLCompileOptions* options = [MTLCompileOptions new];
                options.languageVersion = MTLLanguageVersion3_1;
                options.mathMode = MTLMathModeSafe;
                id<MTLLibrary> library = [impl_->reader->device()
                    newLibraryWithSource:@(source) options:options error:&error];
                MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
                descriptor.vertexFunction = [library newFunctionWithName:@"present_vertex"];
                descriptor.fragmentFunction = [library newFunctionWithName:@"present_rgba8"];
                descriptor.colorAttachments[0].pixelFormat = impl_->layer.pixelFormat;
                impl_->rgba8_present = [impl_->reader->device()
                    newRenderPipelineStateWithDescriptor:descriptor error:&error];
                if (!impl_->rgba8_present)
                    throw std::runtime_error(std::format("Metal presenter pipeline creation failed: {}",
                                                         error ? error.localizedDescription.UTF8String : "unknown error"));
                impl_->rgba8_clear = [impl_->reader->device()
                    newComputePipelineStateWithFunction:[library newFunctionWithName:@"clear_rgba8"] error:&error];
                if (!impl_->rgba8_clear)
                    throw std::runtime_error(std::format("Metal frame clear pipeline creation failed: {}",
                                                         error ? error.localizedDescription.UTF8String : "unknown error"));
                impl_->ensureImage();
                return true;
            } catch (const std::exception& error) {
                impl_->setError(error.what(), true);
                shutdown();
                return false;
            }
        }
        impl_->setError("LichtFeld Studio requires macOS 26 and Metal 4; no Vulkan fallback is available in this build.", true);
        return false;
    }

    bool MetalGraphicsContext::initializeHeadless() {
        if (@available(macOS 26.0, *)) {
            try {
                if (!lfs::core::gpu_backend_available(lfs::core::GpuBackend::Metal)) {
                    impl_->setError("Headless rendering requires macOS 26 and a Metal 4 GPU.", true);
                    return false;
                }
                impl_->headless = true;
                impl_->reader = std::make_unique<lfs::core::MetalTensorReader>();
                impl_->ensureImage();
                return true;
            } catch (const std::exception& error) {
                impl_->setError(error.what(), true);
                return false;
            }
        }
        impl_->setError("Headless rendering requires macOS 26 and Metal 4.", true);
        return false;
    }

    void MetalGraphicsContext::shutdown() {
        if (!impl_ || impl_->shutdown)
            return;
        impl_->shutdown = true;
        if (impl_->active) {
            impl_->active = false;
        }
        // The event loop can reserve the next slot before beginFrame. Release
        // that reservation even when no frame was subsequently started.
        impl_->releaseFrameSlot();
        static_cast<void>(waitForSubmittedFrames());
        impl_->final_image = {};
        impl_->rgba8_present = nil;
        impl_->rgba8_clear = nil;
        impl_->reader.reset();
        impl_->layer = nil;
        if (impl_->view) {
            SDL_Metal_DestroyView(impl_->view);
            impl_->view = nullptr;
        }
        impl_->window = nullptr;
    }

    void MetalGraphicsContext::notifyFramebufferResized(
        const int width, const int height, GraphicsResizeIntent) {
        impl_->width = std::max(width, 0);
        impl_->height = std::max(height, 0);
        if (width > 0 && height > 0) {
            impl_->resizeLayer();
            impl_->ensureImage();
        }
    }

    // Resizes apply synchronously in notifyFramebufferResized.
    bool MetalGraphicsContext::hasPendingResize() const { return false; }
    bool MetalGraphicsContext::pendingResizeReady() const { return true; }
    double MetalGraphicsContext::secondsUntilPendingResizeReady() const { return 0.0; }

    lfs::Result<std::optional<GraphicsFrame>> MetalGraphicsContext::beginFrame(
        const std::array<float, 4>& clear_color) {
        if (impl_->shared->failed.load(std::memory_order_acquire))
            return metalGraphicsError(lfs::ErrorCode::Internal, impl_->error());
        if (impl_->active)
            return metalGraphicsError(lfs::ErrorCode::InvalidArgument,
                                      "beginFrame called while another frame is active");
        if (impl_->width <= 0 || impl_->height <= 0)
            return std::optional<GraphicsFrame>{};
        if (!waitForNextFrameSlot())
            return metalGraphicsError(lfs::ErrorCode::Unavailable, impl_->error());
        try {
            impl_->clear_color = clear_color;
            impl_->ensureImage();
            impl_->clearImage();
            impl_->active = true;
            return std::optional<GraphicsFrame>{GraphicsFrame{
                .frame_slot = static_cast<std::size_t>(impl_->generation % kFramesInFlight),
                .width = static_cast<std::uint32_t>(impl_->width),
                .height = static_cast<std::uint32_t>(impl_->height),
                .generation = ++impl_->generation,
            }};
        } catch (const std::exception& error) {
            impl_->releaseFrameSlot();
            impl_->setError(error.what(), true);
            return metalGraphicsError(lfs::ErrorCode::Internal, error.what());
        }
    }

    lfs::Status MetalGraphicsContext::endFrame() {
        if (!impl_->active)
            return lfs::Status::failure(metalGraphicsError(
                lfs::ErrorCode::InvalidArgument, "endFrame called without an active frame"));
        impl_->active = false;
        if (impl_->headless) {
            ++impl_->submitted;
            publishCompleted(impl_->shared->completed, impl_->submitted);
            impl_->releaseFrameSlot();
            return {};
        }
        try {
            id<CAMetalDrawable> drawable = [impl_->layer nextDrawable];
            if (!drawable) {
                impl_->releaseFrameSlot();
                ++impl_->consecutive_acquire_failures;
                const bool terminal = impl_->consecutive_acquire_failures >=
                                      kMaximumConsecutiveAcquireFailures;
                impl_->setError(std::format(
                    "CAMetalLayer drawable acquisition failed ({}/{} consecutive failures)",
                    impl_->consecutive_acquire_failures,
                    kMaximumConsecutiveAcquireFailures), terminal);
                return lfs::Status::failure(metalGraphicsError(
                    terminal ? lfs::ErrorCode::Internal : lfs::ErrorCode::Unavailable,
                    impl_->error()));
            }
            impl_->consecutive_acquire_failures = 0;
            const std::uint64_t serial = ++impl_->submitted;
            const auto state = impl_->shared;
            const auto slots = impl_->frame_slots;
            const auto pipeline = impl_->rgba8_present;
            const std::uint32_t width = static_cast<std::uint32_t>(impl_->width);
            const std::uint32_t height = static_cast<std::uint32_t>(impl_->height);
            const lfs::core::Tensor* image = &impl_->final_image;
            const std::array<const lfs::core::Tensor*, 1> tensors{image};
            id<MTLCommandBuffer> command = impl_->reader->submit(
                tensors, [=](id<MTLCommandBuffer> buffer,
                             std::span<const lfs::core::MetalTensorView> views) {
                    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
                    pass.colorAttachments[0].texture = drawable.texture;
                    pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
                    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                    id<MTLRenderCommandEncoder> encoder = [buffer renderCommandEncoderWithDescriptor:pass];
                    [encoder setRenderPipelineState:pipeline];
                    [encoder setFragmentBuffer:views[0].buffer offset:views[0].offset atIndex:0];
                    const std::array<std::uint32_t, 2> dimensions{static_cast<std::uint32_t>(width),
                                                                 static_cast<std::uint32_t>(height)};
                    [encoder setFragmentBytes:dimensions.data() length:sizeof(dimensions) atIndex:1];
                    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                    [encoder endEncoding];
                    [buffer presentDrawable:drawable];
                    [buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                        if (completed.status == MTLCommandBufferStatusError) {
                            {
                                std::lock_guard lock(state->error_mutex);
                                state->error = std::format(
                                    "Metal presentation command {} failed: {}", serial,
                                    completed.error
                                        ? completed.error.localizedDescription.UTF8String
                                        : "unknown command-buffer failure");
                            }
                            state->failed.store(true, std::memory_order_release);
                        }
                        publishCompleted(state->completed, serial);
                        dispatch_semaphore_signal(slots);
                    }];
                });
            if (!command)
                throw std::runtime_error("Metal tensor reader returned no presentation command buffer");
            impl_->frame_slot_held = false;
            return {};
        } catch (const std::exception& error) {
            impl_->releaseFrameSlot();
            impl_->setError(error.what(), true);
            return lfs::Status::failure(metalGraphicsError(lfs::ErrorCode::Internal,
                                                           error.what()));
        }
    }

    lfs::core::Tensor* MetalGraphicsContext::finalImageTensor(const GraphicsFrame& frame) {
        return impl_->active && frame.generation == impl_->generation ? &impl_->final_image : nullptr;
    }

    const lfs::core::Tensor* MetalGraphicsContext::finalImageTensor(const GraphicsFrame& frame) const {
        return impl_->active && frame.generation == impl_->generation ? &impl_->final_image : nullptr;
    }

    bool MetalGraphicsContext::presentBootstrapFrame(const float r, const float g,
                                                     const float b, const float a) {
        auto frame = beginFrame({r, g, b, a});
        if (!frame || !*frame)
            return false;
        return static_cast<bool>(endFrame());
    }

    bool MetalGraphicsContext::hasActiveFrame() const noexcept { return impl_->active; }

    std::future<lfs::Result<WindowCapture>> MetalGraphicsContext::captureFinalFrameRgbaAsync() {
        const auto tensor = impl_->final_image;
        const int width = impl_->width;
        const int height = impl_->height;
        return std::async(std::launch::async, [tensor, width, height]() -> lfs::Result<WindowCapture> {
            try {
                if (!tensor.is_valid() || width <= 0 || height <= 0)
                    return metalGraphicsError(lfs::ErrorCode::NotFound,
                                              "No final Metal image is available");
                const auto host = tensor.to_pageable_host();
                WindowCapture capture{.width = width, .height = height};
                capture.rgba.resize(static_cast<std::size_t>(width) * height * 4);
                if (host.dtype() == lfs::core::DataType::UInt8) {
                    std::memcpy(capture.rgba.data(), host.data_ptr(), capture.rgba.size());
                } else {
                    const auto converted = host.to(lfs::core::DataType::UInt8);
                    std::memcpy(capture.rgba.data(), converted.data_ptr(), capture.rgba.size());
                }
                return capture;
            } catch (const std::exception& error) {
                return metalGraphicsError(lfs::ErrorCode::Internal, error.what());
            }
        });
    }

    lfs::Result<WindowCapture> MetalGraphicsContext::captureFinalFrameRgba() {
        return captureFinalFrameRgbaAsync().get();
    }

    bool MetalGraphicsContext::waitForNextFrameSlot() {
        if (impl_->frame_slot_held)
            return true;
        const auto timeout = dispatch_time(DISPATCH_TIME_NOW,
                                           std::chrono::duration_cast<std::chrono::nanoseconds>(kFrameWait).count());
        if (dispatch_semaphore_wait(impl_->frame_slots, timeout) != 0) {
            impl_->setError("Timed out after 5 seconds waiting for a Metal presentation frame slot", true);
            return false;
        }
        impl_->frame_slot_held = true;
        return true;
    }

    bool MetalGraphicsContext::waitForSubmittedFrames() {
        return waitForCompletedSubmitSerial(impl_->submitted);
    }

    bool MetalGraphicsContext::waitIdle() {
        if (!waitForSubmittedFrames())
            return false;
        try {
            if (lfs::core::gpu_backend_live(lfs::core::GpuBackend::Metal))
                lfs::core::gpu_device_barrier(lfs::core::GpuBackend::Metal);
            return true;
        } catch (const std::exception& error) {
            impl_->setError(error.what(), true);
            return false;
        }
    }

    std::uint64_t MetalGraphicsContext::completedSubmitSerial() const {
        return impl_->shared->completed.load(std::memory_order_acquire);
    }

    bool MetalGraphicsContext::waitForCompletedSubmitSerial(const std::uint64_t serial) {
        if (serial == 0)
            return true;
        const auto deadline = std::chrono::steady_clock::now() + kFrameWait;
        while (completedSubmitSerial() < serial) {
            if (std::chrono::steady_clock::now() >= deadline) {
                impl_->setError(std::format(
                    "Timed out after 5 seconds waiting for Metal presentation serial {} (completed {})",
                    serial, completedSubmitSerial()), true);
                return false;
            }
            std::this_thread::yield();
        }
        return !impl_->shared->failed.load(std::memory_order_acquire);
    }

    RendererTerminalState MetalGraphicsContext::terminalState() const noexcept {
        return impl_->shared->failed.load(std::memory_order_acquire)
                   ? RendererTerminalState::Quarantined
                   : RendererTerminalState::Running;
    }

    const std::string& MetalGraphicsContext::lastError() const noexcept {
        // GraphicsContext's legacy API returns a reference. Keep a thread-local
        // snapshot so completion handlers can continue updating shared state.
        thread_local std::string snapshot;
        snapshot = impl_->error();
        return snapshot;
    }

    GraphicsCapabilities MetalGraphicsContext::capabilities() const noexcept {
        return {
            .native_metal = true,
            .mesh_rendering = true,
            // The tensor wireframe is drawn in the fragment shader at any width.
            .wireframe = true,
            .wide_lines = true,
            .external_memory_interop = false,
            .external_semaphore_interop = false,
            .environment_map = true,
            .split_view = false,
            .temporal_upscaling = false,
            .mesh2splat = false,
            .hdr_libplacebo = false,
        };
    }

    void MetalGraphicsContext::flushPipelineCache() {}
    // The native Metal viewer reads Metal tensors directly, so renderer-visible
    // splat storage is ordinary Metal storage with the requested capacity.
    lfs::core::SplatTensorAllocator MetalGraphicsContext::splatTensorAllocator(bool) {
        return [](lfs::core::TensorShape shape, const std::size_t capacity, const lfs::core::DataType dtype,
                  std::string_view name) {
            const lfs::core::GpuBackendScope scope(lfs::core::GpuBackend::Metal);
            auto tensor = lfs::core::Tensor::empty(std::move(shape), lfs::core::Device::GPU, dtype);
            if (tensor.ndim() > 0 && capacity > tensor.size(0))
                tensor.reserve(capacity);
            tensor.set_name(std::string(name));
            return tensor;
        };
    }
    void MetalGraphicsContext::connectTensorBackend() {}
    void MetalGraphicsContext::disconnectTensorBackend() {}

    std::unique_ptr<GraphicsContext> createGraphicsContext() {
        return std::make_unique<MetalGraphicsContext>();
    }
} // namespace lfs::vis
