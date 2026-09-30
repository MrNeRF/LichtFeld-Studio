/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "splat_preprocessor.hpp"

namespace lfs::rendering::metal {
    enum class RasterMode : uint32_t { Gaussian,
                                       Points,
                                       Discs,
                                       Gut };
    enum class RasterError : uint32_t { None,
                                        InstanceCapacityExceeded };
    struct RasterStatus {
        uint64_t required_instances;
        RasterError error;
        uint32_t unused;
    };
    static_assert(sizeof(RasterStatus) == 16);

    // One reservation per in-flight frame. All GPU buffers and textures are allocated
    // up front; encode neither allocates frame storage nor reads counts back to CPU.
    // Completion releases the reservation even if its public wrapper is destroyed.
    class RasterFrame {
    public:
        RasterFrame(id<MTLDevice> device, uint32_t width, uint32_t height,
                    uint32_t max_splats, uint32_t max_instances);
        ~RasterFrame();
        RasterFrame(const RasterFrame&) = delete;
        RasterFrame& operator=(const RasterFrame&) = delete;
        [[nodiscard]] bool busy() const;
        // Read only after command completion. An overflow frame is a clear background,
        // never a partially sorted scene. Retry in a larger reservation if desired.
        [[nodiscard]] RasterStatus status() const;
        [[nodiscard]] id<MTLTexture> color() const;       // RGBA16Float, premultiplied
        [[nodiscard]] id<MTLTexture> depth() const;       // RGBA32Float: weighted Z, alpha, first Z (or expected-depth weight), median Z
        [[nodiscard]] id<MTLTexture> pick() const;        // R32Uint: first contributing source ID
        [[nodiscard]] id<MTLBuffer> statusBuffer() const; // GPU consumers only; CPU reads use status().

    private:
        friend class TileRasterizer;
        struct Impl;
        std::shared_ptr<Impl> impl_;
    };

    class TileRasterizer {
    public:
        explicit TileRasterizer(id<MTLDevice> device);
        ~TileRasterizer();
        TileRasterizer(const TileRasterizer&) = delete;
        TileRasterizer& operator=(const TileRasterizer&) = delete;
        void encode(id<MTLCommandBuffer> command, BufferSlice projected, uint32_t count,
                    RasterMode mode, simd_float4 background, RasterFrame& frame, const OverlayBuffers& overlay = {},
                    BufferSlice gut = {}, const Projection& projection = {});

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::rendering::metal
