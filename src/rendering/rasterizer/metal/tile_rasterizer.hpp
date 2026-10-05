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
        uint32_t blend_threads; // completed GPU dispatch: 32 or 64; reuses reserved status space
        uint32_t maximum_tile_instances;
    };
    static_assert(sizeof(RasterStatus) == 24);

    // Tracked buffers may be shared by frames submitted serially to one command
    // queue. Outputs/status remain frame-owned; growth retains encoded resources
    // through command-buffer ownership and commits only a complete replacement.
    class RasterScratch {
    public:
        explicit RasterScratch(id<MTLDevice>);
        ~RasterScratch();
        [[nodiscard]] bool fits(uint32_t width, uint32_t height, uint32_t splats, uint32_t instances) const;

    private:
        friend class RasterFrame;
        friend class TileRasterizer;
        void reserve(uint32_t tiles, uint32_t splats, uint32_t instances);
        struct Impl;
        std::shared_ptr<Impl> impl_;
    };

    // One output/status reservation per in-flight frame. Scratch and outputs are allocated
    // up front. Optional parallel summaries follow already completed dense counts
    // within the device working set; encode never waits for a GPU count readback.
    // Completion releases the reservation even if its public wrapper is destroyed.
    class RasterFrame {
    public:
        RasterFrame(id<MTLDevice> device, uint32_t width, uint32_t height,
                    uint32_t max_splats, uint32_t max_instances, std::shared_ptr<RasterScratch> scratch = {});
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
        // Vulkan's legacy GUT chain omits the saturating color/alpha update,
        // while retaining median and expected-depth contributions. Native
        // analytic/Spark rendering and the GS macro reference include it.
        // Transparent desktop 3DGS can match the macro reference's half
        // footprint and half batches with FP32 composition. The default remains FP32.
        void encode(id<MTLCommandBuffer> command, BufferSlice projected, uint32_t count,
                    RasterMode mode, simd_float4 background, RasterFrame& frame, const OverlayBuffers& overlay = {},
                    BufferSlice gut = {}, const Projection& projection = {}, const LodSelection& lod = {},
                    bool omit_saturating_color = false, bool macro_half_display = false, GpuProfile* profile = nullptr, bool exact_median = false);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::rendering::metal
