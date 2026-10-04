/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "viewport_frame_desc.hpp"
#include <memory>
namespace lfs::vis {
    class GraphicsContext;
    struct GraphicsFrame;
    struct ViewRenderState;
    class LFS_VIS_API ViewportReferenceResources {
    public:
        ViewportReferenceResources();
        ~ViewportReferenceResources();

    private:
        friend class ViewportReferenceRenderer;
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
    // Temporary reference adapter, NOT an extensible per-backend compositor.
    // Delete after the single tensor-program compositor reaches parity.
    class LFS_VIS_API ViewportReferenceRenderer {
    public:
        explicit ViewportReferenceRenderer(std::shared_ptr<ViewportReferenceResources> resources = {});
        ~ViewportReferenceRenderer();
        [[nodiscard]] bool initialize(GraphicsContext&);
        void prepare(GraphicsContext&, const ViewportFrameDesc&, ViewRenderState&);
        [[nodiscard]] bool hasPreRenderWork(const ViewportFrameDesc&) const;
        [[nodiscard]] bool recordPreRenderWork(const GraphicsFrame&, const ViewportFrameDesc&);
        void record(const GraphicsFrame&, const ViewportFrameDesc&);
        void recordFrame(const GraphicsFrame&, const ViewportFrameDesc&, ViewRenderState&);
        void prepareImport(GraphicsContext&, const ViewportFrameDesc&, ViewRenderState&, ViewportReferenceRenderer* resident);
        void discardImportMesh(uint64_t mesh_id);
        [[nodiscard]] SceneUpscalerSelection sceneUpscalerSelection() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
    LFS_VIS_API void snapshotViewportReference(const ViewRenderState&, ViewportFrameDesc&);
    LFS_VIS_API void prepareViewportReference(GraphicsContext&, ViewRenderState&, bool export_locked, bool resize_deferring);
    LFS_VIS_API void clearViewportReference(ViewRenderState&);
    LFS_VIS_API void shutdownViewportReference(ViewRenderState&, GraphicsContext*);
    [[nodiscard]] LFS_VIS_API size_t viewportReferenceFramesInFlight(GraphicsContext&);
    [[nodiscard]] LFS_VIS_API size_t viewportReferenceMemoryBytes(GraphicsContext&, size_t additional_blocks, size_t additional_allocations);
    [[nodiscard]] LFS_VIS_API std::unique_ptr<ViewportReferenceRenderer> createViewportReferenceRenderer(
        GraphicsContext&, std::shared_ptr<ViewportReferenceResources>&);
} // namespace lfs::vis
