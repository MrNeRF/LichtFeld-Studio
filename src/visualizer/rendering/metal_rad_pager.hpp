/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#if defined(__APPLE__) && defined(__OBJC__)
#include "core/splat_data.hpp"
#include "core/tensor_rad.hpp"
#include "lod_page_cache.hpp"
#import <Metal/Metal.h>
#include <memory>
#include <span>

namespace lfs::vis {
    class VulkanContext;
    // Uses the shared decode scheduler and tensor upload engine. Residency is
    // published only after GPU completion; no full-tree CPU traversal is needed.
    class LFS_VIS_API MetalRadPager {
    public:
        struct Settings {
            size_t pool_splats = 0;
            float vram_fraction = .15f;
            uint32_t fade_frames = 12;
            bool operator==(const Settings&) const = default;
        };
        explicit MetalRadPager(id<MTLDevice>);
        ~MetalRadPager();
        MetalRadPager(const MetalRadPager&) = delete;
        MetalRadPager& operator=(const MetalRadPager&) = delete;
        void configure(const core::SplatData&, VulkanContext&, void* consumer_timeline, Settings);
        void advance(std::span<const uint32_t> touches = {});
        // Call only after the consuming native command has been committed.
        void noteRendererCompletion(uint64_t submitted_value);
        const core::RadPagePool& pool() const;
        const LodPageCache& cache() const;
        bool rootReady() const;
        bool pending() const;
        bool frozen() const;
        void waitForRoot();
        uint32_t nodes() const;
        uint32_t physicalNodes() const;
        uint64_t signature() const;
        uint32_t fadeFrames() const;
        // The loader's coarse resident prefix keeps the first frame useful while
        // the pinned root is decoded. Only that prefix is migrated, once.
        const std::array<core::Tensor, 7>& preview(const core::SplatData&);
        const core::Tensor& deleted(const core::SplatData&);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
#endif
