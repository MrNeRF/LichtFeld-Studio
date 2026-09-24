/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "core/tensor_rad.hpp"
#include "core/tensor_upload.hpp"
#include "lod_page_cache.hpp"
#include <memory>
#include <span>
namespace lfs::vis {
    // One bounded producer/consumer engine for both tensor backends. Decode
    // workers fill CPU tensor slots; a single upload worker batches the ready
    // pages on the tensor backend's queue, separate from renderer compute.
    class LFS_VIS_API LodUploadEngine {
    public:
        struct DeviceLayout {
            lfs::core::RadPagePool pool;
            [[nodiscard]] bool valid() const { return pool.regions[0].is_valid(); }
        };
        struct StagingSlot {
            uint8_t* data = nullptr;
        };
        struct ResidentPage {
            uint32_t page = 0, offset = 0, count = 0;
        };
        LodUploadEngine();
        ~LodUploadEngine();
        LodUploadEngine(const LodUploadEngine&) = delete;
        LodUploadEngine& operator=(const LodUploadEngine&) = delete;
        std::vector<LodPageCache::PendingUpload> configure(DeviceLayout layout, void* device = nullptr, void* consumer_timeline = nullptr);
        [[nodiscard]] bool configured() const;
        [[nodiscard]] bool idle() const;
        [[nodiscard]] size_t stagingBytes() const;
        [[nodiscard]] StagingSlot* acquireStagingSlot();
        void releaseSlot(StagingSlot*);
        void submitPackedPage(StagingSlot*, const lfs::core::RadPagePackedDesc&, uint32_t page, uint64_t generation);
        [[nodiscard]] std::vector<LodPageCache::PendingUpload> collectPublished();
        std::vector<LodPageCache::PendingUpload> drainAndSync();
        [[nodiscard]] uint64_t lastPublishedSignalValue() const;
        [[nodiscard]] void* timeline() const;
        // Value must already be submitted by the renderer. New uploads wait
        // GPU-side before reusing pool slots; publication waits in reverse.
        void noteRendererCompletion(uint64_t value);
        // Resident roots use the same queue. Caller publishes with a GPU wait.
        lfs::core::TensorCompletion quantizeResident(const lfs::core::RadPageSources& source,
                                                     std::span<const ResidentPage> pages);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
