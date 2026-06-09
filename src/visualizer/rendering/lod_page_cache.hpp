/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "io/formats/rad.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <optional>
#include <span>
#include <vector>

namespace lfs::vis {

    class LodPageCache {
    public:
        static constexpr std::uint32_t kInvalidPage = lfs::core::SplatLodTree::kInvalidPage;
        static constexpr std::size_t   kChunkSplats = lfs::core::SplatLodTree::kChunkSplats;

        struct Snapshot {
            std::vector<std::uint32_t> page_to_chunk;
            std::vector<std::uint32_t> chunk_to_page;
            std::size_t logical_chunks = 0;
            std::size_t physical_pages = 0;
            std::size_t resident_chunks = 0;
            std::uint64_t generation = 0;
        };

        struct PendingUpload {
            std::uint32_t page = kInvalidPage;
            std::uint32_t chunk = kInvalidPage;
            std::uint64_t generation = 0;
            std::optional<lfs::io::RadDecodedChunk> decoded_chunk;
            std::string error;
        };

        void reset();
        void configure(std::size_t logical_chunk_count,
                       std::size_t physical_page_capacity,
                       std::size_t root_chunk_count = 1);
        void setRadSource(const lfs::core::SplatLodTree::RadSource* source,
                          int max_sh_degree,
                          bool lod_opacity_encoded);
        void submitTraversalPriority(std::span<const std::uint32_t> chunks,
                                     std::span<const std::uint32_t> protected_chunks = {});
        [[nodiscard]] std::vector<PendingUpload> drainPendingUploads();
        void completeUploads(std::span<const PendingUpload> uploads);

        [[nodiscard]] const Snapshot& snapshot() const { return snapshot_; }
        [[nodiscard]] bool configured() const { return snapshot_.logical_chunks > 0; }
        [[nodiscard]] bool fullyResident() const {
            return snapshot_.resident_chunks == snapshot_.logical_chunks;
        }

    private:
        struct PageSlot {
            std::uint32_t chunk = kInvalidPage;
            std::uint32_t loading_chunk = kInvalidPage;
            std::uint64_t last_used = 0;
            bool pinned = false;
        };
        struct RadSourceSnapshot {
            std::filesystem::path path;
            std::vector<lfs::core::SplatLodTree::ChunkFileRange> chunks;
            int max_sh_degree = 0;
            bool lod_opacity_encoded = false;

            [[nodiscard]] bool valid() const { return !path.empty() && !chunks.empty(); }
        };
        struct DecodeJob {
            std::uint32_t page = kInvalidPage;
            std::uint32_t chunk = kInvalidPage;
            std::uint64_t generation = 0;
            std::future<std::expected<lfs::io::RadDecodedChunk, std::string>> future;
        };

        bool requestResident(std::uint32_t chunk,
                             bool pin,
                             std::span<const std::uint8_t> protected_pages = {});
        [[nodiscard]] std::size_t chooseEvictionSlot(
            std::span<const std::uint8_t> protected_pages = {}) const;
        void reserveUpload(std::size_t page, std::uint32_t chunk, bool pin);
        void publishPage(std::size_t page, std::uint32_t chunk, bool pin);
        void invalidateResidentPage(std::size_t page);
        void collectFinishedDecodes();
        [[nodiscard]] bool decodeInFlight(std::uint32_t chunk) const;

        std::vector<PageSlot> pages_;
        std::vector<PendingUpload> pending_uploads_;
        std::vector<DecodeJob> decode_jobs_;
        RadSourceSnapshot rad_source_;
        Snapshot snapshot_;
        std::uint64_t clock_ = 0;
    };

} // namespace lfs::vis
