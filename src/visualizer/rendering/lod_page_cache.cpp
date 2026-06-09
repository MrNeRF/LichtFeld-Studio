/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lod_page_cache.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <utility>

namespace lfs::vis {

    void LodPageCache::reset() {
        pages_.clear();
        pending_uploads_.clear();
        decode_jobs_.clear();
        rad_source_ = {};
        snapshot_ = {};
        clock_ = 0;
    }

    void LodPageCache::configure(const std::size_t logical_chunk_count,
                                 std::size_t physical_page_capacity,
                                 const std::size_t root_chunk_count) {
        reset();
        if (logical_chunk_count == 0) {
            return;
        }

        if (physical_page_capacity == 0 || physical_page_capacity > logical_chunk_count) {
            physical_page_capacity = logical_chunk_count;
        }

        pages_.resize(physical_page_capacity);
        snapshot_.logical_chunks = logical_chunk_count;
        snapshot_.physical_pages = physical_page_capacity;
        snapshot_.page_to_chunk.assign(physical_page_capacity, kInvalidPage);
        snapshot_.chunk_to_page.assign(logical_chunk_count, kInvalidPage);

        const std::size_t pinned_roots = std::min(root_chunk_count, logical_chunk_count);
        for (std::uint32_t chunk = 0; chunk < pinned_roots; ++chunk) {
            const std::size_t page = chooseEvictionSlot();
            if (page >= pages_.size()) {
                break;
            }
            reserveUpload(page, chunk, true);
            publishPage(page, chunk, true);
        }

        // Full-capacity native loads retain the current all-resident behavior and
        // still exercise the same maps used by paged RAD streaming.
        if (physical_page_capacity == logical_chunk_count) {
            for (std::uint32_t chunk = 0; chunk < logical_chunk_count; ++chunk) {
                if (chunk < pinned_roots) {
                    continue;
                }
                const std::size_t page = chooseEvictionSlot();
                if (page >= pages_.size()) {
                    break;
                }
                reserveUpload(page, chunk, false);
                publishPage(page, chunk, false);
            }
        }
    }

    void LodPageCache::setRadSource(const lfs::core::SplatLodTree::RadSource* const source,
                                    const int max_sh_degree,
                                    const bool lod_opacity_encoded) {
        if (source == nullptr || !source->valid()) {
            rad_source_ = {};
            return;
        }
        if (rad_source_.path == source->path &&
            rad_source_.chunks.size() == source->chunks.size() &&
            rad_source_.max_sh_degree == max_sh_degree &&
            rad_source_.lod_opacity_encoded == lod_opacity_encoded) {
            return;
        }
        rad_source_.path = source->path;
        rad_source_.chunks = source->chunks;
        rad_source_.max_sh_degree = max_sh_degree;
        rad_source_.lod_opacity_encoded = lod_opacity_encoded;
    }

    void LodPageCache::submitTraversalPriority(const std::span<const std::uint32_t> chunks,
                                               const std::span<const std::uint32_t> protected_chunks) {
        if (!configured()) {
            return;
        }
        collectFinishedDecodes();

        std::vector<std::uint8_t> protected_pages;
        if (!protected_chunks.empty()) {
            protected_pages.assign(pages_.size(), 0);
            for (const std::uint32_t chunk : protected_chunks) {
                if (chunk >= snapshot_.chunk_to_page.size()) {
                    continue;
                }
                const std::uint32_t page = snapshot_.chunk_to_page[chunk];
                if (page != kInvalidPage && page < protected_pages.size()) {
                    protected_pages[page] = 1;
                }
            }
        }

        for (const std::uint32_t chunk : chunks) {
            if (chunk >= snapshot_.logical_chunks) {
                continue;
            }
            requestResident(chunk, false, protected_pages);
        }
    }

    std::vector<LodPageCache::PendingUpload> LodPageCache::drainPendingUploads() {
        collectFinishedDecodes();
        std::vector<PendingUpload> uploads;
        uploads.swap(pending_uploads_);
        return uploads;
    }

    void LodPageCache::completeUploads(const std::span<const PendingUpload> uploads) {
        for (const auto& upload : uploads) {
            if (!upload.error.empty() ||
                upload.page == kInvalidPage ||
                upload.chunk == kInvalidPage ||
                upload.page >= pages_.size() ||
                upload.chunk >= snapshot_.logical_chunks) {
                continue;
            }
            auto& slot = pages_[upload.page];
            if (slot.loading_chunk != upload.chunk &&
                slot.chunk != upload.chunk) {
                continue;
            }
            publishPage(upload.page, upload.chunk, slot.pinned);
        }
    }

    bool LodPageCache::requestResident(const std::uint32_t chunk,
                                       const bool pin,
                                       const std::span<const std::uint8_t> protected_pages) {
        if (chunk >= snapshot_.logical_chunks || pages_.empty()) {
            return false;
        }

        const std::uint32_t current_page = snapshot_.chunk_to_page[chunk];
        if (current_page != kInvalidPage && current_page < pages_.size()) {
            pages_[current_page].last_used = ++clock_;
            pages_[current_page].pinned = pages_[current_page].pinned || pin;
            return true;
        }
        if (decodeInFlight(chunk)) {
            return true;
        }
        for (auto& pending : pending_uploads_) {
            if (pending.chunk == chunk) {
                if (pending.page < pages_.size()) {
                    pages_[pending.page].last_used = ++clock_;
                    pages_[pending.page].pinned = pages_[pending.page].pinned || pin;
                }
                return true;
            }
        }

        const std::size_t page = chooseEvictionSlot(protected_pages);
        if (page >= pages_.size()) {
            return false;
        }

        reserveUpload(page, chunk, pin);
        return true;
    }

    std::size_t LodPageCache::chooseEvictionSlot(
        const std::span<const std::uint8_t> protected_pages) const {
        for (std::size_t page = 0; page < pages_.size(); ++page) {
            if (page < protected_pages.size() && protected_pages[page] != 0) {
                continue;
            }
            if (pages_[page].chunk == kInvalidPage &&
                pages_[page].loading_chunk == kInvalidPage) {
                return page;
            }
        }

        std::size_t best_page = pages_.size();
        std::uint64_t best_time = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t page = 0; page < pages_.size(); ++page) {
            if (page < protected_pages.size() && protected_pages[page] != 0) {
                continue;
            }
            const auto& slot = pages_[page];
            if (slot.pinned || slot.loading_chunk != kInvalidPage) {
                continue;
            }
            if (slot.last_used < best_time) {
                best_time = slot.last_used;
                best_page = page;
            }
        }
        return best_page;
    }

    void LodPageCache::reserveUpload(const std::size_t page,
                                     const std::uint32_t chunk,
                                     const bool pin) {
        invalidateResidentPage(page);

        auto& slot = pages_[page];
        slot.loading_chunk = chunk;
        slot.last_used = ++clock_;
        slot.pinned = slot.pinned || pin;

        const auto enqueue_pending = [&] {
            pending_uploads_.push_back({
                .page = static_cast<std::uint32_t>(page),
                .chunk = chunk,
                .generation = snapshot_.generation,
                .decoded_chunk = std::nullopt,
                .error = {},
            });
        };

        // Root pages are made visible before upload so the first exact traversal can
        // still render root-only while the renderer fills the pinned physical page.
        // Keep that bootstrap upload synchronous via the renderer's resident tensor
        // fallback; all non-root RAD pages use the async decode queue.
        if (pin || !rad_source_.valid() || chunk >= rad_source_.chunks.size()) {
            enqueue_pending();
            return;
        }

        const auto path = rad_source_.path;
        const auto range = rad_source_.chunks[chunk];
        const int max_sh_degree = rad_source_.max_sh_degree;
        const bool lod_opacity_encoded = rad_source_.lod_opacity_encoded;
        decode_jobs_.push_back({
            .page = static_cast<std::uint32_t>(page),
            .chunk = chunk,
            .generation = snapshot_.generation,
            .future = std::async(std::launch::async,
                                 [path, range, max_sh_degree, lod_opacity_encoded] {
                                     return lfs::io::load_rad_chunk(path,
                                                                    range,
                                                                    max_sh_degree,
                                                                    lod_opacity_encoded);
                                 }),
        });
    }

    void LodPageCache::publishPage(const std::size_t page,
                                   const std::uint32_t chunk,
                                   const bool pin) {
        auto& slot = pages_[page];
        if (slot.chunk == chunk &&
            chunk < snapshot_.chunk_to_page.size() &&
            snapshot_.chunk_to_page[chunk] == page) {
            slot.loading_chunk = kInvalidPage;
            slot.last_used = ++clock_;
            slot.pinned = slot.pinned || pin;
            return;
        }
        invalidateResidentPage(page);

        slot.chunk = chunk;
        slot.loading_chunk = kInvalidPage;
        slot.last_used = ++clock_;
        slot.pinned = slot.pinned || pin;

        snapshot_.page_to_chunk[page] = chunk;
        snapshot_.chunk_to_page[chunk] = static_cast<std::uint32_t>(page);
        ++snapshot_.resident_chunks;
        ++snapshot_.generation;
    }

    void LodPageCache::invalidateResidentPage(const std::size_t page) {
        if (page >= pages_.size()) {
            return;
        }
        auto& slot = pages_[page];
        if (slot.chunk == kInvalidPage) {
            return;
        }
        if (slot.chunk < snapshot_.chunk_to_page.size() &&
            snapshot_.chunk_to_page[slot.chunk] == page) {
            snapshot_.chunk_to_page[slot.chunk] = kInvalidPage;
        }
        if (page < snapshot_.page_to_chunk.size()) {
            snapshot_.page_to_chunk[page] = kInvalidPage;
        }
        slot.chunk = kInvalidPage;
        if (snapshot_.resident_chunks > 0) {
            --snapshot_.resident_chunks;
        }
        ++snapshot_.generation;
    }

    void LodPageCache::collectFinishedDecodes() {
        using namespace std::chrono_literals;
        auto it = decode_jobs_.begin();
        while (it != decode_jobs_.end()) {
            if (!it->future.valid() ||
                it->future.wait_for(0ms) != std::future_status::ready) {
                ++it;
                continue;
            }
            PendingUpload upload{
                .page = it->page,
                .chunk = it->chunk,
                .generation = it->generation,
                .decoded_chunk = std::nullopt,
                .error = {},
            };
            auto decoded = it->future.get();
            if (decoded) {
                upload.decoded_chunk = std::move(*decoded);
            } else {
                upload.error = decoded.error();
                if (it->page < pages_.size() &&
                    pages_[it->page].loading_chunk == it->chunk) {
                    pages_[it->page].loading_chunk = kInvalidPage;
                }
            }
            pending_uploads_.push_back(std::move(upload));
            it = decode_jobs_.erase(it);
        }
    }

    bool LodPageCache::decodeInFlight(const std::uint32_t chunk) const {
        for (const auto& job : decode_jobs_) {
            if (job.chunk == chunk) {
                return true;
            }
        }
        for (const auto& slot : pages_) {
            if (slot.loading_chunk == chunk) {
                return true;
            }
        }
        return false;
    }

} // namespace lfs::vis
