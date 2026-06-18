/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lod_page_cache.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <expected>
#include <fstream>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace lfs::vis {
    namespace {

        std::size_t pageRequestBudgetFor(const std::size_t physical_pages) {
            if (physical_pages == 0) {
                return 0;
            }
            const std::size_t lower = std::min<std::size_t>(8, physical_pages);
            const std::size_t upper = std::min<std::size_t>(64, physical_pages);
            return std::clamp<std::size_t>(physical_pages / 4, lower, upper);
        }

        std::size_t decodeWorkerCount() {
            const std::size_t hw = std::max<std::size_t>(std::thread::hardware_concurrency(), 1);
            return std::clamp<std::size_t>(hw / 2, 2, 8);
        }

        constexpr std::size_t kStaleRequestFrames = 8;
        std::size_t staleRequestFrames() { return kStaleRequestFrames; }

    } // namespace

    // Bounded worker pool decoding RAD chunks by descending priority. Queue
    // entries are re-prioritized (or expired) every submit from the latest
    // traversal interest; jobs already running are never cancelled.
    struct LodPageCache::DecodeScheduler {
        struct Entry {
            std::uint32_t chunk = kInvalidPage;
            std::uint32_t page = kInvalidPage;
            std::uint32_t priority = 0;
            std::uint64_t enqueued_frame = 0;
            std::uint64_t generation = 0;
            lfs::core::SplatLodTree::ChunkFileRange range{};
        };
        struct Result {
            std::uint32_t chunk = kInvalidPage;
            std::uint32_t page = kInvalidPage;
            std::uint64_t generation = 0;
            // Empty = the page sink accepted the chunk (publish flows through
            // the upload engine); non-empty = read/decode/sink failure.
            std::string error;
        };

        explicit DecodeScheduler(RadSourceSnapshot source)
            : source_(std::move(source)) {
            const std::size_t worker_count = decodeWorkerCount();
            workers_.reserve(worker_count);
            for (std::size_t i = 0; i < worker_count; ++i) {
                workers_.emplace_back([this] { workerLoop(); });
            }
        }

        ~DecodeScheduler() {
            {
                std::lock_guard lock(mutex_);
                stop_ = true;
            }
            cv_.notify_all();
            for (auto& worker : workers_) {
                worker.join();
            }
        }

        void enqueue(Entry entry) {
            {
                std::lock_guard lock(mutex_);
                queue_.push_back(entry);
                std::push_heap(queue_.begin(), queue_.end(), byPriority);
            }
            cv_.notify_one();
        }

        // Refreshes priorities from the latest interest map and expires queued
        // entries that nothing asked for within `stale_frames`. Returns the
        // dropped entries so the cache can release their reserved page slots.
        std::vector<Entry> reprioritize(const std::unordered_map<std::uint32_t, std::uint32_t>& interest,
                                        const std::uint64_t frame,
                                        const std::uint64_t stale_frames) {
            std::vector<Entry> dropped;
            std::lock_guard lock(mutex_);
            std::size_t write = 0;
            for (std::size_t read = 0; read < queue_.size(); ++read) {
                Entry entry = queue_[read];
                if (const auto it = interest.find(entry.chunk); it != interest.end()) {
                    entry.priority = it->second;
                    entry.enqueued_frame = frame;
                } else if (frame > entry.enqueued_frame + stale_frames) {
                    dropped.push_back(entry);
                    continue;
                }
                queue_[write++] = entry;
            }
            queue_.resize(write);
            std::make_heap(queue_.begin(), queue_.end(), byPriority);
            return dropped;
        }

        std::vector<Result> drainCompleted() {
            std::lock_guard lock(mutex_);
            return std::exchange(completed_, {});
        }

        [[nodiscard]] std::size_t outstanding() const {
            std::lock_guard lock(mutex_);
            return queue_.size() + in_flight_ + completed_.size();
        }

        [[nodiscard]] bool chunkInFlight(const std::uint32_t chunk) const {
            std::lock_guard lock(mutex_);
            for (const auto& entry : queue_) {
                if (entry.chunk == chunk) {
                    return true;
                }
            }
            for (const auto& result : completed_) {
                if (result.chunk == chunk) {
                    return true;
                }
            }
            return in_flight_chunks_.count(chunk) > 0;
        }

        // One-lock snapshot of every chunk this scheduler currently owns;
        // submitInternal screens requests against it instead of paying a
        // mutex + queue scan per request.
        void snapshotChunks(std::unordered_set<std::uint32_t>& out) const {
            std::lock_guard lock(mutex_);
            for (const auto& entry : queue_) {
                out.insert(entry.chunk);
            }
            for (const auto& result : completed_) {
                out.insert(result.chunk);
            }
            out.insert(in_flight_chunks_.begin(), in_flight_chunks_.end());
        }

        void setSink(std::shared_ptr<const PageSink> sink) {
            std::lock_guard lock(mutex_);
            sink_ = std::move(sink);
        }

    private:
        static bool byPriority(const Entry& a, const Entry& b) { return a.priority < b.priority; }

        void workerLoop() {
            std::ifstream stream;
            std::vector<std::uint8_t> raw;
            std::unique_lock lock(mutex_);
            while (true) {
                cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
                if (stop_) {
                    return;
                }
                std::pop_heap(queue_.begin(), queue_.end(), byPriority);
                const Entry entry = queue_.back();
                queue_.pop_back();
                ++in_flight_;
                in_flight_chunks_.insert(entry.chunk);
                const std::shared_ptr<const PageSink> sink = sink_;
                lock.unlock();

                Result result{.chunk = entry.chunk, .page = entry.page, .generation = entry.generation, .error = {}};
                if (!stream.is_open()) {
                    (void)lfs::core::open_file_for_read(source_.path, std::ios::binary, stream);
                }
                if (sink == nullptr || !(*sink)) {
                    result.error = "page sink not installed";
                } else if (!stream.is_open()) {
                    result.error = "Failed to open RAD source for chunk decode";
                } else {
                    raw.resize(entry.range.file_bytes);
                    stream.clear();
                    stream.seekg(static_cast<std::streamoff>(entry.range.file_offset), std::ios::beg);
                    stream.read(reinterpret_cast<char*>(raw.data()),
                                static_cast<std::streamsize>(raw.size()));
                    if (!stream.good()) {
                        result.error = "Failed to read RAD chunk bytes";
                        stream.close();
                    } else {
                        result.error = (*sink)(entry.chunk, entry.page, entry.generation,
                                               std::span<const std::uint8_t>(raw));
                    }
                }

                lock.lock();
                --in_flight_;
                in_flight_chunks_.erase(entry.chunk);
                if (!result.error.empty()) {
                    completed_.push_back(std::move(result));
                }
            }
        }

        RadSourceSnapshot source_;
        std::shared_ptr<const PageSink> sink_;
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::vector<Entry> queue_;
        std::vector<Result> completed_;
        std::unordered_multiset<std::uint32_t> in_flight_chunks_;
        std::size_t in_flight_ = 0;
        bool stop_ = false;
        std::vector<std::thread> workers_;
    };

    LodPageCache::LodPageCache() = default;

    LodPageCache::~LodPageCache() = default;

    void LodPageCache::reset() {
        scheduler_.reset();
        pages_.clear();
        pending_uploads_.clear();
        rad_source_ = {};
        snapshot_ = {};
        clock_ = 0;
        frame_ = 0;
        page_payload_bytes_ = 0;
        deferred_requests_ = 0;
        admitted_requests_ = 0;
        last_admission_log_frame_ = 0;
    }

    void LodPageCache::configure(const std::size_t logical_chunk_count,
                                 std::size_t physical_page_capacity,
                                 const std::size_t root_chunk_count,
                                 const std::size_t page_payload_bytes,
                                 const bool disk_backed) {
        reset();
        if (logical_chunk_count == 0) {
            return;
        }
        protect_window_frames_ = kDefaultProtectWindowFrames;
        page_payload_bytes_ = page_payload_bytes;
        disk_backed_ = disk_backed;

        if (physical_page_capacity == 0 || physical_page_capacity > logical_chunk_count) {
            physical_page_capacity = logical_chunk_count;
        }

        pages_.resize(physical_page_capacity);
        snapshot_.logical_chunks = logical_chunk_count;
        snapshot_.physical_pages = physical_page_capacity;
        snapshot_.page_to_chunk.assign(physical_page_capacity, kInvalidPage);
        snapshot_.chunk_to_page.assign(logical_chunk_count, kInvalidPage);
        snapshot_.page_resident_frame.assign(physical_page_capacity, 0);

        const std::size_t pinned_roots = std::min(root_chunk_count, logical_chunk_count);
        root_chunk_count_ = pinned_roots;

        // Full-capacity TENSOR-backed pools pre-publish everything: their
        // uploads cover all pages from resident tensors. Disk-backed pools
        // must stream regardless of capacity - only a preview prefix exists
        // in memory, so pre-publishing would tell the selector chunks are
        // resident that no upload will ever fill.
        if (!disk_backed_ && physical_page_capacity == logical_chunk_count) {
            for (std::uint32_t chunk = 0; chunk < logical_chunk_count; ++chunk) {
                const bool pin = chunk < pinned_roots;
                const std::size_t page = chooseEvictionSlot();
                if (page >= pages_.size()) {
                    break;
                }
                reserveUpload(page, chunk, pin, 0);
                publishPage(page, chunk, pin);
            }
        }
    }

    void LodPageCache::setRadSource(const lfs::core::SplatLodTree::RadSource* const source,
                                    const int max_sh_degree,
                                    const bool lod_opacity_encoded) {
        if (source == nullptr || !source->valid()) {
            rad_source_ = {};
            scheduler_.reset();
            return;
        }
        if (rad_source_.path == source->path &&
            rad_source_.chunks.size() == source->chunks.size() &&
            rad_source_.max_sh_degree == max_sh_degree &&
            rad_source_.lod_opacity_encoded == lod_opacity_encoded) {
            return;
        }
        scheduler_.reset();
        rad_source_.path = source->path;
        rad_source_.chunks = source->chunks;
        rad_source_.max_sh_degree = max_sh_degree;
        rad_source_.lod_opacity_encoded = lod_opacity_encoded;
    }

    void LodPageCache::setPageSink(PageSink sink) {
        page_sink_ = sink ? std::make_shared<const PageSink>(std::move(sink)) : nullptr;
        if (scheduler_) {
            scheduler_->setSink(page_sink_);
        }
    }

    void LodPageCache::ensureRootResidency() {
        if (pages_.empty() ||
            (!disk_backed_ && snapshot_.physical_pages >= snapshot_.logical_chunks)) {
            return;
        }
        for (std::uint32_t chunk = 0;
             chunk < static_cast<std::uint32_t>(root_chunk_count_); ++chunk) {
            // Self-healing: a failed stream (sink not yet installed, decode
            // error) released the reservation; retry until the root is in.
            (void)requestResident(chunk, true,
                                  std::numeric_limits<std::uint32_t>::max());
        }
    }

    void LodPageCache::beginFrame() {
        ++frame_;
        deferred_requests_ = 0;
        admitted_requests_ = 0;
        ensureRootResidency();
    }

    void LodPageCache::submitTraversalPriority(const std::span<const std::uint32_t> chunks,
                                               const std::span<const std::uint32_t> protected_chunks) {
        // Caller order encodes priority; synthesize descending values so the
        // scheduler preserves it.
        std::vector<ChunkRequest> requests;
        requests.reserve(chunks.size());
        const std::uint32_t top = std::numeric_limits<std::uint32_t>::max() - 1;
        std::uint32_t rank = 0;
        for (const std::uint32_t chunk : chunks) {
            requests.push_back({.chunk = chunk, .priority = top - std::min(rank, top)});
            ++rank;
        }
        submitInternal(requests, protected_chunks, /*stamp_resident_requests=*/false);
    }

    void LodPageCache::submitTraversalPriority(const std::span<const ChunkRequest> requests,
                                               const std::span<const std::uint32_t> protected_chunks) {
        submitInternal(requests, protected_chunks, /*stamp_resident_requests=*/true);
    }

    void LodPageCache::submitInternal(const std::span<const ChunkRequest> requests,
                                      const std::span<const std::uint32_t> protected_chunks,
                                      const bool stamp_resident_requests) {
        if (!configured()) {
            return;
        }
        const auto submit_start = std::chrono::steady_clock::now();
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
                stampProtection(chunk);
            }
        }

        const std::size_t request_budget = requestBudgetPages();

        // Reprioritize queued decodes from this frame's interest; expire
        // requests nothing asked for recently and release their reservations.
        // Requests arrive priority-descending and the queue only ever holds
        // budget-recent admissions, so the top few-budgets prefix carries
        // every entry the queue could match; chunks past it expire as stale,
        // which is the wanted outcome when the view moved that far.
        if (scheduler_) {
            const std::size_t interest_cap =
                std::min(requests.size(), request_budget * 4);
            std::unordered_map<std::uint32_t, std::uint32_t> interest;
            interest.reserve(interest_cap);
            for (std::size_t i = 0; i < interest_cap; ++i) {
                auto [it, inserted] =
                    interest.try_emplace(requests[i].chunk, requests[i].priority);
                if (!inserted) {
                    it->second = std::max(it->second, requests[i].priority);
                }
            }
            for (const auto& dropped : scheduler_->reprioritize(interest, frame_, staleRequestFrames())) {
                if (dropped.page < pages_.size() &&
                    pages_[dropped.page].loading_chunk == dropped.chunk) {
                    pages_[dropped.page].loading_chunk = kInvalidPage;
                }
            }
        }

        eviction_scratch_enabled_ = true;
        eviction_scratch_dirty_ = true;
        std::size_t new_requests = 0;
        std::size_t wants = 0;
        std::size_t budget_deferred = 0;
        std::size_t refused_no_slot = 0;
        std::size_t refused_margin = 0;

        // Per-submit activity snapshot: chunks pending upload, queued or
        // running in the scheduler, or loading into a slot. Screening against
        // a set keeps the request loop O(1) per request — the per-request
        // decodeInFlight/pending scans measured 800 ms/frame at 95K pages.
        std::unordered_set<std::uint32_t> active_chunks;
        active_chunks.reserve(pending_uploads_.size() + requests.size());
        for (const auto& upload : pending_uploads_) {
            active_chunks.insert(upload.chunk);
        }
        if (scheduler_) {
            scheduler_->snapshotChunks(active_chunks);
        }
        for (const auto& slot : pages_) {
            if (slot.loading_chunk != kInvalidPage) {
                active_chunks.insert(slot.loading_chunk);
            }
        }
        const std::size_t outstanding_base = outstandingWorkCount();

        for (const auto& request : requests) {
            const std::uint32_t chunk = request.chunk;
            if (chunk >= snapshot_.logical_chunks) {
                continue;
            }
            const bool resident =
                chunk < snapshot_.chunk_to_page.size() &&
                snapshot_.chunk_to_page[chunk] != kInvalidPage;
            if (resident && stamp_resident_requests) {
                stampProtection(chunk, request.priority);
            }
            // Once the budget is gone no later request can admit (the list is
            // priority-descending); resident stamps above stay complete, the
            // rest is deferral bookkeeping only — skip the hash work.
            if (!resident &&
                (new_requests >= request_budget ||
                 outstanding_base + new_requests >= request_budget)) {
                ++wants;
                ++deferred_requests_;
                ++budget_deferred;
                continue;
            }
            const bool active_request = resident || active_chunks.count(chunk) > 0;
            if (!active_request) {
                ++wants;
                if (new_requests >= request_budget ||
                    outstanding_base + new_requests >= request_budget) {
                    ++deferred_requests_;
                    ++budget_deferred;
                    continue;
                }
            }
            const AdmitResult result =
                requestResident(chunk, false, request.priority, protected_pages,
                                /*screened_inactive=*/!active_request);
            if (!active_request) {
                switch (result) {
                case AdmitResult::Admitted:
                    ++new_requests;
                    ++admitted_requests_;
                    active_chunks.insert(chunk);
                    break;
                case AdmitResult::NoSlot:
                    ++deferred_requests_;
                    ++refused_no_slot;
                    break;
                case AdmitResult::MarginRefused:
                    ++deferred_requests_;
                    ++refused_margin;
                    break;
                case AdmitResult::Invalid:
                    ++deferred_requests_;
                    break;
                case AdmitResult::AlreadyActive:
                    break; // raced into residency/flight between the checks
                }
            }
        }

        eviction_scratch_enabled_ = false;

        // Admission telemetry: a frozen pool (wants but zero admissions with
        // margin/no-slot refusals) is indistinguishable from healthy flow
        // control in the HUD's single deferred number; this line tells the
        // refusal mode and whether any page is even evictable.
        if (wants > 0) {
            const bool frozen = new_requests == 0 && (refused_no_slot + refused_margin) > 0;
            const std::uint64_t interval = frozen ? 16 : 64;
            if (frame_ >= last_admission_log_frame_ + interval) {
                last_admission_log_frame_ = frame_;
                std::size_t freeable = 0;
                for (const auto& slot : pages_) {
                    if (slot.pinned || slot.chunk == kInvalidPage ||
                        slot.loading_chunk != kInvalidPage ||
                        slot.protected_until_frame > frame_) {
                        continue;
                    }
                    if (effectivePriority(slot) == 0u) {
                        ++freeable;
                    }
                }
                LOG_PERF("vksplat.lod_admission frame={} wants={} admitted={} no_slot={} "
                         "margin={} budget={} freeable={} resident={}/{}{}",
                         frame_,
                         wants,
                         new_requests,
                         refused_no_slot,
                         refused_margin,
                         budget_deferred,
                         freeable,
                         snapshot_.resident_chunks,
                         snapshot_.physical_pages,
                         frozen ? " FROZEN" : "");
            }
        }

        // Render-thread cost spike log: chooseEvictionSlot is O(pages) per
        // admission, so streaming frames at large pools are the suspect for
        // stutter. Fires only when this call actually got expensive.
        const double submit_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - submit_start)
                                     .count();
        if (submit_ms > 2.0) {
            LOG_PERF("vksplat.lod_submit_slow ms={:.2f} requests={} wants={} admitted={} "
                     "no_slot={} margin={} pages={}",
                     submit_ms, requests.size(), wants, new_requests,
                     refused_no_slot, refused_margin, pages_.size());
        }
    }

    std::vector<LodPageCache::PendingUpload> LodPageCache::drainPendingUploads(const std::size_t max_bytes) {
        collectFinishedDecodes();
        if (max_bytes == 0 || page_payload_bytes_ == 0 ||
            pending_uploads_.size() <= max_bytes / page_payload_bytes_) {
            std::vector<PendingUpload> uploads;
            uploads.swap(pending_uploads_);
            return uploads;
        }
        const std::size_t take = std::max<std::size_t>(max_bytes / page_payload_bytes_, 1);
        std::vector<PendingUpload> uploads(
            std::make_move_iterator(pending_uploads_.begin()),
            std::make_move_iterator(pending_uploads_.begin() + static_cast<std::ptrdiff_t>(take)));
        pending_uploads_.erase(pending_uploads_.begin(),
                               pending_uploads_.begin() + static_cast<std::ptrdiff_t>(take));
        return uploads;
    }

    void LodPageCache::completeUploads(const std::span<const PendingUpload> uploads) {
        for (const auto& upload : uploads) {
            if (upload.page == kInvalidPage ||
                upload.chunk == kInvalidPage ||
                upload.page >= pages_.size() ||
                upload.chunk >= snapshot_.logical_chunks) {
                continue;
            }
            auto& slot = pages_[upload.page];
            if (!upload.error.empty()) {
                // Failed past the decode stage (async copy error, pool
                // reconfigured mid-flight): release the reservation or the
                // slot would never become evictable again.
                if (slot.loading_chunk == upload.chunk) {
                    slot.loading_chunk = kInvalidPage;
                }
                continue;
            }
            if (slot.loading_chunk != upload.chunk &&
                slot.chunk != upload.chunk) {
                continue;
            }
            publishPage(upload.page, upload.chunk, slot.pinned);
        }
    }

    std::size_t LodPageCache::outstandingWorkCount() const {
        return (scheduler_ ? scheduler_->outstanding() : 0) + pending_uploads_.size();
    }

    void LodPageCache::stampProtection(const std::uint32_t chunk, const std::uint32_t priority) {
        if (chunk >= snapshot_.chunk_to_page.size()) {
            return;
        }
        const std::uint32_t page = snapshot_.chunk_to_page[chunk];
        if (page == kInvalidPage || page >= pages_.size()) {
            return;
        }
        auto& slot = pages_[page];
        slot.protected_until_frame = frame_ + protect_window_frames_;
        slot.last_used = ++clock_;
        slot.last_priority = std::max(effectivePriority(slot), priority);
        slot.last_touch_frame = frame_;
    }

    std::size_t LodPageCache::requestBudgetPages() const {
        if (page_payload_bytes_ == 0) {
            return pageRequestBudgetFor(pages_.size());
        }
        // Decode workers (≤8) starve when the queue runs dry, so the depth
        // must track page size: smaller pages decode proportionally faster.
        // The byte cap equals the old 32-page ceiling at 65K-splat pages.
        constexpr std::size_t kRequestBudgetBytes = 180u << 20;
        const std::size_t byte_cap =
            std::max<std::size_t>(kRequestBudgetBytes / page_payload_bytes_, 32);
        return std::clamp<std::size_t>(pages_.size() / 16,
                                       std::min<std::size_t>(32, pages_.size()),
                                       byte_cap);
    }

    LodPageCache::AdmitResult LodPageCache::requestResident(
        const std::uint32_t chunk,
        const bool pin,
        const std::uint32_t priority,
        const std::span<const std::uint8_t> protected_pages,
        const bool screened_inactive) {
        if (chunk >= snapshot_.logical_chunks || pages_.empty()) {
            return AdmitResult::Invalid;
        }

        const std::uint32_t current_page = snapshot_.chunk_to_page[chunk];
        if (current_page != kInvalidPage && current_page < pages_.size()) {
            pages_[current_page].last_used = ++clock_;
            pages_[current_page].pinned = pages_[current_page].pinned || pin;
            return AdmitResult::AlreadyActive;
        }
        if (!screened_inactive) {
            if (decodeInFlight(chunk)) {
                return AdmitResult::AlreadyActive;
            }
            for (auto& pending : pending_uploads_) {
                if (pending.chunk == chunk) {
                    if (pending.page < pages_.size()) {
                        pages_[pending.page].last_used = ++clock_;
                        pages_[pending.page].pinned = pages_[pending.page].pinned || pin;
                    }
                    return AdmitResult::AlreadyActive;
                }
            }
        }

        const std::size_t page = chooseEvictionSlot(protected_pages);
        if (page >= pages_.size()) {
            // Every slot is pinned, loading, or inside its protection window:
            // defer rather than churn a page the current cut renders from.
            return AdmitResult::NoSlot;
        }

        // Priority admission: stealing an occupied slot must buy more screen
        // quality than it destroys, or a working set larger than the pool
        // rotates through the model forever (region refines, neighbor
        // coarsens, repeat). Deferred refinements keep rendering their parent
        // stand-ins; idle pages decay so a camera move reclaims them.
        if (frame_ > 0 && !pin && pages_[page].chunk != kInvalidPage) {
            // Hysteresis: displacement must buy at least 2x pixel scale
            // (one float exponent step), or near-equal sibling regions swap
            // pages forever at full pool.
            constexpr std::uint32_t kAdmissionMargin = 0x00800000;
            const std::uint32_t resistance = effectivePriority(pages_[page]);
            const std::uint32_t bar =
                resistance > std::numeric_limits<std::uint32_t>::max() - kAdmissionMargin
                    ? std::numeric_limits<std::uint32_t>::max()
                    : resistance + kAdmissionMargin;
            if (priority <= bar) {
                return AdmitResult::MarginRefused;
            }
        }

        reserveUpload(page, chunk, pin, priority);
        return AdmitResult::Admitted;
    }

    std::uint32_t LodPageCache::effectivePriority(const PageSlot& slot) const {
        // The selector readback re-stamps every page the live cut renders
        // from each frame, so an untouched page is ground truth for "not part
        // of the current view" — it surrenders its admission resistance
        // entirely once past a short flicker guard. Post-pause reallocation
        // is then bounded by stream throughput (sub-second), not a decay
        // clock (a gradual halving per 30 idle frames made new-view wants
        // arrive seconds late). Touched pages keep full resistance; the
        // admission margin alone guards same-view steals at full pool.
        constexpr std::uint64_t kIdleFreeFrames = 12;
        const std::uint64_t age = frame_ - std::min(frame_, slot.last_touch_frame);
        return age > kIdleFreeFrames ? 0u : slot.last_priority;
    }

    void LodPageCache::buildEvictionScratch(
        const std::span<const std::uint8_t> protected_pages) const {
        eviction_free_scratch_.clear();
        eviction_heap_.clear();
        const auto heap_after = [](const EvictionCandidate& a, const EvictionCandidate& b) {
            return a.priority > b.priority ||
                   (a.priority == b.priority && a.last_used > b.last_used);
        };
        for (std::size_t page = pages_.size(); page-- > 0;) {
            if (page < protected_pages.size() && protected_pages[page] != 0) {
                continue;
            }
            const auto& slot = pages_[page];
            if (slot.loading_chunk != kInvalidPage) {
                continue;
            }
            if (slot.chunk == kInvalidPage) {
                // Descending so back() pops the lowest index, matching the
                // legacy first-free-by-index choice.
                eviction_free_scratch_.push_back(static_cast<std::uint32_t>(page));
                continue;
            }
            if (slot.pinned || slot.protected_until_frame > frame_) {
                continue;
            }
            eviction_heap_.push_back({
                .priority = frame_ > 0 ? effectivePriority(slot) : 0u,
                .last_used = slot.last_used,
                .page = static_cast<std::uint32_t>(page),
            });
        }
        std::make_heap(eviction_heap_.begin(), eviction_heap_.end(), heap_after);
    }

    std::size_t LodPageCache::chooseEvictionSlot(
        const std::span<const std::uint8_t> protected_pages) const {
        if (eviction_scratch_enabled_) {
            if (eviction_scratch_dirty_) {
                buildEvictionScratch(protected_pages);
                eviction_scratch_dirty_ = false;
            }
            while (!eviction_free_scratch_.empty()) {
                const std::uint32_t page = eviction_free_scratch_.back();
                eviction_free_scratch_.pop_back();
                if (pages_[page].chunk == kInvalidPage &&
                    pages_[page].loading_chunk == kInvalidPage) {
                    return page;
                }
            }
            const auto heap_after = [](const EvictionCandidate& a, const EvictionCandidate& b) {
                return a.priority > b.priority ||
                       (a.priority == b.priority && a.last_used > b.last_used);
            };
            while (!eviction_heap_.empty()) {
                std::pop_heap(eviction_heap_.begin(), eviction_heap_.end(), heap_after);
                const EvictionCandidate candidate = eviction_heap_.back();
                eviction_heap_.pop_back();
                // Revalidate: an earlier admission this submit may have made
                // the slot loading, or a resident-want stamp protected it.
                const auto& slot = pages_[candidate.page];
                if (slot.pinned || slot.loading_chunk != kInvalidPage ||
                    slot.chunk == kInvalidPage || slot.protected_until_frame > frame_) {
                    continue;
                }
                return candidate.page;
            }
            return pages_.size();
        }
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
        std::uint32_t best_priority = std::numeric_limits<std::uint32_t>::max();
        std::uint64_t best_time = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t page = 0; page < pages_.size(); ++page) {
            if (page < protected_pages.size() && protected_pages[page] != 0) {
                continue;
            }
            const auto& slot = pages_[page];
            if (slot.pinned || slot.loading_chunk != kInvalidPage) {
                continue;
            }
            if (slot.protected_until_frame > frame_) {
                continue;
            }
            const std::uint32_t priority = frame_ > 0 ? effectivePriority(slot) : 0u;
            if (priority < best_priority ||
                (priority == best_priority && slot.last_used < best_time)) {
                best_priority = priority;
                best_time = slot.last_used;
                best_page = page;
            }
        }
        return best_page;
    }

    void LodPageCache::reserveUpload(const std::size_t page,
                                     const std::uint32_t chunk,
                                     const bool pin,
                                     const std::uint32_t priority) {
        auto& slot = pages_[page];
        slot.loading_chunk = chunk;
        slot.last_used = ++clock_;
        slot.last_priority = priority;
        slot.last_touch_frame = frame_;
        slot.pinned = slot.pinned || pin;

        const auto enqueue_pending = [&] {
            pending_uploads_.push_back({
                .page = static_cast<std::uint32_t>(page),
                .chunk = chunk,
                .generation = snapshot_.generation,
                .error = {},
            });
        };

        // In-core pages (no RAD source) are filled from the resident tensors
        // by the renderer (D2D); everything else — pinned roots included —
        // streams from disk through the decode scheduler into the upload
        // engine.
        if (!rad_source_.valid() || chunk >= rad_source_.chunks.size()) {
            enqueue_pending();
            return;
        }

        if (!scheduler_) {
            scheduler_ = std::make_unique<DecodeScheduler>(rad_source_);
            scheduler_->setSink(page_sink_);
        }
        scheduler_->enqueue({
            .chunk = chunk,
            .page = static_cast<std::uint32_t>(page),
            .priority = priority,
            .enqueued_frame = frame_,
            .generation = snapshot_.generation,
            .range = rad_source_.chunks[chunk],
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
        // A freshly streamed page is about to be rendered from; protect it
        // until the (one-frame-stale) selector readback can list it itself.
        // Only under a frame-driven caller — legacy per-call protection
        // semantics stay untouched when beginFrame is never used.
        if (frame_ > 0) {
            slot.protected_until_frame = frame_ + protect_window_frames_;
        }

        snapshot_.page_to_chunk[page] = chunk;
        snapshot_.chunk_to_page[chunk] = static_cast<std::uint32_t>(page);
        snapshot_.page_resident_frame[page] = static_cast<std::uint32_t>(frame_);
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
        if (!scheduler_) {
            return;
        }
        // Success flows through the upload engine; only failures come back
        // here. Releasing the reservation lets the chunk re-request later.
        for (auto& result : scheduler_->drainCompleted()) {
            LOG_WARN("LOD chunk {} stream failed: {}", result.chunk, result.error);
            if (result.page < pages_.size() &&
                pages_[result.page].loading_chunk == result.chunk) {
                pages_[result.page].loading_chunk = kInvalidPage;
            }
        }
    }

    bool LodPageCache::decodeInFlight(const std::uint32_t chunk) const {
        if (scheduler_ && scheduler_->chunkInFlight(chunk)) {
            return true;
        }
        for (const auto& slot : pages_) {
            if (slot.loading_chunk == chunk) {
                return true;
            }
        }
        return false;
    }

} // namespace lfs::vis
