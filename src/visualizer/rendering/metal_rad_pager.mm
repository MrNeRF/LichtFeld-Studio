/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "metal_rad_pager.hpp"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "core/sh_layout.hpp"
#include "core/tensor_backend.hpp"
#include "frame_budget.hpp"
#include "lod_upload_engine.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>
#include <thread>

namespace lfs::vis {
    namespace {
        std::atomic<uint64_t> next_signature{1};
        constexpr size_t kPage = core::SplatLodTree::kChunkSplats;
    } // namespace
    struct MetalRadPager::Impl {
        id<MTLDevice> device;
        // Destruction order: decode workers must finish before the engine dies.
        LodUploadEngine engine;
        LodPageCache cache;
        core::RadPagePool pool;
        const core::SplatData* model = nullptr;
        const core::SplatLodTree* tree = nullptr;
        core::SplatLodTree::NodeMetaView meta;
        Settings settings;
        uint64_t signature = 0;
        uint32_t nodes = 0;
        std::array<core::Tensor, 7> preview;
        core::Tensor deleted;
        uint64_t deleted_version = 0;
        const void* deleted_pointer = nullptr;
        uint32_t frozen_frames = 0;
        uint64_t last_publish_frame = 0;
        explicit Impl(id<MTLDevice> value) : device(value) {}
    };
    MetalRadPager::MetalRadPager(id<MTLDevice> device) : impl_(std::make_unique<Impl>(device)) {}
    MetalRadPager::~MetalRadPager() = default;
    void MetalRadPager::configure(const core::SplatData& model, void* device,
                                  void* consumer, Settings settings) {
        auto& i = *impl_;
        if (!std::isfinite(settings.vram_fraction) || settings.vram_fraction <= 0 || settings.vram_fraction > 1)
            throw std::invalid_argument(std::format("Metal RAD pool fraction must be in (0, 1] (fraction={})", settings.vram_fraction));
        if (i.model == &model && i.tree == model.lod_tree.get() && i.settings.pool_splats == settings.pool_splats && i.settings.vram_fraction == settings.vram_fraction) {
            i.settings.fade_frames = settings.fade_frames;
            return;
        }
        const auto* tree = model.lod_tree.get();
        if (!tree || !tree->rad_source.valid() || tree->rad_source.chunk_size < kPage || tree->rad_source.chunk_size % kPage != 0 ||
            !tree->has_tree() || !tree->total_nodes() || tree->total_nodes() > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument(std::format("Metal RAD paging requires the native ordered chunk hierarchy (tree_present={}, source_valid={}, chunk_splats={}, required_page_splats={}, ordered_tree={}, nodes={})", tree != nullptr, tree && tree->rad_source.valid(), tree ? tree->rad_source.chunk_size : 0, kPage, tree && tree->has_tree(), tree ? tree->total_nodes() : 0));
        auto meta = tree->meta_view;
        if (!meta.valid()) {
            auto opened = io::open_rad_meta_sidecar(tree->rad_source.path);
            if (!opened) {
                const auto built = io::build_rad_meta_sidecar(tree->rad_source.path);
                if (!built)
                    throw std::runtime_error(std::format("Metal RAD metadata sidecar build failed (path={}, error={})", tree->rad_source.path.string(), built.error().message));
                opened = io::open_rad_meta_sidecar(tree->rad_source.path);
            }
            if (!opened)
                throw std::runtime_error(std::format("Metal RAD metadata sidecar unavailable (path={}, error={})", tree->rad_source.path.string(), opened.error()));
            meta = std::move(*opened);
        }
        const size_t nodes = tree->total_nodes(), chunks = (nodes + kPage - 1) / kPage;
        if (meta.node_count != nodes || meta.chunk_count != chunks || tree->rad_source.chunks.size() != chunks)
            throw std::invalid_argument(std::format("Metal RAD metadata and payload chunk extents differ (meta_nodes={}, tree_nodes={}, meta_chunks={}, expected_chunks={}, source_chunks={})", meta.node_count, nodes, meta.chunk_count, chunks, tree->rad_source.chunks.size()));
        // Stop producers before draining or replacing storage. Neither worker
        // calls back into the renderer; no renderer lock is acquired here.
        i.cache.reset();
        (void)i.engine.configure({});
        i.model = nullptr;
        i.pool = {};
        i.preview = {};
        i.deleted = {};
        i.deleted_pointer = nullptr;
        i.frozen_frames = 0;
        i.last_publish_frame = 0;
        const auto rest = uint32_t(model.max_sh_coeffs_rest());
        const auto slots = core::sh_float4_slots_for_rest(rest);
        const uint64_t page_bytes = kPage * uint64_t(12 + 8 + slots * 4 + 8 + 8 + 2 + 8 + 12) + 64;
        const uint64_t staging = 32 * (sizeof(core::RadPagePackedDesc) + core::rad_page_staging_bytes(kPage));
        const uint64_t recommended = i.device.recommendedMaxWorkingSetSize;
        const uint64_t budget = recommended ? recommended - recommended / 5 : std::numeric_limits<uint64_t>::max();
        const uint64_t allocated = i.device.currentAllocatedSize;
        const uint64_t free = budget > allocated ? budget - allocated : 0;
        const uint64_t pool_budget = uint64_t(double(free) * settings.vram_fraction);
        const size_t requested = settings.pool_splats ? (settings.pool_splats / kPage + (settings.pool_splats % kPage != 0)) : chunks;
        const size_t affordable = pool_budget > staging ? size_t((pool_budget - staging) / page_bytes) : 0;
        const size_t pages = std::min({chunks, requested, affordable, size_t(std::numeric_limits<uint32_t>::max() / kPage)});
        if (!pages)
            throw core::MemoryAllocationError({.domain = core::MemoryDomain::MetalDevice, .requested_bytes = staging + page_bytes * std::max<size_t>(1, requested), .label = "viewer.rad.pool", .operation = "rad.pool.admission"});
        core::RadPagePool pool;
        pool.page_splats = uint32_t(kPage);
        pool.sh_slots = slots;
        const size_t n = pages * kPage;
        const std::array<size_t, 9> sizes{n * 12, n * 8, n * slots * 4, n * 8, n * 8, n * 2, pages * 64, n * 8, n * 12};
        const core::GpuBackendScope scope(core::GpuBackend::Metal);
        for (size_t j = 0; j < sizes.size(); ++j)
            if (sizes[j])
                pool.regions[j] = core::Tensor::zeros({sizes[j]}, core::Device::GPU, core::DataType::UInt8);
        (void)i.engine.configure({pool}, device, consumer);
        // The captured mapped-file owner outlives every decode job, including
        // model replacement. The engine remains alive until cache.reset joins.
        i.cache.configure(chunks, pages, 1, size_t(page_bytes), true);
        i.cache.setRadSource(&tree->rad_source, model.get_max_sh_degree(), tree->lod_opacity_encoded);
        auto* engine = &i.engine;
        i.cache.setPageSink([engine, meta, degree = model.get_max_sh_degree(), encoded = tree->lod_opacity_encoded](uint32_t chunk, uint32_t page, uint64_t generation,
                                                                                                                    std::span<const uint8_t> bytes) -> std::string {
            auto* slot = engine->acquireStagingSlot();
            if (!slot)
                return std::format("Metal RAD upload engine unavailable (chunk={}, page={}, generation={}, bytes={})", chunk, page, generation, bytes.size());
            try {
                auto decoded = io::decode_rad_chunk_packed(bytes, degree, encoded, kPage, meta, chunk,
                                                           std::span<uint8_t>(slot->data, engine->stagingBytes()));
                if (!decoded) {
                    engine->releaseSlot(slot);
                    return std::move(decoded.error());
                }
                engine->submitPackedPage(slot, *decoded, page, generation);
                return {};
            } catch (const std::exception& error) {
                engine->releaseSlot(slot);
                return error.what();
            }
        });
        i.pool = std::move(pool);
        i.model = &model;
        i.tree = tree;
        i.meta = std::move(meta);
        i.settings = settings;
        i.nodes = uint32_t(nodes);
        i.signature = next_signature.fetch_add(1);
        LOG_INFO("Metal RAD page cache configured: nodes={} logical_chunks={} physical_pages={}", nodes, chunks, pages);
    }
    void MetalRadPager::advance(std::span<const uint32_t> touches) {
        auto& i = *impl_;
        if (!i.model)
            throw std::logic_error(std::format("Metal RAD pager is not configured (signature={}, nodes={}, model_present={})", i.signature, i.nodes, i.model != nullptr));
        i.cache.beginFrame();
        auto published = i.engine.collectPublished();
        for (const auto& page : published)
            if (!page.error.empty())
                LOG_ERROR("Metal RAD page upload failed (chunk {}): {}", page.chunk, page.error);
        i.cache.completeUploads(published);
        if (std::any_of(published.begin(), published.end(), [](const auto& page) { return page.error.empty(); }))
            i.last_publish_frame = i.cache.frameIndex();
        // Drain decoder failures; successful pages arrive exclusively through
        // collectPublished after completion, never through this pending list.
        const auto pending = i.cache.drainPendingUploads();
        if (!pending.empty())
            throw std::logic_error(std::format("Disk-backed Metal RAD cache produced a resident-tensor upload (uploads={}, frame={})", pending.size(), i.cache.frameIndex()));
        std::vector<LodPageCache::ChunkRequest> requests;
        std::vector<uint32_t> protected_chunks;
        for (size_t chunk = 0; chunk < std::min(touches.size(), i.cache.snapshot().logical_chunks); ++chunk) {
            if (touches[chunk] == LodPageCache::kInvalidPage)
                protected_chunks.push_back(uint32_t(chunk));
            else if (touches[chunk])
                requests.push_back({uint32_t(chunk), touches[chunk]});
        }
        i.cache.submitTraversalPriority(requests, protected_chunks);
        const bool frozen = i.cache.deferredRequestCount() > 0 && i.cache.admittedRequestCount() == 0 &&
                            !i.cache.hasOutstandingWork() && i.engine.idle();
        i.frozen_frames = frozen ? std::min(i.frozen_frames + 1, 30u) : 0;
    }
    void MetalRadPager::noteRendererCompletion(uint64_t value) { impl_->engine.noteRendererCompletion(value); }
    const core::RadPagePool& MetalRadPager::pool() const { return impl_->pool; }
    const LodPageCache& MetalRadPager::cache() const { return impl_->cache; }
    bool MetalRadPager::rootReady() const { return !impl_->cache.snapshot().chunk_to_page.empty() && impl_->cache.snapshot().chunk_to_page[0] != LodPageCache::kInvalidPage; }
    bool MetalRadPager::pending() const {
        const auto& i = *impl_;
        return !i.engine.idle() || i.cache.hasOutstandingWork() ||
               (i.cache.deferredRequestCount() > 0 && !frozen()) ||
               (i.settings.fade_frames && i.cache.frameIndex() < i.last_publish_frame + i.settings.fade_frames);
    }
    bool MetalRadPager::frozen() const { return impl_->frozen_frames >= 30; }
    void MetalRadPager::waitForRoot() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!rootReady()) {
            advance();
            if (rootReady())
                return;
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error(std::format("Metal RAD pinned root did not become resident before capture (timeout_ms=5000, nodes={}, chunks={}, pending={}, frozen={})", impl_->nodes, impl_->cache.snapshot().logical_chunks, pending(), frozen()));
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    uint32_t MetalRadPager::nodes() const { return impl_->nodes; }
    uint32_t MetalRadPager::physicalNodes() const { return uint32_t(impl_->cache.snapshot().physical_pages * kPage); }
    uint64_t MetalRadPager::signature() const { return impl_->signature; }
    uint32_t MetalRadPager::fadeFrames() const { return impl_->settings.fade_frames; }
    const std::array<core::Tensor, 7>& MetalRadPager::preview(const core::SplatData& model) {
        auto& i = *impl_;
        if (!i.preview[0].is_valid()) {
            const core::GpuBackendScope scope(core::GpuBackend::Metal);
            const std::array<const core::Tensor*, 7> sources{&model.means_raw(), &model.scaling_raw(), &model.rotation_raw(), &model.opacity_raw(), &model.sh0_raw(), &model.shN_raw(), &model.shN_value_bounds()};
            for (size_t j = 0; j < sources.size(); ++j)
                if (sources[j]->is_valid())
                    i.preview[j] = sources[j]->to(core::Device::GPU);
        }
        return i.preview;
    }
    const core::Tensor& MetalRadPager::deleted(const core::SplatData& model) {
        auto& i = *impl_;
        const auto& mask = model.deleted();
        const auto pointer = mask.is_valid() ? mask.data_ptr() : nullptr;
        if (i.deleted_pointer != pointer || i.deleted_version != model.deleted_mask_version()) {
            const core::GpuBackendScope scope(core::GpuBackend::Metal);
            i.deleted = mask.is_valid() ? mask.to(core::Device::GPU) : core::Tensor{};
            i.deleted_version = model.deleted_mask_version();
            i.deleted_pointer = pointer;
        }
        return i.deleted;
    }
} // namespace lfs::vis
