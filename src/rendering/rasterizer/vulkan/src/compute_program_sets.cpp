/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/guarded_task.hpp"

// Custom compute program sets: built on a worker, adopted at a render boundary
// and retired with the same in-flight timeline keys as grown buffers.

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "gs_pipeline.h"
#include "shader_variants.hpp"

#include <algorithm>
#include <format>

VulkanGSPipeline::ComputeProgramSet::~ComputeProgramSet() {
    for (auto* list : {&programs, &fixed})
        for (auto& program : *list)
            destroyComputeProgram(context.device, context.dispatch, program);
}

lfs::Result<VulkanGSPipeline::ProgramSetHandle> VulkanGSPipeline::prepareProgramSet(
    std::vector<ComputeProgramSource> programs, std::function<void()> finished, const bool live) {
    if (device == VK_NULL_HANDLE || programs.empty())
        return vksplatProgramError(lfs::ErrorCode::FailedPrecondition, "Program sets require a device and hosts", LFS_SOURCE_SITE_CURRENT());
    auto fixed = missingFixedVariants(programs);
    auto& store = program_sets_;
    const std::lock_guard lock(store.mutex);
    if (store.stopping)
        return vksplatProgramError(lfs::ErrorCode::Cancelled, "Program-set worker has stopped", LFS_SOURCE_SITE_CURRENT());
    const auto id = ++store.next_key;
    ProgramSetHandle handle(new ProgramSetKey{id, store.generation.load()}, [releases = store.releases](const ProgramSetKey* key) {
        const std::lock_guard lock(releases->first);
        releases->second.push_back(key->id);
        delete key;
    });
    if (live)
        std::erase_if(store.queued, [](const auto& job) { return job.live; });
    store.queued.push_back({computeBuildContext(), std::move(programs), std::move(fixed), handle->generation, std::move(finished), id, live, handle});
    if (!store.worker.joinable())
        store.worker = std::thread(runProgramSetWorker, std::ref(store));
    store.changed.notify_all();
    return handle;
}

void VulkanGSPipeline::runProgramSetWorker(ProgramSetStore& store) {
    // The cache belongs to this thread; every job of one worker uses one device.
    VkPipelineCache cache = VK_NULL_HANDLE;
    VkDevice cache_device = VK_NULL_HANDLE;
    const auto build = [&](const ProgramSetJob& job) -> lfs::Result<std::unique_ptr<ComputeProgramSet>> {
        auto set = std::make_unique<ComputeProgramSet>();
        set->context = job.context;
        set->programs.reserve(job.programs.size());
        set->fixed.reserve(job.fixed.size());
        std::vector<_ComputePipeline*> programs;
        for (const auto& source : job.programs) {
            auto& program = set->programs.emplace_back(source.spec.bindings);
            if (auto prepared = prepareComputeProgram(job.context, program, source.spec, source.spirv); !prepared)
                return std::move(prepared).error();
            programs.push_back(&program);
        }
        for (const auto& source : job.fixed) {
            std::vector<uint32_t> spirv;
            try {
                spirv = loadSpirv(source.path);
            } catch (const lfs::Exception& e) {
                return lfs::Error{e.error()}.with_context(std::format("load custom variant '{}'", source.spec.label),
                                                          LFS_SOURCE_SITE_CURRENT());
            }
            auto& program = set->fixed.emplace_back(source.spec.bindings);
            if (auto prepared = prepareComputeProgram(job.context, program, source.spec, spirv); !prepared)
                return std::move(prepared).error();
            programs.push_back(&program);
        }
        if (auto created = createComputePrograms(job.context, cache, programs); !created)
            return std::move(created).error();
        return set;
    };

    const lfs::core::TaskContext context{.name = "user shader pipelines", .domain = lfs::ErrorDomain::Rendering, .operation_id = {}, .site = LFS_SOURCE_SITE_CURRENT()};
    std::uint64_t active_generation = 0, active_key = 0;
    std::function<void()> active_finished;
    lfs::core::run_guarded<void>(context, [&]() -> lfs::Status {
    std::unique_lock lock(store.mutex);
    for (;;) {
        store.changed.wait(lock, [&] { return store.stopping || !store.queued.empty(); });
        if (store.stopping)
            break;
        ProgramSetJob job = std::move(store.queued.front());
        active_generation = job.generation;
        active_key = job.key;
        active_finished = job.finished;
        store.queued.pop_front();
        store.building = true;
        lock.unlock();

        std::optional<lfs::Result<std::unique_ptr<ComputeProgramSet>>> outcome;
        lfs::core::run_guarded<std::unique_ptr<ComputeProgramSet>>(context, [&]() -> lfs::Result<std::unique_ptr<ComputeProgramSet>> {
            if (cache_device == VK_NULL_HANDLE) {
                cache_device = job.context.device;
                VkPipelineCacheCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
                if (vkCreatePipelineCache(cache_device, &info, nullptr, &cache) != VK_SUCCESS) cache = VK_NULL_HANDLE;
            }
            return build(job);
        }, [&](auto&& result) { outcome.emplace(std::move(result)); });
        auto built = std::move(*outcome);

        lock.lock();
        store.building = false;
        // A stale set is destroyed here, while the device is still alive:
        // shutdown bumps the generation and then joins this thread.
        const bool current = job.generation == store.generation.load() && !job.owner.expired();
        if (current) {
            store.built.emplace(job.key, std::move(built));
        }
        store.changed.notify_all();
        if (current && job.finished) {
            lock.unlock();
            job.finished();
            lock.lock();
        }
    }
    lock.unlock();
    return {}; }, [&](lfs::Status&& result) {
        if (result) return;
        std::unique_lock lock(store.mutex);
        store.building = false;
        store.stopping = true;
        if (active_generation == store.generation.load()) store.built.emplace(active_key, result.error());
        for (const auto& job : store.queued) store.built.emplace(job.key, result.error());
        auto queued = std::move(store.queued);
        store.queued.clear();
        store.changed.notify_all();
        lock.unlock();
        // Even a failed wake callback cannot strand other owners' wait signals.
        const auto wake = [](const std::function<void()>& finished) {
            try {
                if (finished) finished();
            } catch (...) {
                LOG_ERROR("A program set wake callback failed after the build worker stopped");
            }
        };
        wake(active_finished);
        for (const auto& job : queued) wake(job.finished); });
    if (cache != VK_NULL_HANDLE)
        vkDestroyPipelineCache(cache_device, cache, nullptr);
}

void VulkanGSPipeline::waitForProgramSetBuilds() {
    auto& store = program_sets_;
    std::unique_lock lock(store.mutex);
    store.changed.wait(lock, [&] { return store.queued.empty() && !store.building; });
}

lfs::Result<bool> VulkanGSPipeline::programSetReady(const ProgramSetHandle& handle) {
    if (commandBatchInProgress)
        return vksplatProgramError(lfs::ErrorCode::ContractViolation, "Program sets are collected outside recording", LFS_SOURCE_SITE_CURRENT());
    if (!handle)
        return true;
    auto& store = program_sets_;
    if (handle->generation != store.generation.load())
        return false;
    drainRetiredProgramSets();
    if (store.sets.contains(handle->id))
        return true;
    std::unique_lock lock(store.mutex);
    const auto found = store.built.find(handle->id);
    if (found == store.built.end())
        return false;
    if (!found->second)
        return found->second.error();
    auto set = std::move(*found->second);
    store.built.erase(found);
    lock.unlock();
    for (auto& program : set->fixed) {
        if (adoptedFixedVariant(program.diagnostic_name)) {
            destroyComputeProgram(set->context.device, set->context.dispatch, program);
            continue;
        }
        if (!custom_fixed_) {
            custom_fixed_ = std::make_unique<ComputeProgramSet>();
            custom_fixed_->context = set->context;
        }
        custom_fixed_->programs.push_back(std::exchange(program, _ComputePipeline(std::vector<int>{})));
    }
    // A ready set is not current yet: useProgramSet resolves it. The current set's fixed
    // variants were adopted before it was selected; later ones serve chains it does not host.
    store.sets.emplace(handle->id, std::move(set));
    return true;
}

void VulkanGSPipeline::useProgramSet(const ProgramSetHandle& handle) {
    LFS_VK_DEBUG_ASSERT(!commandBatchInProgress, "Program sets are selected before recording");
    drainRetiredProgramSets();
    auto& store = program_sets_;
    auto* set = handle && handle->generation == store.generation.load() && store.sets.contains(handle->id) ? store.sets.at(handle->id).get() : nullptr;
    if (store.current == set)
        return;
    store.current = set;
    ++program_set_serial_;
    resolveCustomVariants();
}

std::vector<VulkanGSPipeline::FixedProgramSource> VulkanGSPipeline::missingFixedVariants(
    const std::span<const ComputeProgramSource> hosts) const {
    using lfs::rendering::CustomChain;
    using lfs::rendering::CustomKind;
    std::array<std::array<bool, 2>, 3> needed{};
    for (const auto& host : hosts)
        for (const auto& row : lfs::rendering::shader_variants)
            if (row.custom_kind == CustomKind::Host && row.chain != CustomChain::None)
                for (const bool banded : {false, true})
                    if (host.spec.label == (banded ? row.custom_banded : row.custom))
                        needed[static_cast<size_t>(row.chain)][banded] = true;
    std::vector<FixedProgramSource> fixed;
    for (const auto* const base : all_compute_pipelines) {
        const auto path = lfs::core::utf8_to_path(base->spirv_path);
        const auto* row = lfs::rendering::shaderVariants(lfs::core::path_to_utf8(path.stem()));
        if (!row || row->custom_kind != CustomKind::Spirv)
            continue;
        for (const bool banded : {false, true}) {
            const auto label = std::string(row->customVariantLabel(banded));
            if (!needed[static_cast<size_t>(row->chain)][banded] || adoptedFixedVariant(label) ||
                std::ranges::any_of(fixed, [&](const FixedProgramSource& f) { return f.spec.label == label; }))
                continue;
            const uint32_t checked = base->expected_workgroup_size_x != 0 ? 1u : 0u;
            FixedProgramSource source{.spec = {.label = label,
                                               .bindings = base->buffer_layouts,
                                               .workgroup = {base->expected_workgroup_size_x, checked, checked},
                                               .required_subgroup_size = base->required_subgroup_size},
                                      .path = lfs::core::path_to_utf8(path.parent_path() / (label + ".spv"))};
            // The rasters read the projection's geometry after their own bindings.
            if (row->chain == CustomChain::Gut)
                source.spec.bindings.push_back(std::ranges::max(base->buffer_layouts) + 1);
            fixed.push_back(std::move(source));
        }
    }
    return fixed;
}

VulkanGSPipeline::_ComputePipeline* VulkanGSPipeline::adoptedFixedVariant(const std::string_view label) const {
    if (!custom_fixed_)
        return nullptr;
    const auto found = std::ranges::find(custom_fixed_->programs, label, &_ComputePipeline::diagnostic_name);
    return found == custom_fixed_->programs.end() ? nullptr : &*found;
}

void VulkanGSPipeline::resolveCustomVariants() {
    custom_variants_.clear();
    auto* const set = program_sets_.current;
    if (!set)
        return;
    for (auto* const base : all_compute_pipelines) {
        const auto* row = lfs::rendering::shaderVariants(
            lfs::core::path_to_utf8(lfs::core::utf8_to_path(base->spirv_path).stem()));
        if (!row || row->custom.empty())
            continue;
        CustomVariant variant{.labels = {std::string(row->customVariantLabel(false)), std::string(row->customVariantLabel(true))}};
        for (std::size_t banded = 0; banded < 2; ++banded) {
            const auto& label = variant.labels[banded];
            const auto host = std::ranges::find(set->programs, label, &_ComputePipeline::diagnostic_name);
            variant.hosts[banded] = host != set->programs.end() ? &*host : adoptedFixedVariant(label);
        }
        custom_variants_.emplace(base, std::move(variant));
    }
}

VulkanGSPipeline::CustomEffectDraw VulkanGSPipeline::setCustomEffect(
    const bool enabled, const std::span<_ComputePipeline* const> programs) {
    custom_effect_active_ = false;
    if (!enabled)
        return {};
    CustomEffectDraw draw;
    if (!program_sets_.current)
        return draw;
    for (auto* const program : programs) {
        const auto found = custom_variants_.find(program);
        if (found != custom_variants_.end() && !found->second.hosts[banded_export_active_])
            return draw;
    }
    custom_effect_active_ = draw.active = true;
    return draw;
}

VulkanGSPipeline::_ComputePipeline& VulkanGSPipeline::customVariant(_ComputePipeline& program) {
    if (!custom_effect_active_)
        return program;
    const auto found = custom_variants_.find(&program);
    if (found == custom_variants_.end())
        return program;
    if (auto* const host = found->second.hosts[banded_export_active_])
        return *host;
    lfs::rendering::throw_renderer_contract(
        std::format("A custom effect draw runs '{}' but setCustomEffect did not require it",
                    found->second.labels[banded_export_active_]),
        LFS_SOURCE_SITE_CURRENT());
}

void VulkanGSPipeline::invalidateProgramSets() noexcept {
    program_sets_.generation.fetch_add(1);
}

void VulkanGSPipeline::stopProgramSetWorker() noexcept {
    auto& store = program_sets_;
    std::unique_lock lock(store.mutex);
    store.generation.fetch_add(1);
    store.stopping = true;
    store.queued.clear();
    store.changed.notify_all();
    lock.unlock();
    if (store.worker.joinable())
        store.worker.join();
    lock.lock();
    store.built.clear();
    store.stopping = false;
}

void VulkanGSPipeline::drainRetiredProgramSets() {
    auto& store = program_sets_;
    std::vector<std::uint64_t> released;
    {
        const std::lock_guard lock(store.releases->first);
        released.swap(store.releases->second);
    }
    for (const auto key : released) {
        {
            const std::lock_guard lock(store.mutex);
            store.built.erase(key);
            std::erase_if(store.queued, [key](const auto& job) { return job.key == key; });
        }
        if (auto entry = store.sets.extract(key); !entry.empty()) {
            if (store.current == entry.mapped().get()) {
                store.current = nullptr;
                resolveCustomVariants();
            }
            store.retired.push_back({std::move(entry.mapped()), inFlightRetireKeys()});
        }
    }
    if (store.retired.size() > kMaxRetiredProgramSets)
        waitForPendingBatch();
    std::erase_if(store.retired, [&](const RetiredProgramSet& retired) { return retireKeysComplete(retired.keys); });
}
