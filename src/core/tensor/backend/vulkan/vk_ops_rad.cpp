/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/rad_ops.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"

namespace lfs::core::internal {
    namespace {
        struct Push {
            std::array<uint64_t, 9> regions;
            uint64_t packed, descriptor;
            uint32_t page, page_splats, slots, padding;
        };
        static_assert(sizeof(Push) == 104);
        void dispatch(const RadPagePool& pool, uint32_t page, StorageRef descriptor, uint64_t packed,
                      std::span<const StorageRef> reads, uint32_t phase) {
            const auto context = acquire_vulkan_context();
            Push p{.packed = packed, .descriptor = vk::address(descriptor), .page = page, .page_splats = pool.page_splats, .slots = pool.sh_slots};
            std::vector<StorageRef> writes;
            for (size_t i = 0; i < 9; ++i)
                if (pool.regions[i].is_valid()) {
                    writes.push_back(storage_ref(pool.regions[i]));
                    p.regions[i] = vk::address(writes.back());
                }
            const std::array constants{phase};
            const auto& pipeline = context->pipelines().specialized("rad_page", sizeof(p), constants);
            context->recorders().record(reads, writes, [&](VkCommandBuffer cmd) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                vkCmdPushConstants(cmd, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
                vkCmdDispatch(cmd, (pool.page_splats + 255) / 256, 1, 1);
            });
        }
    } // namespace
    void vulkan_rad_page_dequant(const Tensor& packed, const RadPagePool& pool, uint32_t page) {
        const auto input = storage_ref(packed);
        const std::array reads{input};
        dispatch(pool, page, input, vk::address(input) + sizeof(RadPagePackedDesc), reads, 0);
    }
    void vulkan_rad_page_quantize(const RadPageSources& src, const RadPagePool& pool, uint32_t page) {
        struct Sources {
            uint64_t means, sh0, shN, rotation, scaling, opacity, sh_bounds;
            uint32_t offset, count, rest, half_sh, quant_sh, padding;
        };
        static_assert(sizeof(Sources) == 80);
        const auto address = [](const Tensor& t) { return t.is_valid() ? vk::address(storage_ref(t)) : uint64_t{0}; };
        const Sources source{address(src.means), address(src.sh0), src.sh_rest ? address(src.shN) : 0,
                             address(src.rotation), address(src.scaling), address(src.opacity), address(src.shN_bounds),
                             src.offset, src.count, src.sh_rest, src.shN.is_valid() && src.shN.dtype() == DataType::Float16, src.sh_q16, 0};
        const auto context = acquire_vulkan_context();
        vk::ScopedAllocation descriptor(*context, sizeof(source));
        context->memory().copy_host_to_device({.src = raw_storage_ref(const_cast<Sources*>(&source)),
                                               .dst = descriptor.storage(),
                                               .bytes = sizeof(source),
                                               .synchronous = false});
        std::vector<StorageRef> reads{descriptor.storage()};
        for (const auto* t : {&src.means, &src.sh0, &src.shN, &src.rotation, &src.scaling, &src.opacity, &src.shN_bounds})
            if (t->is_valid())
                reads.push_back(storage_ref(*t));
        dispatch(pool, page, descriptor.storage(), 0, reads, 1);
        if (source.shN && source.rest)
            dispatch(pool, page, descriptor.storage(), 0, reads, 2);
        dispatch(pool, page, descriptor.storage(), 0, reads, 3);
    }
} // namespace lfs::core::internal
