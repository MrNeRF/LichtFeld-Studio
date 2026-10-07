/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/structure_vulkan.hpp"
#include "structure_gpu.hpp"
#include "vulkan/dispatch.hpp"
#include <vector>
namespace lfs::training {
    namespace {
        struct Backend {
            static uint64_t address(const core::Tensor& tensor) {
                return tensor.is_valid() ? vulkan::address(tensor) : 0;
            }
            static void launch(uint32_t operation, const structure_detail::Parameters& p,
                               std::initializer_list<const core::Tensor*> reads,
                               std::initializer_list<const core::Tensor*> writes, size_t items) {
                std::vector<vulkan::StorageRef> r, w;
                for (auto* t : reads)
                    if (t->is_valid())
                        r.push_back(vulkan::ref(*t));
                for (auto* t : writes)
                    if (t->is_valid())
                        w.push_back(vulkan::ref(*t));
                vulkan::dispatch("structure", p, r, w, static_cast<uint32_t>((items + 255) / 256), operation);
            }
        };
    } // namespace
    const gpu_ops::StructureOps& vulkan_structure_ops() {
        return structure_detail::Implementation<Backend>::table();
    }
} // namespace lfs::training
