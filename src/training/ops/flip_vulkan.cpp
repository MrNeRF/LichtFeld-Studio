/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "flip_gpu.hpp"
#include "vulkan/dispatch.hpp"
#include <vector>
namespace lfs::training {
    namespace {
        struct Backend {
            static uint64_t address(const core::Tensor& t) { return vulkan::address(t); }
            static void launch(const flip_detail::Parameters& p, std::initializer_list<const core::Tensor*> reads,
                               std::initializer_list<const core::Tensor*> writes, size_t groups) {
                std::vector<vulkan::StorageRef> r, w;
                for (auto* t : reads)
                    r.push_back(vulkan::ref(*t));
                for (auto* t : writes)
                    w.push_back(vulkan::ref(*t));
                LFS_ASSERT(groups <= std::numeric_limits<uint32_t>::max());
                // Vulkan guarantees at least 65535 workgroups per dispatch dimension.
                // Keep the same row mapping for full-resolution evaluation images.
                for (size_t offset = 0; offset < groups; offset += 65535) {
                    auto chunk = p;
                    chunk.group_offset = static_cast<uint32_t>(offset);
                    vulkan::dispatch("flip", chunk, r, w, static_cast<uint32_t>(std::min<size_t>(65535, groups - offset)));
                }
            }
        };
    } // namespace
    core::Tensor vulkan_flip_error_map(const core::Tensor& a, const core::Tensor& b, float ppd) { return flip_detail::Implementation<Backend>::map(a, b, ppd); }
    core::Tensor vulkan_flip_error_image(const core::Tensor& a) { return flip_detail::Implementation<Backend>::image(a); }
} // namespace lfs::training
