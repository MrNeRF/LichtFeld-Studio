/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "flip_gpu.hpp"
#include "metal_kernels.hpp"
#include <vector>
namespace lfs::training {
    namespace {
        struct Backend {
            static uint64_t address(const core::Tensor& t) { return metal::address(t); }
            static void launch(const flip_detail::Parameters& p, std::initializer_list<const core::Tensor*> reads,
                               std::initializer_list<const core::Tensor*> writes, size_t groups) {
                std::vector<const core::Tensor*> uses(reads);
                uses.insert(uses.end(), writes.begin(), writes.end());
                LFS_ASSERT(groups <= std::numeric_limits<uint32_t>::max());
                metal::kernels().launch({.function = "flip_main", .params = std::as_bytes(std::span(&p, 1)), .uses = uses, .groups = {static_cast<uint32_t>(groups), 1, 1}, .group = {256, 1, 1}});
            }
        };
    } // namespace
    core::Tensor metal_flip_error_map(const core::Tensor& a, const core::Tensor& b, float ppd) { return flip_detail::Implementation<Backend>::map(a, b, ppd); }
    core::Tensor metal_flip_error_image(const core::Tensor& a) { return flip_detail::Implementation<Backend>::image(a); }
} // namespace lfs::training
