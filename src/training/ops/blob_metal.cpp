/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "blob_gpu.hpp"
#include "metal_families.hpp"
#include "metal_kernels.hpp"
#include <vector>
namespace lfs::training {
    namespace {
        struct Backend {
            static uint64_t address(const core::Tensor& tensor) { return metal::address(tensor); }
            static void launch(uint32_t operation, const blob_detail::Parameters& p,
                               std::initializer_list<const core::Tensor*> reads,
                               std::initializer_list<const core::Tensor*> writes, size_t items) {
                std::vector<const core::Tensor*> uses;
                for (auto* t : reads)
                    if (t->is_valid())
                        uses.push_back(t);
                for (auto* t : writes)
                    if (t->is_valid())
                        uses.push_back(t);
                const std::pair<uint32_t, uint32_t> constant{1, operation};
                metal::kernels().launch({.function = "blob_main", .params = std::as_bytes(std::span(&p, 1)), .uses = uses, .groups = {static_cast<uint32_t>((items + 255) / 256), 1, 1}, .group = {256, 1, 1}, .constants = std::span(&constant, 1)});
            }
        };
    } // namespace
    const gpu_ops::BlobOps& metal_blob_ops() {
        return blob_detail::Implementation<Backend>::table();
    }
} // namespace lfs::training
