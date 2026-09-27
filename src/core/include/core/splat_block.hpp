/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/exportable_storage.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/tensor.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace lfs::core {

    // Backend half of exportable splat storage. Layout, views, growth
    // bookkeeping, rebind, and provenance stay in SplatExportableStorage.
    // CUDA reserves exportable VMM. Every other session reserves one device
    // allocation and treats commit as bookkeeping.
    class SplatBlockOps {
    public:
        virtual ~SplatBlockOps() = default;

        [[nodiscard]] virtual std::size_t granularity(int device) const = 0;

        [[nodiscard]] virtual std::shared_ptr<ExportableBlock>
        reserve(std::size_t initial_commit, int device, std::size_t reserve_bytes) = 0;

        virtual void commit_range(const std::shared_ptr<ExportableBlock>& block,
                                  std::size_t offset, std::size_t bytes) = 0;

        // Device-wide wait before a CUDA commit. Other backends have nothing
        // to drain ahead of a no-op commit.
        virtual void prepare_growth() = 0;

        // Initialize opacity and rotation rows added by grow. CUDA keeps the
        // existing stream copies; other backends fill through tensor ops.
        virtual void init_growth_slack(const std::shared_ptr<ExportableBlock>& block,
                                       std::size_t opacity_offset,
                                       std::size_t rotation_offset,
                                       std::size_t n_slack) = 0;

        [[nodiscard]] virtual Tensor bind_region(const std::shared_ptr<ExportableBlock>& block,
                                                 void* data,
                                                 std::size_t byte_offset,
                                                 TensorShape shape,
                                                 std::size_t capacity,
                                                 DataType dtype,
                                                 std::string external_kind) = 0;
    };

    [[nodiscard]] SplatBlockOps& splat_block_ops(GpuBackend backend);

} // namespace lfs::core
