/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"
#include "kernels/tensor_labels.hpp"
namespace lfs::core::internal {
    namespace {
        template <typename T>
        T* data(const StorageRef& storage) {
            return reinterpret_cast<T*>(static_cast<uint8_t*>(storage.data) + storage.byte_offset);
        }
        template <typename T>
        T* data(const std::optional<StorageRef>& storage) {
            return storage ? data<T>(*storage) : nullptr;
        }
    } // namespace
    void CudaBackendOps::update_labels(const StorageRef output, const StorageRef selected, const LabelUpdateProgram& p,
                                       const ExecContext context) {
        LFS_FACADE_TRACE(update_labels);
        if (p.indices) {
            if (p.existing)
                copy_device_to_device(CopyRequest{.src = *p.existing,
                                                  .dst = output,
                                                  .bytes = p.output_count,
                                                  .synchronous = false,
                                                  .context = context});
            else
                memset(FillRequest{.dst = output, .bytes = p.output_count, .value = 0, .context = context});
        }
        tensor_ops::launch_update_labels({.output = data<uint8_t>(output),
                                          .selected = data<uint8_t>(selected),
                                          .existing = data<uint8_t>(p.existing),
                                          .locked = data<uint8_t>(p.locked),
                                          .allowed = data<uint8_t>(p.allowed),
                                          .indices = data<int32_t>(p.indices),
                                          .categories = data<int32_t>(p.categories),
                                          .count = static_cast<uint32_t>(p.count),
                                          .output_count = static_cast<uint32_t>(p.output_count),
                                          .allowed_count = static_cast<uint32_t>(p.allowed_count),
                                          .label = p.label,
                                          .mode = p.mode},
                                         context.cuda_stream);
    }
} // namespace lfs::core::internal
