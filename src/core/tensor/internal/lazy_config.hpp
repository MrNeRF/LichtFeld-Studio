/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <cstdint>

namespace lfs::core {

    struct LazyTelemetrySnapshot {
        uint64_t expr_nodes_created = 0;
        uint64_t materializations = 0;
        uint64_t kernel_launches = 0;
        uint64_t allocated_bytes = 0;
        uint64_t expr_nodes_live = 0;
        uint64_t tensor_mappings_live = 0;
        uint64_t expr_nodes_peak = 0;
        uint64_t tensor_mappings_peak = 0;
        uint64_t expr_nodes_dropped = 0;
        uint64_t expr_node_limit = 0;
    };

    namespace internal {

        LFS_CORE_API void reset_lazy_telemetry();
        LFS_CORE_API LazyTelemetrySnapshot lazy_telemetry_snapshot();

        LFS_CORE_API void telemetry_record_expr_node(uint64_t count = 1);
        LFS_CORE_API void telemetry_record_expr_node_drop(uint64_t count = 1);
        LFS_CORE_API void telemetry_publish_expr_registry_counts(uint64_t live_nodes,
                                                                 uint64_t live_tensor_mappings,
                                                                 uint64_t node_limit);
        LFS_CORE_API void telemetry_record_materialization(uint64_t bytes);
        LFS_CORE_API void telemetry_record_kernel_launch(uint64_t count = 1);

    } // namespace internal

} // namespace lfs::core
