/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <cstdint>
#include <optional>
#include <string_view>

namespace lfs::core {

    enum class LazyMode : uint8_t {
        Off = 0,
        Shadow = 1,
        On = 2
    };

    struct LazyTelemetrySnapshot {
        uint64_t expr_nodes_created = 0;
        uint64_t materializations = 0;
        uint64_t eager_fallbacks = 0;
        uint64_t eager_fallback_host_read = 0;
        uint64_t eager_fallback_device_transfer = 0;
        uint64_t eager_fallback_mutation = 0;
        uint64_t eager_fallback_interop = 0;
        uint64_t eager_fallback_other = 0;
        uint64_t eager_fallback_size_heuristic = 0;
        uint64_t stateful_op_eager = 0;
        uint64_t kernel_launches = 0;
        uint64_t allocated_bytes = 0;
    };

    namespace internal {

        enum class LazyFallbackReason : uint8_t {
            Unspecified = 0,
            HostRead = 1,
            DeviceTransfer = 2,
            Mutation = 3,
            Interop = 4,
            Other = 5,
            SizeHeuristic = 6
        };

        class LFS_CORE_API LazyFallbackReasonScope {
        public:
            explicit LazyFallbackReasonScope(LazyFallbackReason reason);
            ~LazyFallbackReasonScope();

            LazyFallbackReasonScope(const LazyFallbackReasonScope&) = delete;
            LazyFallbackReasonScope& operator=(const LazyFallbackReasonScope&) = delete;

        private:
            LazyFallbackReason previous_ = LazyFallbackReason::Unspecified;
        };

        LFS_CORE_API LazyMode parse_lazy_mode_string(std::string_view value, LazyMode fallback = LazyMode::Off);
        LFS_CORE_API const char* lazy_mode_name(LazyMode mode);

        LFS_CORE_API LazyMode current_lazy_mode();
        LFS_CORE_API bool lazy_mode_enabled();
        LFS_CORE_API bool lazy_mode_shadow_enabled();

        // Testing helper: override mode at runtime. Pass std::nullopt to clear.
        LFS_CORE_API void set_lazy_mode_override_for_testing(std::optional<LazyMode> mode);
        LFS_CORE_API void clear_lazy_mode_cache_for_testing();

        LFS_CORE_API void reset_lazy_telemetry();
        LFS_CORE_API LazyTelemetrySnapshot lazy_telemetry_snapshot();

        LFS_CORE_API void telemetry_record_expr_node(uint64_t count = 1);
        LFS_CORE_API void telemetry_record_materialization(uint64_t bytes);
        LFS_CORE_API void telemetry_record_eager_fallback(uint64_t count = 1);
        LFS_CORE_API void telemetry_record_kernel_launch(uint64_t count = 1);
        LFS_CORE_API void telemetry_record_stateful_op_eager(uint64_t count = 1);

    } // namespace internal

} // namespace lfs::core
