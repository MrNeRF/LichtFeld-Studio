/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/lazy_config.hpp"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <string>

namespace lfs::core::internal {

    namespace {

        struct LazyTelemetryCounters {
            std::atomic<uint64_t> expr_nodes_created{0};
            std::atomic<uint64_t> materializations{0};
            std::atomic<uint64_t> eager_fallbacks{0};
            std::atomic<uint64_t> eager_fallback_host_read{0};
            std::atomic<uint64_t> eager_fallback_device_transfer{0};
            std::atomic<uint64_t> eager_fallback_mutation{0};
            std::atomic<uint64_t> eager_fallback_interop{0};
            std::atomic<uint64_t> eager_fallback_other{0};
            std::atomic<uint64_t> kernel_launches{0};
            std::atomic<uint64_t> allocated_bytes{0};
        };

        struct LazyRuntimeState {
            std::atomic<int> override_mode{static_cast<int>(-1)};
            std::atomic<int> cached_env_mode{static_cast<int>(LazyMode::Off)};
            std::atomic<bool> has_cached_env_mode{false};
            LazyTelemetryCounters telemetry{};
        };

        LazyRuntimeState& lazy_runtime_state() {
            static LazyRuntimeState state;
            return state;
        }

        std::string trim_ascii_lower(std::string_view value) {
            size_t begin = 0;
            while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
                ++begin;
            }

            size_t end = value.size();
            while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
                --end;
            }

            std::string result(value.substr(begin, end - begin));
            std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return result;
        }

        bool is_valid_mode_value(int value) {
            return value >= static_cast<int>(LazyMode::Off) &&
                   value <= static_cast<int>(LazyMode::On);
        }

        LazyFallbackReason& current_fallback_reason() {
            static thread_local LazyFallbackReason reason = LazyFallbackReason::Unspecified;
            return reason;
        }

    } // namespace

    LazyFallbackReasonScope::LazyFallbackReasonScope(LazyFallbackReason reason)
        : previous_(current_fallback_reason()) {
        // Preserve an already-classified boundary reason so nested helper calls do
        // not overwrite top-level diagnostics attribution.
        if (previous_ == LazyFallbackReason::Unspecified &&
            reason != LazyFallbackReason::Unspecified) {
            current_fallback_reason() = reason;
        }
    }

    LazyFallbackReasonScope::~LazyFallbackReasonScope() {
        current_fallback_reason() = previous_;
    }

    LazyMode parse_lazy_mode_string(std::string_view value, LazyMode fallback) {
        const std::string normalized = trim_ascii_lower(value);
        if (normalized.empty()) {
            return fallback;
        }

        if (normalized == "off" || normalized == "eager" || normalized == "0" ||
            normalized == "false" || normalized == "no") {
            return LazyMode::Off;
        }
        if (normalized == "shadow" || normalized == "compare" || normalized == "1") {
            return LazyMode::Shadow;
        }
        if (normalized == "on" || normalized == "lazy" || normalized == "2" ||
            normalized == "true" || normalized == "yes") {
            return LazyMode::On;
        }

        return fallback;
    }

    const char* lazy_mode_name(LazyMode mode) {
        switch (mode) {
        case LazyMode::Off: return "off";
        case LazyMode::Shadow: return "shadow";
        case LazyMode::On: return "on";
        default: return "off";
        }
    }

    LazyMode current_lazy_mode() {
        auto& state = lazy_runtime_state();

        const int override_mode = state.override_mode.load(std::memory_order_acquire);
        if (is_valid_mode_value(override_mode)) {
            return static_cast<LazyMode>(override_mode);
        }

        if (!state.has_cached_env_mode.load(std::memory_order_acquire)) {
            const char* env_value = std::getenv("TENSOR_LAZY_MODE");
            const LazyMode parsed = env_value
                                        ? parse_lazy_mode_string(env_value, LazyMode::Off)
                                        : LazyMode::Off;
            state.cached_env_mode.store(static_cast<int>(parsed), std::memory_order_release);
            state.has_cached_env_mode.store(true, std::memory_order_release);
        }

        const int cached_mode = state.cached_env_mode.load(std::memory_order_acquire);
        if (!is_valid_mode_value(cached_mode)) {
            return LazyMode::Off;
        }
        return static_cast<LazyMode>(cached_mode);
    }

    bool lazy_mode_enabled() {
        return current_lazy_mode() == LazyMode::On;
    }

    bool lazy_mode_shadow_enabled() {
        return current_lazy_mode() == LazyMode::Shadow;
    }

    void set_lazy_mode_override_for_testing(std::optional<LazyMode> mode) {
        auto& state = lazy_runtime_state();
        if (mode.has_value()) {
            state.override_mode.store(static_cast<int>(*mode), std::memory_order_release);
            return;
        }
        state.override_mode.store(-1, std::memory_order_release);
    }

    void clear_lazy_mode_cache_for_testing() {
        auto& state = lazy_runtime_state();
        state.cached_env_mode.store(static_cast<int>(LazyMode::Off), std::memory_order_release);
        state.has_cached_env_mode.store(false, std::memory_order_release);
    }

    void reset_lazy_telemetry() {
        auto& telemetry = lazy_runtime_state().telemetry;
        telemetry.expr_nodes_created.store(0, std::memory_order_relaxed);
        telemetry.materializations.store(0, std::memory_order_relaxed);
        telemetry.eager_fallbacks.store(0, std::memory_order_relaxed);
        telemetry.eager_fallback_host_read.store(0, std::memory_order_relaxed);
        telemetry.eager_fallback_device_transfer.store(0, std::memory_order_relaxed);
        telemetry.eager_fallback_mutation.store(0, std::memory_order_relaxed);
        telemetry.eager_fallback_interop.store(0, std::memory_order_relaxed);
        telemetry.eager_fallback_other.store(0, std::memory_order_relaxed);
        telemetry.kernel_launches.store(0, std::memory_order_relaxed);
        telemetry.allocated_bytes.store(0, std::memory_order_relaxed);
    }

    LazyTelemetrySnapshot lazy_telemetry_snapshot() {
        const auto& telemetry = lazy_runtime_state().telemetry;
        return LazyTelemetrySnapshot{
            telemetry.expr_nodes_created.load(std::memory_order_relaxed),
            telemetry.materializations.load(std::memory_order_relaxed),
            telemetry.eager_fallbacks.load(std::memory_order_relaxed),
            telemetry.eager_fallback_host_read.load(std::memory_order_relaxed),
            telemetry.eager_fallback_device_transfer.load(std::memory_order_relaxed),
            telemetry.eager_fallback_mutation.load(std::memory_order_relaxed),
            telemetry.eager_fallback_interop.load(std::memory_order_relaxed),
            telemetry.eager_fallback_other.load(std::memory_order_relaxed),
            telemetry.kernel_launches.load(std::memory_order_relaxed),
            telemetry.allocated_bytes.load(std::memory_order_relaxed)};
    }

    void telemetry_record_expr_node(uint64_t count) {
        lazy_runtime_state().telemetry.expr_nodes_created.fetch_add(count, std::memory_order_relaxed);
    }

    void telemetry_record_materialization(uint64_t bytes) {
        auto& telemetry = lazy_runtime_state().telemetry;
        telemetry.materializations.fetch_add(1, std::memory_order_relaxed);
        telemetry.allocated_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }

    void telemetry_record_eager_fallback(uint64_t count) {
        auto& telemetry = lazy_runtime_state().telemetry;
        telemetry.eager_fallbacks.fetch_add(count, std::memory_order_relaxed);

        switch (current_fallback_reason()) {
        case LazyFallbackReason::HostRead:
            telemetry.eager_fallback_host_read.fetch_add(count, std::memory_order_relaxed);
            break;
        case LazyFallbackReason::DeviceTransfer:
            telemetry.eager_fallback_device_transfer.fetch_add(count, std::memory_order_relaxed);
            break;
        case LazyFallbackReason::Mutation:
            telemetry.eager_fallback_mutation.fetch_add(count, std::memory_order_relaxed);
            break;
        case LazyFallbackReason::Interop:
            telemetry.eager_fallback_interop.fetch_add(count, std::memory_order_relaxed);
            break;
        case LazyFallbackReason::Other:
        case LazyFallbackReason::Unspecified:
        default:
            telemetry.eager_fallback_other.fetch_add(count, std::memory_order_relaxed);
            break;
        }
    }

    void telemetry_record_kernel_launch(uint64_t count) {
        lazy_runtime_state().telemetry.kernel_launches.fetch_add(count, std::memory_order_relaxed);
    }

} // namespace lfs::core::internal
