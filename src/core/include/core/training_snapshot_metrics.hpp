/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/uuid.hpp"
#include <cstddef>
#include <cstdint>
namespace lfs::training {
    struct TrainingSnapshotPauseMetrics {
        lfs::core::Uuid snapshot_uuid;
        int iteration = 0;
        std::uint64_t checkpoint_bytes = 0;
        std::uint64_t device_snapshot_bytes = 0;
        std::uint64_t pinned_peak_bytes = 0;
        std::uint64_t host_staging_bytes = 0;
        std::uint64_t host_rss_delta_bytes = 0;
        std::uint64_t host_memory_available_bytes = 0;
        std::uint64_t host_memory_required_bytes = 0;
        std::size_t tensor_piece_count = 0;
        std::size_t cpu_piece_count = 0;
        double service_initialization_ms = 0.0;
        double prepare_stall_ms = 0.0;
        // Compatibility name retained for the P4 MCP surface. It is exactly
        // the measured on-training-thread prepare stall.
        double preparation_ms = 0.0;
        double safe_point_entry_ms = 0.0;
        double stream_sync_ms = 0.0;
        double additional_cpu_state_ms = 0.0;
        double scng_ms = 0.0;
        double selm_ms = 0.0;
        double prms_ms = 0.0;
        double serialize_and_issue_ms = 0.0;
        double last_d2h_wait_ms = 0.0;
        double pause_ms = 0.0;
        double cold_path_ms = 0.0;
        double final_drain_ms = 0.0;
        double measured_pinned_d2h_bytes_per_second = 0.0;
        double rig_gate_ms = 0.0;
        bool cold_first_snapshot = false;
        bool pause_within_rig_gate = false;
        bool cold_path_within_rig_gate = false;
        bool host_memory_preflight_passed = false;
        bool host_ram_within_gate = false;
        bool consistency_proven = false;
    };

    struct TrainingSnapshotServiceMetrics {
        std::uint64_t completed_snapshots = 0;
        double pause_p95_ms = 0.0;
        std::size_t p95_n = 0;
        TrainingSnapshotPauseMetrics last;
    };

} // namespace lfs::training
