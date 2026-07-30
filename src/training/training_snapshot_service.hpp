/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/parameters.hpp"
#include "core/uuid.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace lfs::training {

    class IStrategy;
    class BilateralGrid;
    class PPISP;
    class PPISPControllerPool;
    class ADMMSparsityOptimizer;

    struct TrainingSnapshotServiceConfig {
        std::size_t ring_slots = 4;
        std::size_t band_bytes = 128ull * 1024 * 1024;
        std::size_t calibration_bytes = 32ull * 1024 * 1024;
        int calibration_iterations = 4;
    };

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
        double preparation_ms = 0.0;
        double safe_point_entry_ms = 0.0;
        double stream_sync_ms = 0.0;
        double additional_cpu_state_ms = 0.0;
        double serialize_and_issue_ms = 0.0;
        double last_d2h_wait_ms = 0.0;
        double pause_ms = 0.0;
        double final_drain_ms = 0.0;
        double measured_pinned_d2h_bytes_per_second = 0.0;
        double rig_gate_ms = 0.0;
        bool cold_first_snapshot = false;
        bool pause_within_rig_gate = false;
        bool host_memory_preflight_passed = false;
        bool host_ram_within_gate = false;
        bool consistency_proven = false;
    };

    struct TrainingSnapshotServiceMetrics {
        std::uint64_t completed_snapshots = 0;
        double pause_p95_ms = 0.0;
        TrainingSnapshotPauseMetrics last;
    };

    struct TrainingSnapshotCaptureRequest {
        int iteration;
        // Optional externally assigned identity. CPU chapters may be
        // pre-staged with this UUID while the optimizer is still running.
        // A nil UUID asks the service to generate one during prepare().
        lfs::core::Uuid snapshot_uuid;
        // Optional origin for the one optimizer-pause clock. The caller sets
        // this immediately before draining model readers/locks so those waits
        // are included with the service-side stream synchronizations.
        std::optional<std::chrono::steady_clock::time_point>
            safe_point_entered_at;
        const IStrategy& strategy;
        const lfs::core::param::TrainingParameters& params;
        const BilateralGrid* bilateral_grid = nullptr;
        const PPISP* ppisp = nullptr;
        const PPISPControllerPool* ppisp_controller_pool = nullptr;
        const ADMMSparsityOptimizer* sparsity_optimizer = nullptr;
        std::span<const cudaStream_t> mutating_streams;
        // Runs inside the measured safe-point clock after all mutation streams
        // are quiescent. The callback must produce owned CPU chapter state and
        // stamp it with the supplied UUID.
        std::function<lfs::Result<void>(
            const lfs::core::Uuid&)>
            capture_additional_cpu_state;
    };

    struct CapturedTrainingSnapshot {
        lfs::core::Uuid snapshot_uuid;
        int iteration = 0;
        std::shared_ptr<const std::vector<std::byte>> checkpoint_bytes;
        TrainingSnapshotPauseMetrics metrics;
    };

    class PreparedTrainingSnapshot {
    public:
        struct Impl;

        PreparedTrainingSnapshot(PreparedTrainingSnapshot&&) noexcept;
        PreparedTrainingSnapshot&
        operator=(PreparedTrainingSnapshot&&) noexcept;
        PreparedTrainingSnapshot(
            const PreparedTrainingSnapshot&) = delete;
        PreparedTrainingSnapshot&
        operator=(const PreparedTrainingSnapshot&) = delete;
        ~PreparedTrainingSnapshot();

        [[nodiscard]] const lfs::core::Uuid&
        snapshot_uuid() const noexcept;
        [[nodiscard]] std::uint64_t
        checkpoint_bytes() const noexcept;

    private:
        friend class TrainingSnapshotService;
        explicit PreparedTrainingSnapshot(
            std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
    };

    class PendingTrainingSnapshot {
    public:
        struct Impl;

        PendingTrainingSnapshot(PendingTrainingSnapshot&&) noexcept;
        PendingTrainingSnapshot&
        operator=(PendingTrainingSnapshot&&) noexcept;
        PendingTrainingSnapshot(
            const PendingTrainingSnapshot&) = delete;
        PendingTrainingSnapshot&
        operator=(const PendingTrainingSnapshot&) = delete;
        ~PendingTrainingSnapshot();

        [[nodiscard]] bool ready() const;
        [[nodiscard]] TrainingSnapshotPauseMetrics
        pause_metrics() const;
        [[nodiscard]] lfs::Result<CapturedTrainingSnapshot>
        wait();

    private:
        friend class TrainingSnapshotService;
        explicit PendingTrainingSnapshot(
            std::shared_ptr<Impl> impl);
        std::shared_ptr<Impl> impl_;
    };

    class TrainingSnapshotService {
    public:
        struct Impl;

        explicit TrainingSnapshotService(
            TrainingSnapshotServiceConfig config = {});
        TrainingSnapshotService(
            const TrainingSnapshotService&) = delete;
        TrainingSnapshotService&
        operator=(const TrainingSnapshotService&) = delete;
        ~TrainingSnapshotService();

        // Preparation runs before the measured optimizer pause: it records the
        // exact LFKP layout, calibrates the rig once, and pre-faults immutable
        // pageable staging plus the bounded pinned ring.
        [[nodiscard]] lfs::Result<PreparedTrainingSnapshot>
        prepare(const TrainingSnapshotCaptureRequest& request);

        // One clock: immediately before all mutating-stream synchronizations
        // through last D2H completion, ending just before the caller may let
        // the optimizer mutate again. Final pinned-to-pageable drain continues
        // behind the returned pending handle.
        [[nodiscard]] lfs::Result<PendingTrainingSnapshot>
        capture(
            PreparedTrainingSnapshot prepared,
            const TrainingSnapshotCaptureRequest& request);

        [[nodiscard]] TrainingSnapshotServiceMetrics metrics() const;

    private:
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::training
