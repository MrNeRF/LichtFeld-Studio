/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once
#include "core/training_snapshot_metrics.hpp"

#include "core/error.hpp"
#include "core/parameters.hpp"
#include "core/uuid.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace lfs::training {

    class IStrategy;
    class BilateralGrid;
    class PPISP;
    class PPISPControllerPool;
    class ADMMSparsityOptimizer;

    struct TrainingSnapshotServiceConfig {
        std::size_t ring_slots = 4;
        std::size_t band_bytes = 64ull * 1024 * 1024;
        std::size_t calibration_bytes = 32ull * 1024 * 1024;
        int calibration_iterations = 4;
    };

    struct TrainingSnapshotCpuStateMetrics {
        double scng_ms = 0.0;
        double selm_ms = 0.0;
        double prms_ms = 0.0;
    };

    struct TrainingStepWindowMetrics {
        int first_iteration = 0;
        int last_iteration = 0;
        std::size_t sample_count = 0;
        double mean_ms = 0.0;
    };

    struct TrainingStepRegressionMetrics {
        TrainingStepWindowMetrics pre_snapshot;
        TrainingStepWindowMetrics post_resume;
        double regression_percent = 0.0;
        bool gate_evaluated = false;
        bool within_gate = false;
    };

    // Selects two contiguous steady-state windows. A topology-changing
    // refinement/densification iteration clears the candidate run, so no
    // reported window can contain such an event.
    class TrainingStepRegressionTracker {
    public:
        explicit TrainingStepRegressionTracker(
            std::size_t window_size = 100);

        void observe(
            int iteration,
            double elapsed_ms,
            bool topology_changed);
        void arm_after_snapshot(int snapshot_iteration);
        [[nodiscard]] TrainingStepRegressionMetrics
        metrics() const noexcept;
        void reset() noexcept;

    private:
        struct Sample {
            int iteration = 0;
            double elapsed_ms = 0.0;
        };

        [[nodiscard]] TrainingStepWindowMetrics
        summarize(const std::deque<Sample>& samples) const noexcept;

        std::size_t window_size_ = 100;
        std::deque<Sample> steady_run_;
        std::optional<TrainingStepWindowMetrics>
            latest_steady_window_;
        std::deque<Sample> post_resume_run_;
        TrainingStepRegressionMetrics metrics_;
        int snapshot_iteration_ = 0;
        bool armed_ = false;
    };

    struct TrainingSnapshotCaptureRequest {
        int iteration;
        // Optional externally assigned identity. The bundle may reserve this
        // UUID inter-step on the training thread; SCNG/SELM/PRMS themselves
        // are captured later inside the measured safe-point window. That
        // inter-step work can overlap rendering, never optimizer mutation.
        // A nil UUID asks the service to generate one during prepare().
        lfs::core::Uuid snapshot_uuid;
        // Explicit saves may use the smaller headroom-only host-memory
        // reserve; autosave captures retain the full safety reserve.
        bool relaxed_host_memory_gate = false;
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
        std::span<void* const> mutating_streams;
        // Runs inside the measured safe-point clock after all mutation streams
        // are quiescent. The callback may only copy detached value state and
        // must stamp it with the supplied UUID. JSON/DOM/chapter assembly runs
        // after capture() returns and the optimizer may mutate again.
        std::function<lfs::Result<
            TrainingSnapshotCpuStateMetrics>(
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

        // Trainer-start initialization owns the cold CUDA work: D2H stream,
        // full-mutating-stream calibration, and bounded pinned ring. It must
        // complete before prepare() is allowed onto a save path.
        [[nodiscard]] lfs::Result<void>
        initialize(
            const TrainingSnapshotCaptureRequest& request);

        // Preparation runs inter-step on the training thread (concurrent with
        // rendering only): it records the exact LFKP layout and pre-faults
        // immutable pageable staging. The residual stall is measured and
        // bounded independently from the safe-point pause.
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

        static void reset_process_pinned_d2h_calibration_for_testing();
        void testing_advance_completed_snapshots(
            std::uint64_t count = 1);

    private:
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::training
