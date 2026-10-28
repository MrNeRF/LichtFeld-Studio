/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/tensor.hpp"
#include <cstdint>
#include <iosfwd>
#include <span>
#include <vector>

namespace lfs::training {

    enum class POPSpaPhase : uint32_t {
        FirstScore,
        FirstPrune,
        Sparsify,
        SecondScore,
        FinalPrune,
        Recover,
        Complete,
    };

    class POPSpaController {
    public:
        struct Config {
            size_t target_count = 1;
            size_t first_prune_count = 0; // 0: max(K, round(sqrt(N*K))).
            uint32_t sparsify_steps = 5000;
            uint32_t refine_steps = 5000;
            float rho = 0.0005f;
            uint32_t projection_interval = 50;
            float erank_weight = 0.01f;
            float thin_scale_weight = 1.0f;
            float erank_epsilon = 1e-7f;
            uint64_t camera_fingerprint = 0;
        };

        // Masks are optional CUDA bool [N]; inactive rows cannot be retained and
        // every frozen row must be active. Budgets include the frozen rows.
        lfs::Status initialize(const Config&, const core::Tensor& opacities,
                               size_t training_view_count,
                               const core::Tensor& active_mask = {},
                               const core::Tensor& frozen_mask = {});
        [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }
        [[nodiscard]] POPSpaPhase phase() const noexcept { return phase_; }
        [[nodiscard]] uint32_t phase_step() const noexcept { return phase_step_; }
        [[nodiscard]] size_t score_view() const noexcept { return score_view_; }
        [[nodiscard]] size_t score_view_count() const noexcept { return view_count_; }
        [[nodiscard]] size_t state_size() const noexcept { return rows_; }
        // Original active primitive count, used to resolve automatic K1.
        [[nodiscard]] size_t input_size() const noexcept { return input_active_count_; }
        [[nodiscard]] const Config& config() const noexcept { return config_; }
        [[nodiscard]] uint64_t additional_iterations() const noexcept;
        // FP64 values in aligned UInt8 [N,8] storage (Tensor has no Float64 dtype).
        [[nodiscard]] core::Tensor& scores() noexcept { return scores_; }
        [[nodiscard]] const core::Tensor& scores() const noexcept { return scores_; }
        [[nodiscard]] const core::Tensor& proxy() const noexcept { return z_; }
        [[nodiscard]] const core::Tensor& dual() const noexcept { return u_; }
        [[nodiscard]] bool projection_due() const noexcept;
        lfs::Status finish_score_view();
        // Exact Int64 CUDA indices, ascending row order; tied scores prefer lower IDs.
        lfs::Result<core::Tensor> prune_indices() const;
        // Call only after the trainer has compacted all Gaussian row state.
        lfs::Status accept_prune(const core::Tensor& opacities,
                                 const core::Tensor& active_mask = {},
                                 const core::Tensor& frozen_mask = {});
        // Adds manual gradients to existing photometric gradients and returns a GPU
        // scalar. Invoke after raster backward and before after_backward()/Adam.
        lfs::Result<core::Tensor> regularization(const core::Tensor& opacities,
                                                 const core::Tensor& log_scales,
                                                 core::Tensor& opacity_grad,
                                                 core::Tensor& scales_grad);
        lfs::Status after_backward(const core::Tensor& opacities);
        // Call after Adam; zero-length phases are skipped by accept_prune().
        lfs::Status finish_optimization_step();
        void reset() noexcept;

        lfs::Status serialize(std::ostream&) const;
        lfs::Status deserialize(std::istream&);
        lfs::Result<size_t> checkpoint_size_bytes(size_t expected_rows) const;
        // Validates and consumes a component; returns its Gaussian row count.
        static lfs::Result<size_t> consume_checkpoint(std::istream&);
        void adopt_checkpoint_state(POPSpaController& loaded) noexcept;

        // CPU oracle used for budget validation and exact-index unit tests.
        static lfs::Result<std::vector<int64_t>> select_keep_indices(
            std::span<const double> scores, size_t count,
            std::span<const uint8_t> active = {}, std::span<const uint8_t> frozen = {});

    private:
        static POPSpaController read_serialized(std::istream&);
        void validate_state(size_t expected_rows, bool inspect_contents = false) const;
        void set_masks(const core::Tensor&, const core::Tensor&);
        void project(const core::Tensor&, bool update_dual);
        Config config_;
        bool initialized_ = false;
        POPSpaPhase phase_ = POPSpaPhase::FirstScore;
        uint32_t phase_step_ = 0;
        size_t score_view_ = 0;
        size_t view_count_ = 0;
        size_t rows_ = 0;
        size_t input_rows_ = 0;
        size_t active_count_ = 0;
        size_t frozen_count_ = 0;
        size_t input_active_count_ = 0;
        core::Tensor scores_;
        core::Tensor z_;
        core::Tensor u_;
        core::Tensor active_;
        core::Tensor frozen_;
        // Derived projection workspace, intentionally omitted from checkpoints.
        core::Tensor projection_values_;
        core::Tensor projection_indices_;
        core::Tensor projection_keep_;
    };
} // namespace lfs::training
