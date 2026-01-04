/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "istrategy.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include <memory>

namespace lfs::training {
    // Forward declarations
    struct RenderOutput;

    /// ADC densification-based optimization strategy. SplatData owned by Scene.
    class ADC : public IStrategy {
    public:
        ADC() = delete;
        /// SplatData must be owned by Scene
        explicit ADC(lfs::core::SplatData& splat_data);

        // Prevent copy/move
        ADC(const ADC&) = delete;
        ADC& operator=(const ADC&) = delete;
        ADC(ADC&&) = delete;
        ADC& operator=(ADC&&) = delete;

        // IStrategy interface implementation
        void initialize(const lfs::core::param::OptimizationParameters& optimParams) override;

        void post_backward(int iter, RenderOutput& render_output) override;

        void step(int iter) override;

        bool is_refining(int iter) const override;

        lfs::core::SplatData& get_model() override { return *_splat_data; }
        const lfs::core::SplatData& get_model() const override { return *_splat_data; }

        void remove_gaussians(const lfs::core::Tensor& mask) override;

        // IStrategy interface - optimizer access
        AdamOptimizer& get_optimizer() override { return *_optimizer; }
        const AdamOptimizer& get_optimizer() const override { return *_optimizer; }
        ExponentialLR* get_scheduler() { return _scheduler.get(); }
        const ExponentialLR* get_scheduler() const { return _scheduler.get(); }

        // Serialization for checkpoints
        void serialize(std::ostream& os) const override;
        void deserialize(std::istream& is) override;
        const char* strategy_type() const override { return "adc"; }

        // Reserve optimizer capacity for future growth (e.g., after checkpoint load)
        void reserve_optimizer_capacity(size_t capacity) override;

        // Get count of active (non-free) Gaussians
        size_t active_count() const;

        // Get count of free slots available for reuse
        size_t free_count() const;

        // Get indices of active (non-free) Gaussians for export
        lfs::core::Tensor get_active_indices() const;

    private:
        // Helper functions
        void duplicate(const lfs::core::Tensor& is_duplicated);

        void split(const lfs::core::Tensor& is_split);

        void grow_gs(int iter);

        void remove(const lfs::core::Tensor& is_prune);

        void prune_gs(int iter);

        void reset_opacity();

        // Reuse free slots for new Gaussians, returns indices that were filled
        // and the count of remaining Gaussians that need to be appended
        std::pair<lfs::core::Tensor, int64_t> fill_free_slots(
            const lfs::core::Tensor& source_indices, int64_t count);

        // Fill free slots with provided data (for split operation)
        // Returns indices that were filled and count of remaining data
        std::pair<lfs::core::Tensor, int64_t> fill_free_slots_with_data(
            const lfs::core::Tensor& positions,
            const lfs::core::Tensor& rotations,
            const lfs::core::Tensor& scales,
            const lfs::core::Tensor& sh0,
            const lfs::core::Tensor& shN,
            const lfs::core::Tensor& opacities,
            int64_t count);

        // Mark slots as free (for soft deletion)
        void mark_as_free(const lfs::core::Tensor& indices);

        // Member variables
        std::unique_ptr<AdamOptimizer> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;
        lfs::core::SplatData* _splat_data = nullptr; // Scene-owned
        std::unique_ptr<const lfs::core::param::OptimizationParameters> _params;

        // Free slot tracking - bool tensor [capacity], true = slot is free for reuse
        lfs::core::Tensor _free_mask;
    };
} // namespace lfs::training
