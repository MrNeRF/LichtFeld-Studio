/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "postprocess_compaction.hpp"

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "istrategy.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "strategy_factory.hpp"
#include "strategy_utils.hpp"

#include <algorithm>
#include <format>
#include <limits>
#include <type_traits>

namespace lfs::training {

    lfs::Status compact_gaussians_for_postprocess(
        IStrategy& strategy,
        const core::param::OptimizationParameters& params,
        const core::Tensor& keep_indices,
        const std::function<lfs::Status(core::SplatData&)>& prepare_state) {
        try {
            auto& model = strategy.get_model();
            const size_t old_size = model.size();
            LFS_ASSERT_MSG(keep_indices.is_valid() && keep_indices.ndim() == 1 &&
                               keep_indices.dtype() == core::DataType::Int64 &&
                               keep_indices.device() == core::Device::CUDA &&
                               keep_indices.numel() > 0 && keep_indices.numel() <= old_size,
                           std::format("postprocess retained indices must be CUDA Int64 [K], 0 < K <= N (K={}, N={})",
                                       keep_indices.numel(), old_size));
            const auto host_indices = keep_indices.cpu().contiguous();
            const auto* kept = host_indices.ptr<int64_t>();
            const size_t count = keep_indices.numel();
            for (size_t i = 0; i < count; ++i) {
                LFS_ASSERT_MSG(kept[i] >= 0 && static_cast<size_t>(kept[i]) < old_size &&
                                   (i == 0 || kept[i - 1] < kept[i]),
                               std::format("retained indices must be unique, ascending and in [0,N) (position={}, index={}, N={})",
                                           i, kept[i], old_size));
            }
            for (const auto& range : model.frozen_ranges()) {
                const size_t end = range.start + std::min(range.count, old_size - std::min(range.start, old_size));
                for (size_t row = range.start; row < end; ++row) {
                    LFS_ASSERT_MSG(std::binary_search(kept, kept + count, static_cast<int64_t>(row)),
                                   std::format("postprocess compaction must retain frozen rows (missing row={}, K={})", row, count));
                }
            }
            if (model.has_deleted_mask()) {
                LFS_ASSERT_MSG(model.deleted_mask_matches_size(),
                               std::format("postprocess deleted mask must match model rows (mask={}, N={})", model.deleted().numel(), old_size));
                const auto retained_deleted = model.deleted().index_select(0, keep_indices).cpu();
                for (size_t i = 0; i < count; ++i) {
                    LFS_ASSERT_MSG(!retained_deleted.ptr<bool>()[i],
                                   std::format("postprocess compaction cannot retain deleted rows (row={})", kept[i]));
                }
            }
            auto* adopter = dynamic_cast<ICheckpointStateAdopter*>(&strategy);
            LFS_ASSERT_MSG(adopter && adopter->has_checkpoint_runtime_state(),
                           std::format("postprocess compaction requires an initialized transactional strategy (strategy={})", strategy.strategy_type()));

            auto sh_rest = core::Tensor::empty(
                {count, model.max_sh_coeffs_rest(), 3}, model.means().device());
            if (model.max_sh_coeffs_rest() > 0) {
                sh_value::gather_shN_to_canonical(model, keep_indices, sh_rest);
            }
            core::SplatData staged_model(
                model.get_max_sh_degree(),
                model.means().index_select(0, keep_indices),
                model.sh0().index_select(0, keep_indices),
                std::move(sh_rest),
                model.scaling_raw().index_select(0, keep_indices),
                model.rotation_raw().index_select(0, keep_indices),
                model.opacity_raw().index_select(0, keep_indices),
                model.get_scene_scale());
            staged_model.set_active_sh_degree(model.get_active_sh_degree());
            staged_model.set_frozen_ranges(model.frozen_ranges());
            remap_frozen_ranges_after_compaction(staged_model, keep_indices, old_size);

            auto staged_result = StrategyFactory::instance().create(strategy.strategy_type(), staged_model);
            if (!staged_result) {
                return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::FailedPrecondition,
                    .domain = lfs::ErrorDomain::Training,
                    .user_message = "Cannot create the strategy for POPSpa compaction.",
                    .detail = staged_result.error(),
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            }
            auto staged_strategy = std::move(*staged_result);
            auto compact_params = params;
            compact_params.max_cap = static_cast<int>(count);
            staged_strategy->initialize(compact_params);
            staged_strategy->set_optimization_params(params);
            apply_frozen_ranges_to_optimizer(staged_model, staged_strategy->get_optimizer(), 0.0f);
            LFS_ASSERT_MSG(adopter->can_adopt_checkpoint_state(*staged_strategy),
                           std::format("strategy must accept staged compaction state (strategy={}, K={})", strategy.strategy_type(), count));
            if (prepare_state) {
                if (auto prepared = prepare_state(staged_model); !prepared)
                    return prepared;
            }
            // Surface asynchronous gather failures while the old graph is intact.
            for (const auto* tensor : {&staged_model.means(), &staged_model.sh0(),
                                       &staged_model.shN_raw(), &staged_model.scaling_raw(),
                                       &staged_model.rotation_raw(), &staged_model.opacity_raw()}) {
                if (tensor->is_valid())
                    LFS_CUDA_CHECK(cudaStreamSynchronize(tensor->stream()));
            }
            static_assert(std::is_nothrow_move_assignable_v<core::SplatData>);
            // Strategy adoption only transfers serialized Adam state. Install
            // the already-prepared runtime mask before the no-throw commit.
            strategy.get_optimizer().set_frozen_mask(staged_strategy->get_optimizer().frozen_mask());
            staged_model.set_capacity_ensure(model.release_capacity_ensure());
            model = std::move(staged_model);
            adopter->adopt_checkpoint_state(*staged_strategy);
            return {};
        } catch (const lfs::Exception& e) {
            return lfs::Status::failure(e.error());
        } catch (const std::exception& e) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::Training,
                .user_message = "POPSpa compaction failed.",
                .detail = e.what(),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
    }

} // namespace lfs::training
