/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "blob_seeding.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "istrategy.hpp"
#include "kernels/mrnf_kernels.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include "strategy_utils.hpp"
#include <cassert>
#include <future>
#include <limits>
#include <memory>
#include <thread>

namespace lfs::training::sh_value {
    class ShNMutationBatch;
}

class MRNFStrategyTest_MortonReorderKeepsFarMaskPublished_Test;
class MRNFStrategyTest_RefineOutOfVramIsUndoneToTheRowsItStartedFrom_Test;
class MRNFStrategyTest_GrowthExchangeUsesMaxOfImpactRanks_Test;
class MRNFStrategyTest_GrowthExchangeSkipsNewbornsUntilScored_Test;
class MRNFStrategyTest_GrowthExchangeMarksSplitParentsAndChildrenNewborn_Test;
class MRNFStrategyTest_RemoveGaussiansKeepsGrowthExchangeStateAligned_Test;
class MRNFStrategyTest_GrowthExchangeFallsBackToCoverageWithoutImpact_Test;
class MRNFStrategyTest_GrowthExchangeMatchesDemandWithoutDeletingParents_Test;
class MRNFStrategyTest_GrowthExchangeCosineQuotaAtStartMiddleAndEnd_Test;
class MRNFStrategyTest_LateLrAnnealDecaysOpacityAndColorAfterGrowth_Test;
class MRNFStrategyTest_PerSplatBiasCountersRestartWhereMomentsRestart_Test;
class MRNFStrategyTest_EdgeGuidanceFactorPrefersHigherPrecomputedEdgeScores_Test;
class MRNFStrategyTest_GrowAndSplitResetsOptimizerStateForParents_Test;
class MRNFStrategyTest_SHDegree0KeepsShNEmptyAndFusedAdamUsableAfterGrowth_Test;
class MRNFStrategyTest_GrowAndSplitUsesIgsPlusSplitRule_Test;
class MRNFStrategyTest_GrowAndSplitWithoutMaxCapExtendsBookkeepingMasks_Test;
class MRNFStrategyTest_DeletedMaskCapacityGrowthPreservesExistingRows_Test;
class MRNFStrategyTest_GrowAndSplitReplacementSkipsZeroWeightCandidates_Test;
class MRNFStrategyTest_GrowAndSplitReusesFreeSlotsBeforeAppending_Test;
class MRNFStrategyTest_ChunkedChildPlacementMatchesSingleChunk_Test;
class MRNFStrategyTest_SerializeRoundTripPreservesFreeMask_Test;
class MRNFStrategyTest_SerializeRoundTripPreservesLrScheduleState_Test;
class MRNFStrategyTest_DeserializeResizesTransientBuffersToLoadedModel_Test;
class MRNFStrategyTest_SetOptimizationParamsRecomputesDecayFromCurrentState_Test;
class MRNFStrategyTest_DegenerateBoundsStayInvalidAndKeepFiniteMeanLearningRate_Test;
class MRNFStrategyTest_LineBoundsUseFiniteSceneScaleForMeanLearningRate_Test;
class CropDampingStrategyTest_MrnfRejectedRowsAreNotRefineCandidatesAtZeroScale_Test;
class CropDampingStrategyTest_RoiGrowthShareSplitsPicksAndSpillsLeftovers_Test;
class CropDampingStrategyTest_RoiGrowthSplitCountsOnlySelectableWeights_Test;
class MRNFStrategyTest_CompactSplatsCorrectAndPeakBelowThreeX_Test;
class MRNFStrategyTest_CompactSplatsFusedPathLeavesGradsEmpty_Test;
class MRNFStrategyTest_ApplyDecaySkipsFrozenRows_Test;
class MRNFStrategyTest_DensificationInfoShapeIsTwoRows_Test;
class MRNFStrategyTest_ZeroVisibilityProducesNoGrowth_Test;
class MRNFStrategyTest_DirectAuxiliaryGrowthPreservesPrefix_Test;
class MRNFStrategyTest_PerSplatMeanStepScalesWithExtentAndClamps_Test;
class MRNFStrategyTest_EdgeWindowNormalizesViewsAndClosesBeforeRefineBackward_Test;
class MRNFStrategyTest_RayDepthStateFollowsItsSplats_Test;
class MRNFStrategyTest_NeedGrowthOffLeavesErrorAndRefillWeightsUnchanged_Test;
class MRNFStrategyTest_NeedGrowthRefillsFromHighErrorGroup_Test;
class MRNFStrategyTest_NeedGrowthPrefersLargeSplatsAtEqualError_Test;
class MRNFStrategyTest_NeedGrowthDonorsAreLowestSummedContribution_Test;
class MRNFStrategyTest_NeedGrowthProtectsNewbornsUntilAFullPass_Test;
class MRNFStrategyTest_NeedGrowthDonorShareTapersToZero_Test;
class MRNFStrategyTest_NeedGrowthFreedSlotsGoToTheNeedDraw_Test;
class MRNFStrategyTest_NeedGrowthResumeRebuildsDonorMemory_Test;
class MRNFStrategyTest_NeedGrowthReceiverSplitsOnlyAtTheCap_Test;
class MRNFStrategyTest_RefineErrorViewMeanOffKeepsTheMaxOverViews_Test;
class MRNFStrategyTest_RefineErrorViewMeanAveragesVisibleViewsAndResetsAtRefine_Test;
class MRNFStrategyTest_RefineErrorViewMeanRestartsTheCountOnResume_Test;
class MRNFStrategyTest_RefineErrorCompressFoldsThePowerOfEachViewOverViewsWithError_Test;
class MRNFStrategyTest_RefineErrorCompressCandidatesAreEveryScoredSplat_Test;
class MRNFStrategyTest_RefineErrorCompressNeedHasNoSparsityFactor_Test;
class MRNFStrategyTest_RefineErrorCompressRanksSmallIntenseOverLargeFaint_Test;
class MRNFStrategyTest_RefineErrorCompressSwitchStartsAFreshWindow_Test;
class MRNFStrategyTest_RefineNeedPowerRaisesTheFoldedNeed_Test;
class MRNFStrategyTest_RefineNeedPowerSharpensTheReceiverDraws_Test;
class MRNFStrategyTest_DonorRankPerViewCountsCoveringViewsInEveryFold_Test;
class MRNFStrategyTest_DonorRankPerViewFadesTheCountWithTheSum_Test;
class MRNFStrategyTest_DonorRankPerViewOffKeepsTheSummedRanking_Test;
class MRNFStrategyTest_DonorRankPerViewRanksBySumPerViewWithAFloor_Test;
class MRNFStrategyTest_DonorRankPerViewCountFollowsItsSplats_Test;
class MRNFStrategyTest_DonorRankPerViewResumeRebuildsTheCount_Test;
class MRNFStrategyTest_DonorRankImpactOffKeepsTheBlendWeightDonors_Test;
class MRNFStrategyTest_DonorRankImpactBackwardWritesImpactUnderNeedGrowth_Test;
class MRNFStrategyTest_DonorRankImpactFadesTheMaximaLikeThePlainExchange_Test;
class MRNFStrategyTest_DonorRankImpactKeepsASmallHighPeakSplat_Test;
class MRNFStrategyTest_DonorRankImpactStateFollowsItsSplats_Test;
class MRNFStrategyTest_DonorRankImpactResumeAndSwitchStartAfresh_Test;
class MRNFStrategyTest_ExploreOffKeepsTheNoiseAndTheChildren_Test;
class MRNFStrategyTest_ExploreWalkCovarianceFollowsTheSplatShape_Test;
class MRNFStrategyTest_ExploreWalkAmplitudeFollowsTheRunAndOnlyRenderedSplatsWalk_Test;
class MRNFStrategyTest_ExploreGatePeakWalksHiddenOpaqueSplatsOnly_Test;
class MRNFStrategyTest_ExploreBackwardWritesVisibilityAndPeakBlendWeight_Test;
class MRNFStrategyTest_ExploreScoutsShareOpacityTaperAndSeed_Test;
class MRNFStrategyTest_ExploreScoutsSkipOpacityDecayUntilRendered_Test;
class MRNFStrategyTest_ExploreRowsFollowTheirSplats_Test;
class MRNFStrategyTest_ReceiverSplitTakesTheOpacityAndCrossScale_Test;
class MRNFStrategyTest_RetireOffKeepsTheRefineBitIdentical_Test;
class MRNFStrategyTest_RetireQuotaStartsAtTheCapAndTapersToGrowthEnd_Test;
class MRNFStrategyTest_RetireTakesTheLowestPeakPastTheGraceAndRefillsTheSlots_Test;
class MRNFStrategyTest_RetireRowsTrackPeakAndRendersAndFollowTheirSplats_Test;
class MRNFStrategyTest_LossErrorMapImpliesTheCompressedViewMean_Test;
class MRNFStrategyTest_LossErrorMapScoreEqualsTheMatchScore_Test;
class MRNFStrategyTest_RetireSpareOffKeepsTheRetireesBitIdentical_Test;
class MRNFStrategyTest_RetireSpareExemptsThinSplatsAndKeepsTheQuota_Test;
class MRNFStrategyTest_RetireSpareWindowFollowsRendersAndSplats_Test;
class MRNFStrategyTest_RetireSpareBackwardWritesTheRawMinorSigma_Test;
class MRNFStrategyTest_RetireRankBothOffKeepsTheRetireesBitIdentical_Test;
class MRNFStrategyTest_RetireRankBothKeepsHighImpactOrHighPeakSplats_Test;
class MRNFStrategyTest_RetireRankBothFoldsTheWindowImpactApartFromTheDonors_Test;
class MRNFStrategyTest_ImpactContrastNormDividesByThePhotoDeviation_Test;
class MRNFStrategyTest_ReceiverKeepLongSplitsShortReceiversAcross_Test;
class MRNFStrategyTest_ReceiverKeepLongWindowFollowsRendersAndSplats_Test;
class MRNFStrategyTest_KeepCoveredOffKeepsTheRowsAndTheRetireesBitIdentical_Test;
class MRNFStrategyTest_KeepCoveredDecayScalesByTheReach_Test;
class MRNFStrategyTest_KeepCoveredTakesCoveredPicksOutOfTheRetireQuota_Test;

namespace lfs::training {

    inline constexpr float kBlobSeedOpacity = 0.5f;
    inline constexpr double kBlobSeedCapacityFraction = 0.01;
    inline constexpr float kFarMaskOrbits = 2.0f;

    class MRNF : public IStrategy, public ICheckpointStateAdopter {
    public:
        MRNF() = delete;
        explicit MRNF(lfs::core::SplatData& splat_data);

        MRNF(const MRNF&) = delete;
        MRNF& operator=(const MRNF&) = delete;
        MRNF(MRNF&&) = delete;
        MRNF& operator=(MRNF&&) = delete;

        void initialize(const lfs::core::param::OptimizationParameters& optimParams) override;
        void pre_step(int iter, RenderOutput& render_output) override;
        void post_render(int iter, RenderOutput& render_output) override;
        void post_backward(int iter, RenderOutput& render_output) override;
        bool is_refining(int iter) const override;
        void step(int iter) override;
        [[nodiscard]] lfs::core::Tensor rendered_support_counts() const override { return _rendered_count; }

        void for_each_gaussian_row(const std::function<void(lfs::core::Tensor&)>& visit) override;

        lfs::core::SplatData& get_model() override { return *_splat_data; }
        const lfs::core::SplatData& get_model() const override { return *_splat_data; }

        void remove_gaussians(const lfs::core::Tensor& mask) override;

        AdamOptimizer& get_optimizer() override {
            assert(_optimizer);
            return *_optimizer;
        }
        const AdamOptimizer& get_optimizer() const override {
            assert(_optimizer);
            return *_optimizer;
        }

        void serialize(std::ostream& os) const override;
        void deserialize(std::istream& is) override;
        bool has_checkpoint_runtime_state() const noexcept override { return static_cast<bool>(_optimizer); }
        bool can_adopt_checkpoint_state(const IStrategy& loaded) const noexcept override;
        void adopt_checkpoint_state(IStrategy& loaded) noexcept override;
        const char* strategy_type() const override { return "mrnf"; }

        void reserve_optimizer_capacity(size_t capacity) override;
        void set_optimization_params(const lfs::core::param::OptimizationParameters& params) override;
        void restart_mean_lr() override;
        void set_training_dataset(std::shared_ptr<CameraDataset> views) override;
        std::shared_ptr<CameraDataset> get_training_dataset() const override { return _views; }
        lfs::core::Tensor edge_score_scratch(int iter) override;
        void on_edge_score_accumulated(int iter) override;
        std::pair<lfs::core::Tensor, lfs::core::Tensor> impact_scratch(int iter) override;
        void on_impact_accumulated(int iter) override;
        std::pair<lfs::core::Tensor, lfs::core::Tensor> explore_scratch(int iter) override;
        lfs::core::Tensor minor_sigma_scratch(int iter) override;
        lfs::core::Tensor major_sigma_scratch(int iter) override;

    private:
        friend class ::MRNFStrategyTest_MortonReorderKeepsFarMaskPublished_Test;
        friend class ::MRNFStrategyTest_RefineOutOfVramIsUndoneToTheRowsItStartedFrom_Test;
        friend class ::MRNFStrategyTest_EdgeWindowNormalizesViewsAndClosesBeforeRefineBackward_Test;
        friend class ::MRNFStrategyTest_EdgeGuidanceFactorPrefersHigherPrecomputedEdgeScores_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitResetsOptimizerStateForParents_Test;
        friend class ::MRNFStrategyTest_SHDegree0KeepsShNEmptyAndFusedAdamUsableAfterGrowth_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitUsesIgsPlusSplitRule_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitWithoutMaxCapExtendsBookkeepingMasks_Test;
        friend class ::MRNFStrategyTest_DeletedMaskCapacityGrowthPreservesExistingRows_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitReplacementSkipsZeroWeightCandidates_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitReusesFreeSlotsBeforeAppending_Test;
        friend class ::MRNFStrategyTest_ChunkedChildPlacementMatchesSingleChunk_Test;
        friend class ::MRNFStrategyTest_SerializeRoundTripPreservesFreeMask_Test;
        friend class ::MRNFStrategyTest_SerializeRoundTripPreservesLrScheduleState_Test;
        friend class ::MRNFStrategyTest_DeserializeResizesTransientBuffersToLoadedModel_Test;
        friend class ::MRNFStrategyTest_SetOptimizationParamsRecomputesDecayFromCurrentState_Test;
        friend class ::MRNFStrategyTest_DegenerateBoundsStayInvalidAndKeepFiniteMeanLearningRate_Test;
        friend class ::MRNFStrategyTest_LineBoundsUseFiniteSceneScaleForMeanLearningRate_Test;
        friend class ::CropDampingStrategyTest_MrnfRejectedRowsAreNotRefineCandidatesAtZeroScale_Test;
        friend class ::CropDampingStrategyTest_RoiGrowthShareSplitsPicksAndSpillsLeftovers_Test;
        friend class ::CropDampingStrategyTest_RoiGrowthSplitCountsOnlySelectableWeights_Test;
        friend class ::MRNFStrategyTest_CompactSplatsCorrectAndPeakBelowThreeX_Test;
        friend class ::MRNFStrategyTest_CompactSplatsFusedPathLeavesGradsEmpty_Test;
        friend class ::MRNFStrategyTest_ApplyDecaySkipsFrozenRows_Test;
        friend class ::MRNFStrategyTest_DensificationInfoShapeIsTwoRows_Test;
        friend class ::MRNFStrategyTest_ZeroVisibilityProducesNoGrowth_Test;
        friend class ::MRNFStrategyTest_DirectAuxiliaryGrowthPreservesPrefix_Test;
        friend class ::MRNFStrategyTest_PerSplatMeanStepScalesWithExtentAndClamps_Test;
        friend class ::MRNFStrategyTest_RayDepthStateFollowsItsSplats_Test;
        friend class ::MRNFStrategyTest_PerSplatBiasCountersRestartWhereMomentsRestart_Test;
        friend class ::MRNFStrategyTest_LateLrAnnealDecaysOpacityAndColorAfterGrowth_Test;
        friend class ::MRNFStrategyTest_GrowthExchangeUsesMaxOfImpactRanks_Test;
        friend class ::MRNFStrategyTest_GrowthExchangeSkipsNewbornsUntilScored_Test;
        friend class ::MRNFStrategyTest_GrowthExchangeMarksSplitParentsAndChildrenNewborn_Test;
        friend class ::MRNFStrategyTest_RemoveGaussiansKeepsGrowthExchangeStateAligned_Test;
        friend class ::MRNFStrategyTest_GrowthExchangeFallsBackToCoverageWithoutImpact_Test;
        friend class ::MRNFStrategyTest_GrowthExchangeMatchesDemandWithoutDeletingParents_Test;
        friend class ::MRNFStrategyTest_GrowthExchangeCosineQuotaAtStartMiddleAndEnd_Test;
        friend class ::MRNFStrategyTest_NeedGrowthOffLeavesErrorAndRefillWeightsUnchanged_Test;
        friend class ::MRNFStrategyTest_NeedGrowthRefillsFromHighErrorGroup_Test;
        friend class ::MRNFStrategyTest_NeedGrowthPrefersLargeSplatsAtEqualError_Test;
        friend class ::MRNFStrategyTest_NeedGrowthDonorsAreLowestSummedContribution_Test;
        friend class ::MRNFStrategyTest_NeedGrowthProtectsNewbornsUntilAFullPass_Test;
        friend class ::MRNFStrategyTest_NeedGrowthDonorShareTapersToZero_Test;
        friend class ::MRNFStrategyTest_NeedGrowthFreedSlotsGoToTheNeedDraw_Test;
        friend class ::MRNFStrategyTest_NeedGrowthResumeRebuildsDonorMemory_Test;
        friend class ::MRNFStrategyTest_NeedGrowthReceiverSplitsOnlyAtTheCap_Test;
        friend class ::MRNFStrategyTest_RefineErrorViewMeanOffKeepsTheMaxOverViews_Test;
        friend class ::MRNFStrategyTest_RefineErrorViewMeanAveragesVisibleViewsAndResetsAtRefine_Test;
        friend class ::MRNFStrategyTest_RefineErrorViewMeanRestartsTheCountOnResume_Test;
        friend class ::MRNFStrategyTest_RefineErrorCompressFoldsThePowerOfEachViewOverViewsWithError_Test;
        friend class ::MRNFStrategyTest_RefineErrorCompressCandidatesAreEveryScoredSplat_Test;
        friend class ::MRNFStrategyTest_RefineErrorCompressNeedHasNoSparsityFactor_Test;
        friend class ::MRNFStrategyTest_RefineErrorCompressRanksSmallIntenseOverLargeFaint_Test;
        friend class ::MRNFStrategyTest_RefineErrorCompressSwitchStartsAFreshWindow_Test;
        friend class ::MRNFStrategyTest_RefineNeedPowerRaisesTheFoldedNeed_Test;
        friend class ::MRNFStrategyTest_RefineNeedPowerSharpensTheReceiverDraws_Test;
        friend class ::MRNFStrategyTest_DonorRankPerViewCountsCoveringViewsInEveryFold_Test;
        friend class ::MRNFStrategyTest_DonorRankPerViewFadesTheCountWithTheSum_Test;
        friend class ::MRNFStrategyTest_DonorRankPerViewOffKeepsTheSummedRanking_Test;
        friend class ::MRNFStrategyTest_DonorRankPerViewRanksBySumPerViewWithAFloor_Test;
        friend class ::MRNFStrategyTest_DonorRankPerViewCountFollowsItsSplats_Test;
        friend class ::MRNFStrategyTest_DonorRankPerViewResumeRebuildsTheCount_Test;
        friend class ::MRNFStrategyTest_DonorRankImpactOffKeepsTheBlendWeightDonors_Test;
        friend class ::MRNFStrategyTest_DonorRankImpactBackwardWritesImpactUnderNeedGrowth_Test;
        friend class ::MRNFStrategyTest_DonorRankImpactFadesTheMaximaLikeThePlainExchange_Test;
        friend class ::MRNFStrategyTest_DonorRankImpactKeepsASmallHighPeakSplat_Test;
        friend class ::MRNFStrategyTest_DonorRankImpactStateFollowsItsSplats_Test;
        friend class ::MRNFStrategyTest_DonorRankImpactResumeAndSwitchStartAfresh_Test;
        friend class ::MRNFStrategyTest_ExploreOffKeepsTheNoiseAndTheChildren_Test;
        friend class ::MRNFStrategyTest_ExploreWalkCovarianceFollowsTheSplatShape_Test;
        friend class ::MRNFStrategyTest_ExploreWalkAmplitudeFollowsTheRunAndOnlyRenderedSplatsWalk_Test;
        friend class ::MRNFStrategyTest_ExploreGatePeakWalksHiddenOpaqueSplatsOnly_Test;
        friend class ::MRNFStrategyTest_ExploreBackwardWritesVisibilityAndPeakBlendWeight_Test;
        friend class ::MRNFStrategyTest_ExploreScoutsShareOpacityTaperAndSeed_Test;
        friend class ::MRNFStrategyTest_ExploreScoutsSkipOpacityDecayUntilRendered_Test;
        friend class ::MRNFStrategyTest_ExploreRowsFollowTheirSplats_Test;
        friend class ::MRNFStrategyTest_ReceiverSplitTakesTheOpacityAndCrossScale_Test;
        friend class ::MRNFStrategyTest_RetireOffKeepsTheRefineBitIdentical_Test;
        friend class ::MRNFStrategyTest_RetireQuotaStartsAtTheCapAndTapersToGrowthEnd_Test;
        friend class ::MRNFStrategyTest_RetireTakesTheLowestPeakPastTheGraceAndRefillsTheSlots_Test;
        friend class ::MRNFStrategyTest_RetireRowsTrackPeakAndRendersAndFollowTheirSplats_Test;
        friend class ::MRNFStrategyTest_LossErrorMapImpliesTheCompressedViewMean_Test;
        friend class ::MRNFStrategyTest_LossErrorMapScoreEqualsTheMatchScore_Test;
        friend class ::MRNFStrategyTest_RetireSpareOffKeepsTheRetireesBitIdentical_Test;
        friend class ::MRNFStrategyTest_RetireSpareExemptsThinSplatsAndKeepsTheQuota_Test;
        friend class ::MRNFStrategyTest_RetireSpareWindowFollowsRendersAndSplats_Test;
        friend class ::MRNFStrategyTest_RetireSpareBackwardWritesTheRawMinorSigma_Test;
        friend class ::MRNFStrategyTest_RetireRankBothOffKeepsTheRetireesBitIdentical_Test;
        friend class ::MRNFStrategyTest_RetireRankBothKeepsHighImpactOrHighPeakSplats_Test;
        friend class ::MRNFStrategyTest_RetireRankBothFoldsTheWindowImpactApartFromTheDonors_Test;
        friend class ::MRNFStrategyTest_ImpactContrastNormDividesByThePhotoDeviation_Test;
        friend class ::MRNFStrategyTest_ReceiverKeepLongSplitsShortReceiversAcross_Test;
        friend class ::MRNFStrategyTest_ReceiverKeepLongWindowFollowsRendersAndSplats_Test;
        friend class ::MRNFStrategyTest_KeepCoveredOffKeepsTheRowsAndTheRetireesBitIdentical_Test;
        friend class ::MRNFStrategyTest_KeepCoveredDecayScalesByTheReach_Test;
        friend class ::MRNFStrategyTest_KeepCoveredTakesCoveredPicksOutOfTheRetireQuota_Test;

        void clear_rendered_support(const lfs::core::Tensor& indices);
        // Frees rows the way refine() prunes them: free mask, deleted mask, zero
        // quaternion (early exit in preprocessing) and fresh optimizer state.
        void soft_delete_rows(const lfs::core::Tensor& indices);

        // Growth exchange preserves splats with high removal impact. Rasterizers without
        // impact output use coverage as the single protection score. Need-guided growth runs
        // on the same plumbing: its donors are the splats with the least summed blend weight
        // over about one pass, on its own schedule.
        [[nodiscard]] bool growth_exchange_active(int iter) const;
        [[nodiscard]] bool need_guided_growth_on() const;
        // Growth ranks by the mean per-view error over the views that saw a splat, not the max.
        [[nodiscard]] bool refine_error_view_mean_on() const;
        // Growth ranks by error intensity: squared error map, per-view sum ^0.2, mean over the
        // views with error. Implies the view-mean accumulation.
        [[nodiscard]] bool refine_error_compress_on() const;
        // The model is full enough for donors; with need-guided growth every split then uses
        // the receiver rule.
        [[nodiscard]] bool at_cap(size_t active) const;
        // Turns the window error in _refine_weight_max into NEED (error first, sparsity as a
        // boost), then raises it to refine_need_power; after it, growth and refill draw by NEED
        // for the rest of the refine.
        void fold_need_into_refine_weights();
        void apply_refine_need_power();
        // Need-guided donors rank by summed blend weight per view that saw the splat, the view
        // count floored at donor_view_floor(), instead of by the summed blend weight.
        [[nodiscard]] bool donor_rank_per_view_on() const;
        // Need-guided donors rank by removal impact, as the plain exchange does: the backward
        // writes the window impact sum and pixel peak, each fades as a maximum, and a row ranks
        // by the later of its two positions. Takes precedence over the per-view rank.
        [[nodiscard]] bool donor_rank_impact_on() const;
        [[nodiscard]] float donor_view_floor() const;
        // Exploration (need-guided growth only): a per-render random walk in units of each splat's
        // own sigma (replaces the position noise), its gate on the peak blend weight, and faint
        // scout children at the cap that skip the opacity decay until rendered a few times.
        [[nodiscard]] bool explore_walk_on() const;
        [[nodiscard]] bool explore_gate_peak_on() const;
        [[nodiscard]] bool explore_scouts_on() const;
        // A(t) = explore_walk_scale x 8 x 0.01^(iter / iterations).
        [[nodiscard]] float explore_walk_amplitude(int iter) const;
        // The scout share at iter: explore_scout_share on the donor quota's cosine to grow_until_iter.
        [[nodiscard]] float explore_scout_share_at(int iter) const;
        // The receiver split's opacity factor at iter: receiver_opacity_scale, or with
        // receiver_opacity_scale_end > 0 the linear ramp to it over [0, grow_until_iter].
        [[nodiscard]] float receiver_opacity_scale_at(int iter) const;
        // Retirement (mrnf_retire_share > 0, FastGS): from the first refine at the cap, each refine also
        // frees the live splats with the lowest peak blend weight among those rendered at least
        // mrnf_retire_grace times since birth; the refine's refill takes their slots like pruned ones.
        // The peak is the exploration peak row, folded from a per-step peak with the match kernels.
        [[nodiscard]] bool retire_on() const;
        // The retire quota at iter: mrnf_retire_share x live, tapering linearly from first_iter (the first
        // refine at the cap) to 0 at grow_until_iter; 0 before the cap.
        [[nodiscard]] static size_t retire_count(const lfs::core::param::OptimizationParameters& params, int iter,
                                                 int first_iter, size_t live);
        // Adds the retirees of this refine to prune_mask (active, trainable rows not already pruned) and
        // returns their count.
        size_t mark_retirees(int iter, lfs::core::Tensor& prune_mask);
        // Thin-splat exemption (mrnf_retire_spare_px > 0 with retirement on): splats whose raw projected
        // minor-axis sigma, at its smallest over the renders of the last refine window that drew them, is
        // below mrnf_retire_spare_px pixels are not retired; the quota is taken from the other splats.
        [[nodiscard]] bool spare_on() const;
        // Closes the refine window of the minor sigma: rows drawn in it take the window's minimum, the others
        // keep their last value; the window starts afresh (+inf).
        void close_spare_window();
        // Retire ranking Both (mrnf_retire_rank both with retirement on): the backward writes the removal impact
        // every step; each refine folds the window into a faded maximum (half-life one pass, as the donor impact
        // memory) and the retirees are the lowest by the latest of their ranks in peak blend weight, summed
        // impact and pixel-peak impact.
        [[nodiscard]] bool retire_rank_both_on() const;
        // Folds the window impact into the retire impact rows; false when no impact was written this window.
        bool fold_retire_impact();
        // Whether the growth exchange ranks by removal impact (the plain exchange, or need-guided donors by impact).
        [[nodiscard]] bool exchange_ranks_by_impact() const;
        // Receiver keep-long (receiver_keep_long > 0 with need-guided growth, FastGS): a receiver whose raw
        // projected major sigma, at its largest over the renders of the last refine window that drew it, is below
        // receiver_keep_long pixels splits along its second axis.
        [[nodiscard]] bool keep_long_on() const;
        // Closes the refine window of the major sigma: rows drawn in it take the window's maximum, the others keep
        // their last value; the window starts afresh (0 = undrawn).
        void close_keep_long_window();
        // [N] flags of the receivers that split across this refine (empty when keep-long is off).
        [[nodiscard]] lfs::core::Tensor keep_long_split_across() const;
        // Keep covered (mrnf_keep_covered, FastGS): the peak and render rows of retirement are kept from the
        // first step, also without the retire quota. A row past the retire grace has reach
        // clamp(peak / max(opacity, 1e-6), 0, 1), the others 1. A row is kept when its reach is below
        // MRNF_COVERED_REACH and its largest axis is at least MRNF_KEEP_COVERED_SIZE_FACTOR x the median largest
        // axis of the live rows at this refine: both its decays scale by its reach, and the quota's pick drops it
        // (not refilled from other rows). Every other row decays and retires as with the switch off.
        [[nodiscard]] bool keep_covered_on() const;
        // Sets the size floor of this refine from the live rows (not free, not in prune_mask).
        void update_covered_size_floor(const lfs::core::Tensor& prune_mask);
        // [N] decay factor: the reach of a kept row, 1 for every other row; reach, when given, gets every row's
        // reach. Empty when keep covered is off or the rows do not match the model.
        [[nodiscard]] lfs::core::Tensor covered_factor(lfs::core::Tensor* reach = nullptr) const;
        // The keep-covered census line of a refine over the live rows that stay (not in prune_mask): rows past the
        // grace, those with reach below MRNF_COVERED_REACH, the median largest axis, the kept rows, and the quota
        // picks spared this refine.
        void log_keep_covered(int iter, const lfs::core::Tensor& prune_mask);
        // Sizes the exploration rows to the model (new rows zero) or drops the ones not in use.
        void ensure_explore_rows();
        // After the backward: walks the splats rendered in this step and clears the step's marks.
        void explore_step(int iter);
        // This window's count of views that covered each splat, or zeros. Read before the
        // window outputs are reset.
        [[nodiscard]] lfs::core::Tensor donor_window_views() const;
        void log_donor_view_counts(const lfs::core::Tensor& donors) const;
        [[nodiscard]] float exchange_importance_decay() const;
        [[nodiscard]] int exchange_warmup_windows() const;
        [[nodiscard]] size_t exchange_quota(int iter, size_t active) const;
        void reset_exchange_state();
        void reset_exchange_window_outputs();
        void update_exchange_importance();
        [[nodiscard]] lfs::core::Tensor select_exchange_victims(
            size_t count,
            const lfs::core::Tensor& excluded = {});
        // Marks rows filled by growth as newborn: no importance yet and not ranked until a
        // full pass over the training views has scored them.
        void record_exchange_births(const lfs::core::Tensor& split_parents = {});
        void refine(int iter);
        struct GrowthResult {
            lfs::core::Tensor split_indices;
            lfs::core::Tensor exchange_victim_indices;
            int replacement_count = 0;
            int growth_count = 0;
            size_t exchange_victims = 0;
            int receiver_splits = 0;
            int candidates = 0;
        };
        // refine_candidates defaults to compute_refine_candidates().
        GrowthResult grow_and_split(
            int iter,
            int pruned_count,
            lfs::core::Tensor refine_candidates = {},
            size_t exchange_quota = 0);
        // One line per refine: counts, the growth weights and the opacity gate. Info with
        // need-guided growth, debug (computed only when enabled) without.
        void log_refine_census(int iter, int pruned, const GrowthResult& growth, size_t quota) const;
        // Splits the given parents and places their children (free slots first,
        // then appended rows) in chunks of at most chunk_rows children.
        // Returns {children placed in free slots, children appended}. The receiver rule makes
        // children smaller on every axis (cross axes x receiver_cross_scale) with
        // receiver_opacity_scale x the parent's opacity.
        // With scout_share > 0 a seeded share of the children become scouts (explore_scouts).
        std::pair<size_t, size_t> split_parents_into_children(const lfs::core::Tensor& split_indices,
                                                              size_t chunk_rows, bool receiver = false,
                                                              float scout_share = 0.0f, uint64_t scout_seed = 0,
                                                              float receiver_opacity_scale = 1.0f);
        [[nodiscard]] bool screen_share_shrink_active(int iter) const;
        [[nodiscard]] lfs::core::Tensor compute_refine_candidates() const;
        void apply_decay(int iter);
        void inject_noise(int iter);
        void compact_splats(const lfs::core::Tensor& keep_mask);
        void compute_bounds();
        void sync_mean_learning_rate();
        void apply_late_lr_anneal(int iter);
        void ensure_densification_info_shape();
        void enforce_max_cap();
        void refresh_decay_schedule_from_current_state();
        void reset_edge_accumulator();
        void start_blob_seeding();
        void cancel_blob_seeding();
        void append_blob_seeds(BlobSeeds seeds);
        [[nodiscard]] lfs::core::Tensor visibility_accumulator() const;
        void refresh_camera_hull();
        void refresh_far_field_mask(size_t n);
        void publish_mean_step_far_mask();
        void ensure_mean_step_far_mask();
        // Keeps one ray-depth row per splat (new rows unobserved) and hands it to the optimizer
        // with the current step bound; releases it when ray_depth_lr is 0.
        void sync_ray_depth();
        void ensure_ray_depth_rows(size_t n);
        [[nodiscard]] size_t densification_row_count() const;
        // Gumbel top-k picks split between the crop box ROI and the rest by the growth share.
        [[nodiscard]] lfs::core::Tensor sample_with_roi_share(
            const lfs::core::Tensor& weights,
            int k,
            uint64_t seed);
        [[nodiscard]] lfs::core::Tensor sample_gumbel_topk(
            const lfs::core::Tensor& weights,
            int k,
            uint64_t seed);
        size_t append_child_rows(
            const lfs::core::Tensor& child_means,
            const lfs::core::Tensor& child_rotations,
            const lfs::core::Tensor& child_log_scales,
            const lfs::core::Tensor& child_sh0,
            const lfs::core::Tensor& child_shN,
            const lfs::core::Tensor& child_raw_opacities,
            size_t append_start,
            size_t K,
            lfs::training::sh_value::ShNMutationBatch* shn_batch = nullptr);
        void publish_vram_attribution() noexcept;
        size_t active_count() const;
        size_t free_count() const;
        [[nodiscard]] lfs::core::Tensor get_active_indices() const;
        void mark_as_free(const lfs::core::Tensor& indices);
        // Writes child shN linear rows directly into resident swizzled splat_data.shN().
        std::pair<lfs::core::Tensor, int64_t> fill_free_slots_with_data(
            const lfs::core::Tensor& positions,
            const lfs::core::Tensor& rotations,
            const lfs::core::Tensor& scales,
            const lfs::core::Tensor& sh0,
            const lfs::core::Tensor& shN,
            const lfs::core::Tensor& opacities,
            int64_t count,
            lfs::training::sh_value::ShNMutationBatch* shn_batch = nullptr);
        [[nodiscard]] mrnf_strategy::EdgeGuidance edge_guidance();

        std::unique_ptr<AdamOptimizer> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;
        lfs::core::SplatData* _splat_data = nullptr;
        std::unique_ptr<const lfs::core::param::OptimizationParameters> _params;

        std::shared_ptr<CameraDataset> _views;

        lfs::core::Tensor _rendered_count;
        // Window error per splat: the max over views, or with refine_error_view_mean the sum over
        // views, turned into the mean at the start of refine().
        lfs::core::Tensor _refine_weight_max;
        lfs::core::Tensor _precomputed_edge_scores;
        bool _edge_precompute_valid = false;
        lfs::core::Tensor _edge_score_sum;
        lfs::core::Tensor _edge_view_scores;
        int _edge_sample_count = 0;
        std::unique_ptr<BlobSeeder> _blob_seeder;
        // Triangulation runs off the training thread; its seeds join at the first refine after it finishes.
        std::future<BlobSeeds> _blob_seeds;
        std::jthread _blob_seed_worker;
        lfs::core::Tensor _far_field_mask;
        // [N, ray_depth::kStateStride] ray-depth rows; transient (fresh after a resume).
        lfs::core::Tensor _ray_depth_state;
        float _cam_centroid[3] = {0.0f, 0.0f, 0.0f};
        float _orbit_radius = 0.0f;
        bool _camera_hull_valid = false;
        lfs::core::Tensor _free_mask;
        bool _topology_frozen = false;

        // Growth exchange state; transient (rebuilt after a resume).
        lfs::core::Tensor _exchange_importance_sum;     // [N] leaky max of impact sum or coverage fallback
        lfs::core::Tensor _exchange_importance_peak;    // [N] leaky max of impact peak; invalid for fallback
        lfs::core::Tensor _exchange_birth;              // [N] window in which the row was last filled
        lfs::core::Tensor _exchange_window_impact_sum;  // [N] window removal-impact sum
        lfs::core::Tensor _exchange_window_impact_peak; // [N] window removal-impact peak
        lfs::core::Tensor _exchange_view_count;         // [N] faded count of covering views (donor_rank_per_view)
        lfs::core::Tensor _exchange_window_views;       // [N] window count of covering views, when no row holds it
        lfs::core::Tensor _exchange_window_coverage;    // [N] window coverage at the last fold (max fold only)
        lfs::core::Tensor _exchange_free_before_growth; // free slots right before growth

        // Exploration rows; transient (fresh after a resume).
        lfs::core::Tensor _explore_rendered;      // [N] views that rendered the splat this step
        lfs::core::Tensor _explore_peak;          // [N] running peak blend weight, halves in ~800 steps
        lfs::core::Tensor _explore_scout_renders; // [N] renders a scout needs before its opacity decays
        lfs::core::Tensor _retire_step_peak;      // [N] largest blend weight of this step (retirement)
        lfs::core::Tensor _retire_renders;        // [N] steps rendered since birth (retirement)
        int _retire_first_iter = -1;              // first refine at the cap; transient
        size_t _last_retired = 0;
        lfs::core::Tensor _spare_window_sigma; // [N] smallest raw minor sigma (px) this refine window, +inf: undrawn
        lfs::core::Tensor _spare_sigma;        // [N] that minimum for the last window that drew the splat, +inf: none
        size_t _last_spared = 0;
        size_t _last_covered_spared = 0;                                        // kept rows taken out of the retire quota's pick this refine
        float _covered_median_axis = 0.0f;                                      // median largest axis of the live rows at the last refine
        float _covered_log_size_floor = std::numeric_limits<float>::infinity(); // log of the kept size floor
        lfs::core::Tensor _retire_impact_sum;                                   // [N] faded max of the window summed removal impact (rank Both)
        lfs::core::Tensor _retire_impact_peak;                                  // [N] faded max of the window pixel-peak removal impact (rank Both)
        int _retire_impact_windows = 0;                                         // windows folded into the retire impact rows
        lfs::core::Tensor _keep_long_window_major;                              // [N] largest raw major sigma (px) this refine window, 0: undrawn
        lfs::core::Tensor _keep_long_major;                                     // [N] that maximum for the last window that drew it, +inf: none
        size_t _last_split_across = 0;
        int _exchange_windows = 0; // windows in the current importance memory
        bool _exchange_impact_written = false;
        int _exchange_first_iter = -1;

        DensifyNScratch _densify_n_scratch;
        lfs::core::Tensor _refine_counts_dev;

        std::size_t _strategy_required_peak_bytes = 0;
        std::size_t _strategy_allocated_peak_bytes = 0;
        std::size_t _densify_n_required_peak_bytes = 0;
        std::size_t _densify_n_allocated_peak_bytes = 0;
        std::size_t _densify_child_required_peak_bytes = 0;
        std::size_t _densify_child_allocated_peak_bytes = 0;

        mrnf_strategy::MRNFBounds _bounds = {};
        bool _bounds_valid = false;
        int _refine_windows_since_bounds = 0;
        float _median_splat_extent = 0.0f;
        bool _median_splat_extent_valid = false;

        // need (optional, [n]) multiplies the weights: need-guided refill.
        [[nodiscard]] lfs::core::Tensor build_replace_parent_weights(
            size_t n,
            const lfs::core::Tensor& active_mask,
            const lfs::core::Tensor& trainable_mask,
            const mrnf_strategy::EdgeGuidance& edge_guidance,
            const lfs::core::Tensor& need = {}) const;

        // Grow MRNF bookkeeping after parameter preflight and before any
        // parameter append. This is intentionally grow-only: refine resets
        // the buffers after the append, while this step preserves their live
        // prefix across a reservation boundary.
        void ensure_auxiliary_capacity_for_growth(size_t target_size);

        // MRNF uses independent exponential schedules for mean and scale learning rates.
        double _mean_lr_unscaled = 0.0;
        double _scale_lr_current = 0.0;
        double _mean_lr_gamma = 1.0;
        double _scale_lr_gamma = 1.0;
    };

} // namespace lfs::training
