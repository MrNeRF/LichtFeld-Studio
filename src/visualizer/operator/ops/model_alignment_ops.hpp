/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "point_pick_operator.hpp"
#include "rendering/rendering_types.hpp"
#include <any>
#include <array>
#include <span>

namespace lfs::vis::op {
    struct ModelAlignmentState {
        bool linked = false;
        std::string display_mode = "separate";
        bool can_swap_roles = false;
        bool can_delete_pair = false;
        bool can_undo = false;
    };

    class LFS_VIS_API ModelAlignmentOperator : public PointPickOperator {
    public:
        ModelAlignmentOperator();
        ~ModelAlignmentOperator() override;
        OperatorResult invoke(OperatorContext&, OperatorProperties&) override;
        void cancel(OperatorContext&) override;
        static LFS_LOCAL_SYMBOL const OperatorDescriptor DESCRIPTOR;
        const OperatorDescriptor& descriptor() const override { return DESCRIPTOR; }
        bool poll(const OperatorContext&, const OperatorProperties* = nullptr) const override;

        struct Session {
            std::array<core::Uuid, 2> nodes;
            std::array<glm::mat4, 2> world;
            SplitViewMode previous_mode;
            float previous_position;
            size_t previous_offset;
            SplitViewPanelId previous_focus;
            Viewport* primary_viewport;
            Viewport previous_primary;
            Viewport previous_secondary;
            std::optional<glm::mat4> preview;
            std::optional<SplitViewPanelId> composite;
            std::optional<std::array<Viewport, 2>> free_views;
            bool linked = false;
        };
        // Non-owning viewer-thread access; lifetime belongs to the modal operator.
        static ModelAlignmentOperator* active() { return active_; }
        const Session* session() const { return session_ ? &*session_ : nullptr; }
        PointPickState& pointState() { return point_state_; }
        const PointPickState& pointState() const { return point_state_; }
        bool pendingPairOnRight() const { return pair_edit_.pending_right; }
        struct PreviewFeedback {
            std::optional<int> focus;
            std::string summary;
        };
        const PreviewFeedback& feedback() const { return feedback_; }
        const std::vector<size_t>& pairColorSlots() const { return pair_edit_.colors; }
        static bool canBegin(const core::Scene&, const core::SceneNode&, const core::SceneNode&);
        bool begin(Viewport&, const core::Scene&, const core::SceneNode&, const core::SceneNode&);
        void end();
        void setPreview(std::optional<glm::mat4>);
        bool canSwapRoles(const core::Scene&) const;
        ModelAlignmentState state() const;
        void swapRoles(const core::Scene&);
        void setComposite(std::optional<SplitViewPanelId>);
        void toggleCameraLink();
        void syncCamera(const Viewport&);

    private:
        friend class ModelAlignmentTest;
        static ModelAlignmentOperator* active_;
        PointPickState point_state_;
        PreviewFeedback feedback_;
        std::optional<Session> session_;
        struct PairEdit {
            bool pending_right = false;
            std::vector<size_t> colors;
        } pair_edit_;
        bool supportsPointEditing() const override { return true; }
        std::any capturePointEditState() const override { return pair_edit_; }
        void restorePointEditState(const std::any& state) override { pair_edit_ = std::any_cast<PairEdit>(state); }
        void pointsCleared() override { pair_edit_ = {}; }
        void appendPickedPoint(const PointSample&) override;
        std::pair<size_t, size_t> pointRemovalRange(size_t, bool) override;
        void showPointMenu(glm::dvec2, int) override;
        bool acceptsPointSample(const PointSample&, std::optional<int>) const override;
        bool applyPickedPoints(OperatorContext&, float) override;
        std::optional<OperatorResult> prepareModal(OperatorContext&) override;
        bool pointEditingBlocked() const override;
        void beforePointEdit(OperatorContext&) override;
        void handleToolUiAction(OperatorContext&, PointPickState::UiAction) override;
        OperatorResult handleToolAction(OperatorContext&, input::Action) override;
        std::optional<glm::mat4> computePairPreviewTransform() const;
        bool updatePreview(OperatorContext&);
        void restorePreview(OperatorContext&);
        bool applyAlignment(OperatorContext&);
        void reportPairFitFailure() const;
        void publishPairPreviewFeedback(const glm::mat4&);
        void publishView();
    };
    inline const ModelAlignmentOperator::Session* modelAlignmentSession() {
        const auto* op = ModelAlignmentOperator::active();
        return op ? op->session() : nullptr;
    }
    inline constexpr bool hasEnoughAlignmentPairs(size_t count) { return count >= 6 && count % 2 == 0; }
    [[nodiscard]] LFS_VIS_API std::optional<glm::dmat4> computePairAlignment(
        std::span<const glm::dvec3> source, std::span<const glm::dvec3> reference, bool adjust_scale);
} // namespace lfs::vis::op
