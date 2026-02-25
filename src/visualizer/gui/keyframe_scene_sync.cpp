/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/keyframe_scene_sync.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "sequencer/keyframe.hpp"
#include "visualizer_impl.hpp"

#include <cassert>
#include <format>

static_assert(static_cast<uint8_t>(lfs::sequencer::EasingType::LINEAR) == 0);
static_assert(static_cast<uint8_t>(lfs::sequencer::EasingType::EASE_IN) == 1);
static_assert(static_cast<uint8_t>(lfs::sequencer::EasingType::EASE_OUT) == 2);
static_assert(static_cast<uint8_t>(lfs::sequencer::EasingType::EASE_IN_OUT) == 3);

namespace lfs::vis::gui {

    KeyframeSceneSync::KeyframeSceneSync(SequencerController& controller, VisualizerImpl* viewer)
        : controller_(controller),
          viewer_(viewer) {
        assert(viewer_);
    }

    KeyframeSceneSync::~KeyframeSceneSync() = default;

    void KeyframeSceneSync::syncToSceneGraph() {
        auto* sm = viewer_->getSceneManager();
        if (!sm)
            return;

        auto& scene = sm->getScene();
        core::Scene::Transaction tx(scene);
        scene.removeKeyframeNodes();

        const auto& timeline = controller_.timeline();
        if (timeline.empty())
            return;

        const auto group_id = scene.addKeyframeGroup("Keyframes");

        size_t visible_index = 0;
        for (size_t i = 0; i < timeline.keyframes().size(); ++i) {
            const auto& kf = timeline.keyframes()[i];
            if (kf.is_loop_point)
                continue;

            auto data = std::make_unique<core::KeyframeData>();
            data->keyframe_index = visible_index;
            data->time = kf.time;
            data->position = kf.position;
            data->rotation = kf.rotation;
            data->focal_length_mm = kf.focal_length_mm;
            data->easing = static_cast<uint8_t>(kf.easing);

            const auto name = std::format("Keyframe {}", visible_index + 1);
            scene.addKeyframe(name, group_id, std::move(data));
            ++visible_index;
        }
    }

    void KeyframeSceneSync::emitNodeSelectedForKeyframe(const size_t index) {
        auto* sm = viewer_->getSceneManager();
        if (!sm)
            return;

        const auto name = std::format("Keyframe {}", index + 1);
        auto& scene = sm->getScene();
        if (scene.getNode(name)) {
            lfs::core::events::ui::NodeSelected{.path = name, .type = "KEYFRAME", .metadata = {}}.emit();
        }
    }

    void KeyframeSceneSync::setupEvents() {
        using namespace lfs::core::events;

        state::KeyframeListChanged::when([this](const auto&) {
            syncToSceneGraph();
        });

        cmd::SequencerGoToKeyframe::when([this](const auto& e) {
            auto& timeline = controller_.timeline();
            if (e.keyframe_index >= timeline.size())
                return;

            controller_.selectKeyframe(e.keyframe_index);
            const auto* kf = timeline.getKeyframe(e.keyframe_index);
            if (!kf)
                return;

            controller_.seek(kf->time);

            auto& vp = viewer_->getViewport();
            vp.camera.R = glm::mat3_cast(kf->rotation);
            vp.camera.t = kf->position;

            if (auto* rm = viewer_->getRenderingManager()) {
                rm->setFocalLength(kf->focal_length_mm);
                rm->markDirty(DirtyFlag::CAMERA);
            }

            emitNodeSelectedForKeyframe(e.keyframe_index);
        });

        cmd::SequencerSelectKeyframe::when([this](const auto& e) {
            auto& timeline = controller_.timeline();
            if (e.keyframe_index >= timeline.size())
                return;

            controller_.selectKeyframe(e.keyframe_index);
            emitNodeSelectedForKeyframe(e.keyframe_index);
        });

        ui::RenderSettingsChanged::when([this](const auto& e) {
            if (!e.focal_length_mm)
                return;
            const auto sel = controller_.selectedKeyframe();
            if (!sel.has_value())
                return;
            auto& timeline = controller_.timeline();
            if (*sel >= timeline.size())
                return;
            timeline.setKeyframeFocalLength(*sel, *e.focal_length_mm);
            controller_.updateLoopKeyframe();
        });

        cmd::SequencerDeleteKeyframe::when([this](const auto& e) {
            if (e.keyframe_index == 0)
                return;

            auto& timeline = controller_.timeline();
            if (e.keyframe_index >= timeline.size())
                return;

            controller_.selectKeyframe(e.keyframe_index);
            controller_.removeSelectedKeyframe();

            state::KeyframeListChanged{.count = timeline.size()}.emit();
        });

        cmd::SequencerSetKeyframeEasing::when([this](const auto& e) {
            auto& timeline = controller_.timeline();
            if (e.keyframe_index >= timeline.size())
                return;

            const auto easing = static_cast<sequencer::EasingType>(e.easing_type);
            timeline.setKeyframeEasing(e.keyframe_index, easing);

            state::KeyframeListChanged{.count = timeline.size()}.emit();
        });
    }

} // namespace lfs::vis::gui
