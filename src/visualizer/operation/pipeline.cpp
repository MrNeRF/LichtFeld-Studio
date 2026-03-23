/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "pipeline.hpp"
#include "core/logger.hpp"
#include "scene/scene_manager.hpp"
#include "undo_history.hpp"

namespace lfs::vis::op {

    Pipeline::Pipeline(std::string name)
        : name_(std::move(name)) {}

    Pipeline& Pipeline::add(OperationFactory factory, OperatorProperties props) {
        stages_.push_back({std::move(factory), std::move(props)});
        return *this;
    }

    Pipeline& Pipeline::add(Stage stage) {
        stages_.push_back(std::move(stage));
        return *this;
    }

    Pipeline& Pipeline::operator|(Stage stage) {
        return add(std::move(stage));
    }

    bool Pipeline::poll(SceneManager& scene) const {
        for (const auto& stage : stages_) {
            auto op = stage.factory();
            if (!op->poll(scene)) {
                return false;
            }
        }
        return true;
    }

    std::string Pipeline::description() const {
        if (!name_.empty()) {
            return name_;
        }

        if (stages_.empty()) {
            return "Empty Pipeline";
        }

        if (stages_.size() == 1) {
            auto op = stages_[0].factory();
            return op->label();
        }

        std::string desc;
        for (size_t i = 0; i < stages_.size(); ++i) {
            auto op = stages_[i].factory();
            if (i > 0) {
                desc += " | ";
            }
            desc += op->id();
        }
        return desc;
    }

    ModifiesFlag Pipeline::collectModifications() const {
        ModifiesFlag mods = ModifiesFlag::NONE;
        for (const auto& stage : stages_) {
            auto op = stage.factory();
            mods = mods | op->modifies();
        }
        return mods;
    }

    OperationResult Pipeline::execute(SceneManager& scene) {
        if (stages_.empty()) {
            return OperationResult::skipped("Empty pipeline");
        }

        if (!poll(scene)) {
            return OperationResult::failure("Poll failed");
        }

        auto snapshot = std::make_unique<SceneSnapshot>(scene, description());

        auto mods = collectModifications();
        if (hasFlag(mods, ModifiesFlag::SELECTION)) {
            snapshot->captureSelection();
        }
        if (hasFlag(mods, ModifiesFlag::TRANSFORMS)) {
            snapshot->captureTransforms(scene.getSelectedNodeNames());
        }
        if (hasFlag(mods, ModifiesFlag::TOPOLOGY)) {
            snapshot->captureTopology();
        }

        std::any data;
        for (auto& stage : stages_) {
            auto op = stage.factory();
            LOG_DEBUG("Executing operation: {}", op->id());

            auto result = op->execute(scene, stage.props, data);
            if (!result.ok()) {
                LOG_ERROR("Operation {} failed: {}", op->id(), result.error);
                return result;
            }
            data = std::move(result.data);
        }

        snapshot->captureAfter();
        pushSceneSnapshotIfChanged(std::move(snapshot));

        return OperationResult::success(std::move(data));
    }

} // namespace lfs::vis::op
