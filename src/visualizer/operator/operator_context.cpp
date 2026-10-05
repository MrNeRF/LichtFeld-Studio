/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "operator_context.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::op {

    OperatorContext::OperatorContext(SceneManager& scene) : scene_(scene) {}

    std::vector<std::string> OperatorContext::selectedNodes() const {
        return scene_.getSelectedNodeNames();
    }

    void OperatorContext::setModalEvent(const ModalEvent& event) {
        current_event_ = std::make_unique<ModalEvent>(event);
    }

} // namespace lfs::vis::op
