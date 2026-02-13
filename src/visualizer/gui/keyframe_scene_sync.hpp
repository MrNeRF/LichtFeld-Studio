/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>

#include "sequencer/sequencer_controller.hpp"

namespace lfs::core {
    class Scene;
}

namespace lfs::vis {
    class VisualizerImpl;
}

namespace lfs::vis::gui {

    class LFS_VIS_API KeyframeSceneSync {
    public:
        KeyframeSceneSync(SequencerController& controller, VisualizerImpl* viewer);
        ~KeyframeSceneSync();

        void setupEvents();
        void syncToSceneGraph();

    private:
        void emitNodeSelectedForKeyframe(size_t index);

        SequencerController& controller_;
        VisualizerImpl* viewer_;
    };

} // namespace lfs::vis::gui
