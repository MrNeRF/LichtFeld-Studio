/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>

namespace lfs::core::param {
    struct TrainingParameters;
}

namespace lfs::app {

    // Refuses drivers and GPUs this build cannot run on. Returns false after reporting the
    // reason; callers must not continue. show_dialog is for the interactive GUI path only —
    // a modal in a CLI or CI run blocks the process forever. viewer_only lets the GUI
    // continue on the Vulkan tensor backend when CUDA is unusable.
    bool preflightGpu(bool show_dialog, bool viewer_only = false);

    class Application {
    public:
        int run(std::unique_ptr<lfs::core::param::TrainingParameters> params);
    };

} // namespace lfs::app
