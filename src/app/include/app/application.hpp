/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <memory>
#include <optional>

namespace lfs::core::param {
    struct TrainingParameters;
}

namespace lfs::app {

    // Refuses drivers and GPUs this build cannot run on. Returns false after reporting the
    // reason; callers must not continue. show_dialog is for the interactive GUI path only —
    // a modal in a CLI or CI run blocks the process forever.
    bool preflightGpu(bool show_dialog);

    // Routes core image loads (masks, depth and normal sidecars) through the cache loader.
    void install_image_loader(bool use_cpu_memory);

    // LFS_LPIPS_WEIGHTS, else the downloaded weights; nullopt with a warning when neither is available.
    [[nodiscard]] std::optional<std::filesystem::path> prepare_lpips_weights(bool allow_download);

    class Application {
    public:
        int run(std::unique_ptr<lfs::core::param::TrainingParameters> params);
    };

} // namespace lfs::app
