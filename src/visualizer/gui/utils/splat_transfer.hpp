/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <atomic>
#include <core/export.hpp>
#include <core/splat_data.hpp>
#include <filesystem>
#include <future>
#include <glm/glm.hpp>
#include <memory>
#include <string>
namespace lfs::vis::gui {
    struct SplatTransferOptions {
        std::string format = "ply";
        std::string name = "splat";
        int sh_degree = -1;
        int sog_iterations = 30;
        bool z_up = true;
    };
    // UI-thread CPU snapshot; worker transforms and atomically commits the selected file format.
    LFS_VIS_API std::future<std::filesystem::path> prepareSplatTransfer(
        const core::SplatData& model, glm::mat4 world,
        std::shared_ptr<std::atomic<bool>> cancelled,
        SplatTransferOptions options = {});
} // namespace lfs::vis::gui
