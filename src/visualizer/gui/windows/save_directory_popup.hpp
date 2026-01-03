/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/filesystem_utils.hpp"
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <imgui.h>

namespace lfs::vis::gui {

    struct DatasetLoadParams {
        std::filesystem::path dataset_path;
        std::filesystem::path output_path;
        std::optional<std::filesystem::path> init_path; // Custom init PLY (replaces points3D)
    };

    class SaveDirectoryPopup {
    public:
        using ConfirmCallback = std::function<void(const DatasetLoadParams& params)>;

        void render(const ImVec2& viewport_pos = {0, 0}, const ImVec2& viewport_size = {0, 0});
        void show(const std::filesystem::path& dataset_path);
        void setOnConfirm(ConfirmCallback callback) { on_confirm_ = std::move(callback); }
        [[nodiscard]] bool isOpen() const { return popup_open_; }

    private:
        [[nodiscard]] static std::filesystem::path deriveDefaultOutputPath(const std::filesystem::path& dataset_path);
        void renderPathRow(const char* label, const std::string& path, float max_width);

        ConfirmCallback on_confirm_;
        lfs::io::DatasetInfo dataset_info_;
        std::string output_path_buffer_;
        std::string init_path_buffer_;
        bool use_custom_init_ = false;
        bool popup_open_ = false;
        bool should_open_ = false;
    };

} // namespace lfs::vis::gui
