/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/filesystem_utils.hpp"
#include "training/checkpoint.hpp"
#include <filesystem>
#include <functional>
#include <string>
#include <imgui.h>

namespace lfs::vis::gui {

    struct CheckpointLoadParams {
        std::filesystem::path checkpoint_path;
        std::filesystem::path dataset_path;
        std::filesystem::path output_path;
    };

    class ResumeCheckpointPopup {
    public:
        using ConfirmCallback = std::function<void(const CheckpointLoadParams& params)>;

        void render(const ImVec2& viewport_pos = {0, 0}, const ImVec2& viewport_size = {0, 0});
        void show(const std::filesystem::path& checkpoint_path);
        void setOnConfirm(ConfirmCallback callback) { on_confirm_ = std::move(callback); }
        [[nodiscard]] bool isOpen() const { return popup_open_; }

    private:
        void renderPathRow(const char* label, const std::string& path, float max_width);

        ConfirmCallback on_confirm_;
        std::filesystem::path checkpoint_path_;
        lfs::training::CheckpointHeader header_;
        std::string stored_dataset_path_;
        std::string dataset_path_buffer_;
        std::string output_path_buffer_;
        bool dataset_valid_ = false;
        bool popup_open_ = false;
        bool should_open_ = false;
    };

} // namespace lfs::vis::gui
