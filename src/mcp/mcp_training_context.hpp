/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "training/trainer.hpp"

#include <array>
#include <atomic>
#include <expected>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace lfs::mcp {

    class LFS_MCP_API TrainingContext {
    public:
        struct SelectionWorkspace {
            core::Tensor& locked_groups_device_mask;
            core::Tensor& selection_scratch_buffer;
            core::Tensor& selection_polygon_vertex_buffer;
            std::array<core::Tensor, 2>& selection_output_buffers;
            size_t& selection_output_buffer_index;
        };

        static TrainingContext& instance();

        std::expected<void, std::string> load_dataset(
            const std::filesystem::path& path,
            const core::param::TrainingParameters& params);

        std::expected<void, std::string> load_checkpoint(
            const std::filesystem::path& path);

        std::expected<void, std::string> save_checkpoint(
            const std::filesystem::path& path);

        std::expected<void, std::string> save_ply(
            const std::filesystem::path& path);

        std::expected<std::string, std::string> render_to_base64(
            int camera_index = 0,
            int width = 0,
            int height = 0);

        std::expected<core::Tensor, std::string> compute_screen_positions(
            int camera_index = 0);

        std::expected<void, std::string> start_training();
        void stop_training();
        void pause_training();
        void resume_training();

        bool is_loaded() const {
            std::lock_guard lock(mutex_);
            return scene_ != nullptr;
        }
        bool is_training() const {
            std::lock_guard lock(mutex_);
            return training_thread_ != nullptr;
        }

        std::shared_ptr<core::Scene> scene() const {
            std::lock_guard lock(mutex_);
            return scene_;
        }
        std::shared_ptr<training::Trainer> trainer() const {
            std::lock_guard lock(mutex_);
            return trainer_;
        }
        core::param::TrainingParameters params() const {
            std::lock_guard lock(mutex_);
            return params_;
        }

        template <typename Fn>
        decltype(auto) with_selection_workspace(Fn&& fn) {
            std::lock_guard lock(selection_mutex_);
            SelectionWorkspace workspace{
                .locked_groups_device_mask = locked_groups_device_mask_,
                .selection_scratch_buffer = selection_scratch_buffer_,
                .selection_polygon_vertex_buffer = selection_polygon_vertex_buffer_,
                .selection_output_buffers = selection_output_buffers_,
                .selection_output_buffer_index = selection_output_buffer_index_,
            };
            return std::forward<Fn>(fn)(workspace);
        }

        void shutdown();

    private:
        TrainingContext() = default;
        ~TrainingContext();
        void stop_training_locked();

        // shared_ptr allows tool lambdas to hold references across async boundaries.
        // INVARIANT: stop_training() must complete before scene_.reset().
        std::shared_ptr<core::Scene> scene_;
        std::shared_ptr<training::Trainer> trainer_;
        core::param::TrainingParameters params_;
        core::Tensor locked_groups_device_mask_;
        core::Tensor selection_scratch_buffer_;
        core::Tensor selection_polygon_vertex_buffer_;
        std::array<core::Tensor, 2> selection_output_buffers_;
        size_t selection_output_buffer_index_ = 0;

        std::unique_ptr<std::jthread> training_thread_;
        mutable std::mutex mutex_;
        mutable std::mutex selection_mutex_;
    };

    LFS_MCP_API void register_scene_tools();

} // namespace lfs::mcp
