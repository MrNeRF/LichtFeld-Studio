/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera_views.hpp"
#include "core/error.hpp"
#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::core {
    class Scene;
    class SplatData;
} // namespace lfs::core

namespace lfs::training {

    struct FastGSFusedExtraGradients;

    // A pose offset per photograph from its pose at construction, refined by Adam from the rasterizer's camera gradient
    // (kernels/camera_pose_math.hpp). After each step the camera draws its refined pose; commit() also writes it into
    // the cameras' R and T, which saves, exports and the viewer read.
    class CameraPoseOffsets {
    public:
        explicit CameraPoseOffsets(std::vector<core::CameraViewGroup> groups);

        [[nodiscard]] const std::vector<core::CameraViewGroup>& groups() const { return groups_; }
        // Where the next backward writes the camera gradient.
        [[nodiscard]] float* grad_w2c() { return grad_w2c_.ptr<float>(); }
        [[nodiscard]] float* grad_cam_position() { return grad_cam_position_.ptr<float>(); }

        // One Adam step of the drawn camera's photograph, on the backward's stream.
        void step(const core::Camera& camera, float lr_rotation, float lr_translation, cudaStream_t stream);
        // R and T (and the device pose) to the refined poses, or back to the start poses. The caller keeps other
        // readers of the cameras' R and T out.
        void commit();
        void restore_start_poses();

        struct HostState {
            std::vector<float> offsets; // [N, 6]: r (radians), d (scene units), in the start camera's axes
            std::vector<float> steps;   // [N]
        };
        [[nodiscard]] HostState download() const;
        // Start pose of `row`, row-major 4x4 world-to-camera.
        [[nodiscard]] const float* start_pose(int row) const { return start_host_.data() + 16 * row; }

    private:
        struct DrawIndex {
            int row;
            int draw;
        };
        std::vector<core::CameraViewGroup> groups_;
        std::unordered_map<int, DrawIndex> draw_of_uid_;
        std::vector<int> first_draw_;
        std::vector<float> rotations_host_;
        core::Tensor rotations_;
        void set_row_pose(size_t row, const float* pose);
        std::vector<float> start_host_;
        lfs::core::Tensor start_;   // [N, 16]
        lfs::core::Tensor offsets_; // [N, 6]
        lfs::core::Tensor adam_m_;  // [N, 6]
        lfs::core::Tensor adam_v_;  // [N, 6]
        lfs::core::Tensor steps_;   // [N]
        lfs::core::Tensor grad_w2c_;
        lfs::core::Tensor grad_cam_position_;
        lfs::core::Tensor refined_w2c_;
        lfs::core::Tensor refined_cam_position_;
        cudaStream_t stream_ = nullptr; // where the steps run
    };

    // --pose-opt: refines every training camera's pose during the run. No offset moves for the first 10% of the run;
    // the rate is full until half way, then halves smoothly down to 1/128 at the last iteration.
    class CameraPoseRefinement {
    public:
        // Fails unless every camera is drawn with the pinhole projection (pinhole, or undistorted).
        static lfs::Result<std::unique_ptr<CameraPoseRefinement>> create(
            std::vector<core::CameraViewGroup> groups, int iterations);
        CameraPoseRefinement(std::vector<core::CameraViewGroup> groups, int iterations);

        // Asks the coming backward for the camera gradient while the rate is above zero.
        void bind_gradients(FastGSFusedExtraGradients& extra, int iteration);
        // After the backward of `camera`; `stream` is the backward's.
        void step(const lfs::core::Camera& camera, int iteration, cudaStream_t stream);
        void commit() { offsets_.commit(); }

        [[nodiscard]] bool reports_at(int iteration) const;
        [[nodiscard]] std::string summary(int iteration) const;
        // One row per photograph: updates, move and turn, with their world-axis components.
        [[nodiscard]] lfs::Status write_table(const std::filesystem::path& path) const;

    private:
        [[nodiscard]] float rate(int iteration) const;

        CameraPoseOffsets offsets_;
        int iterations_;
        int start_;
        int hold_;
    };

    // Fits each held-out photograph to the frozen model, cycling through its views for a fixed number of steps, so a model trained
    // with --pose-opt is scored from where its views fit. apply() and restore() move R and T as commit() does.
    class RefittedCameraPoses {
    public:
        struct Settings {
            int resize_factor = 1;
            int max_width = 0;
            float lambda_dssim = 0.2f;
            bool mip_filter = false;
        };

        RefittedCameraPoses(std::vector<core::CameraViewGroup> groups,
                            lfs::core::SplatData& model,
                            lfs::core::Tensor& background,
                            const Settings& settings);
        void apply() { offsets_.commit(); }
        void restore() { offsets_.restore_start_poses(); }

    private:
        void refit(lfs::core::SplatData& model, lfs::core::Tensor& background, const Settings& settings);

        CameraPoseOffsets offsets_;
    };

    // The scene's cameras as COLMAP sparse files in output_sparse_path, written over the dataset's model.
    [[nodiscard]] lfs::Status write_scene_cameras_colmap(
        const lfs::core::Scene& scene,
        const std::filesystem::path& dataset_path,
        const std::filesystem::path& output_sparse_path);

} // namespace lfs::training
