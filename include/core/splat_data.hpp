/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/point_cloud.hpp"
#include <expected>
#include <filesystem>
#include <future>
#include <geometry/bounding_box.hpp>
#include <glm/glm.hpp>
#include <mutex>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace gs {
    namespace param {
        struct TrainingParameters;
    }

    class SplatData {
    public:
        SplatData() = default;
        ~SplatData();

        // Delete copy operations
        SplatData(const SplatData&) = delete;
        SplatData& operator=(const SplatData&) = delete;

        // Custom move operations (needed because of mutex)
        SplatData(SplatData&& other) noexcept;
        SplatData& operator=(SplatData&& other) noexcept;

        // Constructor
        SplatData(int sh_degree,
                  torch::Tensor means,
                  torch::Tensor sh0,
                  torch::Tensor shN,
                  torch::Tensor scaling,
                  torch::Tensor rotation,
                  torch::Tensor opacity,
                  float scene_scale);

        // Static factory method to create from PointCloud
        static std::expected<SplatData, std::string> init_model_from_pointcloud(
            const gs::param::TrainingParameters& params,
            torch::Tensor scene_center,
            const PointCloud& point_cloud);

        // Computed getters (implemented in cpp)
        torch::Tensor get_means() const;
        torch::Tensor get_opacity() const;
        torch::Tensor get_rotation() const;
        torch::Tensor get_scaling() const;
        torch::Tensor get_shs() const;

        // that's really a stupid hack for now. This stuff must go into a CUDA kernel
        SplatData& transform(const glm::mat4& transform_matrix);

        // Simple inline getters
        int get_active_sh_degree() const { return _active_sh_degree; }
        float get_scene_scale() const { return _scene_scale; }
        int64_t size() const { return _means.size(0); }

        // Raw tensor access for optimization (inline for performance)
        inline torch::Tensor& means() { return _means; }
        inline const torch::Tensor& means() const { return _means; }
        inline torch::Tensor& opacity_raw() { return _opacity; }
        inline const torch::Tensor& opacity_raw() const { return _opacity; }
        inline torch::Tensor& rotation_raw() { return _rotation; }
        inline const torch::Tensor& rotation_raw() const { return _rotation; }
        inline torch::Tensor& scaling_raw() { return _scaling; }
        inline const torch::Tensor& scaling_raw() const { return _scaling; }
        inline torch::Tensor& sh0() { return _sh0; }
        inline const torch::Tensor& sh0() const { return _sh0; }
        inline torch::Tensor& shN() { return _shN; }
        inline const torch::Tensor& shN() const { return _shN; }

        // Utility methods
        void increment_sh_degree();
        void set_active_sh_degree(int sh_degree);

        // Export methods - join_threads controls sync vs async
        // if stem is not empty save splat as stem.ply
        void save_ply(const std::filesystem::path& root, int iteration, bool join_threads = true, std::string stem = "") const;
        std::filesystem::path save_sog(const std::filesystem::path& root, int iteration, int kmeans_iterations = 10, bool join_threads = true) const;

        // Get attribute names for the PLY format
        std::vector<std::string> get_attribute_names() const;

        SplatData crop_by_cropbox(const gs::geometry::BoundingBox& bounding_box) const;

    public:
        // Holds the magnitude of the screen space gradient
        torch::Tensor _densification_info = torch::empty({0});

    private:
        int _active_sh_degree = 0;
        int _max_sh_degree = 0;
        float _scene_scale = 0.f;

        torch::Tensor _means;
        torch::Tensor _sh0;
        torch::Tensor _shN;
        torch::Tensor _scaling;
        torch::Tensor _rotation;
        torch::Tensor _opacity;

        // Async save management
        mutable std::mutex _save_mutex;
        mutable std::vector<std::future<void>> _save_futures;

        // Convert to point cloud for export
        PointCloud to_point_cloud() const;

        // Helper methods for async save management
        void wait_for_saves() const;
        void cleanup_finished_saves() const;
    };
} // namespace gs