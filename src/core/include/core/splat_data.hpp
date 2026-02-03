/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/point_cloud.hpp"
#include "core/tensor.hpp"

#include <expected>
#include <filesystem>
#include <glm/fwd.hpp>
#include <string>
#include <vector>

namespace lfs::geometry {
    class BoundingBox;
}

namespace lfs::core {

    namespace param {
        struct TrainingParameters;
    }

    /**
     * @brief Core data structure for Gaussian splat representation
     *
     * Contains the fundamental attributes of a Gaussian splat scene:
     * - Positions (means)
     * - Spherical harmonics coefficients (sh0, shN)
     * - Scaling factors
     * - Rotation quaternions
     * - Opacity values
     *
     * Note: Gradients are managed by AdamOptimizer, not SplatData.
     */
    class SplatData {
    public:
        SplatData() = default;
        ~SplatData();

        // Delete copy operations
        SplatData(const SplatData&) = delete;
        SplatData& operator=(const SplatData&) = delete;

        // Custom move operations
        SplatData(SplatData&& other) noexcept;
        SplatData& operator=(SplatData&& other) noexcept;

        // Constructor
        SplatData(int sh_degree,
                  Tensor means,
                  Tensor sh0,
                  Tensor shN,
                  Tensor scaling,
                  Tensor rotation,
                  Tensor opacity,
                  float scene_scale);

        // ========== Computed getters ==========
        Tensor get_means() const;
        Tensor get_opacity() const;  // Returns sigmoid(opacity_raw)
        Tensor get_rotation() const; // Returns normalized quaternions
        Tensor get_scaling() const;  // Returns exp(scaling_raw)
        Tensor get_shs() const;      // Returns concatenated sh0 + shN

        // ========== Simple inline getters ==========
        int get_active_sh_degree() const { return _active_sh_degree; }
        int get_max_sh_degree() const { return _max_sh_degree; }
        float get_scene_scale() const { return _scene_scale; }
        unsigned long size() const { return static_cast<unsigned long>(_means.shape()[0]); }

        // ========== Raw tensor access (for optimization) ==========
        inline Tensor& means() { return _means; }
        inline const Tensor& means() const { return _means; }
        inline Tensor& means_raw() { return _means; }
        inline const Tensor& means_raw() const { return _means; }
        inline Tensor& opacity_raw() { return _opacity; }
        inline const Tensor& opacity_raw() const { return _opacity; }
        inline Tensor& rotation_raw() { return _rotation; }
        inline const Tensor& rotation_raw() const { return _rotation; }
        inline Tensor& scaling_raw() { return _scaling; }
        inline const Tensor& scaling_raw() const { return _scaling; }
        inline Tensor& sh0() { return _sh0; }
        inline const Tensor& sh0() const { return _sh0; }
        inline Tensor& sh0_raw() { return _sh0; }
        inline const Tensor& sh0_raw() const { return _sh0; }
        inline Tensor& shN() { return _shN; }
        inline const Tensor& shN() const { return _shN; }
        inline Tensor& shN_raw() { return _shN; }
        inline const Tensor& shN_raw() const { return _shN; }

        // ========== Soft deletion (for undo/redo crop support) ==========
        Tensor& deleted() { return _deleted; }
        [[nodiscard]] const Tensor& deleted() const { return _deleted; }
        [[nodiscard]] bool has_deleted_mask() const { return _deleted.is_valid(); }
        [[nodiscard]] unsigned long visible_count() const;

        // Mark gaussians as deleted, returns previous state for undo
        Tensor soft_delete(const Tensor& mask);
        void undelete(const Tensor& mask);
        void clear_deleted();

        // Permanently remove deleted gaussians (compacts data)
        // Returns number of gaussians removed
        size_t apply_deleted();

        // ========== Capacity management ==========
        // Reserve capacity for parameter tensors (for MCMC densification)
        void reserve_capacity(size_t capacity);

        // ========== SH degree management ==========
        void increment_sh_degree();
        void set_active_sh_degree(int sh_degree);
        void set_max_sh_degree(int sh_degree) { _max_sh_degree = sh_degree; }

        // ========== Serialization ==========
        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);

    public:
        // Holds the magnitude of the screen space gradient (used for densification)
        Tensor _densification_info;

    private:
        int _active_sh_degree = 0;
        int _max_sh_degree = 0;
        float _scene_scale = 0.f;

        // Parameters
        Tensor _means;
        Tensor _sh0;
        Tensor _shN;
        Tensor _scaling;
        Tensor _rotation;
        Tensor _opacity;

        // Soft deletion mask: bool tensor [N], true = hidden from rendering
        Tensor _deleted;

        // Allow free functions in splat_data_export.cpp and splat_data_transform.cpp
        // to access private members
        friend void save_ply(const SplatData&, const std::filesystem::path&, int, bool, std::string);
        friend std::filesystem::path save_sog(const SplatData&, const std::filesystem::path&, int, int, bool);
        friend PointCloud to_point_cloud(const SplatData&);
        friend std::vector<std::string> get_attribute_names(const SplatData&);
        friend SplatData& transform(SplatData&, const glm::mat4&);
        friend SplatData crop_by_cropbox(const SplatData&, const lfs::geometry::BoundingBox&, bool);
        friend SplatData extract_by_mask(const SplatData&, const Tensor&);
        friend void random_choose(SplatData&, int, int);
    };

    // ========== Free function: Factory ==========

    /**
     * @brief Create SplatData from a PointCloud
     * @param params Training parameters (SH degree, init settings)
     * @param scene_center Center of the scene
     * @param point_cloud Source point cloud
     * @param capacity If > 0, pre-allocate for this many gaussians (bypasses memory pool)
     * @return SplatData on success, error string on failure
     */
    std::expected<SplatData, std::string> init_model_from_pointcloud(
        const param::TrainingParameters& params,
        Tensor scene_center,
        const PointCloud& point_cloud,
        int capacity = 0);

} // namespace lfs::core
