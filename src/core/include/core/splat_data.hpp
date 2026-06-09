/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/point_cloud.hpp"
#include "core/tensor.hpp"

#include <atomic>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <glm/glm.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::geometry {
    class BoundingBox;
}

namespace lfs::core {

    namespace param {
        struct TrainingParameters;
    }

    struct SplatLodTree {
        static constexpr std::size_t kChunkSplats = 65'536;
        static constexpr uint32_t    kInvalidPage = 0xFFFFFFFFu;

        struct ChunkFileRange {
            uint64_t file_offset = 0;
            uint64_t file_bytes = 0;
            uint64_t payload_offset = 0;
            uint64_t payload_bytes = 0;
            uint64_t base = 0;
            uint64_t count = 0;
        };

        struct RadSource {
            std::filesystem::path path;
            uint32_t chunk_size = static_cast<uint32_t>(kChunkSplats);
            uint64_t metadata_bytes = 0;
            std::vector<ChunkFileRange> chunks;

            [[nodiscard]] bool valid() const { return !path.empty() && !chunks.empty(); }
        };

        std::vector<uint16_t> child_count;
        std::vector<uint32_t> child_start;
        std::vector<uint8_t>  lod_level;
        std::vector<glm::vec3> centers;
        std::vector<float>     sizes;
        std::vector<uint32_t> page_to_chunk;
        std::vector<uint32_t> chunk_to_page;
        RadSource             rad_source;
        bool                   lod_opacity_encoded = false;

        size_t total_nodes() const { return child_count.size(); }
        size_t chunk_count() const { return (total_nodes() + kChunkSplats - 1) / kChunkSplats; }
        bool   has_tree() const  { return !child_count.empty(); }
    };

    using SplatTensorAllocator = std::function<Tensor(TensorShape shape,
                                                      size_t capacity,
                                                      DataType dtype,
                                                      std::string_view name)>;

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
    class LFS_CORE_API SplatData {
    public:
        enum class ShNLayout {
            Canonical,
            Swizzled
        };

        struct FrozenRange {
            std::size_t start = 0;
            std::size_t count = 0;
        };

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
                  float scene_scale,
                  ShNLayout shN_layout = ShNLayout::Canonical);

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
        void set_scene_scale(float scene_scale) { _scene_scale = scene_scale; }
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

        // shN is stored in vksplat-style float4-packed swizzled layout: 1D float tensor of
        // sh_swizzled_float_count(N, max_rest) = ceil(N / SH_REORDER_SIZE)
        //                                        * slots_for_max_rest
        //                                        * SH_REORDER_SIZE * 4 floats.
        // SH0 allocates no shN rest storage; SH1/SH2/SH3 allocate 3/6/12 float4 slots per
        // primitive. sh_swizzled_index(p, k, max_rest) / shAt(p, k, slots) returns a
        // float4-slot index (multiply by 4 for the float offset).
        // shN() / shN_raw() return the swizzled tensor directly. Use shN_canonical() to
        // materialise a deswizzled [N, K, 3] view for I/O / transforms / scene merge.
        inline Tensor& shN() { return _shN; }
        inline const Tensor& shN() const { return _shN; }
        inline Tensor& shN_raw() { return _shN; }
        inline const Tensor& shN_raw() const { return _shN; }

        // Materialise a deswizzled [N, K, 3] copy of resident shN storage where
        // K = sh_rest_coeffs of the max SH degree. Always allocates a new tensor — not a view.
        Tensor shN_canonical() const;

        // Host-side variant for export/checkpoint paths. Copies the resident swizzled buffer
        // to CPU first and unpacks there, avoiding a full canonical SH allocation on CUDA.
        Tensor shN_canonical_cpu() const;

        // Replace _shN with the swizzled form of a canonical-layout source tensor.
        // `canonical` may be [N, K, 3] or [N, K*3]; K may be 0 for SH degree 0. The
        // swizzled buffer is allocated/resized to fit N with optional `capacity`.
        void shN_set_from_canonical(const Tensor& canonical, size_t capacity = 0);

        // Number of "rest" SH coefficients implied by the current active SH degree
        // (0 / 3 / 8 / 15 for degree 0 / 1 / 2 / 3).
        size_t active_sh_coeffs_rest() const;

        // Number of resident "rest" SH coefficients implied by the current max SH degree.
        size_t max_sh_coeffs_rest() const;

        // ========== Soft deletion (for undo/redo crop support) ==========
        Tensor& deleted() { return _deleted; }
        [[nodiscard]] const Tensor& deleted() const { return _deleted; }
        [[nodiscard]] bool has_deleted_mask() const { return _deleted.is_valid(); }
        [[nodiscard]] unsigned long visible_count() const;

        // Cached count of deleted gaussians, refreshed by the owner (trainer) on its
        // own thread/stream via refresh_deleted_count(). Lets other threads (e.g. the
        // Vulkan viewer) decide whether the soft-delete path is needed with a plain
        // atomic read — no GPU reduction on the shared mask, which would race the
        // strategy's writes and can deadlock against the render interop handshake.
        [[nodiscard]] std::size_t deleted_count() const {
            return _deleted_count.load(std::memory_order_relaxed);
        }
        void refresh_deleted_count();

        [[nodiscard]] const std::vector<FrozenRange>& frozen_ranges() const { return _frozen_ranges; }
        [[nodiscard]] bool has_frozen_ranges() const { return !_frozen_ranges.empty(); }
        void set_frozen_ranges(std::vector<FrozenRange> ranges) { _frozen_ranges = std::move(ranges); }
        void clear_frozen_ranges() { _frozen_ranges.clear(); }
        void remap_frozen_ranges_after_keep(size_t old_size, const std::vector<int>& kept_old_indices);
        void remap_frozen_ranges_after_keep(size_t old_size, const std::vector<int64_t>& kept_old_indices);

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
        void set_max_sh_degree(int sh_degree);
        bool set_sh_degree(int sh_degree);

        // ========== Serialization ==========
        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is, SplatTensorAllocator tensor_allocator = {});

        // Allocator used to back the parameter tensors (e.g. Vulkan-external interop
        // storage). Retained so edits that rebuild tensors (apply_deleted) can keep
        // them in the same storage the renderer requires, instead of falling back to
        // the default device allocator.
        void set_tensor_allocator(SplatTensorAllocator allocator) {
            _tensor_allocator = std::move(allocator);
        }

    public:
        // Holds the magnitude of the screen space gradient (used for densification)
        Tensor _densification_info;

        // Optional LOD tree (populated by RAD loader, null for training/non-RAD scenes)
        std::unique_ptr<SplatLodTree> lod_tree;

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
        // Cached nonzero count of _deleted; see refresh_deleted_count(). Atomic so
        // the render thread can read it without a data race on the writer.
        std::atomic<std::size_t> _deleted_count{0};

        // Backing allocator for parameter tensors (see set_tensor_allocator).
        SplatTensorAllocator _tensor_allocator;
        std::vector<FrozenRange> _frozen_ranges;

        // Allow free functions in splat_data_transform.cpp to access private members
        friend LFS_CORE_API SplatData& transform(SplatData&, const glm::mat4&);
        friend LFS_CORE_API SplatData crop_by_cropbox(const SplatData&, const lfs::geometry::BoundingBox&, bool);
        friend LFS_CORE_API SplatData extract_by_mask(const SplatData&, const Tensor&);
        friend LFS_CORE_API void random_choose(SplatData&, int, int);
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
    LFS_CORE_API std::expected<SplatData, std::string> init_model_from_pointcloud(
        const param::TrainingParameters& params,
        Tensor scene_center,
        const PointCloud& point_cloud,
        int capacity = 0,
        SplatTensorAllocator tensor_allocator = {});

} // namespace lfs::core
