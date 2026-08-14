/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/mapped_file.hpp"
#include "core/point_cloud.hpp"
#include "core/tensor.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
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

    // Per-node tree metadata in the exact layouts the GPU traversal consumes.
    struct NodeBoundsRecord {
        float x = 0.0f, y = 0.0f, z = 0.0f, size = 0.0f;
    };
    struct NodeLinksRecord {
        uint32_t child_start = 0;
        // child_count:16 | level:8 | flags:8 (bit0 leaf, bit1 root, bit2 lodOpacity)
        uint32_t packed = 0;
        uint32_t parent = 0xFFFFFFFFu;
        uint32_t logical = 0xFFFFFFFFu;

        [[nodiscard]] uint32_t childCount() const { return packed & 0xffffu; }
        [[nodiscard]] uint32_t level() const { return (packed >> 16u) & 0xffu; }
    };
    static_assert(sizeof(NodeBoundsRecord) == 16);
    static_assert(sizeof(NodeLinksRecord) == 16);

    // On-disk .rad.meta sidecar records (20 B/node): bounds quantized against
    // a per-chunk frame so 1B-leaf trees fit a ~34 GB cache. Centers are u16
    // against the chunk AABB (error ≤ extent/65535); sizes are u16 over the
    // chunk's log-size range (relative error ≤ range/65535).
    struct RadMetaBoundsQ {
        uint16_t qx = 0, qy = 0, qz = 0, qsize = 0;
    };
    struct RadMetaLinksQ {
        uint32_t child_start = 0;
        // Same encoding as NodeLinksRecord::packed; logical is derived as
        // chunk * kChunkSplats + offset at expansion time.
        uint32_t packed = 0;
        uint32_t parent = 0xFFFFFFFFu;
    };
    // Per-chunk payload range + dequantization frame (sidecar chunk table).
    struct RadMetaChunkRecord {
        uint64_t payload_offset = 0;
        uint64_t payload_bytes = 0;
        float bbox_min[3] = {0.0f, 0.0f, 0.0f};
        float bbox_extent[3] = {0.0f, 0.0f, 0.0f};
        float log_size_min = 0.0f;
        float log_size_range = 0.0f;

        [[nodiscard]] glm::vec3 dequantCenter(const RadMetaBoundsQ& q) const {
            constexpr float kInv = 1.0f / 65535.0f;
            return {bbox_min[0] + static_cast<float>(q.qx) * kInv * bbox_extent[0],
                    bbox_min[1] + static_cast<float>(q.qy) * kInv * bbox_extent[1],
                    bbox_min[2] + static_cast<float>(q.qz) * kInv * bbox_extent[2]};
        }
        [[nodiscard]] float dequantSize(const RadMetaBoundsQ& q) const {
            constexpr float kInv = 1.0f / 65535.0f;
            return std::exp(log_size_min + static_cast<float>(q.qsize) * kInv * log_size_range);
        }
    };
    static_assert(sizeof(RadMetaBoundsQ) == 8);
    static_assert(sizeof(RadMetaLinksQ) == 12);
    static_assert(sizeof(RadMetaChunkRecord) == 48);

    struct SplatLodTree {
        static constexpr std::size_t kChunkSplats = 2'048;
        static constexpr uint32_t kInvalidPage = 0xFFFFFFFFu;

        struct ChunkFileRange {
            uint64_t file_offset = 0;
            uint64_t file_bytes = 0;
            uint64_t payload_offset = 0;
            uint64_t payload_bytes = 0;
            uint64_t file_base = 0;
            uint64_t file_count = 0;
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

        // Memory-mapped per-node metadata from a .rad.meta sidecar; replaces
        // the in-RAM vectors for out-of-core models (23 B/node does not scale
        // to billions of nodes). Copies share the mapping.
        struct NodeMetaView {
            std::shared_ptr<MappedFile> file;
            const RadMetaBoundsQ* bounds = nullptr;
            const RadMetaLinksQ* links = nullptr;
            const RadMetaChunkRecord* chunks = nullptr;
            std::size_t node_count = 0;
            std::size_t chunk_count = 0;
            std::size_t leaf_count = 0;

            [[nodiscard]] bool valid() const {
                return file != nullptr && bounds != nullptr && links != nullptr &&
                       chunks != nullptr && node_count > 0;
            }
            [[nodiscard]] const RadMetaChunkRecord& chunkOf(const std::size_t node) const {
                return chunks[node / kChunkSplats];
            }
        };

        std::vector<uint16_t> child_count;
        std::vector<uint32_t> child_start;
        std::vector<uint8_t> lod_level;
        std::vector<glm::vec3> centers;
        std::vector<float> sizes;
        std::vector<uint32_t> page_to_chunk;
        std::vector<uint32_t> chunk_to_page;
        RadSource rad_source;
        NodeMetaView meta_view;
        bool lod_opacity_encoded = false;

        size_t total_nodes() const {
            return meta_view.valid() ? meta_view.node_count : child_count.size();
        }
        size_t chunk_count() const { return (total_nodes() + kChunkSplats - 1) / kChunkSplats; }
        bool has_tree() const { return total_nodes() > 0; }
        // True when the per-node vectors are materialized in RAM (legacy and
        // in-core models); false when only the sidecar view backs the tree.
        bool nodes_in_memory() const { return !child_count.empty(); }

        [[nodiscard]] uint32_t child_start_at(const std::size_t i) const {
            return meta_view.valid() ? meta_view.links[i].child_start : child_start[i];
        }
        [[nodiscard]] uint16_t child_count_at(const std::size_t i) const {
            return meta_view.valid() ? static_cast<uint16_t>(meta_view.links[i].packed & 0xffffu)
                                     : child_count[i];
        }
        [[nodiscard]] uint8_t level_at(const std::size_t i) const {
            return meta_view.valid()
                       ? static_cast<uint8_t>((meta_view.links[i].packed >> 16u) & 0xffu)
                       : (i < lod_level.size() ? lod_level[i] : 0);
        }
        [[nodiscard]] glm::vec3 center_at(const std::size_t i) const {
            return meta_view.valid() ? meta_view.chunkOf(i).dequantCenter(meta_view.bounds[i])
                                     : centers[i];
        }
        [[nodiscard]] float size_at(const std::size_t i) const {
            return meta_view.valid() ? meta_view.chunkOf(i).dequantSize(meta_view.bounds[i])
                                     : sizes[i];
        }
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

        // Deep-copies scalars and tensors, including q16 bounds. Skips allocator,
        // capacity hook, LOD tree, frozen ranges, and layout generation.
        [[nodiscard]] SplatData clone() const;

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

        // float2 bounds per 256-splat block when SH value quant is ON.
        inline Tensor& shN_value_bounds() { return _shN_value_bounds; }
        inline const Tensor& shN_value_bounds() const { return _shN_value_bounds; }
        // q16 pad-dropped codes: Float16 bit-patterns + per-block bounds.
        // IEEE f16 float4-swizzle (exportable viewer path) is also Float16 but has
        // no bounds — do not treat it as q16.
        [[nodiscard]] bool shN_value_quantized() const {
            return _shN.is_valid() && _shN.dtype() == DataType::Float16 &&
                   _shN_value_bounds.is_valid() && _shN_value_bounds.numel() > 0;
        }
        // IEEE half float4-swizzle (same topology as fp32, 2 B/component). Used by
        // the GUI exportable block so the Vulkan viewer can zero-copy bind half SH.
        [[nodiscard]] bool shN_ieee_f16() const {
            return _shN.is_valid() && _shN.dtype() == DataType::Float16 && !shN_value_quantized();
        }

        // Non-SH display attrs (rotation / log-scale / logit opacity) as IEEE f16.
        // Means stay Float32. Used by the viewer layout when a path stores half
        // attrs (lodq pool / future exportable packing). Training exportable
        // currently keeps these as Float32 and zero-copies them into the viewport.
        [[nodiscard]] bool non_sh_attrs_f16() const {
            return _rotation.is_valid() && _rotation.dtype() == DataType::Float16 &&
                   _scaling.is_valid() && _scaling.dtype() == DataType::Float16 &&
                   _opacity.is_valid() && _opacity.dtype() == DataType::Float16;
        }

        // Materialise a deswizzled [N, K, 3] copy of resident shN storage where
        // K = sh_rest_coeffs of the max SH degree. Always allocates a new tensor — not a view.
        // When quantized, dequants to fp32 first (PLY/checkpoint bit-compat).
        Tensor shN_canonical() const;

        // Host-side variant for export/checkpoint paths. Copies the resident swizzled buffer
        // to CPU first and unpacks there, avoiding a full canonical SH allocation on CUDA.
        Tensor shN_canonical_cpu() const;

        // Synchronize every unique non-null CUDA home stream, then re-home all valid CUDA
        // tensor members onto the default stream. Call before a trainer destroys its streams.
        void detach_from_streams();

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
        [[nodiscard]] std::uint64_t deleted_mask_version() const {
            return _deleted_mask_version.load(std::memory_order_relaxed);
        }
        void refresh_deleted_count();

        [[nodiscard]] const std::vector<FrozenRange>& frozen_ranges() const { return _frozen_ranges; }
        [[nodiscard]] bool has_frozen_ranges() const { return !_frozen_ranges.empty(); }
        void set_frozen_ranges(std::vector<FrozenRange> ranges) { _frozen_ranges = std::move(ranges); }
        void clear_frozen_ranges() { _frozen_ranges.clear(); }
        void remap_frozen_ranges_after_keep(size_t old_size, const std::vector<int>& kept_old_indices);
        void remap_frozen_ranges_after_keep(size_t old_size, const std::vector<int64_t>& kept_old_indices);

        // Mark gaussians as deleted, returns newly deleted mask for undo
        Tensor soft_delete(const Tensor& mask);
        void undelete(const Tensor& mask);
        void clear_deleted();

        // VkSplat (and other consumers) require: if a deleted mask is present it
        // must be a contiguous CUDA bool tensor of numel == size(). Call this
        // after any N-mutating path (densify grow, compact, load, random_choose)
        // that may have left the mask stale. Legal outcomes:
        //   - mask matches the contract (possibly rebuilt / resized in place)
        //   - mask cleared (has_deleted_mask()==false)
        // Bumps deleted_mask_version() when the mask storage or content changes.
        void reconcile_deleted_mask();

        // True when has_deleted_mask() and the packer contract holds.
        [[nodiscard]] bool deleted_mask_matches_size() const;

        // Bump deleted_mask_version after external writers replace/mutate
        // deleted() storage (strategies, compact gathers, exportable rebind).
        void notify_deleted_mask_changed();

        // Permanently remove deleted gaussians (compacts data)
        // Returns number of gaussians removed
        size_t apply_deleted();

        // ========== Capacity management ==========
        // Reserve capacity for parameter tensors (for MCMC densification)
        void reserve_capacity(size_t capacity);

        // ========== SH degree management ==========
        void increment_sh_degree();
        // Non-q16 shN verify/resize shared by degree setters (fail-loud on
        // unclassifiable Float16 storage). Internal maintenance helper.
        void verify_or_resize_non_q16_shN(size_t n, size_t cap, uint32_t layout_rest);
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
        [[nodiscard]] bool has_tensor_allocator() const noexcept {
            return static_cast<bool>(_tensor_allocator);
        }
        // Allocate a named param tensor via the active backing allocator when set
        // (exportable / Vulkan-external), otherwise zeros_direct. Used by SH q16
        // encode to keep codes+bounds inside the exportable block.
        [[nodiscard]] Tensor allocate_named_param(
            const TensorShape& shape,
            std::size_t capacity,
            DataType dtype,
            std::string_view name);

        // Optional hook for exportable / external storage growth.
        // When densification needs more rows than the committed exportable block,
        // AdamOptimizer calls this before falling back to Tensor::cat (which would
        // leave the Vulkan zero-copy path). Returns true if capacity >= needed.
        using CapacityEnsureFn = std::function<bool(std::size_t needed_rows)>;
        void set_capacity_ensure(CapacityEnsureFn fn) {
            _capacity_ensure = std::move(fn);
        }
        // Full-model rebuilds must transfer and reinstall this hook. A fresh
        // SplatData has none, so densification would otherwise abort when its
        // committed capacity is exhausted.
        [[nodiscard]] bool has_capacity_ensure() const noexcept {
            return static_cast<bool>(_capacity_ensure);
        }
        [[nodiscard]] CapacityEnsureFn release_capacity_ensure() {
            return std::move(_capacity_ensure);
        }
        // Bumped by rebindSplatData / exportable grow so densify code that holds
        // Tensor& locals across ensure_param_capacity can detect invalidated
        // references after a grow.
        [[nodiscard]] std::uint64_t param_layout_generation() const noexcept {
            return _param_layout_generation;
        }
        void note_param_layout_changed() noexcept {
            ++_param_layout_generation;
        }
        // needed_rows growth hook. When layout_changed is non-null, sets it true
        // iff the hook rebased parameter tensors (generation advanced). Callers
        // that held Tensor& across the call MUST re-fetch via get_param / means_raw.
        [[nodiscard]] bool ensure_param_capacity(std::size_t needed_rows,
                                                 bool* layout_changed = nullptr) {
            if (layout_changed) {
                *layout_changed = false;
            }
            if (means_raw().is_valid() && means_raw().capacity() >= needed_rows) {
                return true;
            }
            if (!_capacity_ensure) {
                return false;
            }
            const std::uint64_t gen0 = _param_layout_generation;
            const bool ok = _capacity_ensure(needed_rows);
            if (layout_changed) {
                *layout_changed = (_param_layout_generation != gen0);
            }
            return ok;
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

        // Parameters — any new Tensor member must be added to detach_from_streams().
        Tensor _means;
        Tensor _sh0;
        // When sh_value quant is ON: Float16 bit-pattern u16 codes,
        // pad-dropped cell-linear layout (n_cells = coeffs_rest*3). When OFF: Float32
        // float4-swizzled as before. Access via shN() / data_ptr(); do not assume dtype.
        Tensor _shN;
        // float2 bounds per 256-splat block (only when quant ON); empty when fp32.
        Tensor _shN_value_bounds;
        Tensor _scaling;
        Tensor _rotation;
        Tensor _opacity;

        // Soft deletion mask: bool tensor [N], true = hidden from rendering
        Tensor _deleted;
        // Cached nonzero count of _deleted; see refresh_deleted_count(). Atomic so
        // the render thread can read it without a data race on the writer.
        std::atomic<std::size_t> _deleted_count{0};
        // Monotonic content revision for the soft-delete mask. The renderer uses
        // this to refresh per-ring opacity copies when the mask is updated in place.
        std::atomic<std::uint64_t> _deleted_mask_version{0};

        // Backing allocator for parameter tensors (see set_tensor_allocator).
        SplatTensorAllocator _tensor_allocator;
        CapacityEnsureFn _capacity_ensure;
        // Monotonic; advanced when exportable rebind rewrites param tensor VAs.
        std::uint64_t _param_layout_generation = 0;
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
