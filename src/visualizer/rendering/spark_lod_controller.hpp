/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/splat_data.hpp"
#include "rendering_types.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace lfs::vis {

    class SparkLodController {
    public:
        struct LodParameters {
            size_t max_splats = DEFAULT_LOD_MAX_SPLATS;
            size_t requested_max_splats = 0;
            float pixel_scale_limit = DEFAULT_LOD_PIXEL_SCALE_LIMIT;
            float lod_render_scale = DEFAULT_LOD_RENDER_SCALE;
            float object_scale = 1.0f;
            float behind_camera_penalty = DEFAULT_LOD_BEHIND_CAMERA_FOVEATION;
            float cone_foveation = DEFAULT_LOD_CONE_FOVEATION;
            float cone_inner_degrees = DEFAULT_LOD_CONE_INNER_DEGREES;
            float cone_outer_degrees = DEFAULT_LOD_CONE_OUTER_DEGREES;
            float outside_view_foveation = DEFAULT_LOD_OUTSIDE_VIEW_FOVEATION;
            float viewport_half_tan_x = 0.0f;
            float viewport_half_tan_y = 0.0f;
            float ortho_half_width = 0.0f;
            float ortho_half_height = 0.0f;
            float prefetch_pixel_scale_ratio = DEFAULT_LOD_PREFETCH_PIXEL_SCALE_RATIO;
            bool viewport_foveation = true;
            bool orthographic = false;
        };

        struct Stats {
            bool available = false;
            bool enabled = false;
            bool active = false;
            bool has_tree = false;
            bool lod_opacity_encoded = false;
            bool async_result_ready = false;
            bool budget_limited = false;
            bool threshold_limited = false;
            bool output_limited = false;
            bool budget_fill_active = false;
            bool budget_repair_active = false;
            bool full_quality_reference = false;
            bool transition_active = false;
            size_t model_splats = 0;
            size_t tree_nodes = 0;
            size_t non_leaf_nodes = 0;
            size_t full_quality_splats = 0;
            size_t selected_splats = 0;
            size_t max_splats = 0;
            size_t requested_max_splats = 0;
            size_t budget_repair_limit = 0;
            size_t output_size = 0;
            size_t frontier_size = 0;
            size_t leaf_count = 0;
            size_t chunk_splats = 0;
            size_t chunk_count = 0;
            size_t touched_chunks = 0;
            size_t resident_chunks = 0;
            size_t outside_view_nodes = 0;
            size_t behind_view_nodes = 0;
            size_t viewport_throttled_nodes = 0;
            uint16_t root_child_count = 0;
            uint16_t max_child_count = 0;
            float pixel_scale_limit = 0.0f;
            float budget_fill_pixel_scale_limit = 0.0f;
            float min_pixel_scale = 0.0f;
            float lod_render_scale = 1.0f;
            float behind_camera_penalty = 0.0f;
            float cone_foveation = 0.0f;
            float cone_inner_degrees = 0.0f;
            float cone_outer_degrees = 0.0f;
            uint64_t selection_hash = 0;
            uint64_t generation = 0;
            std::vector<std::pair<uint8_t, size_t>> level_histogram;
            // True when the GPU selector drives the rendered cut; selection counts
            // and residency then come from the renderer, not this controller.
            bool gpu_selection = false;
            size_t gpu_output_capacity = 0;
            size_t gpu_overflow = 0;
            float gpu_pixel_scale_feedback = 1.0f;
            size_t pool_pages = 0;
            size_t streaming_jobs = 0;
            size_t miss_chunks = 0;
            size_t deferred_requests = 0;
            bool admission_frozen = false;
        };

        SparkLodController();
        ~SparkLodController();

        // Attach to a SplatData that has a lod_tree
        void attach(const lfs::core::SplatData& data);
        void detach();

        // Synchronous traversal. Returns selected count.
        size_t update(const glm::mat4& view_matrix, const LodParameters& params);
        void updateAsync(const glm::mat4& view_matrix, const LodParameters& params);
        bool swapAsyncResults(bool allow_transition = true,
                              bool remap_physical_indices = true,
                              bool update_cpu_publish_state = true);
        bool hasReadyResults() const;
        void invalidatePendingWork();
        void setReadyCallback(std::function<void()> callback);
        void applyPageMaps(const std::vector<uint32_t>& page_to_chunk,
                           const std::vector<uint32_t>& chunk_to_page,
                           bool remap_physical_indices = true);

        // Accessors
        bool hasTree() const;
        const std::vector<uint32_t>& selectedIndices() const;
        const std::vector<uint32_t>& selectedLogicalIndices() const;
        const std::vector<uint32_t>& selectedLevels() const;
        const std::vector<float>& selectedWeights() const;
        const std::vector<uint32_t>& fullQualityIndices() const;
        const std::vector<uint32_t>& fullQualityLogicalIndices() const;
        const std::vector<uint32_t>& fullQualityLevels() const;
        void advanceTransition();
        bool transitionActive() const;
        std::vector<uint32_t> touchedChunks() const;
        uint64_t selectionHash() const;
        uint64_t statsGeneration() const;
        bool pageMappingActive() const;
        void activateFullQualityReference();
        Stats stats() const;

    private:
        struct LodTreeNode {
            glm::vec3 center;
            float size;
            uint32_t child_start;
            uint32_t parent_index;
            uint16_t child_count;
            uint8_t lod_level;
        };

        struct TraversalView {
            glm::vec3 origin;
            glm::vec3 forward;
        };

        struct PreparedTraversal {
            TraversalView view;
            float object_scale = 1.0f;
            float behind_camera_penalty = DEFAULT_LOD_BEHIND_CAMERA_FOVEATION;
            float cone_foveation = DEFAULT_LOD_CONE_FOVEATION;
            float cone_dot0 = 1.0f;
            float cone_dot = 1.0f;
            float cone_blend_denominator = 0.0f;
            bool cone_blend_valid = false;
            bool cone_tail_valid = true;
            glm::mat4 object_to_view{1.0f};
            float outside_view_foveation = DEFAULT_LOD_OUTSIDE_VIEW_FOVEATION;
            float viewport_half_tan_x = 0.0f;
            float viewport_half_tan_y = 0.0f;
            float ortho_half_width = 0.0f;
            float ortho_half_height = 0.0f;
            bool viewport_foveation = false;
            bool orthographic = false;
        };

        struct HeapNode {
            uint32_t index;
            float pixel_scale;
        };

        struct HeapCompare {
            bool operator()(const HeapNode& a, const HeapNode& b) const {
                return a.pixel_scale < b.pixel_scale;
            }
        };

        struct PixelScaleResult {
            float pixel_scale = 0.0f;
            bool outside_view = false;
            bool behind_view = false;
            bool viewport_throttled = false;
        };

        struct TraversalScratch {
            std::vector<HeapNode> frontier_heap;
            std::vector<std::uint8_t> touched_chunk_bitmap;
            std::vector<float> touched_chunk_priority;
            std::vector<uint32_t> touched_chunks;
            std::vector<size_t> level_counts;
        };

        struct PageMapSnapshot {
            std::vector<uint32_t> page_to_chunk;
            std::vector<uint32_t> chunk_to_page;
            size_t resident_chunks = 0;
        };

        struct TraversalGuidance {
            std::vector<std::uint8_t> exact_selected;
            std::vector<std::uint8_t> selected_subtree;
        };

        PixelScaleResult computePixelScale(uint32_t node_index,
                                           const PreparedTraversal& traversal) const;
        static TraversalView makeTraversalView(const glm::mat4& object_to_view);
        static PreparedTraversal prepareTraversal(const glm::mat4& object_to_view,
                                                  const LodParameters& params);
        struct TraverseResult {
            size_t count = 0;
            bool cancelled = false;
            Stats stats;
        };

        struct WorkItem;

        TraverseResult traverse(const glm::mat4& view_matrix,
                                const LodParameters& params,
                                TraversalScratch& scratch,
                                std::vector<uint32_t>& out_indices,
                                std::vector<uint32_t>& out_logical_indices,
                                std::shared_ptr<const TraversalGuidance> guidance,
                                std::uint64_t cancel_generation = 0) const;
        bool childRangeResident(const PageMapSnapshot& page_maps,
                                uint32_t child_start,
                                uint16_t child_count) const;
        uint32_t renderIndexForNode(const PageMapSnapshot& page_maps,
                                    uint32_t node_index) const;
        uint32_t nodeIndexForRenderIndex(const PageMapSnapshot& page_maps,
                                         uint32_t render_index) const;
        void remapLogicalIndicesForPages(const PageMapSnapshot& page_maps,
                                         const std::vector<uint32_t>& logical_indices,
                                         std::vector<uint32_t>& out_indices) const;
        void remapCurrentSelectionsForPagesLocked(const PageMapSnapshot& page_maps);
        PageMapSnapshot pageMapsSnapshot() const;
        bool publishAsyncResult(const WorkItem& work, const TraverseResult& result);
        void buildLevelsForIndices(const std::vector<uint32_t>& indices,
                                   std::vector<uint32_t>& out_levels) const;
        [[nodiscard]] LodParameters stabilizeParametersLocked(const glm::mat4& view_matrix,
                                                              LodParameters params);
        [[nodiscard]] std::shared_ptr<const TraversalGuidance> buildGuidanceForIndicesLocked(
            const std::vector<uint32_t>& logical_indices) const;
        void publishTargetGuidanceLocked(const std::vector<uint32_t>& logical_indices);
        bool beginTransitionLocked(const std::vector<uint32_t>& new_indices,
                                   const std::vector<uint32_t>& new_logical_indices,
                                   const std::vector<uint32_t>& new_touched_chunks);
        void rebuildTransitionLocked(float t);
        void finishTransitionLocked();
        void setUnityWeightsLocked();
        void workerLoop(std::stop_token stop_token);

        struct WorkItem {
            glm::mat4 view_matrix;
            LodParameters params;
            std::shared_ptr<const TraversalGuidance> guidance;
            uint64_t generation = 0;
        };

        static bool equivalentWork(const WorkItem& a, const WorkItem& b);

        std::vector<LodTreeNode> nodes_;
        std::vector<uint32_t> page_to_chunk_;
        std::vector<uint32_t> chunk_to_page_;
        mutable std::mutex page_maps_mutex_;
        std::vector<uint32_t> full_quality_indices_;
        std::vector<uint32_t> full_quality_logical_indices_;
        std::vector<uint32_t> full_quality_levels_;
        std::vector<uint32_t> full_quality_touched_chunks_;
        std::uint64_t full_quality_hash_ = 0;
        std::vector<uint32_t> selected_indices_;
        std::vector<uint32_t> selected_logical_indices_;
        std::vector<float> selected_weights_;
        mutable std::vector<uint32_t> selected_levels_;
        std::vector<uint32_t> selected_touched_chunks_;
        mutable bool selected_levels_dirty_ = true;
        std::vector<uint32_t> target_logical_indices_;
        std::shared_ptr<const TraversalGuidance> target_guidance_;
        bool transition_active_ = false;
        std::chrono::steady_clock::time_point transition_start_time_{};
        std::chrono::milliseconds transition_duration_{120};
        std::vector<uint32_t> transition_old_indices_;
        std::vector<uint32_t> transition_old_logical_indices_;
        std::vector<uint32_t> transition_new_indices_;
        std::vector<uint32_t> transition_new_logical_indices_;
        std::vector<uint32_t> transition_new_touched_chunks_;
        std::vector<uint32_t> transition_indices_;
        std::vector<uint32_t> transition_logical_indices_;
        std::vector<float> transition_start_weights_;
        std::vector<float> transition_end_weights_;
        TraversalScratch sync_scratch_;
        TraversalScratch async_scratch_;
        mutable std::mutex mutex_;
        std::condition_variable_any cv_;
        std::jthread worker_;
        std::optional<WorkItem> pending_work_;
        bool worker_busy_ = false;
        std::optional<WorkItem> last_requested_work_;
        size_t smoothed_max_splats_ = 0;
        std::chrono::steady_clock::time_point budget_smoothing_time_{};
        TraversalView previous_motion_view_{};
        bool has_previous_motion_view_ = false;
        bool ready_available_ = false;
        std::vector<uint32_t> async_indices_;
        std::vector<uint32_t> async_logical_indices_;
        std::vector<uint32_t> ready_swap_indices_;
        std::vector<uint32_t> ready_swap_logical_indices_;
        std::vector<uint32_t> ready_swap_touched_chunks_;
        uint64_t latest_published_async_generation_ = 0;
        Stats base_stats_;
        Stats current_stats_;
        Stats ready_swap_stats_;
        std::function<void()> ready_callback_;
        uint64_t next_work_generation_ = 0;
        std::atomic<uint64_t> latest_requested_generation_{0};
        std::atomic<uint64_t> min_valid_work_generation_{0};
        uint64_t stats_generation_ = 0;
    };

} // namespace lfs::vis
