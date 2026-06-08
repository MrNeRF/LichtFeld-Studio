/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/splat_data.hpp"
#include "rendering_types.hpp"
#include <atomic>
#include <functional>
#include <vector>
#include <cstdint>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <utility>
#include <glm/glm.hpp>

namespace lfs::vis {

class SparkLodController {
public:
    struct LodParameters {
        size_t max_splats = DEFAULT_LOD_MAX_SPLATS;
        float pixel_scale_limit = DEFAULT_LOD_PIXEL_SCALE_LIMIT;
        float lod_render_scale = DEFAULT_LOD_RENDER_SCALE;
        float object_scale = 1.0f;
        float behind_camera_penalty = DEFAULT_LOD_BEHIND_CAMERA_FOVEATION;
        float cone_foveation = DEFAULT_LOD_CONE_FOVEATION;
        float cone_inner_degrees = DEFAULT_LOD_CONE_INNER_DEGREES;
        float cone_outer_degrees = DEFAULT_LOD_CONE_OUTER_DEGREES;
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
        bool full_quality_reference = false;
        size_t model_splats = 0;
        size_t tree_nodes = 0;
        size_t non_leaf_nodes = 0;
        size_t full_quality_splats = 0;
        size_t selected_splats = 0;
        size_t max_splats = 0;
        size_t requested_max_splats = 0;
        size_t output_size = 0;
        size_t frontier_size = 0;
        size_t leaf_count = 0;
        size_t chunk_splats = 0;
        size_t chunk_count = 0;
        size_t touched_chunks = 0;
        size_t resident_chunks = 0;
        uint16_t root_child_count = 0;
        uint16_t max_child_count = 0;
        float pixel_scale_limit = 0.0f;
        float min_pixel_scale = 0.0f;
        float lod_render_scale = 1.0f;
        float behind_camera_penalty = 0.0f;
        float cone_foveation = 0.0f;
        float cone_inner_degrees = 0.0f;
        float cone_outer_degrees = 0.0f;
        uint64_t selection_hash = 0;
        uint64_t generation = 0;
        std::vector<std::pair<uint8_t, size_t>> level_histogram;
    };

    SparkLodController();
    ~SparkLodController();

    // Attach to a SplatData that has a lod_tree
    void attach(const lfs::core::SplatData& data);
    void detach();

    // Synchronous traversal. Returns selected count.
    size_t update(const glm::mat4& view_matrix, const LodParameters& params);
    void updateAsync(const glm::mat4& view_matrix, const LodParameters& params);
    bool swapAsyncResults();
    bool hasReadyResults() const;
    void invalidatePendingWork();
    void setReadyCallback(std::function<void()> callback);

    // Accessors
    bool hasTree() const;
    const std::vector<uint32_t>& selectedIndices() const;
    const std::vector<uint32_t>& fullQualityIndices() const;
    void activateFullQualityReference();
    Stats stats() const;

private:
    struct LodTreeNode {
        glm::vec3 center;
        float size;
        uint32_t child_start;
        uint16_t child_count;
        uint8_t lod_level;
    };

    struct TraversalView {
        glm::vec3 origin;
        glm::vec3 forward;
    };

    float computePixelScale(uint32_t node_index,
                           const TraversalView& view,
                           const LodParameters& params) const;
    static TraversalView makeTraversalView(const glm::mat4& object_to_view);
    struct TraverseResult {
        size_t count = 0;
        bool cancelled = false;
        Stats stats;
    };

    struct WorkItem;

    TraverseResult traverse(const glm::mat4& view_matrix,
                            const LodParameters& params,
                            std::vector<uint32_t>& out_indices,
                            std::uint64_t cancel_generation = 0) const;
    TraverseResult traverseDynamic(const glm::mat4& view_matrix,
                                   const LodParameters& params,
                                   std::vector<uint32_t>& out_indices,
                                   std::uint64_t cancel_generation = 0) const;
    bool publishAsyncResult(const WorkItem& work, const TraverseResult& result);
    bool shouldPublishDynamicPreview() const;
    void workerLoop(std::stop_token stop_token);

    struct WorkItem {
        glm::mat4 view_matrix;
        LodParameters params;
        uint64_t generation = 0;
    };

    static bool equivalentWork(const WorkItem& a, const WorkItem& b);

    std::vector<LodTreeNode> nodes_;
    std::vector<uint32_t> full_quality_indices_;
    std::size_t full_quality_touched_chunks_ = 0;
    std::uint64_t full_quality_hash_ = 0;
    std::vector<uint32_t> selected_indices_;
    mutable std::mutex mutex_;
    std::condition_variable_any cv_;
    std::jthread worker_;
    std::optional<WorkItem> pending_work_;
    std::optional<WorkItem> last_requested_work_;
    bool ready_available_ = false;
    std::vector<uint32_t> async_indices_;
    std::vector<uint32_t> ready_swap_indices_;
    Stats base_stats_;
    Stats current_stats_;
    Stats ready_swap_stats_;
    std::function<void()> ready_callback_;
    uint64_t next_work_generation_ = 0;
    std::atomic<uint64_t> latest_requested_generation_{0};
    uint64_t stats_generation_ = 0;
};

} // namespace lfs::vis
