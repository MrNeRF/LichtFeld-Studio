/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "spark_lod_controller.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>

namespace lfs::vis {
    namespace {

        constexpr std::size_t kSparkLodChunkSplats = lfs::core::SplatLodTree::kChunkSplats;
        constexpr std::uint32_t kSparkInvalidPage = lfs::core::SplatLodTree::kInvalidPage;
        constexpr std::uint32_t kInvalidNodeIndex = std::numeric_limits<std::uint32_t>::max();
        constexpr float kLodExpandHysteresis = 1.15f;
        constexpr float kLodCollapseHysteresis = 0.82f;
        constexpr float kLodBudgetFillTargetRatio = 1.0f;
        constexpr float kLodBudgetFillPixelScaleRatio = 0.0f;
        constexpr float kLodBudgetRepairOverheadRatio = 0.20f;
        constexpr float kLodBudgetSmoothingTimeSeconds = 0.16f;
        constexpr std::chrono::milliseconds kLodBudgetSmoothingFallbackDt(16);
        constexpr bool kEnableLodTransitionBlending = true;
        constexpr float kLodTransitionMaxOverheadRatio = 0.20f;
        constexpr float kLodTransitionWeightFloor = 0.04f;
        constexpr std::size_t kLodTransitionMinCutSize = 4096;

        uint64_t hashSelectedIndices(const std::vector<uint32_t>& indices) {
            uint64_t hash = 1469598103934665603ull;
            const auto mix = [&hash](const uint64_t value) {
                hash ^= value;
                hash *= 1099511628211ull;
            };
            const size_t size = indices.size();
            mix(static_cast<uint64_t>(size));
            if (size == 0) {
                return hash;
            }

            constexpr size_t kMaxSamples = 4096;
            const size_t sample_count = std::min(size, kMaxSamples);
            if (sample_count == 1) {
                mix(indices.front());
                return hash;
            }

            for (size_t sample = 0; sample < sample_count; ++sample) {
                const size_t index = (sample * (size - 1)) / (sample_count - 1);
                mix(static_cast<uint64_t>(index));
                mix(static_cast<uint64_t>(indices[index]));
            }
            return hash;
        }

        std::vector<uint32_t> makeSequentialChunks(const std::size_t chunk_count) {
            std::vector<uint32_t> chunks;
            chunks.resize(chunk_count);
            std::iota(chunks.begin(), chunks.end(), 0u);
            return chunks;
        }

        bool almostEqual(const float a, const float b) {
            if (a == b) {
                return true;
            }
            if (!std::isfinite(a) || !std::isfinite(b)) {
                return false;
            }
            constexpr float kAbsEpsilon = 1.0e-6f;
            constexpr float kRelEpsilon = 1.0e-5f;
            const float scale = std::max({1.0f, std::abs(a), std::abs(b)});
            return std::abs(a - b) <= std::max(kAbsEpsilon, kRelEpsilon * scale);
        }

        bool equivalentParams(const SparkLodController::LodParameters& a,
                              const SparkLodController::LodParameters& b) {
            return a.max_splats == b.max_splats &&
                   a.requested_max_splats == b.requested_max_splats &&
                   almostEqual(a.pixel_scale_limit, b.pixel_scale_limit) &&
                   almostEqual(a.lod_render_scale, b.lod_render_scale) &&
                   almostEqual(a.object_scale, b.object_scale) &&
                   almostEqual(a.behind_camera_penalty, b.behind_camera_penalty) &&
                   almostEqual(a.cone_foveation, b.cone_foveation) &&
                   almostEqual(a.cone_inner_degrees, b.cone_inner_degrees) &&
                   almostEqual(a.cone_outer_degrees, b.cone_outer_degrees) &&
                   almostEqual(a.outside_view_foveation, b.outside_view_foveation) &&
                   almostEqual(a.viewport_half_tan_x, b.viewport_half_tan_x) &&
                   almostEqual(a.viewport_half_tan_y, b.viewport_half_tan_y) &&
                   almostEqual(a.ortho_half_width, b.ortho_half_width) &&
                   almostEqual(a.ortho_half_height, b.ortho_half_height) &&
                   almostEqual(a.prefetch_pixel_scale_ratio, b.prefetch_pixel_scale_ratio) &&
                   a.viewport_foveation == b.viewport_foveation &&
                   a.orthographic == b.orthographic;
        }

        bool equivalentMatrix(const glm::mat4& a, const glm::mat4& b) {
            for (int col = 0; col < 4; ++col) {
                for (int row = 0; row < 4; ++row) {
                    if (!almostEqual(a[col][row], b[col][row])) {
                        return false;
                    }
                }
            }
            return true;
        }

    } // namespace

    SparkLodController::SparkLodController() {
        worker_ = std::jthread([this](std::stop_token stop_token) {
            workerLoop(stop_token);
        });
    }

    bool SparkLodController::equivalentWork(const WorkItem& a, const WorkItem& b) {
        return equivalentMatrix(a.view_matrix, b.view_matrix) &&
               equivalentParams(a.params, b.params);
    }

    SparkLodController::~SparkLodController() {
        {
            std::scoped_lock lock(mutex_);
            pending_work_.reset();
            ready_callback_ = nullptr;
            const uint64_t generation = ++next_work_generation_;
            latest_requested_generation_.store(generation, std::memory_order_release);
            min_valid_work_generation_.store(generation, std::memory_order_release);
        }
        worker_.request_stop();
        cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    void SparkLodController::attach(const lfs::core::SplatData& data) {
        detach();
        if (!data.lod_tree || !data.lod_tree->has_tree()) {
            return;
        }

        const auto& tree = *data.lod_tree;
        const size_t n = tree.total_nodes();
        if (n == 0 || n > static_cast<size_t>(data.size())) {
            // Out-of-core model: the GPU selector drives the cut and CPU
            // traversal stays detached, but the model/tree stats are still
            // this controller's to report.
            if (n > 0 && (tree.meta_view.valid() || tree.child_count.size() >= n)) {
                size_t leaf_count = 0;
                if (tree.meta_view.valid()) {
                    leaf_count = tree.meta_view.leaf_count;
                } else {
                    for (size_t i = 0; i < n; ++i) {
                        leaf_count += tree.child_count[i] == 0 ? 1 : 0;
                    }
                }
                Stats stats{};
                stats.available = true;
                stats.has_tree = true;
                stats.lod_opacity_encoded = tree.lod_opacity_encoded;
                stats.model_splats = leaf_count;
                stats.tree_nodes = n;
                stats.non_leaf_nodes = n - leaf_count;
                stats.full_quality_splats = leaf_count;
                stats.chunk_splats = kSparkLodChunkSplats;
                stats.chunk_count = tree.chunk_count();
                base_stats_ = stats;
                std::scoped_lock lock(mutex_);
                current_stats_ = stats;
            }
            return;
        }
        if (tree.child_start.size() < n || tree.child_count.size() < n) {
            detach();
            return;
        }
        nodes_.resize(n);
        full_quality_indices_.clear();
        full_quality_indices_.reserve(n);
        full_quality_logical_indices_.clear();
        full_quality_logical_indices_.reserve(n);
        full_quality_levels_.clear();
        full_quality_levels_.reserve(n);
        full_quality_hash_ = 0;
        full_quality_touched_chunks_.clear();

        const size_t chunk_count = (n + kSparkLodChunkSplats - 1) / kSparkLodChunkSplats;
        {
            std::scoped_lock lock(page_maps_mutex_);
            if (tree.chunk_to_page.empty()) {
                chunk_to_page_.resize(chunk_count);
                page_to_chunk_.resize(chunk_count);
                std::iota(chunk_to_page_.begin(), chunk_to_page_.end(), 0u);
                std::iota(page_to_chunk_.begin(), page_to_chunk_.end(), 0u);
            } else {
                chunk_to_page_ = tree.chunk_to_page;
                if (chunk_to_page_.size() < chunk_count) {
                    chunk_to_page_.resize(chunk_count, kSparkInvalidPage);
                } else if (chunk_to_page_.size() > chunk_count) {
                    chunk_to_page_.resize(chunk_count);
                }

                page_to_chunk_ = tree.page_to_chunk;
                if (page_to_chunk_.empty()) {
                    std::uint32_t max_page = 0;
                    bool has_page = false;
                    for (const std::uint32_t page : chunk_to_page_) {
                        if (page == kSparkInvalidPage) {
                            continue;
                        }
                        max_page = std::max(max_page, page);
                        has_page = true;
                    }

                    if (has_page) {
                        page_to_chunk_.assign(static_cast<std::size_t>(max_page) + 1, kSparkInvalidPage);
                        for (std::size_t chunk = 0; chunk < chunk_to_page_.size(); ++chunk) {
                            const std::uint32_t page = chunk_to_page_[chunk];
                            if (page != kSparkInvalidPage) {
                                page_to_chunk_[page] = static_cast<std::uint32_t>(chunk);
                            }
                        }
                    }
                }
            }
        }
        const PageMapSnapshot page_maps = pageMapsSnapshot();
        const std::size_t resident_chunk_count = page_maps.resident_chunks;

        const bool has_cached_centers = tree.centers.size() >= n;
        const bool has_cached_sizes = tree.sizes.size() >= n;
        const float* means_ptr = nullptr;
        const float* scales_ptr = nullptr;
        lfs::core::Tensor means_cpu;
        lfs::core::Tensor scaling_cpu;
        if (!has_cached_centers) {
            means_cpu = data.means().cpu();
            means_ptr = means_cpu.ptr<float>();
        }
        if (!has_cached_sizes) {
            scaling_cpu = data.scaling_raw().cpu();
            scales_ptr = scaling_cpu.ptr<float>();
        }

        for (size_t i = 0; i < n; ++i) {
            if (has_cached_centers) {
                nodes_[i].center = tree.centers[i];
            } else {
                nodes_[i].center = glm::vec3(
                    means_ptr[i * 3 + 0],
                    means_ptr[i * 3 + 1],
                    means_ptr[i * 3 + 2]);
            }

            if (has_cached_sizes) {
                nodes_[i].size = tree.sizes[i];
            } else {
                float sx = std::exp(scales_ptr[i * 3 + 0]);
                float sy = std::exp(scales_ptr[i * 3 + 1]);
                float sz = std::exp(scales_ptr[i * 3 + 2]);
                nodes_[i].size = 2.0f * std::max({sx, sy, sz});
            }

            nodes_[i].child_start = tree.child_start[i];
            nodes_[i].parent_index = kInvalidNodeIndex;
            nodes_[i].child_count = tree.child_count[i];
            nodes_[i].lod_level = (i < tree.lod_level.size()) ? tree.lod_level[i] : 0;
            if (nodes_[i].child_count == 0) {
                full_quality_indices_.push_back(renderIndexForNode(page_maps, static_cast<uint32_t>(i)));
                full_quality_logical_indices_.push_back(static_cast<uint32_t>(i));
            }
        }

        for (size_t i = 0; i < n; ++i) {
            const auto& node = nodes_[i];
            for (uint32_t c = 0; c < node.child_count; ++c) {
                const uint32_t child_idx = node.child_start + c;
                if (child_idx < n) {
                    nodes_[child_idx].parent_index = static_cast<uint32_t>(i);
                }
            }
        }

        // Compute lod_level via BFS if not provided by loader
        if (tree.lod_level.empty()) {
            std::vector<uint8_t> bfs_level(n, 0);
            std::queue<uint32_t> q;
            q.push(0);
            bfs_level[0] = 0;
            while (!q.empty()) {
                uint32_t idx = q.front();
                q.pop();
                uint8_t level = bfs_level[idx];
                nodes_[idx].lod_level = level;
                for (uint32_t c = 0; c < nodes_[idx].child_count; ++c) {
                    uint32_t child_idx = nodes_[idx].child_start + c;
                    if (child_idx < n) {
                        bfs_level[child_idx] = level + 1;
                        q.push(child_idx);
                    }
                }
            }
        }

        buildLevelsForIndices(full_quality_indices_, full_quality_levels_);

        std::size_t non_leaf_count = 0;
        std::uint16_t max_child_count = 0;
        for (const auto& node : nodes_) {
            if (node.child_count > 0) {
                ++non_leaf_count;
                max_child_count = std::max(max_child_count, node.child_count);
            }
        }
        LOG_INFO(
            "LOD attach: nodes={} non_leaf_nodes={} root_child_count={} max_child_count={} resident_chunks={}/{}",
            nodes_.size(),
            non_leaf_count,
            nodes_.empty() ? 0u : static_cast<unsigned>(nodes_[0].child_count),
            static_cast<unsigned>(max_child_count),
            resident_chunk_count,
            chunk_count);
        full_quality_hash_ = hashSelectedIndices(full_quality_indices_);
        full_quality_touched_chunks_ = makeSequentialChunks(chunk_count);

        SparkLodController::Stats stats;
        stats.has_tree = !nodes_.empty();
        stats.lod_opacity_encoded = tree.lod_opacity_encoded;
        stats.model_splats = data.size();
        stats.tree_nodes = nodes_.size();
        stats.non_leaf_nodes = non_leaf_count;
        stats.full_quality_splats = full_quality_indices_.size();
        stats.chunk_splats = kSparkLodChunkSplats;
        stats.chunk_count = chunk_count;
        stats.resident_chunks = resident_chunk_count;
        stats.root_child_count = nodes_.empty() ? 0 : nodes_[0].child_count;
        stats.max_child_count = max_child_count;
        base_stats_ = stats;
        {
            std::scoped_lock lock(mutex_);
            selected_indices_.clear();
            selected_logical_indices_.clear();
            selected_weights_.clear();
            selected_levels_.clear();
            selected_touched_chunks_.clear();
            target_logical_indices_.clear();
            target_guidance_.reset();
            transition_active_ = false;
            transition_old_indices_.clear();
            transition_old_logical_indices_.clear();
            transition_new_indices_.clear();
            transition_new_logical_indices_.clear();
            transition_new_touched_chunks_.clear();
            transition_indices_.clear();
            transition_logical_indices_.clear();
            transition_start_weights_.clear();
            transition_end_weights_.clear();
            if (!nodes_.empty()) {
                selected_indices_.push_back(renderIndexForNode(page_maps, 0));
                selected_logical_indices_.push_back(0);
                selected_weights_.push_back(1.0f);
                selected_levels_.push_back(nodes_[0].lod_level);
                selected_touched_chunks_.push_back(0);
                target_logical_indices_ = selected_logical_indices_;
                target_guidance_ = buildGuidanceForIndicesLocked(target_logical_indices_);
                selected_levels_dirty_ = false;
                stats.active = true;
                stats.selected_splats = 1;
                stats.output_size = 1;
                stats.frontier_size = 1;
                stats.touched_chunks = 1;
                stats.max_splats = 1;
                stats.selection_hash = hashSelectedIndices(selected_indices_);
            }
            current_stats_ = stats;
            ready_swap_stats_ = {};
            ready_swap_indices_.clear();
            ready_swap_logical_indices_.clear();
            ready_swap_touched_chunks_.clear();
            last_requested_work_.reset();
            latest_published_async_generation_ = 0;
            next_work_generation_ = 0;
            latest_requested_generation_.store(0, std::memory_order_release);
            min_valid_work_generation_.store(0, std::memory_order_release);
            stats_generation_ = 0;
        }
    }

    void SparkLodController::detach() {
        {
            std::unique_lock lock(mutex_);
            pending_work_.reset();
            ready_available_ = false;
            const uint64_t generation = ++next_work_generation_;
            latest_requested_generation_.store(generation, std::memory_order_release);
            min_valid_work_generation_.store(generation, std::memory_order_release);
            cv_.notify_all();
            cv_.wait(lock, [this] { return !worker_busy_; });
        }

        // The worker traverses these vectors without holding mutex_. They are safe to clear only
        // after the cancellation generation has been observed and the active traversal has left.
        nodes_.clear();
        {
            std::scoped_lock lock(page_maps_mutex_);
            page_to_chunk_.clear();
            chunk_to_page_.clear();
        }
        full_quality_indices_.clear();
        full_quality_logical_indices_.clear();
        full_quality_levels_.clear();
        full_quality_hash_ = 0;
        full_quality_touched_chunks_.clear();
        selected_indices_.clear();
        selected_logical_indices_.clear();
        selected_weights_.clear();
        selected_levels_.clear();
        selected_touched_chunks_.clear();
        target_logical_indices_.clear();
        target_guidance_.reset();
        selected_levels_dirty_ = false;
        transition_active_ = false;
        transition_old_indices_.clear();
        transition_old_logical_indices_.clear();
        transition_new_indices_.clear();
        transition_new_logical_indices_.clear();
        transition_new_touched_chunks_.clear();
        transition_indices_.clear();
        transition_logical_indices_.clear();
        transition_start_weights_.clear();
        transition_end_weights_.clear();
        {
            std::scoped_lock lock(mutex_);
            pending_work_.reset();
            ready_available_ = false;
            async_indices_.clear();
            async_logical_indices_.clear();
            ready_swap_indices_.clear();
            ready_swap_logical_indices_.clear();
            ready_swap_touched_chunks_.clear();
            target_logical_indices_.clear();
            target_guidance_.reset();
            last_requested_work_.reset();
            latest_published_async_generation_ = 0;
            smoothed_max_splats_ = 0;
            budget_smoothing_time_ = {};
            has_previous_motion_view_ = false;
            base_stats_ = {};
            current_stats_ = {};
            ready_swap_stats_ = {};
            next_work_generation_ = 0;
            latest_requested_generation_.store(0, std::memory_order_release);
            min_valid_work_generation_.store(0, std::memory_order_release);
            stats_generation_ = 0;
        }
    }

    SparkLodController::TraversalView SparkLodController::makeTraversalView(const glm::mat4& object_to_view) {
        const glm::mat4 view_to_object = glm::inverse(object_to_view);
        glm::vec3 forward = -glm::vec3(view_to_object[2]);
        const float forward_length = glm::length(forward);
        if (forward_length > 1.0e-6f) {
            forward /= forward_length;
        } else {
            forward = {0.0f, 0.0f, -1.0f};
        }

        return {.origin = glm::vec3(view_to_object[3]),
                .forward = forward};
    }

    SparkLodController::PreparedTraversal SparkLodController::prepareTraversal(
        const glm::mat4& object_to_view,
        const LodParameters& params) {
        PreparedTraversal traversal;
        traversal.object_to_view = object_to_view;
        traversal.view = makeTraversalView(object_to_view);
        traversal.object_scale = std::isfinite(params.object_scale) && params.object_scale > 0.0f
                                     ? params.object_scale
                                     : 1.0f;
        traversal.cone_foveation = params.cone_foveation;
        traversal.outside_view_foveation =
            std::clamp(params.outside_view_foveation, 0.0f, 1.0f);
        traversal.viewport_foveation = params.viewport_foveation;
        traversal.behind_camera_penalty =
            std::clamp(params.behind_camera_penalty, 0.0f, 1.0f);
        if (traversal.viewport_foveation) {
            traversal.behind_camera_penalty =
                std::min(traversal.behind_camera_penalty, traversal.outside_view_foveation);
        }
        traversal.orthographic = params.orthographic;
        traversal.viewport_half_tan_x = params.viewport_half_tan_x;
        traversal.viewport_half_tan_y = params.viewport_half_tan_y;
        traversal.ortho_half_width = params.ortho_half_width;
        traversal.ortho_half_height = params.ortho_half_height;

        const float inner_degrees = std::clamp(params.cone_inner_degrees, 0.0f, 180.0f);
        const float outer_degrees = std::clamp(params.cone_outer_degrees, 0.0f, 180.0f);
        traversal.cone_dot0 = inner_degrees > 0.0f
                                  ? std::cos(glm::radians(inner_degrees * 0.5f))
                                  : 1.0f;
        traversal.cone_dot = outer_degrees > 0.0f
                                 ? std::cos(glm::radians(outer_degrees * 0.5f))
                                 : 1.0f;
        traversal.cone_dot = std::min(traversal.cone_dot, traversal.cone_dot0);
        traversal.cone_blend_denominator = traversal.cone_dot0 - traversal.cone_dot;
        traversal.cone_blend_valid = traversal.cone_blend_denominator >= 1.0e-6f;
        traversal.cone_tail_valid = traversal.cone_dot >= 1.0e-6f;
        return traversal;
    }

    SparkLodController::PixelScaleResult SparkLodController::computePixelScale(
        uint32_t node_index,
        const PreparedTraversal& traversal) const {
        PixelScaleResult result;
        const auto& node = nodes_[node_index];
        const glm::vec3 delta = node.center - traversal.view.origin;
        float radial_dist = glm::length(delta);
        if (radial_dist <= 0.0f) {
            result.pixel_scale = std::numeric_limits<float>::max();
            return result;
        }

        const float radius = std::max(node.size * traversal.object_scale, 1.0e-6f);
        float distance_for_scale = radial_dist;
        const float discovery_floor =
            node.child_count > 0 && node.lod_level <= 2 ? 0.05f : 0.0f;

        // Foveation: match Spark's compute_pixel_scale exactly.
        float forward_dot = glm::dot(delta, traversal.view.forward);
        float foveate;
        if (forward_dot <= 0.0f) {
            foveate = traversal.behind_camera_penalty;
        } else {
            float inv_distance = 1.0f / radial_dist;
            float dot = forward_dot * inv_distance;

            if (dot >= traversal.cone_dot0) {
                foveate = 1.0f;
            } else if (dot >= traversal.cone_dot) {
                if (!traversal.cone_blend_valid) {
                    foveate = 1.0f;
                } else {
                    float t = (dot - traversal.cone_dot) / traversal.cone_blend_denominator;
                    foveate = traversal.cone_foveation + (1.0f - traversal.cone_foveation) * t;
                }
            } else {
                if (!traversal.cone_tail_valid) {
                    foveate = traversal.behind_camera_penalty;
                } else {
                    float t = dot / traversal.cone_dot;
                    foveate = traversal.behind_camera_penalty +
                              (traversal.cone_foveation - traversal.behind_camera_penalty) * t;
                }
            }
        }
        foveate = std::max(foveate, discovery_floor);

        if (traversal.viewport_foveation) {
            const glm::vec3 view_center =
                glm::vec3(traversal.object_to_view * glm::vec4(node.center, 1.0f));
            float viewport_foveate = 1.0f;
            const float outside_floor =
                std::max(traversal.outside_view_foveation, discovery_floor);
            if (traversal.orthographic) {
                const float half_width = traversal.ortho_half_width;
                const float half_height = traversal.ortho_half_height;
                if (half_width > 0.0f && half_height > 0.0f) {
                    const float overflow_x = std::abs(view_center.x) - (half_width + radius);
                    const float overflow_y = std::abs(view_center.y) - (half_height + radius);
                    const float overflow = std::max(overflow_x, overflow_y);
                    if (overflow > 0.0f) {
                        result.outside_view = true;
                        const float blend_width = std::max(std::max(half_width, half_height) * 0.08f,
                                                           radius * 0.5f);
                        const float t = 1.0f - std::clamp(overflow / blend_width, 0.0f, 1.0f);
                        viewport_foveate = outside_floor + (1.0f - outside_floor) * t;
                    }
                }
            } else if (traversal.viewport_half_tan_x > 0.0f &&
                       traversal.viewport_half_tan_y > 0.0f) {
                const float view_depth = -view_center.z;
                if (view_depth + radius <= 1.0e-6f) {
                    result.behind_view = true;
                    result.outside_view = true;
                    viewport_foveate = outside_floor;
                } else {
                    const float depth_for_extent = std::max(view_depth, 1.0e-6f);
                    const float half_width = depth_for_extent * traversal.viewport_half_tan_x;
                    const float half_height = depth_for_extent * traversal.viewport_half_tan_y;
                    const float overflow_x = std::abs(view_center.x) - (half_width + radius);
                    const float overflow_y = std::abs(view_center.y) - (half_height + radius);
                    const float overflow = std::max(overflow_x, overflow_y);
                    if (overflow > 0.0f) {
                        result.outside_view = true;
                        const float blend_width = std::max(std::max(half_width, half_height) * 0.08f,
                                                           radius * 0.5f);
                        const float t = 1.0f - std::clamp(overflow / blend_width, 0.0f, 1.0f);
                        viewport_foveate = outside_floor + (1.0f - outside_floor) * t;
                    }
                    if (view_depth > 1.0e-6f) {
                        distance_for_scale = view_depth;
                    }
                }
            }
            result.viewport_throttled = viewport_foveate < 0.999f;
            foveate = std::min(foveate, viewport_foveate);
        }

        result.behind_view = result.behind_view || forward_dot <= 0.0f;
        result.pixel_scale = (radius / distance_for_scale) * foveate;
        return result;
    }

    size_t SparkLodController::update(const glm::mat4& view_matrix, const LodParameters& params) {
        LodParameters stabilized_params;
        std::shared_ptr<const TraversalGuidance> guidance;
        {
            std::scoped_lock lock(mutex_);
            stabilized_params = stabilizeParametersLocked(view_matrix, params);
            guidance = target_guidance_;
            pending_work_.reset();
            ready_available_ = false;
            const uint64_t generation = ++next_work_generation_;
            latest_requested_generation_.store(generation, std::memory_order_release);
            min_valid_work_generation_.store(generation, std::memory_order_release);
            latest_published_async_generation_ = generation;
            last_requested_work_ = WorkItem{view_matrix, stabilized_params, guidance, generation};
        }
        const auto result = traverse(view_matrix,
                                     stabilized_params,
                                     sync_scratch_,
                                     selected_indices_,
                                     selected_logical_indices_,
                                     guidance);
        selected_levels_dirty_ = true;
        {
            std::scoped_lock lock(mutex_);
            transition_active_ = false;
            transition_old_indices_.clear();
            transition_old_logical_indices_.clear();
            transition_new_indices_.clear();
            transition_new_logical_indices_.clear();
            transition_new_touched_chunks_.clear();
            transition_indices_.clear();
            transition_logical_indices_.clear();
            transition_start_weights_.clear();
            transition_end_weights_.clear();
            setUnityWeightsLocked();
            publishTargetGuidanceLocked(selected_logical_indices_);
            selected_touched_chunks_ = sync_scratch_.touched_chunks;
            current_stats_ = result.stats;
            current_stats_.generation = ++stats_generation_;
        }
        return result.count;
    }

    void SparkLodController::updateAsync(const glm::mat4& view_matrix, const LodParameters& params) {
        WorkItem work{};
        work.view_matrix = view_matrix;
        work.params = params;
        {
            std::scoped_lock lock(mutex_);
            work.params = stabilizeParametersLocked(view_matrix, params);
            work.guidance = target_guidance_;
            if (last_requested_work_ && equivalentWork(*last_requested_work_, work)) {
                return;
            }
            const uint64_t generation = ++next_work_generation_;
            work.generation = generation;
            latest_requested_generation_.store(generation, std::memory_order_release);
            pending_work_ = work;
            last_requested_work_ = work;
        }
        cv_.notify_one();
    }

    bool SparkLodController::swapAsyncResults(const bool allow_transition,
                                              const bool remap_physical_indices,
                                              const bool update_cpu_publish_state) {
        std::scoped_lock lock(mutex_);
        if (!ready_available_) {
            return false;
        }
        if (remap_physical_indices) {
            const PageMapSnapshot page_maps = pageMapsSnapshot();
            remapLogicalIndicesForPages(page_maps, ready_swap_logical_indices_, ready_swap_indices_);
        }
        if (transition_active_) {
            finishTransitionLocked();
        }
        const bool transition_started =
            allow_transition &&
            beginTransitionLocked(ready_swap_indices_, ready_swap_logical_indices_, ready_swap_touched_chunks_);
        if (!transition_started) {
            selected_indices_.swap(ready_swap_indices_);
            selected_logical_indices_.swap(ready_swap_logical_indices_);
            if (update_cpu_publish_state) {
                setUnityWeightsLocked();
            } else {
                selected_weights_.clear();
            }
            selected_touched_chunks_.swap(ready_swap_touched_chunks_);
        }
        if (update_cpu_publish_state) {
            publishTargetGuidanceLocked(transition_started
                                            ? transition_new_logical_indices_
                                            : selected_logical_indices_);
        } else {
            target_logical_indices_.clear();
            target_guidance_.reset();
        }
        selected_levels_dirty_ = true;
        current_stats_ = ready_swap_stats_;
        current_stats_.selected_splats = selected_indices_.size();
        current_stats_.output_size = selected_indices_.size();
        current_stats_.touched_chunks = selected_touched_chunks_.size();
        current_stats_.selection_hash = hashSelectedIndices(selected_indices_);
        current_stats_.transition_active = transition_active_;
        current_stats_.generation = ++stats_generation_;
        ready_available_ = false;
        return true;
    }

    bool SparkLodController::hasReadyResults() const {
        std::scoped_lock lock(mutex_);
        return ready_available_;
    }

    void SparkLodController::invalidatePendingWork() {
        std::scoped_lock lock(mutex_);
        pending_work_.reset();
        ready_available_ = false;
        last_requested_work_.reset();
        const uint64_t generation = ++next_work_generation_;
        latest_requested_generation_.store(generation, std::memory_order_release);
        min_valid_work_generation_.store(generation, std::memory_order_release);
        latest_published_async_generation_ = generation;
    }

    void SparkLodController::setReadyCallback(std::function<void()> callback) {
        std::scoped_lock lock(mutex_);
        ready_callback_ = std::move(callback);
    }

    void SparkLodController::applyPageMaps(const std::vector<uint32_t>& page_to_chunk,
                                           const std::vector<uint32_t>& chunk_to_page,
                                           const bool remap_physical_indices) {
        const std::size_t resident_chunks = static_cast<std::size_t>(
            std::count_if(chunk_to_page.begin(), chunk_to_page.end(), [](const std::uint32_t page) {
                return page != kSparkInvalidPage;
            }));
        {
            std::scoped_lock lock(page_maps_mutex_);
            page_to_chunk_ = page_to_chunk;
            chunk_to_page_ = chunk_to_page;
        }
        const PageMapSnapshot page_maps = pageMapsSnapshot();
        {
            std::scoped_lock lock(mutex_);
            base_stats_.resident_chunks = resident_chunks;
            current_stats_.resident_chunks = resident_chunks;
            ready_swap_stats_.resident_chunks = resident_chunks;
            if (remap_physical_indices) {
                remapCurrentSelectionsForPagesLocked(page_maps);
            } else {
                current_stats_.generation = ++stats_generation_;
            }
            // New resident pages can enable a finer cut even when the camera did not
            // change. Keep in-flight/ready work, but force the next async request to
            // traverse against the updated page table.
            last_requested_work_.reset();
        }
    }

    bool SparkLodController::publishAsyncResult(const WorkItem& work, const TraverseResult& result) {
        if (result.cancelled) {
            return false;
        }

        std::function<void()> ready_callback;
        {
            std::scoped_lock lock(mutex_);
            if (work.generation < min_valid_work_generation_.load(std::memory_order_acquire) ||
                work.generation <= latest_published_async_generation_) {
                return false;
            }
            ready_swap_indices_.swap(async_indices_);
            ready_swap_logical_indices_.swap(async_logical_indices_);
            ready_swap_touched_chunks_ = async_scratch_.touched_chunks;
            ready_swap_stats_ = result.stats;
            ready_available_ = true;
            latest_published_async_generation_ = work.generation;
            ready_callback = ready_callback_;
        }
        if (ready_callback) {
            ready_callback();
        }
        return true;
    }

    void SparkLodController::workerLoop(std::stop_token stop_token) {
        while (true) {
            WorkItem work{};
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, stop_token, [this]() {
                    return pending_work_.has_value();
                });
                if (stop_token.stop_requested()) {
                    return;
                }
                work = *pending_work_;
                pending_work_.reset();
                worker_busy_ = true;
            }

            try {
                const auto result = traverse(work.view_matrix,
                                             work.params,
                                             async_scratch_,
                                             async_indices_,
                                             async_logical_indices_,
                                             work.guidance,
                                             work.generation);
                if (!result.cancelled && !stop_token.stop_requested()) {
                    publishAsyncResult(work, result);
                }
            } catch (const std::exception& error) {
                LOG_ERROR("LOD traversal worker failed: {}", error.what());
            } catch (...) {
                LOG_ERROR("LOD traversal worker failed with an unknown error");
            }
            {
                std::scoped_lock lock(mutex_);
                worker_busy_ = false;
            }
            cv_.notify_all();
        }
    }

    SparkLodController::PageMapSnapshot SparkLodController::pageMapsSnapshot() const {
        PageMapSnapshot snapshot;
        std::scoped_lock lock(page_maps_mutex_);
        snapshot.page_to_chunk = page_to_chunk_;
        snapshot.chunk_to_page = chunk_to_page_;
        snapshot.resident_chunks = static_cast<std::size_t>(
            std::count_if(snapshot.chunk_to_page.begin(),
                          snapshot.chunk_to_page.end(),
                          [](const std::uint32_t page) {
                              return page != kSparkInvalidPage;
                          }));
        return snapshot;
    }

    bool SparkLodController::childRangeResident(const PageMapSnapshot& page_maps,
                                                const uint32_t child_start,
                                                const uint16_t child_count) const {
        if (child_count == 0 || page_maps.chunk_to_page.empty()) {
            return true;
        }

        const std::size_t first_chunk = static_cast<std::size_t>(child_start) / kSparkLodChunkSplats;
        const std::uint64_t last_child =
            static_cast<std::uint64_t>(child_start) + static_cast<std::uint64_t>(child_count) - 1u;
        const std::size_t last_chunk = static_cast<std::size_t>(last_child / kSparkLodChunkSplats);
        if (last_chunk >= page_maps.chunk_to_page.size()) {
            return false;
        }

        for (std::size_t chunk = first_chunk; chunk <= last_chunk; ++chunk) {
            if (page_maps.chunk_to_page[chunk] == kSparkInvalidPage) {
                return false;
            }
        }
        return true;
    }

    uint32_t SparkLodController::renderIndexForNode(const PageMapSnapshot& page_maps,
                                                    const uint32_t node_index) const {
        if (page_maps.chunk_to_page.empty()) {
            return node_index;
        }

        const std::size_t chunk = static_cast<std::size_t>(node_index) / kSparkLodChunkSplats;
        if (chunk >= page_maps.chunk_to_page.size()) {
            return node_index;
        }
        const std::uint32_t page = page_maps.chunk_to_page[chunk];
        if (page == kSparkInvalidPage) {
            // Evicted chunk: the raw node index would address the pool tensor
            // out of range and render an unrelated splat. The projection
            // shader skips the invalid sentinel cleanly.
            return kSparkInvalidPage;
        }
        return page * static_cast<std::uint32_t>(kSparkLodChunkSplats) +
               (node_index & static_cast<std::uint32_t>(kSparkLodChunkSplats - 1));
    }

    uint32_t SparkLodController::nodeIndexForRenderIndex(const PageMapSnapshot& page_maps,
                                                         const uint32_t render_index) const {
        if (page_maps.page_to_chunk.empty()) {
            return render_index;
        }

        const std::size_t page = static_cast<std::size_t>(render_index) / kSparkLodChunkSplats;
        if (page >= page_maps.page_to_chunk.size()) {
            return render_index;
        }
        const std::uint32_t chunk = page_maps.page_to_chunk[page];
        if (chunk == kSparkInvalidPage) {
            return render_index;
        }
        return chunk * static_cast<std::uint32_t>(kSparkLodChunkSplats) +
               (render_index & static_cast<std::uint32_t>(kSparkLodChunkSplats - 1));
    }

    void SparkLodController::remapLogicalIndicesForPages(
        const PageMapSnapshot& page_maps,
        const std::vector<uint32_t>& logical_indices,
        std::vector<uint32_t>& out_indices) const {
        out_indices.clear();
        out_indices.reserve(logical_indices.size());
        for (const std::uint32_t logical : logical_indices) {
            out_indices.push_back(renderIndexForNode(page_maps, logical));
        }
    }

    void SparkLodController::remapCurrentSelectionsForPagesLocked(
        const PageMapSnapshot& page_maps) {
        remapLogicalIndicesForPages(page_maps, full_quality_logical_indices_, full_quality_indices_);
        full_quality_hash_ = hashSelectedIndices(full_quality_indices_);

        if (!selected_logical_indices_.empty()) {
            remapLogicalIndicesForPages(page_maps, selected_logical_indices_, selected_indices_);
        }
        if (!ready_swap_logical_indices_.empty()) {
            remapLogicalIndicesForPages(page_maps, ready_swap_logical_indices_, ready_swap_indices_);
            ready_swap_stats_.selection_hash = hashSelectedIndices(ready_swap_indices_);
        }
        if (!transition_old_logical_indices_.empty()) {
            remapLogicalIndicesForPages(page_maps, transition_old_logical_indices_, transition_old_indices_);
        }
        if (!transition_new_logical_indices_.empty()) {
            remapLogicalIndicesForPages(page_maps, transition_new_logical_indices_, transition_new_indices_);
        }
        if (!transition_logical_indices_.empty()) {
            remapLogicalIndicesForPages(page_maps, transition_logical_indices_, transition_indices_);
            if (transition_active_) {
                selected_indices_ = transition_indices_;
            }
        }

        selected_levels_dirty_ = true;
        current_stats_.selected_splats = selected_indices_.size();
        current_stats_.output_size = selected_indices_.size();
        current_stats_.selection_hash = hashSelectedIndices(selected_indices_);
        current_stats_.generation = ++stats_generation_;
    }

    SparkLodController::TraverseResult SparkLodController::traverse(
        const glm::mat4& view_matrix,
        const LodParameters& params,
        TraversalScratch& scratch,
        std::vector<uint32_t>& out_indices,
        std::vector<uint32_t>& out_logical_indices,
        std::shared_ptr<const TraversalGuidance> guidance,
        const std::uint64_t cancel_generation) const {
        TraverseResult result;
        result.stats = base_stats_;
        auto& stats = result.stats;
        stats.async_result_ready = false;
        stats.budget_limited = false;
        stats.threshold_limited = false;
        stats.output_limited = false;
        stats.budget_fill_active = false;
        stats.budget_repair_active = false;
        stats.selected_splats = 0;
        stats.output_size = 0;
        stats.frontier_size = 0;
        stats.leaf_count = 0;
        stats.touched_chunks = 0;
        stats.min_pixel_scale = 0.0f;
        stats.outside_view_nodes = 0;
        stats.behind_view_nodes = 0;
        stats.viewport_throttled_nodes = 0;
        stats.selection_hash = 0;
        stats.max_splats = params.max_splats;
        stats.requested_max_splats =
            params.requested_max_splats == 0 ? params.max_splats : params.requested_max_splats;
        stats.budget_repair_limit = static_cast<std::size_t>(
            std::ceil(static_cast<double>(params.max_splats) *
                      (1.0 + static_cast<double>(kLodBudgetRepairOverheadRatio))));
        stats.pixel_scale_limit = params.pixel_scale_limit;
        stats.budget_fill_pixel_scale_limit =
            params.pixel_scale_limit * kLodBudgetFillPixelScaleRatio;
        stats.lod_render_scale = params.lod_render_scale;
        stats.behind_camera_penalty =
            params.viewport_foveation
                ? std::min(std::clamp(params.behind_camera_penalty, 0.0f, 1.0f),
                           std::clamp(params.outside_view_foveation, 0.0f, 1.0f))
                : params.behind_camera_penalty;
        stats.cone_foveation = params.cone_foveation;
        stats.cone_inner_degrees = params.cone_inner_degrees;
        stats.cone_outer_degrees = params.cone_outer_degrees;

        out_indices.clear();
        std::uint32_t cancel_poll = 0;
        const auto should_cancel = [&]() {
            if (cancel_generation == 0) {
                return false;
            }
            if ((++cancel_poll & 0x3ffu) != 0) {
                return false;
            }
            return cancel_generation < min_valid_work_generation_.load(std::memory_order_acquire);
        };
        const auto cancel_traversal = [&]() {
            out_indices.clear();
            out_logical_indices.clear();
            result.count = 0;
            result.cancelled = true;
            return result;
        };

        if (nodes_.empty() || params.max_splats == 0) {
            stats.output_limited = params.max_splats == 0;
            stats.selection_hash = hashSelectedIndices(out_indices);
            out_logical_indices.clear();
            return result;
        }

        const PageMapSnapshot page_maps = pageMapsSnapshot();
        stats.resident_chunks = page_maps.resident_chunks;
        out_indices.reserve(params.max_splats);
        out_logical_indices.clear();
        out_logical_indices.reserve(params.max_splats);
        const auto emit_node = [&](const uint32_t node_index) {
            out_indices.push_back(renderIndexForNode(page_maps, node_index));
            out_logical_indices.push_back(node_index);
        };
        scratch.touched_chunk_bitmap.assign(stats.chunk_count, 0);
        scratch.touched_chunk_priority.assign(stats.chunk_count, 0.0f);
        scratch.touched_chunks.clear();
        const auto touch_chunk_index = [&](const std::size_t chunk_index, const float priority) {
            if (chunk_index >= scratch.touched_chunk_bitmap.size()) {
                return;
            }
            if (scratch.touched_chunk_bitmap[chunk_index] == 0) {
                scratch.touched_chunk_bitmap[chunk_index] = 1;
                scratch.touched_chunks.push_back(static_cast<uint32_t>(chunk_index));
                ++stats.touched_chunks;
            }
            if (std::isfinite(priority)) {
                scratch.touched_chunk_priority[chunk_index] =
                    std::max(scratch.touched_chunk_priority[chunk_index], priority);
            } else if (priority > 0.0f) {
                scratch.touched_chunk_priority[chunk_index] = priority;
            }
        };
        const auto touch_node_chunk = [&](const std::size_t node_index, const float priority) {
            touch_chunk_index(node_index / kSparkLodChunkSplats, priority);
        };
        const auto touch_child_range = [&](const std::size_t child_start,
                                           const std::size_t child_count,
                                           const float priority) {
            if (child_count == 0) {
                return;
            }
            const std::size_t first_chunk = child_start / kSparkLodChunkSplats;
            const std::size_t last_chunk = (child_start + child_count - 1) / kSparkLodChunkSplats;
            for (std::size_t chunk = first_chunk; chunk <= last_chunk; ++chunk) {
                touch_chunk_index(chunk, priority);
            }
        };
        const auto exact_selected = [&](const std::uint32_t node_index) {
            return guidance &&
                   node_index < guidance->exact_selected.size() &&
                   guidance->exact_selected[node_index] != 0;
        };
        const auto selected_subtree = [&](const std::uint32_t node_index) {
            return guidance &&
                   node_index < guidance->selected_subtree.size() &&
                   guidance->selected_subtree[node_index] != 0;
        };
        const auto stop_limit_for_node = [&](const std::uint32_t node_index) {
            if (exact_selected(node_index)) {
                return params.pixel_scale_limit * kLodExpandHysteresis;
            }
            if (selected_subtree(node_index)) {
                return params.pixel_scale_limit * kLodCollapseHysteresis;
            }
            return params.pixel_scale_limit;
        };
        const std::size_t budget_fill_target =
            static_cast<std::size_t>(std::ceil(static_cast<double>(params.max_splats) *
                                               static_cast<double>(kLodBudgetFillTargetRatio)));
        const auto should_stop_for_threshold = [&](const std::uint32_t node_index,
                                                   const float pixel_scale,
                                                   const std::size_t projected_splats) {
            const float stop_limit = stop_limit_for_node(node_index);
            if (pixel_scale > stop_limit) {
                return false;
            }

            if (projected_splats < budget_fill_target) {
                stats.budget_fill_active = true;
                return false;
            }

            stats.threshold_limited = true;
            return true;
        };
        const auto touch_predictive_children = [&](const LodTreeNode& node, const float pixel_scale) {
            if (node.child_count == 0) {
                return;
            }
            const float ratio = std::clamp(params.prefetch_pixel_scale_ratio, 0.1f, 1.0f);
            if (pixel_scale > params.pixel_scale_limit * ratio) {
                touch_child_range(node.child_start, node.child_count, pixel_scale);
            }
        };
        const PreparedTraversal traversal = prepareTraversal(view_matrix, params);
        const auto compute_node_pixel_scale = [&](const std::uint32_t node_index) {
            const PixelScaleResult scale = computePixelScale(node_index, traversal);
            stats.outside_view_nodes += scale.outside_view ? 1u : 0u;
            stats.behind_view_nodes += scale.behind_view ? 1u : 0u;
            stats.viewport_throttled_nodes += scale.viewport_throttled ? 1u : 0u;
            return scale;
        };

        auto& heap = scratch.frontier_heap;
        heap.clear();
        heap.reserve(std::min(params.max_splats, nodes_.size()));
        std::vector<HeapNode> child_candidates;
        child_candidates.reserve(base_stats_.max_child_count);
        constexpr HeapCompare heap_compare{};
        const auto push_heap_node = [&](const HeapNode node) {
            heap.push_back(node);
            std::push_heap(heap.begin(), heap.end(), heap_compare);
        };
        const auto pop_heap_node = [&]() {
            std::pop_heap(heap.begin(), heap.end(), heap_compare);
            const HeapNode node = heap.back();
            heap.pop_back();
            return node;
        };

        // Seed with root node
        const PixelScaleResult root_scale = compute_node_pixel_scale(0);
        push_heap_node({.index = 0,
                        .pixel_scale = root_scale.pixel_scale});
        touch_node_chunk(0, std::numeric_limits<float>::infinity());

        // Matches Spark semantics: this tracks output size after draining frontier.
        size_t num_splats = 1;
        float min_pixel_scale = std::numeric_limits<float>::max();

        while (!heap.empty()) {
            if (should_cancel()) {
                return cancel_traversal();
            }
            const auto top = heap.front();
            min_pixel_scale = std::min(min_pixel_scale, top.pixel_scale);
            if (should_stop_for_threshold(top.index, top.pixel_scale, num_splats)) {
                break;
            }

            const auto current = pop_heap_node();
            const auto& node = nodes_[current.index];

            if (node.child_count == 0) {
                // Leaf: output directly.
                emit_node(current.index);
                ++stats.leaf_count;
                continue;
            } else {
                // Classify children before the budget test. View/frustum foveation
                // changes priority, but residency and budget decide whether we can refine.
                touch_child_range(node.child_start, node.child_count, current.pixel_scale);
                if (!childRangeResident(page_maps, node.child_start, node.child_count)) {
                    emit_node(current.index);
                    continue;
                }
                child_candidates.clear();
                for (uint32_t c = 0; c < node.child_count; ++c) {
                    if (should_cancel()) {
                        return cancel_traversal();
                    }
                    const uint32_t child_idx = node.child_start + c;
                    if (child_idx < nodes_.size()) {
                        const PixelScaleResult child_scale = compute_node_pixel_scale(child_idx);
                        min_pixel_scale = std::min(min_pixel_scale, child_scale.pixel_scale);
                        touch_node_chunk(child_idx, child_scale.pixel_scale);
                        child_candidates.push_back({.index = child_idx,
                                                    .pixel_scale = child_scale.pixel_scale});
                    }
                }
                const size_t expanded_num_splats = num_splats - 1 + child_candidates.size();
                if (expanded_num_splats > params.max_splats) {
                    if (expanded_num_splats <= stats.budget_repair_limit) {
                        stats.budget_repair_active = true;
                    } else {
                        emit_node(current.index);
                        touch_predictive_children(node, current.pixel_scale);
                        stats.budget_limited = true;
                        break;
                    }
                }
                if (child_candidates.empty()) {
                    num_splats -= 1;
                    continue;
                }
                for (const HeapNode& child : child_candidates) {
                    if (should_stop_for_threshold(child.index, child.pixel_scale, expanded_num_splats)) {
                        emit_node(child.index);
                    } else {
                        push_heap_node(child);
                    }
                }
                num_splats = expanded_num_splats;
            }
        }

        stats.output_size = out_indices.size();
        stats.frontier_size = heap.size();

        // Spark drains the whole remaining frontier after the budget/threshold loop.
        // The expansion test above is what keeps this set within the requested cap.
        while (!heap.empty()) {
            if (should_cancel()) {
                return cancel_traversal();
            }
            emit_node(pop_heap_node().index);
        }

        stats.selected_splats = out_indices.size();
        stats.min_pixel_scale =
            min_pixel_scale == std::numeric_limits<float>::max() ? 0.0f : min_pixel_scale;
        std::stable_sort(scratch.touched_chunks.begin(),
                         scratch.touched_chunks.end(),
                         [&](const std::uint32_t a, const std::uint32_t b) {
                             const float pa =
                                 a < scratch.touched_chunk_priority.size()
                                     ? scratch.touched_chunk_priority[a]
                                     : 0.0f;
                             const float pb =
                                 b < scratch.touched_chunk_priority.size()
                                     ? scratch.touched_chunk_priority[b]
                                     : 0.0f;
                             if (pa == pb) {
                                 return a < b;
                             }
                             return pa > pb;
                         });
        stats.selection_hash = hashSelectedIndices(out_indices);
        {
            scratch.level_counts.assign(256, 0);
            for (const uint32_t index : out_indices) {
                if (should_cancel()) {
                    return cancel_traversal();
                }
                const uint32_t node_index = nodeIndexForRenderIndex(page_maps, index);
                if (node_index < nodes_.size()) {
                    ++scratch.level_counts[nodes_[node_index].lod_level];
                }
            }
            for (size_t level = 0; level < scratch.level_counts.size(); ++level) {
                if (scratch.level_counts[level] > 0) {
                    stats.level_histogram.emplace_back(static_cast<uint8_t>(level),
                                                       scratch.level_counts[level]);
                }
            }
        }
        result.count = out_indices.size();
        return result;
    }

    bool SparkLodController::hasTree() const {
        return !nodes_.empty();
    }

    const std::vector<uint32_t>& SparkLodController::selectedIndices() const {
        return selected_indices_;
    }

    const std::vector<uint32_t>& SparkLodController::selectedLogicalIndices() const {
        return selected_logical_indices_;
    }

    const std::vector<float>& SparkLodController::selectedWeights() const {
        return selected_weights_;
    }

    const std::vector<uint32_t>& SparkLodController::selectedLevels() const {
        if (selected_levels_dirty_) {
            buildLevelsForIndices(selected_indices_, selected_levels_);
            selected_levels_dirty_ = false;
        }
        return selected_levels_;
    }

    const std::vector<uint32_t>& SparkLodController::fullQualityIndices() const {
        return full_quality_indices_;
    }

    const std::vector<uint32_t>& SparkLodController::fullQualityLogicalIndices() const {
        return full_quality_logical_indices_;
    }

    const std::vector<uint32_t>& SparkLodController::fullQualityLevels() const {
        return full_quality_levels_;
    }

    bool SparkLodController::transitionActive() const {
        std::scoped_lock lock(mutex_);
        return transition_active_;
    }

    std::vector<uint32_t> SparkLodController::touchedChunks() const {
        std::scoped_lock lock(mutex_);
        return selected_touched_chunks_;
    }

    uint64_t SparkLodController::selectionHash() const {
        std::scoped_lock lock(mutex_);
        return current_stats_.selection_hash;
    }

    uint64_t SparkLodController::statsGeneration() const {
        std::scoped_lock lock(mutex_);
        return current_stats_.generation;
    }

    bool SparkLodController::pageMappingActive() const {
        std::scoped_lock lock(page_maps_mutex_);
        for (std::size_t chunk = 0; chunk < chunk_to_page_.size(); ++chunk) {
            const std::uint32_t page = chunk_to_page_[chunk];
            if (page == kSparkInvalidPage || page != chunk) {
                return true;
            }
        }
        return false;
    }

    void SparkLodController::buildLevelsForIndices(const std::vector<uint32_t>& indices,
                                                   std::vector<uint32_t>& out_levels) const {
        const PageMapSnapshot page_maps = pageMapsSnapshot();
        out_levels.clear();
        out_levels.reserve(indices.size());
        for (const uint32_t index : indices) {
            const uint32_t node_index = nodeIndexForRenderIndex(page_maps, index);
            out_levels.push_back(node_index < nodes_.size() ? nodes_[node_index].lod_level : 0u);
        }
    }

    SparkLodController::LodParameters SparkLodController::stabilizeParametersLocked(
        const glm::mat4& view_matrix,
        LodParameters params) {
        const size_t requested_max = params.max_splats;
        params.requested_max_splats = requested_max;

        const auto now = std::chrono::steady_clock::now();
        if (smoothed_max_splats_ == 0 || budget_smoothing_time_.time_since_epoch().count() == 0) {
            smoothed_max_splats_ = requested_max;
            budget_smoothing_time_ = now;
        } else if (smoothed_max_splats_ != requested_max) {
            auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - budget_smoothing_time_);
            if (dt.count() <= 0) {
                dt = kLodBudgetSmoothingFallbackDt;
            }
            budget_smoothing_time_ = now;
            const float seconds = static_cast<float>(dt.count()) * 0.001f;
            const float alpha = 1.0f - std::exp(-seconds / kLodBudgetSmoothingTimeSeconds);
            const double smoothed =
                static_cast<double>(smoothed_max_splats_) +
                (static_cast<double>(requested_max) - static_cast<double>(smoothed_max_splats_)) *
                    static_cast<double>(std::clamp(alpha, 0.0f, 1.0f));
            const size_t next = static_cast<size_t>(std::llround(std::max(0.0, smoothed)));
            const size_t diff = next > requested_max ? next - requested_max : requested_max - next;
            smoothed_max_splats_ = diff <= 4096 ? requested_max : std::max<size_t>(1, next);
        } else {
            budget_smoothing_time_ = now;
        }
        params.max_splats = smoothed_max_splats_;

        const TraversalView motion_view = makeTraversalView(view_matrix);
        float motion_score = 0.0f;
        if (has_previous_motion_view_) {
            const float angular =
                1.0f - std::clamp(glm::dot(previous_motion_view_.forward, motion_view.forward), -1.0f, 1.0f);
            const float translation = glm::length(motion_view.origin - previous_motion_view_.origin);
            motion_score = angular * 8.0f + std::min(translation * 0.2f, 1.0f);
        }
        previous_motion_view_ = motion_view;
        has_previous_motion_view_ = true;

        params.prefetch_pixel_scale_ratio =
            motion_score > 0.02f ? 0.45f : DEFAULT_LOD_PREFETCH_PIXEL_SCALE_RATIO;
        return params;
    }

    std::shared_ptr<const SparkLodController::TraversalGuidance>
    SparkLodController::buildGuidanceForIndicesLocked(
        const std::vector<uint32_t>& logical_indices) const {
        if (nodes_.empty() || logical_indices.empty()) {
            return {};
        }

        auto guidance = std::make_shared<TraversalGuidance>();
        guidance->exact_selected.assign(nodes_.size(), 0);
        guidance->selected_subtree.assign(nodes_.size(), 0);

        for (const std::uint32_t logical_index : logical_indices) {
            if (logical_index >= nodes_.size()) {
                continue;
            }
            guidance->exact_selected[logical_index] = 1;
            std::uint32_t cursor = logical_index;
            while (cursor != kInvalidNodeIndex && cursor < nodes_.size()) {
                if (guidance->selected_subtree[cursor] != 0) {
                    break;
                }
                guidance->selected_subtree[cursor] = 1;
                cursor = nodes_[cursor].parent_index;
            }
        }
        return guidance;
    }

    void SparkLodController::publishTargetGuidanceLocked(
        const std::vector<uint32_t>& logical_indices) {
        target_logical_indices_ = logical_indices;
        target_guidance_ = buildGuidanceForIndicesLocked(target_logical_indices_);
    }

    void SparkLodController::setUnityWeightsLocked() {
        selected_weights_.assign(selected_indices_.size(), 1.0f);
    }

    bool SparkLodController::beginTransitionLocked(const std::vector<uint32_t>& new_indices,
                                                   const std::vector<uint32_t>& new_logical_indices,
                                                   const std::vector<uint32_t>& new_touched_chunks) {
        if (!kEnableLodTransitionBlending) {
            transition_active_ = false;
            return false;
        }

        if (selected_indices_.empty() ||
            new_indices.empty() ||
            selected_logical_indices_ == new_logical_indices ||
            selected_logical_indices_.size() != selected_indices_.size() ||
            new_logical_indices.size() != new_indices.size()) {
            transition_active_ = false;
            return false;
        }

        const std::size_t base_count = std::max(selected_indices_.size(), new_indices.size());
        const std::size_t smaller_count = std::min(selected_indices_.size(), new_indices.size());
        if (smaller_count < kLodTransitionMinCutSize) {
            transition_active_ = false;
            return false;
        }
        const std::size_t extra_budget = static_cast<std::size_t>(
            std::ceil(static_cast<double>(base_count) * static_cast<double>(kLodTransitionMaxOverheadRatio)));
        const std::size_t transition_cap =
            base_count > std::numeric_limits<std::size_t>::max() - extra_budget
                ? std::numeric_limits<std::size_t>::max()
                : base_count + extra_budget;

        const auto old_guidance = buildGuidanceForIndicesLocked(selected_logical_indices_);
        const auto new_guidance = buildGuidanceForIndicesLocked(new_logical_indices);
        if (!old_guidance || !new_guidance) {
            transition_active_ = false;
            return false;
        }

        const auto exact_selected = [](const std::shared_ptr<const TraversalGuidance>& guidance,
                                       const std::uint32_t node_index) {
            return guidance &&
                   node_index < guidance->exact_selected.size() &&
                   guidance->exact_selected[node_index] != 0;
        };
        const auto has_selected_relation = [this, &exact_selected](
                                               const std::shared_ptr<const TraversalGuidance>& guidance,
                                               const std::uint32_t node_index) {
            if (!guidance || node_index >= nodes_.size()) {
                return false;
            }
            if (node_index < guidance->selected_subtree.size() &&
                guidance->selected_subtree[node_index] != 0) {
                return true;
            }
            std::uint32_t cursor = nodes_[node_index].parent_index;
            while (cursor != kInvalidNodeIndex && cursor < nodes_.size()) {
                if (exact_selected(guidance, cursor)) {
                    return true;
                }
                cursor = nodes_[cursor].parent_index;
            }
            return false;
        };

        transition_old_indices_ = selected_indices_;
        transition_old_logical_indices_ = selected_logical_indices_;
        transition_new_indices_ = new_indices;
        transition_new_logical_indices_ = new_logical_indices;
        transition_new_touched_chunks_ = new_touched_chunks;
        transition_indices_.clear();
        transition_logical_indices_.clear();
        transition_start_weights_.clear();
        transition_end_weights_.clear();
        const std::size_t reserve_count = std::min(
            transition_cap,
            selected_indices_.size() > std::numeric_limits<std::size_t>::max() - new_indices.size()
                ? std::numeric_limits<std::size_t>::max()
                : selected_indices_.size() + new_indices.size());
        transition_indices_.reserve(reserve_count);
        transition_logical_indices_.reserve(reserve_count);
        transition_start_weights_.reserve(reserve_count);
        transition_end_weights_.reserve(reserve_count);

        bool has_weight_transition = false;
        const auto append_transition_node = [&](const std::uint32_t render_index,
                                                const std::uint32_t logical,
                                                const float start_weight,
                                                const float end_weight) {
            transition_indices_.push_back(render_index);
            transition_logical_indices_.push_back(logical);
            transition_start_weights_.push_back(start_weight);
            transition_end_weights_.push_back(end_weight);
            has_weight_transition = has_weight_transition ||
                                    std::abs(start_weight - end_weight) > 1.0e-4f;
        };

        for (std::size_t i = 0; i < new_indices.size(); ++i) {
            const std::uint32_t logical = new_logical_indices[i];
            const bool already_selected = exact_selected(old_guidance, logical);
            const bool related_to_old = already_selected || has_selected_relation(old_guidance, logical);
            append_transition_node(new_indices[i],
                                   logical,
                                   already_selected ? 1.0f : (related_to_old ? kLodTransitionWeightFloor : 1.0f),
                                   1.0f);
        }
        for (std::size_t i = 0; i < selected_indices_.size() && transition_indices_.size() < transition_cap; ++i) {
            const std::uint32_t logical = selected_logical_indices_[i];
            if (exact_selected(new_guidance, logical) ||
                !has_selected_relation(new_guidance, logical)) {
                continue;
            }
            append_transition_node(selected_indices_[i],
                                   logical,
                                   1.0f,
                                   kLodTransitionWeightFloor);
        }

        if (!has_weight_transition) {
            transition_active_ = false;
            transition_indices_.clear();
            transition_logical_indices_.clear();
            transition_start_weights_.clear();
            transition_end_weights_.clear();
            return false;
        }

        std::vector<std::uint8_t> touched_bitmap(base_stats_.chunk_count, 0);
        std::vector<std::uint32_t> transition_touched;
        transition_touched.reserve(selected_touched_chunks_.size() +
                                   new_touched_chunks.size() +
                                   transition_logical_indices_.size());
        const auto add_chunk = [&](const std::uint32_t chunk) {
            if (chunk >= touched_bitmap.size() || touched_bitmap[chunk] != 0) {
                return;
            }
            touched_bitmap[chunk] = 1;
            transition_touched.push_back(chunk);
        };
        for (const std::uint32_t chunk : selected_touched_chunks_) {
            add_chunk(chunk);
        }
        for (const std::uint32_t chunk : new_touched_chunks) {
            add_chunk(chunk);
        }
        for (const std::uint32_t logical : transition_logical_indices_) {
            add_chunk(logical / static_cast<std::uint32_t>(kSparkLodChunkSplats));
        }
        selected_touched_chunks_ = std::move(transition_touched);

        transition_start_time_ = std::chrono::steady_clock::now();
        transition_active_ = true;
        rebuildTransitionLocked(0.0f);
        return true;
    }

    void SparkLodController::rebuildTransitionLocked(const float t) {
        if (!transition_active_) {
            return;
        }

        const float linear_new_weight = std::clamp(t, 0.0f, 1.0f);
        const float blend = linear_new_weight * linear_new_weight * (3.0f - 2.0f * linear_new_weight);
        selected_indices_ = transition_indices_;
        selected_logical_indices_ = transition_logical_indices_;
        selected_weights_.clear();
        selected_weights_.reserve(transition_start_weights_.size());
        for (std::size_t i = 0; i < transition_start_weights_.size(); ++i) {
            const float start = transition_start_weights_[i];
            selected_weights_.push_back(start + (transition_end_weights_[i] - start) * blend);
        }
    }

    void SparkLodController::finishTransitionLocked() {
        if (!transition_active_) {
            return;
        }
        selected_indices_ = transition_new_indices_;
        selected_logical_indices_ = transition_new_logical_indices_;
        transition_active_ = false;
        transition_old_indices_.clear();
        transition_old_logical_indices_.clear();
        transition_new_indices_.clear();
        transition_new_logical_indices_.clear();
        selected_touched_chunks_ = transition_new_touched_chunks_;
        transition_new_touched_chunks_.clear();
        transition_indices_.clear();
        transition_logical_indices_.clear();
        transition_start_weights_.clear();
        transition_end_weights_.clear();
        setUnityWeightsLocked();
    }

    void SparkLodController::advanceTransition() {
        std::scoped_lock lock(mutex_);
        if (!transition_active_) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - transition_start_time_);
        const float t = transition_duration_.count() <= 0
                            ? 1.0f
                            : static_cast<float>(elapsed.count()) / static_cast<float>(transition_duration_.count());
        if (t >= 1.0f) {
            finishTransitionLocked();
        } else {
            rebuildTransitionLocked(t);
        }
        selected_levels_dirty_ = true;
        current_stats_.selected_splats = selected_indices_.size();
        current_stats_.output_size = selected_indices_.size();
        current_stats_.touched_chunks = selected_touched_chunks_.size();
        current_stats_.selection_hash = hashSelectedIndices(selected_indices_);
        current_stats_.transition_active = transition_active_;
        current_stats_.generation = ++stats_generation_;
    }

    void SparkLodController::activateFullQualityReference() {
        std::scoped_lock lock(mutex_);
        pending_work_.reset();
        last_requested_work_.reset();
        ready_available_ = false;
        const uint64_t generation = ++next_work_generation_;
        latest_requested_generation_.store(generation, std::memory_order_release);
        min_valid_work_generation_.store(generation, std::memory_order_release);
        latest_published_async_generation_ = generation;
        if (current_stats_.full_quality_reference &&
            current_stats_.selected_splats == full_quality_indices_.size()) {
            return;
        }

        Stats stats = base_stats_;
        stats.active = true;
        stats.enabled = false;
        stats.full_quality_reference = true;
        stats.selected_splats = full_quality_indices_.size();
        stats.output_size = full_quality_indices_.size();
        stats.leaf_count = full_quality_indices_.size();
        stats.max_splats = full_quality_indices_.size();
        stats.touched_chunks = full_quality_touched_chunks_.size();
        stats.selection_hash = full_quality_hash_;
        {
            const PageMapSnapshot page_maps = pageMapsSnapshot();
            std::vector<size_t> counts(256, 0);
            for (const uint32_t index : full_quality_indices_) {
                const uint32_t node_index = nodeIndexForRenderIndex(page_maps, index);
                if (node_index < nodes_.size()) {
                    ++counts[nodes_[node_index].lod_level];
                }
            }
            for (size_t level = 0; level < counts.size(); ++level) {
                if (counts[level] > 0) {
                    stats.level_histogram.emplace_back(static_cast<uint8_t>(level), counts[level]);
                }
            }
        }

        current_stats_ = stats;
        current_stats_.generation = ++stats_generation_;
        transition_active_ = false;
        transition_old_indices_.clear();
        transition_old_logical_indices_.clear();
        transition_new_indices_.clear();
        transition_new_logical_indices_.clear();
        selected_indices_ = full_quality_indices_;
        selected_logical_indices_ = full_quality_logical_indices_;
        setUnityWeightsLocked();
        publishTargetGuidanceLocked(selected_logical_indices_);
        selected_touched_chunks_ = full_quality_touched_chunks_;
    }

    SparkLodController::Stats SparkLodController::stats() const {
        std::scoped_lock lock(mutex_);
        auto stats = current_stats_;
        stats.async_result_ready = ready_available_;
        return stats;
    }

} // namespace lfs::vis
