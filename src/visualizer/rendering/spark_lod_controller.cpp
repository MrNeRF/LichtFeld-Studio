/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "spark_lod_controller.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>

namespace lfs::vis {
namespace {

constexpr std::size_t kSparkLodChunkSplats = 65'536;

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

std::size_t countTouchedChunks(const std::vector<uint32_t>& indices) {
    if (indices.empty()) {
        return 0;
    }

    std::size_t touched = 0;
    std::size_t last_chunk = std::numeric_limits<std::size_t>::max();
    for (const uint32_t index : indices) {
        const std::size_t chunk = static_cast<std::size_t>(index) / kSparkLodChunkSplats;
        if (chunk != last_chunk) {
            ++touched;
            last_chunk = chunk;
        }
    }
    return touched;
}

} // namespace

SparkLodController::SparkLodController() {
    worker_ = std::jthread([this](std::stop_token stop_token) {
        workerLoop(stop_token);
    });
}

SparkLodController::~SparkLodController() {
    cv_.notify_all();
}

void SparkLodController::attach(const lfs::core::SplatData& data) {
    detach();
    if (!data.lod_tree || !data.lod_tree->has_tree()) {
        return;
    }

    const auto& tree = *data.lod_tree;
    const size_t n = tree.total_nodes();
    if (n == 0 || n > static_cast<size_t>(data.size())) {
        detach();
        return;
    }
    if (tree.child_start.size() < n || tree.child_count.size() < n) {
        detach();
        return;
    }
    nodes_.resize(n);
    full_quality_indices_.clear();
    full_quality_indices_.reserve(n);
    full_quality_hash_ = 0;
    full_quality_touched_chunks_ = 0;

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
        nodes_[i].child_count = tree.child_count[i];
        nodes_[i].lod_level = (i < tree.lod_level.size()) ? tree.lod_level[i] : 0;
        if (nodes_[i].child_count == 0) {
            full_quality_indices_.push_back(static_cast<uint32_t>(i));
        }
    }

    // Compute lod_level via BFS if not provided by loader
    if (tree.lod_level.empty()) {
        std::vector<uint8_t> bfs_level(n, 0);
        std::queue<uint32_t> q;
        q.push(0);
        bfs_level[0] = 0;
        while (!q.empty()) {
            uint32_t idx = q.front(); q.pop();
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

    std::size_t non_leaf_count = 0;
    std::uint16_t max_child_count = 0;
    for (const auto& node : nodes_) {
        if (node.child_count > 0) {
            ++non_leaf_count;
            max_child_count = std::max(max_child_count, node.child_count);
        }
    }
    LOG_INFO(
        "LOD attach: nodes={} non_leaf_nodes={} root_child_count={} max_child_count={}",
        nodes_.size(),
        non_leaf_count,
        nodes_.empty() ? 0u : static_cast<unsigned>(nodes_[0].child_count),
        static_cast<unsigned>(max_child_count));
    full_quality_hash_ = hashSelectedIndices(full_quality_indices_);
    full_quality_touched_chunks_ = countTouchedChunks(full_quality_indices_);

    SparkLodController::Stats stats;
    stats.has_tree = !nodes_.empty();
    stats.lod_opacity_encoded = tree.lod_opacity_encoded;
    stats.model_splats = data.size();
    stats.tree_nodes = nodes_.size();
    stats.non_leaf_nodes = non_leaf_count;
    stats.full_quality_splats = full_quality_indices_.size();
    stats.chunk_splats = kSparkLodChunkSplats;
    stats.chunk_count = (nodes_.size() + kSparkLodChunkSplats - 1) / kSparkLodChunkSplats;
    stats.resident_chunks = stats.chunk_count;
    stats.root_child_count = nodes_.empty() ? 0 : nodes_[0].child_count;
    stats.max_child_count = max_child_count;
    base_stats_ = stats;
    {
        std::scoped_lock lock(mutex_);
        current_stats_ = stats;
        ready_swap_stats_ = {};
        next_work_generation_ = 0;
        latest_requested_generation_ = 0;
        stats_generation_ = 0;
    }
}

void SparkLodController::detach() {
    nodes_.clear();
    full_quality_indices_.clear();
    full_quality_hash_ = 0;
    full_quality_touched_chunks_ = 0;
    selected_indices_.clear();
    {
        std::scoped_lock lock(mutex_);
        pending_work_.reset();
        ready_available_ = false;
        async_indices_.clear();
        ready_swap_indices_.clear();
        base_stats_ = {};
        current_stats_ = {};
        ready_swap_stats_ = {};
        next_work_generation_ = 0;
        latest_requested_generation_ = 0;
        stats_generation_ = 0;
    }
}

float SparkLodController::computePixelScale(uint32_t node_index,
                                             const glm::mat4& view_matrix,
                                             const LodParameters& params) const {
    const auto& node = nodes_[node_index];
    glm::vec4 center_vs = view_matrix * glm::vec4(node.center, 1.0f);
    float radial_dist = glm::length(glm::vec3(center_vs));
    if (radial_dist <= 0.0f) {
        return std::numeric_limits<float>::max();
    }

    const float object_scale = std::isfinite(params.object_scale) && params.object_scale > 0.0f
                                   ? params.object_scale
                                   : 1.0f;
    float pixel_scale = (node.size * object_scale) / radial_dist;

    // Foveation: match Spark's compute_pixel_scale exactly.
    float forward_dot = -center_vs.z;  // dot(center_vs, -z_axis)
    float foveate;
    if (forward_dot <= 0.0f) {
        // Behind camera: apply behind-camera penalty
        foveate = params.behind_camera_penalty;
    } else {
        float inv_distance = 1.0f / radial_dist;
        float dot = forward_dot * inv_distance;
        float inner_degrees = std::clamp(params.cone_inner_degrees, 0.0f, 180.0f);
        float outer_degrees = std::clamp(params.cone_outer_degrees, 0.0f, 180.0f);
        float cone_dot0 = inner_degrees > 0.0f ? std::cos(glm::radians(inner_degrees * 0.5f)) : 1.0f;
        float cone_dot = outer_degrees > 0.0f ? std::cos(glm::radians(outer_degrees * 0.5f)) : 1.0f;
        cone_dot = std::min(cone_dot, cone_dot0);

        if (dot >= cone_dot0) {
            foveate = 1.0f;
        } else if (dot >= cone_dot) {
            float denom = cone_dot0 - cone_dot;
            if (denom < 1.0e-6f) {
                foveate = 1.0f;
            } else {
                float t = (dot - cone_dot) / denom;
                foveate = params.cone_foveation + (1.0f - params.cone_foveation) * t;
            }
        } else {
            if (cone_dot < 1.0e-6f) {
                foveate = params.behind_camera_penalty;
            } else {
                float t = dot / cone_dot;
                foveate = params.behind_camera_penalty + (params.cone_foveation - params.behind_camera_penalty) * t;
            }
        }
    }

    pixel_scale *= foveate;
    return pixel_scale;
}

size_t SparkLodController::update(const glm::mat4& view_matrix, const LodParameters& params) {
    {
        std::scoped_lock lock(mutex_);
        pending_work_.reset();
        ready_available_ = false;
        latest_requested_generation_ = ++next_work_generation_;
    }
    const auto result = traverse(view_matrix, params, selected_indices_);
    {
        std::scoped_lock lock(mutex_);
        current_stats_ = result.stats;
        current_stats_.generation = ++stats_generation_;
    }
    return result.count;
}

void SparkLodController::updateAsync(const glm::mat4& view_matrix, const LodParameters& params) {
    {
        std::scoped_lock lock(mutex_);
        const uint64_t generation = ++next_work_generation_;
        latest_requested_generation_ = generation;
        ready_available_ = false;
        pending_work_ = WorkItem{view_matrix, params, generation};
    }
    cv_.notify_one();
}

bool SparkLodController::swapAsyncResults() {
    std::scoped_lock lock(mutex_);
    if (!ready_available_) {
        return false;
    }
    selected_indices_.swap(ready_swap_indices_);
    current_stats_ = ready_swap_stats_;
    current_stats_.generation = ++stats_generation_;
    ready_available_ = false;
    return true;
}

bool SparkLodController::hasReadyResults() const {
    std::scoped_lock lock(mutex_);
    return ready_available_;
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
        }

        const auto result = traverse(work.view_matrix, work.params, async_indices_);

        {
            std::scoped_lock lock(mutex_);
            if (work.generation == latest_requested_generation_) {
                ready_swap_indices_.swap(async_indices_);
                ready_swap_stats_ = result.stats;
                ready_available_ = true;
            }
        }
    }
}

SparkLodController::TraverseResult SparkLodController::traverse(
    const glm::mat4& view_matrix,
    const LodParameters& params,
    std::vector<uint32_t>& out_indices) const {
    TraverseResult result;
    result.stats = base_stats_;
    auto& stats = result.stats;
    stats.async_result_ready = false;
    stats.budget_limited = false;
    stats.threshold_limited = false;
    stats.output_limited = false;
    stats.selected_splats = 0;
    stats.output_size = 0;
    stats.frontier_size = 0;
    stats.leaf_count = 0;
    stats.touched_chunks = 0;
    stats.min_pixel_scale = 0.0f;
    stats.selection_hash = 0;
    stats.max_splats = params.max_splats;
    stats.pixel_scale_limit = params.pixel_scale_limit;
    stats.lod_render_scale = params.lod_render_scale;
    stats.behind_camera_penalty = params.behind_camera_penalty;
    stats.cone_foveation = params.cone_foveation;
    stats.cone_inner_degrees = params.cone_inner_degrees;
    stats.cone_outer_degrees = params.cone_outer_degrees;

    out_indices.clear();
    if (nodes_.empty() || params.max_splats == 0) {
        stats.output_limited = params.max_splats == 0;
        stats.selection_hash = hashSelectedIndices(out_indices);
        return result;
    }

    out_indices.reserve(params.max_splats);
    std::vector<std::uint8_t> touched_chunks(stats.chunk_count, 0);
    const auto touch_chunk = [&](const std::size_t node_index) {
        const std::size_t chunk_index = node_index / kSparkLodChunkSplats;
        if (chunk_index < touched_chunks.size() && touched_chunks[chunk_index] == 0) {
            touched_chunks[chunk_index] = 1;
            ++stats.touched_chunks;
        }
    };
    const auto touch_child_range = [&](const std::size_t child_start, const std::size_t child_count) {
        if (child_count == 0) {
            return;
        }
        touch_chunk(child_start);
        touch_chunk(child_start + child_count - 1);
    };

    struct HeapNode {
        uint32_t index;
        float pixel_scale;
    };

    struct HeapCompare {
        bool operator()(const HeapNode& a, const HeapNode& b) const {
            return a.pixel_scale < b.pixel_scale; // max-heap: larger scale first
        }
    };

    std::priority_queue<HeapNode, std::vector<HeapNode>, HeapCompare> heap;

    // Seed with root node
    heap.push({0, computePixelScale(0, view_matrix, params)});
    touch_chunk(0);

    // Matches Spark semantics: this tracks output size after draining frontier.
    size_t num_splats = 1;
    float min_pixel_scale = std::numeric_limits<float>::max();

    while (!heap.empty()) {
        const auto top = heap.top();
        min_pixel_scale = std::min(min_pixel_scale, top.pixel_scale);
        if (top.pixel_scale <= params.pixel_scale_limit) {
            stats.threshold_limited = true;
            break;
        }

        heap.pop();
        const auto& node = nodes_[top.index];

        if (node.child_count == 0) {
            // Leaf: output directly.
            out_indices.push_back(top.index);
            ++stats.leaf_count;
            if (out_indices.size() >= params.max_splats) {
                stats.output_limited = true;
                break;
            }
        } else {
            // Internal node: check budget before expanding.
            const size_t new_num_splats = num_splats - 1 + static_cast<size_t>(node.child_count);
            if (new_num_splats > params.max_splats) {
                // Keep this node in the frontier output (Spark behavior).
                heap.push(top);
                stats.budget_limited = true;
                break;
            }

            // Expand children. Children already below threshold go directly to output.
            touch_child_range(node.child_start, node.child_count);
            for (uint32_t c = 0; c < node.child_count; ++c) {
                const uint32_t child_idx = node.child_start + c;
                if (child_idx < nodes_.size()) {
                    const float scale = computePixelScale(child_idx, view_matrix, params);
                    min_pixel_scale = std::min(min_pixel_scale, scale);
                    if (scale <= params.pixel_scale_limit) {
                        out_indices.push_back(child_idx);
                    } else {
                        heap.push({child_idx, scale});
                    }
                }
            }
            num_splats = new_num_splats;
            if (out_indices.size() >= params.max_splats) {
                stats.output_limited = true;
                break;
            }
        }
    }

    stats.output_size = out_indices.size();
    stats.frontier_size = heap.size();

    // Spark drains the whole remaining frontier after the budget/threshold loop.
    // The expansion test above is what keeps this set within the requested cap.
    while (!heap.empty()) {
        out_indices.push_back(heap.top().index);
        heap.pop();
    }

    stats.selected_splats = out_indices.size();
    stats.min_pixel_scale =
        min_pixel_scale == std::numeric_limits<float>::max() ? 0.0f : min_pixel_scale;
    stats.selection_hash = hashSelectedIndices(out_indices);
    {
        std::vector<size_t> counts(256, 0);
        for (const uint32_t index : out_indices) {
            if (index < nodes_.size()) {
                ++counts[nodes_[index].lod_level];
            }
        }
        for (size_t level = 0; level < counts.size(); ++level) {
            if (counts[level] > 0) {
                stats.level_histogram.emplace_back(static_cast<uint8_t>(level), counts[level]);
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

const std::vector<uint32_t>& SparkLodController::fullQualityIndices() const {
    return full_quality_indices_;
}

void SparkLodController::activateFullQualityReference() {
    std::scoped_lock lock(mutex_);
    pending_work_.reset();
    ready_available_ = false;
    latest_requested_generation_ = ++next_work_generation_;
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
    stats.touched_chunks = full_quality_touched_chunks_;
    stats.selection_hash = full_quality_hash_;
    {
        std::vector<size_t> counts(256, 0);
        for (const uint32_t index : full_quality_indices_) {
            if (index < nodes_.size()) {
                ++counts[nodes_[index].lod_level];
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
}

SparkLodController::Stats SparkLodController::stats() const {
    std::scoped_lock lock(mutex_);
    auto stats = current_stats_;
    stats.async_result_ready = ready_available_;
    return stats;
}

} // namespace lfs::vis
