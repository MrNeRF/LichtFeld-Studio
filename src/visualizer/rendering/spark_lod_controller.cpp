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
           almostEqual(a.pixel_scale_limit, b.pixel_scale_limit) &&
           almostEqual(a.lod_render_scale, b.lod_render_scale) &&
           almostEqual(a.object_scale, b.object_scale) &&
           almostEqual(a.behind_camera_penalty, b.behind_camera_penalty) &&
           almostEqual(a.cone_foveation, b.cone_foveation) &&
           almostEqual(a.cone_inner_degrees, b.cone_inner_degrees) &&
           almostEqual(a.cone_outer_degrees, b.cone_outer_degrees);
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
        latest_requested_generation_.store(++next_work_generation_, std::memory_order_release);
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
        nodes_[i].child_count = tree.child_count[i];
        nodes_[i].lod_level = (i < tree.lod_level.size()) ? tree.lod_level[i] : 0;
        if (nodes_[i].child_count == 0) {
            full_quality_indices_.push_back(renderIndexForNode(page_maps, static_cast<uint32_t>(i)));
            full_quality_logical_indices_.push_back(static_cast<uint32_t>(i));
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
        selected_levels_.clear();
        selected_touched_chunks_.clear();
        if (!nodes_.empty()) {
            selected_indices_.push_back(renderIndexForNode(page_maps, 0));
            selected_logical_indices_.push_back(0);
            selected_levels_.push_back(nodes_[0].lod_level);
            selected_touched_chunks_.push_back(0);
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
        next_work_generation_ = 0;
        latest_requested_generation_.store(0, std::memory_order_release);
        stats_generation_ = 0;
    }
}

void SparkLodController::detach() {
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
    selected_levels_.clear();
    selected_touched_chunks_.clear();
    selected_levels_dirty_ = false;
    {
        std::scoped_lock lock(mutex_);
        pending_work_.reset();
        ready_available_ = false;
        async_indices_.clear();
        async_logical_indices_.clear();
        ready_swap_indices_.clear();
        ready_swap_logical_indices_.clear();
        ready_swap_touched_chunks_.clear();
        last_requested_work_.reset();
        base_stats_ = {};
        current_stats_ = {};
        ready_swap_stats_ = {};
        next_work_generation_ = 0;
        latest_requested_generation_.store(0, std::memory_order_release);
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
    traversal.view = makeTraversalView(object_to_view);
    traversal.object_scale = std::isfinite(params.object_scale) && params.object_scale > 0.0f
                                 ? params.object_scale
                                 : 1.0f;
    traversal.behind_camera_penalty = params.behind_camera_penalty;
    traversal.cone_foveation = params.cone_foveation;

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

float SparkLodController::computePixelScale(uint32_t node_index,
                                             const PreparedTraversal& traversal) const {
    const auto& node = nodes_[node_index];
    const glm::vec3 delta = node.center - traversal.view.origin;
    float radial_dist = glm::length(delta);
    if (radial_dist <= 0.0f) {
        return std::numeric_limits<float>::max();
    }

    float pixel_scale = (node.size * traversal.object_scale) / radial_dist;

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

    pixel_scale *= foveate;
    return pixel_scale;
}

size_t SparkLodController::update(const glm::mat4& view_matrix, const LodParameters& params) {
    {
        std::scoped_lock lock(mutex_);
        pending_work_.reset();
        ready_available_ = false;
        const uint64_t generation = ++next_work_generation_;
        latest_requested_generation_.store(generation, std::memory_order_release);
        last_requested_work_ = WorkItem{view_matrix, params, generation};
    }
    const auto result = traverse(view_matrix, params, sync_scratch_, selected_indices_, selected_logical_indices_);
    selected_levels_dirty_ = true;
    {
        std::scoped_lock lock(mutex_);
        selected_touched_chunks_ = sync_scratch_.touched_chunks;
        current_stats_ = result.stats;
        current_stats_.generation = ++stats_generation_;
    }
    return result.count;
}

void SparkLodController::updateAsync(const glm::mat4& view_matrix, const LodParameters& params) {
    WorkItem work{view_matrix, params, 0};
    {
        std::scoped_lock lock(mutex_);
        if (last_requested_work_ && equivalentWork(*last_requested_work_, work)) {
            return;
        }
        const uint64_t generation = ++next_work_generation_;
        work.generation = generation;
        latest_requested_generation_.store(generation, std::memory_order_release);
        ready_available_ = false;
        pending_work_ = work;
        last_requested_work_ = work;
    }
    cv_.notify_one();
}

bool SparkLodController::swapAsyncResults() {
    std::scoped_lock lock(mutex_);
    if (!ready_available_) {
        return false;
    }
    selected_indices_.swap(ready_swap_indices_);
    selected_logical_indices_.swap(ready_swap_logical_indices_);
    selected_touched_chunks_.swap(ready_swap_touched_chunks_);
    selected_levels_dirty_ = true;
    current_stats_ = ready_swap_stats_;
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
    latest_requested_generation_.store(++next_work_generation_, std::memory_order_release);
}

void SparkLodController::setReadyCallback(std::function<void()> callback) {
    std::scoped_lock lock(mutex_);
    ready_callback_ = std::move(callback);
}

void SparkLodController::applyPageMaps(const std::vector<uint32_t>& page_to_chunk,
                                       const std::vector<uint32_t>& chunk_to_page) {
    const std::size_t resident_chunks = static_cast<std::size_t>(
        std::count_if(chunk_to_page.begin(), chunk_to_page.end(), [](const std::uint32_t page) {
            return page != kSparkInvalidPage;
        }));
    {
        std::scoped_lock lock(page_maps_mutex_);
        page_to_chunk_ = page_to_chunk;
        chunk_to_page_ = chunk_to_page;
    }
    {
        std::scoped_lock lock(mutex_);
        base_stats_.resident_chunks = resident_chunks;
        current_stats_.resident_chunks = resident_chunks;
        ready_swap_stats_.resident_chunks = resident_chunks;
        pending_work_.reset();
        ready_available_ = false;
        last_requested_work_.reset();
        latest_requested_generation_.store(++next_work_generation_, std::memory_order_release);
    }
}

bool SparkLodController::publishAsyncResult(const WorkItem& work, const TraverseResult& result) {
    if (result.cancelled) {
        return false;
    }

    std::function<void()> ready_callback;
    {
        std::scoped_lock lock(mutex_);
        if (work.generation != latest_requested_generation_.load(std::memory_order_acquire)) {
            return false;
        }
        ready_swap_indices_.swap(async_indices_);
        ready_swap_logical_indices_.swap(async_logical_indices_);
        ready_swap_touched_chunks_ = async_scratch_.touched_chunks;
        ready_swap_stats_ = result.stats;
        ready_available_ = true;
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
        }

        const auto result = traverse(work.view_matrix,
                                     work.params,
                                     async_scratch_,
                                     async_indices_,
                                     async_logical_indices_,
                                     work.generation);
        if (result.cancelled || stop_token.stop_requested()) {
            continue;
        }
        publishAsyncResult(work, result);
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

    return page_maps.chunk_to_page[first_chunk] != kSparkInvalidPage &&
           page_maps.chunk_to_page[last_chunk] != kSparkInvalidPage;
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
        return node_index;
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

SparkLodController::TraverseResult SparkLodController::traverse(
    const glm::mat4& view_matrix,
    const LodParameters& params,
    TraversalScratch& scratch,
    std::vector<uint32_t>& out_indices,
    std::vector<uint32_t>& out_logical_indices,
    const std::uint64_t cancel_generation) const {
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
    std::uint32_t cancel_poll = 0;
    const auto should_cancel = [&]() {
        if (cancel_generation == 0) {
            return false;
        }
        if ((++cancel_poll & 0x3ffu) != 0) {
            return false;
        }
        return latest_requested_generation_.load(std::memory_order_acquire) != cancel_generation;
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
    scratch.touched_chunks.clear();
    const auto touch_chunk = [&](const std::size_t node_index) {
        const std::size_t chunk_index = node_index / kSparkLodChunkSplats;
        if (chunk_index < scratch.touched_chunk_bitmap.size() && scratch.touched_chunk_bitmap[chunk_index] == 0) {
            scratch.touched_chunk_bitmap[chunk_index] = 1;
            scratch.touched_chunks.push_back(static_cast<uint32_t>(chunk_index));
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

    auto& heap = scratch.frontier_heap;
    heap.clear();
    heap.reserve(std::min(params.max_splats, nodes_.size()));
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
    const PreparedTraversal traversal = prepareTraversal(view_matrix, params);

    // Seed with root node
    push_heap_node({0, computePixelScale(0, traversal)});
    touch_chunk(0);

    // Matches Spark semantics: this tracks output size after draining frontier.
    size_t num_splats = 1;
    float min_pixel_scale = std::numeric_limits<float>::max();

    while (!heap.empty()) {
        if (should_cancel()) {
            return cancel_traversal();
        }
        const auto top = heap.front();
        min_pixel_scale = std::min(min_pixel_scale, top.pixel_scale);
        if (top.pixel_scale <= params.pixel_scale_limit) {
            stats.threshold_limited = true;
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
            // Internal node: check budget before expanding.
            const size_t new_num_splats = num_splats - 1 + static_cast<size_t>(node.child_count);
            if (new_num_splats > params.max_splats) {
                // Keep this node in the frontier output (Spark behavior).
                push_heap_node(current);
                stats.budget_limited = true;
                break;
            }

            // Expand children. Children already below threshold go directly to output.
            touch_child_range(node.child_start, node.child_count);
            if (!childRangeResident(page_maps, node.child_start, node.child_count)) {
                emit_node(current.index);
                continue;
            }
            for (uint32_t c = 0; c < node.child_count; ++c) {
                if (should_cancel()) {
                    return cancel_traversal();
                }
                const uint32_t child_idx = node.child_start + c;
                if (child_idx < nodes_.size()) {
                    const float scale = computePixelScale(child_idx, traversal);
                    min_pixel_scale = std::min(min_pixel_scale, scale);
                    if (scale <= params.pixel_scale_limit) {
                        emit_node(child_idx);
                    } else {
                        push_heap_node({child_idx, scale});
                    }
                }
            }
            num_splats = new_num_splats;
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

void SparkLodController::activateFullQualityReference() {
    std::scoped_lock lock(mutex_);
    pending_work_.reset();
    last_requested_work_.reset();
    ready_available_ = false;
    latest_requested_generation_.store(++next_work_generation_, std::memory_order_release);
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
    selected_touched_chunks_ = full_quality_touched_chunks_;
}

SparkLodController::Stats SparkLodController::stats() const {
    std::scoped_lock lock(mutex_);
    auto stats = current_stats_;
    stats.async_result_ready = ready_available_;
    return stats;
}

} // namespace lfs::vis
