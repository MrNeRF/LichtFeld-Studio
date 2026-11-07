/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/batch_trainer.hpp"
#include "batch_process.hpp"

#include "core/argument_parser.hpp"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include "core/scene_partitioner.hpp"
#include "core/splat_data.hpp"
#include "core/tile_exposure.hpp"
#include "core/camera.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"
#include "io/ply_stream_merge.hpp"
#include "training/rasterization/fast_rasterizer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <print>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>

#ifdef _WIN32
#include "python/windows_process_utils.hpp"
#include <windows.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace lfs::app {

    namespace {

        using std::filesystem::path;
        using json = nlohmann::json;

        std::atomic<int> g_interrupt_count{0};
#ifndef _WIN32
        std::atomic<int> g_interrupt_signal{SIGINT};

        void handle_sigint(int signal) {
            g_interrupt_signal.store(signal);
            ++g_interrupt_count;
        }
#else
        BOOL WINAPI handle_console_ctrl(DWORD ctrl_type) {
            if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT) {
                ++g_interrupt_count;
                return TRUE;
            }
            return FALSE;
        }
#endif

        struct ScopedSignalHandler {
            ScopedSignalHandler() {
                g_interrupt_count.store(0);
#ifndef _WIN32
                g_interrupt_signal.store(SIGINT);
                struct sigaction sa {};
                sa.sa_handler = handle_sigint;
                sigemptyset(&sa.sa_mask);
                sa.sa_flags = 0;
                sigaction(SIGINT, &sa, &old_sa_int_);
                sigaction(SIGTERM, &sa, &old_sa_term_);
#else
                SetConsoleCtrlHandler(handle_console_ctrl, TRUE);
#endif
            }

            ~ScopedSignalHandler() {
#ifndef _WIN32
                sigaction(SIGINT, &old_sa_int_, nullptr);
                sigaction(SIGTERM, &old_sa_term_, nullptr);
#else
                SetConsoleCtrlHandler(handle_console_ctrl, FALSE);
#endif
            }

        private:
#ifndef _WIN32
            struct sigaction old_sa_int_ {};
            struct sigaction old_sa_term_ {};
#endif
        };

        struct TileFeasibility {
            bool feasible = false;
            int camera_count = 0;
            int multi_view_cells = 0;
            std::vector<std::string> reasons;
        };

        TileFeasibility assess_tile_feasibility(
            const std::vector<int>& cameras,
            const lfs::core::TileCameraSelectionDiagnostics& diagnostics,
            const lfs::core::param::TrainingParameters& params) {
            TileFeasibility result;
            result.camera_count = static_cast<int>(cameras.size());
            result.multi_view_cells = diagnostics.multi_view_covered_cells;
            if (result.camera_count < params.tile_min_cameras) {
                result.reasons.push_back(std::format(
                    "selected cameras {} < required {}", result.camera_count,
                    params.tile_min_cameras));
            }
            if (result.multi_view_cells < params.tile_min_multi_view_cells) {
                result.reasons.push_back(std::format(
                    "multi-view cells {} < required {}", result.multi_view_cells,
                    params.tile_min_multi_view_cells));
            }
            result.feasible = result.reasons.empty();
            return result;
        }

        glm::mat3 cropbox_rotation(const glm::vec3& degrees) {
            return glm::mat3(glm::eulerAngleXYZ(glm::radians(degrees.x),
                                                glm::radians(degrees.y),
                                                glm::radians(degrees.z)));
        }

        struct LocalTileBounds {
            glm::vec3 low{0.0f};
            glm::vec3 high{0.0f};
        };

        // TileSpec stores the world-space centre as min/max midpoint and its
        // local half extents as their difference.  Keep all partition topology
        // operations in that local frame; comparing world X/Y is wrong for an
        // OBB/PCA-aligned plan.
        LocalTileBounds local_tile_bounds(const lfs::core::TileSpec& tile,
                                          const bool context = false) {
            const glm::vec3 world_low = context && tile.has_context_bounds
                ? tile.context_min : tile.min;
            const glm::vec3 world_high = context && tile.has_context_bounds
                ? tile.context_max : tile.max;
            const glm::vec3 world_center = (world_low + world_high) * 0.5f;
            const glm::vec3 local_center =
                glm::transpose(cropbox_rotation(tile.rotation_degrees)) * world_center;
            const glm::vec3 half = (world_high - world_low) * 0.5f;
            return {local_center - half, local_center + half};
        }

        void set_local_tile_bounds(lfs::core::TileSpec& tile,
                                   const LocalTileBounds bounds,
                                   const bool context = false) {
            const glm::vec3 local_center = (bounds.low + bounds.high) * 0.5f;
            const glm::vec3 world_center = cropbox_rotation(tile.rotation_degrees) * local_center;
            const glm::vec3 half = (bounds.high - bounds.low) * 0.5f;
            if (context) {
                tile.context_min = world_center - half;
                tile.context_max = world_center + half;
                tile.has_context_bounds = true;
            } else {
                tile.min = world_center - half;
                tile.max = world_center + half;
            }
        }

        float tile_geometry_epsilon(const lfs::core::TileSpec& a,
                                    const lfs::core::TileSpec& b) {
            const auto a_bounds = local_tile_bounds(a);
            const auto b_bounds = local_tile_bounds(b);
            float scale = 1.0f;
            for (int axis = 0; axis < 3; ++axis) {
                // Scale with a tile dimension, never with its distance from
                // the world origin.  The latter can be millions of metres in
                // geo-referenced data and would turn an adjacency tolerance
                // into a genuine spatial gap.
                scale = std::max(scale, a_bounds.high[axis] - a_bounds.low[axis]);
                scale = std::max(scale, b_bounds.high[axis] - b_bounds.low[axis]);
            }
            // The additional ULP term preserves the nextafter ownership
            // convention after world/local coordinate round trips.
            const float coordinate_scale = std::max({
                std::abs(a_bounds.low.x), std::abs(a_bounds.low.y), std::abs(a_bounds.low.z),
                std::abs(a_bounds.high.x), std::abs(a_bounds.high.y), std::abs(a_bounds.high.z),
                std::abs(b_bounds.low.x), std::abs(b_bounds.low.y), std::abs(b_bounds.low.z),
                std::abs(b_bounds.high.x), std::abs(b_bounds.high.y), std::abs(b_bounds.high.z)});
            return std::max({1e-5f, scale * 1e-5f,
                             coordinate_scale * std::numeric_limits<float>::epsilon() * 4.0f});
        }

        constexpr float TILE_ROTATION_EPSILON_DEGREES = 1e-3f;

        bool tiles_make_rectangular_union(const lfs::core::TileSpec& a,
                                          const lfs::core::TileSpec& b) {
            const float epsilon = tile_geometry_epsilon(a, b);
            if (glm::length(a.rotation_degrees - b.rotation_degrees) > TILE_ROTATION_EPSILON_DEGREES)
                return false;
            const auto a_bounds = local_tile_bounds(a);
            const auto b_bounds = local_tile_bounds(b);
            for (int join_axis = 0; join_axis < 3; ++join_axis) {
                const bool faces_touch =
                    std::abs(a_bounds.high[join_axis] - b_bounds.low[join_axis]) <= epsilon ||
                    std::abs(b_bounds.high[join_axis] - a_bounds.low[join_axis]) <= epsilon;
                if (!faces_touch) continue;
                bool same_other_intervals = true;
                for (int axis = 0; axis < 3; ++axis) {
                    if (axis == join_axis) continue;
                    same_other_intervals = same_other_intervals &&
                        std::abs(a_bounds.low[axis] - b_bounds.low[axis]) <= epsilon &&
                        std::abs(a_bounds.high[axis] - b_bounds.high[axis]) <= epsilon;
                }
                if (same_other_intervals) return true;
            }
            return false;
        }

        lfs::core::TileSpec coalesce_tiles(const lfs::core::TileSpec& a,
                                           const lfs::core::TileSpec& b) {
            lfs::core::TileSpec merged = a;
            const auto a_bounds = local_tile_bounds(a);
            const auto b_bounds = local_tile_bounds(b);
            LocalTileBounds merged_bounds;
            for (int axis = 0; axis < 3; ++axis) {
                merged_bounds.low[axis] = std::min(a_bounds.low[axis], b_bounds.low[axis]);
                merged_bounds.high[axis] = std::max(a_bounds.high[axis], b_bounds.high[axis]);
            }
            set_local_tile_bounds(merged, merged_bounds);
            if (a.has_context_bounds || b.has_context_bounds) {
                const auto a_context = local_tile_bounds(a, true);
                const auto b_context = local_tile_bounds(b, true);
                LocalTileBounds merged_context;
                for (int axis = 0; axis < 3; ++axis) {
                    merged_context.low[axis] = std::min(a_context.low[axis], b_context.low[axis]);
                    merged_context.high[axis] = std::max(a_context.high[axis], b_context.high[axis]);
                }
                set_local_tile_bounds(merged, merged_context, true);
            }
            return merged;
        }

        std::vector<lfs::core::TileSpec> split_tile_on_neighbour_boundaries(
            const lfs::core::TileSpec& tile,
            const lfs::core::TileSpec& neighbour,
            const int join_axis) {
            const auto core = local_tile_bounds(tile);
            const auto neighbour_core = local_tile_bounds(neighbour);
            const auto context = local_tile_bounds(tile, true);
            const float epsilon = tile_geometry_epsilon(tile, neighbour);
            std::array<std::vector<float>, 3> cuts;
            for (int axis = 0; axis < 3; ++axis) {
                cuts[axis] = {core.low[axis], core.high[axis]};
                if (axis == join_axis) continue;
                for (const float cut : {neighbour_core.low[axis], neighbour_core.high[axis]}) {
                    if (cut > core.low[axis] + epsilon && cut < core.high[axis] - epsilon)
                        cuts[axis].push_back(cut);
                }
                std::sort(cuts[axis].begin(), cuts[axis].end());
                cuts[axis].erase(std::unique(cuts[axis].begin(), cuts[axis].end(),
                                             [epsilon](const float lhs, const float rhs) {
                                                 return std::abs(lhs - rhs) <= epsilon;
                                             }), cuts[axis].end());
            }
            const glm::vec3 lower_margin = core.low - context.low;
            const glm::vec3 upper_margin = context.high - core.high;
            std::vector<lfs::core::TileSpec> fragments;
            for (size_t x = 0; x + 1 < cuts[0].size(); ++x) {
                for (size_t y = 0; y + 1 < cuts[1].size(); ++y) {
                    for (size_t z = 0; z + 1 < cuts[2].size(); ++z) {
                        LocalTileBounds fragment_bounds{
                            {cuts[0][x], cuts[1][y], cuts[2][z]},
                            {cuts[0][x + 1], cuts[1][y + 1], cuts[2][z + 1]}};
                        if (fragment_bounds.high.x - fragment_bounds.low.x <= epsilon ||
                            fragment_bounds.high.y - fragment_bounds.low.y <= epsilon ||
                            fragment_bounds.high.z - fragment_bounds.low.z <= epsilon)
                            continue;
                        auto fragment = tile;
                        set_local_tile_bounds(fragment, fragment_bounds);
                        set_local_tile_bounds(fragment,
                                              {fragment_bounds.low - lower_margin,
                                               fragment_bounds.high + upper_margin}, true);
                        fragments.push_back(std::move(fragment));
                    }
                }
            }
            return fragments;
        }

        struct RectangularClosure {
            std::vector<lfs::core::TileSpec> failing_fragments;
            std::vector<lfs::core::TileSpec> neighbour_fragments;
            size_t failing_fragment = 0;
            size_t neighbour_fragment = 0;
            int join_axis = 0;
        };

        std::optional<RectangularClosure> make_rectangular_closure(
            const lfs::core::TileSpec& failing,
            const lfs::core::TileSpec& neighbour) {
            const float epsilon = tile_geometry_epsilon(failing, neighbour);
            if (glm::length(failing.rotation_degrees - neighbour.rotation_degrees) > TILE_ROTATION_EPSILON_DEGREES)
                return std::nullopt;
            const auto failing_bounds = local_tile_bounds(failing);
            const auto neighbour_bounds = local_tile_bounds(neighbour);
            for (int join_axis = 0; join_axis < 3; ++join_axis) {
                const bool faces_touch =
                    std::abs(failing_bounds.high[join_axis] - neighbour_bounds.low[join_axis]) <= epsilon ||
                    std::abs(neighbour_bounds.high[join_axis] - failing_bounds.low[join_axis]) <= epsilon;
                if (!faces_touch) continue;
                bool positive_overlap = true;
                for (int axis = 0; axis < 3; ++axis) {
                    if (axis == join_axis) continue;
                    positive_overlap = positive_overlap &&
                        std::min(failing_bounds.high[axis], neighbour_bounds.high[axis]) -
                            std::max(failing_bounds.low[axis], neighbour_bounds.low[axis]) > epsilon;
                }
                if (!positive_overlap) continue;

                auto failing_fragments = split_tile_on_neighbour_boundaries(
                    failing, neighbour, join_axis);
                auto neighbour_fragments = split_tile_on_neighbour_boundaries(
                    neighbour, failing, join_axis);
                size_t best_failing = failing_fragments.size();
                size_t best_neighbour = neighbour_fragments.size();
                float best_face_area = -1.0f;
                for (size_t left = 0; left < failing_fragments.size(); ++left) {
                    for (size_t right = 0; right < neighbour_fragments.size(); ++right) {
                        if (!tiles_make_rectangular_union(failing_fragments[left], neighbour_fragments[right]))
                            continue;
                        const auto bounds = local_tile_bounds(failing_fragments[left]);
                        float face_area = 1.0f;
                        for (int axis = 0; axis < 3; ++axis)
                            if (axis != join_axis)
                                face_area *= bounds.high[axis] - bounds.low[axis];
                        if (face_area > best_face_area) {
                            best_face_area = face_area;
                            best_failing = left;
                            best_neighbour = right;
                        }
                    }
                }
                if (best_failing != failing_fragments.size()) {
                    return RectangularClosure{std::move(failing_fragments),
                                              std::move(neighbour_fragments),
                                              best_failing, best_neighbour, join_axis};
                }
            }
            return std::nullopt;
        }

        // Extend only outer ownership faces.  This happens strictly after
        // camera allocation, so a boundary safety margin cannot cause one tile
        // to steal another tile's cameras or alter feasibility decisions.
        std::vector<std::array<bool, 4>> extend_boundary_tile_cores(
            std::vector<lfs::core::TileSpec>& tiles, const float padding) {
            std::vector<std::array<bool, 4>> expanded(tiles.size());
            if (padding <= 0.0f || tiles.empty()) return expanded;
            const glm::vec3 rotation = tiles.front().rotation_degrees;
            LocalTileBounds global = local_tile_bounds(tiles.front());
            for (const auto& tile : tiles) {
                // Automatic plans share one orientation.  A hand-authored plan
                // with mixed frames cannot have a well-defined common outward
                // direction, so leave those exceptional tiles untouched.
                if (glm::length(tile.rotation_degrees - rotation) > TILE_ROTATION_EPSILON_DEGREES)
                    continue;
                const auto bounds = local_tile_bounds(tile);
                global.low = glm::min(global.low, bounds.low);
                global.high = glm::max(global.high, bounds.high);
            }
            const float scale = std::max(1.0f, glm::length(global.high - global.low));
            const float coordinate_scale = std::max({
                std::abs(global.low.x), std::abs(global.low.y), std::abs(global.low.z),
                std::abs(global.high.x), std::abs(global.high.y), std::abs(global.high.z)});
            const float epsilon = std::max({1e-5f, scale * 1e-5f,
                coordinate_scale * std::numeric_limits<float>::epsilon() * 4.0f});
            // Auto tiling splits the horizontal local plane.  Extending Z for
            // every tile would duplicate the same safety margin everywhere.
            for (size_t index = 0; index < tiles.size(); ++index) {
                auto& tile = tiles[index];
                if (glm::length(tile.rotation_degrees - rotation) > TILE_ROTATION_EPSILON_DEGREES)
                    continue;
                auto bounds = local_tile_bounds(tile);
                for (int axis = 0; axis < 2; ++axis) {
                    if (std::abs(bounds.low[axis] - global.low[axis]) <= epsilon) {
                        bounds.low[axis] -= padding;
                        expanded[index][static_cast<size_t>(axis) * 2] = true;
                    }
                    if (std::abs(bounds.high[axis] - global.high[axis]) <= epsilon) {
                        bounds.high[axis] += padding;
                        expanded[index][static_cast<size_t>(axis) * 2 + 1] = true;
                    }
                }
                set_local_tile_bounds(tile, bounds);
            }
            return expanded;
        }

        struct DenseCoarseRender {
            int width = 0;
            int height = 0;
            std::vector<float> alpha;
            std::vector<float> depth;
        };

        // A full-model render is reused by overlapping tiles, but keeping one
        // for every image makes a high-resolution dataset consume unbounded
        // host memory.  At the largest supported mask size this cap is about
        // 32 MiB for the two float planes.
        constexpr size_t DENSE_MASK_FULL_RENDER_CACHE_LIMIT = 64;

        bool point_in_tile_core(const glm::vec3& point,
                                const lfs::core::TileSpec& tile) {
            const auto inverse = glm::transpose(cropbox_rotation(tile.rotation_degrees));
            const glm::vec3 center = (tile.min + tile.max) * 0.5f;
            const glm::vec3 half = (tile.max - tile.min) * 0.5f;
            const glm::vec3 local = inverse * (point - center);
            return local.x >= -half.x && local.x <= half.x &&
                   local.y >= -half.y && local.y <= half.y &&
                   local.z >= -half.z && local.z <= half.z;
        }

        std::unique_ptr<lfs::core::SplatData> coarse_owner_model(
            const lfs::core::SplatData& coarse,
            const lfs::core::Tensor& coarse_means_cpu,
            const lfs::core::TileSpec& tile) {
            if (coarse_means_cpu.ndim() != 2 || coarse_means_cpu.size(1) < 3) return {};
            const auto* values = coarse_means_cpu.ptr<float>();
            const size_t count = coarse_means_cpu.size(0);
            const size_t stride = coarse_means_cpu.size(1);
            std::vector<int> selected;
            selected.reserve(std::min<size_t>(count, 100000));
            for (size_t index = 0; index < count; ++index) {
                if ((index % 4096) == 0 && g_interrupt_count.load() > 0)
                    throw std::runtime_error("cancelled while selecting coarse owner Gaussians");
                const glm::vec3 point{values[index * stride], values[index * stride + 1],
                                      values[index * stride + 2]};
                if (std::isfinite(point.x) && std::isfinite(point.y) &&
                    std::isfinite(point.z) && point_in_tile_core(point, tile)) {
                    selected.push_back(static_cast<int>(index));
                }
            }
            if (selected.empty()) return {};
            const auto indices = lfs::core::Tensor::from_vector(
                selected, lfs::core::TensorShape({selected.size()}), coarse.means().device());
            auto canonical_sh = coarse.shN_canonical();
            auto selected_sh = canonical_sh.is_valid() && canonical_sh.numel() > 0
                                   ? canonical_sh.index_select(0, indices)
                                   : lfs::core::Tensor{};
            auto result = std::make_unique<lfs::core::SplatData>(
                coarse.get_max_sh_degree(),
                coarse.means().index_select(0, indices),
                coarse.sh0().index_select(0, indices),
                std::move(selected_sh),
                coarse.scaling_raw().index_select(0, indices),
                coarse.rotation_raw().index_select(0, indices),
                coarse.opacity_raw().index_select(0, indices),
                coarse.get_scene_scale(), lfs::core::SplatData::ShNLayout::Canonical);
            result->set_active_sh_degree(coarse.get_active_sh_degree());
            return result;
        }

        std::optional<DenseCoarseRender> rasterize_coarse_depth_opacity(
            const lfs::core::SplatData& model,
            const lfs::core::TileCameraEvidence& view,
            const int resolution,
            lfs::core::Tensor& background) {
            if (!view.source || !view.pinhole || view.width <= 0 || view.height <= 0 ||
                resolution < 16 || resolution > 256 || model.size() == 0) {
                return std::nullopt;
            }
            const int width = resolution;
            const int height = std::max(16, static_cast<int>(std::lround(
                static_cast<double>(resolution) * view.height / view.width)));
            if (height > 256) return std::nullopt;
            const float sx = static_cast<float>(width) / static_cast<float>(view.width);
            const float sy = static_cast<float>(height) / static_cast<float>(view.height);
            const auto empty_distortion = lfs::core::Tensor::empty(
                {0}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
            lfs::core::Camera render_camera(
                view.source->R().cpu().contiguous(), view.source->T().cpu().contiguous(),
                view.fx * sx, view.fy * sy, view.cx * sx, view.cy * sy,
                empty_distortion, empty_distortion,
                lfs::core::CameraModelType::PINHOLE,
                std::format("{}_coarse_mask", view.image_name), {}, {}, width, height,
                view.uid, view.source->camera_id());
            auto rendered = lfs::training::fast_rasterize(render_camera,
                                                           const_cast<lfs::core::SplatData&>(model),
                                                           background);
            if (!rendered.alpha.is_valid() || !rendered.depth.is_valid() ||
                rendered.alpha.numel() != static_cast<size_t>(width) * height ||
                rendered.depth.numel() != static_cast<size_t>(width) * height) {
                return std::nullopt;
            }
            const auto alpha = rendered.alpha.cpu().contiguous();
            const auto depth = rendered.depth.cpu().contiguous();
            DenseCoarseRender result;
            result.width = width;
            result.height = height;
            result.alpha.assign(alpha.ptr<float>(), alpha.ptr<float>() + alpha.numel());
            result.depth.assign(depth.ptr<float>(), depth.ptr<float>() + depth.numel());
            return result;
        }

        lfs::core::TileVisibilityMask dense_tile_visibility_mask(
            const lfs::core::TileCameraEvidence& view,
            const lfs::core::SplatData& coarse,
            const lfs::core::SplatData* owner,
            const int resolution,
            const float minimum_alpha,
            const float foreign_weight,
            lfs::core::Tensor& background,
            std::unordered_map<int, DenseCoarseRender>& full_render_cache,
            bool* const rasterized_successfully = nullptr) {
            if (rasterized_successfully) *rasterized_successfully = false;
            DenseCoarseRender full;
            if (const auto cached = full_render_cache.find(view.uid);
                cached != full_render_cache.end()) {
                full = cached->second;
            } else {
                const auto rendered = rasterize_coarse_depth_opacity(
                    coarse, view, resolution, background);
                if (!rendered) return {};
                full = *rendered;
                if (full_render_cache.size() >= DENSE_MASK_FULL_RENDER_CACHE_LIMIT)
                    full_render_cache.clear();
                full_render_cache.emplace(view.uid, full);
            }
            if (rasterized_successfully) *rasterized_successfully = true;
            lfs::core::TileVisibilityMask result;
            result.width = full.width;
            result.height = full.height;
            result.weights.assign(full.alpha.size(), 1.0f);
            // A coarse model may have no owner Gaussian in a very sparse core.
            // Full coverage then belongs outside that owner, but the mask is
            // still a valid dense raster result and must not degrade to the
            // old centre-only estimate.
            if (owner == nullptr) {
                for (size_t index = 0; index < result.weights.size(); ++index) {
                    if (std::isfinite(full.alpha[index]) && full.alpha[index] >= minimum_alpha)
                        result.weights[index] = foreign_weight;
                }
                return result;
            }
            const auto owner_render = rasterize_coarse_depth_opacity(
                *owner, view, resolution, background);
            if (!owner_render || owner_render->width != full.width ||
                owner_render->height != full.height ||
                owner_render->alpha.size() != full.alpha.size()) {
                if (rasterized_successfully) *rasterized_successfully = false;
                return {};
            }
            for (size_t index = 0; index < result.weights.size(); ++index) {
                const float full_alpha = full.alpha[index];
                const float owner_alpha = owner_render->alpha[index];
                const float full_depth = full.depth[index];
                const float owner_depth = owner_render->depth[index];
                if (!std::isfinite(full_alpha) || !std::isfinite(owner_alpha) ||
                    full_alpha < minimum_alpha) continue;
                const bool foreign_dominates =
                    owner_alpha < std::max(minimum_alpha * 0.5f, full_alpha * 0.35f);
                const bool owner_is_behind =
                    owner_alpha >= minimum_alpha && std::isfinite(full_depth) &&
                    std::isfinite(owner_depth) && owner_depth > full_depth * 1.01f;
                if (foreign_dominates || owner_is_behind) {
                    result.weights[index] = foreign_weight;
                }
            }
            // Preserve an all-one dense mask too.  Its presence tells the
            // worker that rasterization succeeded and prevents it from
            // silently replacing a valid "no foreign surface" result with a
            // separate sparse fallback estimate.
            return result;
        }

        std::string format_duration(const double total_seconds) {
            if (total_seconds < 0.0)
                return "00s";
            const auto sec_int = static_cast<int64_t>(total_seconds);
            const int64_t hours = sec_int / 3600;
            const int64_t minutes = (sec_int % 3600) / 60;
            const int64_t seconds = sec_int % 60;

            if (hours > 0) {
                return std::format("{:02d}h {:02d}m {:02d}s", hours, minutes, seconds);
            }
            if (minutes > 0) {
                return std::format("{:02d}m {:02d}s", minutes, seconds);
            }
            return std::format("{:02d}s", seconds);
        }

        std::string current_iso_timestamp() {
            const auto now = std::chrono::system_clock::now();
            const auto in_time_t = std::chrono::system_clock::to_time_t(now);
            std::tm tm_buf{};
#ifdef _WIN32
            localtime_s(&tm_buf, &in_time_t);
#else
            localtime_r(&in_time_t, &tm_buf);
#endif
            std::ostringstream ss;
            ss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
            return ss.str();
        }

        std::string case_key(std::string value) {
            std::ranges::transform(value, value.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return value;
        }

        path resolved_path(const path& value) {
            return lfs::core::native_fs::weakly_canonical(lfs::core::native_fs::current_path() / value);
        }

        std::string path_key(const path& value) {
            return case_key(lfs::core::path_to_utf8(resolved_path(value)));
        }

        std::pair<path, path> report_paths(const lfs::core::param::BatchTrainingParameters& params) {
            return {params.summary_file.empty() ? params.output_dir / "batch_summary.json" : params.summary_file,
                    params.report_file.empty() ? params.output_dir / "batch_report.txt" : params.report_file};
        }

        // Visit options with their values, recognizing aliases and joined short values.
        template <typename Visit>
        void visit_training_args(const std::vector<std::string>& args, Visit visit) {
            for (size_t i = 0; i < args.size();) {
                const auto* flag = lfs::core::args::training_flag(args[i]);
                const size_t count = flag && flag->takes_value && args[i] == flag->name && i + 1 < args.size() ? 2 : 1;
                visit(i, count, flag);
                i += count;
            }
        }

        bool is_hidden_or_output_dir(const path& path) {
            const auto filename = lfs::core::path_to_utf8(path.filename());
            if (filename.empty())
                return false;
            if (filename.front() == '.')
                return true;
            if (filename == "output" || filename == "outputs")
                return true;
            return false;
        }

        void discover_datasets_recursive(const path& root, int depth, int max_depth,
                                         std::vector<path>& discovered) {
            if (depth > max_depth)
                return;
            if (!lfs::core::native_fs::exists(root) || !lfs::core::native_fs::is_directory(root))
                return;

            try {
                for (const auto& entry : lfs::core::native_fs::directory_iterator(root)) {
                    const auto entry_path = lfs::core::native_fs::entry_path(root, entry.path());
                    if (!lfs::core::native_fs::entry_is_directory(root, entry))
                        continue;
                    if (is_hidden_or_output_dir(entry_path))
                        continue;

                    if (lfs::io::Loader::isDatasetPath(entry_path)) {
                        discovered.push_back(entry_path);
                    } else if (depth < max_depth) {
                        discover_datasets_recursive(entry_path, depth + 1, max_depth, discovered);
                    }
                }
            } catch (const std::exception& e) {
                LOG_WARN("Could not read directory {}: {}", lfs::core::path_to_utf8(root), e.what());
            }
        }

        std::vector<lfs::core::param::BatchJobItem> resolve_jobs_internal(
            const lfs::core::param::BatchTrainingParameters& params) {
            std::vector<lfs::core::param::BatchJobItem> jobs = params.jobs;
            std::vector<path> candidate_paths;

            for (const auto& input : params.input_paths) {
                if (!lfs::core::native_fs::exists(input))
                    throw std::runtime_error(std::format("Batch input path does not exist: {}", lfs::core::path_to_utf8(input)));

                // If input is directly a dataset (e.g. contains sparse/ or transforms.json)
                if (lfs::io::Loader::isDatasetPath(input)) {
                    candidate_paths.push_back(input);
                    continue;
                }

                // Otherwise, input is a directory containing subdirectories
                if (lfs::core::native_fs::is_directory(input)) {
                    if (params.recursive) {
                        discover_datasets_recursive(input, 1, 5, candidate_paths);
                    } else {
                        try {
                            for (const auto& entry : lfs::core::native_fs::directory_iterator(input)) {
                                const auto entry_path = lfs::core::native_fs::entry_path(input, entry.path());
                                if (!lfs::core::native_fs::entry_is_directory(input, entry))
                                    continue;
                                if (is_hidden_or_output_dir(entry_path))
                                    continue;

                                if (lfs::io::Loader::isDatasetPath(entry_path)) {
                                    candidate_paths.push_back(entry_path);
                                }
                            }
                        } catch (const std::exception& e) {
                            LOG_WARN("Could not read directory {}: {}", lfs::core::path_to_utf8(input), e.what());
                        }
                    }
                } else {
                    throw std::runtime_error(std::format("Batch input is not a dataset or directory: {}", lfs::core::path_to_utf8(input)));
                }
            }

            // Sort candidate paths alphabetically for reproducible queue ordering
            std::sort(candidate_paths.begin(), candidate_paths.end());

            // Convert discovered dataset paths into BatchJobItems
            for (const auto& path : candidate_paths) {
                lfs::core::param::BatchJobItem item;
                item.dataset_path = path;
                jobs.push_back(std::move(item));
            }

            // Deduplicate jobs by canonical dataset path
            std::vector<lfs::core::param::BatchJobItem> unique_jobs;
            std::vector<std::string> seen_canonical_paths;

            for (auto& job : jobs) {
                if (job.dataset_path.empty() || !lfs::core::native_fs::exists(job.dataset_path))
                    throw std::runtime_error(std::format("Batch dataset does not exist: {}", lfs::core::path_to_utf8(job.dataset_path)));
                std::error_code ec;
                auto canonical = lfs::core::native_fs::weakly_canonical(job.dataset_path, ec);
                std::string path_str = lfs::core::path_to_utf8(ec ? job.dataset_path : canonical);

                if (std::find(seen_canonical_paths.begin(), seen_canonical_paths.end(), path_str) != seen_canonical_paths.end()) {
                    continue;
                }
                seen_canonical_paths.push_back(path_str);
                unique_jobs.push_back(std::move(job));
            }

            // Check for duplicate folder names among jobs without explicit custom names
            std::unordered_map<std::string, int> base_name_counts;
            for (const auto& job : unique_jobs) {
                if (job.name.empty()) {
                    std::string base = lfs::core::path_to_utf8(job.dataset_path.stem());
                    if (base.empty() || base == ".") {
                        base = "dataset";
                    }
                    base_name_counts[case_key(base)]++;
                }
            }

            // Reserve all explicit names first so generated names never collide with them
            std::unordered_set<std::string> reserved_names;
            for (const auto& job : unique_jobs) {
                if (!job.name.empty()) {
                    reserved_names.insert(case_key(job.name));
                }
            }

            // Assign unique names
            std::unordered_set<std::string> assigned_names;
            for (auto& job : unique_jobs) {
                if (job.name.empty()) {
                    std::string base = lfs::core::path_to_utf8(job.dataset_path.stem());
                    if (base.empty() || base == ".") {
                        base = "dataset";
                    }
                    if (base_name_counts[case_key(base)] > 1) {
                        auto parent = job.dataset_path.parent_path().stem();
                        std::string parent_str = lfs::core::path_to_utf8(parent);
                        if (!parent_str.empty() && parent_str != "." && parent_str != "/" && parent_str != "\\") {
                            base = parent_str + "_" + base;
                        }
                    }

                    std::string candidate = base;
                    if (assigned_names.contains(case_key(candidate)) || reserved_names.contains(case_key(candidate))) {
                        int suffix = 2;
                        while (true) {
                            candidate = base + "_" + std::to_string(suffix);
                            if (!assigned_names.contains(case_key(candidate)) && !reserved_names.contains(case_key(candidate))) {
                                break;
                            }
                            ++suffix;
                        }
                    }
                    job.name = candidate;
                    assigned_names.insert(case_key(candidate));
                } else {
                    // Disambiguate duplicate explicit names if any
                    std::string candidate = job.name;
                    if (assigned_names.contains(case_key(candidate))) {
                        int suffix = 2;
                        while (true) {
                            candidate = job.name + "_" + std::to_string(suffix);
                            if (!assigned_names.contains(case_key(candidate)) && !reserved_names.contains(case_key(candidate))) {
                                break;
                            }
                            ++suffix;
                        }
                        job.name = candidate;
                    }
                    assigned_names.insert(case_key(job.name));
                }
            }

            // Assign output paths and ensure no collisions
            std::vector<std::string> seen_output_paths;
            for (auto& job : unique_jobs) {
                if (job.output_path.empty()) {
                    if (!params.output_dir.empty()) {
                        job.output_path = params.output_dir / job.name;
                    } else {
                        job.output_path = job.dataset_path / "output";
                    }
                }

                std::error_code ec;
                auto canonical_out = lfs::core::native_fs::weakly_canonical(lfs::core::native_fs::current_path() / job.output_path, ec);
                std::string out_str = case_key(lfs::core::path_to_utf8(ec ? job.output_path : canonical_out));

                if (std::find(seen_output_paths.begin(), seen_output_paths.end(), out_str) != seen_output_paths.end()) {
                    int suffix = 2;
                    path base_out = job.output_path;
                    while (true) {
                        path cand = base_out;
                        cand += "_" + std::to_string(suffix);
                        auto cand_canonical = lfs::core::native_fs::weakly_canonical(lfs::core::native_fs::current_path() / cand, ec);
                        std::string cand_str = case_key(lfs::core::path_to_utf8(ec ? cand : cand_canonical));
                        if (std::find(seen_output_paths.begin(), seen_output_paths.end(), cand_str) == seen_output_paths.end()) {
                            job.output_path = cand;
                            seen_output_paths.push_back(cand_str);
                            break;
                        }
                        suffix++;
                    }
                } else {
                    seen_output_paths.push_back(out_str);
                }
            }

            return unique_jobs;
        }

        struct JobExecutionResult {
            std::string name;
            path dataset_path;
            path output_path;
            bool success = false;
            bool cancelled = false;
            bool timed_out = false;
            int exit_code = 0;
            double duration_seconds = 0.0;
            std::string error_message;
        };

        void cancel_remaining_jobs(
            const std::vector<lfs::core::param::BatchJobItem>& jobs,
            const size_t from_index,
            std::vector<JobExecutionResult>& results,
            const std::string_view reason) {
            for (size_t i = from_index; i < jobs.size(); ++i) {
                JobExecutionResult cancelled_res;
                cancelled_res.name = jobs[i].name;
                cancelled_res.dataset_path = jobs[i].dataset_path;
                cancelled_res.output_path = jobs[i].output_path;
                cancelled_res.cancelled = true;
                cancelled_res.exit_code = 130;
                cancelled_res.error_message = std::string(reason);
                results.push_back(std::move(cancelled_res));
            }
        }

        JobExecutionResult execute_job(
            const lfs::core::param::BatchJobItem& job,
            const lfs::core::param::BatchTrainingParameters& params,
            const path& executable) {
            JobExecutionResult result;
            result.name = job.name;
            result.dataset_path = job.dataset_path;
            result.output_path = job.output_path;

            if (g_interrupt_count.load() > 0) {
                result.cancelled = true;
                result.exit_code = 130;
                result.error_message = "Cancelled by user";
                return result;
            }

            std::error_code ec;
            lfs::core::native_fs::create_directories(job.output_path, ec);
            if (ec) {
                result.success = false;
                result.exit_code = 1;
                result.error_message = std::format("Failed to create output directory: {}", ec.message());
                return result;
            }

            const auto log_path = job.output_path / "training_log.txt";
            std::ofstream log_file(lfs::core::native_fs::native_path(log_path), std::ios::out | std::ios::trunc);
            if (!log_file.is_open()) {
                LOG_WARN("Could not open log file at {}", lfs::core::path_to_utf8(log_path));
            }

            const auto child_args = build_child_args(job, params);
            const auto start_time = std::chrono::steady_clock::now();

            if (g_interrupt_count.load() > 0) {
                result.cancelled = true;
                result.exit_code = 130;
                result.error_message = "Cancelled by user";
                return result;
            }

#ifdef _WIN32
            SECURITY_ATTRIBUTES sa;
            sa.nLength = sizeof(SECURITY_ATTRIBUTES);
            sa.bInheritHandle = TRUE;
            sa.lpSecurityDescriptor = nullptr;

            HANDLE hReadPipe, hWritePipe;
            if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
                result.success = false;
                result.exit_code = 1;
                result.error_message = "Failed to create process pipe";
                return result;
            }
            if (!SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0)) {
                const DWORD error = GetLastError();
                CloseHandle(hReadPipe);
                CloseHandle(hWritePipe);
                result.exit_code = 1;
                result.error_message = std::format("Failed to start child process (system error {})", error);
                return result;
            }

            STARTUPINFOW si = {};
            si.cb = sizeof(si);
            si.hStdOutput = hWritePipe;
            si.hStdError = hWritePipe;
            si.dwFlags |= STARTF_USESTDHANDLES;

            PROCESS_INFORMATION pi = {};
            std::wstring cmdline = lfs::python::detail::build_win32_cmdline(executable, child_args);

            HANDLE hJob = nullptr;
            const DWORD creation_flags = CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED;

            if (!CreateProcessW(executable.c_str(), cmdline.data(), nullptr, nullptr, TRUE,
                                creation_flags, nullptr, nullptr, &si, &pi)) {
                const DWORD error = GetLastError();
                CloseHandle(hReadPipe);
                CloseHandle(hWritePipe);
                result.success = false;
                result.exit_code = 1;
                result.error_message = std::format("Failed to start child process (system error {})", error);
                return result;
            }

            CloseHandle(hWritePipe);

            const DWORD setup_error = detail::prepare_batch_child(pi, hJob, g_interrupt_count);
            if (setup_error != ERROR_SUCCESS) {
                if (hJob != nullptr)
                    CloseHandle(hJob);
                CloseHandle(pi.hProcess);
                CloseHandle(pi.hThread);
                CloseHandle(hReadPipe);
                result.exit_code = 1;
                result.error_message = std::format("Failed to start child process (system error {})", setup_error);
                return result;
            }

            int forwarded_interrupts = 0;
            const auto forward_cancel = [&]() {
                return detail::forward_pending_batch_cancel(g_interrupt_count.load(), forwarded_interrupts,
                                                            [&](const bool force) {
                                                                return (!force && GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pi.dwProcessId)) ||
                                                                       detail::stop_batch_child(pi.hProcess, hJob, 130);
                                                            });
            };

            HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
            char buffer[4096];
            DWORD bytesRead = 0;
            bool pipe_open = true;
            bool child_done = false;
            bool cancel_message_printed = false;
            detail::BatchTimeout timeout;
            constexpr DWORD MAX_DRAIN_PER_CYCLE = 64 * 1024;

            while (!child_done) {
                // Also delivers an interrupt received during ResumeThread, before the first wait.
                if (!forward_cancel()) {
                    result.exit_code = 1;
                    result.error_message = std::format("Failed to stop child process (system error {})", GetLastError());
                    break;
                }
                DWORD total_bytes_read_this_iteration = 0;
                if (pipe_open) {
                    while (total_bytes_read_this_iteration < MAX_DRAIN_PER_CYCLE) {
                        DWORD bytesAvailable = 0;
                        if (!PeekNamedPipe(hReadPipe, nullptr, 0, nullptr, &bytesAvailable, nullptr)) {
                            pipe_open = false;
                            break;
                        }
                        if (bytesAvailable == 0) {
                            break;
                        }
                        DWORD toRead = std::min<DWORD>(static_cast<DWORD>(sizeof(buffer) - 1), bytesAvailable);
                        toRead = std::min<DWORD>(toRead, MAX_DRAIN_PER_CYCLE - total_bytes_read_this_iteration);
                        if (ReadFile(hReadPipe, buffer, toRead, &bytesRead, nullptr) && bytesRead > 0) {
                            buffer[bytesRead] = '\0';
                            DWORD bytesWritten = 0;
                            WriteFile(hStdOut, buffer, bytesRead, &bytesWritten, nullptr);
                            if (log_file.is_open()) {
                                log_file.write(buffer, bytesRead);
                                log_file.flush();
                            }
                            total_bytes_read_this_iteration += bytesRead;
                        } else {
                            pipe_open = false;
                            break;
                        }
                    }
                }

                const DWORD wait_time = (total_bytes_read_this_iteration > 0) ? 0 : 50;
                DWORD wait_res = WaitForSingleObject(pi.hProcess, wait_time);
                if (wait_res == WAIT_FAILED) {
                    result.exit_code = 1;
                    result.error_message = std::format("Failed to stop child process (system error {})", GetLastError());
                    detail::stop_batch_child(pi.hProcess, hJob, 1);
                    break;
                }
                if (wait_res == WAIT_OBJECT_0) {
                    child_done = true;

                    if (pipe_open) {
                        DWORD drained = 0;
                        while (drained < MAX_DRAIN_PER_CYCLE) {
                            DWORD bytesAvailable = 0;
                            if (!PeekNamedPipe(hReadPipe, nullptr, 0, nullptr, &bytesAvailable, nullptr) || bytesAvailable == 0) {
                                break;
                            }
                            DWORD toRead = std::min<DWORD>(static_cast<DWORD>(sizeof(buffer) - 1), bytesAvailable);
                            toRead = std::min<DWORD>(toRead, MAX_DRAIN_PER_CYCLE - drained);
                            if (ReadFile(hReadPipe, buffer, toRead, &bytesRead, nullptr) && bytesRead > 0) {
                                drained += bytesRead;
                                buffer[bytesRead] = '\0';
                                DWORD bytesWritten = 0;
                                WriteFile(hStdOut, buffer, bytesRead, &bytesWritten, nullptr);
                                if (log_file.is_open()) {
                                    log_file.write(buffer, bytesRead);
                                    log_file.flush();
                                }
                            } else {
                                break;
                            }
                        }
                    }

                    if (g_interrupt_count.load() > 0) {
                        result.cancelled = true;
                        result.error_message = "Cancelled by user";
                        result.exit_code = 130;
                    }
                    break;
                }

                if (g_interrupt_count.load() > 0) {
                    result.cancelled = true;
                    result.error_message = "Cancelled by user";
                    if (result.exit_code == 0) {
                        result.exit_code = 130;
                    }

                    const int count = g_interrupt_count.load();
                    if (count >= 2 && !cancel_message_printed) {
                        std::println(stderr, "\n[Batch] Second interrupt received. Force terminating child process...");
                        cancel_message_printed = true;
                    } else if (!cancel_message_printed) {
                        std::println(stderr, "\n[Batch] Interrupt received. Waiting for child process to save and shut down cleanly... (Press Ctrl+C again to force kill)");
                        cancel_message_printed = true;
                    }
                }

                if (timeout.stopped_at || g_interrupt_count.load() == 0) {
                    const bool stopped = timeout.update(start_time, std::chrono::steady_clock::now(), params.timeout_minutes,
                                                        [&](const bool force) {
                                                            return (!force && GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pi.dwProcessId)) ||
                                                                   detail::stop_batch_child(pi.hProcess, hJob, 124);
                                                        });
                    if (timeout.stopped_at && !result.timed_out) {
                        result.timed_out = true;
                        result.exit_code = 124;
                        result.error_message = batch_timeout_message(params.timeout_minutes);
                        std::println(stderr, "\n[Batch] {}. Sending a stop signal; force termination follows after {} seconds.",
                                     result.error_message, detail::BATCH_STOP_GRACE.count());
                    }
                    if (!stopped) {
                        result.error_message = "Failed to stop timed-out child process";
                        break;
                    }
                }
            }

            if (!child_done) {
                if (detail::stop_batch_child(pi.hProcess, hJob, 1))
                    WaitForSingleObject(pi.hProcess, INFINITE);
                else
                    LOG_ERROR("Failed to terminate child process (system error {})", GetLastError());
            }
            DWORD exitCodeDword = 0;
            GetExitCodeProcess(pi.hProcess, &exitCodeDword);
            if (!result.timed_out && result.exit_code == 0) {
                result.exit_code = static_cast<int>(exitCodeDword);
            }
            if (result.cancelled && result.exit_code == 0) {
                result.exit_code = 130;
            }

            if (hJob != nullptr) {
                CloseHandle(hJob);
            }
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            CloseHandle(hReadPipe);

#else
            int pipe_fds[2];
            if (pipe(pipe_fds) != 0) {
                result.success = false;
                result.exit_code = 1;
                result.error_message = "Failed to create pipe";
                return result;
            }

            fcntl(pipe_fds[0], F_SETFL, O_NONBLOCK);

            const std::string program_utf8 = lfs::core::path_to_utf8(executable);
            std::vector<char*> c_argv;
            c_argv.reserve(child_args.size() + 2);
            c_argv.push_back(const_cast<char*>(program_utf8.c_str()));
            for (const auto& arg : child_args) {
                c_argv.push_back(const_cast<char*>(arg.c_str()));
            }
            c_argv.push_back(nullptr);

            sigset_t block_mask, orig_mask;
            sigemptyset(&block_mask);
            sigaddset(&block_mask, SIGINT);
            sigaddset(&block_mask, SIGTERM);
            sigprocmask(SIG_BLOCK, &block_mask, &orig_mask);

            if (g_interrupt_count.load() > 0) {
                sigprocmask(SIG_SETMASK, &orig_mask, nullptr);
                close(pipe_fds[0]);
                close(pipe_fds[1]);
                result.cancelled = true;
                result.exit_code = 130;
                result.error_message = "Cancelled by user";
                return result;
            }

            const pid_t pid = fork();
            if (pid == -1) {
                sigprocmask(SIG_SETMASK, &orig_mask, nullptr);
                close(pipe_fds[0]);
                close(pipe_fds[1]);
                result.success = false;
                result.exit_code = 1;
                result.error_message = "Failed to fork process";
                return result;
            }

            if (pid == 0) {
                if (!detail::prepare_batch_child(orig_mask))
                    _exit(127);
                dup2(pipe_fds[1], STDOUT_FILENO);
                dup2(pipe_fds[1], STDERR_FILENO);
                close(pipe_fds[0]);
                close(pipe_fds[1]);

                execvp(program_utf8.c_str(), c_argv.data());
                _exit(127);
            }

            if (!detail::register_batch_child(pid)) {
                const int error = errno;
                detail::cleanup_batch_group(pid);
                kill(pid, SIGKILL);
                while (waitpid(pid, nullptr, 0) == -1 && errno == EINTR) {}
                sigprocmask(SIG_SETMASK, &orig_mask, nullptr);
                close(pipe_fds[0]);
                close(pipe_fds[1]);
                result.exit_code = 1;
                result.error_message = std::format("Failed to start child process (system error {})", error);
                return result;
            }

            int forwarded_interrupts = 0;
            const auto forward_cancel = [&]() {
                return detail::forward_pending_batch_cancel(g_interrupt_count.load(), forwarded_interrupts,
                                                            [&](const bool force) {
                                                                return detail::signal_batch_child(pid, force ? SIGKILL : g_interrupt_signal.load());
                                                            });
            };
            forward_cancel();
            sigprocmask(SIG_SETMASK, &orig_mask, nullptr);

            close(pipe_fds[1]);

            char buffer[4096];
            bool child_done = false;
            bool cancel_message_printed = false;
            detail::BatchTimeout timeout;
            constexpr size_t MAX_DRAIN_PER_CYCLE = 64 * 1024;

            while (!child_done) {
                if (!forward_cancel()) {
                    result.exit_code = 1;
                    result.error_message = std::format("Failed to stop child process (system error {})", errno);
                    break;
                }
                struct pollfd pfd {};
                pfd.fd = pipe_fds[0];
                pfd.events = POLLIN | POLLHUP;

                int poll_res = poll(&pfd, 1, 50);
                if (poll_res > 0 && (pfd.revents & POLLIN)) {
                    size_t total_read = 0;
                    ssize_t bytes_read = 0;
                    while (total_read < MAX_DRAIN_PER_CYCLE &&
                           (bytes_read = read(pipe_fds[0], buffer, std::min(sizeof(buffer) - 1, MAX_DRAIN_PER_CYCLE - total_read))) > 0) {
                        buffer[bytes_read] = '\0';
                        ssize_t written = ::write(STDOUT_FILENO, buffer, bytes_read);
                        static_cast<void>(written);
                        std::fflush(stdout);

                        if (log_file.is_open()) {
                            log_file.write(buffer, bytes_read);
                            log_file.flush();
                        }
                        total_read += static_cast<size_t>(bytes_read);
                    }
                }

                sigprocmask(SIG_BLOCK, &block_mask, nullptr);
                int status = 0;
                pid_t wait_res = waitpid(pid, &status, WNOHANG);
                const int wait_error = errno;
                if (wait_res < 0 && wait_error != EINTR) {
                    sigprocmask(SIG_SETMASK, &orig_mask, nullptr);
                    result.exit_code = 1;
                    result.error_message = std::format("Failed to wait for child process (system error {})", wait_error);
                    break;
                }
                if (wait_res == pid) {
                    child_done = true;
                    sigprocmask(SIG_SETMASK, &orig_mask, nullptr);

                    // Drain remaining output without waiting for inherited pipe writers.
                    size_t drained = 0;
                    ssize_t bytes_read = 0;
                    while (drained < MAX_DRAIN_PER_CYCLE &&
                           (bytes_read = read(pipe_fds[0], buffer, std::min(sizeof(buffer) - 1, MAX_DRAIN_PER_CYCLE - drained))) > 0) {
                        drained += static_cast<size_t>(bytes_read);
                        buffer[bytes_read] = '\0';
                        ssize_t written = ::write(STDOUT_FILENO, buffer, bytes_read);
                        static_cast<void>(written);
                        std::fflush(stdout);
                        if (log_file.is_open()) {
                            log_file.write(buffer, bytes_read);
                            log_file.flush();
                        }
                    }

                    if (!result.timed_out) {
                        if (WIFEXITED(status)) {
                            result.exit_code = WEXITSTATUS(status);
                        } else if (WIFSIGNALED(status)) {
                            result.exit_code = 128 + WTERMSIG(status);
                        }
                    }

                    if (g_interrupt_count.load() > 0) {
                        result.cancelled = true;
                        result.error_message = "Cancelled by user";
                        result.exit_code = 130;
                    }
                    break;
                }
                sigprocmask(SIG_SETMASK, &orig_mask, nullptr);

                if (!child_done && g_interrupt_count.load() > 0) {
                    result.cancelled = true;
                    result.error_message = "Cancelled by user";
                    if (result.exit_code == 0) {
                        result.exit_code = 130;
                    }

                    const int count = g_interrupt_count.load();
                    if (count >= 2 && !cancel_message_printed) {
                        std::println(stderr, "\n[Batch] Second interrupt received. Force killing child process (PID {})...", pid);
                        cancel_message_printed = true;
                    } else if (!cancel_message_printed) {
                        std::println(stderr, "\n[Batch] Interrupt received. Waiting for child process to save and shut down cleanly... (Press Ctrl+C again to force kill)");
                        cancel_message_printed = true;
                    }
                }

                if (timeout.stopped_at || g_interrupt_count.load() == 0) {
                    const bool stopped = timeout.update(start_time, std::chrono::steady_clock::now(), params.timeout_minutes,
                                                        [&](const bool force) {
                                                            return detail::signal_batch_child(pid, force ? SIGKILL : SIGTERM);
                                                        });
                    if (timeout.stopped_at && !result.timed_out) {
                        result.timed_out = true;
                        result.exit_code = 124;
                        result.error_message = batch_timeout_message(params.timeout_minutes);
                        std::println(stderr, "\n[Batch] {}. Sending a stop signal; force termination follows after {} seconds.",
                                     result.error_message, detail::BATCH_STOP_GRACE.count());
                    }
                    if (!stopped) {
                        result.error_message = "Failed to stop timed-out child process";
                        break;
                    }
                }
            }

            if (!detail::cleanup_batch_group(pid)) {
                result.exit_code = 1;
                result.error_message = std::format("Failed to clean up child process group (system error {})", errno);
            }
            if (!child_done) {
                detail::signal_batch_child(pid, SIGKILL);
                while (waitpid(pid, nullptr, 0) == -1 && errno == EINTR) {}
            }
            close(pipe_fds[0]);
#endif

            const auto end_time = std::chrono::steady_clock::now();
            result.duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
            result.success = (result.exit_code == 0 && !result.cancelled && !result.timed_out);
            if (result.cancelled && result.exit_code == 0) {
                result.exit_code = 130;
            }
            if (result.timed_out && result.exit_code == 0) {
                result.exit_code = 124;
            }
            if (!result.success && result.error_message.empty()) {
                result.error_message = std::format("Process exited with code {}", result.exit_code);
            }

            return result;
        }

        bool write_summary_files(
            const lfs::core::param::BatchTrainingParameters& params,
            const std::vector<JobExecutionResult>& results,
            const double total_duration_seconds,
            const std::string& timestamp) {

            const auto [summary_path, report_path] = report_paths(params);

            json summary_json;
            summary_json["timestamp"] = timestamp;
            summary_json["total_jobs"] = results.size();

            size_t succeeded = 0, failed = 0, cancelled = 0;
            for (const auto& r : results) {
                if (r.cancelled)
                    ++cancelled;
                else if (r.success)
                    ++succeeded;
                else
                    ++failed;
            }

            summary_json["succeeded"] = succeeded;
            summary_json["failed"] = failed;
            summary_json["cancelled"] = cancelled;
            summary_json["total_duration_seconds"] = total_duration_seconds;
            summary_json["total_duration_formatted"] = format_duration(total_duration_seconds);

            json jobs_array = json::array();
            for (size_t i = 0; i < results.size(); ++i) {
                const auto& r = results[i];
                json job_obj;
                job_obj["index"] = i + 1;
                job_obj["name"] = r.name;
                job_obj["dataset_path"] = lfs::core::path_to_utf8(r.dataset_path);
                job_obj["output_path"] = lfs::core::path_to_utf8(r.output_path);
                job_obj["status"] = r.cancelled ? "CANCELLED" : (r.success ? "SUCCESS" : "FAILED");
                job_obj["exit_code"] = r.exit_code;
                job_obj["duration_seconds"] = r.duration_seconds;
                job_obj["duration_formatted"] = format_duration(r.duration_seconds);
                if (!r.error_message.empty()) {
                    job_obj["error"] = r.error_message;
                }
                jobs_array.push_back(std::move(job_obj));
            }
            summary_json["jobs"] = std::move(jobs_array);

            const bool summary_written = write_batch_report(summary_path, summary_json.dump(2) + "\n");
            std::ostringstream txt_out;
            txt_out << "================================================================================\n";
            txt_out << "LichtFeld Studio - Batch Training Report\n";
            txt_out << "Timestamp: " << timestamp << "\n";
            txt_out << "Total Jobs: " << results.size() << " | Succeeded: " << succeeded
                    << " | Failed: " << failed << " | Cancelled: " << cancelled << "\n";
            txt_out << "Total Duration: " << format_duration(total_duration_seconds) << "\n";
            txt_out << "================================================================================\n\n";

            for (size_t i = 0; i < results.size(); ++i) {
                const auto& r = results[i];
                const std::string status_str = r.cancelled ? "Cancelled" : (r.success ? "Success  " : "Failed   ");
                txt_out << std::format("[{:2d}] {} {:<24} ({}) -> {}\n",
                                       i + 1, status_str, r.name,
                                       format_duration(r.duration_seconds),
                                       lfs::core::path_to_utf8(r.output_path));
                if (!r.success && !r.error_message.empty()) {
                    txt_out << "     Error: " << r.error_message << "\n";
                }
            }
            txt_out << "\n================================================================================\n";
            const bool report_written = write_batch_report(report_path, txt_out.str());
            return summary_written && report_written;
        }

    } // namespace

    bool validate_batch_report_paths(const lfs::core::param::BatchTrainingParameters& params, std::string& error) {
        const auto [summary, report] = report_paths(params);
        std::error_code ec;
        if (path_key(summary) == path_key(report) || lfs::core::native_fs::equivalent(summary, report, ec)) {
            error = "Batch summary and report must point to different files";
            return false;
        }
        return true;
    }

    bool write_batch_report(const path& destination, const std::string_view contents) {
        try {
            if (destination.has_parent_path())
                lfs::core::native_fs::create_directories(destination.parent_path());
            std::ofstream out;
            out.exceptions(std::ios::failbit | std::ios::badbit);
            out.open(lfs::core::native_fs::native_path(destination));
            out << contents;
            out.flush();
            out.close();
            LOG_INFO("Wrote batch report to {}", lfs::core::path_to_utf8(destination));
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to write batch report {}: {}", lfs::core::path_to_utf8(destination), e.what());
            return false;
        }
    }

    std::vector<lfs::core::param::BatchJobItem> resolve_batch_jobs(
        const lfs::core::param::BatchTrainingParameters& params) {
        return resolve_jobs_internal(params);
    }

    std::vector<std::string> build_child_args(
        const lfs::core::param::BatchJobItem& job,
        const lfs::core::param::BatchTrainingParameters& params) {
        std::vector<std::string> args;
        args.reserve(16 + params.shared_args.size() + job.extra_args.size());

        args.push_back("--headless");
        args.push_back("-d");
        args.push_back(lfs::core::path_to_utf8(job.dataset_path));
        args.push_back("-o");
        args.push_back(lfs::core::path_to_utf8(job.output_path));

        std::unordered_set<std::string> overridden;
        visit_training_args(job.extra_args, [&](size_t, size_t, const auto* flag) {
            if (flag)
                overridden.insert(flag->canonical_name);
        });
        if (job.config_file && !job.config_file->empty())
            overridden.insert(lfs::core::args::training_flag("--config")->canonical_name);
        visit_training_args(params.shared_args, [&](const size_t i, const size_t count, const auto* flag) {
            if (!flag || !overridden.contains(flag->canonical_name))
                args.insert(args.end(), params.shared_args.begin() + i, params.shared_args.begin() + i + count);
        });

        // The job's own config and arguments come last so they win over shared ones
        if (job.config_file && !job.config_file->empty()) {
            args.push_back("--config");
            args.push_back(lfs::core::path_to_utf8(*job.config_file));
        }
        for (const auto& extra : job.extra_args) {
            args.push_back(extra);
        }

        return args;
    }

    bool validate_job_arguments(
        const lfs::core::param::BatchJobItem& job,
        const lfs::core::param::BatchTrainingParameters& params,
        std::string& error) {
        bool reserved = false;
        const auto check_reserved = [&](const std::vector<std::string>& args) {
            visit_training_args(args, [&](const size_t i, size_t, const auto* flag) {
                if (flag && (flag->name == "-d" || flag->name == "--data-path" ||
                             flag->name == "-o" || flag->name == "--output-path")) {
                    error = std::format("Reserved training option '{}': use the batch input/output options or the job's dataset/output fields", args[i]);
                    reserved = true;
                }
            });
        };
        check_reserved(params.shared_args);
        check_reserved(job.extra_args);
        if (reserved)
            return false;

        const auto child_args = build_child_args(job, params);
        std::vector<std::string> validate_args;
        validate_args.reserve(child_args.size() + 1);
        validate_args.push_back("lichtfeld-studio");
        for (const auto& a : child_args) {
            validate_args.push_back(a);
        }
        std::vector<const char*> validate_argv;
        validate_argv.reserve(validate_args.size());
        for (const auto& a : validate_args) {
            validate_argv.push_back(a.c_str());
        }

        auto val_res = lfs::core::args::parse_args_and_params(
            static_cast<int>(validate_argv.size()), validate_argv.data(), /*create_output_directory=*/false, /*validate_only=*/true);
        if (!val_res) {
            error = val_res.error();
            return false;
        }
        const auto& parsed = **val_res;
        if (!parsed.optimization.headless || !parsed.view_paths.empty() || parsed.project_path ||
            parsed.resume_checkpoint || parsed.resume_project || parsed.render_path ||
            parsed.import_cameras_path || parsed.evaluate_only) {
            error = "Batch jobs must select headless training of the job's dataset; viewer, resume and other modes are not supported";
            return false;
        }
        // Also catch path overrides embedded in short-option clusters.
        const auto& dataset = parsed.dataset_project ? *parsed.dataset_project : parsed.dataset.data_path;
        if (resolved_path(dataset) != resolved_path(job.dataset_path) ||
            resolved_path(parsed.dataset.output_path) != resolved_path(job.output_path)) {
            error = "Reserved dataset/output override: use the batch input/output options or the job's dataset/output fields";
            return false;
        }
        return true;
    }

    std::string batch_timeout_message(const int minutes) {
        return std::format("Job timed out after {} {}", minutes, minutes == 1 ? "minute" : "minutes");
    }

    int run_batch_training(const lfs::core::param::BatchTrainingParameters& params) {
        ScopedSignalHandler signal_handler;

        path executable;
        try {
            executable = lfs::core::getExecutablePath();
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to locate LichtFeld Studio executable: {}", e.what());
            std::println(stderr, "Error: Failed to locate LichtFeld Studio executable: {}", e.what());
            return 1;
        }

        std::vector<lfs::core::param::BatchJobItem> jobs;
        try {
            std::string error;
            if (!validate_batch_report_paths(params, error)) {
                LOG_ERROR("{}", error);
                return 1;
            }
            jobs = resolve_batch_jobs(params);
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to resolve batch inputs: {}", e.what());
            return 1;
        }
        if (jobs.empty()) {
            LOG_ERROR("Batch training: No valid datasets found.");
            std::println(stderr, "\nError: No valid datasets found.");
            std::println(stderr, "Specify dataset directories containing sparse/ or transforms.json, or pass a --manifest file.");
            std::println(stderr, "Run 'LichtFeld-Studio batch-train --help' for details.\n");
            return 1;
        }

        for (const auto& job : jobs) {
            std::string job_error;
            if (!validate_job_arguments(job, params, job_error)) {
                LOG_ERROR("Job '{}' has invalid training arguments: {}", job.name, job_error);
                std::println(stderr, "\nError in job '{}' arguments: {}\n", job.name, job_error);
                return 1;
            }
        }

        if (params.dry_run) {
            std::println("\n================================================================================");
            std::println("LichtFeld Studio - Batch Training Dry Run (Resolved Queue: {} jobs)", jobs.size());
            std::println("================================================================================");
            if (!params.shared_args.empty()) {
                std::string shared_str;
                for (const auto& a : params.shared_args) {
                    shared_str += a + " ";
                }
                std::println("Shared arguments: {}", shared_str);
            }
            if (params.timeout_minutes > 0) {
                std::println("Timeout per job: {} minutes", params.timeout_minutes);
            }
            std::println("Continue on failure: {}", params.continue_on_failure ? "yes" : "no");
            std::println("--------------------------------------------------------------------------------");

            for (size_t i = 0; i < jobs.size(); ++i) {
                const auto& job = jobs[i];
                std::println("[{:2d}] Job: {}", i + 1, job.name);
                std::println("     Dataset: {}", lfs::core::path_to_utf8(job.dataset_path));
                std::println("     Output:  {}", lfs::core::path_to_utf8(job.output_path));
                if (job.config_file) {
                    std::println("     Config:  {}", lfs::core::path_to_utf8(*job.config_file));
                }
                if (!job.extra_args.empty()) {
                    std::string extra_str;
                    for (const auto& a : job.extra_args)
                        extra_str += a + " ";
                    std::println("     Extra:   {}", extra_str);
                }
            }
            std::println("================================================================================");
            std::println("Dry run complete. {} job(s) ready to train.\n", jobs.size());
            return 0;
        }

        const auto timestamp = current_iso_timestamp();
        const auto batch_start_time = std::chrono::steady_clock::now();

        std::println("\n================================================================================");
        std::println("LichtFeld Studio - Batch Training");
        std::println("Queue size: {} dataset(s) | Started at: {}", jobs.size(), timestamp);
        if (!params.shared_args.empty()) {
            std::string shared_str;
            for (const auto& a : params.shared_args) {
                shared_str += a + " ";
            }
            std::println("Shared options: {}", shared_str);
        }
        std::println("Failure policy: {}", params.continue_on_failure ? "Continue next on failure" : "Stop on first failure");
        std::println("================================================================================\n");

        std::vector<JobExecutionResult> results;
        results.reserve(jobs.size());

        for (size_t i = 0; i < jobs.size(); ++i) {
            if (g_interrupt_count.load() > 0) {
                std::println("\n[Batch] Training interrupted by user. Aborting remaining {} job(s)...", jobs.size() - i);
                cancel_remaining_jobs(jobs, i, results, "Cancelled before start");
                break;
            }

            const auto& job = jobs[i];
            std::println("\n================================================================================");
            std::println("[Batch Job {}/{}] Starting: {}", i + 1, jobs.size(), job.name);
            std::println("Dataset: {}", lfs::core::path_to_utf8(job.dataset_path));
            std::println("Output:  {}", lfs::core::path_to_utf8(job.output_path));
            std::println("Time:    {}", current_iso_timestamp());
            std::println("================================================================================");

            auto res = execute_job(job, params, executable);
            results.push_back(res);

            if (res.success) {
                std::println("\n[Batch Job {}/{}] Completed in {}: {}",
                             i + 1, jobs.size(), format_duration(res.duration_seconds), res.name);
            } else if (res.cancelled) {
                std::println("\n[Batch Job {}/{}] Cancelled: {}", i + 1, jobs.size(), res.name);
                cancel_remaining_jobs(jobs, i + 1, results, "Cancelled by user");
                break;
            } else {
                std::println("\n[Batch Job {}/{}] Failed (exit code {}): {}",
                             i + 1, jobs.size(), res.exit_code, res.name);
                if (!res.error_message.empty()) {
                    std::println("  Detail: {}", res.error_message);
                }

                if (!params.continue_on_failure) {
                    std::println("\n[Batch] Job failed and continue-on-failure is disabled. Stopping batch.");
                    cancel_remaining_jobs(jobs, i + 1, results, "Cancelled due to previous failure");
                    break;
                }
            }
        }

        const auto batch_end_time = std::chrono::steady_clock::now();
        const auto total_duration_seconds = std::chrono::duration<double>(batch_end_time - batch_start_time).count();

        size_t succeeded_count = 0, failed_count = 0, cancelled_count = 0;
        for (const auto& r : results) {
            if (r.cancelled)
                ++cancelled_count;
            else if (r.success)
                ++succeeded_count;
            else
                ++failed_count;
        }

        std::println("\n================================================================================");
        std::println("Batch Training Summary");
        std::println("Total: {} | Succeeded: {} | Failed: {} | Cancelled: {}",
                     results.size(), succeeded_count, failed_count, cancelled_count);
        std::println("Total Time: {}", format_duration(total_duration_seconds));
        std::println("--------------------------------------------------------------------------------");

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            const std::string status_label = r.cancelled ? "Cancelled" : (r.success ? "Success  " : "Failed   ");
            std::println("[{:2d}] {} {:<24} ({}) -> {}",
                         i + 1, status_label, r.name,
                         format_duration(r.duration_seconds),
                         lfs::core::path_to_utf8(r.output_path));
        }
        std::println("================================================================================");

        const bool reports_written = write_summary_files(params, results, total_duration_seconds, timestamp);

        return (failed_count > 0 || cancelled_count > 0 || !reports_written) ? 1 : 0;
    }

    int run_auto_tile_training(const lfs::core::param::TrainingParameters& params,
                               const std::vector<std::string>& command_line) {
        using lfs::core::path_to_utf8;
        const auto& dataset = params.dataset.data_path;
        const auto& output = params.dataset.output_path;
        if (dataset.empty() || output.empty() || !lfs::io::Loader::isDatasetPath(dataset)) {
            LOG_ERROR("Auto-tiling needs a dataset folder and an output directory");
            return 1;
        }
        if (params.dataset.centralize_dataset != "off") {
            LOG_ERROR("Auto-tiling currently requires --centralize off so crop boxes and exports share dataset coordinates");
            return 1;
        }
        if (!params.auto_tile_crop && params.auto_tile_merge) {
            LOG_ERROR("A merged auto-tile export requires the strict core crop; remove --no-tile-crop or add --no-tile-merge");
            return 1;
        }
        if (params.resume_checkpoint || params.resume_project || params.dataset_project || params.cropbox ||
            params.init_path || params.optimization.random || !params.add_splat_paths.empty()) {
            LOG_ERROR("Auto-tiling owns the initialization and requires fresh dataset training without a manual crop box, --init/--random, or added splats");
            return 1;
        }
        // Covers planning, dense-mask generation and all child jobs so a GUI
        // cancellation never waits for an uninterruptible parent-side phase.
        ScopedSignalHandler signal_handler;

        // Preserve ordinary training options, including --config, without forwarding the
        // parent's dataset/output and tiling options to the tile processes.
        std::vector<std::string> shared;
        for (size_t n = 1; n < command_line.size();) {
            const auto* flag = lfs::core::args::training_flag(command_line[n]);
            if (!flag) {
                LOG_ERROR("Cannot forward auto-tiling option: {}", command_line[n]);
                return 1;
            }
            const bool separate_value = flag->takes_value &&
                                        command_line[n] == flag->name && n + 1 < command_line.size();
            const size_t count = separate_value ? 2 : 1;
            const std::string_view name = flag->name;
            const bool parent_only = name == "-d" || name == "--data-path" || name == "-o" ||
                                     name == "--output-path" || name == "--output-name" ||
                                     name == "--export" || name == "--headless" ||
                                     name == "--crop" ||
                                     name == "--auto-tile" || name == "--tile-size" ||
                                     name == "--tile-overlap" || name == "--no-tile-pca" ||
                                     name == "--no-tile-crop" || name == "--no-tile-merge" ||
                                     name == "--tile-outside-loss-weight" || name == "--tile-plan" ||
                                     name == "--tile-align" || name == "--tile-source" ||
                                     name == "--tile-quantile" || name == "--tile-coarse-iterations" ||
                                     name == "--tile-coarse-resize" || name == "--tile-depth-masks" ||
                                     name == "--tile-camera-uids" || name == "--tile-exposure-gains" ||
                                     name == "--tile-contextbox" || name == "--tile-child" ||
                                     name == "--tile-min-cameras" ||
                                     name == "--tile-min-multiview-cells" ||
                                     name == "--no-tile-merge-infeasible" ||
                                     name == "--tile-boundary-padding" ||
                                     name == "--tile-visibility-mask-resolution" ||
                                     name == "--tile-visibility-min-alpha" ||
                                     name == "--tile-visibility-foreign-weight" ||
                                     name == "--no-tile-require-dense-masks" ||
                                     name == "--tile-lock-exposure-correction" ||
                                     name == "--train";
            if (!parent_only)
                shared.insert(shared.end(), command_line.begin() + n, command_line.begin() + n + count);
            n += count;
        }

        std::vector<lfs::core::TileSpec> tiles;
        std::vector<lfs::core::TileCameraEvidence> camera_evidence;
        std::vector<lfs::core::TilePlanDiagnostics> planning_diagnostics;
        bool plan_has_boundary_padding = false;
        try {
            if (params.tile_plan_path) {
                std::ifstream input(lfs::core::native_fs::native_path(*params.tile_plan_path));
                if (!input) throw std::runtime_error("Cannot open tile plan");
                const auto plan = json::parse(input);
                if (!plan.contains("tiles") || !plan["tiles"].is_array())
                    throw std::runtime_error("Tile plan needs a tiles array");
                if (plan.contains("boundary_padding_applied") &&
                    !plan["boundary_padding_applied"].is_boolean())
                    throw std::runtime_error("Tile plan boundary_padding_applied must be a boolean");
                plan_has_boundary_padding = plan.value("boundary_padding_applied", false);
                if (plan["tiles"].size() == 0 || plan["tiles"].size() > 1000)
                    throw std::runtime_error("Tile plan must have 1 to 1000 boxes");
                std::unordered_set<std::string> tile_names;
                for (const auto& item : plan["tiles"]) {
                    const auto lo = item.at("min").get<std::array<float, 3>>();
                    const auto hi = item.at("max").get<std::array<float, 3>>();
                    const auto rotation = item.at("rotation_degrees").get<std::array<float, 3>>();
                    lfs::core::TileSpec tile;
                    tile.i = item.at("i").get<int>();
                    tile.j = item.at("j").get<int>();
                    tile.name = std::format("tile_{}_{}", tile.i, tile.j);
                    tile.min = {lo[0], lo[1], lo[2]};
                    tile.max = {hi[0], hi[1], hi[2]};
                    tile.rotation_degrees = {rotation[0], rotation[1], rotation[2]};
                    if (item.contains("context_min") || item.contains("context_max")) {
                        if (!item.contains("context_min") || !item.contains("context_max"))
                            throw std::runtime_error("Tile plan needs both context_min and context_max");
                        const auto context_lo = item.at("context_min").get<std::array<float, 3>>();
                        const auto context_hi = item.at("context_max").get<std::array<float, 3>>();
                        tile.context_min = {context_lo[0], context_lo[1], context_lo[2]};
                        tile.context_max = {context_hi[0], context_hi[1], context_hi[2]};
                        tile.has_context_bounds = true;
                        for (int axis = 0; axis < 3; ++axis)
                            if (!std::isfinite(context_lo[axis]) || !std::isfinite(context_hi[axis]) ||
                                context_lo[axis] >= context_hi[axis])
                                throw std::runtime_error("Tile plan contains invalid context bounds");
                    }
                    for (int axis = 0; axis < 3; ++axis) {
                        if (!std::isfinite(lo[axis]) || !std::isfinite(hi[axis]) ||
                            !std::isfinite(rotation[axis]) || lo[axis] >= hi[axis])
                            throw std::runtime_error("Tile plan contains invalid bounds or angles");
                    }
                    if (tile.i < 0 || tile.j < 0 || tile.i > 99 || tile.j > 99 ||
                        (params.auto_tile_grid &&
                         (tile.i >= params.auto_tile_grid->first ||
                          tile.j >= params.auto_tile_grid->second)))
                        throw std::runtime_error("Tile plan contains an invalid grid index");
                    if (!tile_names.insert(tile.name).second)
                        throw std::runtime_error("Tile plan contains duplicate grid indices");
                    tiles.push_back(tile);
                }
                if (params.auto_tile_grid &&
                    tiles.size() != static_cast<size_t>(params.auto_tile_grid->first * params.auto_tile_grid->second))
                    throw std::runtime_error("Tile plan count differs from --auto-tile grid");
                if (params.auto_tile_count > 0 && tiles.size() != static_cast<size_t>(params.auto_tile_count))
                    throw std::runtime_error("Tile plan count differs from --auto-tile count");
            } else {
            lfs::io::LoadOptions load_options;
            load_options.images_folder = params.dataset.images;
            load_options.min_track_length = params.dataset.min_track_length;
            auto loaded = lfs::io::Loader::create()->load(dataset, load_options);
            if (!loaded)
                throw std::runtime_error(std::format("Cannot inspect dataset: {}", loaded.error().format()));
            auto* scene = std::get_if<lfs::io::LoadedScene>(&loaded->data);
            if (!scene)
                throw std::runtime_error("Auto-tiling expects a dataset with cameras and optional sparse points");
            std::vector<glm::vec3> centers;
            centers.reserve(scene->cameras.size());
            for (const auto& camera : scene->cameras) {
                if (!camera || !camera->has_image()) continue;
                const auto position = camera->cam_position().cpu().contiguous();
                if (position.numel() >= 3) {
                    const auto* p = position.ptr<float>();
                    centers.emplace_back(p[0], p[1], p[2]);
                }
            }
            const lfs::core::PointCloud empty_points;
            const auto& points = scene->point_cloud ? *scene->point_cloud : empty_points;
            camera_evidence = lfs::core::collect_tile_camera_evidence(scene->cameras);
            lfs::core::OrientationMode mode = lfs::core::OrientationMode::Obb3D;
            if (params.auto_tile_align_mode == "planar_yaw" || params.auto_tile_align_mode == "yaw") {
                mode = lfs::core::OrientationMode::PlanarYaw;
            } else if (params.auto_tile_align_mode == "planar_pitch" || params.auto_tile_align_mode == "pitch") {
                mode = lfs::core::OrientationMode::PlanarPitch;
            } else if (params.auto_tile_align_mode == "planar_roll" || params.auto_tile_align_mode == "roll") {
                mode = lfs::core::OrientationMode::PlanarRoll;
            } else if (params.auto_tile_align_mode == "axis_aligned" || params.auto_tile_align_mode == "none" || !params.auto_tile_align_pca) {
                mode = lfs::core::OrientationMode::AxisAligned;
            }
            lfs::core::BoundsSource source = lfs::core::BoundsSource::Points;
            if (params.auto_tile_source == "points") {
                source = lfs::core::BoundsSource::Points;
            } else if (params.auto_tile_source == "cameras") {
                source = lfs::core::BoundsSource::Cameras;
            }
            auto bounds = lfs::core::compute_robust_scene_bounds_obb(
                points, centers, params.auto_tile_quantile, mode, source);
            const auto outer_bounds = lfs::core::compute_outer_scene_bounds(
                points, centers, bounds, params.auto_tile_quantile, source);
            const auto extent = bounds.max - bounds.min;
            if (params.auto_tile_quantile > 0.0f) {
                // Keep edge geometry and the full local Z range just beyond the quantiles.
                const glm::vec3 padding = glm::max(extent * 0.02f, glm::vec3(0.01f));
                bounds.min -= padding;
                bounds.max += padding;
            }
            lfs::core::TileGridConfig grid;
            if (params.auto_tile_grid) {
                grid.columns = params.auto_tile_grid->first;
                grid.rows = params.auto_tile_grid->second;
            }
            grid.target_size = params.auto_tile_size;
            grid.overlap_ratio = params.tile_overlap;
            grid.align_pca = params.auto_tile_align_pca;
            grid.orientation_mode = mode;
            grid.quantile = params.auto_tile_quantile;
            tiles = params.auto_tile_count > 0
                ? lfs::core::generate_adaptive_tiles(bounds, params.auto_tile_count,
                      params.tile_overlap, outer_bounds, points, camera_evidence,
                      &planning_diagnostics,
                      {.resize_factor = params.dataset.resize_factor,
                       .max_width = params.dataset.max_width})
                : lfs::core::generate_tile_grid(bounds, grid, outer_bounds);
            }
            if (camera_evidence.empty()) {
                lfs::io::LoadOptions load_options;
                load_options.images_folder = params.dataset.images;
                load_options.min_track_length = params.dataset.min_track_length;
                auto loaded = lfs::io::Loader::create()->load(dataset, load_options);
                if (!loaded)
                    throw std::runtime_error(std::format("Cannot inspect dataset cameras: {}", loaded.error().format()));
                auto* scene = std::get_if<lfs::io::LoadedScene>(&loaded->data);
                if (!scene) throw std::runtime_error("Tile plan needs a dataset with cameras");
                camera_evidence = lfs::core::collect_tile_camera_evidence(scene->cameras);
            }
            if (camera_evidence.size() < 2)
                throw std::runtime_error("Auto-tiling needs at least two usable cameras");
        } catch (const std::exception& error) {
            LOG_ERROR("Cannot partition dataset: {}", error.what());
            return 1;
        }

        const size_t requested_tile_count = tiles.size();
        std::vector<std::vector<int>> tile_cameras;
        std::vector<lfs::core::TileCameraSelectionDiagnostics> camera_selection_diagnostics;
        std::vector<TileFeasibility> tile_feasibility;
        std::vector<std::vector<std::string>> coalesced_from;
        coalesced_from.reserve(tiles.size());
        for (const auto& tile : tiles)
            coalesced_from.push_back({tile.name});
        std::unordered_map<int, const lfs::core::TileCameraEvidence*> views_by_uid;
        for (const auto& view : camera_evidence)
            if (!views_by_uid.emplace(view.uid, &view).second) {
                LOG_ERROR("Dataset contains duplicate camera UID {}", view.uid);
                return 1;
            }
        const auto assign_tile_cameras = [&] {
            tile_cameras.clear();
            camera_selection_diagnostics.clear();
            tile_feasibility.clear();
            tile_cameras.reserve(tiles.size());
            camera_selection_diagnostics.reserve(tiles.size());
            tile_feasibility.reserve(tiles.size());
            for (const auto& tile : tiles) {
                lfs::core::TileCameraSelectionDiagnostics diagnostics;
                auto selected = lfs::core::select_tile_camera_uids(
                    tile, camera_evidence, &diagnostics);
                tile_feasibility.push_back(assess_tile_feasibility(selected, diagnostics, params));
                tile_cameras.push_back(std::move(selected));
                camera_selection_diagnostics.push_back(std::move(diagnostics));
            }
        };
        assign_tile_cameras();

        // Make a rectangular closure around an infeasible cell.  When a
        // partial neighbour prevents a direct union, split both rectangles on
        // their shared local-frame boundaries, merge the aligned pair, and
        // leave the remaining fragments explicit.  This preserves ownership;
        // unlike an L-shaped bounding box it never steals a third tile's core.
        json topology_operations = json::array();
        std::string topology_blocker;
        size_t fragment_serial = 0;
        size_t topology_operation_count = 0;
        const size_t max_topology_operations = std::max<size_t>(64, requested_tile_count * 8);
        while (true) {
            if (g_interrupt_count.load() > 0) {
                LOG_WARN("Auto-tile feasibility processing cancelled");
                return 1;
            }
            const auto unfeasible = std::ranges::find_if(
                tile_feasibility, [](const TileFeasibility& status) { return !status.feasible; });
            if (unfeasible == tile_feasibility.end()) break;
            const size_t failing_index = static_cast<size_t>(unfeasible - tile_feasibility.begin());
            if (!params.tile_merge_infeasible || tiles.size() <= 1) break;

            size_t best_index = tiles.size();
            std::optional<RectangularClosure> best_closure;
            float best_distance = std::numeric_limits<float>::infinity();
            size_t best_fragment_count = std::numeric_limits<size_t>::max();
            const glm::vec3 failing_center = (tiles[failing_index].min + tiles[failing_index].max) * 0.5f;
            for (size_t candidate = 0; candidate < tiles.size(); ++candidate) {
                if (candidate == failing_index)
                    continue;
                auto closure = make_rectangular_closure(tiles[failing_index], tiles[candidate]);
                if (!closure) continue;
                const glm::vec3 candidate_center =
                    (tiles[candidate].min + tiles[candidate].max) * 0.5f;
                const float distance = glm::length(candidate_center - failing_center);
                const size_t fragment_count = closure->failing_fragments.size() +
                                              closure->neighbour_fragments.size();
                if (fragment_count < best_fragment_count ||
                    (fragment_count == best_fragment_count &&
                     (distance < best_distance ||
                      (distance == best_distance &&
                     (best_index == tiles.size() ||
                      tiles[candidate].name < tiles[best_index].name))))) {
                    best_distance = distance;
                    best_index = candidate;
                    best_fragment_count = fragment_count;
                    best_closure = std::move(closure);
                }
            }
            if (best_index == tiles.size()) break;
            if (++topology_operation_count > max_topology_operations) {
                topology_blocker = std::format(
                    "stopped after {} rectangular-closure operations", max_topology_operations);
                break;
            }

            auto closure = std::move(*best_closure);
            const size_t prospective_count = tiles.size() +
                closure.failing_fragments.size() + closure.neighbour_fragments.size() - 3;
            if (prospective_count > 1000) {
                topology_blocker = "rectangular closure would exceed the 1000 tile safety limit";
                break;
            }

            const std::string failing_name = tiles[failing_index].name;
            const std::string neighbour_name = tiles[best_index].name;
            auto merged = coalesce_tiles(
                closure.failing_fragments[closure.failing_fragment],
                closure.neighbour_fragments[closure.neighbour_fragment]);
            // Keep the infeasible tile's name for the joined region. Its
            // remaining pieces receive stable fragment names in the manifest.
            merged.name = failing_name;
            auto merged_origins = coalesced_from[failing_index];
            merged_origins.insert(merged_origins.end(),
                                  coalesced_from[best_index].begin(),
                                  coalesced_from[best_index].end());
            std::vector<lfs::core::TileSpec> fragments;
            std::vector<std::vector<std::string>> fragment_origins;
            const auto append_remaining = [&](const std::vector<lfs::core::TileSpec>& source,
                                              const size_t merged_fragment,
                                              const std::vector<std::string>& origins,
                                              const std::string_view base_name) {
                for (size_t index = 0; index < source.size(); ++index) {
                    if (index == merged_fragment) continue;
                    auto fragment = source[index];
                    fragment.name = std::format("{}__fragment_{}", base_name, fragment_serial++);
                    fragments.push_back(std::move(fragment));
                    fragment_origins.push_back(origins);
                }
            };
            append_remaining(closure.failing_fragments, closure.failing_fragment,
                             coalesced_from[failing_index], failing_name);
            append_remaining(closure.neighbour_fragments, closure.neighbour_fragment,
                             coalesced_from[best_index], neighbour_name);
            tiles[failing_index] = std::move(merged);
            coalesced_from[failing_index] = std::move(merged_origins);
            tiles.erase(tiles.begin() + static_cast<std::ptrdiff_t>(best_index));
            coalesced_from.erase(coalesced_from.begin() + static_cast<std::ptrdiff_t>(best_index));
            tiles.insert(tiles.end(), std::make_move_iterator(fragments.begin()),
                         std::make_move_iterator(fragments.end()));
            coalesced_from.insert(coalesced_from.end(),
                                  std::make_move_iterator(fragment_origins.begin()),
                                  std::make_move_iterator(fragment_origins.end()));
            // Adaptive work estimates describe the pre-coalescing leaves and
            // must not be incorrectly attached to their union.
            planning_diagnostics.clear();
            topology_operations.push_back({
                {"kind", best_fragment_count == 2 ? "coalesce" : "rectangular_closure"},
                {"failing", failing_name}, {"neighbour", neighbour_name},
                {"join_axis", closure.join_axis},
                {"failing_fragments", closure.failing_fragments.size()},
                {"neighbour_fragments", closure.neighbour_fragments.size()},
                {"resolved_tile_count", tiles.size()}});
            LOG_WARN("Auto-tile feasibility: joined '{}' with '{}' via {} ({} requested -> {} current tiles)",
                     failing_name, neighbour_name,
                     best_fragment_count == 2 ? "direct rectangular coalescing" : "rectangular closure",
                     requested_tile_count, tiles.size());
            assign_tile_cameras();
        }

        json feasibility_report;
        feasibility_report["requested_tile_count"] = requested_tile_count;
        feasibility_report["resolved_tile_count"] = tiles.size();
        feasibility_report["minimum_cameras"] = params.tile_min_cameras;
        feasibility_report["minimum_multi_view_cells"] = params.tile_min_multi_view_cells;
        feasibility_report["coalesce_infeasible"] = params.tile_merge_infeasible;
        feasibility_report["topology_operations"] = topology_operations;
        if (!topology_blocker.empty())
            feasibility_report["topology_blocker"] = topology_blocker;
        feasibility_report["tiles"] = json::array();
        bool all_tiles_feasible = true;
        for (size_t index = 0; index < tiles.size(); ++index) {
            const auto& status = tile_feasibility[index];
            feasibility_report["tiles"].push_back({
                {"name", tiles[index].name},
                {"feasible", status.feasible},
                {"camera_count", status.camera_count},
                {"multi_view_cells", status.multi_view_cells},
                {"reasons", status.reasons},
                {"coalesced_from", coalesced_from[index]}});
            all_tiles_feasible = all_tiles_feasible && status.feasible;
        }
        if (!write_batch_report(output / "tile_feasibility.json", feasibility_report.dump(2) + "\n"))
            return 1;
        if (!all_tiles_feasible) {
            for (size_t index = 0; index < tiles.size(); ++index) {
                if (!tile_feasibility[index].feasible) {
                    LOG_ERROR("Tile '{}' remains infeasible: {}", tiles[index].name,
                              tile_feasibility[index].reasons.empty()
                                  ? "unknown feasibility error"
                                  : tile_feasibility[index].reasons.front());
                }
            }
            LOG_ERROR("Auto-tile feasibility gate failed before the coarse pass; see {}",
                      path_to_utf8(output / "tile_feasibility.json"));
            return 1;
        }
        if (tiles.size() != requested_tile_count) {
            LOG_WARN("Auto-tile feasibility resolved {} requested tiles to {} rectangular tiles",
                     requested_tile_count, tiles.size());
        }

        // Keep the unextended cores in the manifest as the exact region used
        // for camera feasibility.  Boundary crop padding is applied only now.
        const std::vector<lfs::core::TileSpec> camera_assignment_tiles = tiles;
        const auto boundary_extensions = plan_has_boundary_padding
            ? std::vector<std::array<bool, 4>>(tiles.size())
            : extend_boundary_tile_cores(tiles, params.tile_boundary_padding);
        if (plan_has_boundary_padding && params.tile_boundary_padding > 0.0f) {
            LOG_INFO("Auto-tile plan already contains boundary padding; not applying it a second time");
        }

        // Solve this once from SfM before the coarse pass.  The same normalized
        // images are then seen by the global prior and every local optimizer;
        // learned exposure correction, when enabled, only needs to model the
        // residual instead of letting tiles choose incompatible baselines.
        lfs::core::TileExposureCalibration exposure;
        size_t exposure_samples = 0;
        try {
            LOG_INFO("Estimating global tile exposure from shared SfM tracks");
            const auto samples = lfs::core::sample_tile_exposure(
                camera_evidence, 2048, [] { return g_interrupt_count.load() > 0; });
            if (g_interrupt_count.load() > 0) return 1;
            exposure_samples = samples.size();
            std::vector<int> camera_uids;
            camera_uids.reserve(camera_evidence.size());
            for (const auto& view : camera_evidence) camera_uids.push_back(view.uid);
            exposure = lfs::core::estimate_tile_exposure(samples, camera_uids);
        } catch (const std::exception& error) {
            LOG_ERROR("Cannot estimate global tile exposure: {}", error.what());
            return 1;
        }
        if (exposure.connected_cameras < 2)
            LOG_WARN("Insufficient shared SfM image evidence for exposure calibration; using unchanged images");
        else
            LOG_INFO("Global exposure calibrated {} of {} images from {} shared points and {} reliable view pairs",
                     exposure.connected_cameras, exposure.cameras.size(),
                     exposure.shared_points, exposure.reliable_edges);
        json exposure_file = {
            {"version", 1},
            {"dataset", path_to_utf8(dataset)},
            {"method", "sfm_shared_track_log_luminance"},
            {"domain", "encoded_rgb"},
            {"sampled_observations", exposure_samples},
            {"shared_points", exposure.shared_points},
            {"reliable_view_pairs", exposure.reliable_edges},
            {"connected_cameras", exposure.connected_cameras},
            {"cameras", json::array()}};
        for (const auto& gain : exposure.cameras) {
            const auto found = views_by_uid.find(gain.camera_uid);
            if (found == views_by_uid.end()) {
                LOG_ERROR("Exposure calibration returned unknown camera UID {}", gain.camera_uid);
                return 1;
            }
            exposure_file["cameras"].push_back({
                {"uid", gain.camera_uid},
                {"image", found->second->image_name},
                {"gain", gain.gain},
                {"shared_points", gain.shared_points}});
        }
        const path exposure_path = output / "tile_exposure.json";
        if (!write_batch_report(exposure_path, exposure_file.dump(2) + "\n")) return 1;

        lfs::core::param::BatchTrainingParameters batch;
        batch.output_dir = output;
        batch.shared_args = shared;
        batch.continue_on_failure = false;
        std::filesystem::path executable;
        try {
            executable = lfs::core::getExecutablePath();
        } catch (const std::exception& error) {
            LOG_ERROR("{}", error.what());
            return 1;
        }
        // CityGaussian/Sun-style global prior: one inexpensive global model
        // fixes the coordinate frame and supplies geometry to every local run.
        lfs::core::param::BatchJobItem coarse_job;
        coarse_job.dataset_path = dataset;
        coarse_job.name = "coarse";
        coarse_job.output_path = output / "coarse";
        json coarse_allowlist = {{"dataset", path_to_utf8(dataset)}, {"cameras", json::array()}};
        for (const auto& view : camera_evidence)
            coarse_allowlist["cameras"].push_back({{"uid", view.uid}, {"image", view.image_name}});
        const path coarse_camera_path = coarse_job.output_path / "coarse_cameras.json";
        if (!write_batch_report(coarse_camera_path, coarse_allowlist.dump(2) + "\n")) return 1;
        coarse_job.extra_args = {
            "--iter", std::format("{}", params.tile_coarse_iterations),
            "--resize_factor", std::format("{}", params.tile_coarse_resize_factor),
            "--tile-camera-uids", path_to_utf8(coarse_camera_path),
            "--tile-exposure-gains", path_to_utf8(exposure_path),
            "--tile-child", "--output-name", "coarse", "--export", "ply"};
        if (params.tile_lock_exposure_correction) {
            // A PLY transfers Gaussian appearance but not optimizer-owned
            // appearance modules. Keep the coarse model in the same globally
            // calibrated photometric space as locked tile workers.
            coarse_job.extra_args.push_back("--no-exposure-correction");
        }
        std::string coarse_error;
        if (!validate_job_arguments(coarse_job, batch, coarse_error)) {
            LOG_ERROR("Invalid coarse auto-tile pass: {}", coarse_error);
            return 1;
        }
        if (g_interrupt_count.load() > 0) return 1;
        std::println("[Auto-tile] Training global coarse prior ({} iterations, resize {})",
                     params.tile_coarse_iterations, params.tile_coarse_resize_factor);
        const auto coarse_result = execute_job(coarse_job, batch, executable);
        if (!coarse_result.success) {
            LOG_ERROR("Global coarse pass failed: {}", coarse_result.error_message);
            return 1;
        }
        const path coarse_export = coarse_job.output_path / "coarse.ply";
        if (!lfs::core::native_fs::exists(coarse_export)) {
            LOG_ERROR("Global coarse pass completed without its expected PLY export: {}", path_to_utf8(coarse_export));
            return 1;
        }

        std::shared_ptr<lfs::core::SplatData> coarse_model;
        lfs::core::Tensor coarse_means_cpu;
        std::vector<glm::vec3> coarse_points;
        try {
            auto loaded = lfs::io::Loader::create()->load(coarse_export);
            if (!loaded) throw std::runtime_error(loaded.error().format());
            const auto* splat = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&loaded->data);
            if (!splat || !*splat) throw std::runtime_error("coarse export is not a Gaussian splat PLY");
            coarse_model = *splat;
            coarse_means_cpu = coarse_model->get_means().cpu().contiguous();
            if (coarse_means_cpu.ndim() != 2 || coarse_means_cpu.size(1) < 3)
                throw std::runtime_error("coarse export has invalid Gaussian means");
            const size_t count = coarse_means_cpu.size(0);
            if (count == 0)
                throw std::runtime_error("coarse export has no Gaussian means");
            if (!params.tile_require_dense_masks) {
                const size_t stride = coarse_means_cpu.size(1);
                const size_t max_samples = 150'000;
                const size_t step = std::max<size_t>(1, (count + max_samples - 1) / max_samples);
                const auto* values = coarse_means_cpu.ptr<float>();
                coarse_points.reserve((count + step - 1) / step);
                for (size_t row = 0; row < count; row += step) {
                    const glm::vec3 point{values[row * stride], values[row * stride + 1], values[row * stride + 2]};
                    if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
                        coarse_points.push_back(point);
                }
                if (coarse_points.empty())
                    throw std::runtime_error("coarse export has no finite Gaussian means for sparse fallback");
            }
            LOG_INFO("Loaded {} coarse Gaussians; cached CPU means once for per-tile owner selection{}",
                     coarse_model->size(), params.tile_require_dense_masks
                         ? ""
                         : std::format(" and retained {} sampled centres for explicit sparse fallback", coarse_points.size()));
        } catch (const std::exception& error) {
            LOG_ERROR("Cannot prepare coarse depth masks: {}", error.what());
            return 1;
        }

        lfs::core::Tensor mask_background;
        try {
            mask_background = lfs::core::Tensor::zeros(
                {3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        } catch (const std::exception& error) {
            LOG_ERROR("Cannot initialize CUDA for mandatory dense coarse visibility masks: {}", error.what());
            return 1;
        }
        std::unordered_map<int, DenseCoarseRender> dense_full_render_cache;
        std::vector<path> tile_depth_mask_paths(tiles.size());
        size_t eligible_dense_mask_views = 0;
        for (size_t tile_index = 0; tile_index < tiles.size(); ++tile_index) {
            for (const int uid : tile_cameras[tile_index]) {
                const auto view = views_by_uid.find(uid);
                if (view != views_by_uid.end() && view->second->source && view->second->pinhole &&
                    view->second->width > 0 && view->second->height > 0)
                    ++eligible_dense_mask_views;
            }
        }
        size_t rendered_dense_mask_views = 0;
        size_t fallback_mask_views = 0;
        size_t ineligible_mask_views = 0;
        size_t unavailable_mask_views = 0;
        size_t processed_dense_mask_views = 0;
        const size_t dense_progress_interval = std::max<size_t>(1, eligible_dense_mask_views / 20);
        for (size_t tile_index = 0; tile_index < tiles.size(); ++tile_index) {
            if (g_interrupt_count.load() > 0) {
                LOG_WARN("Auto-tile dense-mask generation cancelled before tile {}", tiles[tile_index].name);
                return 1;
            }
            json depth_masks = {
                {"version", 3},
                {"method", "coarse_gaussian_dense_depth_opacity"},
                {"coarse_model", path_to_utf8(coarse_export)},
                {"resolution", params.tile_visibility_mask_resolution},
                {"minimum_alpha", params.tile_visibility_min_alpha},
                {"foreign_weight", params.tile_visibility_foreign_weight},
                {"require_dense", params.tile_require_dense_masks},
                {"masks", json::array()}};
            // This expensive full-model scan/index-select is deliberately once
            // per tile. The resulting owner model is reused for every selected
            // camera rather than rebuilt inside each rasterization call.
            std::unique_ptr<lfs::core::SplatData> owner_model;
            try {
                owner_model = coarse_owner_model(*coarse_model, coarse_means_cpu, tiles[tile_index]);
            } catch (const std::exception& error) {
                LOG_ERROR("Cannot prepare coarse owner model for tile '{}': {}",
                          tiles[tile_index].name, error.what());
                return 1;
            }
            size_t tile_dense_masks = 0;
            size_t tile_fallback_masks = 0;
            size_t tile_ineligible_masks = 0;
            size_t tile_unavailable_masks = 0;
            json ineligible_uids = json::array();
            json unavailable_uids = json::array();
            for (const int uid : tile_cameras[tile_index]) {
                if (g_interrupt_count.load() > 0) {
                    LOG_WARN("Auto-tile dense-mask generation cancelled at tile '{}' camera {}",
                             tiles[tile_index].name, uid);
                    return 1;
                }
                const auto view = views_by_uid.find(uid);
                if (view == views_by_uid.end()) {
                    LOG_ERROR("Tile '{}' refers to unknown camera UID {}", tiles[tile_index].name, uid);
                    return 1;
                }
                if (!view->second->source || !view->second->pinhole ||
                    view->second->width <= 0 || view->second->height <= 0) {
                    ++tile_ineligible_masks;
                    ++ineligible_mask_views;
                    ineligible_uids.push_back(uid);
                    continue;
                }
                lfs::core::TileVisibilityMask mask;
                std::string method = "dense_gaussian_depth_opacity";
                bool dense_rasterized = false;
                try {
                    mask = dense_tile_visibility_mask(
                        *view->second, *coarse_model, owner_model.get(),
                        params.tile_visibility_mask_resolution,
                        params.tile_visibility_min_alpha,
                        params.tile_visibility_foreign_weight,
                        mask_background, dense_full_render_cache, &dense_rasterized);
                } catch (const std::exception& error) {
                    LOG_WARN("Dense coarse visibility rasterization failed for tile '{}' camera {}: {}",
                             tiles[tile_index].name, uid, error.what());
                }
                if (mask.weights.empty() && !dense_rasterized) {
                    if (params.tile_require_dense_masks) {
                        LOG_ERROR("Tile '{}' camera {} could not produce its required dense coarse depth/opacity mask",
                                  tiles[tile_index].name, uid);
                        return 1;
                    }
                    method = "gaussian_center_zbuffer_fallback";
                    mask = lfs::core::estimate_tile_depth_visibility_mask(
                        tiles[tile_index], *view->second, coarse_points,
                        params.tile_visibility_mask_resolution,
                        params.tile_visibility_mask_resolution);
                    if (mask.weights.empty()) {
                        LOG_WARN("Tile '{}' camera {} has no usable sparse fallback mask",
                                 tiles[tile_index].name, uid);
                        ++tile_unavailable_masks;
                        ++unavailable_mask_views;
                        unavailable_uids.push_back(uid);
                    } else {
                        ++tile_fallback_masks;
                        ++fallback_mask_views;
                    }
                } else {
                    ++tile_dense_masks;
                    ++rendered_dense_mask_views;
                }
                if (!mask.weights.empty()) {
                    depth_masks["masks"].push_back({
                        {"uid", uid}, {"width", mask.width}, {"height", mask.height},
                        {"method", method}, {"weights", mask.weights}});
                }
                ++processed_dense_mask_views;
                if (processed_dense_mask_views == eligible_dense_mask_views ||
                    processed_dense_mask_views % dense_progress_interval == 0) {
                    LOG_INFO("Auto-tile dense masks: {}/{} eligible camera views processed",
                             processed_dense_mask_views, eligible_dense_mask_views);
                }
            }
            depth_masks["coverage"] = {
                {"eligible", tile_dense_masks + tile_fallback_masks + tile_unavailable_masks},
                {"dense", tile_dense_masks},
                {"fallback", tile_fallback_masks},
                {"unavailable", tile_unavailable_masks},
                {"ineligible", tile_ineligible_masks},
                {"ineligible_uids", ineligible_uids},
                {"unavailable_uids", unavailable_uids}};
            tile_depth_mask_paths[tile_index] = output / "tiles" / tiles[tile_index].name / "tile_depth_masks.json";
            if (!write_batch_report(tile_depth_mask_paths[tile_index], depth_masks.dump() + "\n"))
                return 1;
        }
        const json dense_mask_coverage = {
            {"require_dense", params.tile_require_dense_masks},
            {"eligible", eligible_dense_mask_views},
            {"dense", rendered_dense_mask_views},
            {"fallback", fallback_mask_views},
            {"unavailable", unavailable_mask_views},
            {"ineligible", ineligible_mask_views}};
        // Dense mask rendering belongs to the parent only. Release its large
        // model and thread-local raster cache before starting local workers.
        coarse_model.reset();
        coarse_means_cpu = {};
        std::vector<glm::vec3>().swap(coarse_points);
        lfs::training::release_fast_rasterizer_thread_local_caches();
        // The parent needs only UID/name metadata after planning. Release loaded
        // camera payloads before starting a separate training process.
        for (auto& view : camera_evidence) {
            view.source.reset();
            std::vector<glm::vec3>().swap(view.observations);
            std::vector<glm::vec2>().swap(view.observation_pixels);
        }
        json manifest;
        manifest["dataset"] = path_to_utf8(dataset);
        manifest["requested_tile_count"] = requested_tile_count;
        manifest["resolved_tile_count"] = tiles.size();
        manifest["feasibility"] = {
            {"path", "tile_feasibility.json"},
            {"minimum_cameras", params.tile_min_cameras},
            {"minimum_multi_view_cells", params.tile_min_multi_view_cells},
            {"topology_operations", topology_operations},
            {"coalesced", tiles.size() != requested_tile_count}};
        manifest["overlap_ratio"] = params.tile_overlap;
        manifest["outside_loss_weight"] = params.tile_outside_loss_weight;
        manifest["crop_after_training"] = params.auto_tile_crop;
        manifest["merge_exports"] = params.auto_tile_merge;
        manifest["ownership"] = "disjoint_core_export_with_overlapping_training_context";
        manifest["coarse_prior"] = {
            {"path", "coarse/coarse.ply"},
            {"iterations", params.tile_coarse_iterations},
            {"resize_factor", params.tile_coarse_resize_factor},
            {"training_seconds", coarse_result.duration_seconds},
            {"transferred_attributes", {"means", "scales", "rotations", "opacity", "spherical_harmonics"}},
            {"tile_initialization", "core_plus_context_gaussian_filter"}};
        manifest["depth_masks"] = {
            {"method", "coarse_gaussian_dense_depth_opacity"},
            {"coverage", dense_mask_coverage},
            {"fallback", "gaussian_center_zbuffer when dense requirement is disabled"}};
        manifest["boundary_padding"] = {
            {"distance", params.tile_boundary_padding},
            {"timing", "after_camera_assignment"},
            {"axes", "local OBB partition X/Y only"},
            {"already_present_in_plan", plan_has_boundary_padding}};
        manifest["partition_mode"] = params.auto_tile_count > 0 ? "adaptive" : "grid";
        manifest["camera_selection"] = "sfm_image_support_tile_coverage_and_projected_airspace";
        manifest["airspace_view_threshold"] = lfs::core::TILE_AIRSPACE_VIEW_THRESHOLD;
        manifest["exposure_calibration"] = {
            {"path", "tile_exposure.json"},
            {"connected_cameras", exposure.connected_cameras},
            {"shared_points", exposure.shared_points},
            {"reliable_view_pairs", exposure.reliable_edges}};
        manifest["tile_exposure_correction_locked"] = params.tile_lock_exposure_correction;
        if (!planning_diagnostics.empty())
            manifest["planning_work_units"] = "relative SfM-point and image-resolution proxy";
        manifest["tiles"] = json::array();
        for (size_t index = 0; index < tiles.size(); ++index) {
            const auto& tile = tiles[index];
            json entry = {{"name", tile.name}, {"i", tile.i}, {"j", tile.j},
                          {"camera_count", tile_cameras[index].size()},
                          {"camera_allowlist", path_to_utf8(path("tiles") / tile.name / "tile_cameras.json")},
                          {"depth_masks", path_to_utf8(path("tiles") / tile.name / "tile_depth_masks.json")},
                          {"min", {tile.min.x, tile.min.y, tile.min.z}},
                          {"max", {tile.max.x, tile.max.y, tile.max.z}},
                          {"rotation_degrees", {tile.rotation_degrees.x,
                                                 tile.rotation_degrees.y,
                                                 tile.rotation_degrees.z}}};
            const auto& assignment_tile = camera_assignment_tiles[index];
            entry["camera_assignment_core_min"] = {
                assignment_tile.min.x, assignment_tile.min.y, assignment_tile.min.z};
            entry["camera_assignment_core_max"] = {
                assignment_tile.max.x, assignment_tile.max.y, assignment_tile.max.z};
            entry["coalesced_from"] = coalesced_from[index];
            entry["feasibility"] = {
                {"camera_count", tile_feasibility[index].camera_count},
                {"multi_view_cells", tile_feasibility[index].multi_view_cells},
                {"feasible", tile_feasibility[index].feasible}};
            entry["boundary_extension"] = {
                {"frame", "tile_local_obb"},
                {"min_x", boundary_extensions[index][0]},
                {"max_x", boundary_extensions[index][1]},
                {"min_y", boundary_extensions[index][2]},
                {"max_y", boundary_extensions[index][3]}};
            if (tile.has_context_bounds) {
                entry["context_min"] = {tile.context_min.x, tile.context_min.y, tile.context_min.z};
                entry["context_max"] = {tile.context_max.x, tile.context_max.y, tile.context_max.z};
            }
            const auto& selection = camera_selection_diagnostics[index];
            entry["camera_selection"] = {
                {"evidence_views", selection.evidence_views},
                {"track_supported_views", selection.track_supported_views},
                {"airspace_supported_views", selection.airspace_supported_views},
                {"observed_cells", selection.observed_cells},
                {"covered_cells", selection.covered_cells},
                {"multi_view_cells", selection.multi_view_cells},
                {"multi_view_covered_cells", selection.multi_view_covered_cells},
                {"sparse_fallback", selection.sparse_fallback}};
            if (index < planning_diagnostics.size()) {
                const auto& d = planning_diagnostics[index];
                entry["planning"] = {{"sampled_points", d.sampled_points},
                                     {"sampled_crop_points", d.sampled_crop_points},
                                     {"supporting_cameras", d.supporting_cameras},
                                     {"estimated_selected_cameras", d.estimated_selected_cameras},
                                     {"occupied_cells", d.occupied_cells},
                                     {"covered_cells", d.covered_cells},
                                     {"image_evidence_views", d.image_evidence_views},
                                     {"observed_image_cells", d.observed_image_cells},
                                     {"mean_image_coverage", d.mean_image_coverage},
                                     {"mean_image_megapixels", d.mean_image_megapixels},
                                     {"estimated_work_units", d.estimated_work_units}};
            }
            manifest["tiles"].push_back(std::move(entry));
        }
        if (!write_batch_report(output / "tiles.json", manifest.dump(2) + "\n"))
            return 1;

        std::vector<path> exports;
        exports.reserve(tiles.size());
        for (size_t tile_index = 0; tile_index < tiles.size(); ++tile_index) {
            const auto& tile = tiles[tile_index];
            lfs::core::param::BatchJobItem job;
            job.dataset_path = dataset;
            job.name = tile.name;
            job.output_path = output / "tiles" / tile.name;
            json allowlist;
            allowlist["dataset"] = path_to_utf8(dataset);
            allowlist["cameras"] = json::array();
            for (const int uid : tile_cameras[tile_index]) {
                const auto found = views_by_uid.find(uid);
                if (found != views_by_uid.end())
                    allowlist["cameras"].push_back({{"uid", uid}, {"image", found->second->image_name}});
            }
            const path camera_path = job.output_path / "tile_cameras.json";
            if (!write_batch_report(camera_path, allowlist.dump(2) + "\n")) return 1;
            const auto box = std::format("{},{},{},{},{},{}", tile.min.x, tile.min.y, tile.min.z,
                                         tile.max.x, tile.max.y, tile.max.z);
            const auto context_min = tile.has_context_bounds ? tile.context_min : tile.min;
            const auto context_max = tile.has_context_bounds ? tile.context_max : tile.max;
            const auto context_box = std::format("{},{},{},{},{},{}",
                                                 context_min.x, context_min.y, context_min.z,
                                                 context_max.x, context_max.y, context_max.z);
            job.extra_args = {"--cropbox=" + box, "--cropbox-loss-weight",
                              std::format("{}", params.tile_outside_loss_weight),
                              "--tile-camera-uids", path_to_utf8(camera_path),
                              "--tile-contextbox=" + context_box,
                              "--tile-exposure-gains", path_to_utf8(exposure_path),
                              "--tile-depth-masks", path_to_utf8(tile_depth_mask_paths[tile_index]),
                              "--init", path_to_utf8(coarse_export),
                              "--tile-child",
                              "--output-name", tile.name, "--export", "ply"};
            if (params.auto_tile_crop)
                job.extra_args.push_back("--crop");
            if (!params.tile_require_dense_masks)
                job.extra_args.push_back("--no-tile-require-dense-masks");
            if (tile.rotation_degrees.x != 0.0f || tile.rotation_degrees.y != 0.0f ||
                tile.rotation_degrees.z != 0.0f) {
                job.extra_args.push_back(std::format("--cropbox-rotation={},{},{}",
                                                     tile.rotation_degrees.x,
                                                     tile.rotation_degrees.y,
                                                     tile.rotation_degrees.z));
            }
            if (params.tile_lock_exposure_correction) {
                // The global SfM gains are still applied to every target image
                // through --tile-exposure-gains.  This only removes the
                // independent learned residual that could otherwise diverge
                // between separately optimized tiles.
                job.extra_args.push_back("--no-exposure-correction");
            }
            std::string error;
            if (!validate_job_arguments(job, batch, error)) {
                LOG_ERROR("Invalid tile {}: {}", tile.name, error);
                return 1;
            }
            if (g_interrupt_count.load() > 0) {
                LOG_WARN("Auto-tiling interrupted before {}", tile.name);
                return 1;
            }
            std::println("[Auto-tile {}/{}] Training {}", exports.size() + 1, tiles.size(), tile.name);
            const auto result = execute_job(job, batch, executable);
            manifest["tiles"][tile_index]["training_seconds"] = result.duration_seconds;
            if (!write_batch_report(output / "tiles.json", manifest.dump(2) + "\n"))
                return 1;
            if (!result.success) {
                LOG_ERROR("Tile {} failed: {}", tile.name, result.error_message);
                return 1;
            }
            const path exported = job.output_path / (tile.name + ".ply");
            if (!lfs::core::native_fs::exists(exported)) {
                LOG_ERROR("Tile {} completed without its expected PLY export: {}", tile.name, path_to_utf8(exported));
                return 1;
            }
            exports.push_back(exported);
        }
        if (!params.auto_tile_merge) return 0;
        if (g_interrupt_count.load() > 0) return 1;

        // Tile PLYs already use the same dataset coordinates. Streaming their
        // identical vertex records avoids holding every trained model in RAM.
        const path destination = output / "merged.ply";
        if (const auto merged = lfs::io::merge_binary_ply_streaming(
                exports, destination, [] { return g_interrupt_count.load() > 0; }); !merged) {
            LOG_ERROR("Cannot merge tile splats: {}", merged.error().format());
            return 1;
        }
        std::println("[Auto-tile] Merged {} tiles into {}", tiles.size(), path_to_utf8(destination));
        return 0;
    }

} // namespace lfs::app
