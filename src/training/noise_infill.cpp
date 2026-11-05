/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "noise_infill.hpp"

#include "dataset.hpp"
#include "rasterization/fast_rasterizer.hpp"

#include "core/collision.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data_transform.hpp"
#include "io/loader.hpp"
#include "rendering/mesh2splat.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <deque>
#include <format>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::training {

    namespace {
        lfs::Error infill_error(std::string message) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::Training,
                .user_message = std::move(message),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        constexpr float kSh0 = 0.28209479177387814f;
        // SH0 coefficients of a colour channel at 0 and at 1.
        constexpr float kChannelOff = -0.5f / kSh0, kChannelOn = 0.5f / kSh0;
        // Neighbouring noise Gaussians overlap, so the filled volume renders opaque.
        constexpr float kScaleOfCell = 0.7f;
        // A cell is surface where the mesh's surface splats accumulate this opacity, as the collision grid judges it.
        constexpr float kSurfaceOpacity = 0.1f;
        // The cell edge puts about this many cells inside the mesh, whatever its size and shape.
        constexpr double kInsideCells = 200000.0;
        // The share of the training before the noise appears.
        constexpr float kStartFraction = 0.3f;

        struct Geometry {
            std::vector<float> vertices; // [V*3]
            std::vector<int> indices;    // [F*3]
        };

        uint64_t edge_key(const int a, const int b) {
            return (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
        }

        struct PositionHash {
            size_t operator()(const std::array<float, 3>& p) const {
                size_t h = 0;
                for (const float c : p)
                    h = h * 0x9E3779B97F4A7C15ull ^ (c == 0.0f ? 0u : std::bit_cast<uint32_t>(c));
                return h;
            }
        };

        // Meshes split a vertex where its texture coordinates or normals change; unwelded, every seam reads as a hole.
        void weld_by_position(Geometry& g) {
            const size_t count = g.vertices.size() / 3;
            std::unordered_map<std::array<float, 3>, int, PositionHash> first;
            first.reserve(count);
            std::vector<int> welded(count);
            for (size_t v = 0; v < count; ++v)
                welded[v] = first.try_emplace({g.vertices[3 * v], g.vertices[3 * v + 1], g.vertices[3 * v + 2]}, int(v))
                                .first->second;
            for (auto& i : g.indices)
                i = welded[size_t(i)];
        }

        // Each open boundary loop is closed by a fan around its centroid; returns the number of loops.
        size_t cap_boundary_loops(Geometry& g) {
            const size_t faces = g.indices.size() / 3;
            const auto undirected = [](const int a, const int b) { return edge_key(std::min(a, b), std::max(a, b)); };
            std::unordered_map<uint64_t, int> faces_on_edge;
            std::unordered_set<uint64_t> directed;
            faces_on_edge.reserve(faces * 3);
            directed.reserve(faces * 3);
            for (size_t f = 0; f < faces; ++f) {
                for (int k = 0; k < 3; ++k) {
                    const int a = g.indices[3 * f + k], b = g.indices[3 * f + (k + 1) % 3];
                    ++faces_on_edge[undirected(a, b)];
                    directed.insert(edge_key(a, b));
                }
            }
            // An edge only one face holds lies on the boundary, however the faces are wound. Walking both ways from vertex to
            // vertex follows a loop across patches wound differently.
            std::unordered_multimap<int, int> boundary;
            for (const auto& [key, count] : faces_on_edge) {
                if (count != 1)
                    continue;
                const int a = int(key >> 32), b = int(uint32_t(key));
                boundary.emplace(a, b);
                boundary.emplace(b, a);
            }
            size_t loops = 0;
            while (!boundary.empty()) {
                std::vector<int> loop;
                const int first = boundary.begin()->first;
                int at = first;
                while (true) {
                    loop.push_back(at);
                    const auto it = boundary.find(at);
                    if (it == boundary.end())
                        break;
                    const int next = it->second;
                    boundary.erase(it);
                    auto [reverse, end] = boundary.equal_range(next);
                    while (reverse != end && reverse->second != at)
                        ++reverse;
                    assert(reverse != end);
                    boundary.erase(reverse);
                    at = next;
                    if (at == first)
                        break;
                }
                if (loop.size() < 3)
                    continue;
                std::array<double, 3> centroid{};
                for (const int v : loop)
                    for (int k = 0; k < 3; ++k)
                        centroid[size_t(k)] += g.vertices[3 * size_t(v) + size_t(k)] / double(loop.size());
                const int center = int(g.vertices.size() / 3);
                for (const double c : centroid)
                    g.vertices.push_back(float(c));
                for (size_t i = 0; i < loop.size(); ++i) {
                    // The cap holds a boundary edge the other way round from the face beside it.
                    const int a = loop[i], b = loop[(i + 1) % loop.size()];
                    if (directed.contains(edge_key(a, b)))
                        g.indices.insert(g.indices.end(), {b, a, center});
                    else
                        g.indices.insert(g.indices.end(), {a, b, center});
                }
                ++loops;
            }
            return loops;
        }

        // Volume the closed, consistently wound surface encloses: the sum of the signed tetrahedra to the origin.
        double enclosed_volume(const Geometry& g) {
            const auto vertex = [&](const int v) {
                return glm::dvec3(g.vertices[3 * size_t(v)], g.vertices[3 * size_t(v) + 1], g.vertices[3 * size_t(v) + 2]);
            };
            double volume = 0.0;
            for (size_t f = 0; f < g.indices.size(); f += 3)
                volume += glm::dot(vertex(g.indices[f]), glm::cross(vertex(g.indices[f + 1]), vertex(g.indices[f + 2])));
            return std::abs(volume) / 6.0;
        }

        struct Grid {
            glm::ivec3 dims{0};
            std::vector<uint8_t> state; // kFree, kSurface or kOutside
            size_t at(const glm::ivec3 p) const { return (size_t(p.z) * size_t(dims.y) + size_t(p.y)) * size_t(dims.x) + size_t(p.x); }
            bool inside(const glm::ivec3 p) const { return glm::all(glm::greaterThanEqual(p, glm::ivec3(0))) && glm::all(glm::lessThan(p, dims)); }
            template <typename Visit>
            void for_each_cell(Visit&& visit) const {
                for (int z = 0; z < dims.z; ++z)
                    for (int y = 0; y < dims.y; ++y)
                        for (int x = 0; x < dims.x; ++x)
                            visit(glm::ivec3{x, y, z});
            }
        };
        constexpr uint8_t kFree = 0, kSurface = 1, kOutside = 2;
        constexpr std::array<glm::ivec3, 6> kNeighbours{glm::ivec3{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

        // Free cells a path of free cells joins to the border are outside.
        void mark_outside(Grid& grid) {
            std::deque<glm::ivec3> queue;
            const auto seed = [&](const glm::ivec3 p) {
                if (auto& s = grid.state[grid.at(p)]; s == kFree) {
                    s = kOutside;
                    queue.push_back(p);
                }
            };
            grid.for_each_cell([&](const glm::ivec3 p) {
                if (!grid.inside(p - glm::ivec3(1)) || !grid.inside(p + glm::ivec3(1)))
                    seed(p);
            });
            while (!queue.empty()) {
                const auto p = queue.front();
                queue.pop_front();
                for (const auto& d : kNeighbours)
                    if (const auto q = p + d; grid.inside(q))
                        seed(q);
            }
        }

        // The cells inside, more than the erosion steps from any surface or outside cell.
        std::vector<glm::ivec3> eroded_inside(const Grid& grid) {
            std::vector<int> distance(grid.state.size(), -1);
            std::deque<glm::ivec3> queue;
            grid.for_each_cell([&](const glm::ivec3 p) {
                if (grid.state[grid.at(p)] != kFree) {
                    distance[grid.at(p)] = 0;
                    queue.push_back(p);
                }
            });
            while (!queue.empty()) {
                const auto p = queue.front();
                queue.pop_front();
                const int next = distance[grid.at(p)] + 1;
                if (next > kNoiseInfillErosion)
                    continue;
                for (const auto& d : kNeighbours)
                    if (const auto q = p + d; grid.inside(q) && distance[grid.at(q)] < 0) {
                        distance[grid.at(q)] = next;
                        queue.push_back(q);
                    }
            }
            std::vector<glm::ivec3> kept;
            grid.for_each_cell([&](const glm::ivec3 p) {
                if (distance[grid.at(p)] < 0)
                    kept.push_back(p);
            });
            return kept;
        }
    } // namespace

    lfs::Result<NoiseInfill> make_noise_infill(const core::MeshData& mesh) {
        if (!mesh.vertices.is_valid() || !mesh.indices.is_valid() || mesh.face_count() == 0)
            return infill_error("The noise infill mesh has no triangles");
        assert(mesh.vertices.ndim() == 2 && mesh.vertices.size(1) == 3);
        assert(mesh.indices.ndim() == 2 && mesh.indices.size(1) == 3);
        Geometry g;
        g.vertices = mesh.vertices.cpu().contiguous().to_vector();
        g.indices = mesh.indices.cpu().contiguous().to_vector_int();
        glm::vec3 low(INFINITY), high(-INFINITY);
        for (size_t v = 0; v < g.vertices.size(); v += 3) {
            const glm::vec3 p{g.vertices[v], g.vertices[v + 1], g.vertices[v + 2]};
            low = glm::min(low, p);
            high = glm::max(high, p);
        }
        const glm::vec3 size = high - low;
        const float extent = std::max({size.x, size.y, size.z});
        if (!(extent > 0.0f) || !std::isfinite(extent))
            return infill_error("The noise infill mesh has no extent");
        weld_by_position(g);
        const size_t loops = cap_boundary_loops(g);
        const auto clamped_cell = [extent](const double wanted) { return std::clamp(float(wanted), extent / 512.0f, extent / 16.0f); };
        float cell = clamped_cell(std::cbrt(enclosed_volume(g) / kInsideCells));

        const auto vertex_count = g.vertices.size() / 3, face_count = g.indices.size() / 3;
        core::MeshData capped(core::Tensor::from_vector(g.vertices, {vertex_count, 3}, core::Device::CPU),
                              core::Tensor::from_vector(g.indices, {face_count, 3}, core::Device::CPU));
        struct InsideCells {
            glm::vec3 origin;
            std::vector<glm::ivec3> cells;
        };
        const auto inside_cells = [&](const float voxel) -> lfs::Result<InsideCells> {
            core::Mesh2SplatOptions options;
            // Surface splats a few to a cell close the surface in the collision grid.
            options.resolution_target = std::clamp(int(std::ceil(4.0f * extent / voxel)), core::Mesh2SplatOptions::kMinResolution, 2048);
            auto surface = rendering::mesh_to_splat(capped, options);
            if (!surface)
                return infill_error(std::format("Noise infill surface splats failed: {}", surface.error()));

            core::CollisionSettings collision;
            collision.voxel_size = voxel;
            collision.opacity = kSurfaceOpacity;
            collision.cleanup = false;
            core::CollisionVolume volume;
            try {
                volume = core::generateCollision(core::captureCollisionSource(**surface), collision);
            } catch (const std::exception& error) {
                return lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::Internal,
                    .domain = lfs::ErrorDomain::Training,
                    .user_message = std::format("Noise infill voxelization failed: {}", error.what()),
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                });
            }

            Grid grid;
            grid.dims = volume.dimensions;
            grid.state.assign(size_t(grid.dims.x) * size_t(grid.dims.y) * size_t(grid.dims.z), kFree);
            grid.for_each_cell([&](const glm::ivec3 p) {
                if (volume.occupied(p))
                    grid.state[grid.at(p)] = kSurface;
            });
            mark_outside(grid);
            return InsideCells{volume.origin, eroded_inside(grid)};
        };
        auto inside = inside_cells(cell);
        // Shells wound against each other enclose far more than their signed volume says: what the grid found sets the cell.
        if (inside && double(inside->cells.size()) > 2.0 * kInsideCells) {
            cell = clamped_cell(cell * std::cbrt(double(inside->cells.size()) / kInsideCells));
            inside = inside_cells(cell);
        }
        if (!inside)
            return std::move(inside.error());
        const auto& kept = inside->cells;
        if (kept.empty())
            return infill_error(std::format("The noise infill mesh encloses no cell clear of its surface at cell {}", cell));

        const size_t count = kept.size();
        std::vector<float> means, rotation(count * 4, 0.0f);
        means.reserve(count * 3);
        for (size_t i = 0; i < count; ++i) {
            const glm::vec3 center = inside->origin + (glm::vec3(kept[i]) + 0.5f) * cell;
            means.insert(means.end(), {center.x, center.y, center.z});
            rotation[4 * i] = 1.0f;
        }
        LOG_INFO("Noise infill: {} Gaussians at cell {}, {} open loops capped", count, cell, loops);
        return NoiseInfill{.splats = core::SplatData(0, core::Tensor::from_vector(means, {count, 3}, core::Device::CUDA),
                                                     core::Tensor::zeros({count, 1, 3}, core::Device::CUDA),
                                                     core::Tensor::zeros({count, 0, 3}, core::Device::CUDA),
                                                     core::Tensor::full({count, 3}, std::log(kScaleOfCell * cell), core::Device::CUDA),
                                                     core::Tensor::from_vector(rotation, {count, 4}, core::Device::CUDA),
                                                     core::Tensor::full({count, 1}, kShownNoiseOpacity, core::Device::CUDA), 1.0f),
                           .cell = cell,
                           .capped_loops = loops};
    }

    lfs::Result<core::SplatData> load_noise_infill(const std::filesystem::path& mesh,
                                                   const std::array<float, 3>& origin) {
        io::LoadOptions options;
        options.mesh_geometry_only = true;
        auto loaded = io::Loader::create()->load(mesh, options);
        if (!loaded)
            return infill_error(std::format("Failed to load the noise infill mesh '{}': {}", core::path_to_utf8(mesh),
                                            loaded.error().format()));
        const auto* const data = std::get_if<std::shared_ptr<core::MeshData>>(&loaded->data);
        if (!data || !*data)
            return infill_error(std::format("'{}' is not a mesh", core::path_to_utf8(mesh)));
        auto infill = make_noise_infill(**data);
        if (!infill)
            return std::move(infill.error());
        auto& means = infill->splats.means();
        means = (means - core::Tensor::from_vector({origin[0], origin[1], origin[2]}, core::TensorShape({1, 3}), means.device())).contiguous();
        return std::move(infill->splats);
    }

    int noise_infill_start_iteration(const core::param::OptimizationParameters& params) {
        return static_cast<int>(kStartFraction * static_cast<float>(params.iterations));
    }

    void randomize_noise_colors(core::SplatData& model, const std::size_t count) {
        if (count == 0)
            return;
        assert(model.sh0().ndim() == 3 && size_t(model.sh0().size(0)) >= count);
        auto noise = model.sh0().slice(0, 0, count);
        const float off = kChannelOff, on = kChannelOn;
        const auto colors = core::Tensor::from_vector(
            std::vector<float>{on, off, off, off, on, off, off, off, on, off, on, on, on, off, on, on, on, off}, {6, 3}, noise.device());
        const auto picks = core::Tensor::randint({count}, 0, 6, noise.device(), core::DataType::Int32).to(core::DataType::Int64);
        const auto rows = static_cast<int>(count);
        noise.copy_from(colors.index_select(0, picks).reshape({rows, 1, 3}));
    }

    bool noise_infill_step(core::SplatData& model, const core::param::OptimizationParameters& params, const int iter) {
        const auto noise = core::find_noise_infill(model);
        // A later step than the start one when the run was shortened past it.
        if (!noise || (noise->hidden && iter < noise_infill_start_iteration(params)))
            return false;
        randomize_noise_colors(model, noise->count);
        if (!noise->hidden)
            return false;
        model.opacity_raw().slice(0, 0, noise->count).fill_(kShownNoiseOpacity);
        return true;
    }

    core::param::OptimizationParameters with_noise_outside_cap(const core::param::OptimizationParameters& params,
                                                               const core::SplatData* const model) {
        auto capped = params;
        if (model && capped.max_cap > 0)
            capped.max_cap += static_cast<int>(noise_infill_rows(*model));
        return capped;
    }

    RowOverride::RowOverride(core::Tensor& tensor, const std::size_t begin, const std::size_t end, const float value)
        : rows_(tensor.slice(0, begin, end)),
          saved_(rows_.clone()) {
        assert(begin <= end && end <= size_t(tensor.size(0)));
        rows_.fill_(value);
    }

    RowOverride::~RowOverride() {
        rows_.copy_from(saved_);
    }

    void paint_noise_green(core::Tensor& noise_sh0) {
        assert(noise_sh0.ndim() == 3 && noise_sh0.size(1) == 1 && noise_sh0.size(2) == 3);
        const auto count = static_cast<std::size_t>(noise_sh0.size(0));
        std::vector<float> green(count * 3, kChannelOff);
        for (std::size_t i = 0; i < count; ++i)
            green[3 * i + 1] = kChannelOn;
        noise_sh0.copy_from(core::Tensor::from_vector(green, {count, 1, 3}, noise_sh0.device()));
    }

    std::optional<double> surface_see_through(core::SplatData& model, const std::size_t noise_rows, const CameraDataset& cameras,
                                              const bool mip_filter, const std::function<bool()>& keep_going) {
        const auto rows = static_cast<std::size_t>(model.size());
        assert(noise_rows > 0 && noise_rows <= rows);
        auto noise_color = model.sh0().slice(0, 0, noise_rows);
        paint_noise_green(noise_color);
        model.sh0().slice(0, noise_rows, rows).fill_(kChannelOff);
        model.set_active_sh_degree(0);
        auto black = core::Tensor::zeros({3}, model.sh0().device());
        const auto green_sum = [&](core::Camera& camera) {
            const auto image = fast_rasterize(camera, model, black, mip_filter).image;
            assert(image.ndim() == 3 && image.size(0) == 3);
            return double(image.slice(0, 1, 2).sum().item<float>());
        };
        double shown = 0.0, covered = 0.0;
        for (std::size_t i = 0; i < cameras.size(); ++i) {
            if (keep_going && !keep_going())
                return std::nullopt;
            auto& camera = *cameras.get_camera(i);
            shown += green_sum(camera);
            const RowOverride hidden_surface(model.opacity_raw(), noise_rows, rows, kHiddenOpacity);
            covered += green_sum(camera);
        }
        if (covered <= 0.0)
            return std::nullopt;
        return shown / covered;
    }

} // namespace lfs::training
