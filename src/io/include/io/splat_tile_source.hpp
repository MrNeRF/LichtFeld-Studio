/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/splat_data.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <glm/glm.hpp>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace lfs::io {

    // One node of a hierarchical splat LOD tree (3D Tiles tile; an SSOG tree
    // node at one level). Geometry is in the source-local frame.
    struct SplatTile {
        static constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();

        glm::vec3 center{0.0f};
        glm::mat3 half_axes{0.0f};    // oriented box: columns are half-extent axes
        glm::mat4 transform{1.0f};    // content frame -> source-local frame
        float geometric_error = 0.0f; // detail missing versus the children, local units
        bool additive = false;        // ADD refinement: children draw with the parent
        std::uint32_t parent = kNone;
        std::uint32_t first_child = 0; // children are contiguous
        std::uint32_t child_count = 0;
        std::uint64_t splat_count = 0; // 0: no content
        int sh_degree = 0;

        // Distance from `point` to the oriented box, 0 inside.
        [[nodiscard]] float distance(const glm::vec3& point) const {
            const glm::vec3 offset = point - center;
            glm::vec3 outside{0.0f};
            for (int axis = 0; axis < 3; ++axis) {
                const float half = glm::length(half_axes[axis]);
                const float along = half > 0.0f ? std::abs(glm::dot(offset, half_axes[axis])) / half : 0.0f;
                outside[axis] = std::max(along - half, 0.0f);
            }
            return glm::length(outside);
        }

        [[nodiscard]] float radius() const {
            return std::sqrt(glm::dot(half_axes[0], half_axes[0]) + glm::dot(half_axes[1], half_axes[1]) +
                             glm::dot(half_axes[2], half_axes[2]));
        }
    };

    // Random-access tile content for view-dependent streaming. Tile 0 is the root.
    class SplatTileSource {
    public:
        virtual ~SplatTileSource() = default;
        [[nodiscard]] virtual std::span<const SplatTile> tiles() const = 0;
        // Decodes one tile's content in its content frame; safe from worker threads.
        [[nodiscard]] virtual std::expected<core::SplatData, std::string> load_tile(std::uint32_t tile) const = 0;
        // Source-local frame -> georeferenced frame (ECEF for 3D Tiles).
        glm::dmat4 local_to_world{1.0};
    };

    struct SplatTileView {
        glm::vec3 camera{0.0f};            // source-local frame
        std::array<glm::vec4, 6> planes{}; // normalized frustum planes, inside >= 0; zero = no culling
        float sse_per_error = 0.0f;        // pixels per unit error at unit distance: H / (2 tan(fov/2))
        float max_sse = 16.0f;             // pixels of error a tile may show before refining
    };

    struct SplatTileSelection {
        std::vector<std::uint32_t> render; // resident tiles to draw, never overlapping
        std::vector<std::uint32_t> wanted; // tiles the view needs, most urgent first
        std::uint64_t render_splats = 0;
        bool complete = false; // every visible region is covered by resident tiles
    };

    // 3D Tiles traversal: a tile refines when its screen-space error, measured to its
    // bounding volume, exceeds max_sse. A REPLACE tile draws until all its visible
    // descendants are resident, so the render set has no holes.
    [[nodiscard]] LFS_IO_API SplatTileSelection select_splat_tiles(
        const SplatTileSource& source, const SplatTileView& view,
        const std::function<bool(std::uint32_t)>& resident);

    // Decodes one tile and moves it to the GPU.
    [[nodiscard]] LFS_IO_API std::expected<core::SplatData, std::string> load_splat_tile_gpu(
        const SplatTileSource& source, std::uint32_t tile);

    // Merges decoded tiles into one model in the source-local frame; null for no tiles.
    // With an allocator the merged tensors are allocated by it (e.g. renderer storage),
    // so the result needs no further migration copy.
    [[nodiscard]] LFS_IO_API std::unique_ptr<core::SplatData> merge_splat_tiles(
        const SplatTileSource& source, std::span<const std::uint32_t> tiles,
        const std::function<const core::SplatData*(std::uint32_t)>& splats,
        const core::SplatTensorAllocator& allocator = {});

    // A 3D Tiles tileset.json (sniffed by content, since the extension is generic).
    [[nodiscard]] LFS_IO_API bool is_tiles3d_path(const std::filesystem::path& path);
    // Parses the tile tree and probes each tile's splat count from its GLB header.
    // Tiles are expressed relative to the root tile's frame.
    [[nodiscard]] LFS_IO_API std::expected<std::shared_ptr<SplatTileSource>, std::string>
    open_tiles3d(const std::filesystem::path& path);

} // namespace lfs::io
