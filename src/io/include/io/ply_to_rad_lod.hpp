/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "io/error.hpp"
#include "io/exporter.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>

namespace lfs::io {

    struct PlyGaussianInfo {
        std::uint64_t vertex_count = 0;
        int sh_degree = 0;
    };

    // Cheap header-only probe of a gaussian-splat PLY.
    [[nodiscard]] LFS_IO_API std::expected<PlyGaussianInfo, std::string> probe_ply_gaussians(
        const std::filesystem::path& input_path);

    // Per-bucket LOD tree builder. kBhatt is the quality-validated default;
    // kOctree (Morton octree + moment matching) builds the same contract
    // orders of magnitude faster but has not passed the real-scene PSNR gate
    // yet. The top tree over bucket roots always uses bhatt (it is tiny and
    // needs the leaf-index plumbing).
    enum class LodBuilder {
        kBhatt,
        kOctree,
    };

    struct PlyToRadLodOptions {
        // Scratch space for Morton buckets and per-bucket subtrees. Empty means
        // a directory next to the output file (same drive, atomic-rename safe).
        std::filesystem::path temp_dir;
        // Splats per spatial bucket; bounds peak RAM of the per-bucket LOD
        // build and its serial merge time, which is the conversion's critical
        // path. Smaller buckets trade a little cross-boundary merge quality
        // for parallelism.
        std::size_t target_bucket_splats = 2'000'000;
        // 0 = derive from available memory.
        std::size_t max_concurrent_buckets = 0;
        float lod_base = 1.25f;
        int compression_level = 6;
        // RAD splats per file chunk. Streamable exports use Spark's 65,536-splat
        // chunks; non-stream exports use LichtFeld's native 2,048-node chunks.
        std::uint32_t chunk_size = kRadStreamableChunkSplats;
        LodBuilder builder = LodBuilder::kBhatt;
        // kOctree only: splats per octree leaf group before binary pairing
        // takes over (OctreeLodBuildOptions::leaf_group_splats, clamped to
        // [2, 64]).
        std::uint32_t octree_leaf_splats = 8;
        // kOctree only: the per-bucket octree pass hands its surviving
        // representatives to the bhatt agglomerative builder once they fit
        // this budget, so the visible coarse levels carry similarity-ordered
        // merges instead of compounded cell-forced pairings. 0 = pure octree
        // to the bucket root (OctreeLodBuildOptions::bhatt_top_nodes).
        std::uint32_t octree_bhatt_top_nodes = 32768;
        // Replicates the source across a tiles_x by tiles_y grid on the X/Y
        // ground plane, offsetting each instance by the scene extent plus a
        // 1% margin. The source PLY is replayed once per tile, so no tiled
        // intermediate file is needed; total splats multiply accordingly.
        std::uint32_t tiles_x = 1;
        std::uint32_t tiles_y = 1;
        ExportProgressCallback progress = nullptr;
    };

    // Out-of-core conversion of a gaussian-splat PLY into a chunked LOD RAD
    // file. Peak memory is bounded by the bucket size, not the input size:
    // splats are Morton-bucketed to disk, each bucket gets a local
    // Bhattacharyya LOD subtree, bucket roots are merged into a top tree, and
    // the combined hierarchy is streamed out as RAD chunks.
    [[nodiscard]] LFS_IO_API Result<void> convert_ply_to_rad_lod(
        const std::filesystem::path& input_path,
        const std::filesystem::path& output_path,
        const PlyToRadLodOptions& options = {});

} // namespace lfs::io
