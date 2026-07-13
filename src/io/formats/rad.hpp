/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "rad_packed_page.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace lfs::io {

    using lfs::core::SplatData;

    struct RadDecodedChunk {
        std::uint64_t base = 0;
        std::uint64_t count = 0;
        int max_sh_degree = 0;
        std::uint32_t sh_coeffs_rest = 0;
        bool lod_opacity_encoded = false;
        std::vector<float> means;
        std::vector<float> opacity_raw;
        std::vector<float> sh0_raw;
        std::vector<float> scaling_raw;
        std::vector<float> rotation_raw;
        std::vector<float> shN_canonical;
        std::vector<std::uint16_t> child_count;
        std::vector<std::uint32_t> child_start;
    };

    // Caller-provided destinations for a zero-copy chunk decode: properties
    // land directly in these buffers (each sized for the caller's capacity)
    // with the same post-decode transforms as load_rad_chunk. shN decodes via
    // the canonical scratch (resized by the decoder); null members are skipped.
    struct RadChunkDsts {
        float* means = nullptr;        // [capacity*3]
        float* opacity_raw = nullptr;  // [capacity]
        float* sh0_raw = nullptr;      // [capacity*3]
        float* scaling_raw = nullptr;  // [capacity*3]
        float* rotation_raw = nullptr; // [capacity*4]
        std::vector<float>* shN_canonical = nullptr;
    };
    struct RadChunkInfo {
        std::uint64_t base = 0;
        std::uint64_t count = 0;
        int max_sh_degree = 0;
        std::uint32_t sh_coeffs_rest = 0;
        bool lod_opacity_encoded = false;
    };
    [[nodiscard]] std::expected<RadChunkInfo, std::string> decode_rad_chunk_into(
        std::span<const std::uint8_t> data,
        int fallback_max_sh,
        bool lod_opacity_encoded,
        std::size_t dst_capacity,
        const RadChunkDsts& dsts);

    // Inflate-only chunk decode for the GPU dequant path: property planes land
    // in `dst` (an upload staging slot) still quantized, dimension-major, with
    // delta variants normalized away; the chunk's sidecar bounds/links planes
    // and dequant frame ride along. The returned descriptor drives the CUDA
    // page-dequant kernel. Streaming-profile only — per-component property
    // layouts and unknown encodings are hard errors, never fallbacks.
    [[nodiscard]] std::expected<RadPagePackedDesc, std::string> decode_rad_chunk_packed(
        std::span<const std::uint8_t> data,
        int fallback_max_sh,
        bool lod_opacity_encoded,
        std::size_t dst_capacity,
        const lfs::core::SplatLodTree::NodeMetaView& meta_view,
        std::uint32_t chunk,
        std::span<std::uint8_t> dst);

    // Load RAD (Random Access Dynamic) format - chunked hierarchical Gaussian splat format.
    // Overrides exist for deterministic tests; production callers use the
    // heuristics (out-of-core when the workset exceeds half of available RAM).
    struct RadLoadOverrides {
        std::optional<bool> out_of_core;
        std::optional<std::size_t> preview_splats;
    };
    std::expected<SplatData, std::string> load_rad(const std::filesystem::path& filepath);
    std::expected<SplatData, std::string> load_rad(const std::filesystem::path& filepath,
                                                   const RadLoadOverrides& overrides);
    std::expected<RadDecodedChunk, std::string> load_rad_chunk(
        const std::filesystem::path& filepath,
        const lfs::core::SplatLodTree::ChunkFileRange& range,
        int max_sh_degree,
        bool lod_opacity_encoded);
    // Reuses an already-open stream; for callers issuing many chunk reads
    // (streaming page caches) where per-chunk open/close costs add up.
    std::expected<RadDecodedChunk, std::string> load_rad_chunk(
        std::istream& in,
        const std::filesystem::path& filepath_for_errors,
        const lfs::core::SplatLodTree::ChunkFileRange& range,
        int max_sh_degree,
        bool lod_opacity_encoded);

    // True when a chunked RAD should keep its leaf tensors on the host and
    // stream pages to the GPU instead of migrating everything to CUDA at load.
    [[nodiscard]] bool rad_paged_load_recommended(const SplatData& data);

    // ------------------------------------------------------------------------
    // Node-metadata sidecar (<file>.rad.meta) — a derived cache, NOT part of
    // the RAD format. Holds per-node bounds/links quantized to 20 B/node
    // (per-chunk dequant frames) so out-of-core opens keep tree metadata on
    // disk (mmap) instead of in RAM, and cached re-opens skip decoding every
    // chunk. 1B-leaf trees fit ~34 GB.
    // ------------------------------------------------------------------------
    [[nodiscard]] std::filesystem::path rad_meta_sidecar_path(const std::filesystem::path& rad_path);
    // Validates magic/version/completeness and that the sidecar matches the
    // RAD file (size + mtime fast check, header hash authoritative).
    [[nodiscard]] std::expected<lfs::core::SplatLodTree::NodeMetaView, std::string>
    open_rad_meta_sidecar(const std::filesystem::path& rad_path);
    [[nodiscard]] Result<void> build_rad_meta_sidecar(
        const std::filesystem::path& rad_path,
        const ExportProgressCallback& progress = nullptr);
    // Header probe for RAD LOD files; flat RADs return std::nullopt.
    [[nodiscard]] std::expected<std::optional<std::uint32_t>, std::string> rad_lod_file_chunk_size(
        const std::filesystem::path& input);
    using RechunkProgressCallback = std::function<bool(float)>;
    // Re-encodes a RAD LOD file at the requested file chunk size. Node order,
    // tree links, and logical indices are unchanged.
    [[nodiscard]] Result<void> rechunk_rad_lod(
        const std::filesystem::path& input,
        const std::filesystem::path& output,
        std::uint32_t target_chunk_size,
        const RechunkProgressCallback& progress = nullptr);
    // Exposed for tests: scatter-derive parent/level over a BFS level-ordered,
    // children-contiguous links plane. child_start may be non-monotone across
    // parents within a level (multi-bucket converter layouts).
    [[nodiscard]] std::expected<std::uint64_t, std::string> derive_rad_meta_parents_levels(
        std::span<lfs::core::RadMetaLinksQ> links);
    // Dequantizes one chunk's sidecar records into expanded node bounds/links
    // records. Production keeps the quantized records resident and dequantizes
    // in the selector; this remains the CPU reference for tests.
    void expand_rad_meta_page(const lfs::core::SplatLodTree::NodeMetaView& view,
                              std::uint32_t chunk,
                              std::size_t node_count,
                              lfs::core::NodeBoundsRecord* out_bounds,
                              lfs::core::NodeLinksRecord* out_links);

    // One chunk of pack-domain splat arrays for streaming RAD export.
    // All values use the on-disk RAD domains: display alpha (lodOpacity),
    // display RGB (0.5 + SH_C0 * sh0_raw), linear scales, normalized [w,x,y,z]
    // quaternions, canonical shN.
    struct RadStreamChunkSource {
        std::uint32_t count = 0;
        const float* means = nullptr;               // [count*3]
        const float* alpha = nullptr;               // [count]
        const float* rgb = nullptr;                 // [count*3]
        const float* scales = nullptr;              // [count*3]
        const float* rotation = nullptr;            // [count*4]
        const float* shN = nullptr;                 // [count*sh_coeffs*3], optional
        const std::uint16_t* child_count = nullptr; // [count], LOD tree only
        const std::uint32_t* child_start = nullptr; // [count], LOD tree only
    };

    enum class RadGpuQuantization {
        Auto,
        Disabled,
    };

    // Streams LOD RAD chunks to disk with bounded memory. The chunk index area
    // is reserved up front (total node count must be known) and backpatched on
    // finish(); the decoder tolerates the trailing space padding.
    // With emit_meta_sidecar (LOD only), the .rad.meta sidecar is written
    // alongside chunk emission from the same decoded values the standalone
    // builder would see, so finished files open without a rebuild pass.
    // Emission is best-effort and never fails the writer.
    class RadStreamWriter {
    public:
        RadStreamWriter(std::filesystem::path output_path,
                        std::uint64_t total_count,
                        int sh_degree,
                        bool lod_tree,
                        int compression_level = 6,
                        bool emit_meta_sidecar = false,
                        std::uint32_t chunk_size = kRadStreamableChunkSplats,
                        RadGpuQuantization gpu_quantization = RadGpuQuantization::Auto);
        ~RadStreamWriter();
        RadStreamWriter(const RadStreamWriter&) = delete;
        RadStreamWriter& operator=(const RadStreamWriter&) = delete;

        [[nodiscard]] std::expected<void, std::string> open();
        [[nodiscard]] std::expected<void, std::string> append(const RadStreamChunkSource& chunk);
        // Compresses the chunks in parallel, then writes them in order. Every
        // chunk except the final one of the file must hold a full chunk's
        // splat count.
        [[nodiscard]] std::expected<void, std::string> append_batch(std::span<const RadStreamChunkSource> chunks);
        [[nodiscard]] std::expected<void, std::string> finish();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::io
