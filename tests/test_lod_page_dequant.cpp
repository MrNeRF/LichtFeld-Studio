/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// CPU/GPU parity for the packed page-dequant path: every chunk of a
// converter-produced RAD file must decode to the same pool contents through
// decode_rad_chunk_packed + the CUDA kernel as through the CPU reference
// decode_rad_chunk_into (+ swizzle + expand_rad_meta_page). Pure-arithmetic
// encodings must match bit-exactly; libm paths (exp/log/trig) within ULPs.
// Fixtures span SH degree 0-3 and both Auto-profile alpha encodings (r8 for
// chunks whose alpha stays <=1, f16 for merged-interior chunks above 1).

#include "core/cuda/sh_layout.cuh"
#include "core/environment.hpp"
#include "io/formats/rad.hpp"
#include "io/formats/rad_dequant_math.hpp"
#include "io/ply_to_rad_lod.hpp"
#include "rendering/lod_page_dequant_cuda.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <span>
#include <vector>

namespace {

    using lfs::core::SplatLodTree;

    constexpr std::size_t kPage = SplatLodTree::kChunkSplats;

    using EncodingKey = std::pair<std::uint32_t, std::uint32_t>;
    using EncodingSet = std::set<EncodingKey>;

    EncodingKey kindEnc(const lfs::io::RadPackedKind kind, const lfs::io::RadPackedEncoding enc) {
        return {static_cast<std::uint32_t>(kind), static_cast<std::uint32_t>(enc)};
    }

    void writeSyntheticShPly(const std::filesystem::path& path,
                             const std::size_t count,
                             const int rest_props,
                             const float opacity_raw) {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> pos(-40.0f, 40.0f);
        std::uniform_real_distribution<float> sh(-0.4f, 0.4f);
        std::uniform_real_distribution<float> log_scale(-6.0f, -2.0f);
        std::uniform_real_distribution<float> quat(-1.0f, 1.0f);

        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.good());
        out << "ply\nformat binary_little_endian 1.0\n"
            << "element vertex " << count << "\n";
        for (const char* name : {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"}) {
            out << "property float " << name << "\n";
        }
        for (int i = 0; i < rest_props; ++i) {
            out << "property float f_rest_" << i << "\n";
        }
        for (const char* name : {"opacity", "scale_0", "scale_1", "scale_2",
                                 "rot_0", "rot_1", "rot_2", "rot_3"}) {
            out << "property float " << name << "\n";
        }
        out << "end_header\n";

        const std::size_t rest = static_cast<std::size_t>(rest_props);
        std::vector<float> row(17 + rest);
        for (std::size_t i = 0; i < count; ++i) {
            row[0] = pos(rng);
            row[1] = pos(rng);
            row[2] = pos(rng);
            row[3] = row[4] = row[5] = 0.0f;
            row[6] = 0.1f + sh(rng);
            row[7] = 0.2f + sh(rng);
            row[8] = 0.3f + sh(rng);
            for (std::size_t c = 0; c < rest; ++c) {
                row[9 + c] = sh(rng) * 0.25f;
            }
            row[9 + rest] = opacity_raw;
            row[10 + rest] = row[11 + rest] = row[12 + rest] = log_scale(rng);
            row[13 + rest] = 1.0f + quat(rng);
            row[14 + rest] = quat(rng) * 0.3f;
            row[15 + rest] = quat(rng) * 0.3f;
            row[16 + rest] = quat(rng) * 0.3f;
            out.write(reinterpret_cast<const char*>(row.data()),
                      static_cast<std::streamsize>(row.size() * sizeof(float)));
        }
        ASSERT_TRUE(out.good());
    }

    // Reference of the retired CPU writeSwizzledShPage: canonical rows packed
    // into float4 slots, lane-swizzled per kShReorderSize block.
    void referenceSwizzle(const std::vector<float>& canonical,
                          const std::size_t count,
                          const std::uint32_t src_rest,
                          const std::uint32_t dst_slots,
                          std::vector<float>& dst) {
        std::fill(dst.begin(), dst.end(), 0.0f);
        if (count == 0 || dst_slots == 0u || src_rest == 0u) {
            return;
        }
        const std::size_t stride = static_cast<std::size_t>(src_rest) * lfs::core::kShChannels;
        for (std::size_t i = 0; i < count; ++i) {
            const float* const row = canonical.data() + i * stride;
            const std::size_t block = i / lfs::core::kShReorderSize;
            const std::size_t lane = i % lfs::core::kShReorderSize;
            for (std::uint32_t slot = 0; slot < dst_slots; ++slot) {
                const std::size_t float4_index =
                    block * dst_slots * lfs::core::kShReorderSize +
                    static_cast<std::size_t>(slot) * lfs::core::kShReorderSize + lane;
                for (std::size_t component = 0; component < 4u; ++component) {
                    const std::size_t source_index = static_cast<std::size_t>(slot) * 4u + component;
                    dst[float4_index * 4u + component] =
                        source_index < stride ? row[source_index] : 0.0f;
                }
            }
        }
    }

    template <typename T>
    std::vector<T> readDevice(const void* const ptr, const std::size_t count) {
        std::vector<T> host(count);
        EXPECT_EQ(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        return host;
    }

    struct DeviceBuffer {
        void* ptr = nullptr;
        explicit DeviceBuffer(const std::size_t bytes) {
            EXPECT_EQ(cudaMalloc(&ptr, bytes), cudaSuccess);
            EXPECT_EQ(cudaMemset(ptr, 0xCD, bytes), cudaSuccess);
        }
        ~DeviceBuffer() { (void)cudaFree(ptr); }
    };

    // Converts a synthetic PLY with `rest_props` f_rest properties, then runs
    // CPU-vs-kernel parity on every chunk. Records each (kind, encoding) the
    // converter emitted so callers can assert matrix coverage.
    void runConvertedFileParity(const std::string& name,
                                const std::size_t splat_count,
                                const int rest_props,
                                const float opacity_raw,
                                EncodingSet& encodings_seen) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lod_page_dequant" / name;
        std::filesystem::remove_all(temp_dir);
        std::filesystem::create_directories(temp_dir);
        const auto ply_path = temp_dir / (name + ".ply");
        const auto rad_path = temp_dir / (name + ".rad");
        writeSyntheticShPly(ply_path, splat_count, rest_props, opacity_raw);
        lfs::io::PlyToRadLodOptions options;
        options.target_bucket_splats = 65'536;
        options.temp_dir = temp_dir / "scratch";
        ASSERT_TRUE(lfs::io::convert_ply_to_rad_lod(ply_path, rad_path, options).has_value());
        ASSERT_TRUE(lfs::io::build_rad_meta_sidecar(rad_path).has_value());

        auto loaded = lfs::io::load_rad(rad_path);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        ASSERT_TRUE(loaded->lod_tree && loaded->lod_tree->rad_source.valid());
        const auto& source = loaded->lod_tree->rad_source;
        const bool lod_opacity = loaded->lod_tree->lod_opacity_encoded;
        const int max_sh = loaded->get_max_sh_degree();
        const int expected_sh =
            rest_props == 45 ? 3 : (rest_props == 24 ? 2 : (rest_props == 9 ? 1 : 0));
        ASSERT_EQ(max_sh, expected_sh);

        auto view = lfs::io::open_rad_meta_sidecar(rad_path);
        ASSERT_TRUE(view.has_value()) << view.error();

        const auto dst_rest =
            static_cast<std::uint32_t>((max_sh + 1) * (max_sh + 1) - 1);
        const std::uint32_t dst_slots = lfs::core::sh_float4_slots_for_rest(dst_rest);
        const std::size_t sh_floats =
            dst_rest > 0u ? lfs::core::sh_swizzled_byte_count(kPage, dst_rest) / sizeof(float)
                          : 0u;

        const std::size_t sh_slot_count = sh_floats / 4u;
        DeviceBuffer d_means(kPage * lfs::vis::lodq::kXyzBytes);
        DeviceBuffer d_sh0(kPage * lfs::vis::lodq::kSh0Bytes);
        DeviceBuffer d_shN(std::max<std::size_t>(sh_slot_count * lfs::vis::lodq::kShNSlotBytes, 16));
        DeviceBuffer d_rot(kPage * lfs::vis::lodq::kRotationBytes);
        DeviceBuffer d_scale(kPage * lfs::vis::lodq::kScalingBytes);
        DeviceBuffer d_opacity(kPage * lfs::vis::lodq::kOpacityBytes);
        DeviceBuffer d_frames(lfs::vis::lodq::kPageFrameBytes);
        DeviceBuffer d_bounds(kPage * lfs::vis::lodq::kMetaBoundsBytes);
        DeviceBuffer d_links(kPage * lfs::vis::lodq::kMetaLinksBytes);
        DeviceBuffer d_slot(64u << 20);

        lfs::vis::LodPoolDeviceView pool{};
        pool.means = static_cast<float*>(d_means.ptr);
        pool.sh0 = static_cast<uint2*>(d_sh0.ptr);
        pool.shN = dst_rest > 0u ? static_cast<std::uint32_t*>(d_shN.ptr) : nullptr;
        pool.rotation = static_cast<uint2*>(d_rot.ptr);
        pool.scaling = static_cast<uint2*>(d_scale.ptr);
        pool.opacity = static_cast<std::uint16_t*>(d_opacity.ptr);
        pool.page_frames = static_cast<float4*>(d_frames.ptr);
        pool.meta_bounds = static_cast<uint2*>(d_bounds.ptr);
        pool.meta_links = static_cast<std::uint32_t*>(d_links.ptr);
        pool.dst_rest = dst_rest;
        pool.dst_slots = dst_slots;

        std::ifstream in(rad_path, std::ios::binary);
        ASSERT_TRUE(in.good());

        std::vector<std::uint8_t> chunk_bytes;
        std::vector<std::uint8_t> slot(64u << 20);
        std::vector<float> ref_means(kPage * 3), ref_sh0(kPage * 3), ref_scale(kPage * 3);
        std::vector<float> ref_rot(kPage * 4), ref_opacity(kPage);
        std::vector<float> shN_canonical;
        std::vector<float> ref_shN(sh_floats);

        for (std::size_t c = 0; c < source.chunks.size(); ++c) {
            const auto& range = source.chunks[c];
            chunk_bytes.resize(range.file_bytes);
            in.seekg(static_cast<std::streamoff>(range.file_offset), std::ios::beg);
            in.read(reinterpret_cast<char*>(chunk_bytes.data()),
                    static_cast<std::streamsize>(chunk_bytes.size()));
            ASSERT_TRUE(in.good());

            const lfs::io::RadChunkDsts dsts{
                .means = ref_means.data(),
                .opacity_raw = ref_opacity.data(),
                .sh0_raw = ref_sh0.data(),
                .scaling_raw = ref_scale.data(),
                .rotation_raw = ref_rot.data(),
                .shN_canonical = &shN_canonical,
            };
            auto info = lfs::io::decode_rad_chunk_into(
                std::span<const std::uint8_t>(chunk_bytes), max_sh, lod_opacity, kPage, dsts);
            ASSERT_TRUE(info.has_value()) << info.error();
            const std::size_t count = static_cast<std::size_t>(info->count);
            referenceSwizzle(shN_canonical, count, info->sh_coeffs_rest, dst_slots, ref_shN);

            auto desc = lfs::io::decode_rad_chunk_packed(
                std::span<const std::uint8_t>(chunk_bytes), max_sh, lod_opacity, kPage,
                *view, static_cast<std::uint32_t>(c), std::span<std::uint8_t>(slot));
            ASSERT_TRUE(desc.has_value()) << desc.error();
            ASSERT_EQ(desc->count, info->count);
            for (std::uint32_t p = 0; p < desc->property_count; ++p) {
                encodings_seen.insert({desc->props[p].kind, desc->props[p].encoding});
            }

            ASSERT_EQ(cudaMemcpy(d_slot.ptr, slot.data(), desc->used_bytes,
                                 cudaMemcpyHostToDevice),
                      cudaSuccess);
            ASSERT_EQ(lfs::vis::launchLodPageDequant(
                          static_cast<const std::uint8_t*>(d_slot.ptr), *desc, pool,
                          /*page=*/0, static_cast<std::uint32_t>(kPage), nullptr),
                      cudaSuccess);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

            const auto gpu_means = readDevice<float>(d_means.ptr, count * 3);
            for (std::size_t i = 0; i < count * 3; ++i) {
                ASSERT_EQ(gpu_means[i], ref_means[i]) << "means bit-parity at " << i;
            }
            namespace radmath = lfs::io::radmath;
            // sh0 reaches f16 through bit-exact fp32 on both sides: the GPU
            // halves must equal floatToHalf(reference) exactly.
            const auto gpu_sh0 = readDevice<uint2>(d_sh0.ptr, count);
            for (std::size_t i = 0; i < count; ++i) {
                const std::uint16_t h[3] = {
                    static_cast<std::uint16_t>(gpu_sh0[i].x & 0xFFFFu),
                    static_cast<std::uint16_t>(gpu_sh0[i].x >> 16),
                    static_cast<std::uint16_t>(gpu_sh0[i].y & 0xFFFFu)};
                for (std::size_t d = 0; d < 3; ++d) {
                    ASSERT_EQ(h[d], radmath::floatToHalf(ref_sh0[i * 3 + d]))
                        << "sh0 half-parity at " << i << "," << d;
                }
            }
            // log-f16 scales pass through from the file; the fp32 reference
            // went f16->exp->log, so allow one f16 step.
            const auto gpu_scale = readDevice<uint2>(d_scale.ptr, count);
            const auto gpu_rot = readDevice<uint2>(d_rot.ptr, count);
            const auto gpu_opacity = readDevice<std::uint16_t>(d_opacity.ptr, count);
            for (std::size_t i = 0; i < count && i < 4096; ++i) {
                const float s[3] = {
                    radmath::halfToFloat(static_cast<std::uint16_t>(gpu_scale[i].x & 0xFFFFu)),
                    radmath::halfToFloat(static_cast<std::uint16_t>(gpu_scale[i].x >> 16)),
                    radmath::halfToFloat(static_cast<std::uint16_t>(gpu_scale[i].y & 0xFFFFu))};
                for (std::size_t d = 0; d < 3; ++d) {
                    ASSERT_NEAR(s[d], ref_scale[i * 3 + d],
                                std::max(2e-3, 2e-3 * std::abs(ref_scale[i * 3 + d])))
                        << "scaling at " << i << "," << d;
                }
                const float r[4] = {
                    radmath::halfToFloat(static_cast<std::uint16_t>(gpu_rot[i].x & 0xFFFFu)),
                    radmath::halfToFloat(static_cast<std::uint16_t>(gpu_rot[i].x >> 16)),
                    radmath::halfToFloat(static_cast<std::uint16_t>(gpu_rot[i].y & 0xFFFFu)),
                    radmath::halfToFloat(static_cast<std::uint16_t>(gpu_rot[i].y >> 16))};
                for (std::size_t d = 0; d < 4; ++d) {
                    ASSERT_NEAR(r[d], ref_rot[i * 4 + d], 2e-3) << "rotation at " << i << "," << d;
                }
                ASSERT_NEAR(radmath::halfToFloat(gpu_opacity[i]), ref_opacity[i],
                            std::max(2e-3, 2e-3 * std::abs(ref_opacity[i])))
                    << "opacity at " << i;
            }
            // s8 SH bytes pass through bit-exact; decoding them with the
            // frame scales must reproduce the reference floats exactly.
            if (dst_rest > 0u) {
                const std::uint32_t valid_floats = info->sh_coeffs_rest * 3u;
                const auto gpu_frames = readDevice<float>(d_frames.ptr, 4);
                const auto gpu_shN = readDevice<std::uint32_t>(d_shN.ptr, sh_slot_count);
                for (std::size_t k = 0; k < sh_floats; ++k) {
                    const std::uint32_t slot_idx = static_cast<std::uint32_t>(k / 4u);
                    const std::uint32_t comp = static_cast<std::uint32_t>(k % 4u);
                    // Swizzled float4 layout is [block][slot][lane].
                    const std::uint32_t slot_in_block =
                        (slot_idx % (dst_slots * lfs::core::kShReorderSize)) /
                        lfs::core::kShReorderSize;
                    const std::uint32_t cf = slot_in_block * 4u + comp;
                    const std::uint32_t band = cf / 3u < 3u ? 0u : (cf / 3u < 8u ? 1u : 2u);
                    const auto b = static_cast<std::int8_t>(
                        (gpu_shN[slot_idx] >> (comp * 8u)) & 0xFFu);
                    const float decoded =
                        cf < valid_floats
                            ? radmath::dequantS8(b, std::max(gpu_frames[band], 1e-6f))
                            : 0.0f;
                    ASSERT_EQ(decoded, ref_shN[k]) << "shN at float " << k;
                }
            }

            // Sidecar records pass through byte-exact; the per-page frame
            // must carry the chunk's dequant frame for the selector.
            const std::size_t logical_start = c * kPage;
            const std::size_t run = std::min(kPage, view->node_count - logical_start);
            const auto gpu_bounds = readDevice<uint2>(d_bounds.ptr, kPage);
            const auto gpu_links = readDevice<std::uint32_t>(d_links.ptr, kPage * 3u);
            for (std::size_t i = 0; i < run; ++i) {
                const auto& bq = view->bounds[logical_start + i];
                EXPECT_EQ(gpu_bounds[i].x,
                          static_cast<std::uint32_t>(bq.qx) |
                              (static_cast<std::uint32_t>(bq.qy) << 16u));
                EXPECT_EQ(gpu_bounds[i].y,
                          static_cast<std::uint32_t>(bq.qz) |
                              (static_cast<std::uint32_t>(bq.qsize) << 16u));
                const auto& lq = view->links[logical_start + i];
                EXPECT_EQ(gpu_links[i * 3u + 0u], lq.child_start);
                EXPECT_EQ(gpu_links[i * 3u + 1u], lq.packed);
                EXPECT_EQ(gpu_links[i * 3u + 2u], lq.parent);
            }
            for (std::size_t i = run; i < kPage; ++i) {
                EXPECT_EQ(gpu_links[i * 3u + 0u], 0xFFFFFFFFu) << "slack links at " << i;
                EXPECT_EQ(gpu_bounds[i].x, 0u) << "slack bounds at " << i;
                EXPECT_EQ(gpu_bounds[i].y, 0u) << "slack bounds at " << i;
            }
            const auto frame_floats = readDevice<float>(d_frames.ptr, 16);
            const auto& record = view->chunks[c];
            for (std::size_t d = 0; d < 3; ++d) {
                EXPECT_EQ(frame_floats[4 + d], record.bbox_min[d]);
                EXPECT_EQ(frame_floats[8 + d], record.bbox_extent[d]);
            }
            EXPECT_EQ(frame_floats[7], record.log_size_min);
            EXPECT_EQ(frame_floats[11], record.log_size_range);
            if (::testing::Test::HasFailure()) {
                return;
            }
        }
    }

    TEST(LodPageDequant, KernelMatchesCpuDecodeOnConvertedFile) {
        EncodingSet seen;
        runConvertedFileParity("sh3", 150'000, 45, /*opacity_raw=*/2.0f, seen);
        using lfs::io::RadPackedEncoding;
        using lfs::io::RadPackedKind;
        // Full Auto-profile matrix for an SH3 LOD tree. Opaque leaves make
        // merged-interior alpha exceed 1, and bhatt levels don't align with
        // chunk boundaries, so every chunk takes the f16 alpha path here; the
        // degree-1 fixture covers r8.
        EXPECT_TRUE(seen.count(kindEnc(RadPackedKind::Means, RadPackedEncoding::F32LeBytes)));
        EXPECT_TRUE(seen.count(kindEnc(RadPackedKind::Alpha, RadPackedEncoding::F16)));
        EXPECT_TRUE(seen.count(kindEnc(RadPackedKind::Sh0, RadPackedEncoding::R8)));
        EXPECT_TRUE(seen.count(kindEnc(RadPackedKind::Scales, RadPackedEncoding::LnF16)));
        EXPECT_TRUE(seen.count(kindEnc(RadPackedKind::Rotation, RadPackedEncoding::Oct88R8)));
        EXPECT_TRUE(seen.count(kindEnc(RadPackedKind::Sh1, RadPackedEncoding::S8)));
        EXPECT_TRUE(seen.count(kindEnc(RadPackedKind::Sh2, RadPackedEncoding::S8)));
        EXPECT_TRUE(seen.count(kindEnc(RadPackedKind::Sh3, RadPackedEncoding::S8)));
    }

    TEST(LodPageDequant, KernelMatchesCpuDecodeDegree2) {
        EncodingSet seen;
        runConvertedFileParity("sh2", 80'000, 24, /*opacity_raw=*/2.0f, seen);
        using lfs::io::RadPackedEncoding;
        using lfs::io::RadPackedKind;
        EXPECT_TRUE(seen.count(kindEnc(RadPackedKind::Sh1, RadPackedEncoding::S8)));
        EXPECT_TRUE(seen.count(kindEnc(RadPackedKind::Sh2, RadPackedEncoding::S8)));
        EXPECT_FALSE(seen.count(kindEnc(RadPackedKind::Sh3, RadPackedEncoding::S8)));
    }

    TEST(LodPageDequant, KernelMatchesCpuDecodeDegree1) {
        EncodingSet seen;
        // Near-transparent leaves keep merged alpha below 1 in every chunk,
        // forcing the Auto profile onto the r8 alpha path.
        runConvertedFileParity("sh1", 80'000, 9, /*opacity_raw=*/-4.0f, seen);
        using lfs::io::RadPackedEncoding;
        using lfs::io::RadPackedKind;
        EXPECT_TRUE(seen.count(kindEnc(RadPackedKind::Alpha, RadPackedEncoding::R8)));
        EXPECT_TRUE(seen.count(kindEnc(RadPackedKind::Sh1, RadPackedEncoding::S8)));
        EXPECT_FALSE(seen.count(kindEnc(RadPackedKind::Sh2, RadPackedEncoding::S8)));
    }

    TEST(LodPageDequant, KernelMatchesCpuDecodeDegree0) {
        EncodingSet seen;
        runConvertedFileParity("sh0", 80'000, 0, /*opacity_raw=*/2.0f, seen);
        using lfs::io::RadPackedEncoding;
        using lfs::io::RadPackedKind;
        EXPECT_TRUE(seen.count(kindEnc(RadPackedKind::Means, RadPackedEncoding::F32LeBytes)));
        EXPECT_FALSE(seen.count(kindEnc(RadPackedKind::Sh1, RadPackedEncoding::S8)));
    }

    // CPU sink-cost comparison on a real converted file: full decode (old
    // sink) vs inflate-only packed decode (new sink). Informational; run with
    // LFS_RAD_BENCH_FILE=<file.rad> pointing at a RAD with a .rad.meta.
    TEST(LodPageDequant, SinkCpuCostBenchmark) {
        const auto bench = lfs::core::environment::value("LFS_RAD_BENCH_FILE");
        if (!bench) {
            GTEST_SKIP() << "set LFS_RAD_BENCH_FILE to run";
        }
        const std::filesystem::path rad_path(*bench);
        auto loaded = lfs::io::load_rad(rad_path);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        ASSERT_TRUE(loaded->lod_tree && loaded->lod_tree->rad_source.valid());
        const auto& source = loaded->lod_tree->rad_source;
        auto view = lfs::io::open_rad_meta_sidecar(rad_path);
        ASSERT_TRUE(view.has_value()) << view.error();
        const bool lod_opacity = loaded->lod_tree->lod_opacity_encoded;
        const int max_sh = loaded->get_max_sh_degree();

        const std::size_t n = std::min<std::size_t>(source.chunks.size(), 256);
        std::ifstream in(rad_path, std::ios::binary);
        std::vector<std::vector<std::uint8_t>> raw(n);
        for (std::size_t c = 0; c < n; ++c) {
            raw[c].resize(source.chunks[c].file_bytes);
            in.seekg(static_cast<std::streamoff>(source.chunks[c].file_offset), std::ios::beg);
            in.read(reinterpret_cast<char*>(raw[c].data()),
                    static_cast<std::streamsize>(raw[c].size()));
            ASSERT_TRUE(in.good());
        }

        // Touch the sidecar planes once so neither loop measures cold mmap
        // page faults; both the old sink (expand_rad_meta_page) and the new
        // one (plane memcpy) read the same bytes in production.
        std::uint64_t warm = 0;
        for (std::size_t c = 0; c < n; ++c) {
            const std::size_t start = c * kPage;
            const std::size_t run = std::min(kPage, view->node_count - start);
            const auto* const b = reinterpret_cast<const std::uint8_t*>(view->bounds + start);
            const auto* const l = reinterpret_cast<const std::uint8_t*>(view->links + start);
            for (std::size_t i = 0; i < run * sizeof(lfs::core::RadMetaBoundsQ); i += 4096) {
                warm += b[i];
            }
            for (std::size_t i = 0; i < run * sizeof(lfs::core::RadMetaLinksQ); i += 4096) {
                warm += l[i];
            }
        }
        ASSERT_GE(warm, 0u);

        std::vector<float> means(kPage * 3), sh0(kPage * 3), scale(kPage * 3);
        std::vector<float> rot(kPage * 4), opacity(kPage);
        std::vector<float> shN_canonical;
        std::vector<float> shN_swizzled(
            lfs::core::sh_swizzled_byte_count(kPage, 15) / sizeof(float));
        std::vector<lfs::core::NodeBoundsRecord> bounds(kPage);
        std::vector<lfs::core::NodeLinksRecord> links(kPage);
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t c = 0; c < n; ++c) {
            const lfs::io::RadChunkDsts dsts{
                .means = means.data(),
                .opacity_raw = opacity.data(),
                .sh0_raw = sh0.data(),
                .scaling_raw = scale.data(),
                .rotation_raw = rot.data(),
                .shN_canonical = &shN_canonical,
            };
            auto info = lfs::io::decode_rad_chunk_into(
                std::span<const std::uint8_t>(raw[c]), max_sh, lod_opacity, kPage, dsts);
            ASSERT_TRUE(info.has_value()) << info.error();
            // The old sink's remaining CPU work: SH swizzle + meta expansion.
            referenceSwizzle(shN_canonical, static_cast<std::size_t>(info->count),
                             info->sh_coeffs_rest,
                             lfs::core::sh_float4_slots_for_rest(15), shN_swizzled);
            std::memset(bounds.data(), 0, bounds.size() * sizeof(bounds[0]));
            std::memset(links.data(), 0xFF, links.size() * sizeof(links[0]));
            const std::size_t start = c * kPage;
            const std::size_t run = std::min(kPage, view->node_count - start);
            lfs::io::expand_rad_meta_page(*view, static_cast<std::uint32_t>(c), run,
                                          bounds.data(), links.data());
        }
        const auto t1 = std::chrono::steady_clock::now();
        std::vector<std::uint8_t> slot(64u << 20);
        for (std::size_t c = 0; c < n; ++c) {
            auto desc = lfs::io::decode_rad_chunk_packed(
                std::span<const std::uint8_t>(raw[c]), max_sh, lod_opacity, kPage,
                *view, static_cast<std::uint32_t>(c), std::span<std::uint8_t>(slot));
            ASSERT_TRUE(desc.has_value()) << desc.error();
        }
        const auto t2 = std::chrono::steady_clock::now();

        const auto ms = [](const auto a, const auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const double full_ms = ms(t0, t1) / static_cast<double>(n);
        const double packed_ms = ms(t1, t2) / static_cast<double>(n);
        std::printf("sink CPU cost over %zu chunks: full decode %.3f ms/chunk "
                    "(%.0f chunks/s/core), packed %.3f ms/chunk (%.0f chunks/s/core)\n",
                    n, full_ms, 1000.0 / full_ms, packed_ms, 1000.0 / packed_ms);
    }

} // namespace
