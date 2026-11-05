/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/formats/rad.hpp"
#include "io/loaders/rad_loader.hpp"
#include "io/ply_to_rad_lod.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <zlib.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

    using lfs::core::NodeBoundsRecord;
    using lfs::core::NodeLinksRecord;
    using lfs::core::RadMetaLinksQ;
    using lfs::core::SplatLodTree;

    RadMetaLinksQ makeNode(const std::uint32_t child_start, const std::uint32_t child_count) {
        return {
            .child_start = child_start,
            .packed = child_count & 0xffffu,
            .parent = 0xFFFFFFFFu,
        };
    }

    std::uint32_t levelOf(const RadMetaLinksQ& rec) { return (rec.packed >> 16u) & 0xffu; }

    TEST(RadMetaSidecar, DeriveParentsHandlesNonMonotoneChildStart) {
        // Level-ordered tree whose level-1 parents point at non-monotone
        // child_start ranges, as the multi-bucket converter layouts produce:
        // node 0 (root) -> [1, 3); node 1 -> [5, 7); node 2 -> [3, 5).
        std::vector<RadMetaLinksQ> links{
            makeNode(1, 2),
            makeNode(5, 2),
            makeNode(3, 2),
            makeNode(0, 0),
            makeNode(0, 0),
            makeNode(0, 0),
            makeNode(0, 0),
        };
        const auto leaf_count = lfs::io::derive_rad_meta_parents_levels(std::span(links));
        ASSERT_TRUE(leaf_count.has_value()) << leaf_count.error();
        EXPECT_EQ(*leaf_count, 4u);

        EXPECT_EQ(links[0].parent, 0xFFFFFFFFu);
        EXPECT_EQ(links[1].parent, 0u);
        EXPECT_EQ(links[2].parent, 0u);
        EXPECT_EQ(links[3].parent, 2u);
        EXPECT_EQ(links[4].parent, 2u);
        EXPECT_EQ(links[5].parent, 1u);
        EXPECT_EQ(links[6].parent, 1u);
        EXPECT_EQ(levelOf(links[0]), 0u);
        EXPECT_EQ(levelOf(links[1]), 1u);
        EXPECT_EQ(levelOf(links[2]), 1u);
        for (std::size_t i = 3; i < links.size(); ++i) {
            EXPECT_EQ(levelOf(links[i]), 2u);
        }
    }

    TEST(RadMetaSidecar, DeriveParentsRejectsCorruptLayouts) {
        // Child range pointing backwards.
        std::vector<RadMetaLinksQ> backwards{makeNode(0, 1), makeNode(0, 0)};
        EXPECT_FALSE(lfs::io::derive_rad_meta_parents_levels(std::span(backwards)).has_value());

        // Orphan node (no parent assigns it).
        std::vector<RadMetaLinksQ> orphan{makeNode(1, 1), makeNode(0, 0), makeNode(0, 0)};
        EXPECT_FALSE(lfs::io::derive_rad_meta_parents_levels(std::span(orphan)).has_value());
    }

    struct SyntheticSplat {
        float x, y, z;
        float nx = 0.0f, ny = 0.0f, nz = 0.0f;
        float dc0, dc1, dc2;
        float opacity;
        float s0, s1, s2;
        float r0, r1, r2, r3;
    };

    void writeSyntheticPly(const std::filesystem::path& path, const std::size_t count) {
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> pos(-50.0f, 50.0f);
        std::uniform_real_distribution<float> log_scale(-6.0f, -2.0f);
        std::vector<SyntheticSplat> splats(count);
        for (auto& s : splats) {
            const float ls = log_scale(rng);
            s = {.x = pos(rng), .y = pos(rng), .z = pos(rng), .dc0 = 0.1f, .dc1 = 0.2f, .dc2 = 0.3f, .opacity = 1.0f, .s0 = ls, .s1 = ls, .s2 = ls, .r0 = 1.0f, .r1 = 0.0f, .r2 = 0.0f, .r3 = 0.0f};
        }
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.good());
        out << "ply\nformat binary_little_endian 1.0\n"
            << "element vertex " << splats.size() << "\n";
        for (const char* name : {"x", "y", "z", "nx", "ny", "nz",
                                 "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                                 "scale_0", "scale_1", "scale_2",
                                 "rot_0", "rot_1", "rot_2", "rot_3"}) {
            out << "property float " << name << "\n";
        }
        out << "end_header\n";
        static_assert(sizeof(SyntheticSplat) == 17 * sizeof(float));
        out.write(reinterpret_cast<const char*>(splats.data()),
                  static_cast<std::streamsize>(splats.size() * sizeof(SyntheticSplat)));
        ASSERT_TRUE(out.good());
    }

    std::filesystem::path makeTestRad(const std::filesystem::path& temp_dir) {
        std::filesystem::create_directories(temp_dir);
        const auto ply_path = temp_dir / "sidecar.ply";
        const auto rad_path = temp_dir / "sidecar.rad";
        writeSyntheticPly(ply_path, 200'000);
        lfs::io::PlyToRadLodOptions options;
        options.target_bucket_splats = 65'536;
        options.temp_dir = temp_dir / "scratch";
        EXPECT_TRUE(lfs::io::convert_ply_to_rad_lod(ply_path, rad_path, options).has_value());
        return rad_path;
    }

    TEST(RadMetaSidecar, ConverterEmitsSidecarInline) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "rad_meta_sidecar_inline";
        std::filesystem::remove_all(temp_dir);
        const auto rad_path = makeTestRad(temp_dir);
        const auto meta_path = lfs::io::rad_meta_sidecar_path(rad_path);

        // The converter publishes the sidecar during chunk emission; opening
        // must succeed without any rebuild pass.
        ASSERT_TRUE(std::filesystem::exists(meta_path));
        {
            auto view = lfs::io::open_rad_meta_sidecar(rad_path);
            ASSERT_TRUE(view.has_value()) << view.error();
            EXPECT_GT(view->node_count, 200'000u);
            EXPECT_EQ(view->leaf_count, 200'000u);
            EXPECT_EQ(view->links[0].parent, 0xFFFFFFFFu);
        }

        // Inline emission must replicate the standalone builder bit for bit.
        const auto read_all = [](const std::filesystem::path& p) {
            std::ifstream in(p, std::ios::binary);
            return std::vector<char>(std::istreambuf_iterator<char>(in),
                                     std::istreambuf_iterator<char>());
        };
        const auto inline_bytes = read_all(meta_path);
        ASSERT_FALSE(inline_bytes.empty());
        std::filesystem::remove(meta_path);
        ASSERT_TRUE(lfs::io::build_rad_meta_sidecar(rad_path).has_value());
        const auto rebuilt_bytes = read_all(meta_path);
        ASSERT_EQ(inline_bytes.size(), rebuilt_bytes.size());
        EXPECT_TRUE(inline_bytes == rebuilt_bytes)
            << "inline sidecar diverges from a standalone rebuild";

        std::filesystem::remove_all(temp_dir);
    }

    TEST(RadMetaSidecar, RoundtripQuantizationAndInvalidation) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "rad_meta_sidecar";
        std::filesystem::remove_all(temp_dir);
        const auto rad_path = makeTestRad(temp_dir);

        // Full in-RAM load is the ground truth the quantized view must match.
        auto full = lfs::io::load_rad(rad_path);
        ASSERT_TRUE(full.has_value()) << full.error();
        ASSERT_TRUE(full->lod_tree && full->lod_tree->nodes_in_memory());
        const auto& tree = *full->lod_tree;

        ASSERT_TRUE(lfs::io::build_rad_meta_sidecar(rad_path).has_value());
        const auto meta_path = lfs::io::rad_meta_sidecar_path(rad_path);
        ASSERT_TRUE(std::filesystem::exists(meta_path));

        auto view = lfs::io::open_rad_meta_sidecar(rad_path);
        ASSERT_TRUE(view.has_value()) << view.error();
        EXPECT_GT(view->node_count, 200'000u);
        EXPECT_EQ(view->leaf_count, 200'000u);
        EXPECT_EQ(view->node_count, tree.total_nodes());
        EXPECT_EQ(view->links[0].parent, 0xFFFFFFFFu);

        // Links bit-exact; bounds within the per-chunk quantization tolerance.
        for (std::size_t i = 0; i < view->node_count; ++i) {
            ASSERT_EQ(view->links[i].child_start, tree.child_start[i]) << "node " << i;
            ASSERT_EQ(view->links[i].packed & 0xffffu, tree.child_count[i]) << "node " << i;
            const auto& frame = view->chunkOf(i);
            const glm::vec3 center = frame.dequantCenter(view->bounds[i]);
            const glm::vec3 truth = tree.centers[i];
            for (int d = 0; d < 3; ++d) {
                const float tolerance = std::max(2.0f * frame.bbox_extent[d] / 65535.0f, 1e-6f);
                ASSERT_NEAR(center[d], truth[d], tolerance) << "node " << i << " dim " << d;
            }
            const float size = frame.dequantSize(view->bounds[i]);
            const float size_truth = tree.sizes[i];
            const float rel_tolerance =
                std::max(2.0f * frame.log_size_range / 65535.0f, 1e-5f);
            ASSERT_NEAR(std::log(std::max(size, 1e-20f)),
                        std::log(std::max(size_truth, 1e-20f)),
                        rel_tolerance)
                << "node " << i;
        }

        // Page expansion produces the exact GPU records, logical included.
        constexpr std::size_t kChunk = SplatLodTree::kChunkSplats;
        const std::size_t expand_chunk = view->chunk_count > 1 ? 1 : 0;
        const std::size_t logical_start = expand_chunk * kChunk;
        const std::size_t run = std::min(kChunk, view->node_count - logical_start);
        std::vector<NodeBoundsRecord> bounds(run);
        std::vector<NodeLinksRecord> links(run);
        lfs::io::expand_rad_meta_page(*view, static_cast<std::uint32_t>(expand_chunk), run,
                                      bounds.data(), links.data());
        for (std::size_t i = 0; i < run; ++i) {
            EXPECT_EQ(links[i].logical, logical_start + i);
            EXPECT_EQ(links[i].child_start, tree.child_start[logical_start + i]);
            EXPECT_EQ(links[i].childCount(), tree.child_count[logical_start + i]);
            // Cross-TU FP contraction may differ by a few ULPs.
            const float expected =
                view->chunkOf(logical_start + i).dequantSize(view->bounds[logical_start + i]);
            EXPECT_NEAR(bounds[i].size, expected, std::abs(expected) * 1e-5f);
        }

        // Touching the RAD file invalidates the sidecar (mtime/hash stamp).
        {
            std::ofstream touch(rad_path, std::ios::binary | std::ios::app);
            touch.put('\0');
        }
        auto stale = lfs::io::open_rad_meta_sidecar(rad_path);
        EXPECT_FALSE(stale.has_value());

        // A rebuild repairs it.
        ASSERT_TRUE(lfs::io::build_rad_meta_sidecar(rad_path).has_value());
        EXPECT_TRUE(lfs::io::open_rad_meta_sidecar(rad_path).has_value());

        // An incomplete build (complete flag cleared) is rejected.
        {
            std::fstream corrupt(meta_path, std::ios::binary | std::ios::in | std::ios::out);
            corrupt.seekp(89, std::ios::beg); // RadMetaHeader::complete byte offset
            corrupt.put('\0');
        }
        auto incomplete = lfs::io::open_rad_meta_sidecar(rad_path);
        EXPECT_FALSE(incomplete.has_value());

        // Wrong version (e.g. a leftover v1 file) is rejected too.
        {
            std::fstream corrupt(meta_path, std::ios::binary | std::ios::in | std::ios::out);
            const std::uint32_t old_version = 1;
            corrupt.seekp(4, std::ios::beg); // RadMetaHeader::version offset
            corrupt.write(reinterpret_cast<const char*>(&old_version), sizeof(old_version));
        }
        auto wrong_version = lfs::io::open_rad_meta_sidecar(rad_path);
        EXPECT_FALSE(wrong_version.has_value());

        std::filesystem::remove_all(temp_dir);
    }

#ifndef _WIN32
    TEST(RadMetaSidecar, BuildFailsCleanlyInReadOnlyDirectory) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "rad_meta_sidecar_ro";
        std::filesystem::remove_all(temp_dir);
        const auto rad_path = makeTestRad(temp_dir);

        std::filesystem::permissions(temp_dir,
                                     std::filesystem::perms::owner_read |
                                         std::filesystem::perms::owner_exec,
                                     std::filesystem::perm_options::replace);
        const auto result = lfs::io::build_rad_meta_sidecar(rad_path);
        std::filesystem::permissions(temp_dir,
                                     std::filesystem::perms::owner_all,
                                     std::filesystem::perm_options::replace);

        ASSERT_FALSE(result.has_value());
        EXPECT_FALSE(result.error().message.empty());

        std::filesystem::remove_all(temp_dir);
    }
#endif

    bool writeRadProbeFixture(const std::filesystem::path& path, const void* data, const size_t size) {
        std::ofstream file(path, std::ios::binary);
        if (!file)
            return false;
        if (size > 0)
            file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        file.flush();
        return file.good();
    }

    class RadLoaderShortProbeTest : public ::testing::TestWithParam<size_t> {};

    TEST_P(RadLoaderShortProbeTest, ValidateOnlyRejectsTruncatedMagicDeterministically) {
        const auto short_bytes = GetParam();
        const auto temp_dir = std::filesystem::temp_directory_path() / "rad_loader_short_probe";
        std::filesystem::remove_all(temp_dir);
        std::filesystem::create_directories(temp_dir);

        constexpr uint8_t rad_magic[4] = {0x52, 0x41, 0x44, 0x30};
        const auto probe_path = temp_dir / std::format("truncated_{}.rad", short_bytes);
        ASSERT_TRUE(writeRadProbeFixture(probe_path, rad_magic, short_bytes));

        lfs::io::RadLoader loader;
        lfs::io::LoadOptions options;
        options.validate_only = true;
        const auto result = loader.load(probe_path, options);

        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, lfs::io::ErrorCode::INVALID_HEADER);
        EXPECT_NE(result.error().message.find(std::format("expected 4 bytes, got {}", short_bytes)),
                  std::string::npos);

        std::filesystem::remove_all(temp_dir);
    }

    INSTANTIATE_TEST_SUITE_P(
        ByteLength,
        RadLoaderShortProbeTest,
        ::testing::Values(size_t{0}, size_t{1}, size_t{2}, size_t{3}));

    TEST(RadLoaderProbeTest, ValidateOnlyAcceptsValidFourByteMagicPrefix) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "rad_loader_valid_probe";
        std::filesystem::remove_all(temp_dir);
        std::filesystem::create_directories(temp_dir);

        constexpr uint8_t rad_magic[4] = {0x52, 0x41, 0x44, 0x30};
        const auto probe_path = temp_dir / "valid_magic.rad";
        ASSERT_TRUE(writeRadProbeFixture(probe_path, rad_magic, sizeof(rad_magic)));

        lfs::io::RadLoader loader;
        lfs::io::LoadOptions options;
        options.validate_only = true;
        const auto result = loader.load(probe_path, options);

        ASSERT_TRUE(result.has_value()) << result.error().format();

        std::filesystem::remove_all(temp_dir);
    }

    TEST(RadLoaderProbeTest, ValidateOnlyRejectsWrongFourByteMagic) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "rad_loader_wrong_probe";
        std::filesystem::remove_all(temp_dir);
        std::filesystem::create_directories(temp_dir);

        constexpr uint8_t wrong_magic[4] = {0x00, 0x00, 0x00, 0x00};
        const auto probe_path = temp_dir / "wrong_magic.rad";
        ASSERT_TRUE(writeRadProbeFixture(probe_path, wrong_magic, sizeof(wrong_magic)));

        lfs::io::RadLoader loader;
        lfs::io::LoadOptions options;
        options.validate_only = true;
        const auto result = loader.load(probe_path, options);

        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, lfs::io::ErrorCode::INVALID_HEADER);
        EXPECT_NE(result.error().message.find("RAD0"), std::string::npos);

        std::filesystem::remove_all(temp_dir);
    }

    // ---- Malformed RAD input: a minimal single-chunk container builder ----

    constexpr std::uint32_t kRadMagic = 0x30444152u;      // "RAD0"
    constexpr std::uint32_t kRadChunkMagic = 0x43444152u; // "RADC"

    std::size_t pad8(const std::size_t n) { return (n + 7u) & ~std::size_t{7u}; }

    void putU32(std::vector<std::uint8_t>& out, const std::uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
        }
    }

    void putU64(std::vector<std::uint8_t>& out, const std::uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
        }
    }

    // JSON blocks are space-padded to 8 bytes; the length field stores the unpadded size.
    void putJsonBlock(std::vector<std::uint8_t>& out, const std::string& json) {
        out.insert(out.end(), json.begin(), json.end());
        out.insert(out.end(), pad8(json.size()) - json.size(), static_cast<std::uint8_t>(' '));
    }

    struct RawProperty {
        std::string name;
        std::string encoding;
        std::vector<std::uint8_t> bytes; // stored bytes, already deflated when `deflated`
        bool deflated = false;
    };

    std::vector<std::uint8_t> floatBytes(const std::vector<float>& values) {
        std::vector<std::uint8_t> out(values.size() * sizeof(float));
        std::memcpy(out.data(), values.data(), out.size());
        return out;
    }

    // Raw deflate stream, the variant rad_decompress tries first.
    std::vector<std::uint8_t> rawDeflate(const std::vector<std::uint8_t>& input) {
        z_stream strm{};
        if (deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
            return {};
        }
        std::vector<std::uint8_t> out(deflateBound(&strm, input.size()));
        strm.next_in = const_cast<Bytef*>(input.data());
        strm.avail_in = static_cast<uInt>(input.size());
        strm.next_out = out.data();
        strm.avail_out = static_cast<uInt>(out.size());
        const int ret = deflate(&strm, Z_FINISH);
        out.resize(strm.total_out);
        deflateEnd(&strm);
        return ret == Z_STREAM_END ? out : std::vector<std::uint8_t>{};
    }

    // Builds a RAD file holding one chunk that covers splats [0, count).
    // `max_sh` is written to the file metadata only when positive.
    std::vector<std::uint8_t> buildRadFile(const std::uint64_t count,
                                           const std::vector<RawProperty>& props,
                                           const int max_sh = 0) {
        std::vector<std::uint8_t> payload;
        nlohmann::json prop_json = nlohmann::json::array();
        for (const auto& prop : props) {
            const std::size_t offset = payload.size();
            payload.insert(payload.end(), prop.bytes.begin(), prop.bytes.end());
            payload.resize(pad8(payload.size()), 0);

            nlohmann::json entry{{"offset", offset},
                                 {"bytes", prop.bytes.size()},
                                 {"property", prop.name},
                                 {"encoding", prop.encoding}};
            if (prop.deflated) {
                entry["compression"] = "gz";
            }
            prop_json.push_back(entry);
        }

        const nlohmann::json chunk_meta{{"version", 1},
                                        {"base", 0},
                                        {"count", count},
                                        {"payloadBytes", payload.size()},
                                        {"properties", prop_json}};
        const std::string chunk_json = chunk_meta.dump();

        std::vector<std::uint8_t> chunk;
        putU32(chunk, kRadChunkMagic);
        putU32(chunk, static_cast<std::uint32_t>(chunk_json.size()));
        putJsonBlock(chunk, chunk_json);
        putU64(chunk, payload.size());
        chunk.insert(chunk.end(), payload.begin(), payload.end());

        nlohmann::json file_meta{{"version", 1},
                                 {"type", "gsplat"},
                                 {"count", count},
                                 {"allChunkBytes", chunk.size()},
                                 {"chunks", nlohmann::json::array({{{"offset", 0},
                                                                    {"bytes", chunk.size()},
                                                                    {"base", 0},
                                                                    {"count", count}}})}};
        if (max_sh > 0) {
            file_meta["maxSh"] = max_sh;
        }
        const std::string meta_json = file_meta.dump();

        std::vector<std::uint8_t> file;
        putU32(file, kRadMagic);
        putU32(file, static_cast<std::uint32_t>(meta_json.size()));
        putJsonBlock(file, meta_json);
        file.insert(file.end(), chunk.begin(), chunk.end());
        return file;
    }

    void writeBytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }

    // Per-test scratch directory, removed when the test ends (also on early ASSERT exits).
    struct ScopedRadDir {
        explicit ScopedRadDir(const std::string& name)
            : path(std::filesystem::temp_directory_path() / ("rad_malformed_" + name)) {
            std::filesystem::remove_all(path);
            std::filesystem::create_directories(path);
        }
        ~ScopedRadDir() { std::filesystem::remove_all(path); }
        std::filesystem::path path;
    };

    // A deflate stream cut in half must make the load fail. Before the fix the
    // inflate loop spins forever on Z_BUF_ERROR, so this test also guards against a hang.
    TEST(RadMalformedInput, TruncatedDeflatePayloadFailsInsteadOfHanging) {
        const ScopedRadDir dir("truncated_deflate");
        constexpr std::uint64_t kCount = 256;

        std::mt19937 rng(7u);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> centers(kCount * 3);
        for (auto& v : centers) {
            v = dist(rng);
        }
        const auto full = rawDeflate(floatBytes(centers));
        ASSERT_GT(full.size(), 16u);
        const std::vector<std::uint8_t> truncated(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(full.size() / 2));

        const auto path = dir.path / "truncated.rad";
        writeBytes(path, buildRadFile(kCount, {{"center", "f32", truncated, true}}));

        const auto loaded = lfs::io::load_rad(path);
        ASSERT_FALSE(loaded.has_value());
        EXPECT_NE(loaded.error().find("decompress"), std::string::npos) << loaded.error();
    }

    // maxSh comes from the file, so an out-of-range value must be clamped to the
    // 4-entry SH table instead of indexing past it.
    TEST(RadMalformedInput, OutOfRangeMaxShIsClampedToDegreeThree) {
        const ScopedRadDir dir("max_sh_clamp");
        constexpr std::uint64_t kCount = 4;
        const std::vector<float> centers(kCount * 3, 0.5f);

        const auto path = dir.path / "max_sh_9.rad";
        writeBytes(path, buildRadFile(kCount, {{"center", "f32", floatBytes(centers), false}}, 9));

        const auto loaded = lfs::io::load_rad(path);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        EXPECT_EQ(loaded->get_max_sh_degree(), 3);
    }

    // A per-component property whose suffix is not 0..2 must be skipped rather than
    // written at a computed index past the end of the means buffer.
    TEST(RadMalformedInput, PerComponentCenterWithInvalidSuffixIsIgnored) {
        const ScopedRadDir dir("center_suffix");
        // Large enough that the bogus component index ('Z' - '0' = 42) also lands inside the means buffer.
        constexpr std::uint64_t kCount = 32;

        // Dimension-major planes, as decode_f32 reads them: plane d holds x, y, z for every splat.
        std::vector<float> planar(kCount * 3);
        for (std::size_t d = 0; d < 3; ++d) {
            for (std::size_t i = 0; i < kCount; ++i) {
                planar[d * kCount + i] = 10.0f * static_cast<float>(d) + static_cast<float>(i) + 0.5f;
            }
        }
        const std::vector<float> rogue(kCount, 1000.0f);

        const auto path = dir.path / "center_z.rad";
        writeBytes(path,
                   buildRadFile(kCount,
                                {{"center", "f32", floatBytes(planar), false},
                                 {"centerZ", "f32", floatBytes(rogue), false}}));

        const auto loaded = lfs::io::load_rad(path);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        const auto means = loaded->get_means().cpu().contiguous().to_vector();
        ASSERT_EQ(means.size(), kCount * 3);
        for (std::size_t i = 0; i < kCount; ++i) {
            for (std::size_t d = 0; d < 3; ++d) {
                EXPECT_FLOAT_EQ(means[i * 3 + d], planar[d * kCount + i]) << "splat " << i << " dim " << d;
            }
        }
    }

} // namespace
