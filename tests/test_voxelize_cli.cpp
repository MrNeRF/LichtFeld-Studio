/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/voxelizer.hpp"
#include "collision_fixture.hpp"
#include "io/argument_parser.hpp"
#include "io/atomic_output.hpp"
#include "io/collision_export.hpp"
#include <array>
#include <bit>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <tuple>

namespace {
    auto parse(std::initializer_list<const char*> args) {
        return lfs::io::args::parse_args(int(args.size()), args.begin());
    }
    TEST(VoxelizeCli, ParsesControlsAndRejectsDeferredOrInvalidOptions) {
        const auto source = PROJECT_ROOT_PATH "/tests/test_voxelize_cli.cpp";
        auto result = parse({"lfs", "voxelize", source, "out.voxel.json", "--auto", "--filter", "none", "--max-gaussian-area", "0"});
        ASSERT_TRUE(result) << result.error();
        const auto& mode = std::get<lfs::io::args::VoxelizeMode>(*result);
        EXPECT_TRUE(mode.settings.auto_resolution);
        EXPECT_FALSE(mode.settings.cleanup);
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", "--filter", "none", "--max-gaussian-area", "1"}));
        for (const auto* flag : {"--seed-pos", "--cluster-resolution", "--fill", "--carve", "--surface"})
            EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", flag, "1"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", "--filter", "connected"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", "--voxel-size", "nan"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", "--auto", "--voxel-size", ".01"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.bin"}));
    }
    TEST(VoxelizeCli, InputFailuresPreserveSource) {
        EXPECT_THROW(lfs::io::loadCollisionSource("missing-collision-input.ply"), std::runtime_error);
        namespace fs = lfs::core::native_fs;
        const auto path = lfs::io::make_atomic_temp_output_path(fs::temp_directory_path() / "voxel_source");
        {
            std::ofstream out(fs::native_path(path));
            out << "source";
        }
        lfs::io::args::VoxelizeMode mode;
        mode.input_path = mode.output_path = path;
        mode.overwrite = true;
        EXPECT_EQ(lfs::app::run_voxelizer(mode), 1);
        std::ifstream in(fs::native_path(path));
        std::string text;
        in >> text;
        EXPECT_EQ(text, "source");
        in.close();
        fs::remove(path);
    }
    namespace fs = lfs::core::native_fs;
    using collision_test::cells;
    using lfs::core::CollisionVolume;
    struct CollisionExportTest : testing::Test {
        fs::path directory = lfs::io::make_atomic_temp_output_path(fs::temp_directory_path() / "voxel_test");
        fs::path path = directory / "scene.voxel.json", binary = directory / "scene.voxel.bin";
        void SetUp() override { fs::create_directories(directory); }
        void TearDown() override { fs::remove_all(directory); }
        static std::string read(const fs::path& p) {
            std::ifstream in(fs::native_path(p), std::ios::binary);
            return {std::istreambuf_iterator<char>(in), {}};
        }
        static void write(const fs::path& p, const std::string& text) {
            std::ofstream out(fs::native_path(p), std::ios::binary);
            out << text;
        }
        static CollisionVolume volume(glm::ivec3 dims = {5, 3, 2}) {
            return *collision_test::box({2, 4, 6}, dims, .25f);
        }
        // Independent little-endian v1.1 decoder, with bounds checks on both pointer spaces.
        std::vector<bool> decode() {
            const auto j = nlohmann::json::parse(read(path));
            EXPECT_EQ(j.at("version"), "1.1");
            EXPECT_EQ(j.at("leafSize"), 4);
            const auto bytes = read(binary);
            const size_t nodes = j.at("nodeCount"), leaves = j.at("leafDataCount");
            EXPECT_EQ(bytes.size(), 4 * (nodes + leaves));
            EXPECT_EQ(leaves, 2 * j.at("numMixedLeaves").get<size_t>());
            auto word = [&](size_t i) {
                if (i >= bytes.size() / 4)
                    throw std::runtime_error("Decoder pointer exceeds binary");
                uint32_t value = 0;
                for (int b = 0; b < 4; ++b)
                    value |= uint32_t(uint8_t(bytes[4 * i + b])) << (8 * b);
                return value;
            };
            const int edge = 4 << j.at("treeDepth").get<int>();
            std::vector<bool> result(size_t(edge) * edge * edge);
            auto visit = [&](auto&& self, size_t index, glm::ivec3 p, int size) -> void {
                if (index >= nodes)
                    throw std::runtime_error("Decoder node exceeds nodeCount");
                const uint32_t value = word(index), mask = value >> 24, pointer = value & 0xffffff;
                if (value == 0xff000000 || size == 4) {
                    if (value != 0xff000000 && size_t(pointer) * 2 + 1 >= leaves)
                        throw std::runtime_error("Decoder leaf exceeds leafDataCount");
                    for (const auto q : cells(glm::ivec3(size))) {
                        const int bit = q.x + 4 * q.y + 16 * q.z;
                        const auto cell = p + q;
                        result[cell.x + edge * (cell.y + edge * cell.z)] =
                            value == 0xff000000 || ((word(nodes + 2 * pointer + bit / 32) >> (bit % 32)) & 1);
                    }
                } else {
                    for (int c = 0; c < 8; ++c)
                        if (mask & (1u << c))
                            self(self, pointer + std::popcount(mask & ((1u << c) - 1)),
                                 p + glm::ivec3(c & 1, (c >> 1) & 1, c >> 2) * (size / 2), size / 2);
                }
            };
            if (nodes)
                visit(visit, 0, {}, edge);
            return result;
        }
    };
    TEST(CollisionExportLimits, MixedLeafBoundary) {
        constexpr uint64_t leaves = 7 * (((uint64_t(1) << 24) - 1) / 8) + 1;
        for (uint64_t count : {uint64_t(0), leaves, uint64_t(14'680'064)})
            EXPECT_NO_THROW(lfs::io::checkCollisionExportMixedBlocks(count));
        for (uint64_t count : {leaves + 8, uint64_t(14'680'065), UINT64_MAX})
            EXPECT_THROW(lfs::io::checkCollisionExportMixedBlocks(count), std::runtime_error);
    }
    TEST_F(CollisionExportTest, MinimumRootKnownLittleEndianBytes) {
        const auto transform = glm::mat4(glm::vec4(-1, 0, 0, 0), glm::vec4(0, -1, 0, 0),
                                         glm::vec4(0, 0, 1, 0), glm::vec4(0, 0, 0, 1));
        const std::array<std::string, 3> bytes{"", std::string("\1\0\0\1\0\0\0\xff", 8),
                                               std::string("\1\0\0\1\0\0\0\0\1\0\0\0\0\0\0\0", 16)};
        for (int kind = 0; kind < 3; ++kind) {
            auto v = volume(glm::ivec3(4));
            v.words = kind == 2 ? std::vector<uint32_t>{CollisionVolume::brick_tag | 1, 1, 0} : std::vector<uint32_t>{uint32_t(kind)};
            v.occupied_count = kind == 0 ? 0 : kind == 1 ? 64
                                                         : 1;
            v.validate();
            lfs::io::exportCollision(v, path, transform);
            EXPECT_EQ(read(binary), bytes[kind]);
            const auto decoded = decode();
            EXPECT_EQ(std::count(decoded.begin(), decoded.end(), true), v.occupied_count);
        }
    }
    TEST_F(CollisionExportTest, EverySignedPermutationPreservesCellsAndLogicalBlockBounds) {
        for (const auto dims : {glm::ivec3(4), glm::ivec3(8), glm::ivec3(24, 16, 12), glm::ivec3(21, 13, 9), glm::ivec3(5, 3, 2)}) {
            auto v = volume(dims);
            const auto padded = ((dims + 3) / 4) * 4;
            const bool sparse = dims == glm::ivec3(4) || dims == glm::ivec3(5, 3, 2);
            if (sparse) {
                const auto q = (dims - 1) % 4;
                const uint64_t bit = uint64_t(1) << (q.x + 4 * q.y + 16 * q.z);
                // A minimum native brick, or an interior with only its +X child populated.
                v.words = dims.x == 4 ? std::vector<uint32_t>{CollisionVolume::brick_tag | 1, uint32_t(bit), uint32_t(bit >> 32)} : std::vector<uint32_t>{3, 0, CollisionVolume::brick_tag | 9, 0, 0, 0, 0, 0, 0, uint32_t(bit), uint32_t(bit >> 32)};
                v.occupied_count = 1;
            }
            auto occupied = [&](glm::ivec3 p) {
                return glm::all(glm::greaterThanEqual(p, glm::ivec3(0))) && glm::all(glm::lessThan(p, dims)) &&
                       (!sparse || p == dims - 1);
            };
            v.validate();
            std::array axes{0, 1, 2};
            do {
                for (int signs = 0; signs < 8; ++signs) {
                    SCOPED_TRACE(::testing::Message() << dims.x << ":" << axes[0] << axes[1] << axes[2] << ":" << signs);
                    glm::mat4 world(0);
                    world[3] = {3, 4, 5, 1};
                    for (int s = 0; s < 3; ++s)
                        world[s][axes[s]] = 2.f * ((signs & (1 << s)) ? -1 : 1) * (axes[s] < 2 ? -1 : 1);
                    lfs::io::exportCollision(v, path, world);
                    const auto decoded = decode();
                    const int edge = std::max(8, v.edge);
                    ASSERT_EQ(decoded.size(), size_t(edge) * edge * edge);
                    for (const auto p : cells(glm::ivec3(edge))) {
                        glm::ivec3 source;
                        for (int s = 0; s < 3; ++s)
                            source[s] = signs & (1 << s) ? padded[s] - 1 - p[axes[s]] : p[axes[s]];
                        ASSERT_EQ(decoded[p.x + edge * (p.y + edge * p.z)], occupied(source));
                    }
                    auto bounds = [&](glm::ivec3 size) {
                        glm::vec3 lo(INFINITY), hi(-INFINITY);
                        for (int c = 0; c < 8; ++c) {
                            const auto corner = v.origin + v.settings.voxel_size * glm::vec3(glm::ivec3(c & 1, (c >> 1) & 1, c >> 2) * size);
                            auto p = glm::vec3(world * glm::vec4(corner, 1));
                            p *= glm::vec3(-1, -1, 1);
                            lo = glm::min(lo, p);
                            hi = glm::max(hi, p);
                        }
                        return nlohmann::json{{"min", {lo.x, lo.y, lo.z}}, {"max", {hi.x, hi.y, hi.z}}};
                    };
                    const auto j = nlohmann::json::parse(read(path));
                    EXPECT_EQ(j["gridBounds"], bounds(padded));
                    EXPECT_EQ(j["sceneBounds"], bounds(dims));
                    EXPECT_EQ(j["voxelResolution"], .5);
                }
            } while (std::next_permutation(axes.begin(), axes.end()));
        }
    }
    TEST_F(CollisionExportTest, InvalidTransformAndPairInstallFailuresPreserveDestinations) {
        for (bool previous : {false, true}) {
            if (previous) {
                write(path, "old-json");
                write(binary, "old-bin");
            }
            auto preserved = [&] {
                EXPECT_EQ(fs::exists(path), previous);
                EXPECT_EQ(fs::exists(binary), previous);
                if (previous) {
                    EXPECT_EQ(read(path), "old-json");
                    EXPECT_EQ(read(binary), "old-bin");
                }
                EXPECT_EQ(std::distance(fs::directory_iterator(directory), std::filesystem::directory_iterator{}), previous ? 2 : 0);
            };
            for (const auto [column, row, value] : {std::tuple{0, 0, 2.f}, {0, 1, .2f}, {0, 3, .1f}, {3, 0, INFINITY}}) {
                glm::mat4 transform(1);
                transform[column][row] = value;
                EXPECT_THROW(lfs::io::exportCollision(volume(), path, transform), std::invalid_argument);
                preserved();
            }
            for (bool cancel : {false, true}) {
                auto inject = [&](float p, const std::string&) {
                    if (p != .95f)
                        return true;
                    if (cancel)
                        return false;
                    for (const auto& entry : fs::directory_iterator(directory))
                        if (entry.path() != path && read(entry.path()).starts_with('{'))
                            fs::remove(entry.path());
                    return true;
                };
                EXPECT_THROW(lfs::io::exportCollision(volume(), path, glm::mat4(1), {}, inject), std::runtime_error);
                preserved();
            }
        }
    }
    TEST_F(CollisionExportTest, LegalLongNamesOverwriteAndErrorsNameTheBlockedPath) {
        path = directory / (std::string(240, 'a') + ".voxel.json");
        lfs::io::exportCollision(volume(), path);
        lfs::io::exportCollision(volume(), path);
        const auto blocked = directory / "file";
        write(blocked, "source-preserved");
        try {
            lfs::io::exportCollision(volume(), blocked / "out.voxel.json");
            FAIL();
        } catch (const std::exception& e) {
            EXPECT_NE(std::string(e.what()).find("file"), std::string::npos);
        }
        EXPECT_EQ(read(blocked), "source-preserved");
    }
} // namespace
