/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "io/atomic_output.hpp"
#include "io/collision_export.hpp"
#include <algorithm>
#include <bit>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <nlohmann/json.hpp>
namespace {
    namespace fs = lfs::core::native_fs;
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
            CollisionVolume v;
            v.settings.voxel_size = .25f;
            v.origin = {2, 4, 6};
            v.dimensions = dims;
            v.edge = 8;
            while (v.edge < std::max({dims.x, dims.y, dims.z}))
                v.edge *= 2;
            v.words = {1};
            v.occupied_count = uint64_t(dims.x) * dims.y * dims.z;
            v.validate();
            return v;
        }
        // Independent wire decoder: no core lookup, export packing or tree helpers.
        std::vector<bool> decode() {
            const auto j = nlohmann::json::parse(read(path));
            EXPECT_EQ(j.at("version"), "1.1");
            EXPECT_EQ(j.at("leafSize"), 4);
            const auto bytes = read(binary);
            const size_t nodes = j.at("nodeCount"), leaves = j.at("leafDataCount");
            EXPECT_EQ(bytes.size(), 4 * (nodes + leaves));
            EXPECT_EQ(leaves, 2 * j.at("numMixedLeaves").get<size_t>());
            auto word = [&](size_t i) {
                if (4 * i + 4 > bytes.size())
                    throw std::runtime_error("Decoder pointer exceeds binary");
                uint32_t value = 0;
                for (int b = 0; b < 4; ++b)
                    value |= uint32_t(uint8_t(bytes[4 * i + b])) << (8 * b);
                return value;
            };
            const int edge = 4 << j.at("treeDepth").get<int>();
            std::vector<bool> result(size_t(edge) * edge * edge);
            for (int z = 0; z < edge; ++z)
                for (int y = 0; y < edge; ++y)
                    for (int x = 0; x < edge; ++x) {
                        size_t index = 0;
                        int px = x, py = y, pz = z;
                        for (int size = edge; nodes && size >= 4; size /= 2) {
                            if (index >= nodes)
                                throw std::runtime_error("Decoder node exceeds nodeCount");
                            const uint32_t value = word(index), mask = value >> 24, pointer = value & 0xffffff;
                            bool solid = value == 0xff000000;
                            if (!solid && size == 4) {
                                const int bit = px + 4 * py + 16 * pz;
                                if (size_t(pointer) * 2 + 1 >= leaves)
                                    throw std::runtime_error("Decoder leaf exceeds leafDataCount");
                                solid = (word(nodes + 2 * pointer + bit / 32) >> (bit % 32)) & 1;
                            } else if (!solid) {
                                const int half = size / 2;
                                const int oct = (px >= half) + 2 * (py >= half) + 4 * (pz >= half);
                                if (!(mask & (1u << oct)))
                                    break;
                                index = pointer + std::popcount(mask & ((1u << oct) - 1));
                                px %= half;
                                py %= half;
                                pz %= half;
                                continue;
                            }
                            result[x + edge * (y + edge * z)] = solid;
                            break;
                        }
                    }
            return result;
        }
    };
    TEST_F(CollisionExportTest, ClippedSolidHasLogicalBoundsAndNoPadding) {
        const auto v = volume();
        lfs::io::exportCollision(v, path);
        const auto decoded = decode();
        for (int z = 0; z < 8; ++z)
            for (int y = 0; y < 8; ++y)
                for (int x = 0; x < 8; ++x)
                    EXPECT_EQ(decoded[x + 8 * (y + 8 * z)], x < 5 && y < 3 && z < 2);
        const auto j = nlohmann::json::parse(read(path));
        EXPECT_EQ(j["gridBounds"]["min"], nlohmann::json::array({-3.25, -4.75, 6}));
        EXPECT_EQ(j["gridBounds"]["max"], nlohmann::json::array({-2, -4, 6.5}));
    }
    TEST_F(CollisionExportTest, ReflectionAndPermutationUseLogicalDimensions) {
        auto v = volume({5, 3, 2});
        v.words.assign(9, 0);
        v.words[0] = 3;
        v.words[2] = CollisionVolume::brick_tag | 9;
        v.words.push_back(1u << (4 + 16));
        v.words.push_back(0);
        v.occupied_count = 1;
        v.validate();
        glm::mat4 transform(0);
        transform[0][1] = -2;
        transform[1][2] = 2;
        transform[2][0] = 2;
        transform[3] = {3, 4, 5, 1};
        lfs::io::exportCollision(v, path, transform);
        const auto decoded = decode();
        EXPECT_EQ(std::count(decoded.begin(), decoded.end(), true), 1);
        EXPECT_TRUE(decoded[0 + 8 * (4 + 8 * 1)]);
        EXPECT_EQ(nlohmann::json::parse(read(path))["voxelResolution"], .5);
    }
    TEST_F(CollisionExportTest, RejectsNonrepresentableTransformsBeforeReplacingFiles) {
        write(path, "old-json");
        write(binary, "old-bin");
        for (int kind = 0; kind < 4; ++kind) {
            glm::mat4 transform(1);
            if (kind == 0)
                transform[0][0] = 2;
            if (kind == 1)
                transform[0][1] = .2f;
            if (kind == 2)
                transform[0][3] = .1f;
            if (kind == 3)
                transform[3][0] = std::numeric_limits<float>::infinity();
            EXPECT_THROW(lfs::io::exportCollision(volume(), path, transform), std::invalid_argument);
            EXPECT_EQ(read(path), "old-json");
            EXPECT_EQ(read(binary), "old-bin");
        }
    }
    TEST_F(CollisionExportTest, SecondInstallFailureAndCancellationRollBackOldAndNewPairs) {
        for (bool previous : {false, true})
            for (bool cancel : {false, true}) {
                if (previous) {
                    write(path, "old-json");
                    write(binary, "old-bin");
                }
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
                EXPECT_EQ(fs::exists(path), previous);
                EXPECT_EQ(fs::exists(binary), previous);
                if (previous) {
                    EXPECT_EQ(read(path), "old-json");
                    EXPECT_EQ(read(binary), "old-bin");
                }
                EXPECT_EQ(std::distance(fs::directory_iterator(directory), std::filesystem::directory_iterator{}), previous ? 2 : 0);
            }
    }
    TEST_F(CollisionExportTest, LongLegalNamesOverwriteAndFailuresNameTheirPath) {
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
    TEST_F(CollisionExportTest, EmptySolidAndMultilevelGridsDecodeIndependently) {
        for (auto dims : {glm::ivec3(1), glm::ivec3(4), glm::ivec3(8), glm::ivec3(9, 5, 3)}) {
            auto v = volume(dims);
            lfs::io::exportCollision(v, path);
            const auto decoded = decode();
            EXPECT_EQ(std::count(decoded.begin(), decoded.end(), true), v.occupied_count);
            v.words = {0};
            v.occupied_count = 0;
            v.validate();
            lfs::io::exportCollision(v, path);
            const auto empty = decode();
            EXPECT_EQ(std::count(empty.begin(), empty.end(), true), 0);
        }
    }
} // namespace
