/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/voxelizer.hpp"
#include "collision_fixture.hpp"
#include "core/argument_parser.hpp"
#include "core/cuda/collision_voxelization_kernels.hpp"
#include "core/native_fs.hpp"
#include "io/collision_export.hpp"
#include "licht_test_support.hpp"
#include <array>
#include <bit>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <tuple>

namespace {
    auto parse(std::initializer_list<const char*> args) {
        return lfs::core::args::parse_args(int(args.size()), args.begin());
    }
    TEST(VoxelizeCli, ParsesControlsAndRejectsDeferredOrInvalidOptions) {
        const auto source = PROJECT_ROOT_PATH "/tests/test_voxelize_cli.cpp";
        auto result = parse({"lfs", "voxelize", source, "out.voxel.json", "--auto", "--filter", "none", "--max-gaussian-area", "0"});
        ASSERT_TRUE(result) << result.error();
        const auto& mode = std::get<lfs::core::args::VoxelizeMode>(*result);
        EXPECT_TRUE(mode.settings.auto_resolution);
        EXPECT_FALSE(mode.settings.cleanup);
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", "--filter", "none", "--max-gaussian-area", "1"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", "--filter", "connected"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", "--voxel-size", "nan"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", "--auto", "--voxel-size", ".01"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.bin"}));
    }
    namespace fs = lfs::core::native_fs;
    using collision_test::cells;
    using lfs::core::CollisionVolume;
    using namespace lfs::test::licht;
    struct CollisionExportTest : testing::Test {
        TemporaryDirectory temporary{"voxel_test"};
        const fs::path& directory = temporary.path;
        fs::path path = directory / "scene.voxel.json", binary = directory / "scene.voxel.bin";
        static std::string read(const fs::path& p) {
            const auto bytes = read_file_bytes(fs::native_path(p));
            return bytes.empty() ? std::string{} : std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        static void write(const fs::path& p, std::string_view text) { write_file_bytes(fs::native_path(p), std::as_bytes(std::span(text))); }
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
            auto visit = [&](auto&& self, size_t index, glm::ivec3 origin, int size) -> void {
                if (index >= nodes || size < 4)
                    throw std::runtime_error("Decoder node exceeds topology bounds");
                const auto value = word(index), mask = value >> 24, pointer = value & 0xffffff;
                if (value == 0xff000000 || size == 4) {
                    if (value != 0xff000000 && size_t(pointer) * 2 + 1 >= leaves)
                        throw std::runtime_error("Decoder leaf exceeds leafDataCount");
                    for (const auto p : cells(glm::ivec3(size))) {
                        const auto cell = origin + p;
                        const int bit = p.x + 4 * p.y + 16 * p.z;
                        result[cell.x + edge * (cell.y + edge * cell.z)] = value == 0xff000000 ||
                                                                           ((word(nodes + 2 * pointer + bit / 32) >> (bit % 32)) & 1);
                    }
                } else {
                    size_t child_index = pointer;
                    for (const auto p : cells(glm::ivec3(2)))
                        if (mask & (1u << (p.x + 2 * p.y + 4 * p.z)))
                            self(self, child_index++, origin + p * (size / 2), size / 2);
                }
            };
            if (nodes)
                visit(visit, 0, {}, edge);
            return result;
        }
    };
    TEST_F(CollisionExportTest, InputFailuresPreserveSource) {
        EXPECT_THROW(lfs::io::loadCollisionSource("missing-collision-input.ply"), std::runtime_error);
        write(path, "source");
        lfs::core::args::VoxelizeMode mode;
        mode.input_path = mode.output_path = path;
        mode.overwrite = true;
        EXPECT_EQ(lfs::app::run_voxelizer(mode), 1);
        EXPECT_EQ(read(path), "source");
    }
    TEST(CollisionExportLimits, MixedLeafBoundary) {
        EXPECT_EQ(lfs::core::cuda::collisionOutputBytes(64'000'000), 64'000'000 * sizeof(lfs::core::cuda::CollisionBrick));
        for (const auto count : {size_t(64'000'000), SIZE_MAX})
            EXPECT_THROW(lfs::core::cuda::collisionOutputBytes(count, 1), lfs::Exception);
        EXPECT_NO_THROW(lfs::io::checkCollisionExportMixedBlocks(14'680'064));
        for (uint64_t count : {uint64_t(14'680'065), UINT64_MAX})
            collision_test::expectError([&] { lfs::io::checkCollisionExportMixedBlocks(count); }, "collision.output_limit");
    }
    TEST_F(CollisionExportTest, EverySignedPermutationPreservesCellsAndLogicalBlockBounds) {
        for (const auto [dims, kind] : {std::tuple{glm::ivec3(4), 0}, {glm::ivec3(4), 1}, {glm::ivec3(4), 2}, {glm::ivec3(8), 1}, {glm::ivec3(24, 16, 12), 1}, {glm::ivec3(21, 13, 9), 1}, {glm::ivec3(5, 3, 2), 2}}) {
            auto v = volume(dims);
            const auto padded = ((dims + 3) / 4) * 4;
            if (kind == 0) {
                v.words = {0};
                v.occupied_count = 0;
            } else if (kind == 2) {
                const auto q = (dims - 1) % 4;
                const uint64_t bit = uint64_t(1) << (q.x + 4 * q.y + 16 * q.z);
                // A minimum native brick, or an interior with only its +X child populated.
                v.words = dims.x == 4 ? std::vector<uint32_t>{CollisionVolume::brick_tag | 1, uint32_t(bit), uint32_t(bit >> 32)} : std::vector<uint32_t>{3, 0, CollisionVolume::brick_tag | 9, 0, 0, 0, 0, 0, 0, uint32_t(bit), uint32_t(bit >> 32)};
                v.occupied_count = 1;
            }
            auto occupied = [&](glm::ivec3 p) {
                return glm::all(glm::greaterThanEqual(p, glm::ivec3(0))) && glm::all(glm::lessThan(p, dims)) &&
                       (kind == 1 || (kind == 2 && p == dims - 1));
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
                    if (dims == glm::ivec3(4) && axes == std::array{0, 1, 2} && signs == 0) {
                        const std::array<std::string, 3> expected{"", std::string("\1\0\0\1\0\0\0\xff", 8),
                                                                  std::string("\1\0\0\1\0\0\0\0\0\0\0\0\0\0\0\x80", 16)};
                        EXPECT_EQ(read(binary), expected[kind]);
                    }
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
                        const auto a = glm::vec3(world * glm::vec4(v.origin, 1)) * glm::vec3(-1, -1, 1);
                        const auto corner = v.origin + v.settings.voxel_size * glm::vec3(size);
                        const auto b = glm::vec3(world * glm::vec4(corner, 1)) * glm::vec3(-1, -1, 1);
                        const auto lo = glm::min(a, b), hi = glm::max(a, b);
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
                EXPECT_THROW(lfs::io::exportCollision(volume(), path, transform), lfs::Exception);
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
                if (cancel)
                    EXPECT_THROW(lfs::io::exportCollision(volume(), path, glm::mat4(1), {}, inject), lfs::core::OperationCancelled);
                else
                    collision_test::expectError([&] { lfs::io::exportCollision(volume(), path, glm::mat4(1), {}, inject); }, "collision.export_publish");
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
