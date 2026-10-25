/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/voxelizer.hpp"
#include "core/argument_parser.hpp"
#include "io/atomic_output.hpp"
#include "io/collision_export.hpp"
#include <fstream>
#include <gtest/gtest.h>
namespace {
    auto parse(std::initializer_list<const char*> args) {
        return lfs::core::args::parse_args(int(args.size()), args.begin());
    }
    TEST(VoxelizeCli, ParsesControlsAndRejectsDeferredOrInvalidOptions) {
        const auto source = PROJECT_ROOT_PATH "/tests/test_voxelize_cli.cpp";
        auto result = parse({"lfs", "voxelize", source, "out.voxel.json", "--auto", "--filter", "none"});
        ASSERT_TRUE(result) << result.error();
        const auto& mode = std::get<lfs::core::args::VoxelizeMode>(*result);
        EXPECT_TRUE(mode.settings.auto_resolution);
        EXPECT_FALSE(mode.settings.cleanup);
        for (const auto* flag : {"--seed-pos", "--cluster-resolution", "--fill", "--carve", "--surface"})
            EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", flag, "1"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", "--filter", "connected"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", "--voxel-size", "nan"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.voxel.json", "--auto", "--voxel-size", ".01"}));
        EXPECT_FALSE(parse({"lfs", "voxelize", source, "out.bin"}));
    }
    TEST(VoxelizeCli, LoaderFailureIsNotSuccessfulEmptyOccupancy) {
        EXPECT_THROW(lfs::io::loadCollisionSource("missing-collision-input.ply"), std::runtime_error);
    }
    TEST(VoxelizeCli, OutputCannotDestroyItsInput) {
        namespace fs = lfs::core::native_fs;
        const auto path = lfs::io::make_atomic_temp_output_path(fs::temp_directory_path() / "voxel_source");
        {
            std::ofstream out(fs::native_path(path));
            out << "source";
        }
        lfs::core::args::VoxelizeMode mode;
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
} // namespace
