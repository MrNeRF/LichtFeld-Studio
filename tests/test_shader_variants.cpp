/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/rasterizer/vulkan/src/shader_variants.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <string>

TEST(ShaderVariants, RegisteredFilesExist) {
    namespace fs = std::filesystem;
    using namespace lfs::rendering;
    const fs::path generated = fs::path(LFS_PROGRAM_SET_TEST_SPIRV_DIR) / "generated";
    const fs::path hosts = LFS_SHADER_PACKAGE_DIR;
    for (const auto& row : shader_variants) {
        for (const auto name : {row.program, row.banded, row.reveal}) {
            if (!name.empty())
                EXPECT_TRUE(fs::is_regular_file(generated / (std::string(name) + ".spv"))) << name;
        }
        if (row.custom_kind == CustomKind::Spirv)
            EXPECT_NE(row.chain, CustomChain::None) << row.program;
        for (const auto name : {row.custom, row.custom_banded}) {
            if (name == row.custom_banded && row.custom_kind == CustomKind::Host && row.chain == CustomChain::None)
                continue;
            if (!name.empty()) {
                const auto& dir = row.custom_kind == CustomKind::Host ? hosts : generated;
                const auto extension = row.custom_kind == CustomKind::Host ? ".slang-module" : ".spv";
                EXPECT_TRUE(fs::is_regular_file(dir / (std::string(name) + extension))) << name;
            }
        }
    }
}
