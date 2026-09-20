/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "app/converter_output_paths.hpp"
#include "core/path_utils.hpp"

namespace {

    namespace fs = std::filesystem;
    using lfs::core::param::OutputFormat;

    TEST(ConverterOutputPathTest, PreservesUnicodeNamesForAllDerivedOutputs) {
        const auto root = fs::temp_directory_path();
        const auto input = root /
                           lfs::core::utf8_to_path("模型_日本語.ply");

        const auto default_ssog = lfs::app::generate_converter_output_path(
            input, {}, OutputFormat::SSOG, "_converted");
        EXPECT_EQ(lfs::core::path_to_utf8(default_ssog.filename()),
                  "模型_日本語.ssog");

        const auto converted = lfs::app::generate_converter_output_path(
            input, {}, OutputFormat::PLY, "_converted");
        EXPECT_EQ(lfs::core::path_to_utf8(converted.filename()),
                  "模型_日本語_converted.ply");

        const auto batch = lfs::app::generate_ssog_batch_output_path(
            root, lfs::core::utf8_to_path("сцена_кириллица.ply"));
        EXPECT_EQ(lfs::core::path_to_utf8(batch.filename()),
                  "сцена_кириллица.ssog");
    }

} // namespace
