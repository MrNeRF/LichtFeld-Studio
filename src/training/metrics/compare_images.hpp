/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "metrics.hpp"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace lfs::training {

    // An image, a folder of images, or a COLMAP dataset folder whose sparse model holds the images' cameras.
    struct ImageSet {
        std::filesystem::path images;
        std::filesystem::path sparse; // Empty when the set has no cameras
        bool recursive = false;       // A dataset's images may sit in sub-folders, as its camera names say
    };

    [[nodiscard]] ImageSet resolve_image_set(const std::filesystem::path& path);

    struct ImagePair {
        std::filesystem::path reference;
        std::filesystem::path test;
        std::string name;          // Reference path relative to its folder, '/'-separated
        std::string reference_key; // Relative paths without extension, lower case, as COLMAP image names give them
        std::string test_key;
    };

    struct ImagePairing {
        std::vector<ImagePair> pairs;
        std::vector<std::filesystem::path> unmatched;
    };

    // Two files form one pair; two folders pair their images by case-insensitive relative path without extension,
    // including sub-folders when recursive.
    [[nodiscard]] ImagePairing pair_images_by_name(const std::filesystem::path& reference,
                                                   const std::filesystem::path& test, bool recursive = false);

    // How the camera of a test image relates to the camera of its reference image.
    enum class TestLens {
        Reference, // The same lens, possibly at another resolution
        Pinhole,   // A pinhole view of a reference with lens distortion: the test was rendered undistorted
        Other,     // Neither; compare cannot map one image onto the other
    };

    [[nodiscard]] TestLens classify_test_lens(const lfs::core::Camera& reference, const lfs::core::Camera& test);

    struct CompareResult {
        EvalMetrics metrics;
        std::vector<std::filesystem::path> unmatched; // Images without a counterpart or without a camera
        std::string lens;                             // How the test images were mapped onto their references
    };

    // Scores every test image against its reference with the evaluation pipeline and writes compare.json and
    // compare_report.txt to params.output_path.
    [[nodiscard]] lfs::Result<CompareResult> compare_image_sets(
        const lfs::core::param::CompareParameters& params, std::optional<std::filesystem::path> lpips_weights);

    // The inputs, the settings, the mean scores and each image's scores, as written to compare.json.
    [[nodiscard]] nlohmann::ordered_json compare_json(const lfs::core::param::CompareParameters& params,
                                                      const CompareResult& result);

    // The inputs, the settings and one line per image with its scores, as printed by the compare command.
    [[nodiscard]] std::string format_compare_report(const lfs::core::param::CompareParameters& params,
                                                    const CompareResult& result);

} // namespace lfs::training
