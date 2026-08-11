/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/argument_parser.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

namespace fs = std::filesystem;

namespace {
    struct Failure final : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    void require(const bool condition, const std::string_view message) {
        if (!condition)
            throw Failure(std::string(message));
    }

    class TemporaryDirectory {
    public:
        TemporaryDirectory() {
            const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            path_ = fs::temp_directory_path() / ("lfs-preprocess-cli-tests-" + std::to_string(stamp));
            fs::create_directories(path_);
        }

        ~TemporaryDirectory() {
            std::error_code ignored;
            fs::remove_all(path_, ignored);
        }

        [[nodiscard]] const fs::path& path() const noexcept { return path_; }

    private:
        fs::path path_;
    };

    void test_full_vulkan_options(const TemporaryDirectory& temporary) {
        const auto model = temporary.path() / "model.onnx";
        std::ofstream(model, std::ios::binary).put('\0');
        const auto dataset_text = temporary.path().string();
        const auto model_text = model.string();
        const char* argv[] = {
            "LichtFeld-Studio", "preprocess", dataset_text.c_str(),
            "--mode", "normals", "--images", "images_8", "--model", model_text.c_str(),
            "--max-side", "392", "--num-tokens", "2048", "--vulkan-device", "2",
            "--png-compression", "4", "--bit-depth", "8", "--overwrite", "--no-download",
        };

        auto parsed = lfs::core::args::parse_args(static_cast<int>(std::size(argv)), argv);
        require(parsed.has_value(), parsed ? "" : parsed.error());
        const auto* mode = std::get_if<lfs::core::args::PreprocessMode>(&*parsed);
        require(mode != nullptr, "preprocess subcommand selected the wrong CLI mode");
        require(mode->params.dataset_path == temporary.path(), "dataset path was not preserved");
        require(mode->params.images_folder == "images_8", "images folder was not parsed");
        require(mode->params.model_path == model, "model path was not parsed");
        require(mode->params.mode == lfs::core::param::PreprocessOutputMode::Normals,
                "output mode was not parsed");
        require(mode->params.max_side == 392 && mode->params.num_tokens == 2048,
                "inference sizing options were not parsed");
        require(mode->params.vulkan_device == 2u, "Vulkan device was not parsed");
        require(mode->params.png_compression == 4 && mode->params.bit_depth == 8,
                "PNG options were not parsed");
        require(mode->params.overwrite && mode->params.no_download,
                "preprocess flags were not parsed");
    }

    void test_download_only() {
        const char* argv[] = {"LichtFeld-Studio", "preprocess", "--download-only"};
        auto parsed = lfs::core::args::parse_args(static_cast<int>(std::size(argv)), argv);
        require(parsed.has_value(), parsed ? "" : parsed.error());
        const auto* mode = std::get_if<lfs::core::args::PreprocessMode>(&*parsed);
        require(mode != nullptr && mode->params.download_only,
                "download-only did not select preprocess mode");
        require(mode->params.dataset_path.empty(), "download-only unexpectedly required a dataset");
    }

    void test_invalid_device(const TemporaryDirectory& temporary) {
        const auto dataset_text = temporary.path().string();
        const char* argv[] = {
            "LichtFeld-Studio", "preprocess", dataset_text.c_str(), "--vulkan-device", "-1"};
        const auto parsed = lfs::core::args::parse_args(static_cast<int>(std::size(argv)), argv);
        require(!parsed.has_value() && parsed.error().contains("--vulkan-device must be 0 or greater"),
                "negative Vulkan device index was accepted");
    }
}

int main() {
    TemporaryDirectory temporary;
    try {
        test_full_vulkan_options(temporary);
        std::cout << "PASS preprocess_vulkan_options\n";
        test_download_only();
        std::cout << "PASS preprocess_download_only\n";
        test_invalid_device(temporary);
        std::cout << "PASS preprocess_invalid_device\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL preprocess_cli: " << error.what() << '\n';
        return 1;
    }
}
