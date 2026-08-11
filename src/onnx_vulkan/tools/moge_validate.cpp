/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs_onnx_vulkan/onnx_vulkan.hpp"

#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {
    constexpr std::string_view kReferenceMagic = "LFSMOGE1";
    constexpr std::string_view kModelSha256 =
        "bbf14e07a30f11e69d36ab861590123f5598ababcbc8946a063eb4a966f35a21";
    constexpr std::string_view kDepthPngSha256 =
        "2e005d8c3e59386b55b4f27d958bb56e0fb69db3a434fc1404c9388b119e2df0";
    constexpr std::string_view kNormalsPngSha256 =
        "8698ac02dbd11d73438076eea6c477ce0ae7776b0471ee4804dc6a172f7c9e71";

    struct Options {
        fs::path model;
        fs::path image = "../data/360_v2/bicycle/images_8/_DSC8679.JPG";
        fs::path references;
        std::optional<std::uint32_t> device;
    };

    struct Image {
        int width = 0;
        int height = 0;
        std::vector<float> rgb;
    };

    struct Reference {
        std::vector<std::int64_t> shape;
        std::vector<float> values;
    };

    [[nodiscard]] fs::path home_directory() {
#ifdef _WIN32
        if (const char* profile = std::getenv("USERPROFILE"); profile && profile[0])
            return profile;
#else
        if (const char* home = std::getenv("HOME"); home && home[0])
            return home;
#endif
        return {};
    }

    [[nodiscard]] Options parse_options(const int argc, char** argv) {
        Options options;
        const auto home = home_directory();
        options.model = home / ".lichtfeld/onnx/moge-2-vitb-normal.onnx";
        options.references = home / ".lichtfeld/onnx/moge-2-vitb-normal-bicycle-reference";
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument = argv[index];
            const auto value = [&]() -> std::string_view {
                if (++index >= argc)
                    throw std::runtime_error("missing value after " + std::string(argument));
                return argv[index];
            };
            if (argument == "--model")
                options.model = value();
            else if (argument == "--image")
                options.image = value();
            else if (argument == "--references")
                options.references = value();
            else if (argument == "--vulkan-device") {
                const auto text = value();
                std::size_t consumed = 0;
                const auto parsed = std::stoul(std::string(text), &consumed);
                if (consumed != text.size() || parsed > std::numeric_limits<std::uint32_t>::max())
                    throw std::runtime_error("invalid Vulkan device index");
                options.device = static_cast<std::uint32_t>(parsed);
            } else if (argument == "--help" || argument == "-h") {
                std::cout << "usage: lfs_moge_validate [--model model.onnx] [--image fixture.jpg] "
                             "[--references directory] [--vulkan-device index]\n";
                std::exit(0);
            } else {
                throw std::runtime_error("unknown argument: " + std::string(argument));
            }
        }
        return options;
    }

    [[nodiscard]] Image load_image(const fs::path& path) {
        auto input = OIIO::ImageInput::open(path.string());
        if (!input)
            throw std::runtime_error("cannot open image '" + path.string() + "': " + OIIO::geterror());
        const auto spec = input->spec();
        if (spec.width <= 0 || spec.height <= 0 || spec.nchannels <= 0)
            throw std::runtime_error("invalid fixture image shape");
        const auto count = static_cast<std::size_t>(spec.width) * spec.height * spec.nchannels;
        std::vector<float> source(count);
        if (!input->read_image(0, 0, 0, spec.nchannels, OIIO::TypeDesc::FLOAT, source.data())) {
            const auto error = input->geterror();
            input->close();
            throw std::runtime_error("cannot read fixture image: " + error);
        }
        input->close();

        Image result{spec.width, spec.height,
                     std::vector<float>(static_cast<std::size_t>(spec.width) * spec.height * 3)};
        for (int y = 0; y < spec.height; ++y) {
            for (int x = 0; x < spec.width; ++x) {
                const auto source_index = (static_cast<std::size_t>(y) * spec.width + x) * spec.nchannels;
                const auto output_index = (static_cast<std::size_t>(y) * spec.width + x) * 3;
                result.rgb[output_index] = source[source_index];
                result.rgb[output_index + 1] = spec.nchannels > 1 ? source[source_index + 1] : source[source_index];
                result.rgb[output_index + 2] = spec.nchannels > 2 ? source[source_index + 2] : source[source_index];
            }
        }
        return result;
    }

    [[nodiscard]] int patch_extent(const int value) {
        constexpr int patch = 14;
        return std::max(patch, (value + patch / 2) / patch * patch);
    }

    [[nodiscard]] Image resize_fixture(const Image& image) {
        constexpr int max_side = 518;
        const float scale = static_cast<float>(max_side) / std::max(image.width, image.height);
        const int width = patch_extent(static_cast<int>(std::round(image.width * scale)));
        const int height = patch_extent(static_cast<int>(std::round(image.height * scale)));
        if (width == image.width && height == image.height)
            return image;
        Image result{width, height, std::vector<float>(static_cast<std::size_t>(width) * height * 3)};
        OIIO::ImageBuf source(OIIO::ImageSpec(image.width, image.height, 3, OIIO::TypeDesc::FLOAT),
                              const_cast<float*>(image.rgb.data()));
        OIIO::ImageBuf destination(OIIO::ImageSpec(width, height, 3, OIIO::TypeDesc::FLOAT),
                                   result.rgb.data());
        const OIIO::ROI roi(0, width, 0, height, 0, 1, 0, 3);
        if (!OIIO::ImageBufAlgo::resample(destination, source, true, roi, 0))
            throw std::runtime_error("fixture resize failed: " + destination.geterror());
        return result;
    }

    [[nodiscard]] std::vector<float> to_nchw(const Image& image) {
        const auto plane = static_cast<std::size_t>(image.width) * image.height;
        std::vector<float> result(plane * 3);
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            result[pixel] = image.rgb[pixel * 3];
            result[plane + pixel] = image.rgb[pixel * 3 + 1];
            result[plane * 2 + pixel] = image.rgb[pixel * 3 + 2];
        }
        return result;
    }

    template <typename T>
    [[nodiscard]] T read_scalar(std::ifstream& stream, const std::string_view label) {
        T value{};
        stream.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!stream)
            throw std::runtime_error("truncated reference " + std::string(label));
        return value;
    }

    [[nodiscard]] Reference load_reference(const fs::path& path) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream)
            throw std::runtime_error("cannot open raw reference '" + path.string() + "'");
        std::array<char, 8> magic{};
        stream.read(magic.data(), magic.size());
        if (!stream || std::string_view(magic.data(), magic.size()) != kReferenceMagic)
            throw std::runtime_error("invalid raw reference magic in '" + path.string() + "'");
        const auto rank = read_scalar<std::uint32_t>(stream, "rank");
        if (rank > 16)
            throw std::runtime_error("raw reference rank exceeds 16");
        Reference result;
        result.shape.resize(rank);
        stream.read(reinterpret_cast<char*>(result.shape.data()),
                    static_cast<std::streamsize>(rank * sizeof(std::int64_t)));
        const auto byte_count = read_scalar<std::uint64_t>(stream, "byte count");
        if (byte_count % sizeof(float) != 0 || byte_count > 256ull * 1024ull * 1024ull)
            throw std::runtime_error("invalid raw reference byte count");
        result.values.resize(static_cast<std::size_t>(byte_count / sizeof(float)));
        stream.read(reinterpret_cast<char*>(result.values.data()), static_cast<std::streamsize>(byte_count));
        if (!stream || stream.peek() != std::ifstream::traits_type::eof())
            throw std::runtime_error("truncated or trailing raw reference data");
        return result;
    }

    [[nodiscard]] bool compare_tensor(const lfs::onnx_vulkan::Tensor& actual,
                                      const Reference& reference,
                                      const std::string_view name) {
        const auto values = actual.data_as<float>();
        if (!std::ranges::equal(actual.shape(), reference.shape) || values.size() != reference.values.size()) {
            std::cerr << name << ": output shape does not match the reference\n";
            return false;
        }
        std::size_t locations_match = 0;
        std::size_t tolerance_match = 0;
        std::size_t classifications_match = 0;
        float maximum_error = 0.0f;
        for (std::size_t index = 0; index < values.size(); ++index) {
            const float actual_value = values[index];
            const float expected_value = reference.values[index];
            const bool same_location = std::isfinite(actual_value) == std::isfinite(expected_value) &&
                                       std::isnan(actual_value) == std::isnan(expected_value) &&
                                       std::isinf(actual_value) == std::isinf(expected_value);
            locations_match += same_location;
            if (std::isfinite(actual_value) && std::isfinite(expected_value)) {
                const float error = std::abs(actual_value - expected_value);
                maximum_error = std::max(maximum_error, error);
                tolerance_match += error <= 1e-3f + 5e-3f * std::abs(expected_value);
            } else {
                tolerance_match += same_location;
            }
            if (name == "mask")
                classifications_match += (actual_value >= 0.5f) == (expected_value >= 0.5f);
        }
        const double ratio = values.empty() ? 1.0 : static_cast<double>(tolerance_match) / values.size();
        const double classification = values.empty() ? 1.0 :
            static_cast<double>(classifications_match) / values.size();
        std::cout << name << ": tolerance=" << ratio * 100.0 << "% max_abs=" << maximum_error;
        if (name == "mask")
            std::cout << " classification=" << classification * 100.0 << '%';
        std::cout << '\n';
        return locations_match == values.size() && ratio >= 0.999 && maximum_error < 5e-2f &&
               (name != "mask" || classification >= 0.999);
    }
}

int main(const int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (!fs::is_regular_file(options.model))
            throw std::runtime_error("model not found: " + options.model.string());
        if (!fs::is_regular_file(options.image))
            throw std::runtime_error("fixture image not found: " + options.image.string());

        const auto image = resize_fixture(load_image(options.image));
        auto input_values = to_nchw(image);
        const std::array<std::int64_t, 4> image_shape{1, 3, image.height, image.width};
        constexpr std::int64_t num_tokens = 1800;
        const std::array<lfs::onnx_vulkan::NamedTensorView, 2> inputs{{
            {"image", {lfs::onnx_vulkan::ElementType::Float32, image_shape,
                       std::as_bytes(std::span(input_values))}},
            {"num_tokens", {lfs::onnx_vulkan::ElementType::Int64, {},
                            std::as_bytes(std::span(&num_tokens, 1))}},
        }};
        lfs::onnx_vulkan::SessionOptions session_options;
        session_options.vulkan_device = options.device;
        auto session = lfs::onnx_vulkan::VulkanSession::create(options.model, session_options);
        if (!session)
            throw std::runtime_error(session.error().message);
        constexpr std::array<std::string_view, 4> requested{"points", "normal", "mask", "metric_scale"};
        auto outputs = session->run(inputs, requested);
        if (!outputs)
            throw std::runtime_error(outputs.error().message);

        bool passed = true;
        for (const auto& output : *outputs) {
            const auto reference = load_reference(options.references / (output.name + ".lfsref"));
            passed &= compare_tensor(output.tensor, reference, output.name);
        }
        std::cout << "device: " << session->device_name() << '\n'
                  << "reference provenance: model=" << kModelSha256 << '\n'
                  << "depth_png=" << kDepthPngSha256 << '\n'
                  << "normals_png=" << kNormalsPngSha256 << '\n';
        if (!passed) {
            std::cerr << "MoGe full-model validation failed\n";
            return 1;
        }
        std::cout << "MoGe full-model validation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "lfs_moge_validate: " << error.what() << '\n';
        return 1;
    }
}
