/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <OpenImageIO/imageio.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {
    constexpr double kPassMae = 1.0e-4;
    constexpr double kHardFailMae = 2.0e-4;
    constexpr double kScale = 65535.0;

    struct Image {
        int width = 0;
        int height = 0;
        int channels = 0;
        std::vector<std::uint16_t> pixels;
    };

    struct WorstPixel {
        std::uint16_t delta = 0;
        fs::path relative_path;
        int x = 0;
        int y = 0;
        int channel = 0;
        std::uint16_t candidate = 0;
        std::uint16_t reference = 0;
    };

    struct Statistics {
        std::uint64_t values = 0;
        std::uint64_t images = 0;
        std::uint64_t above_1e4 = 0;
        long double sum_absolute = 0.0;
        long double sum_squared = 0.0;
        std::vector<std::uint64_t> histogram = std::vector<std::uint64_t>(65536);
        WorstPixel worst;
        fs::path worst_image;
        double worst_image_mae = 0.0;
    };

    [[nodiscard]] std::string path_string(const fs::path& path) {
#ifdef _WIN32
        const auto text = path.u8string();
        return {reinterpret_cast<const char*>(text.data()), text.size()};
#else
        return path.string();
#endif
    }

    [[nodiscard]] std::map<fs::path, fs::path> collect_pngs(const fs::path& root) {
        if (!fs::is_directory(root))
            throw std::runtime_error("Not a directory: " + path_string(root));
        std::map<fs::path, fs::path> files;
        for (const auto& entry : fs::recursive_directory_iterator(root)) {
            if (!entry.is_regular_file()) continue;
            auto extension = entry.path().extension().string();
            std::ranges::transform(extension, extension.begin(), [](const unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (extension == ".png") files.emplace(entry.path().lexically_relative(root), entry.path());
        }
        if (files.empty())
            throw std::runtime_error("No PNG files under " + path_string(root));
        return files;
    }

    [[nodiscard]] Image read_u16(const fs::path& path, const int expected_channels) {
        auto input = OIIO::ImageInput::open(path_string(path));
        if (!input)
            throw std::runtime_error("Failed to open " + path_string(path) + ": " + OIIO::geterror());
        const auto spec = input->spec();
        if (spec.width <= 0 || spec.height <= 0 || spec.nchannels != expected_channels) {
            input->close();
            throw std::runtime_error(std::format("Unexpected shape/channels in {}: {}x{}x{} (expected {} channels)",
                                                path_string(path), spec.width, spec.height, spec.nchannels,
                                                expected_channels));
        }
        if (spec.format != OIIO::TypeDesc::UINT16) {
            input->close();
            throw std::runtime_error("Expected uint16 PNG: " + path_string(path));
        }
        Image image{spec.width, spec.height, spec.nchannels};
        image.pixels.resize(static_cast<std::size_t>(spec.width) * spec.height * spec.nchannels);
        if (!input->read_image(0, 0, 0, spec.nchannels, OIIO::TypeDesc::UINT16, image.pixels.data())) {
            const auto error = input->geterror();
            input->close();
            throw std::runtime_error("Failed to read " + path_string(path) + ": " + error);
        }
        input->close();
        return image;
    }

    [[nodiscard]] std::uint16_t percentile(const Statistics& stats, const double quantile) {
        if (stats.values == 0) return 0;
        const auto rank = static_cast<std::uint64_t>(std::ceil(quantile * static_cast<double>(stats.values)));
        std::uint64_t cumulative = 0;
        for (std::size_t delta = 0; delta < stats.histogram.size(); ++delta) {
            cumulative += stats.histogram[delta];
            if (cumulative >= rank) return static_cast<std::uint16_t>(delta);
        }
        return 65535;
    }

    [[nodiscard]] Statistics compare_directories(const fs::path& candidate_root,
                                                  const fs::path& reference_root,
                                                  const int channels) {
        const auto candidate_files = collect_pngs(candidate_root);
        const auto reference_files = collect_pngs(reference_root);
        if (candidate_files.size() != reference_files.size())
            throw std::runtime_error(std::format("File count differs: candidate={} reference={}",
                                                candidate_files.size(), reference_files.size()));
        Statistics stats;
        for (const auto& [relative, candidate_path] : candidate_files) {
            const auto reference_it = reference_files.find(relative);
            if (reference_it == reference_files.end())
                throw std::runtime_error("Reference is missing " + path_string(relative));
            const auto candidate = read_u16(candidate_path, channels);
            const auto reference = read_u16(reference_it->second, channels);
            if (candidate.width != reference.width || candidate.height != reference.height)
                throw std::runtime_error("Image dimensions differ for " + path_string(relative));

            long double image_absolute = 0.0;
            for (std::size_t index = 0; index < candidate.pixels.size(); ++index) {
                const auto lhs = candidate.pixels[index];
                const auto rhs = reference.pixels[index];
                const auto delta = static_cast<std::uint16_t>(lhs > rhs ? lhs - rhs : rhs - lhs);
                ++stats.histogram[delta];
                stats.sum_absolute += delta;
                image_absolute += delta;
                stats.sum_squared += static_cast<long double>(delta) * delta;
                stats.above_1e4 += static_cast<double>(delta) / kScale > kPassMae;
                if (delta > stats.worst.delta) {
                    const auto pixel = index / channels;
                    stats.worst = {delta, relative,
                                   static_cast<int>(pixel % candidate.width),
                                   static_cast<int>(pixel / candidate.width),
                                   static_cast<int>(index % channels), lhs, rhs};
                }
            }
            const auto image_mae = static_cast<double>(image_absolute / candidate.pixels.size()) / kScale;
            if (image_mae > stats.worst_image_mae) {
                stats.worst_image_mae = image_mae;
                stats.worst_image = relative;
            }
            stats.values += candidate.pixels.size();
            ++stats.images;
        }
        return stats;
    }

    [[nodiscard]] std::string_view classification(const double mae) {
        if (mae <= kPassMae) return "PASS";
        if (mae <= kHardFailMae) return "BORDERLINE";
        return "FAIL";
    }

    [[nodiscard]] double print_statistics(const std::string_view label, const Statistics& stats) {
        const auto mae = static_cast<double>(stats.sum_absolute / stats.values) / kScale;
        const auto rmse = std::sqrt(static_cast<double>(stats.sum_squared / stats.values)) / kScale;
        const auto exceedance = 100.0 * static_cast<double>(stats.above_1e4) / static_cast<double>(stats.values);
        std::cout << std::format(
            "{}: {} images, {} values, {}\n"
            "  normalized MAE={:.9g} RMSE={:.9g} max={:.9g} (>1e-4: {:.6f}%)\n"
            "  abs-delta p50={} p90={} p95={} p99={} p99.9={} max={}\n"
            "  worst pixel: {} ({},{},ch{}) candidate={} reference={}\n"
            "  worst image MAE={:.9g}: {}\n",
            label, stats.images, stats.values, classification(mae), mae, rmse,
            static_cast<double>(stats.worst.delta) / kScale, exceedance,
            percentile(stats, 0.50), percentile(stats, 0.90), percentile(stats, 0.95),
            percentile(stats, 0.99), percentile(stats, 0.999), stats.worst.delta,
            path_string(stats.worst.relative_path), stats.worst.x, stats.worst.y,
            stats.worst.channel, stats.worst.candidate, stats.worst.reference,
            stats.worst_image_mae, path_string(stats.worst_image));
        return mae;
    }
} // namespace

int main(const int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: lfs_preprocess_compare <candidate-depth> <candidate-normals> "
                     "<reference-depth> <reference-normals>\n";
        return 2;
    }
    try {
        const auto depth = compare_directories(argv[1], argv[3], 1);
        const auto normals = compare_directories(argv[2], argv[4], 3);
        const auto depth_mae = print_statistics("depth", depth);
        const auto normals_mae = print_statistics("normals", normals);
        const auto combined_sum = depth.sum_absolute + normals.sum_absolute;
        const auto combined_values = depth.values + normals.values;
        const auto combined_mae = static_cast<double>(combined_sum / combined_values) / kScale;
        std::cout << std::format("combined normalized MAE={:.9g}: {}\n", combined_mae,
                                 classification(combined_mae));
        return depth_mae > kHardFailMae || normals_mae > kHardFailMae ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "Comparison failed: " << error.what() << '\n';
        return 2;
    }
}
