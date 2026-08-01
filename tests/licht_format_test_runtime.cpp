/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/failure_report.hpp"
#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "licht_test_support.hpp"

#include <cstddef>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace lfs::core::detail {

    [[noreturn]] void assertion_failed(
        const std::string_view contract, const std::string_view expression,
        const std::string_view message, const SourceSite location) {
        std::string error = std::format("{} failed: {}", contract, expression);
        if (!message.empty()) {
            error += " — ";
            error += message;
        }
        error += std::format(" ({}:{})", location.file_name(), location.line());
        throw std::runtime_error(error);
    }

} // namespace lfs::core::detail

namespace lfs::core {

    struct Logger::Impl {};

    Logger::Logger()
        : impl_(std::make_unique<Impl>()) {}

    Logger::~Logger() = default;

    Logger& Logger::get() {
        static Logger logger;
        return logger;
    }

    void Logger::log(LogLevel, const SourceSite&, std::string_view) {}

} // namespace lfs::core

namespace lfs::test::licht {

    std::vector<std::byte> one_pixel_png() {
        static constexpr char PNG[] =
            "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52"
            "\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
            "\x89\x00\x00\x00\x0a\x49\x44\x41\x54\x78\x9c\x63\x60\x00\x00\x00"
            "\x02\x00\x01\xe5\x27\xd4\xa2\x00\x00\x00\x00\x49\x45\x4e\x44\xae"
            "\x42\x60\x82";
        const auto bytes = std::as_bytes(std::span(PNG, sizeof(PNG) - 1));
        return {bytes.begin(), bytes.end()};
    }

    std::shared_ptr<core::PointCloud> make_point_cloud(
        const std::size_t count) {
        std::vector<float> means(count * 3, 0.0f);
        std::vector<float> colors(count * 3, 0.5f);
        for (std::size_t index = 0; index < count; ++index) {
            means[index * 3] = static_cast<float>(index + 10);
        }
        return std::make_shared<core::PointCloud>(
            core::Tensor::from_vector(
                means, {count, std::size_t{3}}, core::Device::CPU),
            core::Tensor::from_vector(
                colors, {count, std::size_t{3}}, core::Device::CPU));
    }

} // namespace lfs::test::licht
