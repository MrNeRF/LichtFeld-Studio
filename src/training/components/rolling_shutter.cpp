/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rolling_shutter.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace lfs::training {

    namespace {
        // GeosunPre's operating range for the linear model, in radians and seconds.
        constexpr double MAX_ROTATION = 0.05;
        constexpr double MAX_READOUT = 0.05;
        constexpr double TRANSLATION_TOLERANCE = 1e-8;

        struct ParseError {
            std::string message;
        };

        const nlohmann::json& field(const nlohmann::json& j, const char* key) {
            const auto it = j.find(key);
            if (it == j.end())
                throw ParseError{std::format("missing {}", key)};
            return *it;
        }

        double number(const nlohmann::json& j, const char* key) {
            const auto& v = field(j, key);
            if (!v.is_number() || !std::isfinite(v.get<double>()))
                throw ParseError{std::format("invalid {}", key)};
            return v.get<double>();
        }

        int integer(const nlohmann::json& j, const char* key) {
            const auto& v = field(j, key);
            if (!v.is_number_integer())
                throw ParseError{std::format("invalid {}", key)};
            return v.get<int>();
        }

        std::array<double, 3> vector3(const nlohmann::json& j, const char* key) {
            const auto& v = field(j, key);
            if (!v.is_array() || v.size() != 3)
                throw ParseError{std::format("invalid {}", key)};
            std::array<double, 3> out{};
            for (size_t k = 0; k < 3; ++k) {
                if (!v[k].is_number() || !std::isfinite(v[k].get<double>()))
                    throw ParseError{std::format("non-finite {}", key)};
                out[k] = v[k].get<double>();
            }
            return out;
        }

        using Mat3 = std::array<double, 9>;

        Mat3 multiply(const Mat3& a, const Mat3& b) {
            Mat3 out{};
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    for (int k = 0; k < 3; ++k)
                        out[r * 3 + c] += a[r * 3 + k] * b[k * 3 + c];
            return out;
        }

        // Rodrigues' formula for exp([w]x).
        Mat3 rotation_from_vector(const std::array<double, 3>& w) {
            const double theta = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
            const Mat3 K = {0.0, -w[2], w[1],
                            w[2], 0.0, -w[0],
                            -w[1], w[0], 0.0};
            const Mat3 K2 = multiply(K, K);
            // sin(t)/t and (1-cos(t))/t^2, by series below 1e-4 rad.
            const double a = theta < 1e-4 ? 1.0 - theta * theta / 6.0 : std::sin(theta) / theta;
            const double b = theta < 1e-4 ? 0.5 - theta * theta / 24.0 : (1.0 - std::cos(theta)) / (theta * theta);
            Mat3 out{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
            for (int i = 0; i < 9; ++i)
                out[i] += a * K[i] + b * K2[i];
            return out;
        }

        // World-to-camera of readout coordinate s for gsplat, which slerps the rotation and
        // lerps the translation between two such poses. The model's translation
        //   (I + s [w]x)(t - s d) = t + s ([w]x t - d) - s^2 [w]x d
        // keeps its linear part, and s^2 its best constant on [-1/2, 1/2], 1/8. The rotation
        // is exp(s [w]x) R: (I + s [w]x) is not a rotation, and the two differ in direction
        // by at most (s |w|)^2 / 2.
        std::array<float, 16> readout_pose(const core::RollingShutterMotion& motion,
                                           const std::array<float, 9>& rotation,
                                           const std::array<float, 3>& translation,
                                           const double s) {
            const std::array<double, 3> w = {motion.rotation[0], motion.rotation[1], motion.rotation[2]};
            const std::array<double, 3> d = {motion.displacement[0], motion.displacement[1], motion.displacement[2]};
            const std::array<double, 3> t = {translation[0], translation[1], translation[2]};
            const auto cross = [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
                return std::array<double, 3>{a[1] * b[2] - a[2] * b[1],
                                             a[2] * b[0] - a[0] * b[2],
                                             a[0] * b[1] - a[1] * b[0]};
            };
            const auto wt = cross(w, t);
            const auto wd = cross(w, d);

            const Mat3 Q = rotation_from_vector({s * w[0], s * w[1], s * w[2]});
            Mat3 R{};
            for (int i = 0; i < 9; ++i)
                R[i] = rotation[i];
            const Mat3 QR = multiply(Q, R);
            std::array<float, 16> out{};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c)
                    out[r * 4 + c] = static_cast<float>(QR[r * 3 + c]);
                out[r * 4 + 3] = static_cast<float>(t[r] + s * (wt[r] - d[r]) - wd[r] / 8.0);
            }
            out[15] = 1.0f;
            return out;
        }
    } // namespace

    std::expected<RollingShutterTable, std::string> parse_rolling_shutter(const std::string_view json_text) {
        try {
            const auto j = nlohmann::json::parse(json_text);
            if (integer(j, "version") != 1)
                throw ParseError{"unsupported version"};
            const auto& model = field(j, "model");
            if (!model.is_string() || model.get<std::string>() != "linear rolling shutter about the reference pose")
                throw ParseError{"unsupported camera model"};
            const auto& axis = field(j, "readout_axis");
            if (!axis.is_string() || (axis.get<std::string>() != "rows" && axis.get<std::string>() != "columns"))
                throw ParseError{"unknown readout_axis"};

            RollingShutterTable table;
            table.readout_rows = axis.get<std::string>() == "rows";

            const auto& cameras = field(j, "cameras");
            if (!cameras.is_array())
                throw ParseError{"invalid cameras"};
            for (const auto& c : cameras) {
                const int id = integer(c, "camera_id");
                const double readout = number(c, "readout_s");
                if (std::abs(readout) > MAX_READOUT)
                    throw ParseError{std::format("readout_s of camera {} is outside +-{} s", id, MAX_READOUT)};
                if (!table.readout_s.emplace(id, readout).second)
                    throw ParseError{std::format("duplicate camera {}", id)};
            }

            const auto& images = field(j, "images");
            if (!images.is_array())
                throw ParseError{"invalid images"};
            for (const auto& row : images) {
                const auto& name_field = field(row, "name");
                if (!name_field.is_string())
                    throw ParseError{"invalid name"};
                const auto name = name_field.get<std::string>();
                (void)integer(row, "image_id");
                RollingShutterTable::Image image;
                if (row.contains("camera_id")) {
                    image.camera_id = integer(row, "camera_id");
                    if (!table.readout_s.contains(image.camera_id))
                        throw ParseError{std::format("image '{}' names unknown camera {}", name, image.camera_id)};
                }
                image.rotation = vector3(row, "rotation_rad");
                image.velocity = vector3(row, "velocity_m_s");
                for (int k = 0; k < 3; ++k) {
                    if (std::abs(image.rotation[k]) > MAX_ROTATION)
                        throw ParseError{std::format("rotation of image '{}' is outside +-{} rad", name, MAX_ROTATION)};
                }
                image.translation = vector3(row, "translation_m");
                if (!table.images.emplace(name, image).second)
                    throw ParseError{std::format("duplicate image '{}'", name)};
            }
            if (table.images.empty() || table.readout_s.empty())
                throw ParseError{"empty camera table"};
            return table;
        } catch (const ParseError& e) {
            return std::unexpected(std::format("{}: {}", ROLLING_SHUTTER_SIDECAR, e.message));
        } catch (const std::exception& e) {
            return std::unexpected(std::format("{}: {}", ROLLING_SHUTTER_SIDECAR, e.what()));
        }
    }

    std::expected<std::optional<RollingShutterTable>, std::string> read_rolling_shutter(
        const std::filesystem::path& dataset) {
        if (dataset.empty())
            return std::nullopt;
        const auto path = dataset / ROLLING_SHUTTER_SIDECAR;
        std::error_code ec;
        if (!core::native_fs::is_regular_file(path, ec))
            return std::nullopt;
        std::ifstream file;
        if (!core::open_file_for_read(path, std::ios::binary, file))
            return std::unexpected(std::format("Cannot read '{}'", core::path_to_utf8(path)));
        std::ostringstream text;
        text << file.rdbuf();
        auto table = parse_rolling_shutter(text.str());
        if (!table)
            return std::unexpected(table.error());
        return std::optional<RollingShutterTable>(std::move(*table));
    }

    std::expected<void, std::string> attach_rolling_shutter(
        const RollingShutterTable& table, const std::span<const std::shared_ptr<core::Camera>> cameras) {
        for (const auto& camera : cameras) {
            if (!camera)
                continue;
            const auto found = table.images.find(camera->image_name());
            if (found == table.images.end())
                return std::unexpected(std::format("{} has no entry for image '{}'",
                                                   ROLLING_SHUTTER_SIDECAR, camera->image_name()));
            const auto& image = found->second;
            if (image.camera_id >= 0 && image.camera_id != camera->camera_id())
                return std::unexpected(std::format("{}: image '{}' belongs to camera {} but COLMAP says {}",
                                                   ROLLING_SHUTTER_SIDECAR, camera->image_name(),
                                                   image.camera_id, camera->camera_id()));
            const auto readout = table.readout_s.find(camera->camera_id());
            if (readout == table.readout_s.end())
                return std::unexpected(std::format("{} has no readout time for camera {}",
                                                   ROLLING_SHUTTER_SIDECAR, camera->camera_id()));
            for (int k = 0; k < 3; ++k) {
                if (std::abs(image.translation[k] - readout->second * image.velocity[k]) > TRANSLATION_TOLERANCE)
                    return std::unexpected(std::format("{}: translation_m of image '{}' differs from readout_s * velocity_m_s",
                                                       ROLLING_SHUTTER_SIDECAR, camera->image_name()));
            }
            core::RollingShutterMotion motion;
            motion.readout_rows = table.readout_rows;
            for (int k = 0; k < 3; ++k) {
                motion.rotation[k] = static_cast<float>(image.rotation[k]);
                motion.displacement[k] = static_cast<float>(readout->second * image.velocity[k]);
            }
            camera->set_rolling_shutter(motion);
        }
        return {};
    }

    void clear_rolling_shutter(const std::span<const std::shared_ptr<core::Camera>> cameras) {
        for (const auto& camera : cameras) {
            if (camera)
                camera->set_rolling_shutter(std::nullopt);
        }
    }

    RollingShutterWindowPoses rolling_shutter_window_poses(
        const core::RollingShutterMotion& motion,
        const std::array<float, 9>& rotation,
        const std::array<float, 3>& translation,
        const int full_extent, const int offset, const int extent) {
        // Pixel centres sit at c + 0.5 and s = c / N - 0.5 at the centre of each line.
        const auto s_of = [&](const int line) {
            return (static_cast<double>(line) + 0.5) / static_cast<double>(full_extent) - 0.5;
        };
        const int last_line = offset + std::max(extent, 1) - 1;
        return {readout_pose(motion, rotation, translation, s_of(offset)),
                readout_pose(motion, rotation, translation, s_of(last_line))};
    }

} // namespace lfs::training
