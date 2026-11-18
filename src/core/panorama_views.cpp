/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/panorama_views.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <numbers>

namespace lfs::core {

    namespace {
        struct Face {
            const char* suffix;
            glm::vec3 forward;
        };
        constexpr std::array<Face, kPanoramaViewCount> faces{{
            {"_px", {1, 0, 0}},
            {"_nx", {-1, 0, 0}},
            {"_py", {0, 1, 0}},
            {"_ny", {0, -1, 0}},
            {"_pz", {0, 0, 1}},
            {"_nz", {0, 0, -1}},
        }};
    } // namespace

    std::string panorama_image_name(const std::string& view_image_name) {
        auto path = utf8_to_path(view_image_name);
        const auto stem = path_to_utf8(path.stem());
        const auto name = stem.substr(0, stem.rfind('_')) + path_to_utf8(path.extension());
        path.replace_filename(utf8_to_path(name));
        return path_to_utf8(path);
    }

    std::array<UndistortParams, kPanoramaViewCount> panorama_view_params(
        const int width, const int height) {
        constexpr float pi = std::numbers::pi_v<float>;
        const float src_fx = width / (2 * pi);
        const float src_fy = height / pi;
        constexpr float fov = kPanoramaViewFovDegrees * (pi / 180.0f);
        const int size = std::max(1, static_cast<int>(std::lround(2 * std::max(src_fx, src_fy) * std::tan(fov / 2))));
        const float focal = fov2focal(fov, size);

        std::array<UndistortParams, kPanoramaViewCount> result{};
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto forward = faces[i].forward;
            const glm::vec3 up_hint = forward.y == 0 ? glm::vec3{0, 1, 0} : glm::vec3{0, 0, 1};
            const auto right = glm::cross(up_hint, forward);
            const auto up = glm::cross(forward, right);
            result[i] = {
                .src_fx = src_fx,
                .src_fy = src_fy,
                .src_cx = width * 0.5f,
                .src_cy = height * 0.5f,
                .dst_fx = focal,
                .dst_fy = focal,
                .dst_cx = size * 0.5f,
                .dst_cy = size * 0.5f,
                .src_width = width,
                .src_height = height,
                .dst_width = size,
                .dst_height = size,
                .model_type = CameraModelType::EQUIRECTANGULAR,
                .distortion = {},
                .num_distortion = 0,
                .view_rotation = {right.x, up.x, forward.x, right.y, up.y, forward.y, right.z, up.z, forward.z},
            };
        }
        return result;
    }

    std::array<std::shared_ptr<Camera>, kPanoramaViewCount> make_panorama_views(
        const Camera& panorama, const int first_uid) {
        const auto params = panorama_view_params(panorama.camera_width(), panorama.camera_height());
        const auto name = utf8_to_path(panorama.image_name());
        std::array<std::shared_ptr<Camera>, kPanoramaViewCount> views;
        for (int i = 0; i < kPanoramaViewCount; ++i) {
            auto view_name = name;
            view_name.replace_extension();
            view_name += faces[i].suffix;
            view_name += name.extension();
            views[i] = panorama.make_pinhole_view(params[i], first_uid + i, path_to_utf8(view_name));
        }
        return views;
    }

} // namespace lfs::core
