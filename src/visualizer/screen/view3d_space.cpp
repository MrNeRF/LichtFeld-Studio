/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "screen/view3d_space.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>

namespace lfs::vis::screen {

    namespace {

        using Json = nlohmann::json;

        Json vec3Json(const glm::vec3& v) { return Json::array({v.x, v.y, v.z}); }

        Json mat3Json(const glm::mat3& m) {
            Json out = Json::array();
            for (int c = 0; c < 3; ++c) {
                for (int r = 0; r < 3; ++r)
                    out.push_back(m[c][r]);
            }
            return out;
        }

        template <std::size_t N>
        std::optional<std::array<float, N>> finiteArray(const Json& json) {
            if (!json.is_array() || json.size() != N)
                return std::nullopt;
            std::array<float, N> out{};
            for (std::size_t i = 0; i < N; ++i) {
                if (!json[i].is_number())
                    return std::nullopt;
                out[i] = json[i].get<float>();
                if (!std::isfinite(out[i]))
                    return std::nullopt;
            }
            return out;
        }

        std::optional<glm::vec3> readVec3(const Json& json) {
            const auto a = finiteArray<3>(json);
            if (!a)
                return std::nullopt;
            return glm::vec3((*a)[0], (*a)[1], (*a)[2]);
        }

        std::optional<glm::mat3> readMat3(const Json& json) {
            const auto a = finiteArray<9>(json);
            if (!a)
                return std::nullopt;
            glm::mat3 m;
            for (int c = 0; c < 3; ++c) {
                for (int r = 0; r < 3; ++r)
                    m[c][r] = (*a)[static_cast<std::size_t>(c * 3 + r)];
            }
            return m;
        }

        // Reads json[key] into `out` when present. A present but invalid
        // value fails the whole read.
        class FieldReader {
        public:
            explicit FieldReader(const Json& json) : json_(json) {}

            [[nodiscard]] bool ok() const { return ok_; }

            void boolean(const char* key, bool& out) {
                read(key, [&](const Json& v) {
                    if (!v.is_boolean())
                        return false;
                    out = v.get<bool>();
                    return true;
                });
            }

            void number(const char* key, float& out, const float lo, const float hi) {
                read(key, [&](const Json& v) {
                    if (!v.is_number())
                        return false;
                    const float value = v.get<float>();
                    if (!std::isfinite(value) || value < lo || value > hi)
                        return false;
                    out = value;
                    return true;
                });
            }

            void integer(const char* key, int& out, const int lo, const int hi) {
                read(key, [&](const Json& v) {
                    if (!v.is_number_integer())
                        return false;
                    const auto value = v.get<long long>();
                    if (value < lo || value > hi)
                        return false;
                    out = static_cast<int>(value);
                    return true;
                });
            }

            void count(const char* key, std::size_t& out) {
                read(key, [&](const Json& v) {
                    if (!v.is_number_unsigned())
                        return false;
                    out = v.get<std::size_t>();
                    return true;
                });
            }

            template <typename Enum>
            void enumeration(const char* key, Enum& out, const int lo, const int hi) {
                int value = static_cast<int>(out);
                integer(key, value, lo, hi);
                out = static_cast<Enum>(value);
            }

            void vec3(const char* key, glm::vec3& out) {
                read(key, [&](const Json& v) {
                    const auto value = readVec3(v);
                    if (!value)
                        return false;
                    out = *value;
                    return true;
                });
            }

            template <typename Fn>
            void read(const char* key, Fn&& fn) {
                const auto it = json_.find(key);
                if (it == json_.end())
                    return;
                if (!fn(*it))
                    ok_ = false;
            }

        private:
            const Json& json_;
            bool ok_ = true;
        };

        constexpr float kHuge = 1.0e9f;

    } // namespace

    std::unique_ptr<SpaceData> View3DSpace::clone() const {
        auto copy = std::make_unique<View3DSpace>();
        copy->camera = camera;
        copy->camera.camera.clearTransientMotion();
        copy->settings = settings;
        copy->auto_orthographic = auto_orthographic;
        return copy;
    }

    Json View3DSpace::save() const {
        return {{"camera",
                 {{"rotation", mat3Json(camera.camera.R)},
                  {"translation", vec3Json(camera.camera.t)},
                  {"pivot", vec3Json(camera.camera.pivot)}}},
                {"settings", viewSettingsToJson(settings)},
                {"auto_orthographic", auto_orthographic}};
    }

    bool View3DSpace::load(const Json& json) {
        if (!json.is_object())
            return false;
        const auto camera_it = json.find("camera");
        const auto settings_it = json.find("settings");
        if (camera_it == json.end() || !camera_it->is_object())
            return false;
        const auto rotation = camera_it->contains("rotation") ? readMat3((*camera_it)["rotation"]) : std::nullopt;
        const auto translation = camera_it->contains("translation") ? readVec3((*camera_it)["translation"]) : std::nullopt;
        const auto pivot = camera_it->contains("pivot") ? readVec3((*camera_it)["pivot"]) : std::nullopt;
        if (!rotation || !translation || !pivot)
            return false;
        std::optional<ViewSettings> restored = settings;
        if (settings_it != json.end()) {
            restored = viewSettingsFromJson(*settings_it, settings);
            if (!restored)
                return false;
        }
        const auto auto_ortho = json.find("auto_orthographic");
        if (auto_ortho != json.end() && !auto_ortho->is_boolean())
            return false;
        camera.setViewMatrix(*rotation, *translation);
        camera.camera.pivot = *pivot;
        settings = *restored;
        auto_orthographic = auto_ortho != json.end() && auto_ortho->get<bool>();
        return true;
    }

    ViewAxis alignedViewAxis(const glm::mat3& rotation) {
        const glm::vec3 forward = lfs::rendering::cameraForward(rotation);
        if (!std::isfinite(forward.x) || !std::isfinite(forward.y) || !std::isfinite(forward.z))
            return ViewAxis::None;
        constexpr float kAligned = 0.99995f; // about 0.6 degrees
        struct Candidate {
            glm::vec3 forward;
            ViewAxis axis;
        };
        static constexpr Candidate kCandidates[] = {
            {{0.0f, -1.0f, 0.0f}, ViewAxis::Top},
            {{0.0f, 1.0f, 0.0f}, ViewAxis::Bottom},
            {{0.0f, 0.0f, -1.0f}, ViewAxis::Front},
            {{0.0f, 0.0f, 1.0f}, ViewAxis::Back},
            {{-1.0f, 0.0f, 0.0f}, ViewAxis::Right},
            {{1.0f, 0.0f, 0.0f}, ViewAxis::Left},
        };
        const glm::vec3 unit = glm::normalize(forward);
        for (const auto& candidate : kCandidates) {
            if (glm::dot(unit, candidate.forward) >= kAligned)
                return candidate.axis;
        }
        return ViewAxis::None;
    }

    std::string viewLabel(const View3DSpace& view) {
        const char* direction = "User";
        switch (alignedViewAxis(view.camera.camera.R)) {
        case ViewAxis::Top: direction = "Top"; break;
        case ViewAxis::Bottom: direction = "Bottom"; break;
        case ViewAxis::Front: direction = "Front"; break;
        case ViewAxis::Back: direction = "Back"; break;
        case ViewAxis::Right: direction = "Right"; break;
        case ViewAxis::Left: direction = "Left"; break;
        case ViewAxis::None: break;
        }
        const char* projection = view.settings.equirectangular ? "Panorama"
                                 : view.settings.orthographic  ? "Orthographic"
                                                               : "Perspective";
        return std::string(direction) + " " + projection;
    }

    void setOrthographic(View3DSpace& view, const bool enabled, const float viewport_height) {
        auto& s = view.settings;
        view.auto_orthographic = false;
        if (enabled == s.orthographic)
            return;
        if (enabled) {
            constexpr float kMinScale = 1.0f;
            constexpr float kMaxScale = 10000.0f;
            const float distance = glm::length(view.camera.camera.pivot - view.camera.camera.t);
            const float half_tan = std::tan(lfs::rendering::focalLengthToVFovRad(s.focal_length_mm) * 0.5f);
            if (viewport_height > 0.0f && std::isfinite(distance) && distance > 0.01f && half_tan > 0.0f)
                s.ortho_scale = std::clamp(viewport_height / (2.0f * distance * half_tan), kMinScale, kMaxScale);
        }
        s.orthographic = enabled;
    }

    void setAxisView(View3DSpace& view, const ViewAxis axis, const float viewport_height) {
        int index = 0;
        bool negative = false;
        switch (axis) {
        case ViewAxis::Top: index = 1; break;
        case ViewAxis::Bottom: index = 1, negative = true; break;
        case ViewAxis::Front: index = 2; break;
        case ViewAxis::Back: index = 2, negative = true; break;
        case ViewAxis::Right: index = 0; break;
        case ViewAxis::Left: index = 0, negative = true; break;
        case ViewAxis::None: return;
        }
        view.camera.camera.setAxisAlignedView(index, negative);
        if (!view.settings.orthographic && !view.settings.equirectangular) {
            setOrthographic(view, true, viewport_height);
            view.auto_orthographic = true;
        }
    }

    bool applyViewCommand(View3DSpace& view, const std::string_view command, const float viewport_height) {
        auto& s = view.settings;
        if (command.starts_with("display:")) {
            const auto mode = command.substr(8);
            if (mode != "splats" && mode != "points" && mode != "rings" && mode != "centers")
                return false;
            s.point_cloud_mode = mode == "points";
            s.show_rings = mode == "rings";
            s.show_center_markers = mode == "centers";
        } else if (command == "depth") {
            s.depth_view = !s.depth_view;
        } else if (command == "projection") {
            setOrthographic(view, !s.orthographic, viewport_height);
        } else if (command.starts_with("axis:")) {
            static constexpr std::pair<std::string_view, ViewAxis> kAxes[] = {
                {"top", ViewAxis::Top},
                {"bottom", ViewAxis::Bottom},
                {"front", ViewAxis::Front},
                {"back", ViewAxis::Back},
                {"right", ViewAxis::Right},
                {"left", ViewAxis::Left},
            };
            const auto name = command.substr(5);
            const auto it = std::find_if(std::begin(kAxes), std::end(kAxes),
                                         [name](const auto& entry) { return entry.first == name; });
            if (it == std::end(kAxes))
                return false;
            setAxisView(view, it->second, viewport_height);
        } else if (command == "overlay:grid") {
            s.show_grid = !s.show_grid;
        } else if (command.starts_with("overlay:grid_plane:")) {
            const auto plane = command.substr(19);
            if (plane.size() != 1 || plane[0] < '0' || plane[0] > '2')
                return false;
            s.grid_plane = plane[0] - '0';
            s.show_grid = true;
        } else if (command == "overlay:axes") {
            s.show_coord_axes = !s.show_coord_axes;
        } else if (command == "overlay:pivot") {
            s.show_pivot = !s.show_pivot;
        } else if (command == "overlay:frustums") {
            s.show_camera_frustums = !s.show_camera_frustums;
        } else {
            return false;
        }
        return true;
    }

    Json viewSettingsToJson(const ViewSettings& s) {
        return {
            {"focal_length_mm", s.focal_length_mm},
            {"equirectangular", s.equirectangular},
            {"orthographic", s.orthographic},
            {"ortho_scale", s.ortho_scale},
            {"show_coord_axes", s.show_coord_axes},
            {"axes_size", s.axes_size},
            {"axes_visibility", Json::array({s.axes_visibility[0], s.axes_visibility[1], s.axes_visibility[2]})},
            {"show_grid", s.show_grid},
            {"grid_plane", s.grid_plane},
            {"grid_opacity", s.grid_opacity},
            {"point_cloud_mode", s.point_cloud_mode},
            {"voxel_size", s.voxel_size},
            {"show_rings", s.show_rings},
            {"ring_width", s.ring_width},
            {"show_center_markers", s.show_center_markers},
            {"show_camera_frustums", s.show_camera_frustums},
            {"camera_frustum_scale", s.camera_frustum_scale},
            {"show_pivot", s.show_pivot},
            {"split_view_mode", static_cast<int>(s.split_view_mode)},
            {"gt_comparison_mode", static_cast<int>(s.gt_comparison_mode)},
            {"split_position", s.split_position},
            {"split_view_offset", s.split_view_offset},
            {"depth_view", s.depth_view},
            {"depth_view_min", s.depth_view_min},
            {"depth_view_max", s.depth_view_max},
            {"depth_visualization_mode", static_cast<int>(s.depth_visualization_mode)},
            {"depth_filter_enabled", s.depth_filter_enabled},
            {"depth_filter_min", vec3Json(s.depth_filter_min)},
            {"depth_filter_max", vec3Json(s.depth_filter_max)},
            {"depth_filter_rotation", mat3Json(glm::mat3_cast(s.depth_filter_transform.getRotation()))},
            {"depth_filter_translation", vec3Json(s.depth_filter_transform.getTranslation())},
            {"depth_filter_scale_x", s.depth_filter_scale_x},
            {"depth_filter_scale_y", s.depth_filter_scale_y},
            {"depth_filter_offset_x", s.depth_filter_offset_x},
            {"depth_filter_offset_y", s.depth_filter_offset_y},
            {"depth_filter_viz_mode", s.depth_filter_viz_mode},
        };
    }

    std::optional<ViewSettings> viewSettingsFromJson(const Json& json, const ViewSettings& base) {
        if (!json.is_object())
            return std::nullopt;
        ViewSettings s = base;
        FieldReader r(json);
        r.number("focal_length_mm", s.focal_length_mm, 1.0f, 10000.0f);
        r.boolean("equirectangular", s.equirectangular);
        r.boolean("orthographic", s.orthographic);
        r.number("ortho_scale", s.ortho_scale, 1.0e-6f, kHuge);
        r.boolean("show_coord_axes", s.show_coord_axes);
        r.number("axes_size", s.axes_size, 0.0f, kHuge);
        r.read("axes_visibility", [&](const Json& v) {
            if (!v.is_array() || v.size() != 3)
                return false;
            for (std::size_t i = 0; i < 3; ++i) {
                if (!v[i].is_boolean())
                    return false;
                s.axes_visibility[i] = v[i].get<bool>();
            }
            return true;
        });
        r.boolean("show_grid", s.show_grid);
        r.integer("grid_plane", s.grid_plane, 0, 2);
        r.number("grid_opacity", s.grid_opacity, 0.0f, 1.0f);
        r.boolean("point_cloud_mode", s.point_cloud_mode);
        r.number("voxel_size", s.voxel_size, 0.0f, kHuge);
        r.boolean("show_rings", s.show_rings);
        r.number("ring_width", s.ring_width, 0.0f, kHuge);
        r.boolean("show_center_markers", s.show_center_markers);
        r.boolean("show_camera_frustums", s.show_camera_frustums);
        r.number("camera_frustum_scale", s.camera_frustum_scale, 0.0f, kHuge);
        r.boolean("show_pivot", s.show_pivot);
        r.enumeration("split_view_mode", s.split_view_mode, 0, static_cast<int>(SplitViewMode::GTComparison));
        r.enumeration("gt_comparison_mode", s.gt_comparison_mode, 0, static_cast<int>(GTComparisonMode::Loss));
        r.number("split_position", s.split_position, 0.0f, 1.0f);
        r.count("split_view_offset", s.split_view_offset);
        r.boolean("depth_view", s.depth_view);
        r.number("depth_view_min", s.depth_view_min, 0.0f, kHuge);
        r.number("depth_view_max", s.depth_view_max, 0.0f, kHuge);
        r.enumeration("depth_visualization_mode", s.depth_visualization_mode, 0,
                      static_cast<int>(lfs::rendering::DepthVisualizationMode::Grayscale));
        r.boolean("depth_filter_enabled", s.depth_filter_enabled);
        r.vec3("depth_filter_min", s.depth_filter_min);
        r.vec3("depth_filter_max", s.depth_filter_max);
        std::optional<glm::mat3> rotation;
        std::optional<glm::vec3> translation;
        r.read("depth_filter_rotation", [&](const Json& v) { return (rotation = readMat3(v)).has_value(); });
        r.read("depth_filter_translation", [&](const Json& v) { return (translation = readVec3(v)).has_value(); });
        r.number("depth_filter_scale_x", s.depth_filter_scale_x, 0.0f, kHuge);
        r.number("depth_filter_scale_y", s.depth_filter_scale_y, 0.0f, kHuge);
        r.number("depth_filter_offset_x", s.depth_filter_offset_x, -kHuge, kHuge);
        r.number("depth_filter_offset_y", s.depth_filter_offset_y, -kHuge, kHuge);
        r.integer("depth_filter_viz_mode", s.depth_filter_viz_mode, 0, 2);
        if (!r.ok())
            return std::nullopt;
        if (rotation || translation) {
            const glm::mat3 R = rotation.value_or(glm::mat3_cast(s.depth_filter_transform.getRotation()));
            const glm::vec3 t = translation.value_or(s.depth_filter_transform.getTranslation());
            s.depth_filter_transform = lfs::geometry::EuclideanTransform(glm::quat_cast(R), t);
        }
        sanitizeDepthViewSettings(s);
        return s;
    }

} // namespace lfs::vis::screen
