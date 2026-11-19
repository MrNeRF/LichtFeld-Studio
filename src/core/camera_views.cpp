/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera_views.hpp"
#include "core/camera_distortion_layout.hpp"
#include "core/distortion_model.hpp"
#include "core/path_utils.hpp"
#include "core/source_site.hpp"
#include <algorithm>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <numbers>
#include <unordered_map>

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

        constexpr double degree = std::numbers::pi / 180;

        double fisheye_reach(const UndistortParams& p, float lens_limit) {
            if (!(p.src_fx > 0 && p.src_fy > 0 && p.src_cx > 0 && p.src_cx < p.src_width &&
                  p.src_cy > 0 && p.src_cy < p.src_height && lens_limit > 0))
                return 0;
            std::array<double, 12> distortion;
            std::ranges::copy(p.distortion, distortion.begin());
            double reach = lens_limit;
            for (int i = 0; i < 720; ++i) {
                const double phi = i * (2 * std::numbers::pi / 720);
                const double x = std::cos(phi), y = std::sin(phi);
                const auto inside = [&](double theta) {
                    double u, v;
                    detail::project_fisheye_ray(std::sin(theta) * x, std::sin(theta) * y, std::cos(theta),
                                                p.model_type, distortion.data(), p.num_distortion, u, v);
                    u = p.src_fx * u + p.src_cx;
                    v = p.src_fy * v + p.src_cy;
                    return u >= 0 && u <= p.src_width && v >= 0 && v <= p.src_height;
                };
                if (inside(reach))
                    continue;
                double lo = 0, hi = reach;
                for (int j = 0; j < 32; ++j) {
                    const double mid = (lo + hi) / 2;
                    if (inside(mid))
                        lo = mid;
                    else
                        hi = mid;
                }
                reach = lo;
            }
            return reach;
        }

        std::vector<PinholeView> plan_panorama(int width, int height) {
            constexpr float pi = std::numbers::pi_v<float>;
            const float src_fx = width / (2 * pi);
            const float src_fy = height / pi;
            constexpr float fov = kPanoramaViewFovDegrees * (pi / 180.0f);
            const int size = std::max(1, static_cast<int>(std::lround(2 * std::max(src_fx, src_fy) * std::tan(fov / 2))));
            const float focal = fov2focal(fov, size);

            std::vector<PinholeView> result;
            result.reserve(faces.size());
            for (const auto& face : faces) {
                const auto forward = face.forward;
                const glm::vec3 up_hint = forward.y == 0 ? glm::vec3{0, 1, 0} : glm::vec3{0, 0, 1};
                const auto right = glm::cross(up_hint, forward);
                const auto up = glm::cross(forward, right);
                const UndistortParams params{
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
                    .view = true,
                    .view_rotation = {right.x, up.x, forward.x, right.y, up.y, forward.y, right.z, up.z, forward.z},
                };
                result.push_back({params, face.suffix});
            }
            return result;
        }
    } // namespace

    lfs::Result<std::vector<CameraViewGroup>> group_camera_views(
        std::span<const std::shared_ptr<Camera>> drawn, std::span<const std::shared_ptr<Camera>> scene) {
        std::unordered_map<std::filesystem::path, std::shared_ptr<Camera>> sources;
        for (const auto& camera : scene)
            if (camera)
                sources.emplace(camera->image_path(), camera);

        std::vector<CameraViewGroup> groups;
        std::unordered_map<const Camera*, size_t> rows;
        for (const auto& camera : drawn) {
            if (!camera)
                continue;
            auto owner = camera;
            if (camera->is_undistort_prepared() && is_view(camera->undistort_params())) {
                const auto found = sources.find(camera->image_path());
                if (found == sources.end())
                    return lfs::make_error({.code = lfs::ErrorCode::FailedPrecondition,
                                            .domain = lfs::ErrorDomain::Training,
                                            .user_message = "A pinhole view has no photograph in the scene",
                                            .detection = LFS_SOURCE_SITE_CURRENT()});
                owner = found->second;
            }
            const auto [it, inserted] = rows.try_emplace(owner.get(), groups.size());
            if (inserted)
                groups.push_back({owner, {}});
            groups[it->second].drawn.push_back(camera);
        }
        return groups;
    }

    std::string photograph_image_name(const std::string& view_image_name) {
        auto path = utf8_to_path(view_image_name);
        const auto stem = path_to_utf8(path.stem());
        const auto name = stem.substr(0, stem.rfind('_')) + path_to_utf8(path.extension());
        path.replace_filename(utf8_to_path(name));
        return path_to_utf8(path);
    }

    std::vector<PinholeView> plan_camera_views(const Camera& source, float lens_limit) {
        std::vector<PinholeView> result;
        const auto model = source.camera_model_type();
        if (model == CameraModelType::EQUIRECTANGULAR)
            return plan_panorama(source.camera_width(), source.camera_height());
        if (model != CameraModelType::FISHEYE && model != CameraModelType::THIN_PRISM_FISHEYE)
            return result;
        UndistortParams p{};
        if (source.is_undistort_precomputed()) {
            p = source.undistort_params();
            p.crop_solve_failed = false;
        } else {
            p.src_fx = source.focal_x();
            p.src_fy = source.focal_y();
            p.src_cx = source.center_x();
            p.src_cy = source.center_y();
            p.src_width = source.camera_width();
            p.src_height = source.camera_height();
            p.model_type = model;
            const auto coefficients = [](const Tensor& tensor) {
                return tensor.is_valid() ? tensor.to_vector() : std::vector<float>{};
            };
            const auto radial = coefficients(source.radial_distortion());
            const auto tangential = coefficients(source.tangential_distortion());
            if (radial.size() > 4 || tangential.size() > 6 ||
                (model == CameraModelType::FISHEYE && std::ranges::any_of(tangential, [](float t) { return t != 0; })))
                return result;
            p.num_distortion = packCameraDistortion(model, radial, tangential, std::span(p.distortion));
        }
        p.view = true;
        const double reach = fisheye_reach(p, lens_limit);
        if (reach < kFisheyeViewThresholdDegrees * degree)
            return result;

        const double focal = std::max(p.src_fx, p.src_fy);
        p.dst_fx = p.dst_fy = focal;
        p.dst_width = p.dst_height = std::max(1, int(std::lround(2 * focal * std::tan(kPanoramaViewFovDegrees * degree / 2))));
        p.dst_cx = p.dst_cy = p.dst_width * 0.5f;
        result.push_back({p, "_f"});

        constexpr double tilt = 60 * degree;
        const double across = std::tan(51 * degree);
        double beta = std::min(93 * degree, reach - degree);
        // A hemisphere needs the far edge on the horizon even when there is no outward margin.
        if (reach >= std::numbers::pi / 2)
            beta = std::max(beta, std::numbers::pi / 2);
        double outer = std::tan(beta - tilt);
        // Below the horizon the far corners lie farther off-axis than the edge midpoint.
        if (reach < std::numbers::pi / 2) {
            double lo = 0, hi = outer;
            for (int i = 0; i < 32; ++i) {
                const double u = (lo + hi) / 2;
                const double z = std::cos(tilt) - u * std::sin(tilt);
                if (z >= std::cos(reach) * std::sqrt(1 + u * u + across * across))
                    lo = u;
                else
                    hi = u;
            }
            outer = lo;
        }
        const int along_size = std::max(1, int(std::floor(focal * (std::tan(45 * degree) + outer))));
        const int across_size = std::max(1, int(std::floor(2 * focal * across)));
        for (int i = 0; i < 4; ++i) {
            const bool horizontal = i < 2;
            const int sign = i % 2 == 0 ? 1 : -1;
            const float c = std::cos(tilt), t = sign * std::sin(tilt);
            const std::array<float, 9> rotation = horizontal
                                                      ? std::array<float, 9>{c, 0, t, 0, 1, 0, -t, 0, c}
                                                      : std::array<float, 9>{1, 0, 0, 0, c, t, 0, -t, c};
            std::copy(rotation.begin(), rotation.end(), p.view_rotation);
            const float centre = sign > 0 ? along_size - focal * outer : focal * outer;
            p.dst_width = horizontal ? along_size : across_size;
            p.dst_height = horizontal ? across_size : along_size;
            p.dst_cx = horizontal ? centre : across_size / 2.f;
            p.dst_cy = horizontal ? across_size / 2.f : centre;
            result.push_back({p, faces[i].suffix});
        }
        return result;
    }

    std::vector<std::shared_ptr<Camera>> make_camera_views(
        const Camera& source, std::span<const PinholeView> plan, int first_uid) {
        const auto name = utf8_to_path(source.image_name());
        std::vector<std::shared_ptr<Camera>> views;
        for (const auto& face : plan) {
            auto view_name = name;
            view_name.replace_extension();
            view_name += face.suffix;
            view_name += name.extension();
            views.push_back(source.make_pinhole_view(face.params, first_uid + static_cast<int>(views.size()),
                                                     path_to_utf8(view_name)));
        }
        return views;
    }

} // namespace lfs::core
