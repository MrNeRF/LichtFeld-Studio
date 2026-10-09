/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "compare_images.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/filesystem_utils.hpp"
#include "io/formats/colmap.hpp"
#include "io/pipelined_image_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <set>

namespace lfs::training {

    namespace {
        constexpr std::array IMAGE_EXTENSIONS{".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".webp", ".bmp"};

        std::string lowercase(std::string text) {
            std::ranges::transform(text, text.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return text;
        }

        // Relative path without extension, so cam0/frame.png pairs with cam0/frame.exr but not with cam1/frame.png.
        std::string pairing_key(const std::filesystem::path& relative) {
            auto key = lowercase(lfs::core::path_to_utf8(relative.parent_path() / relative.stem()));
            std::ranges::replace(key, '\\', '/');
            return key;
        }

        std::string image_name(const std::filesystem::path& relative) {
            auto name = lfs::core::path_to_utf8(relative);
            std::ranges::replace(name, '\\', '/');
            return name;
        }

        bool is_image(const std::filesystem::path& path) {
            const auto extension = lowercase(lfs::core::path_to_utf8(path.extension()));
            return std::ranges::find(IMAGE_EXTENSIONS, extension) != IMAGE_EXTENSIONS.end();
        }

        std::map<std::string, std::filesystem::path> images_by_key(const std::filesystem::path& folder,
                                                                   const bool recursive,
                                                                   std::vector<std::filesystem::path>& duplicates) {
            std::vector<std::filesystem::path> files;
            const auto collect = [&](const auto& entries) {
                for (const auto& entry : entries) {
                    if (entry.is_regular_file() && is_image(entry.path()))
                        files.push_back(entry.path());
                }
            };
            if (recursive)
                collect(std::filesystem::recursive_directory_iterator(folder));
            else
                collect(std::filesystem::directory_iterator(folder));
            std::ranges::sort(files);
            std::map<std::string, std::filesystem::path> images;
            for (const auto& file : files) {
                if (!images.emplace(pairing_key(file.lexically_relative(folder)), file).second)
                    duplicates.push_back(file);
            }
            return images;
        }

        lfs::Result<std::map<std::string, std::shared_ptr<lfs::core::Camera>>> cameras_by_key(
            const std::filesystem::path& sparse) {
            std::map<std::string, std::shared_ptr<lfs::core::Camera>> cameras;
            if (sparse.empty())
                return cameras;
            auto loaded = lfs::io::read_colmap_cameras_only(sparse);
            if (!loaded)
                return evaluation_error(std::format("Failed to read the COLMAP model {}: {}",
                                                    lfs::core::path_to_utf8(sparse), loaded.error().format()),
                                        LFS_SOURCE_SITE_CURRENT());
            for (auto& camera : std::get<0>(*loaded))
                cameras.emplace(pairing_key(lfs::core::utf8_to_path(camera->image_name())), std::move(camera));
            return cameras;
        }

        std::vector<float> values(const lfs::core::Tensor& tensor) {
            if (!tensor.is_valid() || tensor.numel() == 0)
                return {};
            return tensor.cpu().contiguous().to(lfs::core::DataType::Float32).to_vector();
        }

        bool nearly_equal(const float a, const float b, const float tolerance) {
            return std::abs(a - b) <= tolerance * std::max({1.0f, std::abs(a), std::abs(b)});
        }

        // Missing trailing coefficients count as zero, so [k1, k2] matches [k1, k2, 0].
        bool same_coefficients(std::vector<float> a, std::vector<float> b) {
            const auto size = std::max(a.size(), b.size());
            a.resize(size, 0.0f);
            b.resize(size, 0.0f);
            return std::ranges::equal(a, b, [](const float x, const float y) { return nearly_equal(x, y, 1.0e-6f); });
        }

        std::shared_ptr<lfs::core::Camera> reference_camera(const ImagePair& pair, const lfs::core::Camera* colmap,
                                                            const int uid) {
            const auto& name = pair.name;
            if (colmap) {
                return std::make_shared<lfs::core::Camera>(
                    colmap->R(), colmap->T(), colmap->focal_x(), colmap->focal_y(), colmap->center_x(),
                    colmap->center_y(), colmap->radial_distortion(), colmap->tangential_distortion(),
                    colmap->camera_model_type(), name, pair.reference, std::filesystem::path{}, colmap->camera_width(),
                    colmap->camera_height(), uid);
            }
            const auto [width, height, channels] = lfs::core::get_image_info(pair.reference);
            const float w = static_cast<float>(width);
            const float h = static_cast<float>(height);
            return std::make_shared<lfs::core::Camera>(
                lfs::core::Tensor::eye(3, lfs::core::Device::CPU), lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
                w, w, 0.5f * w, 0.5f * h, lfs::core::Tensor(), lfs::core::Tensor(), lfs::core::CameraModelType::PINHOLE,
                name, pair.reference, std::filesystem::path{}, width, height, uid);
        }

        // Whether two cameras look from the same place in the same direction.
        bool same_camera_pose(const lfs::core::Camera& a, const lfs::core::Camera& b) {
            const auto close = [](const lfs::core::Tensor& x, const lfs::core::Tensor& y) {
                return std::ranges::equal(values(x), values(y),
                                          [](const float p, const float q) { return nearly_equal(p, q, 1.0e-4f); });
            };
            return close(a.R(), b.R()) && close(a.T(), b.T());
        }

        // The reference lens undistorted onto the pinhole grid the test images were rendered with.
        lfs::core::UndistortParams undistortion_onto(lfs::core::Camera& reference, const lfs::core::Camera& test) {
            reference.precompute_undistortion();
            auto params = reference.undistort_params();
            params.dst_fx = test.focal_x();
            params.dst_fy = test.focal_y();
            params.dst_cx = test.center_x();
            params.dst_cy = test.center_y();
            params.dst_width = test.camera_width();
            params.dst_height = test.camera_height();
            return params;
        }

        struct TestImage {
            std::filesystem::path path;
            bool undistort_like_reference = false;
        };

        lfs::Result<EvaluationRenderResult> load_test_image(lfs::io::PipelinedImageLoader& loader, const TestImage& test,
                                                            const lfs::core::Camera& view,
                                                            const lfs::core::Camera& render_camera,
                                                            const lfs::core::param::TrainingParameters& params) {
            lfs::core::Tensor image;
            try {
                image = loader.load_image_immediate(
                    test.path,
                    evaluation_load_params_like_reference(test.path, view, params, test.undistort_like_reference));
            } catch (const std::exception& e) {
                // LFS-CENSUS-OK(empty-catch): converted into a typed evaluation error
                return evaluation_error(std::format("cannot read test image {}: {}", lfs::core::path_to_utf8(test.path),
                                                    e.what()),
                                        LFS_SOURCE_SITE_CURRENT());
            }
            if (!image.is_valid() || image.ndim() != 3 || image.shape()[0] != 3)
                return evaluation_error(std::format("cannot read test image {} as RGB", lfs::core::path_to_utf8(test.path)),
                                        LFS_SOURCE_SITE_CURRENT());
            if (image.dtype() == lfs::core::DataType::UInt8)
                image = image.to(lfs::core::DataType::Float32) / 255.0f;
            const int width = static_cast<int>(image.shape()[2]);
            const int height = static_cast<int>(image.shape()[1]);
            const int target_width = render_camera.image_width();
            const int target_height = render_camera.image_height();
            if (width != target_width || height != target_height) {
                if (std::abs(static_cast<double>(width) * target_height - static_cast<double>(height) * target_width) >
                    static_cast<double>(std::max(width, target_width)))
                    return evaluation_error(std::format("test image {} is {}x{} but its reference is {}x{}",
                                                        lfs::core::path_to_utf8(test.path), width, height, target_width,
                                                        target_height),
                                            LFS_SOURCE_SITE_CURRENT());
                image = lfs::core::lanczos_resize_float_chw(image, target_height, target_width, 2,
                                                            lfs::core::getCurrentCUDAStream());
            }
            RenderOutput output;
            output.image = image;
            output.width = target_width;
            output.height = target_height;
            return EvaluationRenderResult{.output = std::move(output), .raw_image = std::move(image)};
        }
        // The normalized mask spec with the mesh: prefix normalization drops; empty stays empty.
        std::string mask_label(const std::string& spec) {
            namespace param = lfs::core::param;
            const bool mesh = !spec.empty() && !param::is_eval_mask_folder(spec) && !param::is_eval_mask_box(spec) &&
                              !param::is_eval_mask_points(spec) && !param::eval_mask_splat_file(spec);
            return mesh ? "mesh:" + spec : spec;
        }
    } // namespace

    ImageSet resolve_image_set(const std::filesystem::path& path) {
        if (!std::filesystem::is_directory(path))
            return {.images = path};
        for (const auto& sparse : lfs::io::get_colmap_search_paths(path)) {
            if (lfs::io::find_file_ci(sparse, "cameras.bin").empty() && lfs::io::find_file_ci(sparse, "cameras.txt").empty())
                continue;
            if (std::filesystem::is_directory(path / "images"))
                return {.images = path / "images", .sparse = sparse, .recursive = true};
            return {.images = path, .sparse = sparse};
        }
        return {.images = path};
    }

    ImagePairing pair_images_by_name(const std::filesystem::path& reference, const std::filesystem::path& test,
                                     const bool recursive) {
        ImagePairing pairing;
        if (!std::filesystem::is_directory(reference)) {
            const auto name = reference.filename();
            pairing.pairs.push_back({reference, test, image_name(name), pairing_key(name),
                                     pairing_key(test.filename())});
            return pairing;
        }
        auto references = images_by_key(reference, recursive, pairing.unmatched);
        auto tests = images_by_key(test, recursive, pairing.unmatched);
        for (auto& [key, path] : references) {
            if (const auto match = tests.find(key); match != tests.end()) {
                pairing.pairs.push_back({path, match->second, image_name(path.lexically_relative(reference)), key,
                                         pairing_key(match->second.lexically_relative(test))});
                tests.erase(match);
            } else {
                pairing.unmatched.push_back(path);
            }
        }
        for (auto& [key, path] : tests)
            pairing.unmatched.push_back(path);
        return pairing;
    }

    TestLens classify_test_lens(const lfs::core::Camera& reference, const lfs::core::Camera& test) {
        using lfs::core::CameraModelType;
        const bool reference_panorama = reference.camera_model_type() == CameraModelType::EQUIRECTANGULAR;
        const bool test_panorama = test.camera_model_type() == CameraModelType::EQUIRECTANGULAR;
        if (reference_panorama || test_panorama)
            return reference_panorama && test_panorama ? TestLens::Reference : TestLens::Other;
        const auto normalized = [](const lfs::core::Camera& camera) {
            const float width = static_cast<float>(camera.camera_width());
            const float height = static_cast<float>(camera.camera_height());
            return std::array{camera.focal_x() / width, camera.focal_y() / height, camera.center_x() / width,
                              camera.center_y() / height};
        };
        const bool same_intrinsics =
            std::ranges::equal(normalized(reference), normalized(test),
                               [](const float a, const float b) { return nearly_equal(a, b, 1.0e-4f); });
        const bool same_distortion =
            reference.camera_model_type() == test.camera_model_type() &&
            same_coefficients(values(reference.radial_distortion()), values(test.radial_distortion())) &&
            same_coefficients(values(reference.tangential_distortion()), values(test.tangential_distortion()));
        if (same_intrinsics && same_distortion)
            return TestLens::Reference;
        if (reference.has_distortion() && test.camera_model_type() == CameraModelType::PINHOLE &&
            !test.has_distortion())
            return TestLens::Pinhole;
        return TestLens::Other;
    }

    lfs::Result<CompareResult> compare_image_sets(const lfs::core::param::CompareParameters& params,
                                                  std::optional<std::filesystem::path> lpips_weights) {
        namespace param = lfs::core::param;
        const auto reference_set = resolve_image_set(params.reference_path);
        const auto test_set = resolve_image_set(params.test_path);
        // A reference folder without cameras borrows the test dataset's.
        const auto reference_sparse = !params.colmap_path.empty()     ? params.colmap_path
                                      : !reference_set.sparse.empty() ? reference_set.sparse
                                                                      : test_set.sparse;
        const auto test_sparse = test_set.sparse == reference_sparse ? std::filesystem::path{} : test_set.sparse;
        if (reference_sparse.empty() && !params.evaluation.eval_mask.empty() && !param::is_eval_mask_folder(params.evaluation.eval_mask))
            return evaluation_error("Mesh, box, point and splat masks need the cameras of the reference images: "
                                    "compare COLMAP datasets or pass --colmap",
                                    LFS_SOURCE_SITE_CURRENT());
        if (reference_sparse.empty() && params.evaluation.undistort)
            return evaluation_error("--undistort needs the cameras of the reference images: compare COLMAP datasets "
                                    "or pass --colmap",
                                    LFS_SOURCE_SITE_CURRENT());
        if (!test_set.sparse.empty() && params.evaluation.undistort)
            return evaluation_error("The test dataset's cameras already tell whether it was rendered undistorted; "
                                    "drop --undistort",
                                    LFS_SOURCE_SITE_CURRENT());

        const auto pairing =
            pair_images_by_name(reference_set.images, test_set.images, reference_set.recursive || test_set.recursive);
        CompareResult result{.unmatched = pairing.unmatched};
        for (const auto& path : pairing.unmatched)
            LOG_WARN("compare: no counterpart for {}", lfs::core::path_to_utf8(path));

        auto reference_cameras = cameras_by_key(reference_sparse);
        if (!reference_cameras)
            return std::move(reference_cameras.error());
        auto test_cameras = cameras_by_key(test_sparse);
        if (!test_cameras)
            return std::move(test_cameras.error());

        std::vector<std::shared_ptr<lfs::core::Camera>> cameras;
        std::map<std::string, TestImage> tests;
        size_t pinhole_tests = 0;
        size_t native_distorted_tests = 0;
        size_t moved_cameras = 0;
        for (const auto& pair : pairing.pairs) {
            const lfs::core::Camera* colmap = nullptr;
            if (!reference_sparse.empty()) {
                const auto found = reference_cameras->find(pair.reference_key);
                if (found == reference_cameras->end()) {
                    LOG_WARN("compare: {} has no camera in {}", lfs::core::path_to_utf8(pair.reference),
                             lfs::core::path_to_utf8(reference_sparse));
                    result.unmatched.push_back(pair.reference);
                    continue;
                }
                colmap = found->second.get();
            }
            auto camera = reference_camera(pair, colmap, static_cast<int>(cameras.size()));
            if (camera->camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR && !params.evaluation.eval_mask.empty() &&
                !param::is_eval_mask_folder(params.evaluation.eval_mask))
                return evaluation_error("Mesh, box, point and splat masks do not follow equirectangular cameras yet; "
                                        "use a masks:<folder> mask",
                                        LFS_SOURCE_SITE_CURRENT());

            auto lens = params.evaluation.undistort && camera->has_distortion() ? TestLens::Pinhole : TestLens::Reference;
            const lfs::core::Camera* test_camera = nullptr;
            if (!test_sparse.empty()) {
                const auto found = test_cameras->find(pair.test_key);
                if (found == test_cameras->end()) {
                    LOG_WARN("compare: {} has no camera in {}", lfs::core::path_to_utf8(pair.test),
                             lfs::core::path_to_utf8(test_sparse));
                    result.unmatched.push_back(pair.test);
                    continue;
                }
                test_camera = found->second.get();
                lens = classify_test_lens(*camera, *test_camera);
                if (!same_camera_pose(*camera, *test_camera))
                    ++moved_cameras;
            }

            switch (lens) {
            case TestLens::Other:
                return evaluation_error(
                    std::format("The camera of test image {} has another lens than its reference; compare maps a test "
                                "image onto its reference only for the same lens or a pinhole render of a distorted "
                                "reference",
                                lfs::core::path_to_utf8(pair.test)),
                    LFS_SOURCE_SITE_CURRENT());
            case TestLens::Pinhole:
                if (test_camera)
                    camera->adopt_undistortion(undistortion_onto(*camera, *test_camera));
                camera->prepare_undistortion();
                ++pinhole_tests;
                break;
            case TestLens::Reference:
                if (camera->has_distortion()) {
                    if (params.evaluation.eval_space == param::EvalSpace::Undistorted)
                        camera->prepare_undistortion();
                    else
                        ++native_distorted_tests;
                }
                break;
            }
            tests.emplace(camera->image_name(), TestImage{.path = pair.test,
                                                          .undistort_like_reference = lens == TestLens::Reference});
            cameras.push_back(std::move(camera));
        }
        if (cameras.empty())
            return evaluation_error("No image pairs to compare", LFS_SOURCE_SITE_CURRENT());
        if (pinhole_tests > 0 && native_distorted_tests > 0)
            return evaluation_error("Some test images were rendered undistorted and others with the lens of their "
                                    "references; compare them separately",
                                    LFS_SOURCE_SITE_CURRENT());
        if (moved_cameras > 0)
            LOG_WARN("compare: {} test cameras sit elsewhere than their reference cameras; masks follow the "
                     "reference cameras",
                     moved_cameras);
        const bool distorted_space = params.evaluation.eval_space == param::EvalSpace::Distorted;
        if (reference_sparse.empty())
            result.lens = "no cameras; compared as stored";
        else if (pinhole_tests > 0)
            result.lens = distorted_space ? "pinhole renders, warped into the reference lens"
                                          : "pinhole renders; references undistorted onto their grid";
        else if (std::ranges::any_of(cameras, [](const auto& c) { return c->has_distortion(); }))
            result.lens = distorted_space ? "the reference lens; compared as stored"
                                          : "the reference lens; both undistorted the same way";
        else
            result.lens = "the reference lens, without distortion; compared as stored";

        param::TrainingParameters evaluation;
        evaluation.dataset.resize_factor = -1;
        evaluation.dataset.max_width = 0;
        // A set of another size is compared at the smaller size, both downscaled by the same loader, whose max_width
        // caps the longer side. A pinhole test is measured against the undistorted grid it was rendered on. When the
        // pairs differ in size in more than one way, each test image is resized to its reference instead.
        std::set<std::pair<int, int>> long_sides;
        for (const auto& camera : cameras) {
            const auto& test = tests.at(camera->image_name());
            const auto [reference_width, reference_height, reference_channels] =
                lfs::core::get_image_info(camera->image_path());
            const int expected = test.undistort_like_reference ? std::max(reference_width, reference_height)
                                                               : std::max(camera->camera_width(), camera->camera_height());
            const auto [test_width, test_height, test_channels] = lfs::core::get_image_info(test.path);
            long_sides.emplace(expected, std::max(test_width, test_height));
        }
        const auto [expected_side, test_side] = *long_sides.begin();
        if (long_sides.size() == 1 && expected_side != test_side)
            evaluation.dataset.max_width = std::min(expected_side, test_side);
        else if (long_sides.size() > 1)
            LOG_WARN("compare: the test images differ in size from their references in more than one way; each is "
                     "resized to its reference");
        evaluation.optimization = params.evaluation;
        evaluation.optimization.enable_eval = true;
        evaluation.optimization.undistort = std::ranges::any_of(cameras, [](const auto& c) {
            return c->is_undistort_prepared();
        });
        // Test images in a distorted reference lens are scored like native GUT renders: masks follow that lens.
        evaluation.optimization.gut = native_distorted_tests > 0;
        std::filesystem::create_directories(params.output_path);
        const auto report_path = params.output_path / "compare_report.txt";
        const auto json_path = params.output_path / "compare.json";
        const auto image_dir = params.output_path / "compare_images";
        for (const auto& previous : {report_path, json_path, image_dir}) {
            std::error_code error;
            std::filesystem::remove_all(previous, error);
            if (error)
                return evaluation_error(std::format("cannot replace {}: {}", lfs::core::path_to_utf8(previous),
                                                    error.message()),
                                        LFS_SOURCE_SITE_CURRENT());
        }

        MetricsEvaluator evaluator(evaluation);
        evaluator.set_lpips_weights_path(std::move(lpips_weights));
        if (!params.evaluation.eval_mask.empty()) {
            auto installed = evaluator.install_eval_mask_source({0.0f, 0.0f, 0.0f});
            if (!installed)
                return std::move(installed.error());
            if (!*installed)
                return evaluation_error(std::format("compare cannot build the '{}' mask: it needs a trained model",
                                                    params.evaluation.eval_mask),
                                        LFS_SOURCE_SITE_CURRENT());
        }

        const auto loader = make_eval_image_loader(evaluation);
        const auto test_loader = make_eval_image_loader(evaluation);
        const auto dataset = std::make_shared<CameraDataset>(cameras, DatasetConfig{}, CameraDataset::Split::ALL);
        const auto background = lfs::core::Tensor::zeros({3}, lfs::core::Device::CUDA);
        EvaluationViewSource source{
            .render = [&](lfs::core::Camera& view, lfs::core::Camera& render_camera, float) {
                auto loaded = load_test_image(*test_loader, tests.at(view.image_name()), view, render_camera, evaluation);
                // The next test image decodes on the host while this view is scored, like the references do.
                if (const auto next = static_cast<size_t>(view.uid()) + 1; next < cameras.size()) {
                    const auto& camera = *cameras[next];
                    const auto& test = tests.at(camera.image_name());
                    test_loader->decode_ahead(test.path, evaluation_load_params_like_reference(
                                                             test.path, camera, evaluation, test.undistort_like_reference));
                }
                return loaded;
            },
            .warp_supersample = 1,
            .image_dir = image_dir};
        result.metrics = evaluator.evaluate_views(0, dataset, source, background, loader.get());
        for (const auto& [path, text] : {std::pair{report_path, format_compare_report(params, result)},
                                         std::pair{json_path, compare_json(params, result).dump(2)}}) {
            std::ofstream file;
            if (!lfs::core::open_file_for_write(path, std::ios::out | std::ios::binary, file) ||
                !(file << text << '\n') || !file.flush())
                return evaluation_error(std::format("cannot write {}", lfs::core::path_to_utf8(path)),
                                        LFS_SOURCE_SITE_CURRENT());
        }
        return result;
    }

    nlohmann::ordered_json compare_json(const lfs::core::param::CompareParameters& params, const CompareResult& result) {
        const auto value = [](const std::optional<float>& score) {
            return score ? nlohmann::ordered_json(*score) : nlohmann::ordered_json(nullptr);
        };
        const auto& evaluation = params.evaluation;
        auto images = nlohmann::ordered_json::object();
        for (const auto& view : result.metrics.views) {
            if (!view.skipped_reason.empty()) {
                images[view.image_name] = {{"skipped", view.skipped_reason}};
                continue;
            }
            images[view.image_name] = {{"width", view.width},
                                       {"height", view.height},
                                       {"bit_depth", view.bit_depth},
                                       {"evaluated_pixel_fraction", view.evaluated_pixel_fraction},
                                       {"psnr", value(view.psnr)},
                                       {"ssim", value(view.ssim)},
                                       {"lpips", value(view.lpips)},
                                       {"flip", value(view.flip)}};
        }
        auto unmatched = nlohmann::ordered_json::array();
        for (const auto& path : result.unmatched)
            unmatched.push_back(lfs::core::path_to_utf8(path));
        const auto& metrics = result.metrics;
        return {{"reference", lfs::core::path_to_utf8(params.reference_path)},
                {"test", lfs::core::path_to_utf8(params.test_path)},
                {"lens", result.lens},
                {"eval_space", lfs::core::param::eval_space_name(evaluation.eval_space)},
                {"eval_mask", mask_label(evaluation.eval_mask)},
                {"eval_mask_invert", evaluation.eval_mask_invert},
                {"eval_bit_depth", lfs::core::param::eval_bit_depth_name(evaluation.eval_bit_depth)},
                {"mean", metrics.valid ? nlohmann::ordered_json{{"psnr", metrics.psnr},
                                                                {"ssim", metrics.ssim},
                                                                {"lpips", value(metrics.lpips)},
                                                                {"flip", value(metrics.flip)}}
                                       : nlohmann::ordered_json(nullptr)},
                {"images", std::move(images)},
                {"unmatched", std::move(unmatched)}};
    }

    std::string format_compare_report(const lfs::core::param::CompareParameters& params, const CompareResult& result) {
        const auto score = [](const std::optional<float>& value) {
            return value ? std::format("{:.4f}", *value) : std::string("-");
        };
        const bool masked = !params.evaluation.eval_mask.empty();
        const auto row = [&](const std::string_view name, const std::optional<float> psnr,
                             const std::optional<float> ssim, const std::optional<float> lpips,
                             const std::optional<float> flip, const std::string_view pixels) {
            return std::format("{:<40} {:>9} {:>8} {:>8}{}{}\n", name, score(psnr), score(ssim), score(lpips),
                               params.evaluation.eval_flip ? std::format(" {:>8}", score(flip)) : "",
                               masked ? std::format(" {:>8}", pixels) : "");
        };
        const auto& metrics = result.metrics;
        const auto skipped = std::ranges::count_if(metrics.views, [](const auto& v) { return !v.skipped_reason.empty(); });
        std::string report;
        report += std::format("reference  {}\n", lfs::core::path_to_utf8(params.reference_path));
        report += std::format("test       {}\n", lfs::core::path_to_utf8(params.test_path));
        report += std::format("lens       {}\n", result.lens);
        report += std::format("space      {}\n", lfs::core::param::eval_space_name(params.evaluation.eval_space));
        report += std::format("mask       {}{}\n", masked ? mask_label(params.evaluation.eval_mask) : "none",
                              params.evaluation.eval_mask_invert ? " (inverted)" : "");
        report += std::format("bit depth  {}\n", lfs::core::param::eval_bit_depth_name(params.evaluation.eval_bit_depth));
        report += std::format("pairs      {} compared, {} skipped, {} without a counterpart\n\n",
                              metrics.views.size() - static_cast<size_t>(skipped), skipped, result.unmatched.size());
        report += std::format("{:<40} {:>9} {:>8} {:>8}{}{}\n", "image", "PSNR", "SSIM", "LPIPS",
                              params.evaluation.eval_flip ? "     FLIP" : "", masked ? "   pixels" : "");
        for (const auto& view : metrics.views) {
            if (!view.skipped_reason.empty())
                report += std::format("{:<40} skipped: {}\n", view.image_name, view.skipped_reason);
            else
                report += row(view.image_name, view.psnr, view.ssim, view.lpips, view.flip,
                              std::format("{:.1f}%", 100.0f * view.evaluated_pixel_fraction));
        }
        if (metrics.valid)
            report += row("mean", metrics.psnr, metrics.ssim, metrics.lpips, metrics.flip, "");
        return report;
    }

} // namespace lfs::training
