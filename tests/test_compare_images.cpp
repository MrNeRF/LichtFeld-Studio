/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "io/cache_image_loader.hpp"
#include "io/pipelined_image_loader.hpp"
#include "training/metrics/compare_images.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <torch/torch.h>

namespace fs = std::filesystem;
using lfs::core::Camera;
using lfs::core::CameraModelType;
using lfs::core::Tensor;
using lfs::core::param::CompareParameters;
using lfs::core::param::EvalSpace;

namespace {
    const fs::path BICYCLE = fs::path(TEST_DATA_DIR) / "bicycle";
    const std::vector<std::string> NAMES{"_DSC8679", "_DSC8680"};

    fs::path fresh_directory(const std::string& name) {
        const auto path = fs::temp_directory_path() / name;
        fs::remove_all(path);
        fs::create_directories(path);
        return path;
    }

    fs::path copy_images_to(const std::string& images, const fs::path& folder) {
        fs::create_directories(folder);
        for (const auto& stem : NAMES)
            fs::copy_file(BICYCLE / images / (stem + ".JPG"), folder / (stem + ".JPG"));
        return folder;
    }

    fs::path copy_images(const std::string& images, const std::string& name) {
        return copy_images_to(images, fresh_directory(name));
    }

    // A COLMAP text model with one camera line per image, all looking down +z from the origin.
    void write_model(const fs::path& dataset, const std::string& camera) {
        const auto sparse = dataset / "sparse" / "0";
        fs::create_directories(sparse);
        std::ofstream cameras(sparse / "cameras.txt");
        std::ofstream images(sparse / "images.txt");
        for (size_t i = 0; i < NAMES.size(); ++i) {
            cameras << i + 1 << ' ' << camera << '\n';
            images << i + 1 << " 1 0 0 0 0 0 " << i << ' ' << i + 1 << ' ' << NAMES[i] << ".JPG\n\n";
        }
        std::ofstream(sparse / "points3D.txt").put('\n');
    }

    void install_image_loader() {
        lfs::io::CacheLoader::getInstance(false);
        lfs::core::set_image_loader([](const lfs::core::ImageLoadParams& p) {
            return lfs::io::CacheLoader::getInstance().load_cached_image(
                p.path, {.resize_factor = p.resize_factor,
                         .max_width = p.max_width,
                         .cuda_stream = p.stream,
                         .output_uint8 = p.output_uint8,
                         .skip_blob_cache = p.skip_blob_cache});
        });
    }

    CompareParameters compare(const fs::path& reference, const fs::path& test, const fs::path& output) {
        CompareParameters params;
        params.reference_path = reference;
        params.test_path = test;
        params.output_path = output;
        return params;
    }

    Camera camera(const CameraModelType model, const float focal, const int width, const int height,
                  const std::vector<float>& radial = {}) {
        return Camera(Tensor::eye(3, lfs::core::Device::CPU), Tensor::zeros({3}, lfs::core::Device::CPU), focal, focal,
                      0.5f * width, 0.5f * height,
                      radial.empty() ? Tensor()
                                     : Tensor::from_vector(radial, {radial.size()}, lfs::core::Device::CPU),
                      Tensor(), model, "view.png", "view.png", {}, width, height, 0);
    }

    void save_png16(const fs::path& path, const Tensor& chw) {
        ASSERT_EQ(chw.ndim(), 3u);
        const int height = static_cast<int>(chw.shape()[1]);
        const int width = static_cast<int>(chw.shape()[2]);
        const auto hwc = chw.clamp(0.0f, 1.0f).permute({1, 2, 0}).contiguous().cpu().to_vector();
        std::vector<uint16_t> pixels(hwc.size());
        for (size_t i = 0; i < hwc.size(); ++i)
            pixels[i] = static_cast<uint16_t>(hwc[i] * 65535.0f + 0.5f);
        ASSERT_TRUE(lfs::core::save_png(path, pixels.data(), width, height, 3, 16, 1));
    }

    // A pinhole dataset holding the reference images undistorted onto the grid of `undistort`, as a renderer with
    // those pinhole cameras would produce them.
    void write_pinhole_renders(const fs::path& reference, const fs::path& dataset,
                               const lfs::core::UndistortParams& undistort) {
        fs::create_directories(dataset / "images");
        const auto loader = lfs::training::make_eval_image_loader(lfs::core::param::TrainingParameters{});
        for (const auto& stem : NAMES) {
            lfs::io::LoadParams load;
            load.resize_factor = -1;
            load.undistort = &undistort;
            load.decode_float = true;
            load.decode_16bit = true;
            load.skip_blob_cache = true;
            save_png16(dataset / "images" / (stem + ".png"),
                       loader->load_image_immediate(reference / "images" / (stem + ".JPG"), load));
        }
        write_model(dataset, std::format("PINHOLE {} {} {} {} {} {}", undistort.dst_width, undistort.dst_height,
                                         undistort.dst_fx, undistort.dst_fy, undistort.dst_cx, undistort.dst_cy));
    }
} // namespace

// Catches pairing that depends on extension or case, or that drops the unmatched files silently.
TEST(CompareImages, PairsFoldersByNameWithoutExtension) {
    const auto reference = fresh_directory("lfs_compare_pair_reference");
    const auto test = fresh_directory("lfs_compare_pair_test");
    for (const auto* name : {"a.exr", "B.png", "only_reference.jpg", "notes.txt"})
        std::ofstream(reference / name).put('\n');
    for (const auto* name : {"A.png", "b.PNG", "only_test.png"})
        std::ofstream(test / name).put('\n');

    const auto pairing = lfs::training::pair_images_by_name(reference, test);
    ASSERT_EQ(pairing.pairs.size(), 2u);
    EXPECT_EQ(pairing.pairs[0].reference.filename(), "a.exr");
    EXPECT_EQ(pairing.pairs[0].test.filename(), "A.png");
    EXPECT_EQ(pairing.pairs[1].reference.filename(), "B.png");
    EXPECT_EQ(pairing.pairs[1].test.filename(), "b.PNG");
    ASSERT_EQ(pairing.unmatched.size(), 2u);
    EXPECT_EQ(pairing.unmatched[0].filename(), "only_reference.jpg");
    EXPECT_EQ(pairing.unmatched[1].filename(), "only_test.png");
}

// Catches dataset images in sub-folders being skipped, or same-named images in different sub-folders paired with
// each other.
TEST(CompareImages, PairsDatasetSubFoldersByRelativePath) {
    const auto reference = fresh_directory("lfs_compare_nested_reference");
    const auto test = fresh_directory("lfs_compare_nested_test");
    for (const auto* folder : {"cam0", "cam1"}) {
        fs::create_directories(reference / folder);
        fs::create_directories(test / folder);
        std::ofstream(reference / folder / "frame.exr").put('\n');
    }
    std::ofstream(test / "cam0" / "frame.png").put('\n');
    std::ofstream(test / "cam1" / "FRAME.png").put('\n');

    const auto nested = lfs::training::pair_images_by_name(reference, test, true);
    ASSERT_EQ(nested.pairs.size(), 2u);
    EXPECT_EQ(nested.pairs[0].name, "cam0/frame.exr");
    EXPECT_EQ(nested.pairs[0].test, test / "cam0" / "frame.png");
    EXPECT_EQ(nested.pairs[1].name, "cam1/frame.exr");
    EXPECT_EQ(nested.pairs[1].test, test / "cam1" / "FRAME.png");
    EXPECT_EQ(nested.pairs[1].reference_key, "cam1/frame");
    EXPECT_TRUE(nested.unmatched.empty());

    EXPECT_TRUE(lfs::training::pair_images_by_name(reference, test).pairs.empty());
}

// Catches a COLMAP dataset folder read as a plain image folder, or a plain folder given cameras it has not.
TEST(CompareImages, ResolvesDatasetsFoldersAndFiles) {
    const auto dataset = fresh_directory("lfs_compare_resolve_dataset");
    copy_images_to("images_8", dataset / "images");
    write_model(dataset, "PINHOLE 100 100 50 50 50 50");
    const auto set = lfs::training::resolve_image_set(dataset);
    EXPECT_EQ(set.images, dataset / "images");
    EXPECT_EQ(set.sparse, dataset / "sparse" / "0");

    const auto folder = lfs::training::resolve_image_set(dataset / "images");
    EXPECT_EQ(folder.images, dataset / "images");
    EXPECT_TRUE(folder.sparse.empty());

    const auto file = lfs::training::resolve_image_set(dataset / "images" / (NAMES[0] + ".JPG"));
    EXPECT_EQ(file.images, dataset / "images" / (NAMES[0] + ".JPG"));
    EXPECT_TRUE(file.sparse.empty());
}

// Catches a test camera at another resolution taken for another lens, a pinhole render of a distorted
// reference not recognised, and lenses compare cannot map accepted.
TEST(CompareImages, ClassifiesTheTestLens) {
    using lfs::training::TestLens;
    const auto reference = camera(CameraModelType::PINHOLE, 800.0f, 1000, 600, {-0.1f, 0.02f});
    EXPECT_EQ(lfs::training::classify_test_lens(reference, camera(CameraModelType::PINHOLE, 800.0f, 1000, 600,
                                                                  {-0.1f, 0.02f, 0.0f})),
              TestLens::Reference);
    EXPECT_EQ(lfs::training::classify_test_lens(reference, camera(CameraModelType::PINHOLE, 400.0f, 500, 300,
                                                                  {-0.1f, 0.02f})),
              TestLens::Reference);
    EXPECT_EQ(lfs::training::classify_test_lens(reference, camera(CameraModelType::PINHOLE, 700.0f, 1100, 650)),
              TestLens::Pinhole);
    EXPECT_EQ(lfs::training::classify_test_lens(reference, camera(CameraModelType::PINHOLE, 800.0f, 1000, 600,
                                                                  {-0.2f, 0.02f})),
              TestLens::Other);
    const auto pinhole = camera(CameraModelType::PINHOLE, 800.0f, 1000, 600);
    EXPECT_EQ(lfs::training::classify_test_lens(pinhole, camera(CameraModelType::PINHOLE, 900.0f, 1000, 600)),
              TestLens::Other);
    const auto panorama = camera(CameraModelType::EQUIRECTANGULAR, 1.0f, 2000, 1000);
    EXPECT_EQ(lfs::training::classify_test_lens(reference, panorama), TestLens::Other);
    EXPECT_EQ(lfs::training::classify_test_lens(panorama, camera(CameraModelType::EQUIRECTANGULAR, 1.0f, 1000, 500)),
              TestLens::Reference);
}

// Catches images above the training width cap being downscaled, a larger test set not being resized to its
// references, a smaller one being upsampled instead of the references downscaled, identical sets not scoring
// perfectly, the reports not being written, and a rerun
// into the same folder keeping the previous run's rows or images.
TEST(CompareImages, ScoresRealImagesAndResizesTheTestSet) {
    if (!torch::cuda::is_available())
        GTEST_SKIP() << "CUDA not available";
    install_image_loader();
    const auto reference = copy_images("images_8", "lfs_compare_reference");
    const auto output = fresh_directory("lfs_compare_output");

    const auto full_resolution = BICYCLE / "images" / (NAMES[0] + ".JPG");
    const auto native = lfs::training::compare_image_sets(compare(full_resolution, full_resolution, output / "native"),
                                                          std::nullopt);
    ASSERT_TRUE(native) << native.error().detail();
    const auto [native_width, native_height, native_channels] = lfs::core::get_image_info(full_resolution);
    ASSERT_GT(native_width, 3840);
    EXPECT_EQ(native->metrics.views[0].width, native_width);

    const auto same = lfs::training::compare_image_sets(compare(reference, reference, output / "same"), std::nullopt);
    ASSERT_TRUE(same) << same.error().detail();
    ASSERT_EQ(same->metrics.views.size(), 2u);
    EXPECT_FLOAT_EQ(same->metrics.psnr, 100.0f);
    EXPECT_NEAR(same->metrics.ssim, 1.0f, 1e-4f);

    const auto larger = copy_images("images_4", "lfs_compare_larger");
    auto params = compare(reference, larger, output / "larger");
    params.evaluation.eval_flip = true;
    params.evaluation.enable_save_eval_images = true;
    const auto resized = lfs::training::compare_image_sets(params, std::nullopt);
    ASSERT_TRUE(resized) << resized.error().detail();
    ASSERT_TRUE(resized->metrics.valid);
    for (const auto& view : resized->metrics.views) {
        EXPECT_TRUE(view.skipped_reason.empty()) << view.skipped_reason;
        ASSERT_TRUE(view.psnr && view.flip);
        EXPECT_GT(*view.psnr, 25.0f) << view.image_name;
        EXPECT_LT(*view.flip, 0.2f) << view.image_name;
    }
    EXPECT_TRUE(fs::exists(output / "larger" / "compare_report.txt"));
    EXPECT_FALSE(fs::exists(output / "larger" / "metrics.csv"));
    EXPECT_TRUE(fs::exists(output / "larger" / "compare_images" / (NAMES[0] + ".png")));
    EXPECT_TRUE(fs::exists(output / "larger" / "compare_images" / (NAMES[0] + "_flip.png")));

    params.evaluation.enable_save_eval_images = false;
    ASSERT_TRUE(lfs::training::compare_image_sets(params, std::nullopt));
    EXPECT_FALSE(fs::exists(output / "larger" / "compare_images"));

    const auto smaller = lfs::training::compare_image_sets(compare(larger, reference, output / "smaller"), std::nullopt);
    ASSERT_TRUE(smaller) << smaller.error().detail();
    ASSERT_TRUE(smaller->metrics.valid);
    const auto [small_width, small_height, small_channels] = lfs::core::get_image_info(reference / (NAMES[0] + ".JPG"));
    for (const auto& view : smaller->metrics.views) {
        EXPECT_EQ(view.width, small_width) << view.image_name;
        EXPECT_GT(*view.psnr, 25.0f) << view.image_name;
    }
    const auto report = nlohmann::json::parse(std::ifstream(output / "larger" / "compare.json"));
    ASSERT_EQ(report["images"].size(), 2u);
    const auto& image = report["images"]["_DSC8679.JPG"];
    EXPECT_EQ(image["width"], small_width);
    EXPECT_NEAR(image["psnr"].get<float>(), *resized->metrics.views[0].psnr, 1e-4f);
    EXPECT_TRUE(image["lpips"].is_null());
    EXPECT_NEAR(report["mean"]["flip"].get<float>(), *resized->metrics.flip, 1e-6f);
}

// Catches identical distorted sets scoring below perfect in either space, an undistorted space that does not
// undistort, a pinhole render of a distorted reference mapped with the default undistorted grid instead of the
// test camera's own (off-centre, other size and focal length), --undistort for a folder without cameras
// disagreeing with a dataset that brings those cameras, and --undistort accepted for a set that has cameras.
TEST(CompareImages, DistortedReferencesCompareInEitherSpace) {
    if (!torch::cuda::is_available())
        GTEST_SKIP() << "CUDA not available";
    install_image_loader();
    const auto root = fresh_directory("lfs_compare_lens");
    const auto output = root / "output";
    const auto [width, height, channels] = lfs::core::get_image_info(BICYCLE / "images_8" / (NAMES[0] + ".JPG"));
    const float focal = 0.8f * static_cast<float>(width);
    const std::vector<float> radial{-0.08f, 0.02f};
    const auto opencv = std::format("OPENCV {} {} {} {} {} {} {} {} 0 0", width, height, focal, focal, 0.5f * width,
                                    0.5f * height, radial[0], radial[1]);
    const auto reference = root / "reference";
    copy_images_to("images_8", reference / "images");
    write_model(reference, opencv);
    const auto same = root / "same";
    copy_images_to("images_8", same / "images");
    write_model(same, opencv);

    auto params = compare(reference, same, output / "same_distorted");
    const auto distorted = lfs::training::compare_image_sets(params, std::nullopt);
    ASSERT_TRUE(distorted) << distorted.error().detail();
    EXPECT_FLOAT_EQ(distorted->metrics.psnr, 100.0f);
    EXPECT_EQ(distorted->metrics.views[0].width, width);

    const auto undistort = lfs::core::compute_undistort_params(
        focal, focal, 0.5f * width, 0.5f * height, width, height,
        Tensor::from_vector(radial, {radial.size()}, lfs::core::Device::CPU), Tensor(), CameraModelType::PINHOLE);
    ASSERT_NE(undistort.dst_width, width);
    params = compare(reference, same, output / "same_undistorted");
    params.evaluation.eval_space = EvalSpace::Undistorted;
    const auto undistorted = lfs::training::compare_image_sets(params, std::nullopt);
    ASSERT_TRUE(undistorted) << undistorted.error().detail();
    EXPECT_FLOAT_EQ(undistorted->metrics.psnr, 100.0f);
    EXPECT_EQ(undistorted->metrics.views[0].width, undistort.dst_width);

    auto custom = undistort;
    custom.dst_fx *= 0.9f;
    custom.dst_fy *= 0.93f;
    custom.dst_cx += 9.5f;
    custom.dst_cy -= 6.25f;
    custom.dst_width += 24;
    custom.dst_height -= 12;
    const auto pinhole = root / "pinhole";
    write_pinhole_renders(reference, pinhole, custom);

    params = compare(reference, pinhole, output / "pinhole_undistorted");
    params.evaluation.eval_space = EvalSpace::Undistorted;
    const auto pinhole_undistorted = lfs::training::compare_image_sets(params, std::nullopt);
    ASSERT_TRUE(pinhole_undistorted) << pinhole_undistorted.error().detail();
    EXPECT_GT(pinhole_undistorted->metrics.psnr, 50.0f);
    EXPECT_EQ(pinhole_undistorted->metrics.views[0].width, custom.dst_width);

    params = compare(reference, pinhole, output / "pinhole_distorted");
    const auto pinhole_distorted = lfs::training::compare_image_sets(params, std::nullopt);
    ASSERT_TRUE(pinhole_distorted) << pinhole_distorted.error().detail();
    EXPECT_GT(pinhole_distorted->metrics.psnr, 28.0f);
    EXPECT_EQ(pinhole_distorted->metrics.views[0].width, width);
    EXPECT_TRUE(pinhole_distorted->metrics.views[0].validity_mask_applied);

    const auto default_grid = root / "default_grid";
    write_pinhole_renders(reference, default_grid, undistort);
    const auto dataset = lfs::training::compare_image_sets(compare(reference, default_grid, output / "dataset"),
                                                           std::nullopt);
    ASSERT_TRUE(dataset) << dataset.error().detail();
    params = compare(reference, default_grid / "images", output / "flag_distorted");
    params.evaluation.undistort = true;
    const auto flag = lfs::training::compare_image_sets(params, std::nullopt);
    ASSERT_TRUE(flag) << flag.error().detail();
    EXPECT_NEAR(flag->metrics.psnr, dataset->metrics.psnr, 1e-4f);

    params = compare(reference, reference, output / "flag_with_cameras");
    params.evaluation.undistort = true;
    EXPECT_FALSE(lfs::training::compare_image_sets(params, std::nullopt));

    write_model(pinhole, std::format("OPENCV {} {} {} {} {} {} -0.3 0 0 0", width, height, focal, focal,
                                     0.5f * width, 0.5f * height));
    EXPECT_FALSE(lfs::training::compare_image_sets(compare(reference, pinhole, output / "other"), std::nullopt));
}

// Catches a mask folder or a box mask that does not restrict the scored pixels, masks that need cameras
// accepted without them, and a test image of another shape compared anyway.
TEST(CompareImages, MasksAndShapeMismatch) {
    if (!torch::cuda::is_available())
        GTEST_SKIP() << "CUDA not available";
    install_image_loader();
    const auto reference = copy_images("images_8", "lfs_compare_mask_reference");
    const auto test = copy_images("images_4", "lfs_compare_mask_test");
    const auto masks = fresh_directory("lfs_compare_masks");
    const auto output = fresh_directory("lfs_compare_mask_output");
    for (const auto& stem : NAMES) {
        const auto [width, height, channels] = lfs::core::get_image_info(reference / (stem + ".JPG"));
        std::vector<uint8_t> mask(static_cast<size_t>(width) * height, 0);
        for (int y = 0; y < height; ++y)
            std::fill_n(mask.begin() + static_cast<size_t>(y) * width, width / 2, uint8_t{255});
        ASSERT_TRUE(lfs::core::save_png(masks / (stem + ".png"), mask.data(), width, height, 1, 8, 1));
    }

    auto params = compare(reference, test, output / "folder");
    params.evaluation.eval_mask = "masks:" + masks.string();
    const auto folder = lfs::training::compare_image_sets(params, std::nullopt);
    ASSERT_TRUE(folder) << folder.error().detail();
    for (const auto& view : folder->metrics.views)
        EXPECT_NEAR(view.evaluated_pixel_fraction, 0.5f, 0.01f) << view.image_name;

    params = compare(reference, test, output / "box");
    params.evaluation.eval_mask = "bbox:-2.4,-0.5,-1.3,3.3,1.75,4.1";
    EXPECT_FALSE(lfs::training::compare_image_sets(params, std::nullopt));
    params.colmap_path = BICYCLE / "sparse" / "0";
    const auto box = lfs::training::compare_image_sets(params, std::nullopt);
    ASSERT_TRUE(box) << box.error().detail();
    ASSERT_EQ(box->metrics.views.size(), 2u);
    for (const auto& view : box->metrics.views) {
        EXPECT_GT(view.evaluated_pixel_fraction, 0.05f) << view.image_name;
        EXPECT_LT(view.evaluated_pixel_fraction, 0.95f) << view.image_name;
    }

    const auto cropped = fresh_directory("lfs_compare_cropped");
    for (const auto& stem : NAMES) {
        auto [data, width, height, channels] = lfs::core::load_image(test / (stem + ".JPG"));
        ASSERT_TRUE(lfs::core::save_png(cropped / (stem + ".png"), data, width, height / 2, channels, 8, 1));
        lfs::core::free_image(data);
    }
    const auto mismatch = lfs::training::compare_image_sets(compare(reference, cropped, output / "crop"), std::nullopt);
    ASSERT_TRUE(mismatch) << mismatch.error().detail();
    EXPECT_FALSE(mismatch->metrics.valid);
    for (const auto& view : mismatch->metrics.views)
        EXPECT_NE(view.skipped_reason.find("but its reference is"), std::string::npos) << view.skipped_reason;
}
