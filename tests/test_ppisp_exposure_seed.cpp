/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "components/holdout_appearance.hpp"
#include "components/ppisp.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "io/argument_parser.hpp"

#include "io/project_document.hpp"
#include "licht_test_support.hpp"
#include "training/trainer.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

    using lfs::core::Device;
    using lfs::training::PPISP;
    using lfs::training::PPISPConfig;

    std::vector<float> exposure_host(const PPISP& ppisp) {
        return ppisp.exposure_params().cpu().contiguous().to_vector();
    }

} // namespace

class PPISPExposureSeedTest : public lfs::test::CudaBackendTest {};

TEST_F(PPISPExposureSeedTest, CentersAndScalesKnownFrames) {
    PPISP ppisp(100);
    ppisp.register_frame(10, 0);
    ppisp.register_frame(20, 0);
    ppisp.register_frame(30, 0);
    ppisp.register_frame(40, 0);
    ppisp.finalize();

    const std::vector<std::pair<int, float>> uid_ev{{10, 0.0f}, {20, 2.0f}, {40, 4.0f}};
    ppisp.seed_exposure(uid_ev);

    const auto values = exposure_host(ppisp);
    ASSERT_EQ(values.size(), 4u);
    const float mean = (0.0f + 2.0f + 4.0f) / 3.0f;
    EXPECT_NEAR(values[0], 0.5f * (0.0f - mean), 1e-6f);
    EXPECT_NEAR(values[1], 0.5f * (2.0f - mean), 1e-6f);
    EXPECT_NEAR(values[2], 0.0f, 1e-6f);
    EXPECT_NEAR(values[3], 0.5f * (4.0f - mean), 1e-6f);
}

TEST_F(PPISPExposureSeedTest, CopyInferenceWeightsOverwritesSeed) {
    PPISPConfig config;
    config.warmup_steps = 0;

    PPISP seeded(100, config);
    seeded.register_frame(1, 0);
    seeded.register_frame(2, 0);
    seeded.register_frame(3, 0);
    seeded.register_frame(4, 0);
    seeded.finalize();
    seeded.seed_exposure({{1, -2.0f}, {2, 0.0f}, {3, 2.0f}});

    PPISP blank(100, config);
    blank.register_frame(1, 0);
    blank.register_frame(2, 0);
    blank.register_frame(3, 0);
    blank.register_frame(4, 0);
    blank.finalize();

    const auto import = seeded.copy_inference_weights_from(blank, {0, 1, 2, 3}, {0});
    ASSERT_TRUE(import) << import.error();

    const auto values = exposure_host(seeded);
    ASSERT_EQ(values.size(), 4u);
    for (float value : values) {
        EXPECT_NEAR(value, 0.0f, 1e-6f);
    }
}

class PPISPApplyWithExposureTest : public lfs::test::CudaBackendTest {};

TEST_F(PPISPApplyWithExposureTest, MatchesApplyForRegisteredFrame) {
    PPISP ppisp(100);
    ppisp.register_frame(10, 7);
    ppisp.register_frame(20, 7);
    ppisp.register_frame(30, 7);
    ppisp.finalize();
    ppisp.seed_exposure({{10, 0.0f}, {20, 2.0f}, {30, 4.0f}});

    const auto values = exposure_host(ppisp);
    ASSERT_EQ(values.size(), 3u);
    const float e = values[0];
    ASSERT_NE(e, 0.0f);

    std::vector<float> pixels(3 * 4 * 4);
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = 0.2f + 0.01f * static_cast<float>(i);
    }
    const auto rgb = lfs::core::Tensor::from_vector(pixels, {3, 4, 4}, Device::GPU);

    const auto from_apply = ppisp.apply(rgb, 7, 10).cpu().contiguous().to_vector();
    const auto from_explicit = ppisp.apply_with_exposure(rgb, 7, e).cpu().contiguous().to_vector();
    ASSERT_EQ(from_apply.size(), from_explicit.size());
    EXPECT_EQ(std::memcmp(from_apply.data(), from_explicit.data(), from_apply.size() * sizeof(float)), 0);

    const auto other = ppisp.apply_with_exposure(rgb, 7, e + 1.0f).cpu().contiguous().to_vector();
    EXPECT_NE(std::memcmp(from_apply.data(), other.data(), from_apply.size() * sizeof(float)), 0);
}

TEST_F(PPISPExposureSeedTest, ClampsToExposureRange) {
    PPISP ppisp(100);
    ppisp.register_frame(1, 0);
    ppisp.register_frame(2, 0);
    ppisp.finalize();
    ppisp.seed_exposure({{1, 0.0f}, {2, 100.0f}});

    const auto values = exposure_host(ppisp);
    ASSERT_EQ(values.size(), 2u);
    EXPECT_GE(values[0], -16.0f);
    EXPECT_LE(values[0], 16.0f);
    EXPECT_GE(values[1], -16.0f);
    EXPECT_LE(values[1], 16.0f);
    EXPECT_FLOAT_EQ(values[1], 16.0f);
}

TEST(PPISPHoldoutAppearanceTest, InterpolationMatchesExplicitParametersOnRealCrop) {
    const char* image_path = std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE");
    if (!image_path)
        GTEST_SKIP() << "set real-image path";
    const auto [bytes, width, height, channels] = lfs::core::load_image(image_path);
    ASSERT_NE(bytes, nullptr);
    ASSERT_GE(width, 16);
    ASSERT_GE(height, 16);
    ASSERT_EQ(channels, 3);
    std::vector<float> pixels(3 * 16 * 16);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 16; ++x)
                pixels[c * 256 + y * 16 + x] = bytes[(y * width + x) * 3 + c] / 255.0f;
    lfs::core::free_image(bytes);
    auto rgb = lfs::core::Tensor::from_vector(pixels, {3, 16, 16}, Device::GPU);
    PPISP ppisp(100);
    ppisp.register_frame(10, 7);
    ppisp.register_frame(20, 9);
    ppisp.finalize();
    ppisp.exposure_params().copy_from(lfs::core::Tensor::from_vector({-0.4f, 0.6f}, {2}, Device::GPU));
    std::vector<float> color(16);
    for (int i = 0; i < 16; ++i)
        color[i] = (i - 8) * 0.002f;
    ppisp.color_params().copy_from(lfs::core::Tensor::from_vector(color, {16}, Device::GPU));
    for (float t : {0.0f, 0.25f, 0.5f, 1.0f}) {
        std::vector<float> values(9);
        values[0] = -0.4f * (1.0f - t) + 0.6f * t;
        for (int i = 0; i < 8; ++i)
            values[i + 1] = color[i] * (1.0f - t) + color[i + 8] * t;
        const int camera_id = t <= 0.5f ? 7 : 9;
        auto explicit_params = lfs::core::Tensor::from_vector(values, {1, 9}, Device::GPU);
        auto expected = ppisp.apply_with_controller_params(rgb, explicit_params, ppisp.camera_index(camera_id)).cpu().to_vector();
        auto actual = ppisp.apply_interpolated_frames(rgb, camera_id, 10, 20, t).cpu().to_vector();
        ASSERT_EQ(expected.size(), actual.size());
        for (size_t i = 0; i < actual.size(); ++i)
            EXPECT_NEAR(actual[i], expected[i], 2e-6f);
    }
    auto direct = ppisp.apply(rgb, 7, 10).cpu().to_vector();
    auto endpoint = ppisp.apply_interpolated_frames(rgb, 7, 10, 10, 0.0f).cpu().to_vector();
    EXPECT_EQ(std::memcmp(direct.data(), endpoint.data(), direct.size() * sizeof(float)), 0);
}

TEST(PPISPHoldoutAppearanceTest, RealCaptureOrderBoundariesAndKnownFrames) {
    const auto* path = std::getenv("LFS_SPLIT_TEST_MODEL");
    if (!path)
        GTEST_SKIP() << "set real-model path";
    using namespace lfs::training;
    using namespace lfs::io::project;
    using namespace lfs::test::licht;
    lfs::core::Scene scene;
    auto document = require_result(ProjectDocument::open(path));
    static_cast<void>(require_result(document.hydrate(scene)));
    auto cameras = scene.getAllCameras();
    std::sort(cameras.begin(), cameras.end(), [](const auto& a, const auto& b) { return a->image_name() < b->image_name(); });
    ASSERT_GE(cameras.size(), 7u);
    std::vector<HoldoutAppearanceFrame> frames;
    for (size_t i = 0; i < 7; ++i)
        frames.push_back({cameras[i].get(), i == 1 || i == 5});
    const auto middle = select_holdout_appearance(frames, *cameras[3]);
    ASSERT_TRUE(middle);
    EXPECT_EQ(middle->left_uid, cameras[1]->uid());
    EXPECT_EQ(middle->right_uid, cameras[5]->uid());
    EXPECT_EQ(middle->camera_id, cameras[1]->camera_id());
    EXPECT_EQ(middle->fraction, .5f);
    EXPECT_EQ(select_holdout_appearance(frames, *cameras[2])->fraction, .25f);
    EXPECT_EQ(select_holdout_appearance(frames, *cameras[4])->camera_id, cameras[5]->camera_id());
    EXPECT_EQ(select_holdout_appearance(frames, *cameras[0])->left_uid, cameras[1]->uid());
    EXPECT_EQ(select_holdout_appearance(frames, *cameras[6])->right_uid, cameras[5]->uid());
    EXPECT_FALSE(select_holdout_appearance(frames, *cameras[1]));
    EXPECT_FALSE(select_holdout_appearance({{cameras[0].get(), false}}, *cameras[0]));
}

TEST(PPISPHoldoutAppearanceTest, DefaultsJsonCliAndProjectRoundTrip) {
    const auto* output = std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT");
    if (!output || !std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE"))
        GTEST_SKIP() << "set test-output path";
    using namespace lfs::core::param;
    for (const auto* strategy : {"mrnf", "mcmc", "igs+"}) {
        auto params = OptimizationParameters::defaults_for_strategy(strategy);
        EXPECT_EQ(params.ppisp_holdout_appearance, PPISPHoldoutAppearance::Nearest);
        for (const auto mode : {PPISPHoldoutAppearance::Mean, PPISPHoldoutAppearance::Nearest}) {
            params.ppisp_holdout_appearance = mode;
            EXPECT_EQ(OptimizationParameters::from_json(params.to_json()).ppisp_holdout_appearance, mode);
            EXPECT_TRUE(params.validate().empty());
        }
        params.ppisp_holdout_appearance = static_cast<PPISPHoldoutAppearance>(-1);
        EXPECT_FALSE(params.validate().empty());
    }
    const auto data = std::filesystem::path(std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE")).parent_path().parent_path().string();
    struct RestoreLogger {
        lfs::core::LogLevel level = lfs::core::Logger::get().level();
        ~RestoreLogger() { lfs::core::Logger::get().init(level); }
    } restore_logger;
    for (const auto* mode : {"mean", "nearest", "invalid"}) {
        std::vector<std::string> args{"test", "--headless", "--train", "--data-path", data, "--output-path", output, "--log-file", "/dev/null", "--ppisp-holdout-appearance", mode};
        std::vector<const char*> argv;
        for (const auto& value : args)
            argv.push_back(value.c_str());
        const auto parsed = lfs::io::args::parse_args_and_params(static_cast<int>(argv.size()), argv.data());
        if (std::string_view(mode) == "invalid") {
            EXPECT_FALSE(parsed);
            continue;
        }
        ASSERT_TRUE(parsed) << parsed.error();
        TrainingParameters target;
        lfs::core::param::apply_explicit_training_overrides(target, (*parsed)->overrides);
        EXPECT_EQ(target.optimization.ppisp_holdout_appearance, *ppisp_holdout_appearance_from_string(mode));
    }
    using namespace lfs::io::project;
    using namespace lfs::test::licht;
    auto document = require_result(ProjectDocument::create(fixed_uuid(4277), 1'700'000'000'000'000'000));
    auto params = require_result(document.parameters().snapshot());
    params.mrnf_current.ppisp_holdout_appearance = params.mrnf_session.ppisp_holdout_appearance = PPISPHoldoutAppearance::Mean;
    require_status(document.edit_parameters().set_snapshot(params));
    ProjectDocumentSaveOptions options;
    options.file_uuid = fixed_uuid(4278);
    options.disk_reserve_bytes = 0;
    const auto path = std::filesystem::path(output) / (lfs::core::generate_uuid_v4().to_string() + ".licht");
    std::filesystem::create_directories(path.parent_path());
    static_cast<void>(require_result(document.save(path, options)));
    auto restored = require_result(ProjectDocument::open(path));
    EXPECT_EQ(require_result(restored.parameters().snapshot()).active_optimization().ppisp_holdout_appearance, PPISPHoldoutAppearance::Mean);
    std::filesystem::remove(path);
}

namespace lfs::training {
    struct TrainerHoldoutAppearanceTestAccess {
        static PPISP& configure(Trainer& trainer, int uid, int camera_id) {
            trainer.params_.optimization.use_exposure_correction = true;
            trainer.ppisp_ = std::make_unique<PPISP>(100);
            trainer.ppisp_->register_frame(uid, camera_id);
            trainer.ppisp_->finalize();
            return *trainer.ppisp_;
        }
        static lfs::core::Tensor evaluate(Trainer& trainer, const lfs::core::Tensor& rgb,
                                          const lfs::core::Camera& camera, lfs::core::param::PPISPHoldoutAppearance mode) {
            trainer.params_.optimization.ppisp_holdout_appearance = mode;
            return trainer.applyPPISPForEval(rgb, camera);
        }
    };
} // namespace lfs::training

TEST(PPISPHoldoutAppearanceTest, MeanAndKnownFrameEvaluationAreByteIdentical) {
    const auto* image_path = std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE");
    const auto* model_path = std::getenv("LFS_SPLIT_TEST_MODEL");
    if (!image_path || !model_path)
        GTEST_SKIP() << "set real-image and real-model paths";
    using namespace lfs::training;
    using namespace lfs::io::project;
    using namespace lfs::test::licht;
    using lfs::core::param::PPISPHoldoutAppearance;
    lfs::core::Scene scene;
    auto document = require_result(ProjectDocument::open(model_path));
    static_cast<void>(require_result(document.hydrate(scene)));
    auto cameras = scene.getAllCameras();
    ASSERT_GE(cameras.size(), 2u);
    auto [data, width, height, channels] = lfs::core::load_image(image_path);
    ASSERT_NE(data, nullptr);
    const std::unique_ptr<void, decltype(&lfs::core::free_image)> owner(data, lfs::core::free_image);
    ASSERT_GE(width, 16);
    ASSERT_GE(height, 16);
    ASSERT_EQ(channels, 3);
    std::vector<float> pixels(3 * 16 * 16);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 16; ++x)
                pixels[c * 256 + y * 16 + x] = data[(y * width + x) * 3 + c] / 255.f;
    const auto rgb = lfs::core::Tensor::from_vector(pixels, {3, 16, 16}, Device::GPU);
    Trainer trainer(scene);
    auto& ppisp = TrainerHoldoutAppearanceTestAccess::configure(trainer, cameras[0]->uid(), cameras[0]->camera_id());
    ppisp.exposure_params().fill_(.35f);
    const auto known = ppisp.apply(rgb, cameras[0]->camera_id(), cameras[0]->uid()).cpu().to_vector();
    const auto old_mean = ppisp.apply_with_exposure(rgb, ppisp.majority_camera_id(), 0).cpu().to_vector();
    for (const auto mode : {PPISPHoldoutAppearance::Mean, PPISPHoldoutAppearance::Nearest}) {
        const auto actual = TrainerHoldoutAppearanceTestAccess::evaluate(trainer, rgb, *cameras[0], mode).cpu().to_vector();
        ASSERT_EQ(actual.size(), known.size());
        EXPECT_EQ(std::memcmp(actual.data(), known.data(), known.size() * sizeof(float)), 0);
    }
    const auto actual_mean = TrainerHoldoutAppearanceTestAccess::evaluate(trainer, rgb, *cameras[1], PPISPHoldoutAppearance::Mean).cpu().to_vector();
    ASSERT_EQ(actual_mean.size(), old_mean.size());
    EXPECT_EQ(std::memcmp(actual_mean.data(), old_mean.data(), old_mean.size() * sizeof(float)), 0);
}

TEST(PPISPHoldoutAppearanceTest, CaptureOrderMatchesReferenceMetadata) {
    const auto* model_path = std::getenv("LFS_SPLIT_TEST_MODEL");
    const auto* reference_path = std::getenv("LFS_HOLDOUT_APPEARANCE_REFERENCE");
    if (!model_path || !reference_path)
        GTEST_SKIP() << "set real-model and reference-metadata paths";
    using namespace lfs::training;
    using namespace lfs::io::project;
    using namespace lfs::test::licht;
    std::ifstream input(reference_path);
    ASSERT_TRUE(input);
    const auto reference = nlohmann::json::parse(input);
    const auto& queries = reference.at("heldout_neighbours");
    std::unordered_set<std::string> heldout_names;
    for (const auto& query : queries)
        heldout_names.insert(query.at("image").get<std::string>());
    lfs::core::Scene scene;
    auto document = require_result(ProjectDocument::open(model_path));
    static_cast<void>(require_result(document.hydrate(scene)));
    std::vector<HoldoutAppearanceFrame> capture;
    std::unordered_map<std::string, const lfs::core::Camera*> by_name;
    std::unordered_map<int, const lfs::core::Camera*> by_uid;
    for (const auto& camera : scene.getAllCameras()) {
        capture.push_back({camera.get(), !heldout_names.contains(camera->image_name())});
        by_name.emplace(camera->image_name(), camera.get());
        by_uid.emplace(camera->uid(), camera.get());
    }
    ASSERT_EQ(capture.size(), reference.at("capture_order_frame_parameters").size() + queries.size());
    auto selections = nlohmann::json::array();
    for (const auto& query : queries) {
        const auto name = query.at("image").get<std::string>();
        SCOPED_TRACE(name);
        ASSERT_TRUE(by_name.contains(name));
        const auto actual = select_holdout_appearance(capture, *by_name.at(name));
        ASSERT_TRUE(actual);
        EXPECT_EQ(by_uid.at(actual->left_uid)->image_name(), query.at("left").get<std::string>());
        EXPECT_EQ(by_uid.at(actual->right_uid)->image_name(), query.at("right").get<std::string>());
        EXPECT_EQ(actual->camera_id, query.at("camera_id").get<int>());
        EXPECT_FLOAT_EQ(actual->fraction, query.at("fraction").get<float>());
        selections.push_back({{"image", name}, {"left_uid", actual->left_uid}, {"right_uid", actual->right_uid}, {"camera_id", actual->camera_id}, {"fraction", actual->fraction}});
    }
    if (const auto* output_path = std::getenv("LFS_HOLDOUT_APPEARANCE_SELECTION_OUTPUT")) {
        std::ofstream output(output_path);
        ASSERT_TRUE(output);
        output << selections.dump(2) << '\n';
    }
}
