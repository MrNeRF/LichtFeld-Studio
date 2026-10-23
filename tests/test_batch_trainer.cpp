/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/batch_trainer.hpp"
#include "core/argument_parser.hpp"
#include "core/parameters.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {

    namespace fs = std::filesystem;

    class TempDirectoryGuard {
    public:
        TempDirectoryGuard() {
            path_ = fs::temp_directory_path() / ("lfs_batch_test_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
            fs::create_directories(path_);
        }

        ~TempDirectoryGuard() {
            std::error_code ec;
            fs::remove_all(path_, ec);
        }

        const fs::path& path() const { return path_; }

    private:
        fs::path path_;
    };

    TEST(BatchTrainerTest, ReadBatchConfigObject) {
        TempDirectoryGuard temp;
        const auto config_path = temp.path() / "batch.json";

        nlohmann::json j = {
            {"output_dir", (temp.path() / "outputs").string()},
            {"continue_on_failure", false},
            {"recursive", true},
            {"dry_run", true},
            {"timeout_minutes", 45},
            {"defaults", {
                {"strategy", "mrnf"},
                {"iterations", 30000},
                {"export", "ply,spz"},
                {"use_depth_loss", true}
            }},
            {"jobs", {
                {
                    {"dataset", (temp.path() / "scene_a").string()},
                    {"name", "custom_scene_a"}
                },
                {
                    {"dataset", (temp.path() / "scene_b").string()},
                    {"strategy", "mcmc"},
                    {"iter", 45000},
                    {"output", (temp.path() / "custom_out_b").string()}
                }
            }}
        };

        {
            std::ofstream out(config_path);
            out << j.dump(2);
        }

        auto res = lfs::core::param::read_batch_training_parameters_from_json(config_path);
        ASSERT_TRUE(res.has_value()) << res.error().user_message();

        const auto& params = *res;
        EXPECT_EQ(params.output_dir, temp.path() / "outputs");
        EXPECT_FALSE(params.continue_on_failure);
        EXPECT_TRUE(params.recursive);
        EXPECT_TRUE(params.dry_run);
        EXPECT_EQ(params.timeout_minutes, 45);

        // Check defaults mapped to shared_args
        EXPECT_NE(std::find(params.shared_args.begin(), params.shared_args.end(), "--strategy"), params.shared_args.end());
        EXPECT_NE(std::find(params.shared_args.begin(), params.shared_args.end(), "mrnf"), params.shared_args.end());
        EXPECT_NE(std::find(params.shared_args.begin(), params.shared_args.end(), "--iter"), params.shared_args.end());
        EXPECT_NE(std::find(params.shared_args.begin(), params.shared_args.end(), "30000"), params.shared_args.end());
        EXPECT_NE(std::find(params.shared_args.begin(), params.shared_args.end(), "--export"), params.shared_args.end());
        EXPECT_NE(std::find(params.shared_args.begin(), params.shared_args.end(), "ply,spz"), params.shared_args.end());
        EXPECT_NE(std::find(params.shared_args.begin(), params.shared_args.end(), "--use-depth-loss"), params.shared_args.end());

        // Check jobs
        ASSERT_EQ(params.jobs.size(), 2u);
        EXPECT_EQ(params.jobs[0].dataset_path, temp.path() / "scene_a");
        EXPECT_EQ(params.jobs[0].name, "custom_scene_a");

        EXPECT_EQ(params.jobs[1].dataset_path, temp.path() / "scene_b");
        EXPECT_EQ(params.jobs[1].output_path, temp.path() / "custom_out_b");
        EXPECT_NE(std::find(params.jobs[1].extra_args.begin(), params.jobs[1].extra_args.end(), "--strategy"), params.jobs[1].extra_args.end());
        EXPECT_NE(std::find(params.jobs[1].extra_args.begin(), params.jobs[1].extra_args.end(), "mcmc"), params.jobs[1].extra_args.end());
        EXPECT_NE(std::find(params.jobs[1].extra_args.begin(), params.jobs[1].extra_args.end(), "--iter"), params.jobs[1].extra_args.end());
        EXPECT_NE(std::find(params.jobs[1].extra_args.begin(), params.jobs[1].extra_args.end(), "45000"), params.jobs[1].extra_args.end());
    }

    TEST(BatchTrainerTest, ReadBatchConfigArray) {
        TempDirectoryGuard temp;
        const auto config_path = temp.path() / "batch_array.json";

        nlohmann::json j = nlohmann::json::array({
            {
                {"dataset", (temp.path() / "scene_1").string()},
                {"output", (temp.path() / "out_1").string()}
            },
            {
                {"path", (temp.path() / "scene_2").string()}
            }
        });

        {
            std::ofstream out(config_path);
            out << j.dump(2);
        }

        auto res = lfs::core::param::read_batch_training_parameters_from_json(config_path);
        ASSERT_TRUE(res.has_value()) << res.error().user_message();

        const auto& params = *res;
        ASSERT_EQ(params.jobs.size(), 2u);
        EXPECT_EQ(params.jobs[0].dataset_path, temp.path() / "scene_1");
        EXPECT_EQ(params.jobs[0].output_path, temp.path() / "out_1");
        EXPECT_EQ(params.jobs[1].dataset_path, temp.path() / "scene_2");
    }

    TEST(BatchTrainerTest, ParseBatchTrainArgsCommandLine) {
        std::vector<const char*> argv = {
            "lichtfeld-studio",
            "batch-train",
            "./my_datasets",
            "-o",
            "./my_outputs",
            "--recursive",
            "--stop-on-failure",
            "--dry-run",
            "--timeout",
            "120",
            "--strategy",
            "mrnf",
            "--iter",
            "20000",
            "--export",
            "ply,spz",
            "--use-depth-loss"
        };

        auto res = lfs::core::args::parse_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(res.has_value()) << res.error();

        ASSERT_TRUE(std::holds_alternative<lfs::core::args::BatchTrainingMode>(*res));
        const auto& batch_mode = std::get<lfs::core::args::BatchTrainingMode>(*res);
        const auto& params = batch_mode.params;

        ASSERT_EQ(params.input_paths.size(), 1u);
        EXPECT_EQ(params.input_paths[0], fs::path("./my_datasets"));
        EXPECT_EQ(params.output_dir, fs::path("./my_outputs"));
        EXPECT_TRUE(params.recursive);
        EXPECT_FALSE(params.continue_on_failure); // --stop-on-failure
        EXPECT_TRUE(params.dry_run);
        EXPECT_EQ(params.timeout_minutes, 120);

        // Verify training options were forwarded into shared_args
        EXPECT_NE(std::find(params.shared_args.begin(), params.shared_args.end(), "--strategy"), params.shared_args.end());
        EXPECT_NE(std::find(params.shared_args.begin(), params.shared_args.end(), "mrnf"), params.shared_args.end());
        EXPECT_NE(std::find(params.shared_args.begin(), params.shared_args.end(), "--iter"), params.shared_args.end());
        EXPECT_NE(std::find(params.shared_args.begin(), params.shared_args.end(), "20000"), params.shared_args.end());
        EXPECT_NE(std::find(params.shared_args.begin(), params.shared_args.end(), "--export"), params.shared_args.end());
        EXPECT_NE(std::find(params.shared_args.begin(), params.shared_args.end(), "ply,spz"), params.shared_args.end());
        EXPECT_NE(std::find(params.shared_args.begin(), params.shared_args.end(), "--use-depth-loss"), params.shared_args.end());
    }

    TEST(BatchTrainerTest, ParseBatchAlias) {
        std::vector<const char*> argv = {
            "lichtfeld-studio",
            "batch",
            "./scene_folder",
            "-o",
            "./outputs"
        };

        auto res = lfs::core::args::parse_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(res.has_value()) << res.error();
        EXPECT_TRUE(std::holds_alternative<lfs::core::args::BatchTrainingMode>(*res));
    }

    TEST(BatchTrainerTest, ParseMissingInputsFails) {
        std::vector<const char*> argv = {
            "lichtfeld-studio",
            "batch-train"
        };

        auto res = lfs::core::args::parse_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_FALSE(res.has_value());
    }

    TEST(BatchTrainerTest, ParseBatchTrainNoDownloadFlag) {
        std::vector<const char*> argv = {
            "lichtfeld-studio",
            "batch-train",
            "--no-download",
            "./my_datasets"
        };

        auto res = lfs::core::args::parse_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(res.has_value()) << res.error();
        ASSERT_TRUE(std::holds_alternative<lfs::core::args::BatchTrainingMode>(*res));
        const auto& batch_mode = std::get<lfs::core::args::BatchTrainingMode>(*res);
        ASSERT_EQ(batch_mode.params.input_paths.size(), 1u);
        EXPECT_EQ(batch_mode.params.input_paths[0], fs::path("./my_datasets"));
        EXPECT_NE(std::find(batch_mode.params.shared_args.begin(), batch_mode.params.shared_args.end(), "--no-download"),
                  batch_mode.params.shared_args.end());
    }

    TEST(BatchTrainerTest, ParseBatchTrainInvalidOptionFails) {
        std::vector<const char*> argv = {
            "lichtfeld-studio",
            "batch-train",
            "./my_datasets",
            "--iters",
            "300"
        };

        auto res = lfs::core::args::parse_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_FALSE(res.has_value());
    }

    TEST(BatchTrainerTest, ResolveBatchJobsDisambiguatesDuplicateNames) {
        TempDirectoryGuard temp;
        const auto site_a = temp.path() / "siteA" / "scene";
        const auto site_b = temp.path() / "siteB" / "scene";
        fs::create_directories(site_a);
        fs::create_directories(site_b);
        std::ofstream(site_a / "transforms.json") << "{}";
        std::ofstream(site_b / "transforms.json") << "{}";

        lfs::core::param::BatchTrainingParameters params;
        params.input_paths = {site_a, site_b};
        params.output_dir = temp.path() / "outputs";

        const auto jobs = lfs::app::resolve_batch_jobs(params);
        ASSERT_EQ(jobs.size(), 2u);
        EXPECT_NE(jobs[0].name, jobs[1].name);
        EXPECT_NE(jobs[0].output_path, jobs[1].output_path);

        EXPECT_EQ(jobs[0].name, "siteA_scene");
        EXPECT_EQ(jobs[1].name, "siteB_scene");
        EXPECT_EQ(jobs[0].output_path, temp.path() / "outputs" / "siteA_scene");
        EXPECT_EQ(jobs[1].output_path, temp.path() / "outputs" / "siteB_scene");
    }

    TEST(BatchTrainerTest, ResolveBatchJobsDeterministicSort) {
        TempDirectoryGuard temp;
        const auto dir_room = temp.path() / "datasets" / "room";
        const auto dir_kitchen = temp.path() / "datasets" / "kitchen";
        const auto dir_bicycle = temp.path() / "datasets" / "bicycle";
        fs::create_directories(dir_room);
        fs::create_directories(dir_kitchen);
        fs::create_directories(dir_bicycle);
        std::ofstream(dir_room / "transforms.json") << "{}";
        std::ofstream(dir_kitchen / "transforms.json") << "{}";
        std::ofstream(dir_bicycle / "transforms.json") << "{}";

        lfs::core::param::BatchTrainingParameters params;
        params.input_paths = {temp.path() / "datasets"};
        params.recursive = false;

        const auto jobs = lfs::app::resolve_batch_jobs(params);
        ASSERT_EQ(jobs.size(), 3u);
        EXPECT_EQ(jobs[0].name, "bicycle");
        EXPECT_EQ(jobs[1].name, "kitchen");
        EXPECT_EQ(jobs[2].name, "room");
    }

} // namespace
