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
            {"args", nlohmann::json::array({"--strategy", "mrnf", "--iter", "30000", "--export", "ply,spz", "--use-depth-loss"})},
            {"jobs", {{{"dataset", (temp.path() / "scene_a").string()}, {"name", "custom_scene_a"}}, {{"dataset", (temp.path() / "scene_b").string()}, {"args", nlohmann::json::array({"--strategy", "mcmc", "--iter", "45000"})}, {"output", (temp.path() / "custom_out_b").string()}}}}};

        {
            std::ofstream out(config_path);
            out << j.dump(2);
        }

        auto res = lfs::core::param::read_batch_training_parameters_from_json(config_path);
        ASSERT_TRUE(res.has_value()) << res.error().detail();

        const auto& params = *res;
        EXPECT_EQ(params.output_dir, temp.path() / "outputs");
        EXPECT_FALSE(params.continue_on_failure);
        EXPECT_TRUE(params.recursive);
        EXPECT_TRUE(params.dry_run);
        EXPECT_EQ(params.timeout_minutes, 45);
        EXPECT_EQ(params.shared_args, (std::vector<std::string>{"--strategy", "mrnf", "--iter", "30000", "--export", "ply,spz", "--use-depth-loss"}));

        ASSERT_EQ(params.jobs.size(), 2u);
        EXPECT_EQ(params.jobs[0].dataset_path, temp.path() / "scene_a");
        EXPECT_EQ(params.jobs[0].name, "custom_scene_a");
        EXPECT_TRUE(params.jobs[0].extra_args.empty());

        EXPECT_EQ(params.jobs[1].dataset_path, temp.path() / "scene_b");
        EXPECT_EQ(params.jobs[1].output_path, temp.path() / "custom_out_b");
        EXPECT_EQ(params.jobs[1].extra_args, (std::vector<std::string>{"--strategy", "mcmc", "--iter", "45000"}));
    }

    TEST(BatchTrainerTest, ReadBatchConfigRejectsUnknownKeysAndWrongTypes) {
        TempDirectoryGuard temp;
        const auto config_path = temp.path() / "batch.json";
        const auto detail_for = [&](const nlohmann::json& j) {
            std::ofstream(config_path) << j.dump();
            auto res = lfs::core::param::read_batch_training_parameters_from_json(config_path);
            return res ? std::string("accepted") : std::string(res.error().detail());
        };

        EXPECT_NE(detail_for({{"jobs", {{{"dataset", "scene"}, {"strategy", "mcmc"}}}}}).find("\"strategy\""), std::string::npos);
        EXPECT_NE(detail_for({{"defaults", {{"iterations", 30000}}}}).find("\"defaults\""), std::string::npos);
        EXPECT_NE(detail_for({{"jobs", {{{"dataset", 5}}}}}).find("\"dataset\""), std::string::npos);
        EXPECT_NE(detail_for({{"jobs", {{{"dataset", "scene"}, {"args", nlohmann::json::array({"--iter", 30000})}}}}}).find("\"args\""), std::string::npos);
        EXPECT_NE(detail_for({{"timeout_minutes", -1}}).find("\"timeout_minutes\""), std::string::npos);
        EXPECT_NE(detail_for({{"jobs", {{{"name", "no_dataset"}}}}}).find("\"dataset\""), std::string::npos);
    }

    TEST(BatchTrainerTest, PositionalManifestArgsApplyToItsJobs) {
        TempDirectoryGuard temp;
        const auto scene = temp.path() / "scene";
        fs::create_directories(scene);
        std::ofstream(scene / "transforms.json") << "{}";
        const auto manifest = temp.path() / "batch.json";
        std::ofstream(manifest) << nlohmann::json{
            {"args", nlohmann::json::array({"--iter", "7000"})},
            {"jobs", {{{"dataset", scene.string()}, {"args", nlohmann::json::array({"--strategy", "mcmc"})}}}}}
                                       .dump();

        lfs::core::param::BatchTrainingParameters params;
        params.input_paths = {manifest};
        params.output_dir = temp.path() / "outputs";
        const auto jobs = lfs::app::resolve_batch_jobs(params);

        ASSERT_EQ(jobs.size(), 1u);
        EXPECT_EQ(jobs[0].extra_args, (std::vector<std::string>{"--iter", "7000", "--strategy", "mcmc"}));
    }

    TEST(BatchTrainerTest, ReadBatchConfigArray) {
        TempDirectoryGuard temp;
        const auto config_path = temp.path() / "batch_array.json";

        nlohmann::json j = nlohmann::json::array({{{"dataset", (temp.path() / "scene_1").string()},
                                                   {"output", (temp.path() / "out_1").string()}},
                                                  {{"dataset", (temp.path() / "scene_2").string()}}});

        {
            std::ofstream out(config_path);
            out << j.dump(2);
        }

        auto res = lfs::core::param::read_batch_training_parameters_from_json(config_path);
        ASSERT_TRUE(res.has_value()) << res.error().detail();

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
            "--use-depth-loss"};

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
            "./outputs"};

        auto res = lfs::core::args::parse_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(res.has_value()) << res.error();
        EXPECT_TRUE(std::holds_alternative<lfs::core::args::BatchTrainingMode>(*res));
    }

    TEST(BatchTrainerTest, ParseMissingInputsFails) {
        std::vector<const char*> argv = {
            "lichtfeld-studio",
            "batch-train"};

        auto res = lfs::core::args::parse_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_FALSE(res.has_value());
    }

    TEST(BatchTrainerTest, ParseBatchTrainNoDownloadFlag) {
        std::vector<const char*> argv = {
            "lichtfeld-studio",
            "batch-train",
            "--no-download",
            "./my_datasets"};

        auto res = lfs::core::args::parse_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(res.has_value()) << res.error();
        ASSERT_TRUE(std::holds_alternative<lfs::core::args::BatchTrainingMode>(*res));
        const auto& batch_mode = std::get<lfs::core::args::BatchTrainingMode>(*res);
        ASSERT_EQ(batch_mode.params.input_paths.size(), 1u);
        EXPECT_EQ(batch_mode.params.input_paths[0], fs::path("./my_datasets"));
        EXPECT_NE(std::find(batch_mode.params.shared_args.begin(), batch_mode.params.shared_args.end(), "--no-download"),
                  batch_mode.params.shared_args.end());
    }

    TEST(BatchTrainerTest, ParseBatchTrainTakesFlagArityFromTrainingParser) {
        std::vector<const char*> argv = {
            "lichtfeld-studio",
            "batch-train",
            "--no-cpu-cache",
            "./first",
            "--images",
            "images_4",
            "-r",
            "4",
            "./second"};

        auto res = lfs::core::args::parse_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(res.has_value()) << res.error();
        const auto& params = std::get<lfs::core::args::BatchTrainingMode>(*res).params;
        EXPECT_EQ(params.input_paths, (std::vector<fs::path>{"./first", "./second"}));
        EXPECT_EQ(params.shared_args, (std::vector<std::string>{"--no-cpu-cache", "--images", "images_4", "-r", "4"}));
        EXPECT_FALSE(params.recursive); // -r is the training resize factor
    }

    TEST(BatchTrainerTest, EveryTrainingFlagReachesTheJobs) {
        const auto flags = lfs::core::args::training_flags();
        ASSERT_GT(flags.size(), 100u);
        for (const auto& flag : flags) {
            if (flag.name == "-h" || flag.name == "--help")
                continue;
            const std::string value = flag.name == "--log-level" ? "info" : "value";
            std::vector<const char*> argv = {"lichtfeld-studio", "batch-train", "./first", flag.name.c_str()};
            if (flag.takes_value)
                argv.push_back(value.c_str());
            argv.push_back("./second");

            auto res = lfs::core::args::parse_args(static_cast<int>(argv.size()), argv.data());
            ASSERT_TRUE(res.has_value()) << flag.name << ": " << res.error();
            const auto& params = std::get<lfs::core::args::BatchTrainingMode>(*res).params;
            if (flag.name == "-d" || flag.name == "--data-path") {
                EXPECT_EQ(params.input_paths, (std::vector<fs::path>{"./first", value, "./second"})) << flag.name;
                continue;
            }
            EXPECT_EQ(params.input_paths, (std::vector<fs::path>{"./first", "./second"})) << flag.name;
            if (flag.name == "-o" || flag.name == "--output-path") {
                EXPECT_EQ(params.output_dir, fs::path(value)) << flag.name;
                continue;
            }
            auto forwarded = std::vector<std::string>{flag.name};
            if (flag.takes_value)
                forwarded.push_back(value);
            EXPECT_EQ(params.shared_args, forwarded) << flag.name;
        }
    }

    TEST(BatchTrainerTest, JobConfigAndArgsComeAfterSharedArgs) {
        lfs::core::param::BatchJobItem job;
        job.dataset_path = "scene";
        job.output_path = "out";
        job.config_file = fs::path("job.json");
        job.extra_args = {"--iter", "7000"};
        lfs::core::param::BatchTrainingParameters params;
        params.shared_args = {"--config", "shared.json", "--iter", "30000"};

        EXPECT_EQ(lfs::app::build_child_args(job, params),
                  (std::vector<std::string>{"--headless", "-d", "scene", "-o", "out", "--config", "shared.json", "--iter", "30000",
                                            "--config", "job.json", "--iter", "7000"}));
    }

    TEST(BatchTrainerTest, ParseBatchTrainFreezeFlag) {
        std::vector<const char*> argv = {
            "lichtfeld-studio",
            "batch-train",
            "--add-splat",
            "model.ply",
            "--freeze",
            "./my_datasets"};

        auto res = lfs::core::args::parse_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(res.has_value()) << res.error();
        ASSERT_TRUE(std::holds_alternative<lfs::core::args::BatchTrainingMode>(*res));
        const auto& batch_mode = std::get<lfs::core::args::BatchTrainingMode>(*res);
        ASSERT_EQ(batch_mode.params.input_paths.size(), 1u);
        EXPECT_EQ(batch_mode.params.input_paths[0], fs::path("./my_datasets"));
        EXPECT_NE(std::find(batch_mode.params.shared_args.begin(), batch_mode.params.shared_args.end(), "--freeze"),
                  batch_mode.params.shared_args.end());
    }

    TEST(BatchTrainerTest, ValidateJobArgumentsCatchesInvalidOption) {
        lfs::core::param::BatchJobItem job;
        job.name = "test_job";
        job.dataset_path = "./my_datasets";
        job.output_path = "./my_outputs";

        lfs::core::param::BatchTrainingParameters params;
        params.shared_args = {"--iters", "300"};

        std::string error;
        EXPECT_FALSE(lfs::app::validate_job_arguments(job, params, error));
    }

    TEST(BatchTrainerTest, ValidateJobArgumentsAcceptsSharedEvalStepsWithPerJobEval) {
        lfs::core::param::BatchJobItem job;
        job.name = "test_job";
        job.dataset_path = "./my_datasets";
        job.output_path = "./my_outputs";
        job.extra_args = {"--eval"};

        lfs::core::param::BatchTrainingParameters params;
        params.shared_args = {"--eval-steps", "5000"};

        std::string error;
        EXPECT_TRUE(lfs::app::validate_job_arguments(job, params, error)) << error;
    }

    TEST(BatchTrainerTest, ResolveBatchJobsReservesExplicitNames) {
        TempDirectoryGuard temp;
        const auto site_a = temp.path() / "siteA" / "scene";
        const auto site_b = temp.path() / "siteB" / "scene";
        const auto site_c = temp.path() / "siteC" / "scene_2";
        fs::create_directories(site_a);
        fs::create_directories(site_b);
        fs::create_directories(site_c);
        std::ofstream(site_a / "transforms.json") << "{}";
        std::ofstream(site_b / "transforms.json") << "{}";
        std::ofstream(site_c / "transforms.json") << "{}";

        lfs::core::param::BatchTrainingParameters params;
        // Three jobs: two default names (will be "scene"), one explicit "scene_2"
        lfs::core::param::BatchJobItem j1;
        j1.dataset_path = site_a;
        lfs::core::param::BatchJobItem j2;
        j2.dataset_path = site_b;
        lfs::core::param::BatchJobItem j3;
        j3.dataset_path = site_c;
        j3.name = "scene_2";

        params.jobs = {j1, j2, j3};
        params.output_dir = temp.path() / "outputs";

        const auto jobs = lfs::app::resolve_batch_jobs(params);
        ASSERT_EQ(jobs.size(), 3u);
        EXPECT_NE(jobs[0].name, jobs[1].name);
        EXPECT_NE(jobs[0].name, jobs[2].name);
        EXPECT_NE(jobs[1].name, jobs[2].name);
        EXPECT_EQ(jobs[2].name, "scene_2");
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

    TEST(BatchTrainerTest, ValidateJobArgumentsRejectsHelpWithoutExiting) {
        lfs::core::param::BatchJobItem job;
        job.dataset_path = "/tmp/fake_dataset";
        job.output_path = "/tmp/fake_output";
        job.extra_args = {"--help"};

        lfs::core::param::BatchTrainingParameters params;
        std::string error;
        const bool valid = lfs::app::validate_job_arguments(job, params, error);
        EXPECT_FALSE(valid);
        EXPECT_FALSE(error.empty());
    }

    TEST(BatchTrainerTest, ValidateJobArgumentsDoesNotCreateLogFile) {
        TempDirectoryGuard temp;
        const auto log_path = temp.path() / "test_validate_no_side_effects.log";
        ASSERT_FALSE(fs::exists(log_path));

        lfs::core::param::BatchJobItem job;
        job.dataset_path = "/tmp/fake_dataset";
        job.output_path = temp.path() / "outputs";
        job.extra_args = {"--log-file", log_path.string()};

        lfs::core::param::BatchTrainingParameters params;
        std::string error;
        const bool valid = lfs::app::validate_job_arguments(job, params, error);

        EXPECT_TRUE(valid);
        EXPECT_FALSE(fs::exists(log_path));
        EXPECT_FALSE(fs::exists(job.output_path));
    }

    TEST(BatchTimeoutTest, OneMinuteUsesSingular) {
        EXPECT_EQ(lfs::app::batch_timeout_message(1), "Job timed out after 1 minute");
    }

    TEST(BatchTimeoutTest, OtherDurationsUsePlural) {
        EXPECT_EQ(lfs::app::batch_timeout_message(2), "Job timed out after 2 minutes");
        EXPECT_EQ(lfs::app::batch_timeout_message(60), "Job timed out after 60 minutes");
    }

} // namespace
