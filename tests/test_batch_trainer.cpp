/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/batch_trainer.hpp"
#include "core/parameters.hpp"
#include "io/argument_parser.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
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

    auto parse_batch(std::vector<std::string> args) {
        args.insert(args.begin(), {"lichtfeld-studio", "batch-train"});
        std::vector<const char*> argv;
        for (const auto& arg : args)
            argv.push_back(arg.c_str());
        return lfs::io::args::parse_args(static_cast<int>(argv.size()), argv.data());
    }

    auto parse_child(const lfs::core::param::BatchJobItem& job,
                     const lfs::core::param::BatchTrainingParameters& params) {
        auto args = lfs::app::build_child_args(job, params);
        args.insert(args.begin(), "lichtfeld-studio");
        std::vector<const char*> argv;
        for (const auto& arg : args)
            argv.push_back(arg.c_str());
        return lfs::io::args::parse_args_and_params(static_cast<int>(argv.size()), argv.data(), false, true);
    }

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

        auto parsed = parse_batch({manifest.string(), "-o", (temp.path() / "outputs").string()});
        ASSERT_TRUE(parsed) << parsed.error();
        const auto& params = std::get<lfs::io::args::BatchTrainingMode>(*parsed).params;
        const auto jobs = lfs::app::resolve_batch_jobs(params);

        ASSERT_EQ(jobs.size(), 1u);
        EXPECT_EQ(params.shared_args, (std::vector<std::string>{"--iter", "7000"}));
        EXPECT_EQ(jobs[0].extra_args, (std::vector<std::string>{"--strategy", "mcmc"}));
        auto child = parse_child(jobs[0], params);
        ASSERT_TRUE(child) << child.error();
        EXPECT_EQ((*child)->optimization.iterations, 7000u);
        EXPECT_EQ((*child)->optimization.strategy, "mcmc");
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

        auto res = lfs::io::args::parse_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(res.has_value()) << res.error();

        ASSERT_TRUE(std::holds_alternative<lfs::io::args::BatchTrainingMode>(*res));
        const auto& batch_mode = std::get<lfs::io::args::BatchTrainingMode>(*res);
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

        auto res = lfs::io::args::parse_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(res.has_value()) << res.error();
        EXPECT_TRUE(std::holds_alternative<lfs::io::args::BatchTrainingMode>(*res));
    }

    TEST(BatchTrainerTest, ParseMissingInputsFails) {
        std::vector<const char*> argv = {
            "lichtfeld-studio",
            "batch-train"};

        auto res = lfs::io::args::parse_args(static_cast<int>(argv.size()), argv.data());
        EXPECT_FALSE(res.has_value());
    }

    TEST(BatchTrainerTest, ParseBatchTrainNoDownloadFlag) {
        std::vector<const char*> argv = {
            "lichtfeld-studio",
            "batch-train",
            "--no-download",
            "./my_datasets"};

        auto res = lfs::io::args::parse_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(res.has_value()) << res.error();
        ASSERT_TRUE(std::holds_alternative<lfs::io::args::BatchTrainingMode>(*res));
        const auto& batch_mode = std::get<lfs::io::args::BatchTrainingMode>(*res);
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

        auto res = lfs::io::args::parse_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(res.has_value()) << res.error();
        const auto& params = std::get<lfs::io::args::BatchTrainingMode>(*res).params;
        EXPECT_EQ(params.input_paths, (std::vector<fs::path>{"./first", "./second"}));
        EXPECT_EQ(params.shared_args, (std::vector<std::string>{"--no-cpu-cache", "--images", "images_4", "-r", "4"}));
        EXPECT_FALSE(params.recursive); // -r is the training resize factor
    }

    TEST(BatchTrainerTest, EveryTrainingFlagReachesTheJobs) {
        const auto flags = lfs::io::args::training_flags();
        ASSERT_GT(flags.size(), 100u);
        for (const auto& flag : flags) {
            // The hidden backend selftest is a mode of its own, chosen before any subcommand.
            if (flag.name == "-h" || flag.name == "--help" || flag.name == "--tensor-backend-selftest")
                continue;
            const std::string value = flag.name == "--log-level" ? "info" : "value";
            std::vector<const char*> argv = {"lichtfeld-studio", "batch-train", "./first", flag.name.c_str()};
            if (flag.takes_value)
                argv.push_back(value.c_str());
            argv.push_back("./second");

            auto res = lfs::io::args::parse_args(static_cast<int>(argv.size()), argv.data());
            ASSERT_TRUE(res.has_value()) << flag.name << ": " << res.error();
            const auto& params = std::get<lfs::io::args::BatchTrainingMode>(*res).params;
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
        params.shared_args = {"--strategy", "mcmc", "--config", "shared.json", "--iter", "30000"};

        EXPECT_EQ(lfs::app::build_child_args(job, params),
                  (std::vector<std::string>{"--headless", "-d", "scene", "-o", "out", "--strategy", "mcmc", "--config", "job.json", "--iter", "7000"}));
    }

    TEST(BatchTrainerTest, ParseBatchTrainFreezeFlag) {
        std::vector<const char*> argv = {
            "lichtfeld-studio",
            "batch-train",
            "--add-splat",
            "model.ply",
            "--freeze",
            "./my_datasets"};

        auto res = lfs::io::args::parse_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(res.has_value()) << res.error();
        ASSERT_TRUE(std::holds_alternative<lfs::io::args::BatchTrainingMode>(*res));
        const auto& batch_mode = std::get<lfs::io::args::BatchTrainingMode>(*res);
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

    TEST(BatchTrainerTest, PositionalManifestAppliesAllSettingsAndCliOverrides) {
        TempDirectoryGuard temp;
        const auto scene = temp.path() / "scene";
        fs::create_directories(scene);
        const auto manifest = temp.path() / "batch.json";
        std::ofstream(manifest) << R"({"dry_run":true,"timeout_minutes":12,"continue_on_failure":false,
            "recursive":true,"output_dir":"out","summary_file":"summary.json","report_file":"report.txt",
            "args":["--iter","1000"],"jobs":[{"dataset":"scene"}]})";
        for (const bool positional : {true, false}) {
            auto args = positional ? std::vector<std::string>{manifest.string()}
                                   : std::vector<std::string>{"--manifest", manifest.string()};
            auto parsed = parse_batch(args);
            ASSERT_TRUE(parsed) << parsed.error();
            const auto& params = std::get<lfs::io::args::BatchTrainingMode>(*parsed).params;
            EXPECT_TRUE(params.dry_run);
            EXPECT_TRUE(params.recursive);
            EXPECT_FALSE(params.continue_on_failure);
            EXPECT_EQ(params.timeout_minutes, 12);
            EXPECT_EQ(params.output_dir, temp.path() / "out");
            EXPECT_EQ(params.summary_file, temp.path() / "summary.json");
            EXPECT_EQ(params.report_file, temp.path() / "report.txt");
            EXPECT_TRUE(params.input_paths.empty());
            ASSERT_EQ(params.jobs.size(), 1u);
            EXPECT_EQ(params.jobs[0].dataset_path, scene);
            EXPECT_EQ(lfs::app::run_batch_training(params), 0);
            EXPECT_FALSE(fs::exists(params.output_dir));

            args.insert(args.end(), {"--timeout", "3", "-o", "cli-out", "--summary", "cli.json", "--report", "cli.txt", "-i2000"});
            auto overridden = parse_batch(args);
            ASSERT_TRUE(overridden) << overridden.error();
            const auto& cli = std::get<lfs::io::args::BatchTrainingMode>(*overridden).params;
            EXPECT_EQ(cli.timeout_minutes, 3);
            EXPECT_EQ(cli.output_dir, fs::path("cli-out"));
            EXPECT_EQ(cli.summary_file, fs::path("cli.json"));
            EXPECT_EQ(cli.report_file, fs::path("cli.txt"));
            auto child = parse_child(lfs::app::resolve_batch_jobs(cli).front(), cli);
            ASSERT_TRUE(child) << child.error();
            EXPECT_EQ((*child)->optimization.iterations, 2000u);

            std::ofstream(manifest) << R"({"dry_run":false,"continue_on_failure":true,"recursive":false,
                "jobs":[{"dataset":"scene"}]})";
            auto switches = parse_batch({manifest.string(), "--dry-run", "--recursive", "--stop-on-failure"});
            ASSERT_TRUE(switches) << switches.error();
            const auto& switched = std::get<lfs::io::args::BatchTrainingMode>(*switches).params;
            EXPECT_TRUE(switched.dry_run);
            EXPECT_TRUE(switched.recursive);
            EXPECT_FALSE(switched.continue_on_failure);
            // Restore the settings for the second invocation spelling.
            std::ofstream(manifest) << R"({"dry_run":true,"timeout_minutes":12,"continue_on_failure":false,
                "recursive":true,"output_dir":"out","summary_file":"summary.json","report_file":"report.txt",
                "args":["--iter","1000"],"jobs":[{"dataset":"scene"}]})";
        }
    }

    TEST(BatchTrainerTest, RejectsMultipleManifestsInEveryCombination) {
        for (const auto& args : std::vector<std::vector<std::string>>{
                 {"a.json", "b.json"},
                 {"--manifest", "a.json", "b.json"},
                 {"a.json", "--manifest=b.json"},
                 {"--manifest", "a.json", "--manifest=b.json"}}) {
            auto parsed = parse_batch(args);
            ASSERT_FALSE(parsed);
            EXPECT_NE(parsed.error().find("Only one batch manifest"), std::string::npos);
        }
    }

    TEST(BatchTrainerTest, ManifestPathsResolveBesideFileRegardlessOfExistenceOrCwd) {
        TempDirectoryGuard temp;
        const auto config_dir = temp.path() / "configs";
        const auto elsewhere = temp.path() / "elsewhere";
        fs::create_directories(config_dir);
        fs::create_directories(elsewhere / "scene");
        std::ofstream(elsewhere / "train.json") << "{}";
        struct RestoreCwd {
            fs::path previous = fs::current_path();
            ~RestoreCwd() { fs::current_path(previous); }
        } restore;
        fs::current_path(elsewhere);
        const auto manifest = config_dir / "batch.json";
        for (const bool absolute : {false, true}) {
            const auto value = [&](const char* name) {
                return (absolute ? temp.path() / name : fs::path(name)).string();
            };
            std::ofstream(manifest) << nlohmann::json{
                {"output_dir", value("outputs")},
                {"summary_file", value("summary.json")},
                {"report_file", value("report.txt")},
                {"jobs", {{{"dataset", value("scene")}, {"config", value("train.json")}, {"output", value("out")}}}}}
                                           .dump();
            auto loaded = lfs::core::param::read_batch_training_parameters_from_json(fs::path("../configs/batch.json"));
            ASSERT_TRUE(loaded) << loaded.error().detail();
            const auto base = absolute ? temp.path() : config_dir;
            EXPECT_EQ(loaded->output_dir, base / "outputs");
            EXPECT_EQ(loaded->summary_file, base / "summary.json");
            EXPECT_EQ(loaded->report_file, base / "report.txt");
            ASSERT_EQ(loaded->jobs.size(), 1u);
            EXPECT_EQ(loaded->jobs[0].dataset_path, base / "scene");
            EXPECT_EQ(loaded->jobs[0].config_file, base / "train.json");
            EXPECT_EQ(loaded->jobs[0].output_path, base / "out");
            EXPECT_THROW(lfs::app::resolve_batch_jobs(*loaded), std::runtime_error);
        }
    }

    TEST(BatchTrainerTest, BadInputsFailBeforeDryRunCanSucceed) {
        TempDirectoryGuard temp;
        const auto scene = temp.path() / "scene";
        fs::create_directories(scene);
        std::ofstream(scene / "transforms.json") << "{}";
        lfs::core::param::BatchTrainingParameters params;
        params.input_paths = {scene, temp.path() / "missing"};
        params.output_dir = temp.path() / "out";
        params.dry_run = true;
        EXPECT_THROW(lfs::app::resolve_batch_jobs(params), std::runtime_error);
        EXPECT_NE(lfs::app::run_batch_training(params), 0);
        EXPECT_FALSE(fs::exists(params.output_dir));
        const auto manifest = temp.path() / "batch.json";
        for (const auto contents : {"not json", "{\"jobs\":[{}]}", "{\"jobs\":[{\"dataset\":\"\"}]}"}) {
            std::ofstream(manifest) << contents;
            EXPECT_FALSE(parse_batch({scene.string(), manifest.string(), "--dry-run"}));
            EXPECT_FALSE(parse_batch({scene.string(), "--manifest", manifest.string(), "--dry-run"}));
        }
        fs::remove(manifest);
        EXPECT_FALSE(parse_batch({scene.string(), manifest.string()}));
        EXPECT_FALSE(parse_batch({scene.string(), "--manifest", manifest.string()}));
    }

    TEST(BatchTrainerTest, KnownValuesKeepLeadingDashAndJoinedShortValuesKeepDataset) {
        auto parsed = parse_batch({"first", "--cropbox", "-1,-1,-1,1,1,1", "-r4", "-i7000", "second", "-oout", "third", "-dfourth", "fifth"});
        ASSERT_TRUE(parsed) << parsed.error();
        const auto& params = std::get<lfs::io::args::BatchTrainingMode>(*parsed).params;
        EXPECT_EQ(params.input_paths, (std::vector<fs::path>{"first", "second", "third", "fourth", "fifth"}));
        EXPECT_EQ(params.output_dir, fs::path("out"));
        EXPECT_EQ(params.shared_args, (std::vector<std::string>{"--cropbox", "-1,-1,-1,1,1,1", "-r4", "-i7000"}));
        lfs::core::param::BatchJobItem job;
        job.dataset_path = "first";
        job.output_path = "out";
        auto child = parse_child(job, params);
        ASSERT_TRUE(child) << child.error();
        ASSERT_TRUE((*child)->cropbox);
        EXPECT_EQ((*(*child)->cropbox)[0], -1.0f);
        EXPECT_EQ((*child)->optimization.iterations, 7000u);
        auto dash_value = parse_batch({"--images", "--help", "scene"});
        ASSERT_TRUE(dash_value) << dash_value.error();
        const auto& dash_params = std::get<lfs::io::args::BatchTrainingMode>(*dash_value).params;
        EXPECT_EQ(dash_params.shared_args, (std::vector<std::string>{"--images", "--help"}));
        EXPECT_EQ(dash_params.input_paths, (std::vector<fs::path>{"scene"}));
        auto unknown = parse_batch({"scene", "--unknown", "value", "next"});
        ASSERT_TRUE(unknown) << unknown.error();
        const auto& unknown_params = std::get<lfs::io::args::BatchTrainingMode>(*unknown).params;
        EXPECT_EQ(unknown_params.shared_args, (std::vector<std::string>{"--unknown", "value"}));
        std::string error;
        EXPECT_FALSE(lfs::app::validate_job_arguments(job, unknown_params, error));
    }

    TEST(BatchTrainerTest, ReservedPathOptionsAreRejectedInEveryArgumentLayerAndSpelling) {
        for (const auto& option : std::vector<std::vector<std::string>>{
                 {"-d", "other"},
                 {"--data-path", "other"},
                 {"-d=other"},
                 {"--data-path=other"},
                 {"-dother"},
                 {"-o", "other"},
                 {"--output-path", "other"},
                 {"-o=other"},
                 {"--output-path=other"},
                 {"-oother"}}) {
            SCOPED_TRACE(option.front());
            for (const bool shared : {true, false}) {
                lfs::core::param::BatchJobItem job;
                job.dataset_path = "scene";
                job.output_path = "out";
                lfs::core::param::BatchTrainingParameters params;
                (shared ? params.shared_args : job.extra_args) = option;
                std::string error;
                EXPECT_FALSE(lfs::app::validate_job_arguments(job, params, error));
                EXPECT_NE(error.find("batch input/output"), std::string::npos);
                EXPECT_NE(error.find("dataset/output"), std::string::npos);
            }
            std::vector<std::string> cli = {"scene", "--"};
            cli.insert(cli.end(), option.begin(), option.end());
            auto parsed = parse_batch(cli);
            ASSERT_TRUE(parsed) << parsed.error();
            lfs::core::param::BatchJobItem job;
            job.dataset_path = "scene";
            job.output_path = "out";
            std::string error;
            EXPECT_FALSE(lfs::app::validate_job_arguments(job, std::get<lfs::io::args::BatchTrainingMode>(*parsed).params, error));
            EXPECT_NE(error.find("Reserved training option"), std::string::npos);
        }
    }

    TEST(BatchTrainerTest, EffectivePathsRejectOverridesInShortOptionClusters) {
        lfs::core::param::BatchJobItem job;
        job.dataset_path = "scene";
        job.output_path = "out";
        lfs::core::param::BatchTrainingParameters params;
        for (const auto option : {"-qdother", "-qoother"}) {
            job.extra_args = {option};
            std::string error;
            EXPECT_FALSE(lfs::app::validate_job_arguments(job, params, error));
            EXPECT_NE(error.find("batch input/output"), std::string::npos);
        }
    }

    TEST(BatchTrainerTest, JobOptionsReplaceSharedListsAndScalarAliases) {
        lfs::core::param::BatchJobItem job;
        job.dataset_path = "scene";
        job.output_path = "out";
        job.extra_args = {"--eval-steps=7000", "--eval-steps", "9000", "-i12000"};
        lfs::core::param::BatchTrainingParameters params;
        params.shared_args = {"--eval", "--eval-steps", "1000", "--eval-steps=2000", "--iter", "30000", "-i40000"};
        EXPECT_EQ(lfs::app::build_child_args(job, params), (std::vector<std::string>{
                                                               "--headless", "-d", "scene", "-o", "out", "--eval", "--eval-steps=7000", "--eval-steps", "9000", "-i12000"}));
        auto child = parse_child(job, params);
        ASSERT_TRUE(child) << child.error();
        EXPECT_EQ((*child)->optimization.eval_steps, (std::vector<size_t>{7000, 9000}));
        EXPECT_EQ((*child)->optimization.iterations, 12000u);
    }

    TEST(BatchTrainerTest, RejectsViewerResumeAndCameraOnlyModesAfterParsing) {
        TempDirectoryGuard temp;
        const auto model = temp.path() / "x.ply";
        const auto checkpoint = temp.path() / "x.resume";
        std::ofstream(model) << "ply\n";
        std::ofstream(checkpoint) << "checkpoint";
        lfs::core::param::BatchJobItem job;
        job.dataset_path = temp.path();
        job.output_path = temp.path() / "out";
        lfs::core::param::BatchTrainingParameters params;
        for (const auto& args : std::vector<std::vector<std::string>>{
                 {"--view", model.string()},
                 {"-v", model.string()},
                 {"--resume", checkpoint.string()},
                 {"--import-cameras", temp.path().string()}}) {
            job.extra_args = args;
            auto parsed = parse_child(job, params);
            ASSERT_TRUE(parsed) << parsed.error();
            std::string error;
            EXPECT_FALSE(lfs::app::validate_job_arguments(job, params, error));
            EXPECT_NE(error.find("headless training"), std::string::npos);
        }
    }

    TEST(BatchTrainerTest, NegativeTimeoutIsRejectedInSeparateAndJoinedForms) {
        for (const auto& args : std::vector<std::vector<std::string>>{
                 {"scene", "--timeout", "-1"},
                 {"scene", "--timeout=-1"}}) {
            auto parsed = parse_batch(args);
            ASSERT_FALSE(parsed);
            EXPECT_NE(parsed.error().find("nonnegative"), std::string::npos);
        }
    }

    TEST(BatchTrainerTest, ReportWritesCheckDirectoryOpenAndFlushFailures) {
        TempDirectoryGuard temp;
        const auto report = temp.path() / "new" / "report.txt";
        ASSERT_TRUE(lfs::app::write_batch_report(report, "complete report"));
        std::ifstream input(report);
        EXPECT_EQ(std::string(std::istreambuf_iterator<char>(input), {}), "complete report");
        EXPECT_FALSE(lfs::app::write_batch_report(report / "blocked.txt", "text"));
        EXPECT_FALSE(lfs::app::write_batch_report(temp.path(), "text"));
#ifdef __linux__
        EXPECT_FALSE(lfs::app::write_batch_report("/dev/full", "text"));
#endif
    }

    TEST(BatchTrainerTest, RejectsSummaryAndReportAliasesBeforeRunning) {
        TempDirectoryGuard temp;
        auto parsed = parse_batch({temp.path().string(), "--summary", (temp.path() / "result").string(),
                                   "--report", (temp.path() / "./result").string(), "--dry-run"});
        ASSERT_TRUE(parsed) << parsed.error();
        auto params = std::get<lfs::io::args::BatchTrainingMode>(*parsed).params;
        std::string error;
        EXPECT_FALSE(lfs::app::validate_batch_report_paths(params, error));
        EXPECT_NE(error.find("different files"), std::string::npos);
        EXPECT_NE(lfs::app::run_batch_training(params), 0);
        EXPECT_FALSE(fs::exists(params.summary_file));
        params.output_dir = temp.path();
        params.summary_file.clear();
        params.report_file = temp.path() / "BATCH_SUMMARY.JSON";
        EXPECT_FALSE(lfs::app::validate_batch_report_paths(params, error));
        params.report_file.clear();
        EXPECT_TRUE(lfs::app::validate_batch_report_paths(params, error));
    }

    TEST(BatchTrainerTest, NamesAndExplicitOutputsCollideCaseInsensitively) {
        TempDirectoryGuard temp;
        lfs::core::param::BatchTrainingParameters params;
        for (const auto name : {"Room", "room", "ROOM_2"}) {
            lfs::core::param::BatchJobItem job;
            job.name = name;
            job.dataset_path = temp.path() / ("scene_" + std::to_string(params.jobs.size()));
            fs::create_directories(job.dataset_path);
            params.jobs.push_back(job);
        }
        params.output_dir = temp.path() / "out";
        auto jobs = lfs::app::resolve_batch_jobs(params);
        ASSERT_EQ(jobs.size(), 3u);
        EXPECT_EQ(jobs[0].name, "Room");
        EXPECT_EQ(jobs[1].name, "room_3");
        EXPECT_EQ(jobs[2].name, "ROOM_2");
        EXPECT_EQ(jobs[1].output_path, params.output_dir / "room_3");
        params.jobs[0].output_path = params.output_dir / "Custom";
        params.jobs[1].output_path = params.output_dir / "custom";
        jobs = lfs::app::resolve_batch_jobs(params);
        EXPECT_EQ(jobs[1].output_path, params.output_dir / "custom_2");
    }

    TEST(BatchTrainerTest, MainHelpUsesManifestInBatchExample) {
        const char* argv[] = {"lichtfeld-studio", "--invalid-option-for-help-test"};
        auto parsed = lfs::io::args::parse_args_and_params(2, argv, false);
        ASSERT_FALSE(parsed);
        EXPECT_NE(parsed.error().find("batch-train --manifest batch.json"), std::string::npos);
        EXPECT_EQ(parsed.error().find("batch-train --config batch.json"), std::string::npos);
    }

    TEST(BatchTimeoutTest, OneMinuteUsesSingular) {
        EXPECT_EQ(lfs::app::batch_timeout_message(1), "Job timed out after 1 minute");
    }

    TEST(BatchTimeoutTest, OtherDurationsUsePlural) {
        EXPECT_EQ(lfs::app::batch_timeout_message(2), "Job timed out after 2 minutes");
        EXPECT_EQ(lfs::app::batch_timeout_message(60), "Job timed out after 60 minutes");
    }

} // namespace
