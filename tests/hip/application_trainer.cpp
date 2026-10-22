/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Exercises the application Trainer and shared ingest, not the fixed-topology
// HIP experiment runner. Keep short pipeline checks distinct from benchmarks.
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include "training/strategies/strategy_utils.hpp"
#include <core/logger.hpp>
#include <core/crash_handler.hpp>
#include <core/scene.hpp>
#include <core/tensor_backend.hpp>
#include <io/dataset_scene_import.hpp>
#include <io/project_document.hpp>
#include <nlohmann/json.hpp>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <chrono>
#include <thread>

namespace {
    unsigned positive(const char* text) {
        unsigned result=0; const std::string_view value(text);
        const auto [end,error]=std::from_chars(value.data(),value.data()+value.size(),result);
        if(error!=std::errc{} || end!=value.data()+value.size() || !result)
            throw std::invalid_argument("Expected a positive integer");
        return result;
    }
}
int main(int argc,char** argv) try {
    using namespace lfs;
    using namespace core;
    if(argc<7) throw std::invalid_argument("Arguments: dataset output steps max_width max_gaussians SH_degree_plus_one [resume_project] [--exercise-mutations] [--exercise-lifecycle] [--require-frozen] [--stop-after=N] [--strategy=mcmc|mrnf|igs+] [--appearance] [--profile]");
    bool exercise_mutations=false;
    bool exercise_lifecycle=false;
    bool require_frozen=false;
    unsigned stop_after=0;
    bool appearance=false;
    bool profile=false;
    std::string strategy(param::kStrategyMCMC);
    std::optional<std::filesystem::path> resume;
    for(int i=7;i<argc;++i) {
        if(std::string_view(argv[i])=="--exercise-mutations")exercise_mutations=true;
        else if(std::string_view(argv[i])=="--exercise-lifecycle")exercise_lifecycle=true;
        else if(std::string_view(argv[i])=="--require-frozen")require_frozen=true;
        else if(std::string_view(argv[i]).starts_with("--stop-after="))stop_after=positive(argv[i]+13);
        else if(std::string_view(argv[i])=="--appearance")appearance=true;
        else if(std::string_view(argv[i])=="--profile")profile=true;
        else if(std::string_view(argv[i]).starts_with("--strategy=")) {
            const auto requested=std::string_view(argv[i]).substr(11);
            if(!param::is_valid_strategy_name(requested))throw std::invalid_argument("Unknown test strategy");
            strategy=std::string(param::canonical_strategy_name(requested));
        }
        else if(!resume)resume=std::filesystem::absolute(argv[i]);
        else throw std::invalid_argument("Unexpected extra argument");
    }
    const std::filesystem::path output=std::filesystem::absolute(argv[2]);
    if(std::filesystem::exists(output) && !std::filesystem::is_empty(output))
        throw std::invalid_argument("Use a new or empty output directory");
    std::filesystem::create_directories(output);
    Logger::get().init(LogLevel::Debug,(output/"runtime.log").string());
    if(auto configured=set_default_gpu_backend(GpuBackend::HIP);!configured)
        throw std::runtime_error(format_for_developer(configured.error()));
    param::TrainingParameters params;
    params.dataset.data_path=std::filesystem::absolute(argv[1]);
    params.dataset.output_path=output;
    params.dataset.output_path_explicit=true;
    params.dataset.max_width=int(positive(argv[4]));
    params.dataset.resize_factor=1;
    params.dataset.loading_params.use_cpu_memory=true;
    params.optimization.iterations=positive(argv[3]);
    if(stop_after>=params.optimization.iterations)throw std::invalid_argument("Stop boundary must precede the configured schedule end");
    params.optimization.max_cap=int(positive(argv[5]));
    params.optimization.sh_degree=int(positive(argv[6]))-1;
    if(params.optimization.sh_degree>3)throw std::invalid_argument("SH degree must be 0 through 3");
    params.optimization.strategy=strategy;
    params.optimization.use_bilateral_grid=appearance;
    params.optimization.use_ppisp=appearance;
    params.optimization.undistort=true;
    params.optimization.headless=true;
    if(profile) {
        params.optimization.perf_bench=true;
        params.optimization.perf_bench_warmup=0;
        std::puts("APPLICATION HIP: startup profiling includes the first step; NOT A THROUGHPUT BENCHMARK");
    }
    params.optimization.save_steps={stop_after?stop_after:params.optimization.iterations};
    params.cli_iterations_set=true;
    if(exercise_mutations) {
        // Deliberately accelerated state transitions for a short correctness
        // test. This mode must never be reported as a comparable benchmark.
        params.optimization.sh_degree_interval=1;
        params.optimization.start_refine=0;
        params.optimization.refine_every=2;
        params.optimization.morton_reorder_interval=2;
        std::puts("APPLICATION HIP: accelerated SH/refinement/Morton validation; NOT A BENCHMARK");
    }
    if(exercise_lifecycle) {
        if(resume || strategy!=param::kStrategyIGSPlus || !exercise_mutations)
            throw std::invalid_argument("Lifecycle fixture requires new IGS training with --exercise-mutations");
        params.optimization.reset_every=4;
        // Keep row identity stable so the frozen-prefix assertions can compare
        // every parameter directly. Morton is covered by the mutation fixture.
        params.optimization.morton_reorder_interval=0;
    }
    // Evaluation remains at its normal default (off) for this pipeline check.
    // A comparison run must explicitly restore its historical evaluation setup.
    int completed_iteration=0;
    {
    Scene scene;
    std::unique_ptr<training::Trainer> owner;
    if(resume) {
        const auto source=*resume;
        params.resume_project=source;
        params.overrides.optimization_json=nlohmann::json{
            {"iterations",params.optimization.iterations},{"save_steps",params.optimization.save_steps},
            {"enable_eval",false}}.dump();
        std::puts("APPLICATION HIP: hydrate project and restore real trainer checkpoint"); std::fflush(stdout);
        auto document=io::project::ProjectDocument::open(source);
        if(!document)throw std::runtime_error(format_for_developer(document.error()));
        const auto bound=document->bound_checkpoint_uuid();
        if(!bound || !*bound)throw std::runtime_error("Source project has no resumable checkpoint");
        const auto hydration=document->hydrate(scene);
        if(!hydration)throw std::runtime_error(format_for_developer(hydration.error()));
        if(!hydration->checkpoint_header)throw std::runtime_error("Source project has no checkpoint header");
        auto restored=training::installTrainerFromProjectCheckpoint(scene,*document,**bound,params,path_to_utf8(source),hydration->checkpoint_header->iteration);
        if(!restored)throw std::runtime_error(restored.error());
        owner=std::move(restored->trainer);
        if(owner->get_current_iteration()!=hydration->checkpoint_header->iteration)
            throw std::runtime_error("Restored trainer did not preserve checkpoint iteration");
        training::grant_headless_project_saves(*owner,params,{},source);
    } else {
        std::puts("APPLICATION HIP: shared dataset ingest"); std::fflush(stdout);
        if(auto loaded=training::loadTrainingDataIntoScene(params,scene);!loaded)
            throw std::runtime_error(loaded.error());
        std::puts("APPLICATION HIP: initialize real training model"); std::fflush(stdout);
        if(auto model=training::initializeTrainingModel(params,scene);!model)
            throw std::runtime_error(model.error());
        owner=std::make_unique<training::Trainer>(scene);
        if(auto initialized=owner->initialize(params);!initialized)
            throw std::runtime_error(initialized.error());
        training::grant_headless_project_saves(*owner,params);
    }
    auto& trainer=*owner;
    auto& model=trainer.get_strategy_mutable().get_model();
    if(exercise_lifecycle) {
        if(model.size()<32)throw std::runtime_error("Lifecycle fixture requires at least 32 initial Gaussians");
        model.set_frozen_ranges({{0,8}});
        training::apply_frozen_ranges_to_optimizer(model,trainer.get_strategy_mutable().get_optimizer());
        std::vector<uint8_t> prune(size_t(model.size()),0);
        std::fill_n(prune.begin(),16,uint8_t(1));
        auto mask=Tensor::from_blob(prune.data(),{prune.size()},Device::CPU,DataType::Bool).to(Device::GPU);
        trainer.get_strategy_mutable().remove_gaussians(mask);
        const auto deleted=model.deleted().to(DataType::UInt8).cpu().to_vector_uint8();
        for(size_t i=0;i<16;++i)if(bool(deleted.at(i))!=(i>=8))
            throw std::runtime_error("Lifecycle prune did not preserve frozen rows and delete trainable rows");
        std::puts("PASS lifecycle prune: frozen rows retained, eight trainable slots released");
    }
    const bool verify_frozen=model.has_frozen_ranges() && model.frozen_ranges().size()==1 &&
        model.frozen_ranges().front().start==0 && model.frozen_ranges().front().count==8 &&
        trainer.getParams().optimization.morton_reorder_interval==0;
    if((exercise_lifecycle || require_frozen) && !verify_frozen)
        throw std::runtime_error("Expected frozen prefix or stable row layout is missing");
    const auto frozen_snapshot=[&] {
        std::vector<std::vector<float>> result;
        for(const Tensor* field:{&model.means(),&model.sh0(),&model.rotation_raw(),&model.opacity_raw(),&model.scaling_raw()})
            result.push_back(field->is_valid() && field->numel()?field->slice(0,0,8).to(DataType::Float32).cpu().to_vector():std::vector<float>{});
        const auto sh=model.shN_canonical_cpu();
        result.push_back(sh.is_valid() && sh.numel()?sh.slice(0,0,8).to_vector():std::vector<float>{});
        return result;
    };
    const auto frozen_before=verify_frozen?frozen_snapshot():std::vector<std::vector<float>>{};
    std::jthread limiter;
    if(stop_after)limiter=std::jthread([&](std::stop_token token) {
        while(!token.stop_requested()) {
            if(trainer.get_progress_iteration()>=int(stop_after)){trainer.request_stop();return;}
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    std::puts("APPLICATION HIP: real Trainer::train"); std::fflush(stdout);
    auto trained=trainer.train();
    limiter.request_stop();if(limiter.joinable())limiter.join();
    if(!trained)
        throw std::runtime_error(format_for_developer(trained.error()));
    completed_iteration=trainer.get_current_iteration();
    if(stop_after?(completed_iteration<int(stop_after) || completed_iteration>=int(params.optimization.iterations)):
        completed_iteration!=int(params.optimization.iterations))
        throw std::runtime_error("Application trainer did not reach requested iteration");
    if(verify_frozen && frozen_snapshot()!=frozen_before)
        throw std::runtime_error("Frozen Gaussian parameters changed during training/refinement/opacity reset");
    if(exercise_lifecycle) {
        const auto deleted=model.deleted().to(DataType::UInt8).cpu().to_vector_uint8();
        size_t reused=0;for(size_t i=8;i<16;++i)reused+=!deleted.at(i);
        if(!reused)throw std::runtime_error("Lifecycle fixture did not exercise released slot reuse");
        std::printf("PASS lifecycle frozen parameters unchanged; reused %zu released slots\n",reused);
    }
    std::printf("APPLICATION HIP: completed %d steps with configured schedule %zu\n",completed_iteration,size_t(params.optimization.iterations));
    trainer.shutdown();
    }
    // Reopen the saved container and inspect the actual bound CKPT. Merely
    // scheduling a save, or observing a file, is not a successful checkpoint.
    auto document=io::project::ProjectDocument::open(params.dataset.project_file(),{.defer_geometry_payloads=true});
    if(!document)throw std::runtime_error(format_for_developer(document.error()));
    const auto bound=document->bound_checkpoint_uuid();
    if(!bound || !*bound)throw std::runtime_error("Saved project has no resumable checkpoint binding");
    const auto* checkpoint=document->find_checkpoint(**bound);
    if(!checkpoint)throw std::runtime_error("Saved project is missing the bound checkpoint payload");
    auto verified=checkpoint->visit_stream([&](std::istream& stream,uint64_t bytes)->lfs::Result<void> {
        const auto header=load_checkpoint_header(stream,bytes);
        if(!header)throw std::runtime_error(header.error());
        if(header->iteration!=completed_iteration || !header->num_gaussians)
            throw std::runtime_error("Saved checkpoint iteration or model count differs from completed training");
        return {};
    });
    if(!verified)throw std::runtime_error(format_for_developer(verified.error()));
    // Use the application's exit protocol: Q16 run scratch and other registered
    // long-lived owners must be released before the explicit backend check.
    core::teardown_gpu_before_exit();
    if(auto closed=shutdown_gpu_backend(GpuBackend::HIP);!closed)
        throw std::runtime_error(format_for_developer(closed.error()));
    std::puts("PASS HIP real application trainer, shared ingest and terminal project save");
    return 0;
} catch(const std::exception& error) {
    std::fprintf(stderr,"APPLICATION HIP FAILED: %s\n",error.what());
    // Release the native context after scene/trainer owners unwind, including
    // failed initialization. A failed test must not leave a live GPU session.
    try {
        lfs::core::teardown_gpu_before_exit();
        if(auto closed=lfs::core::shutdown_gpu_backend(lfs::core::GpuBackend::HIP);!closed)
            std::fprintf(stderr,"HIP cleanup failed: %s\n",lfs::format_for_developer(closed.error()).c_str());
    } catch(const std::exception& cleanup) {
        std::fprintf(stderr,"HIP cleanup failed: %s\n",cleanup.what());
    }
    return 1;
}
