/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Exercises the application Trainer and shared ingest, not the fixed-topology
// HIP experiment runner. Keep short pipeline checks distinct from benchmarks.
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include <core/logger.hpp>
#include <core/scene.hpp>
#include <core/tensor_backend.hpp>
#include <io/dataset_scene_import.hpp>
#include <io/project_document.hpp>
#include <nlohmann/json.hpp>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

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
    if(argc<7) throw std::invalid_argument("Arguments: dataset output steps max_width max_gaussians SH_degree_plus_one [resume_project] [--exercise-mutations] [--strategy=mcmc|mrnf|igs+] [--appearance]");
    bool exercise_mutations=false;
    bool appearance=false;
    std::string strategy(param::kStrategyMCMC);
    std::optional<std::filesystem::path> resume;
    for(int i=7;i<argc;++i) {
        if(std::string_view(argv[i])=="--exercise-mutations")exercise_mutations=true;
        else if(std::string_view(argv[i])=="--appearance")appearance=true;
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
    params.optimization.max_cap=int(positive(argv[5]));
    params.optimization.sh_degree=int(positive(argv[6]))-1;
    if(params.optimization.sh_degree>3)throw std::invalid_argument("SH degree must be 0 through 3");
    params.optimization.strategy=strategy;
    params.optimization.use_bilateral_grid=appearance;
    params.optimization.use_ppisp=appearance;
    params.optimization.undistort=true;
    params.optimization.headless=true;
    params.optimization.save_steps={params.optimization.iterations};
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
    // Evaluation remains at its normal default (off) for this pipeline check.
    // A comparison run must explicitly restore its historical evaluation setup.
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
    std::puts("APPLICATION HIP: real Trainer::train"); std::fflush(stdout);
    if(auto trained=trainer.train();!trained)
        throw std::runtime_error(format_for_developer(trained.error()));
    if(trainer.get_current_iteration()!=int(params.optimization.iterations))
        throw std::runtime_error("Application trainer did not reach requested iteration");
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
        if(header->iteration!=int(params.optimization.iterations) || !header->num_gaussians)
            throw std::runtime_error("Saved checkpoint iteration or model count differs from completed training");
        return {};
    });
    if(!verified)throw std::runtime_error(format_for_developer(verified.error()));
    if(auto closed=shutdown_gpu_backend(GpuBackend::HIP);!closed)
        throw std::runtime_error(format_for_developer(closed.error()));
    std::puts("PASS HIP real application trainer, shared ingest and terminal project save");
    return 0;
} catch(const std::exception& error) {
    std::fprintf(stderr,"APPLICATION HIP FAILED: %s\n",error.what());
    // Release the native context after scene/trainer owners unwind, including
    // failed initialization. A failed test must not leave a live GPU session.
    try {
        if(auto closed=lfs::core::shutdown_gpu_backend(lfs::core::GpuBackend::HIP);!closed)
            std::fprintf(stderr,"HIP cleanup failed: %s\n",lfs::format_for_developer(closed.error()).c_str());
    } catch(const std::exception& cleanup) {
        std::fprintf(stderr,"HIP cleanup failed: %s\n",cleanup.what());
    }
    return 1;
}
