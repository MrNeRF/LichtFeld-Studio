/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "training/components/popspa_controller.hpp"
#include "lfs/training/ops/popspa_hip.hpp"
#include "lfs/training/ops/registry.hpp"
#include <core/tensor_backend.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>
using namespace lfs;using namespace lfs::core;using namespace lfs::training;
// Use the real registry with only POPSpa enabled for this contract. The full
// application remains closed pending end-to-end trainer validation.
namespace lfs::training {
unsigned other_tables=0;
const TrainingOps& cuda_training_ops_table(){++other_tables;static const TrainingOps table{.backend=GpuBackend::CUDA};return table;}
const TrainingOps& vulkan_training_ops_table(){++other_tables;static const TrainingOps table{.backend=GpuBackend::Vulkan};return table;}
const TrainingOps& metal_training_ops_table(){++other_tables;static const TrainingOps table{.backend=GpuBackend::Metal};return table;}
}
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class T>static void ok(const T& result){if(!result)throw std::runtime_error("POPSpa controller rejected a valid transition");}
static Tensor mask(const std::vector<unsigned char>& values){auto host=Tensor::empty({values.size()},Device::CPU,DataType::Bool);std::memcpy(host.data_ptr(),values.data(),values.size());return host.gpu();}
static void roundtrip(POPSpaController& controller){
    std::fprintf(stderr,"Controller checkpoint phase=%u rows=%zu\n",unsigned(controller.phase()),controller.state_size());
    std::stringstream stream;ok(controller.serialize(stream));const auto bytes=stream.str();
    auto expected_size=controller.checkpoint_size_bytes(controller.state_size());ok(expected_size);require(expected_size.value()==bytes.size(),"Controller checkpoint size mismatch");
    POPSpaController restored;ok(restored.deserialize(stream));
    require(restored.phase()==controller.phase() && restored.phase_step()==controller.phase_step() &&
            restored.score_view()==controller.score_view() && restored.state_size()==controller.state_size() &&
            restored.config().camera_fingerprint==controller.config().camera_fingerprint,"Controller checkpoint metadata changed");
    require(restored.proxy().to_vector()==controller.proxy().to_vector() && restored.dual().to_vector()==controller.dual().to_vector(),"Controller checkpoint values changed");
    std::stringstream second;ok(restored.serialize(second));require(second.str()==bytes,"Controller checkpoint bytes are not stable");
    auto consumed=POPSpaController::consume_checkpoint(second);ok(consumed);require(consumed.value()==controller.state_size(),"Checkpoint consumer row mismatch");
    std::stringstream truncated(bytes.substr(0,bytes.size()/2));require(!restored.deserialize(truncated),"Truncated controller checkpoint was accepted");
    std::stringstream after;ok(restored.serialize(after));require(after.str()==bytes,"Failed checkpoint read mutated the controller");
}
int main(int argc,char** argv)try{
    std::setvbuf(stdout,nullptr,_IONBF,0);
    std::fprintf(stderr,"START HIP POPSpa controller contract argc=%d\n",argc);
    require(argc==2,"Pass the absolute native HIP training module");
#ifdef _WIN32
    require(_putenv_s("LFS_HIP_TRAINING_MODULE",argv[1])==0,"Cannot select HIP module");
#else
    require(setenv("LFS_HIP_TRAINING_MODULE",argv[1],1)==0,"Cannot select HIP module");
#endif
    require(training_ops(GpuBackend::HIP).popspa && !gpu_backend_live(GpuBackend::HIP),"Controller registry discovery initialized HIP or missed its family");
    std::fprintf(stderr,"Controller registry admitted without context initialization\n");
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    std::fprintf(stderr,"Controller HIP selected\n");
    {
        const GpuBackendScope scope(GpuBackend::HIP);
        POPSpaController c;POPSpaController::Config config{.target_count=2,.sparsify_steps=2,.refine_steps=1,.projection_interval=2,.camera_fingerprint=12345};
        auto raw=Tensor::zeros({8,1},Device::GPU),active=mask({1,1,1,1,1,1,1,1}),frozen=mask({0,0,0,0,0,0,0,1});
        std::fprintf(stderr,"Controller fixture allocated\n");
        ok(c.initialize(config,raw,2,active,frozen));std::fprintf(stderr,"Controller initialized\n");require(c.config().first_prune_count==4,"Automatic first prune budget changed");roundtrip(c);
        require(!c.prune_indices(),"Controller allowed pruning before score completion");ok(c.finish_score_view());roundtrip(c);ok(c.finish_score_view());
        require(c.phase()==POPSpaPhase::FirstPrune,"First scoring phase did not finish");roundtrip(c);
        auto keep=c.prune_indices();ok(keep);require(keep.value().to_vector_int64()==std::vector<int64_t>({0,1,2,7}),"First prune lost stable ties or frozen row");
        raw=Tensor::zeros({4,1},Device::GPU);active=mask({1,1,1,1});frozen=mask({0,0,0,1});ok(c.accept_prune(raw,active,frozen));
        require(c.phase()==POPSpaPhase::Sparsify && c.proxy().to_vector()==std::vector<float>({.5f,0,0,.5f}),"Initial projection lost frozen row");roundtrip(c);
        auto scales=Tensor::full({4,3},-2.f,Device::GPU),go=Tensor::full({4,1},.125f,Device::GPU),gs=Tensor::full({4,3},.25f,Device::GPU);
        auto loss=c.regularization(raw,scales,go,gs);ok(loss);require(std::isfinite(loss.value().item()) && loss.value().item()>=0,"Controller regularization loss is invalid");
        require(go.cpu().to_vector().back()==.125f,"Frozen opacity gradient changed");
        const auto gradients=gs.cpu().to_vector();for(size_t i=9;i<12;++i)require(gradients[i]==.25f,"Frozen scale gradient changed");
        ok(c.finish_optimization_step());require(c.projection_due(),"Projection interval was not honored");ok(c.after_backward(raw));
        require(c.dual().to_vector()==std::vector<float>({0,.5f,.5f,0}),"Controller dual update changed");roundtrip(c);
        ok(c.finish_optimization_step());require(c.phase()==POPSpaPhase::SecondScore,"Sparsify did not enter second score phase");roundtrip(c);
        auto scores=Tensor::empty({4,8},Device::CPU,DataType::UInt8);const double values[]{0,0,4,3};
        for(size_t i=0;i<4;++i)gpu_ops::encode_pop_score(values[i],static_cast<std::byte*>(scores.data_ptr())+8*i,gpu_ops::PopScoreEncoding::Float64);c.scores()=scores.gpu();
        ok(c.finish_score_view());ok(c.finish_score_view());require(c.phase()==POPSpaPhase::FinalPrune,"Second score did not finish");roundtrip(c);
        keep=c.prune_indices();ok(keep);require(keep.value().to_vector_int64()==std::vector<int64_t>({2,3}),"Final prune ignored score order or frozen row");
        raw=Tensor::zeros({2,1},Device::GPU);active=mask({1,1});frozen=mask({0,1});ok(c.accept_prune(raw,active,frozen));require(c.phase()==POPSpaPhase::Recover,"Final prune did not enter recovery");roundtrip(c);
        ok(c.finish_optimization_step());require(c.phase()==POPSpaPhase::Complete,"Recovery did not complete");roundtrip(c);
    }
    require(other_tables==0,"HIP controller queried another backend table");
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"Controller retained HIP storage");std::puts("PASS native HIP POPSpa controller: all phases, frozen rows, regularization, exact checkpoint roundtrips and rejected-read transaction");return 0;
}catch(const std::exception& error){std::fprintf(stderr,"FAIL %s\n",error.what());return 1;}
