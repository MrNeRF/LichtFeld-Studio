/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "training_session_fixture.hpp"
#include "training/hip/session.hpp"
#include "core/tensor/backend/hip/external_access.hpp"
#include "lfs/training/ops/registry.hpp"
#include <core/tensor_backend.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
using namespace lfs;
using namespace lfs::core;
namespace lfs::training {
unsigned other_tables = 0;
const TrainingOps& cuda_training_ops_table() { ++other_tables; static const TrainingOps t{.backend=GpuBackend::CUDA}; return t; }
const TrainingOps& vulkan_training_ops_table() { ++other_tables; static const TrainingOps t{.backend=GpuBackend::Vulkan}; return t; }
const TrainingOps& metal_training_ops_table() { ++other_tables; static const TrainingOps t{.backend=GpuBackend::Metal}; return t; }
}
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main(int argc, char** argv) try {
    require(argc==2 && std::filesystem::path(argv[1]).is_absolute(),"Pass the absolute native HIP training module");
#ifdef _WIN32
    require(_putenv_s("LFS_HIP_TRAINING_MODULE",argv[1])==0,"Cannot select HIP application module");
#else
    require(setenv("LFS_HIP_TRAINING_MODULE",argv[1],1)==0,"Cannot select HIP application module");
#endif
    const auto configured=configured_gpu_backend();
    const auto& table=training::training_ops(GpuBackend::HIP);
    require(table.fast && !table.adam && !table.photometric && !table.refine && training::other_tables==0,
            "HIP FastGS registration fell through or admitted missing training families");
    require(!gpu_backend_live(GpuBackend::HIP),"Registry discovery initialized the HIP context");
    {
        const GpuBackendScope scope(GpuBackend::HIP);
        // The private module must borrow Core-owned scratch rather than hide
        // allocations from Tensor accounting. Failed allocations publish no
        // owner/address and must not invalidate a previously acquired owner.
        {
            const auto before=gpu_backend_memory_info(GpuBackend::HIP).allocated_bytes;
            uint64_t address=0;void* owner=nullptr;LfsHipError error{};
            require(!internal::hip_tensor_scratch_allocate(nullptr,0,0,4096,&address,&owner,&error)&&address&&owner,
                    "Core HIP scratch allocation failed");
            const auto live=gpu_backend_memory_info(GpuBackend::HIP).allocated_bytes;
            require(live>=before+4096,"HIP scratch is absent from Tensor accounting");
            uint64_t rejected=123;void* rejected_owner=owner;
            require(internal::hip_tensor_scratch_allocate(nullptr,-1,0,4096,&rejected,&rejected_owner,&error)&&
                    !rejected&&!rejected_owner,"HIP scratch accepted a foreign device or published failure outputs");
            require(gpu_backend_memory_info(GpuBackend::HIP).allocated_bytes==live,"Rejected HIP scratch changed live storage");
            internal::hip_tensor_scratch_release(nullptr,owner);
            require(gpu_backend_memory_info(GpuBackend::HIP).allocated_bytes==before,"Core HIP scratch owner did not release storage");
            std::puts("PASS Tensor-owned scratch accounting, device rejection, failure rollback and release");
        }
        using namespace hip_fixture;
        Fixture fixture;
        fixture.perturb();
        std::array<Tensor,5> parameters;
        constexpr unsigned attributes[]{3,4,3,3,1};
        for (size_t i=0;i<5;++i)
            parameters[i]=Tensor::from_vector(fixture.values[i],{n,attributes[i]},Device::GPU);
        auto view=Tensor::eye(4);
        auto camera=Tensor::zeros({3},Device::GPU);
        Tensor absent,image,alpha,depth,normal,screen_share;
        gpu_ops::SplatInputs inputs{parameters[0],parameters[2],parameters[1],parameters[4],parameters[3],absent,absent};
        gpu_ops::RenderOutputs outputs{image,alpha,depth,normal};
        gpu_ops::FastParams options{};
        options.full_image={height,width}; options.intrinsics={32,32,width/2.f,height/2.f}; options.render_depth=false;
        gpu_ops::FastSaved saved{table.fast->create()};
        auto result=table.fast->forward(saved,inputs,view,camera,absent,absent,options,outputs,screen_share);
        require(result.code==gpu_ops::RasterResult::Code::Success && result.has_work,"Real HIP TrainingOps forward failed");
        const auto initial=image.cpu().to_vector();
        {
            const auto before=gpu_backend_memory_info(GpuBackend::HIP).allocated_bytes;
            const auto layout=training::hip::Session::model_layout(std::filesystem::absolute(argv[1]),n);
            const LfsHipTrainConfig limited{sizeof(limited),0,n,layout.required_bytes+4};
            bool exhausted=false;
            try {training::hip::Session rejected(std::filesystem::absolute(argv[1]),limited,fixture.model());}
            catch(const training::hip::TrainingError& e){exhausted=e.status==LFS_HIP_TRAIN_RESOURCE_EXHAUSTED;}
            require(exhausted,"HIP scratch budget failure lost ResourceExhausted status");
            require(gpu_backend_memory_info(GpuBackend::HIP).allocated_bytes==before,
                    "Partial HIP scratch construction leaked Tensor storage");
            require(image.cpu().to_vector()==initial,"Rejected scratch construction changed the live render");
            std::puts("PASS partial workspace budget exhaustion preserves live tensors and releases Core scratch");
        }
        const LfsHipTrainConfig config{sizeof(config),0,n,64*1024*1024};
        training::hip::Session reference(std::filesystem::absolute(argv[1]),config,fixture.model());
        reference.set_rasterizer(1); reference.set_optimizer(1); reference.set_frame(fixture.frame());
        std::vector<float> control(channels);
        reference.render(control);
        float image_error=0;
        for (size_t i=0;i<initial.size();++i) image_error=std::max(image_error,std::abs(initial[i]-control[i]));
        std::printf("Application/private forward max error %.9g\n",image_error);
        require(image_error<2e-6f,"HIP TrainingOps forward differs from validated private raster");
        auto target=Tensor::from_vector(fixture.target,{3,height,width},Device::GPU);
        auto gradient=image.sub(target).mul(2.f/channels);
        std::array<Tensor,6> packed,bounds;
        constexpr unsigned mapping[]{0,2,1,4,3};
        constexpr float rates[]{.00016f,.005f,.001f,.05f,.0025f};
        std::array<gpu_ops::BackwardAdamParam,6> groups{
            gpu_ops::BackwardAdamParam{parameters[0],packed[0],bounds[0],absent,absent,absent,absent},
            gpu_ops::BackwardAdamParam{parameters[2],packed[1],bounds[1],absent,absent,absent,absent},
            gpu_ops::BackwardAdamParam{parameters[1],packed[2],bounds[2],absent,absent,absent,absent},
            gpu_ops::BackwardAdamParam{parameters[4],packed[3],bounds[3],absent,absent,absent,absent},
            gpu_ops::BackwardAdamParam{parameters[3],packed[4],bounds[4],absent,absent,absent,absent},
            gpu_ops::BackwardAdamParam{absent,packed[5],bounds[5],absent,absent,absent,absent}};
        for (size_t i=0;i<5;++i) {
            const auto attr=attributes[mapping[i]];
            packed[i]=Tensor::zeros({size_t(n)*attr*4},Device::GPU,DataType::UInt8);
            bounds[i]=Tensor::zeros({(n+255)/256,4},Device::GPU);
            auto& g=groups[i]; g.joint_bits=16; g.primitives=n; g.elements=n*attr; g.attributes=attr;
            g.step_size=rates[i]/(1-std::pow(.9f,1.f)); g.bc2_sqrt_rcp=1/std::sqrt(1-std::pow(.999f,1.f)); g.enabled=true;
        }
        gpu_ops::BackwardAdam adam{groups,absent,absent};
        adam.eps=1e-8f;
        table.fast->backward(saved,{gradient,absent,absent,absent},absent,absent,absent,absent,adam,DensificationType::None);
        reference.step({.00016f,.001f,.005f,.0025f,.05f});
        float parameter_error=0;
        for (size_t i=0;i<5;++i) {
            std::vector<float> expected(fixture.values[i].size());
            reference.read_parameter(i,expected);
            const auto actual=parameters[i].cpu().to_vector();
            for (size_t k=0;k<actual.size();++k) parameter_error=std::max(parameter_error,std::abs(actual[k]-expected[k]));
        }
        std::printf("Application/private one-step fused Adam max error %.9g\n",parameter_error);
        require(parameter_error<3e-6f,"HIP TrainingOps backward/Adam differs from validated private training");
        table.fast->release(saved);
        // The current API can capture raw gradients without updating parameters
        // or moments. Compare that path with accumulation and a finite difference.
        std::array<std::vector<float>,5> unchanged,unchanged_packed,unchanged_bounds;
        for(size_t i=0;i<5;++i) {
            unchanged[i]=parameters[i].cpu().to_vector();
            unchanged_packed[i]=packed[i].cpu().to_vector();
            unchanged_bounds[i]=bounds[i].cpu().to_vector();
        }
        gpu_ops::BackwardAdam capture{groups,absent,absent};
        for(size_t i=0;i<5;++i) {
            auto& group=capture.groups[i];group.enabled=false;
            group.gradient=Tensor::full({n,size_t(group.attributes)},.03125f,Device::GPU);
        }
        std::array<std::vector<float>,5> first_gradient;
        auto forward_again=[&] {
            const auto next=table.fast->forward(saved,inputs,view,camera,absent,absent,options,outputs,screen_share);
            require(next.code==gpu_ops::RasterResult::Code::Success && next.has_work,"Capture forward failed");
        };
        for(int pass=0;pass<2;++pass) {
            forward_again();
            table.fast->backward(saved,{gradient,absent,absent,absent},absent,absent,absent,absent,capture,DensificationType::None);
            for(size_t i=0;i<5;++i) {
                auto values=capture.groups[i].gradient.cpu().to_vector();
                if(pass==0)first_gradient[i]=values;
                else for(size_t j=0;j<values.size();++j)
                    require(std::abs(values[j]-(2.f*first_gradient[i][j]-.03125f))<3e-6f,"Raw gradient capture did not accumulate");
            }
        }
        bool captured_nonzero=false;
        for(const auto& group:first_gradient)for(const auto value:group)captured_nonzero|=std::abs(value-.03125f)>1e-7f;
        require(captured_nonzero,"Capture contract did not exercise nonzero gradients");
        auto assert_unchanged=[&] {
            for(size_t i=0;i<5;++i) {
                require(parameters[i].cpu().to_vector()==unchanged[i],"Capture updated model parameters");
                require(packed[i].cpu().to_vector()==unchanged_packed[i],"Capture updated Adam moments");
                require(bounds[i].cpu().to_vector()==unchanged_bounds[i],"Capture updated Adam bounds");
            }
        };
        assert_unchanged();
        // Impact scoring must preserve raw gradients and use the original
        // two-pixel reduction, independently of the optimized RGB dispatch.
        auto impact_sum=Tensor::zeros({n},Device::GPU),impact_peak=Tensor::zeros({n},Device::GPU);
        capture.impact_sum=impact_sum;capture.impact_peak=impact_peak;
        for(size_t i=0;i<5;++i)capture.groups[i].gradient=Tensor::full({n,size_t(capture.groups[i].attributes)},.03125f,Device::GPU);
        forward_again();const auto full_render=image.cpu().to_vector();
        require(table.fast->pop_scores,"HIP FastGS POPSpa callback missing");
        auto pop=Tensor::zeros({n,8},Device::GPU,DataType::UInt8);
        table.fast->pop_scores(saved,absent,absent,pop);
        const auto pop_bytes=pop.cpu().to_vector_uint8();
        std::vector<double> pop_values(n);std::memcpy(pop_values.data(),pop_bytes.data(),n*sizeof(double));
        table.fast->pop_scores(saved,absent,absent,pop);
        const auto accumulated_bytes=pop.cpu().to_vector_uint8();
        std::vector<double> accumulated(n);std::memcpy(accumulated.data(),accumulated_bytes.data(),n*sizeof(double));
        for(size_t i=0;i<n;++i)require(std::isfinite(pop_values[i]) && pop_values[i]>=0 &&
            std::abs(accumulated[i]-2*pop_values[i])<=1e-10+1e-6*pop_values[i],"HIP FastGS POPSpa did not accumulate");
        table.fast->backward(saved,{gradient,absent,absent,absent},absent,absent,absent,absent,capture,DensificationType::None);
        const auto sums=impact_sum.cpu().to_vector(),peaks=impact_peak.cpu().to_vector();
        const auto selected=size_t(std::max_element(sums.begin(),sums.end())-sums.begin());
        require(sums[selected]>1e-6f,"Impact scoring did not exercise visible contributors");
        for(size_t i=0;i<sums.size();++i)
            require(std::isfinite(sums[i]) && std::isfinite(peaks[i]) && peaks[i]>=0 && peaks[i]<=sums[i]+2e-5f,"Invalid impact sum/peak");
        for(size_t i=0;i<5;++i) {
            const auto values=capture.groups[i].gradient.cpu().to_vector();
            for(size_t j=0;j<values.size();++j)
                require(std::abs(values[j]-first_gradient[i][j])<=3e-6f+1e-3f*std::abs(first_gradient[i][j]-.03125f),"Impact scoring changed raw gradients");
        }
        assert_unchanged();
        const auto fixed_gradient=gradient.cpu().to_vector();
        auto objective=[&](float opacity_delta,bool remove) {
            auto values=unchanged[4];values[selected]=remove ? -80.f : values[selected]+opacity_delta;
            parameters[4]=Tensor::from_vector(values,{n,1},Device::GPU);
            forward_again();const auto rendered=image.cpu().to_vector();
            table.fast->release(saved);
            double value=0;for(size_t j=0;j<rendered.size();++j)value+=double(rendered[j])*fixed_gradient[j];
            return std::pair{value,rendered};
        };
        const auto plus=objective(.002f,false),minus=objective(-.002f,false),removed=objective(0,true);
        const double numerical=(plus.first-minus.first)/.004;
        const double analytic=first_gradient[3][selected]-.03125f;
        require(std::abs(numerical-analytic)<=2e-5+0.03*std::abs(numerical),"Captured opacity gradient differs from finite difference");
        double removal_sum=0,removal_peak=0,removal_energy=0;
        for(size_t pixel=0;pixel<size_t(height)*width;++pixel) {
            double value=0;for(size_t channel=0;channel<3;++channel) {
                const size_t j=channel*height*width+pixel;
                value+=std::abs(double(full_render[j])-removed.second[j]);
                const double delta=double(full_render[j])-removed.second[j];removal_energy+=delta*delta;
            }
            removal_sum+=value;removal_peak=std::max(removal_peak,value);
        }
        require(std::abs(removal_sum-sums[selected])<=3e-4+0.003*removal_sum,"Impact sum differs from a removed Gaussian render");
        require(std::abs(removal_peak-peaks[selected])<=3e-5+0.003*removal_peak,"Impact peak differs from a removed Gaussian render");
        require(removal_energy>1e-8 && std::abs(removal_energy-pop_values[selected])<=1e-7+0.003*removal_energy,
                "HIP FastGS POPSpa removal energy differs from a removed Gaussian render");
        std::printf("PASS FastGS POPSpa accumulation/live backward/removal energy %.12g vs %.12g\n",pop_values[selected],removal_energy);
        parameters[4]=Tensor::from_vector(unchanged[4],{n,1},Device::GPU);
        std::printf("PASS raw capture/accumulation/Adam preservation, finite difference %.9g vs %.9g, impact removal %.9g vs %.9g\n",analytic,numerical,double(sums[selected]),removal_sum);
        // Exercise shape changes and optional depth/normal through the real table.
        options.render_depth=true; options.render_normal=true;
        options.tile_w=19; options.tile_h=17; options.tile_x=3; options.tile_y=2;
        result=table.fast->forward(saved,inputs,view,camera,absent,absent,options,outputs,screen_share);
        require(result.code==gpu_ops::RasterResult::Code::Success && depth.numel()==19*17 && normal.numel()==19*17*3,"HIP TrainingOps tile/depth/normal binding failed");
        for (const auto& output : {image,alpha,depth,normal})
            for (float value : output.cpu().to_vector()) require(std::isfinite(value),"HIP TrainingOps optional output not finite");
        table.fast->release(saved); table.fast->release_caches(saved);
        reference.close();
    }
    require(configured_gpu_backend()==configured,"HIP TrainingOps changed the process default");
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"HIP TrainingOps leaked model/queue/module owners");
    std::puts("PASS real HIP TrainingOps: isolated registration, native forward, fused Adam/private equivalence, tile/depth/normal, cache release and complete owner teardown; full admission remains closed");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr,"HIP APPLICATION FAST CONTRACT FAILED: %s\n",error.what());
    return 1;
}
