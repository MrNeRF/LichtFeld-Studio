/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/registry.hpp"
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>
using namespace lfs;
using namespace core;
namespace lfs::training {
unsigned other_tables=0;
const TrainingOps& cuda_training_ops_table() { ++other_tables; static const TrainingOps t{.backend=GpuBackend::CUDA}; return t; }
const TrainingOps& metal_training_ops_table() { ++other_tables; static const TrainingOps t{.backend=GpuBackend::Metal}; return t; }
const TrainingOps& vulkan_training_ops_table() { ++other_tables; static const TrainingOps t{.backend=GpuBackend::Vulkan}; return t; }
}
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
constexpr int n=2,c=3,h=17,w=19;
constexpr size_t hw=h*w,count=n*c*hw;
constexpr double gaussian[]{.001028380123898387,.0075987582094967365,.036000773310661316,.10936068743467331,.21300552785396576,.26601171493530273,.21300552785396576,.10936068743467331,.036000773310661316,.0075987582094967365,.001028380123898387};
// Independent double-precision, direct 2-D reference: no extracted kernel code,
// GPU reductions, separable convolutions or cached half derivatives.
double reference(const std::vector<float>& image,const std::vector<float>& raw,const std::vector<float>& target,const std::vector<float>& mask,unsigned path,bool padding=true,double denominator=0) {
    const bool masked=path==4 || path==5,decoupled=path==3 || path==5;
    double loss=0,normalizer=0;
    for(int batch=0;batch<n;++batch) for(int channel=0;channel<c;++channel) for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
        if(!masked && path!=0 && padding && (y<5 || y>=h-5 || x<5 || x>=w-5)) continue;
        const size_t index=((batch*c+channel)*h+y)*w+x;
        const double weight=masked ? mask[y*w+x] : 1.;
        normalizer+=weight;
        if(weight==0) continue;
        const double l1=std::abs(double(image[index])-target[index]);
        if(path==0) { loss+=l1; continue; }
        double mu=0,mur=0,mut=0,rr=0,tt=0,rt=0,mass=0;
        for(int dy=-5;dy<=5;++dy) for(int dx=-5;dx<=5;++dx) {
            const int yy=y+dy,xx=x+dx;
            const bool inside=yy>=0 && yy<h && xx>=0 && xx<w;
            const double kernel=gaussian[dy+5]*gaussian[dx+5]*(masked && inside ? mask[yy*w+xx] : 1.);
            mass+=kernel;
            if(!inside) continue;
            const size_t j=((batch*c+channel)*h+yy)*w+xx;
            const double a=image[j],r=decoupled ? raw[j] : a,t=target[j];
            mu+=a*kernel; mur+=r*kernel; mut+=t*kernel; rr+=r*r*kernel; tt+=t*t*kernel; rt+=r*t*kernel;
        }
        if(masked) { mu/=mass; mur/=mass; mut/=mass; rr/=mass; tt/=mass; rt/=mass; }
        const double luminance=(2*mu*mut+.0001)/(mu*mu+mut*mut+.0001);
        const double cs=(2*(rt-mur*mut)+.0009)/(rr-mur*mur+tt-mut*mut+.0009);
        const double ssim=luminance*cs;
        loss+=weight*(path==1 ? 1-ssim : .8*l1+.2*(1-ssim));
    }
    return loss/(denominator>0 ? denominator : normalizer+(masked ? 1e-8 : 0));
}
int main(int argc,char** argv) try {
    require((argc==2 || (argc==3 && std::string_view(argv[2]).starts_with("--plane-dump="))) && std::filesystem::path(argv[1]).is_absolute(),"Pass absolute HIP training module [--plane-dump=absolute-directory]");
#ifdef _WIN32
    require(_putenv_s("LFS_HIP_TRAINING_MODULE",argv[1])==0,"Cannot set HIP module");
#else
    require(setenv("LFS_HIP_TRAINING_MODULE",argv[1],1)==0,"Cannot set HIP module");
#endif
    const auto configured=configured_gpu_backend();
    const auto& table=training::training_ops(GpuBackend::HIP);
    require(table.photometric && !table.fast && !training::other_tables,"Photometric discovery fell through");
    require(!gpu_backend_live(GpuBackend::HIP),"Discovery initialized GPU");
    {
        const GpuBackendScope scope(GpuBackend::HIP);
        const auto& ops=*table.photometric;
        gpu_ops::PhotoSaved saved; saved.backend=ops.create();
        std::vector<float> host(count),raw(count),target(count),mask(hw);
        for(size_t i=0;i<count;++i) { host[i]=.2f+.5f*float((i*19+3)%73)/73; raw[i]=.18f+.6f*float((i*23+7)%83)/83; target[i]=.1f+.7f*float((i*31+11)%89)/89; }
        for(size_t i=0;i<hw;++i) mask[i]=i%7==0 ? 0.f : (i%5==0 ? .25f : 1.5f);
        const TensorShape shape{n,c,h,w};
        auto image=Tensor::from_vector(host,shape,Device::GPU), raw_image=Tensor::from_vector(raw,shape,Device::GPU);
        auto truth=Tensor::from_vector(target,shape,Device::GPU), support=Tensor::from_vector(mask,{h,w},Device::GPU);
        for(unsigned path=0;path<6;++path) {
            gpu_ops::PhotoParams params; params.path=static_cast<gpu_ops::PhotoPath>(path); params.ssim_weight=.2f;
            Tensor loss,gradient,raw_gradient;
            ops.evaluate(saved,image,raw_image,truth,support,params,loss,gradient,raw_gradient);
            const double actual=loss.item(),expected=reference(host,raw,target,mask,path);
            std::printf("Photo path %u loss %.9g CPU %.9g delta %.3g\n",path,actual,expected,std::abs(actual-expected));
            require(std::abs(actual-expected)<2e-5,"Photometric loss differs from independent CPU reference");
            require(gradient.shape()==shape,"Photometric gradient shape changed");
            const auto values=gradient.cpu().to_vector();
            double maximum_error=0;
            for(size_t index:{size_t(0),size_t(117),size_t(299),size_t(701),size_t(1709)}) {
                auto plus=host,minus=host; plus[index]+=.001f; minus[index]-=.001f;
                const double finite=(reference(plus,raw,target,mask,path)-reference(minus,raw,target,mask,path))/(double(plus[index])-minus[index]);
                maximum_error=std::max(maximum_error,std::abs(values[index]-finite));
                require(std::abs(values[index]-finite)<3e-6+.005*std::abs(finite),"Corrected-image gradient differs from independent finite difference");
            }
            if(path==3 || path==5) {
                require(raw_gradient.shape()==shape,"Missing decoupled raw gradient");
                const auto raw_values=raw_gradient.cpu().to_vector();
                for(size_t index:{size_t(0),size_t(117),size_t(299),size_t(701),size_t(1709)}) {
                    auto plus=raw,minus=raw; plus[index]+=.001f; minus[index]-=.001f;
                    const double finite=(reference(host,plus,target,mask,path)-reference(host,minus,target,mask,path))/(double(plus[index])-minus[index]);
                    maximum_error=std::max(maximum_error,std::abs(raw_values[index]-finite));
                    require(std::abs(raw_values[index]-finite)<3e-6+.005*std::abs(finite),"Raw-image gradient differs from independent finite difference");
                }
            } else require(!raw_gradient.is_valid(),"Nondecoupled path returned raw gradient");
            std::printf("Photo path %u gradient max sampled absolute error %.3g\n",path,maximum_error);
        }
        const double metric=ops.metric(saved,image,truth,true,true).item();
        require(std::abs(metric-(1-reference(host,raw,target,mask,1)))<2e-5,"SSIM metric differs from CPU reference");
        auto map=ops.masked_metric_map(image,truth,support);
        require(map.shape()==shape,"Masked metric map shape changed");
        for(float value:map.cpu().to_vector()) require(std::isfinite(value),"Nonfinite masked metric map");
        auto single=image.slice(0,0,1),single_target=truth.slice(0,0,1); Tensor error;
        ops.error_map(saved,single,single_target,error,false);
        require(error.shape()==TensorShape({h,w}),"Error-map dimensions incorrect");
        for(float value:error.cpu().to_vector()) require(std::isfinite(value) && value>=0,"Invalid error map");
        auto zero=Tensor::zeros({h,w},Device::GPU);
        gpu_ops::PhotoParams params; params.path=gpu_ops::PhotoPath::MaskedDecoupled; params.ssim_weight=.2f;
        Tensor loss,gradient,raw_gradient;
        ops.evaluate(saved,image,raw_image,truth,zero,params,loss,gradient,raw_gradient);
        require(saved.empty_mask && loss.item()==0,"Empty mask was not diagnosed/zeroed");
        for(const auto* tensor:{&gradient,&raw_gradient}) for(float value:tensor->cpu().to_vector()) require(value==0,"Empty mask produced gradient");
        const auto allocated=ops.workspace_bytes(saved).allocated;
        ops.evaluate(saved,image,raw_image,truth,support,params,loss,gradient,raw_gradient);
        require(ops.workspace_bytes(saved).allocated==allocated,"Same-shape photometric workspace grew");
        std::vector<int> bytes(count);
        std::vector<float> decoded(count);
        for(size_t i=0;i<count;++i) { bytes[i]=int(target[i]*255); decoded[i]=float(bytes[i])*(1.f/255.f); }
        auto byte_truth=Tensor::from_vector(bytes,shape,Device::GPU).to(DataType::UInt8);
        auto float_truth=Tensor::from_vector(decoded,shape,Device::GPU);
        auto boolean_mask=Tensor::ones({h,w},Device::GPU,DataType::Bool);
        params.path=gpu_ops::PhotoPath::MaskedFused; params.denominator=float(count);
        ops.evaluate(saved,image,raw_image,byte_truth,boolean_mask,params,loss,gradient,raw_gradient);
        const auto byte_loss=loss.item();
        const auto byte_gradient=gradient.cpu().to_vector();
        auto float_mask=Tensor::ones({h,w},Device::GPU);
        ops.evaluate(saved,image,raw_image,float_truth,float_mask,params,loss,gradient,raw_gradient);
        require(std::abs(loss.item()-byte_loss)<2e-6,"Byte target/mask loss normalization differs");
        const auto float_gradient=gradient.cpu().to_vector();
        for(size_t i=0;i<count;++i) require(std::abs(float_gradient[i]-byte_gradient[i])<2e-6,"Byte target/mask gradient differs");
        bool rejected=false;
        try { ops.evaluate(saved,image.cpu(),raw_image,truth,support,params,loss,gradient,raw_gradient); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"Photometric accepted non-HIP prediction");
        ops.shrink_to_required(saved); ops.reset(saved);
        require(ops.workspace_bytes(saved).allocated==0 && !saved.ssim_map.is_valid(),"Photometric workspace reset retained allocations");
        if(argc==3) {
            // The Python parity runner compares these whole-gradient dumps in
            // separate processes. A HIP DLL may use a separate CRT environment;
            // changing this executable's getenv state is not a valid opt-out.
            const auto directory=std::filesystem::path(std::string_view(argv[2]).substr(13));
            require(directory.is_absolute() && std::filesystem::is_directory(directory),"Plane dump directory must exist and be absolute");
            for(auto extent:{std::array{1025,1089},std::array{1440,1920}})
            for(bool byte:{false,true})for(bool padding:{false,true}) {
                const size_t height=extent[0],width=extent[1],pixels=3*height*width;
                std::vector<float> input(pixels),target_values(pixels);
                for(size_t i=0;i<pixels;++i){input[i]=.15f+.7f*float((i*19+3)%997)/997;target_values[i]=.1f+.8f*float((i*31+11)%991)/991;}
                const TensorShape dimensions{1,3,height,width};
                auto prediction=Tensor::from_vector(input,dimensions,Device::GPU);
                auto target_tensor=Tensor::from_vector(target_values,dimensions,Device::GPU);
                if(byte)target_tensor=target_tensor.mul(255.f).to(DataType::UInt8);
                gpu_ops::PhotoParams comparison;comparison.path=gpu_ops::PhotoPath::Fused;
                comparison.ssim_weight=.2f;comparison.valid_padding=padding;
                ops.evaluate(saved,prediction,{},target_tensor,{},comparison,loss,gradient,raw_gradient);
                const auto selected=gradient.cpu().to_vector();
                for(float value:selected)require(std::isfinite(value),"Plane gradient is not finite");
                const float actual_loss=loss.item();
                const uint64_t workspace=ops.workspace_bytes(saved).allocated,elements=selected.size();
                const auto name=std::to_string(height)+"x"+std::to_string(width)+"-byte"+std::to_string(byte)+"-padding"+std::to_string(padding)+".bin";
                std::ofstream file(directory/name,std::ios::binary);
                file.write(reinterpret_cast<const char*>(&actual_loss),sizeof(actual_loss));
                file.write(reinterpret_cast<const char*>(&workspace),sizeof(workspace));
                file.write(reinterpret_cast<const char*>(&elements),sizeof(elements));
                file.write(reinterpret_cast<const char*>(selected.data()),std::streamsize(selected.size()*sizeof(float)));
                require(bool(file),"Cannot write plane gradient comparison");
                std::printf("PASS plane dump %zux%zu byte=%d padding=%d\n",height,width,int(byte),int(padding));
                ops.reset(saved);
            }
        }
    }
    require(configured_gpu_backend()==configured,"Photometric changed default backend");
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"Photometric leaked resources");
    std::puts("PASS HIP application photometric: all six paths against CPU loss/finite differences, metrics, masks, error maps, workspace reuse/reset and teardown");
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"HIP PHOTO CONTRACT FAILED: %s\n",e.what()); return 1; }
