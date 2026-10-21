/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor_backend.hpp>
#include <lfs/training/ops/registry.hpp>
#include <lfs/training/ops/ppisp.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
using namespace lfs::core;using namespace lfs::training;
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
static void near(double a,double b,double t=3e-4){require(std::isfinite(a)&&std::abs(a-b)<=t*(1+std::abs(b)),"ISP differs from reference");}
struct PhotoResult {std::vector<float> output,grad_rgb;std::array<std::vector<float>,4> parameters,gradients;};
static PhotoResult evaluate(GpuBackend backend) {
    const GpuBackendScope scope(backend);const auto* ops=training_ops(backend).ppisp;require(ops,"Missing ISP family");
    constexpr size_t h=17,w=19,n=h*w;
    std::array<Tensor,4> p{Tensor::zeros({3},Device::GPU),Tensor::zeros({2,3,5},Device::GPU),Tensor::zeros({3,8},Device::GPU),Tensor::zeros({2,3,4},Device::GPU)};
    const lfs::gpu_ops::PPISPOutputs outputs{p[0],p[1],p[2],p[3]};ops->initialize(outputs);
    std::vector<float> pixels(3*n),gradient(3*n);for(size_t i=0;i<pixels.size();++i){pixels[i]=.2f+float(i%73)/130;gradient[i]=float(int(i%11)-5)*.007f;}
    auto rgb=Tensor::from_vector(pixels,{3,h,w},Device::GPU),go=Tensor::from_vector(gradient,{3,h,w},Device::GPU),out=Tensor::empty({3,h,w},Device::GPU);
    const lfs::gpu_ops::PPISPInputs inputs{p[0],p[1],p[2],p[3]};const lfs::gpu_ops::PPISPRegion region{0,int(h),2,3,1,2,0,int(w)};
    ops->forward(inputs,rgb,out,region);const auto neutral=out.cpu().to_vector();for(size_t i=0;i<neutral.size();++i)near(neutral[i],pixels[i],3e-5);
    p[0].add_(.07f);p[1].add_(.005f);p[2].add_(.003f);p[3].add_(.01f);
    std::array<Tensor,4> g{Tensor::full(p[0].shape(),.125f,Device::GPU),Tensor::full(p[1].shape(),.125f,Device::GPU),Tensor::full(p[2].shape(),.125f,Device::GPU),Tensor::full(p[3].shape(),.125f,Device::GPU)};
    auto gi=Tensor::full({3,h,w},99.f,Device::GPU);const lfs::gpu_ops::PPISPOutputs gradients{g[0],g[1],g[2],g[3]};
    ops->forward(inputs,rgb,out,region);ops->backward(inputs,rgb,go,gradients,gi,2,3,1,2);
    PhotoResult result{out.cpu().to_vector(),gi.cpu().to_vector()};
    for(size_t i=0;i<4;++i){result.parameters[i]=p[i].cpu().to_vector();result.gradients[i]=g[i].cpu().to_vector();}
    if(backend==GpuBackend::HIP) {
        auto aliased=go.clone();std::array<Tensor,4> other{Tensor::full(p[0].shape(),.125f,Device::GPU),Tensor::full(p[1].shape(),.125f,Device::GPU),Tensor::full(p[2].shape(),.125f,Device::GPU),Tensor::full(p[3].shape(),.125f,Device::GPU)};
        ops->backward(inputs,rgb,aliased,{other[0],other[1],other[2],other[3]},aliased,2,3,1,2);
        const auto inplace=aliased.cpu().to_vector();for(size_t i=0;i<inplace.size();++i)near(inplace[i],result.grad_rgb[i],1e-6);
        auto crop=rgb.slice(1,4,11).slice(2,3,12).contiguous(),cropped=Tensor::empty({3,7,9},Device::GPU);
        ops->forward(inputs,crop,cropped,{4,int(h),2,3,1,2,3,int(w)});
        const auto cv=cropped.cpu().to_vector();for(size_t c=0;c<3;++c)for(size_t y=0;y<7;++y)for(size_t x=0;x<9;++x)near(cv[(c*7+y)*9+x],result.output[(c*h+y+4)*w+x+3],2e-6);
        // Central differences of the full image dot product, independent of
        // the backward implementation and its parameter reduction ordering.
        const size_t selected[]{2,15+7,16+3,12+2};
        for(size_t group=0;group<4;++group) {
            double scores[2]{};constexpr float epsilon=.001f;
            for(int sign=0;sign<2;++sign) {
                auto values=result.parameters[group];values[selected[group]]+=(sign?epsilon:-epsilon);
                auto parameter=Tensor::from_vector(values,p[group].shape(),Device::GPU);p[group].copy_from(parameter);
                ops->forward(inputs,rgb,out,region);const auto render=out.cpu().to_vector();for(size_t i=0;i<render.size();++i)scores[sign]+=double(render[i])*gradient[i];
            }
            near(result.gradients[group][selected[group]]-.125,(scores[1]-scores[0])/(2*.001),2e-3);
            p[group].copy_from(Tensor::from_vector(result.parameters[group],p[group].shape(),Device::GPU));
        }
    }
    return result;
}
int main()try {
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    const auto native=evaluate(GpuBackend::HIP),reference=evaluate(GpuBackend::Vulkan);
    for(size_t i=0;i<native.output.size();++i){near(native.output[i],reference.output[i]);near(native.grad_rgb[i],reference.grad_rgb[i]);}
    for(size_t g=0;g<4;++g){for(size_t i=0;i<native.parameters[g].size();++i)near(native.parameters[g][i],reference.parameters[g][i],2e-6);for(size_t i=0;i<native.gradients[g].size();++i)near(native.gradients[g][i],reference.gradients[g][i],5e-4);}
    {
        const auto* ops=training_ops(GpuBackend::HIP).ppisp;
        std::vector<float> values(30);for(size_t i=0;i<values.size();++i)values[i]=float(int(i%11)-5)*.02f;
        auto parameters=Tensor::from_vector(values,{2,3,5},Device::GPU),gradient=Tensor::full({2,3,5},.125f,Device::GPU),loss=Tensor::zeros({1},Device::GPU);ops->vignetting_regularization(parameters,gradient,loss,.7f,.4f,.2f);
        std::vector<double> expected(30,.125);double expected_loss=0;
        for(size_t camera=0;camera<2;++camera) {
            for(size_t channel=0;channel<3;++channel) {
                for(size_t p=0;p<2;++p){const auto i=camera*15+channel*5+p;expected_loss+=.7*values[i]*values[i]/6;expected[i]+=1.4*values[i]/6;}
                for(size_t p=2;p<5;++p){const auto i=camera*15+channel*5+p;if(values[i]>0){expected_loss+=.2*values[i]/18;expected[i]+=.2/18;}}
            }
            for(size_t p=0;p<5;++p) {
                double mean=0;for(size_t channel=0;channel<3;++channel)mean+=values[camera*15+channel*5+p]/3.;
                for(size_t channel=0;channel<3;++channel){const auto i=camera*15+channel*5+p;const double diff=values[i]-mean;expected_loss+=.4*diff*diff/30;expected[i]+=.8*diff/30;}
            }
        }
        near(loss.cpu().item<float>(),expected_loss,2e-6);const auto actual=gradient.cpu().to_vector();for(size_t i=0;i<actual.size();++i)near(actual[i],expected[i],2e-6);
        Tensor none;auto only_gradient=Tensor::full({2,3,5},.125f,Device::GPU);ops->vignetting_regularization(parameters,only_gradient,none,.7f,.4f,.2f);const auto optional=only_gradient.cpu().to_vector();for(size_t i=0;i<optional.size();++i)near(optional[i],actual[i],0);
        auto exposure=Tensor::from_vector(std::vector<float>{1,2,5},{3},Device::GPU);std::vector<float> colors(24);for(size_t i=0;i<24;++i)colors[i]=float(i/8)*.3f+float(i%8)*.01f;auto color=Tensor::from_vector(colors,{3,8},Device::GPU);ops->project_mean(exposure,color);
        const auto centered=exposure.cpu().to_vector();near(centered[0],1-8./3,1e-6);near(centered[1],2-8./3,1e-6);near(centered[2],5-8./3,1e-6);
        const auto cc=color.cpu().to_vector();for(size_t i=0;i<24;++i)near(cc[i],double(i/8)*.3-.3,1e-6);
        std::array<Tensor,4> p,m,v,g;for(size_t i=0;i<4;++i){p[i]=Tensor::full({17+i},.3f,Device::GPU);m[i]=Tensor::full({17+i},.2f,Device::GPU);v[i]=Tensor::full({17+i},.4f,Device::GPU);g[i]=Tensor::full({17+i},-.7f,Device::GPU);}
        std::array<lfs::gpu_ops::PPISPAdamGroup,4> groups{{{p[0],m[0],v[0],g[0]},{p[1],m[1],v[1],g[1]},{p[2],m[2],v[2],g[2]},{none,m[3],v[3],g[3]}}};
        ops->adam_batch(groups,{.01f,.9f,.99f,2.f,3.f,1e-6f});const double mm=.9*.2+.1*-.7,vv=.99*.4+.01*.49,updated=.3-.01*mm*2/(std::sqrt(vv*9)+1e-6);
        for(size_t i=0;i<4;++i)for(float x:p[i].cpu().to_vector())near(x,i==3?.3:updated,1e-6);
        ops->adam({p[3],m[3],v[3],g[3]},{.01f,.9f,.99f,2.f,3.f,1e-6f});for(float x:p[3].cpu().to_vector())near(x,updated,1e-6);
    }
    require(bool(shutdown_gpu_backend(GpuBackend::Vulkan)),"ISP reference retained storage");require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"ISP retained storage");
    std::puts("PASS HIP ISP: identity, Vulkan forward/backward parity, parameter central differences, crop coordinates, in-place backward, CPU regularization/Adam/project references and optional outputs/groups");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());(void)shutdown_gpu_backend(GpuBackend::HIP);return 1;}
