/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor_backend.hpp>
#include <lfs/training/ops/registry.hpp>
#include <lfs/training/ops/bilateral.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
using namespace lfs::core;using namespace lfs::training;
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
static void near(double a,double b,double tolerance=5e-5){require(std::isfinite(a)&&std::abs(a-b)<=tolerance*(1+std::abs(b)),"Bilateral differs from reference");}
struct Slice {std::vector<float> output,grid_gradient,rgb_gradient;};
static Slice evaluate(GpuBackend backend,bool chw,bool exposure,const std::vector<float>& grid_values,const std::vector<float>& rgb_values,const std::vector<float>& gradients,const std::vector<float>& offsets,int h,int w) {
    const GpuBackendScope scope(backend);const auto* ops=training_ops(backend).bilateral;require(ops,"Missing bilateral family");
    const size_t channels=exposure?9:12;
    auto grid=Tensor::from_vector(grid_values,{1,channels,3,2,4},Device::GPU);
    const auto shape=chw?TensorShape{3,size_t(h),size_t(w)}:TensorShape{size_t(h),size_t(w),3};
    auto rgb=Tensor::from_vector(rgb_values,shape,Device::GPU),gradient=Tensor::from_vector(gradients,shape,Device::GPU);
    auto offset=Tensor::from_vector(offsets,{channels},Device::GPU),output=Tensor::empty(shape,Device::GPU);
    auto gg=Tensor::full(grid.shape(),.125f,Device::GPU),gi=Tensor::full(shape,99.f,Device::GPU);
    lfs::gpu_ops::GridSliceParams params{chw?lfs::gpu_ops::Layout::CHW:lfs::gpu_ops::Layout::HWC,exposure?lfs::gpu_ops::GridTransform::ExposureChroma:lfs::gpu_ops::GridTransform::Affine,true};
    ops->slice_forward(grid,rgb,offset,output,params);ops->slice_backward(grid,rgb,gradient,offset,gg,gi,params);
    Slice result{output.cpu().to_vector(),gg.cpu().to_vector(),gi.cpu().to_vector()};
    if(backend==GpuBackend::HIP) {
        auto in_place=rgb.clone();ops->slice_forward(grid,in_place,offset,in_place,params);
        const auto values=in_place.cpu().to_vector();for(size_t i=0;i<values.size();++i)near(values[i],result.output[i],1e-7);
        auto in_gradient=gradient.clone(),other_grid=Tensor::full(grid.shape(),.125f,Device::GPU);
        ops->slice_backward(grid,rgb,in_gradient,offset,other_grid,in_gradient,params);
        const auto other=in_gradient.cpu().to_vector();for(size_t i=0;i<other.size();++i)near(other[i],result.rgb_gradient[i],1e-6);
    }
    return result;
}
static Slice affine_cpu(bool chw,const std::vector<float>& grid,const std::vector<float>& rgb,const std::vector<float>& grad,const std::vector<float>& offset,int h,int w) {
    const size_t n=size_t(h)*w;Slice out{std::vector<float>(3*n),std::vector<float>(grid.size(),.125f),std::vector<float>(3*n)};
    std::vector<double> gg(grid.size(),.125);constexpr double guide[3]{.299,.587,.114};
    auto at=[&](size_t p,int c){return chw?c*n+p:p*3+c;};
    for(size_t p=0;p<n;++p) {
        double color[4]{rgb[at(p,0)],rgb[at(p,1)],rgb[at(p,2)],1};
        const double x=w>1?double(p%w)/(w-1)*3:0,y=h>1?double(p/w)/(h-1):0;
        const double grey=std::clamp(guide[0]*color[0]+guide[1]*color[1]+guide[2]*color[2],0.,1.),z=grey*2;
        const int x0=int(std::floor(x)),x1=std::min(x0+1,3),y0=int(std::floor(y)),y1=std::min(y0+1,1),z0=int(std::floor(z)),z1=std::min(z0+1,2);
        const double fx=x-x0,fy=y-y0,fz=z-z0;
        double sampled[12]{},dz[12]{};
        for(int c=0;c<12;++c) {
            sampled[c]=offset[c];
            for(int iz=0;iz<2;++iz)for(int iy=0;iy<2;++iy)for(int ix=0;ix<2;++ix) {
                const auto index=((c*3+(iz?z1:z0))*2+(iy?y1:y0))*4+(ix?x1:x0);
                const auto xy=(ix?fx:1-fx)*(iy?fy:1-fy),weight=xy*(iz?fz:1-fz);
                sampled[c]+=weight*grid[index];dz[c]+=xy*(iz?1:-1)*grid[index];
                gg[index]+=weight*double(grad[at(p,c/4)])*color[c%4];
            }
        }
        for(int row=0;row<3;++row)for(int c=0;c<4;++c)out.output[at(p,row)]+=float(sampled[row*4+c]*color[c]);
        double dguide=0;for(int c=0;c<12;++c)dguide+=dz[c]*double(grad[at(p,c/4)])*color[c%4]*2;
        for(int c=0;c<3;++c) {
            double value=dguide*guide[c];for(int row=0;row<3;++row)value+=sampled[row*4+c]*grad[at(p,row)];out.rgb_gradient[at(p,c)]=float(value);
        }
    }
    for(size_t i=0;i<gg.size();++i)out.grid_gradient[i]=float(gg[i]);return out;
}
int main()try {
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    for(bool chw:{false,true})for(bool exposure:{false,true}) {
        constexpr int h=17,w=19;constexpr size_t n=h*w;const size_t channels=exposure?9:12;
        std::vector<float> grid(channels*3*2*4),rgb(3*n),gradient(3*n),offset(channels);
        for(size_t i=0;i<grid.size();++i)grid[i]=float(int(i%17)-8)*.013f;
        for(size_t i=0;i<channels;++i)offset[i]=float(int(i%5)-2)*.007f;
        for(size_t i=0;i<3*n;++i){rgb[i]=.15f+float(i%97)/140;gradient[i]=float(int(i%11)-5)*.013f;}
        const auto native=evaluate(GpuBackend::HIP,chw,exposure,grid,rgb,gradient,offset,h,w);
        const auto reference=exposure?evaluate(GpuBackend::Vulkan,chw,true,grid,rgb,gradient,offset,h,w):affine_cpu(chw,grid,rgb,gradient,offset,h,w);
        for(size_t i=0;i<native.output.size();++i){near(native.output[i],reference.output[i]);near(native.rgb_gradient[i],reference.rgb_gradient[i],2e-4);}
        for(size_t i=0;i<native.grid_gradient.size();++i)near(native.grid_gradient[i],reference.grid_gradient[i],2e-4);
    }
    {
        const auto* ops=training_ops(GpuBackend::HIP).bilateral;
        constexpr size_t n=2,c=4,l=3,h=2,w=4,plane=l*h*w;std::vector<float> values(n*c*plane);
        for(size_t i=0;i<values.size();++i)values[i]=float(int(i%17)-8)*.07f;
        auto grid=Tensor::from_vector(values,{n,c,l,h,w},Device::GPU),gradient=Tensor::full({n,c,l,h,w},.125f,Device::GPU),loss=Tensor::zeros({1},Device::GPU),temp=Tensor::zeros({2048},Device::GPU);
        ops->tv_forward(grid,loss,temp,3);ops->tv_backward(grid,gradient,.75f,3);
        std::vector<double> expected(values.size(),.125);double total=0;
        for(size_t a=0;a<n;++a)for(size_t b=0;b<c;++b)for(size_t z=0;z<l;++z)for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x) {
            const auto i=(((a*c+b)*l+z)*h+y)*w+x;
            for(const auto edge:{std::array<size_t,3>{x,1,l*h*(w-1)},std::array<size_t,3>{y,w,l*(h-1)*w},std::array<size_t,3>{z,w*h,(l-1)*h*w}})if(edge[0]) {
                const auto j=i-edge[1];const double diff=double(values[i])-values[j],scale=1./(c*3*edge[2]);total+=diff*diff*scale;expected[i]+=1.5*diff*scale;expected[j]-=1.5*diff*scale;
            }
        }
        near(loss.cpu().item<float>(),total,2e-6);const auto actual=gradient.cpu().to_vector();for(size_t i=0;i<actual.size();++i)near(actual[i],expected[i],2e-6);
        auto parameter=Tensor::full({17},.3f,Device::GPU),m=Tensor::full({17},.2f,Device::GPU),v=Tensor::full({17},.4f,Device::GPU),g=Tensor::full({17},-.7f,Device::GPU);
        ops->adam(parameter,m,v,g,{.01f,.9f,.99f,2.f,3.f,1e-6f});
        const double mm=.9*.2+.1*-.7,vv=.99*.4+.01*.49;
        for(float value:parameter.cpu().to_vector())near(value,.3-.01*mm*2/(std::sqrt(vv*9)+1e-6),1e-6);
        ops->scale_moments(m,v,1.5f,.25f);for(float value:m.cpu().to_vector())near(value,mm*1.5,1e-6);for(float value:v.cpu().to_vector())near(value,vv*.25,1e-6);
        auto host=Tensor::from_vector(std::vector<float>{0,1,2,3,4,5,6,7,8,9,10,11,12},{13},Device::CPU),device=Tensor::full({13},99.f,Device::GPU);
        ops->upload_slice(host,device,2,4,5);auto destination=Tensor::full({13},-1.f,Device::CPU);ops->download_slice(destination,device,3,4,5);
        const auto copied=destination.to_vector();for(size_t i=0;i<13;++i)near(copied[i],i>=3&&i<8?double(i-1):-1,0);
        auto mean=Tensor::full({n*c},.2f,Device::GPU),identity=Tensor::full({c},.5f,Device::GPU);ops->project_mean(grid,mean,identity,1);const auto projected=grid.cpu().to_vector();for(size_t i=0;i<values.size();++i)near(projected[i],values[i]+.3,1e-6);
        auto sum=Tensor::full({c},2.f,Device::GPU),offset=Tensor::zeros({c},Device::GPU),old=Tensor::full({c},.3f,Device::GPU),updated=Tensor::full({c},.7f,Device::GPU);
        ops->update_offset(sum,offset,identity,old,updated,4.f,.125f);for(float value:sum.cpu().to_vector())near(value,3.6,1e-6);for(float value:offset.cpu().to_vector())near(value,.05,1e-6);
    }
    require(bool(shutdown_gpu_backend(GpuBackend::Vulkan)),"Bilateral reference retained storage");require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"Bilateral retained storage");
    std::puts("PASS HIP Bilateral: CPU affine forward/analytical gradients/TV, exposure-chroma Vulkan parity, CHW/HWC, in-place paths, Adam/moments, projection/offset and bounded transfers");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());(void)shutdown_gpu_backend(GpuBackend::HIP);return 1;}
