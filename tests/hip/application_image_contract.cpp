/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/registry.hpp"
#include <core/sh_layout.hpp>
#include "core/tensor/backend/hip/hip_backend_ops.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <core/tensor_execution.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
using namespace lfs;
using namespace core;
namespace lfs::training {
unsigned other_tables=0;
const TrainingOps& cuda_training_ops_table() { ++other_tables; static const TrainingOps t{.backend=GpuBackend::CUDA}; return t; }
const TrainingOps& metal_training_ops_table() { ++other_tables; static const TrainingOps t{.backend=GpuBackend::Metal}; return t; }
const TrainingOps& vulkan_training_ops_table() { ++other_tables; static const TrainingOps t{.backend=GpuBackend::Vulkan}; return t; }
}
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
uint32_t hash32(uint32_t v) { uint32_t s=v*747796405u+2891336453u; uint32_t w=((s>>((s>>28)+4))^s)*277803737u;return (w>>22)^w; }
int main(int argc,char** argv) try {
    require(argc==2,"Pass absolute HIP module");
#ifdef _WIN32
    require(_putenv_s("LFS_HIP_TRAINING_MODULE",argv[1])==0,"Cannot set module");
#else
    require(setenv("LFS_HIP_TRAINING_MODULE",argv[1],1)==0,"Cannot set module");
#endif
    auto configured=configured_gpu_backend();const auto& table=training::training_ops(GpuBackend::HIP);
    require(table.training_image && !training::other_tables && !gpu_backend_live(GpuBackend::HIP),"Image discovery initialized GPU");
    {
        GpuBackendScope scope(GpuBackend::HIP);const auto& ops=*table.training_image;
        auto latest=Tensor::full({7},-1.f,Device::GPU),ema=latest.clone();
        auto loss=Tensor::full({1},2.f,Device::GPU);ops.heatmap(loss,latest,ema,3,.25f);
        loss=Tensor::full({1},4.f,Device::GPU);ops.heatmap(loss,latest,ema,3,.25f);ops.heatmap(loss,latest,ema,-1,.25f);
        auto lv=latest.cpu().to_vector(),ev=ema.cpu().to_vector();
        for(int i=0;i<7;++i) require(lv[i]==(i==3?4.f:-1.f) && ev[i]==(i==3?2.5f:-1.f),"Camera EMA differs");
        std::vector<float> v{-2,-.1f,0,.0001f,.25f,.5f,.9f,1,2};
        auto input=Tensor::from_vector(v,{v.size()},Device::GPU);
        for(float levels:{3.f,255.f,4095.f}) {
            auto q=ops.quantize_to_grid(input,levels).cpu().to_vector();
            for(size_t i=0;i<v.size();++i) {volatile float product=std::clamp(v[i],0.f,1.f)*levels;volatile float plus=product+.5f;float expected=std::trunc(plus)/levels;require(q[i]==expected,"Quantization rounding differs");}
        }
        auto values=input.clone(),zero=Tensor::zeros({1},Device::GPU),two=Tensor::full({1},2.f,Device::GPU);
        ops.normalize_scalar(values,zero,0);require(values.cpu().to_vector()==v,"Zero normalization changed values");
        ops.normalize_scalar(values,two,0);auto normalized=values.cpu().to_vector();for(size_t i=0;i<v.size();++i)require(normalized[i]==v[i]/2,"Normalization differs");
        constexpr int h=35,w=37,hw=h*w;auto background=Tensor::empty({3,h,w},Device::GPU);
        ops.random_background(background,1234);auto random=background.cpu().to_vector();
        for(int c=0;c<3;++c)for(int i=0;i<hw;++i) {uint32_t offset=c==0?0:c==1?0x9e3779b9u:0x6c8e9cf9u;float expected=float(hash32((1234u^uint32_t(i))+offset))*(1.f/4294967296.f);require(random[c*hw+i]==expected,"Random background differs from CPU hash");}
        ops.random_background(background,1234);require(background.cpu().to_vector()==random,"Random seed not repeatable");
        std::vector<float> source(3*7*9);for(size_t i=0;i<source.size();++i)source[i]=float((i*71)%257)/256;
        auto src=Tensor::from_vector(source,{3,7,9},Device::GPU),dst=Tensor::empty({3,19,23},Device::GPU);ops.resize_background(src,dst);
        auto resized=dst.cpu().to_vector();float max_delta=0;
        for(int c=0;c<3;++c)for(int y=0;y<19;++y)for(int x=0;x<23;++x){double sy=std::clamp((y+.5)*7/19-.5,0.,6.),sx=std::clamp((x+.5)*9/23-.5,0.,8.);int y0=int(sy),x0=int(sx);double fy=sy-y0,fx=sx-x0,expected=0;for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)expected+=source[c*63+std::min(y0+dy,6)*9+std::min(x0+dx,8)]*(dy?fy:1-fy)*(dx?fx:1-fx);max_delta=std::max(max_delta,float(std::abs(resized[c*19*23+y*23+x]-expected)));}
        require(max_delta<2e-6,"Bilinear background differs from CPU reference");
        auto edges=Tensor::empty({h,w},Device::GPU),mask=Tensor::zeros({h,w},Device::GPU);
        ops.canny(background,mask,edges);for(float value:edges.cpu().to_vector())require(value==0,"Masked Canny wrote edges");
        auto constant=Tensor::full({3,h,w},.5f,Device::GPU);ops.canny(constant,Tensor{},edges);for(float value:edges.cpu().to_vector())require(std::isfinite(value) && std::abs(value)<1e-6,"Constant Canny produced edge");
        // The ingest primitive must run on the real Core Tensor allocation/queue.
        {
            auto& backend=static_cast<internal::HipBackendOps&>(internal::backend_ops(GpuBackend::HIP));
            std::vector<uint8_t> bytes(17*19*3);for(size_t i=0;i<bytes.size();++i)bytes[i]=uint8_t(i*71);
            auto hwc=Tensor::from_vector(std::vector<float>(bytes.begin(),bytes.end()),{17,19,3},Device::GPU).to(DataType::UInt8);
            auto chw=Tensor::empty({3,17,19},Device::GPU);
            LfsHipSharedImage descriptor{};descriptor.struct_bytes=sizeof(descriptor);descriptor.height=17;descriptor.width=19;descriptor.channels=3;
            backend.shared_image(descriptor,internal::storage_ref(hwc),internal::storage_ref(chw),{},{});
            auto resized_image=backend.shared_image_resize(hwc,13,17,0,3,{});
            require(resized_image.shape()==TensorShape({3,13,17}),"Core image resize bridge shape differs");
            auto planar_image=backend.shared_image_resize(chw,13,17,2,3,{});
            auto a=resized_image.cpu().to_vector(),b=planar_image.cpu().to_vector();for(size_t i=0;i<a.size();++i)require(std::abs(a[i]-b[i])<2e-6,"Core HWC/CHW resize bridge differs");
            auto depth=Tensor::full({7,9},2.f,Device::GPU);auto depth_resized=backend.shared_image_resize(depth,13,17,3,0,{});
            for(float value:depth_resized.cpu().to_vector())require(std::abs(value-2.f)<1e-6,"Core depth prior bridge differs");
            auto got=chw.cpu().to_vector();for(size_t c=0;c<3;++c)for(size_t i=0;i<17*19;++i)require(got[c*17*19+i]==float(bytes[i*3+c])*(1.f/255.f),"Core Tensor image ingest bridge differs");
        }
        // Independent CPU convolution over an extended image, without GPU tiling/shared-memory layout.
        auto gray=[&](int x,int y){x=std::clamp(x,0,w-1);y=std::clamp(y,0,h-1);return .299*random[y*w+x]+.587*random[hw+y*w+x]+.114*random[2*hw+y*w+x];};
        constexpr int gaussian[25]{2,4,5,4,2,4,9,12,9,4,5,12,15,12,5,4,9,12,9,4,2,4,5,4,2};
        auto blur=[&](int x,int y){double sum=0;for(int j=-2;j<=2;++j)for(int i=-2;i<=2;++i)sum+=gray(x+i,y+j)*gaussian[(j+2)*5+i+2]/159.;return sum;};
        auto gradient=[&](int x,int y){std::array<double,2> g{};constexpr int smooth[3]{1,2,1};for(int j=-1;j<=1;++j)for(int i=-1;i<=1;++i){double b=blur(x+i,y+j);g[0]+=i*smooth[j+1]*b;g[1]+=j*smooth[i+1]*b;}return g;};
        ops.canny(background,Tensor{},edges);auto actual_edges=edges.cpu().to_vector();double canny_error=0;
        for(int y=0;y<h;++y)for(int x=0;x<w;++x){auto g=gradient(x,y);double mag=std::hypot(g[0],g[1]);if(mag>0){int dx=int(std::round(g[0]/mag)),dy=int(std::round(g[1]/mag));auto a=gradient(x+dx,y+dy),b=gradient(x-dx,y-dy);if(mag<std::hypot(a[0],a[1]) || mag<std::hypot(b[0],b[1]))mag=0;}canny_error=std::max(canny_error,std::abs(actual_edges[y*w+x]-mag));}
        require(canny_error<3e-6,"Canny differs from independent CPU convolution/NMS");
        auto byte_image=background.mul(255.f).to(DataType::UInt8);auto float_image=byte_image.to(DataType::Float32).mul(1.f/255.f);
        ops.canny(byte_image,Tensor{},edges);auto byte_edges=edges.cpu().to_vector();ops.canny(float_image,Tensor{},edges);auto float_edges=edges.cpu().to_vector();
        for(int i=0;i<hw;++i)require(std::abs(byte_edges[i]-float_edges[i])<2e-6,"Canny byte and normalized float paths differ");
        std::vector<float> identity(16);for(int i=0;i<4;++i)identity[i*5]=1;
        auto view=Tensor::from_vector(identity,{16},Device::GPU),camera=Tensor::zeros({3},Device::GPU),weights=Tensor::empty({h,w},Device::GPU);
        gpu_ops::RoiParams roi{};roi.image={h,w};roi.intrinsics={20.f,21.f,18.2f,17.1f};std::copy(identity.begin(),identity.end(),roi.world_to_cropbox.begin());roi.minimum={-.51f,-.47f,1.1f};roi.maximum={.49f,.53f,2.3f};roi.outside_weight=.2f;
        for(float translation:{0.f,.37f})for(bool inverse:{false,true}){
            roi.world_to_cropbox[12]=translation;roi.inverse=inverse;ops.roi(view,camera,Tensor{},Tensor{},Tensor{},weights,roi);auto got=weights.cpu().to_vector();
            for(int y=0;y<h;++y)for(int x=0;x<w;++x){double enter=0,leave=1e30;std::array<double,3> direction{(x+.5-roi.intrinsics.cx)/roi.intrinsics.fx,(y+.5-roi.intrinsics.cy)/roi.intrinsics.fy,1};bool hit=true;
                for(int axis=0;axis<3;++axis){double origin=axis==0?translation:0;if(std::abs(direction[axis])<1e-8){if(origin<roi.minimum[axis] || origin>roi.maximum[axis])hit=false;}else{double a=(roi.minimum[axis]-origin)/direction[axis],b=(roi.maximum[axis]-origin)/direction[axis];enter=std::max(enter,std::min(a,b));leave=std::min(leave,std::max(a,b));}}
                hit=hit && leave>=enter;require(got[y*w+x]==((hit!=inverse)?1.f:.2f),"Translated/inverted ROI differs from CPU slab test");}
        }
        // Explicit lens rays and alpha-weighted surface depth follow the
        // current ROI contract; invalid rays and absent surfaces stay outside.
        {
            constexpr size_t pixels=12;
            std::vector<float> rays(pixels*3,0.f),surface(pixels,.25f),opacity(pixels,1.f);
            for(size_t i=0;i<pixels;++i)rays[3*i+2]=1.f;
            surface[1]=2.f;opacity[2]=.4f;
            rays[9]=std::numeric_limits<float>::quiet_NaN();
            rays[12]=2.f;surface[4]=.4f;
            surface[5]=.5f;opacity[5]=.5f;
            surface[6]=std::numeric_limits<float>::quiet_NaN();
            opacity[7]=std::numeric_limits<float>::quiet_NaN();
            surface[8]=0.f;surface[9]=-.1f;
            rays[30]=std::numeric_limits<float>::infinity();
            surface[11]=std::numeric_limits<float>::infinity();
            auto lens=Tensor::from_vector(rays,{1,pixels,3},Device::GPU);
            auto surface_tensor=Tensor::from_vector(surface,{1,pixels},Device::GPU);
            auto alpha_tensor=Tensor::from_vector(opacity,{1,pixels},Device::GPU);
            auto roi_weights=Tensor::empty({1,pixels},Device::GPU);
            gpu_ops::RoiParams p{};p.image={1,int(pixels)};p.intrinsics={1e6f,1e6f,0.f,0.f};
            std::copy(identity.begin(),identity.end(),p.world_to_cropbox.begin());
            p.minimum={-.5f,-.5f,0.f};p.maximum={.5f,.5f,1.f};p.outside_weight=.2f;
            for(bool inverse:{false,true})for(bool use_lens:{false,true}) {
                p.inverse=inverse;
                ops.roi(view,camera,use_lens?lens:Tensor{},surface_tensor,alpha_tensor,roi_weights,p);
                const auto got=roi_weights.cpu().to_vector();
                for(size_t i=0;i<pixels;++i) {
                    const float x=use_lens?rays[3*i]:(float(i)+.5f)/1e6f;
                    const float y=use_lens?rays[3*i+1]:.5f/1e6f;
                    const float z=use_lens?rays[3*i+2]:1.f;
                    const float t=surface[i]/opacity[i];
                    const bool inside=std::isfinite(x)&&std::isfinite(y)&&std::isfinite(z)&&opacity[i]>=.5f &&
                        x*t>=-.5f&&x*t<=.5f&&y*t>=-.5f&&y*t<=.5f&&z*t>=0.f&&z*t<=1.f;
                    require(got[i]==((inside!=inverse)?1.f:.2f),"Lens/surface ROI differs from explicit 3D membership");
                }
            }
            p.inverse=false;
            ops.roi(view,camera,lens,Tensor{},Tensor{},roi_weights,p);
            auto got=roi_weights.cpu().to_vector();
            for(size_t i=0;i<pixels;++i)
                require(got[i]==((i==3 || i==10)?.2f:1.f),"Lens ray-only ROI differs at an invalid ray");
        }
        std::printf("Image extended contract: Canny max error %.9g; byte/float agreement and translated/inverted ROI pass\n",canny_error);
        bool rejected=false;try{ops.resize_background(src.cpu(),dst);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"CPU image accepted by native path");
        std::printf("Image contract: bilinear max error %.9g; hash/EMA/quantization exact\n",max_delta);
    }
    require(configured_gpu_backend()==configured && bool(shutdown_gpu_backend(GpuBackend::HIP)),"Image changed default or leaked resources");
    std::puts("PASS HIP application image: independent CPU references, Canny tile boundaries, ROI, native input rejection and teardown");return 0;
} catch(const std::exception& e){std::fprintf(stderr,"IMAGE CONTRACT FAILED: %s\n",e.what());return 1;}
