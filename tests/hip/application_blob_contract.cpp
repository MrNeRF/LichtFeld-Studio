/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor_backend.hpp>
#include <core/memory_pressure.hpp>
#include <core/gpu_device_info.hpp>
#include "core/tensor/backend/hip/external_access.hpp"
#include <lfs/training/ops/registry.hpp>
#include <lfs/training/ops/blob.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <stdexcept>
using namespace lfs::core;
using namespace lfs::training;
using namespace lfs::training::kernels::blob_seeding;
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
static void close(double a,double b,double t,const char* m){require(std::isfinite(a)&&std::abs(a-b)<=t,m);}
struct Detection {std::vector<float> peaks;std::vector<uint32_t> bitmap;std::array<float,2> density;};
static Detection detect(GpuBackend backend,const std::vector<float>& image,size_t h,size_t w) {
    const GpuBackendScope scope(backend);
    DetectionWorkspace workspace;
    auto source=Tensor::from_vector(image,{3,h,w},Device::GPU);
    auto result=training_ops(backend).blob->detect(source,3,workspace);
    Detection d;
    if(result.peaks.is_valid())d.peaks=result.peaks.cpu().to_vector();
    const auto bitmap=result.bitmap.cpu();
    d.bitmap.assign(bitmap.ptr<uint32_t>(),bitmap.ptr<uint32_t>()+bitmap.numel());
    d.density=result.density;
    std::vector<std::array<float,7>> rows(d.peaks.size()/7);
    for(size_t i=0;i<rows.size();++i)std::copy_n(d.peaks.data()+i*7,7,rows[i].begin());
    std::sort(rows.begin(),rows.end());
    for(size_t i=0;i<rows.size();++i)std::copy_n(rows[i].begin(),7,d.peaks.data()+i*7);
    return d;
}
int main()try {
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    {
        std::vector<std::future<std::string>> inventory;
        for(int i=0;i<8;++i)inventory.push_back(std::async(std::launch::async,[] {
            const auto devices=enumerate_hip_hardware();
            require(!devices.empty(),"Concurrent HIP inventory returned no devices");
            return devices.front().uuid;
        }));
        const auto identity=inventory.front().get();
        for(size_t i=1;i<inventory.size();++i)
            require(inventory[i].get()==identity,"Concurrent HIP inventory identity changed");
        require(!gpu_backend_live(GpuBackend::HIP),"HIP inventory created a Tensor context");
    }
    {
        require(device_memory_domain(GpuBackend::HIP)==MemoryDomain::HipDevice,"HIP pressure mapped to CUDA");
        require(is_device_heap(MemoryDomain::HipDevice),"HIP pressure is not a device heap");
        auto& pressure=MemoryPressureCoordinator::instance();
        pressure.reset_for_testing();
        size_t free=0,foreign_queries=0;
        pressure.set_free_memory_probe([&](MemoryDomain domain) {
            if(domain!=MemoryDomain::HipDevice)++foreign_queries;
            return free;
        });
        AllocationFailure failure{.domain=MemoryDomain::HipDevice,.requested_bytes=1024,.label="HIP pressure contract"};
        const auto error=to_error(failure,LFS_SOURCE_SITE_CURRENT());
        require(error.domain()==lfs::ErrorDomain::Tensor,"HIP allocation error was classified as CUDA");
        pressure.run_episode(failure,PressureContext::RenderThread);
        require(pressure.pressure_active(),"HIP pressure lease was not established");
        free=pressure.reserve_bytes(MemoryDomain::HipDevice)*2;
        pressure.maybe_recover();
        require(!pressure.pressure_active()&&foreign_queries==0,"HIP pressure recovery queried another backend");
        pressure.reset_for_testing();
    }
    {
        auto live=Tensor::full({17},.75f,Device::GPU);
        auto& pressure=MemoryPressureCoordinator::instance();
        pressure.set_allocation_probe([](MemoryDomain domain,size_t){return domain==MemoryDomain::HipDevice;});
        bool rejected=false;
        try{(void)Tensor::empty({4097},Device::GPU);}
        catch(const MemoryAllocationError& e){rejected=e.domain()==MemoryDomain::HipDevice;}
        uint64_t address=123;
        void* owner=reinterpret_cast<void*>(uintptr_t{1});
        LfsHipError native_error{};
        const auto callback_status=internal::hip_tensor_scratch_allocate(nullptr,selected_hip_device(),0,4097,&address,&owner,&native_error);
        pressure.reset_for_testing();
        require(rejected,"HIP allocator did not preserve its typed memory domain");
        require(callback_status!=0&&native_error.native_code==2&&!address&&!owner,
                "HIP scratch callback lost ResourceExhausted or published a failed owner");
        for(const auto value:live.cpu().to_vector())require(value==.75f,"HIP allocation failure damaged a live Tensor");
    }
    {
        const auto* ops=training_ops(GpuBackend::HIP).blob;
        require(ops&&ops->detect&&ops->downsample&&ops->sweep,"Incomplete HIP Blob family");
        constexpr size_t h=19,w=37,n=h*w;
        std::vector<uint8_t> bytes(3*n);
        std::vector<float> values(3*n);
        for(size_t i=0;i<3*n;++i){bytes[i]=uint8_t((i*31+17)%256);values[i]=bytes[i]/255.f;}
        for(int factor:{1,2,3,7})for(bool byte:{false,true}) {
            DetectionWorkspace workspace;
            auto source=byte?Tensor::from_blob(bytes.data(),{3,h,w},Device::CPU,DataType::UInt8).to(Device::GPU):Tensor::from_vector(values,{3,h,w},Device::GPU);
            const auto actual=ops->downsample(source,factor,workspace).cpu().to_vector();
            const auto oh=h/factor,ow=w/factor;
            for(size_t c=0;c<3;++c)for(size_t y=0;y<oh;++y)for(size_t x=0;x<ow;++x) {
                double sum=0;
                for(int dy=0;dy<factor;++dy)for(int dx=0;dx<factor;++dx)sum+=values[(c*h+y*factor+dy)*w+x*factor+dx];
                close(actual[(c*oh+y)*ow+x],sum/(factor*factor),3e-7,"Blob downsample differs from CPU");
            }
        }
        DetectionWorkspace workspace;
        auto uniform=Tensor::full({3,h,w},.5f,Device::GPU);
        const auto empty=ops->detect(uniform,0,workspace);
        require(!empty.peaks.is_valid()&&empty.density[0]==0&&empty.density[1]==0,"Uniform image generated peaks");
        const auto empty_bitmap=empty.bitmap.cpu();
        for(size_t i=0;i<empty_bitmap.numel();++i)require(empty_bitmap.ptr<uint32_t>()[i]==0,"Uniform image generated bitmap hits");
        constexpr size_t views_count=10,neighbors_count=9;
        std::vector<SweepView> views(views_count);
        std::vector<int32_t> neighbors(views_count*neighbors_count,-1);
        const size_t words=(n+15)/16;
        for(size_t i=0;i<views_count;++i) {
            auto& v=views[i];v.R[0]=v.R[4]=v.R[8]=1;v.fx=v.fy=25;v.cx=w*.5f;v.cy=h*.5f;
            v.width=w;v.height=h;v.bitmap_offset=i*words;v.z_min=.2f;v.z_max=2.f;
            for(size_t j=0;j<neighbors_count;++j)neighbors[i*neighbors_count+j]=int32_t(j+1);
        }
        std::vector<uint8_t> view_bytes(views.size()*sizeof(SweepView));std::memcpy(view_bytes.data(),views.data(),view_bytes.size());
        auto view_tensor=Tensor::from_blob(view_bytes.data(),{view_bytes.size()},Device::CPU,DataType::UInt8).to(Device::GPU);
        auto neighbor_tensor=Tensor::from_vector(neighbors,{views_count,neighbors_count},Device::GPU);
        std::vector<uint32_t> bitmap_words(views_count*words,~uint32_t{});
        auto bitmap=Tensor::from_blob(bitmap_words.data(),{views_count*words},Device::CPU,DataType::UInt32).to(Device::GPU);
        auto peaks=Tensor::from_vector(std::vector<float>{18,9,0,0,.5f,.5f,.5f},{1,7},Device::GPU);
        const auto accepted=ops->sweep(peaks,view_tensor,neighbor_tensor,bitmap).cpu().to_vector();
        close(accepted[0],.2,2e-6,"Blob sweep depth differs from analytical identical-camera case");
        close(accepted[1],1,0,"Blob sweep did not accept nine supporting views");
        // Background HIP batching must preserve every row, including a tail
        // after the cooperative batch boundary. This exercises accepted rows
        // rather than only the all-zero rejection case.
        constexpr size_t batch=4096, total=batch+17;
        std::vector<float> repeated(total*7);
        const std::array<float,7> row{18,9,0,0,.5f,.5f,.5f};
        for(size_t i=0;i<total;++i)std::copy(row.begin(),row.end(),repeated.begin()+i*7);
        auto many=Tensor::from_vector(repeated,{total,7},Device::GPU);
        const auto whole=ops->sweep(many,view_tensor,neighbor_tensor,bitmap).cpu().to_vector();
        std::vector<float> chunked;
        for(size_t first=0;first<total;first+=batch) {
            const auto part=ops->sweep(many.slice(0,first,std::min(total,first+batch)).contiguous(),view_tensor,neighbor_tensor,bitmap).cpu().to_vector();
            chunked.insert(chunked.end(),part.begin(),part.end());
        }
        require(whole==chunked,"Cooperative HIP blob sweep changed output rows");
        bitmap.zero_();
        const auto rejected=ops->sweep(peaks,view_tensor,neighbor_tensor,bitmap).cpu().to_vector();
        require(rejected[0]==0&&rejected[1]==0,"Blob sweep accepted an unsupported peak");
    }
    constexpr size_t h=31,w=43,n=h*w;
    std::vector<float> pattern(3*n,.45f);
    for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x)for(size_t c=0;c<3;++c) {
        const double bright=std::exp(-((double(x)-12)*(x-12.)+(double(y)-10)*(y-10.))/4.);
        const double dark=std::exp(-((double(x)-32)*(x-32.)+(double(y)-23)*(y-23.))/5.);
        pattern[(c*h+y)*w+x]=float(.45+.5*bright-.4*dark);
    }
    const auto native=detect(GpuBackend::HIP,pattern,h,w);
    require(!native.peaks.empty(),"Blob test image did not exercise peak emission");
    const auto portable=detect(GpuBackend::Vulkan,pattern,h,w);
    require(native.bitmap==portable.bitmap&&native.peaks.size()==portable.peaks.size(),"Blob detections differ from Vulkan");
    for(size_t i=0;i<native.peaks.size();++i)close(native.peaks[i],portable.peaks[i],2e-6,"Blob peak differs from Vulkan");
    for(int i=0;i<2;++i)close(native.density[i],portable.density[i],0,"Blob density differs from Vulkan");
    require(bool(shutdown_gpu_backend(GpuBackend::Vulkan)),"Blob contract retained Vulkan storage");
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"Blob contract retained native storage");
    std::puts("PASS HIP Blob CPU downsample, constant detection, analytical sweep, Vulkan peak/bitmap parity");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
