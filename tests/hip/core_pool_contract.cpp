/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/nn/ops.hpp>
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <core/splat_exportable_storage.hpp>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
using namespace lfs::core;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static void verify_splat_storage() {
    using S = SplatExportableStorage;
    auto created=S::create(37,3,-1,129);
    require(bool(created),"HIP GUI splat storage allocation failed");
    auto storage=std::move(*created);
    const auto offsets=storage.region_offsets;
    auto* address=storage.block->device_ptr;
    auto backing=std::static_pointer_cast<Tensor>(storage.block->state);
    require(backing && gpu_backend_of(*backing)==GpuBackend::HIP,"Splat block selected a foreign GPU backend");
    auto means=Tensor::view_sharing_storage(*backing,offsets[S::Means],{37,3},37,DataType::Float32,"splat.exportable");
    means.fill_(7.f);
    const auto grown=storage.grow(93);
    require(grown && *grown,"HIP GUI splat storage failed to grow");
    require(storage.block->device_ptr==address && storage.region_offsets==offsets,"Splat growth moved live regions");
    require(means.cpu().to_vector()==std::vector<float>(111,7.f),"Splat growth corrupted existing rows");
    auto opacity=Tensor::view_sharing_storage(*backing,offsets[S::Opacity],{93,1},93,DataType::Float32,"splat.exportable").cpu().to_vector();
    auto rotation=Tensor::view_sharing_storage(*backing,offsets[S::Rotation],{93,4},93,DataType::Float32,"splat.exportable").cpu().to_vector();
    for(size_t row=37;row<93;++row) {
        require(std::isinf(opacity[row]) && opacity[row]<0,"New splat opacity was not initialized inactive");
        for(size_t c=0;c<4;++c)require(rotation[row*4+c]==(c==0?1.f:0.f),"New splat quaternion was not initialized to identity");
    }
    const auto repeated=storage.grow(93);
    require(repeated && !*repeated,"Repeated splat growth changed capacity");
    const auto oversized=storage.grow(130);
    require(!oversized && storage.capacity()==93 && storage.block->device_ptr==address,"Rejected splat growth changed live storage");
    std::puts("PASS HIP GUI splat storage: native backing, pointer-stable growth, retained rows, inactive slack, identity quaternions and rejected overflow");
}
static void verify_where() {
    std::vector<unsigned char> flags{1,0,1,0,1};
    auto condition=Tensor::from_blob(flags.data(),{5,1},Device::CPU,DataType::Bool).to(Device::GPU);
    for(const auto dtype:{DataType::Float32,DataType::Float16,DataType::Int32,DataType::Int64,DataType::UInt8,DataType::Bool,DataType::UInt32}) {
        const size_t width=dtype==DataType::Float16?2:dtype==DataType::Int64?8:
                           (dtype==DataType::UInt8 || dtype==DataType::Bool)?1:4;
        auto source=Tensor::empty({3,5},Device::CPU,dtype),other=Tensor::empty({1,3},Device::CPU,dtype);
        for(size_t i=0;i<18;++i) {
            uint64_t bits;
            if(dtype==DataType::Float32)bits=uint64_t{0x7fc01000}+i; // Preserve NaN payloads.
            else if(dtype==DataType::Float16)bits=uint64_t{0x7e01}+i;
            else if(dtype==DataType::Int64)bits=uint64_t{0x0020000000000001}+i; // Beyond Float64 integer precision.
            else if(dtype==DataType::Bool)bits=i%2;
            else bits=uint64_t{0x81020304}+i;
            auto* bytes=static_cast<unsigned char*>(i<15?source.data_ptr():other.data_ptr());
            std::memcpy(bytes+(i<15?i:i-15)*width,&bits,width);
        }
        auto x=source.to(Device::GPU).transpose(0,1),y=other.to(Device::GPU);
        const auto actual=Tensor::where(condition,x,y).cpu();
        require(actual.shape()==TensorShape{5,3} && actual.dtype()==dtype,"HIP where changed broadcast shape or dtype");
        for(size_t row=0;row<5;++row)for(size_t col=0;col<3;++col) {
            const auto* expected=static_cast<const unsigned char*>(flags[row]?source.data_ptr():other.data_ptr())+
                                 (flags[row]?col*5+row:col)*width;
            require(std::memcmp(static_cast<const unsigned char*>(actual.data_ptr())+(row*3+col)*width,expected,width)==0,
                    "HIP broadcast where changed source bits or selected the wrong element");
        }
    }
    auto scalar=Tensor::full({},0.f,Device::GPU,DataType::Bool);
    auto x=Tensor::full({1,3},7.f,Device::GPU),y=Tensor::full({5,1},9.f,Device::GPU);
    require(Tensor::where(scalar,x,y).cpu().to_vector()==std::vector<float>(15,9.f),"HIP scalar condition broadcast failed");
    const auto empty=Tensor::where(Tensor::empty({0,1},Device::GPU,DataType::Bool),Tensor::empty({0,3},Device::GPU),x);
    require(empty.numel()==0 && empty.shape()==TensorShape{0,3},"HIP empty where broadcast changed shape");
    std::puts("PASS HIP where: seven dtypes, NaN payloads and exact Int64 bits, three-way broadcasting, transposed inputs, scalar conditions and zero extents");
}
int main()try{
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    {
        verify_splat_storage();
        verify_where();
        constexpr int n=2,c=3,h=7,w=11;
        std::vector<float> values(n*c*h*w);
        for(size_t i=0;i<values.size();++i)values[i]=float(int(i*31%127)-63)/32.f;
        auto input=Tensor::from_vector(values,{n,c,h,w},Device::GPU);
        unsigned cases=0;
        for(bool average:{false,true})for(bool include:{false,true})for(int padding:{0,1})for(int stride:{1,2}) {
            constexpr int kh=3,kw=2;
            const int oh=(h+2*padding-kh)/stride+1,ow=(w+2*padding-kw)/stride+1;
            const auto output=average?nn::avg_pool2d(input,kh,kw,stride,stride,padding,padding,include):
                                      nn::max_pool2d(input,kh,kw,stride,stride,padding,padding);
            const auto actual=output.cpu().to_vector();
            require(actual.size()==size_t(n*c*oh*ow),"Pooling output shape differs from reference");
            for(int plane=0;plane<n*c;++plane)for(int y=0;y<oh;++y)for(int x=0;x<ow;++x) {
                double expected=average?0:-std::numeric_limits<double>::infinity();int count=0;
                for(int ky=0;ky<kh;++ky)for(int kx=0;kx<kw;++kx) {
                    const int iy=y*stride-padding+ky,ix=x*stride-padding+kx;
                    if(iy<0 || iy>=h || ix<0 || ix>=w)continue;
                    const auto value=values[(plane*h+iy)*w+ix];
                    expected=average?expected+value:std::max(expected,double(value));++count;
                }
                if(average)expected/=std::max(1,include?kh*kw:count);
                require(std::abs(actual[(plane*oh+y)*ow+x]-expected)<2e-7,"HIP pooling differs from independent CPU reference");
            }
            ++cases;
        }
        std::printf("PASS HIP pooling: %u CPU reference cases, NCHW batches, odd dimensions, non-square kernels, max/mean, stride and padding policies\n",cases);
    }
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"HIP pooling leaked storage");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"HIP POOL FAILED: %s\n",e.what());return 1;}
