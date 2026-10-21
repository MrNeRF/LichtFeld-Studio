/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor.hpp>
#include <core/detail/gpu_backend_ops.hpp>
#include <core/tensor_backend.hpp>
#include <core/tensor_spatial.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace lfs::core;
void equal(const Tensor& gpu, const Tensor& cpu, bool approximate=false) {
    auto result=gpu.cpu();
    if(result.numel()!=cpu.numel() || result.dtype()!=cpu.dtype()) throw std::runtime_error("Spatial shape/type differs");
    if(approximate) {
        for(size_t i=0;i<cpu.numel();++i) if(std::abs(result.ptr<float>()[i]-cpu.ptr<float>()[i])>1e-6f)
            throw std::runtime_error("Spatial spacing differs");
    } else if(cpu.numel() && std::memcmp(result.data_ptr(),cpu.data_ptr(),cpu.numel()*(cpu.dtype()==DataType::Bool || cpu.dtype()==DataType::UInt8 ? 1 : 4)))
        throw std::runtime_error("Spatial result differs from CPU");
}
void low_level_contract() {
    using namespace lfs::core::internal;
    auto& gpu=backend_ops(GpuBackend::HIP);
    struct Buffer {
        GpuBackendOps& gpu; StorageRef storage;
        Buffer(GpuBackendOps& g,size_t bytes,DataType dtype):gpu(g),storage(g.allocate(bytes,256,{})) {storage.dtype=dtype;}
        ~Buffer(){gpu.deallocate(storage,{});}
    };
    const auto host=[](void* p){return StorageRef{GpuBackend::HIP,p,0,DataType::UInt8,nullptr,STORAGE_REF_HOST_MEMORY};};
    Buffer p(gpu,48,DataType::Float32),r(gpu,4,DataType::UInt8),h(gpu,4,DataType::Int32),n(gpu,16,DataType::Int32),o(gpu,24,DataType::Int32);
    std::vector<float> xyz{0,0,0, .5f,0,0, 1,0,0, 3,0,0};
    std::vector<uint8_t> mask(4,1);
    std::vector<int32_t> guards(6,-77),observed(6);
    gpu.copy_host_to_device({host(xyz.data()),p.storage,48});
    gpu.copy_host_to_device({host(mask.data()),r.storage,4});
    gpu.copy_host_to_device({host(guards.data()),o.storage,24});
    auto out=o.storage;out.byte_offset=4;
    const auto run=[&](size_t buckets,float radius,StorageRef result) {
        gpu.radius_neighbor_counts(p.storage,r.storage,h.storage,n.storage,result,4,buckets,radius,100,{},{});
    };
    const auto reject=[&](auto operation) {
        bool rejected=false;try{operation();}catch(const std::exception&){rejected=true;}
        if(!rejected) throw std::runtime_error("Invalid HIP spatial request accepted");
    };
    reject([&]{run(3,.75f,out);});
    reject([&]{run(1,0,out);});
    reject([&]{run(1,NAN,out);});
    auto short_out=out;short_out.byte_offset=20;
    reject([&]{run(1,.75f,short_out);});
    auto aliased=p.storage;aliased.dtype=DataType::Int32;
    reject([&]{run(1,.75f,aliased);});
    auto unaligned=out;unaligned.byte_offset=1;
    reject([&]{run(1,.75f,unaligned);});
    gpu.copy_device_to_host({o.storage,host(observed.data()),24});
    if(observed!=guards) throw std::runtime_error("Rejected HIP spatial request modified output");
    for(int repeat=0;repeat<4;++repeat) {
        run(1,.75f,out);
        gpu.copy_device_to_host({o.storage,host(observed.data()),24});
        if(observed!=std::vector<int32_t>{-77,1,2,1,0,-77})
            throw std::runtime_error("HIP spatial forced collision, output guard or repeated workspace reuse failed");
    }
    auto floating_out=out;floating_out.dtype=DataType::Float32;
    gpu.point_neighbor_spacing(p.storage,r.storage,h.storage,n.storage,floating_out,4,1,2.f,{});
    std::vector<float> floats(6);
    gpu.copy_device_to_host({o.storage,host(floats.data()),24});
    const float spacing[]{1.5f,7.f/6.f,7.f/6.f,2.5f};
    for(size_t i=0;i<4;++i) if(std::abs(floats[i+1]-spacing[i])>1e-6f) throw std::runtime_error("HIP grid spacing differs");
    Buffer v(gpu,16,DataType::Float32),rad(gpu,16,DataType::Float32);
    std::vector<float> values{4,3,2,1},radii{1,1,.25f,1};
    mask={1,1,0,1};
    gpu.copy_host_to_device({host(values.data()),v.storage,16});
    gpu.copy_host_to_device({host(radii.data()),rad.storage,16});
    gpu.copy_host_to_device({host(mask.data()),r.storage,4});
    gpu.radius_neighbor_min(p.storage,v.storage,r.storage,h.storage,n.storage,floating_out,4,1,.75f,rad.storage,{});
    gpu.copy_device_to_host({o.storage,host(floats.data()),24});
    if(floats[1]!=3 || floats[2]!=3 || floats[3]!=2 || floats[4]!=1) throw std::runtime_error("HIP mutual radii minimum differs");
    xyz={-.5f,0,0, .5f,0,0, -.5f,0,0, 10000,0,0};
    gpu.copy_host_to_device({host(xyz.data()),p.storage,48});
    gpu.nearest_point_indices(p.storage,p.storage,h.storage,n.storage,out,4,4,1,.5f,{});
    gpu.copy_device_to_host({o.storage,host(observed.data()),24});
    if(observed[1]!=0 || observed[2]!=1 || observed[3]!=0 || observed[4]!=3)
        throw std::runtime_error("HIP nearest exact ties or large empty gap failed");
}
int main() try {
    GpuBackendScope scope(GpuBackend::HIP);
    for(size_t n : {size_t(0),size_t(1),size_t(17),size_t(257),size_t(4099)}) {
        std::vector<float> xyz(n*3), values(n);
        std::vector<int32_t> integers(n);
        std::vector<bool> refs(n), queries(n);
        for(size_t i=0;i<n;++i) {
            xyz[3*i]=float(int(i%19)-9)*0.125f;
            xyz[3*i+1]=float(int(i%13)-6)*0.25f;
            xyz[3*i+2]=float(int(i%7)-3)*0.125f;
            values[i]=float(int(i%31)-15);integers[i]=int32_t(i%31)-15;
            refs[i]=i%3!=0;queries[i]=i%5!=0;
        }
        if(n>10) {xyz[3]=std::numeric_limits<float>::quiet_NaN();xyz[6]=INFINITY;xyz[9]=-INFINITY;}
        const auto p=Tensor::from_vector(xyz,{n,3},Device::CPU);
        const auto r=Tensor::from_vector(refs,{n},Device::CPU),q=Tensor::from_vector(queries,{n},Device::CPU);
        const auto v=Tensor::from_vector(values,{n},Device::CPU),iv=Tensor::from_vector(integers,{n},Device::CPU);
        auto pg=p.to(Device::GPU),rg=r.to(Device::GPU),qg=q.to(Device::GPU),vg=v.to(Device::GPU),ig=iv.to(Device::GPU);

        std::vector<float> camera_values{
            1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,0,
            1,0,0,1, 0,1,0,1, 0,0,1,2, 0,0,-2,0};
        const auto cameras=Tensor::from_vector(camera_values,{2,16},Device::CPU);
        const auto cameras_gpu=cameras.to(Device::GPU);
        for(float maximum : {0.f,0.5f,4.f})
            equal(camera_frustum_counts(pg,cameras_gpu,maximum),camera_frustum_counts(p,cameras,maximum));
        for(float radius : {0.125f,0.5f,4.f}) {
            for(bool exclude : {false,true}) {
                equal(radius_neighbors(pg,rg,radius,exclude),radius_neighbors(p,r,radius,exclude));
                equal(radius_neighbors(pg,rg,radius,exclude,&qg),radius_neighbors(p,r,radius,exclude,&q));
            }
            for(int maximum : {1,3,9999}) {
                equal(radius_neighbor_counts(pg,rg,radius,maximum),radius_neighbor_counts(p,r,radius,maximum));
                equal(radius_neighbor_counts(pg,rg,radius,maximum,&qg),radius_neighbor_counts(p,r,radius,maximum,&q));
            }
            equal(radius_neighbor_min(pg,vg,radius),radius_neighbor_min(p,v,radius));
            equal(radius_neighbor_min(pg,ig,radius),radius_neighbor_min(p,iv,radius));


        }
    }
    low_level_contract();

    std::puts("PASS HIP spatial: hash collisions, masks, self exclusion, saturation, Int32/Float32 minima, direct mutual radii/grid spacing/nearest ties, public camera coverage, nonfinite and empty points");
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"HIP spatial failed: %s\n",e.what());return 1;}

