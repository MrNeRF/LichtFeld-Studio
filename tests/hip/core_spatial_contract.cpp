/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor.hpp>
#include <core/detail/gpu_backend_ops.hpp>
#include <core/tensor_backend.hpp>
#include <core/tensor_spatial.hpp>
#include <core/tensor_ppisp.hpp>
#include <core/tensor_splat.hpp>
#include <core/tensor_export.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace lfs::core;
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
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
    Buffer boxes(gpu,24,DataType::Float32),visit(gpu,16,DataType::Int32);
    std::vector<float> bounds{0,0,0,3,0,0};std::vector<int32_t> order{0,1,2,3};
    gpu.copy_host_to_device({host(bounds.data()),boxes.storage,24});
    gpu.copy_host_to_device({host(order.data()),visit.storage,16});
    PointTreeProgram tree{};tree.points=4;tree.references=4;tree.levels=1;tree.level_count[0]=1;tree.radius=2.f;tree.max_count=100;
    require(gpu.point_tree_spacing(p.storage,p.storage,boxes.storage,visit.storage,floating_out,tree,{}),"HIP point-tree spacing declined native dispatch");
    gpu.copy_device_to_host({o.storage,host(floats.data()),24});
    for(size_t i=0;i<4;++i) if(std::abs(floats[i+1]-spacing[i])>1e-6f) throw std::runtime_error("HIP native tree spacing differs");
    require(gpu.point_tree_counts(p.storage,p.storage,boxes.storage,visit.storage,rad.storage,{},out,tree,{}),"HIP point-tree counts declined native dispatch");
    gpu.copy_device_to_host({o.storage,host(observed.data()),24});
    if(observed[1]!=2 || observed[2]!=2 || observed[3]!=0 || observed[4]!=0) throw std::runtime_error("HIP native tree counts differ");
    auto bad_tree=tree;bad_tree.level_count[0]=2;
    reject([&]{gpu.point_tree_counts(p.storage,p.storage,boxes.storage,visit.storage,rad.storage,{},out,bad_tree,{});});
    std::vector<int32_t> initial_labels{-77,0,1,2,3,-77};
    gpu.copy_host_to_device({host(initial_labels.data()),o.storage,24});
    require(gpu.radius_connected_components(p.storage,r.storage,h.storage,n.storage,out,4,1,.75f,{}),"HIP radius components declined native dispatch");
    gpu.copy_device_to_host({o.storage,host(observed.data()),24});
    if(observed!=std::vector<int32_t>{-77,0,0,2,3,-77}) throw std::runtime_error("HIP native radius components differ");
    Buffer reach(gpu,4,DataType::Float32);float reach_value=1.f;
    gpu.copy_host_to_device({host(&reach_value),reach.storage,4});
    gpu.copy_host_to_device({host(initial_labels.data()),o.storage,24});
    require(gpu.point_tree_components(p.storage,p.storage,boxes.storage,reach.storage,visit.storage,rad.storage,rad.storage,out,tree,{}),"HIP tree components declined native dispatch");
    gpu.copy_device_to_host({o.storage,host(observed.data()),24});
    if(observed!=std::vector<int32_t>{-77,0,0,2,3,-77}) throw std::runtime_error("HIP native mutual components differ");
    xyz={-.5f,0,0, .5f,0,0, -.5f,0,0, 10000,0,0};
    gpu.copy_host_to_device({host(xyz.data()),p.storage,48});
    Buffer query(gpu,48,DataType::Float32);
    std::vector<float> query_points{0,0,0, -.5f,0,0, -100000,0,0, NAN,0,0};
    gpu.copy_host_to_device({host(query_points.data()),query.storage,48});
    gpu.nearest_point_indices(query.storage,p.storage,h.storage,n.storage,out,4,4,1,.5f,{});
    gpu.copy_device_to_host({o.storage,host(observed.data()),24});
    if(observed[1]!=0 || observed[2]!=0 || observed[3]!=0 || observed[4]!=-1)
        throw std::runtime_error("HIP nearest exact ties or large empty gap failed");
}
std::array<double,9> splat_covariance(const float* scales,const float* quat) {
    double w=quat[0],x=quat[1],y=quat[2],z=quat[3];
    const double n=std::sqrt(w*w+x*x+y*y+z*z);w/=n;x/=n;y/=n;z/=n;
    const double r[]{1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                     2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                     2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)};
    std::array<double,9> result{};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k)
        result[i*3+j]+=r[i*3+k]*r[j*3+k]*std::exp(2.*scales[k]);
    return result;
}
void affine_contract() {
    constexpr size_t n=257;
    std::vector<float> scales(n*3),quats(n*4);
    for(size_t i=0;i<n;++i) {
        for(size_t j=0;j<3;++j) scales[i*3+j]=-12.f+16.f*float((i*37+j*71)%257)/256.f;
        for(size_t j=0;j<4;++j) quats[i*4+j]=std::sin(float(i*4+j)+.3f);
    }
    auto cs=Tensor::from_vector(scales,{n,3},Device::CPU),cq=Tensor::from_vector(quats,{n,4},Device::CPU);
    auto gs=Tensor::from_vector(scales,{n,3},Device::GPU),gq=Tensor::from_vector(quats,{n,4},Device::GPU);
    const std::array<splat_transform::LinearTransform,4> matrices{{{{1,0,0,0,1,0,0,0,1}},{{1,.7f,0,0,1,-.4f,0,0,1}},
                                                                {{-1,0,0,0,2,0,0,0,.4f}},{{0,0,0,0,0,0,0,0,0}}}};
    const auto compare=[&](const Tensor& es,const Tensor& eq,const Tensor& os,const Tensor& oq,bool zero) {
        auto actual_s=os.cpu(),actual_q=oq.cpu();
        for(size_t i=0;i<n;++i) {
            for(size_t k=0;k<4;++k) require(std::isfinite(actual_q.ptr<float>()[i*4+k]),"HIP affine nonfinite quaternion");
            if(zero) {
                for(size_t k=0;k<3;++k) require(actual_s.ptr<float>()[i*3+k]==-INFINITY,"HIP affine zero extent changed");
                continue;
            }
            const auto expected=splat_covariance(es.ptr<float>()+i*3,eq.ptr<float>()+i*4);
            const auto actual=splat_covariance(actual_s.ptr<float>()+i*3,actual_q.ptr<float>()+i*4);
            double error=0,norm=0,volume=0;
            for(int k=0;k<9;++k) {error+=(actual[k]-expected[k])*(actual[k]-expected[k]);norm+=expected[k]*expected[k];}
            for(int k=0;k<3;++k) volume+=double(actual_s.ptr<float>()[i*3+k])-es.ptr<float>()[i*3+k];
            require(std::sqrt(error/norm)<3e-6,"HIP affine covariance differs from CPU");
            require(std::abs(volume)<4e-5,"HIP affine short-axis volume differs from CPU");
        }
    };
    for(size_t m=0;m<matrices.size();++m) {
        auto es=Tensor::zeros({n,3},Device::CPU),eq=Tensor::zeros({n,4},Device::CPU);
        affine_splat_geometry(matrices[m],cs,cq,es,eq);
        auto os=Tensor::zeros({n,3},Device::GPU),oq=Tensor::zeros({n,4},Device::GPU);
        affine_splat_geometry(matrices[m],gs,gq,os,oq);compare(es,eq,os,oq,m==3);
        auto inplace_s=gs.clone(),inplace_q=gq.clone();
        affine_splat_geometry(matrices[m],inplace_s,inplace_q,inplace_s,inplace_q);compare(es,eq,inplace_s,inplace_q,m==3);
        std::vector<float> batch(n*9);
        for(size_t i=0;i<n;++i) std::copy_n(matrices[m].rows,9,batch.data()+i*9);
        auto transforms=Tensor::from_vector(batch,{n,9},Device::GPU);
        affine_splat_geometry(transforms,gs,gq,os,oq);compare(es,eq,os,oq,m==3);
    }
}
int main() try {
    GpuBackendScope scope(GpuBackend::HIP);
    for(size_t n : {size_t(0),size_t(1),size_t(17),size_t(257),size_t(4099)}) {
        std::vector<float> xyz(n*3), values(n), radii(n);
        std::vector<int32_t> integers(n);
        std::vector<bool> refs(n), queries(n);
        for(size_t i=0;i<n;++i) {
            xyz[3*i]=float(int(i%19)-9)*0.125f;
            xyz[3*i+1]=float(int(i%13)-6)*0.25f;
            xyz[3*i+2]=float(int(i%7)-3)*0.125f;
            values[i]=float(int(i%31)-15);integers[i]=int32_t(i%31)-15;
            refs[i]=i%3!=0;queries[i]=i%5!=0;radii[i]=.125f*float(1+i%7);
        }
        if(n>10) {xyz[3]=std::numeric_limits<float>::quiet_NaN();xyz[6]=INFINITY;xyz[9]=-INFINITY;}
        const auto p=Tensor::from_vector(xyz,{n,3},Device::CPU);
        const auto r=Tensor::from_vector(refs,{n},Device::CPU),q=Tensor::from_vector(queries,{n},Device::CPU);
        const auto v=Tensor::from_vector(values,{n},Device::CPU),iv=Tensor::from_vector(integers,{n},Device::CPU);
        const auto rad=Tensor::from_vector(radii,{n},Device::CPU);
        auto pg=p.to(Device::GPU),rg=r.to(Device::GPU),qg=q.to(Device::GPU),vg=v.to(Device::GPU),ig=iv.to(Device::GPU),radg=rad.to(Device::GPU);
        equal(nearest_point_indices(pg,pg),nearest_point_indices(p,p));

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
            equal(radius_neighbor_min(pg,vg,radius,&radg),radius_neighbor_min(p,v,radius,&rad));
            equal(radius_neighbor_min(pg,ig,radius,&radg),radius_neighbor_min(p,iv,radius,&rad));
            equal(point_neighbor_spacing(pg,radius),point_neighbor_spacing(p,radius),true);
            equal(radius_neighbor_counts(pg,rg,radg,9999),radius_neighbor_counts(p,r,rad,9999));
            equal(radius_neighbor_counts(pg,rg,radg,3,&qg),radius_neighbor_counts(p,r,rad,3,&q));
            if(n<=257) {
                equal(radius_connected_components(pg,radius,rg),radius_connected_components(p,radius,r));
                equal(mutual_radius_components(pg,radg),mutual_radius_components(p,rad));
            }


        }
    }
    // Compare the public native PPISP path against the canonical CPU math.
    for(int width : {1,17,257}) {
        const int height=19;
        std::vector<float> pixels(size_t(width)*height*3);
        for(size_t i=0;i<pixels.size();++i) pixels[i]=float(int(i%113)-7)/97.f;
        auto cpu=Tensor::from_vector(pixels,{3,size_t(height),size_t(width)},Device::CPU);
        auto gpu=Tensor::from_vector(pixels,{3,size_t(height),size_t(width)},Device::GPU);
        for(bool adjusted : {false,true}) {
            PpispParams params;
            if(adjusted) {
                params.exposure_factor=1.37f;params.x_offset=3;params.y_offset=5;
                params.full_width=width+7;params.full_height=height+11;
                for(int c=0;c<3;++c) {
                    params.vignetting[c*5]=.02f*c;params.vignetting[c*5+1]=-.01f*c;
                    params.vignetting[c*5+2]=-.15f;params.vignetting[c*5+3]=.03f;
                    params.crf[c*5]=1.13f;params.crf[c*5+1]=.93f;params.crf[c*5+2]=1.07f;
                }
            }
            auto expected=ppisp_apply(cpu,params);auto result=ppisp_apply(gpu,params).cpu();
            for(size_t i=0;i<pixels.size();++i)
                require(std::isfinite(result.ptr<float>()[i]) && std::abs(result.ptr<float>()[i]-expected.ptr<float>()[i])<3e-6f,"HIP PPISP differs from CPU");
        }
    }
    for(size_t n : {size_t(0),size_t(1),size_t(257),size_t(4099)}) {
        for(bool coincident : {false,true}) {
            std::vector<float> xyz(n*3);
            for(size_t i=0;i<n;++i) for(size_t a=0;a<3;++a)
                xyz[i*3+a]=coincident ? 1.f : float(int(((i/3)*37+a*71)%257)-128)*.37f;
            auto cpu=Tensor::from_vector(xyz,{n,3},Device::CPU),gpu=Tensor::from_vector(xyz,{n,3},Device::GPU);
            Tensor expected_keys,actual_keys;
            auto expected=morton_sort_indices(cpu,&expected_keys);
            auto actual=morton_sort_indices(gpu,&actual_keys).cpu();auto observed_keys=actual_keys.cpu();
            require(actual.dtype()==DataType::Int32 && actual_keys.dtype()==DataType::Int64,"HIP Morton wrong output types");
            require(!n || (!std::memcmp(actual.data_ptr(),expected.data_ptr(),n*4) && !std::memcmp(observed_keys.data_ptr(),expected_keys.data_ptr(),n*8)),
                    "HIP Morton keys/order differ from stable CPU order");
        }
    }
    affine_contract();
    low_level_contract();
    std::vector<float> chain(1025*3,0.f);
    for(size_t i=0;i<1025;++i) chain[3*i]=.5f*float(i);
    auto chain_gpu=Tensor::from_vector(chain,{1025,3},Device::GPU);
    auto chain_radii=Tensor::from_vector(std::vector<float>(1025,.6f),{1025},Device::GPU);
    for(unsigned repeat=0;repeat<8;++repeat) {
        auto a=radius_connected_components(chain_gpu,.6f).cpu();
        auto b=mutual_radius_components(chain_gpu,chain_radii).cpu();
        for(size_t i=0;i<1025;++i) require(a.ptr<int32_t>()[i]==0 && b.ptr<int32_t>()[i]==0,"HIP concurrent union chain did not reach minimum root");
    }

    std::puts("PASS HIP spatial, PPISP, stable Morton63 and affine covariance/volume/inplace/batched: hash collisions, masks, self exclusion, saturation, Int32/Float32 minima, direct grid spacing/nearest ties, public mutual radii/spacing/nearest/camera coverage, atomic radius/mutual components, eight concurrent 1025-point chains, nonfinite and empty points");
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"HIP spatial failed: %s\n",e.what());return 1;}

