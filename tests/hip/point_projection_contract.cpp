/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <core/tensor_spatial.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
using namespace lfs::core;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static void near(const Tensor& actual,const Tensor& reference) {
    const auto a=actual.cpu().to_vector(),b=reference.cpu().to_vector();require(a.size()==b.size(),"Projection output size changed");
    for(size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i])&&std::isfinite(b[i]),"Projection produced nonfinite coordinates");
        require(std::abs(a[i]-b[i])<=2e-4f+2e-6f*std::abs(b[i]),"HIP projection differs from shared CPU reference");
    }
}
int main()try {
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    {
        constexpr size_t count=8193;std::vector<float> xyz(count*3);std::vector<int> ids(count);
        for(size_t i=0;i<count;++i){xyz[3*i]=float(int(i%17)-8)*.1f;xyz[3*i+1]=float(int(i%13)-6)*.1f;
            xyz[3*i+2]=-(.5f+float(i%23)*.1f);ids[i]=i%11==0?-1:i%13==0?7:int(i%2);}
        xyz[0]=std::numeric_limits<float>::quiet_NaN();xyz[3]=std::numeric_limits<float>::infinity();
        xyz[6]=xyz[7]=xyz[8]=0;xyz[11]=1;
        const auto points=Tensor::from_vector(xyz,{count,3},Device::CPU),gpu_points=points.to(Device::GPU);
        const auto indices=Tensor::from_vector(ids,{count},Device::CPU),gpu_indices=indices.to(Device::GPU);
        const auto visible=Tensor::from_vector(std::vector<bool>{true,false},{2},Device::CPU),gpu_visible=visible.to(Device::GPU);
        const std::vector<float> matrices{1,0,0,.4f,0,1,0,0,0,0,1,0,0,0,0,1,
                                          1,0,0,0,0,1,0,-.3f,0,0,1,0,0,0,0,1};
        const auto transforms=Tensor::from_vector(matrices,{2,4,4},Device::CPU),gpu_transforms=transforms.to(Device::GPU);
        PointProjection projection{.translation={.1f,-.2f,.3f},.focal_x=120,.focal_y=130,.center_x=190,.center_y=95,
                                   .ortho_scale=20,.width=400,.height=200};
        for(auto model:{PointProjectionModel::Pinhole,PointProjectionModel::Orthographic,PointProjectionModel::Equirectangular})
            for(bool transform:{false,true})for(bool visibility:{false,true}) {
                projection.model=model;
                auto cpu=project_points(points,projection,transform?&transforms:nullptr,&indices,visibility?&visible:nullptr);
                auto gpu=project_points(gpu_points,projection,transform?&gpu_transforms:nullptr,&gpu_indices,visibility?&gpu_visible:nullptr);
                near(gpu,cpu);
                PointRegion2D window{.kind=PointRegion2DKind::Rectangle,.x0=20,.y0=20,.x1=380,.y1=180,.minimum_coordinate=-1000};
                auto expected=Tensor::zeros({count},Device::CPU,DataType::UInt8),actual=Tensor::zeros({count},Device::GPU,DataType::UInt8);
                mark_points_2d(expected,cpu,window);mark_points_2d(actual,gpu,window);
                require(actual.to_vector_uint8()==expected.to_vector_uint8(),"Projection-to-region chained selection differs");
            }
        const auto strided_source=Tensor::from_vector(std::vector<float>{99,0,0,-1,99,1,0,-2,99,0,1,-3},{3,4},Device::CPU);
        const auto strided=strided_source.slice(1,1,4),gpu_strided=strided_source.to(Device::GPU).slice(1,1,4);
        near(project_points(gpu_strided,projection),project_points(strided,projection));
    }
    {
        const auto nan=std::numeric_limits<float>::quiet_NaN(),inf=std::numeric_limits<float>::infinity();
        const std::vector<float> xy{.5f,.5f,2,.5f,.5f,2,2,2,5,5,nan,0,inf,0,-1e8f,0,0,0};
        const auto points=Tensor::from_vector(xy,{9,2},Device::CPU),gpu_points=points.to(Device::GPU);
        const auto polygon=Tensor::from_vector(std::vector<float>{0,0,3,0,3,1,1,1,1,3,0,3},{6,2},Device::CPU);
        const auto centers=Tensor::from_vector(std::vector<float>{2,2,5,5},{2,2},Device::CPU);
        for(auto kind:{PointRegion2DKind::Disk,PointRegion2DKind::Rectangle,PointRegion2DKind::Polygon,PointRegion2DKind::Disks}) {
            PointRegion2D region{.kind=kind,.x0=0,.y0=0,.x1=1,.y1=1,.radius=kind==PointRegion2DKind::Disks?0.f:1.f,.minimum_coordinate=-1000};
            const auto geometry=kind==PointRegion2DKind::Polygon?polygon:centers,gpu_geometry=geometry.to(Device::GPU);
            std::vector<float> bytes(22,77);for(size_t i=0;i<9;++i)bytes[2+2*i]=7;
            auto cpu_parent=Tensor::from_vector(bytes,{11,2},Device::CPU).to(DataType::UInt8),gpu_parent=cpu_parent.to(Device::GPU);
            auto expected=cpu_parent.slice(0,1,10).slice(1,0,1).squeeze(1),actual=gpu_parent.slice(0,1,10).slice(1,0,1).squeeze(1);
            mark_points_2d(expected,points,region,&geometry);mark_points_2d(actual,gpu_points,region,&gpu_geometry);
            require(gpu_parent.flatten().to_vector_uint8()==cpu_parent.flatten().to_vector_uint8(),"Region boundary, invalid coordinate or strided output mismatch");
            if(kind==PointRegion2DKind::Polygon||kind==PointRegion2DKind::Disks){
                const auto before=actual.to_vector_uint8();const auto empty=Tensor::empty({0,2},Device::GPU);
                mark_points_2d(actual,gpu_points,region,&empty);require(actual.to_vector_uint8()==before,"Empty region geometry changed the mask");
            }
        }
    }
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"HIP projection context retained resources");
    std::puts("PASS HIP point projection/regions: 12 camera/transform/visibility cases, chained selection, tails, strides, disk/rectangle/polygon/disks and invalid coordinates");
    return 0;
}catch(const std::exception& error){std::fprintf(stderr,"FAIL %s\n",error.what());return 1;}
