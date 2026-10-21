/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <core/tensor_filters.hpp>
#include <core/tensor_labels.hpp>
#include <core/tensor_histogram.hpp>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <stdexcept>
using namespace lfs::core;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static void exercise(Device device) {
    const auto bytes=[&](std::vector<float> v){return Tensor::from_vector(v,{v.size()},device).to(DataType::UInt8);};
    const auto floats=[&](std::vector<float> v,TensorShape shape){return Tensor::from_vector(v,shape,device);};
    const auto matrix=floats({1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1},{4,4});
    {
        const auto a=Tensor::from_vector(std::vector<int>{-3,0,1,16777216,16777217,2147483647},{6},device);
        const auto b=Tensor::from_vector(std::vector<int>{-3,1,0,16777217,16777216,2147483646},{6},device);
        require(a.ne(b).to_vector_bool()==std::vector<bool>{false,true,true,true,true,true},"Int32 tensor predicate lost integer precision");
        require(a.eq(16777217).to_vector_bool()==std::vector<bool>{false,false,false,false,true,false},"Int32 scalar predicate lost integer precision");
        require(a.eq(1.9f).to_vector_bool()==std::vector<bool>(6,false),"Mixed float/integer predicate did not follow public promotion semantics");
        constexpr int64_t large=int64_t{1}<<55;
        const auto wide_tensor=[&](std::vector<int64_t> values){
            auto cpu=Tensor::empty({values.size()},Device::CPU,DataType::Int64);
            std::copy(values.begin(),values.end(),cpu.ptr<int64_t>());return cpu.to(device);
        };
        const auto wide=wide_tensor({large,large+1,large+2,-large});
        const auto other=wide_tensor({large+1,large,large+2,large});
        require(wide.ne(other).to_vector_bool()==std::vector<bool>{true,true,false,true},"Int64 tensor comparison passed through floating point");
        const auto u=Tensor::from_vector(std::vector<float>{0,1,65535},{3},device).to(DataType::UInt32);
        const auto u_limit=floats({1,1,1},{3}).to(DataType::UInt32);
        require(u.gt(u_limit).to_vector_bool()==std::vector<bool>{false,false,true},"UInt32 predicate mismatch");
        const auto half=floats({-1,0,1,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()},{5}).to(DataType::Float16);
        const auto half_zero=Tensor::zeros({5},device).to(DataType::Float16);
        require(half.gt(half_zero).to_vector_bool()==std::vector<bool>{false,false,true,true,false},"Float16 predicate NaN/infinity mismatch");
    }
    {
        std::vector<float> labels(8193);std::vector<int> expected(260,0);
        for(size_t i=0;i<labels.size();++i){const auto value=int(i%256);labels[i]=float(value);if(value)++expected[value];}
        const auto values=bytes(labels);auto parent=Tensor::full({268},99,device,DataType::Int32);
        auto counts=parent.slice(0,4,264);histogram_u8(values,counts);
        const auto result=parent.to_vector_int();
        require(std::equal(expected.begin(),expected.end(),result.begin()+4),"Byte histogram counts or zero bin mismatch");
        require(result[0]==99&&result[3]==99&&result[264]==99&&result[267]==99,"Histogram changed adjacent bins");
        histogram_u8(bytes({}),counts);require(counts.to_vector_int()==std::vector<int>(260,0),"Empty histogram did not clear bins");
    }
    {
        const auto selected=bytes({1,1,0,1,1,0,1,1,1}),existing=bytes({2,7,1,0,255,7,0,31,1});
        const auto categories=Tensor::from_vector(std::vector<int>{0,0,0,1,0,-1,2,9,0},{9},device);
        const auto allowed=bytes({1,0,1});
        std::vector<bool> flags(256,false);for(int label:{1,2,31,255})flags[label]=true;
        const auto locked=Tensor::from_vector(flags,{256},device);
        const std::vector<std::vector<uint8_t>> expected{{2,1,1,0,255,7,1,31,1},{2,7,1,0,255,7,0,31,0},{2,1,0,0,255,7,1,31,1}};
        for(int offset=0;offset<4;++offset)for(int mode=0;mode<3;++mode) {
            auto storage=Tensor::full({16},99,device,DataType::UInt8);auto output=storage.slice(0,offset,offset+9);
            update_labels(output,selected,{.label=1,.mode=static_cast<LabelUpdateMode>(mode),.existing=&existing,
                .locked=&locked,.categories=&categories,.allowed=&allowed});
            std::vector<uint8_t> reference(16,99);std::copy(expected[mode].begin(),expected[mode].end(),reference.begin()+offset);
            require(storage.to_vector_uint8()==reference,"Dense labels changed modes, locks, category restrictions or adjacent bytes");
        }
        auto alias=bytes({1,0,3});update_labels(alias,alias,{.label=4,.existing=&alias});
        require(alias.to_vector_uint8()==std::vector<uint8_t>{4,0,4},"Aliased label input changed before consumption");
    }
    {
        const auto selected=bytes({0,1,1,1,1,1,1}),existing=bytes({0,255,1,1,1,1,1,1,1,1});
        const auto indices=Tensor::from_vector(std::vector<int>{4,4,0,9,-1,99,1},{7},device);
        std::vector<bool> flags(256,false);flags[255]=true;const auto locks=Tensor::from_vector(flags,{256},device);
        auto out=Tensor::empty({10},device,DataType::UInt8);
        update_labels(out,selected,{.label=1,.mode=LabelUpdateMode::Replace,.existing=&existing,.locked=&locks,.indices=&indices});
        require(out.to_vector_uint8()==std::vector<uint8_t>{1,255,1,1,1,1,1,1,1,1},"Indexed label replacement lost selected duplicates or untouched labels");
        const auto empty=bytes({}),categories=Tensor::zeros({7},device,DataType::Int32);
        update_labels(out,selected,{.label=2,.existing=&existing,.indices=&indices,.categories=&categories,.allowed=&empty});
        require(out.to_vector_uint8()==existing.to_vector_uint8(),"Empty label category table changed existing values");
        auto parent=bytes({1,99,2,99,3,99}).reshape({3,2});auto strided=parent.slice(1,0,1).squeeze(1);
        const auto sel=bytes({1,0,2});update_labels(strided,sel,{.label=2,.mode=LabelUpdateMode::Replace,.existing=&strided});
        require(parent.flatten().to_vector_uint8()==std::vector<uint8_t>{2,99,0,99,2,99},"Strided label output changed adjacent bytes");
        auto live_mask=bytes({255,7,0,3}),live=bytes({1,0,255,2});live_mask.and_live_(live);
        require(live_mask.to_vector_uint8()==std::vector<uint8_t>{255,0,0,3},"Live mask filtering changed retained labels");
        live_mask.and_live_(live_mask);
        require(live_mask.to_vector_uint8()==std::vector<uint8_t>{255,0,0,3},"Aliased live mask changed its own labels");
    }
    {
        const auto ids=Tensor::from_vector(std::vector<int>{0,1,-1,3,0,9},{6},device);
        const auto allowed=bytes({255,0,1,2}),empty=bytes({});
        for(int offset=0;offset<4;++offset) {
            auto storage=Tensor::full({12},255,device,DataType::UInt8);
            auto mask=storage.slice(0,offset,offset+6);
            filter_points(mask,nullptr,{.indices=&ids,.allowed=&allowed});
            std::vector<uint8_t> expected(12,255);expected[offset+1]=expected[offset+2]=expected[offset+5]=0;
            require(storage.to_vector_uint8()==expected,"Node filter changed offset bytes or labels");
            filter_points(mask,nullptr,{.indices=&ids,.allowed=&empty});
            require(mask.to_vector_uint8()==std::vector<uint8_t>(6,0),"Empty allowed table did not reject every node");
        }
        auto alias=bytes({1,1,1});
        const auto indices=Tensor::from_vector(std::vector<int>{3,0,0},{3},device);
        filter_points(alias,nullptr,{.indices=&indices,.allowed=&alias});
        require(alias.to_vector_uint8()==std::vector<uint8_t>{0,1,1},"Aliased allowed flags changed during filtering");
    }
    const auto lo=floats({-1,-1,-1},{3}),hi=floats({1,1,1},{3});
    {
        const auto points=floats({-2,0,0,-1,0,0,0,0,0,1,0,0,2,0,0,0,2,0},{6,3});
        const auto radii=floats({1,1,1},{3});
        for(bool inverse:{false,true})for(bool ellipsoid:{false,true}) {
            auto mask=bytes({1,2,3,4,5,6});
            PointFilter filter{.box_inverse=inverse,.ellipsoid_inverse=inverse};
            if(ellipsoid){filter.ellipsoid_transform=&matrix;filter.ellipsoid_radii=&radii;}
            else{filter.box_transform=&matrix;filter.box_min=&lo;filter.box_max=&hi;}
            filter_points(mask,&points,filter);
            require(mask.to_vector_uint8()==(inverse?std::vector<uint8_t>{1,0,0,0,5,6}:std::vector<uint8_t>{0,2,3,4,0,0}),
                    "Shape filter boundary or inversion mismatch");
        }
        auto storage=bytes({255,77,128,77,2,77,1,77}).reshape({4,2});
        auto mask=storage.slice(1,0,1).squeeze(1);
        const auto strided=floats({-2,0,0,99,0,0,0,99,1,0,0,99,2,0,0,99},{4,4}).slice(1,0,3);
        filter_points(mask,&strided,{.box_transform=&matrix,.box_min=&lo,.box_max=&hi});
        require(storage.flatten().to_vector_uint8()==std::vector<uint8_t>{0,77,128,77,2,77,0,77},"Strided filter changed adjacent bytes");
    }
    {
        const auto points=floats({0,0,-1,0,0,-2,0,0,-3,0,0,1,1,0,-1,0,0,0},{6,3});
        PointFilterWindow window{.projection={.focal_x=10,.focal_y=10,.center_x=60,.center_y=50,.ortho_scale=10,.width=100,.height=100},
                                .near_depth=1,.far_depth=2,.scale_x=.4f,.scale_y=.5f};
        for(auto model:{PointProjectionModel::Pinhole,PointProjectionModel::Orthographic,PointProjectionModel::Equirectangular}) {
            window.projection.model=model;auto mask=bytes({1,2,3,4,5,6});
            filter_points(mask,&points,{.window=&window});
            require(mask.to_vector_uint8()==std::vector<uint8_t>{1,2,0,0,5,0},"Projection/depth window mismatch");
        }
    }
    {
        std::vector<Tensor> outputs;
        for(int call=0;call<24;++call) {
            auto mask=Tensor::full({8193},float(call+1),device,DataType::UInt8);
            const auto points=Tensor::zeros({8193,3},device);
            const auto minimum=floats({float(call%2),-1,-1},{3});
            filter_points(mask,&points,{.box_transform=&matrix,.box_min=&minimum,.box_max=&hi});
            outputs.push_back(mask);
        }
        for(int call=0;call<24;++call)
            require(outputs[call].to_vector_uint8()==std::vector<uint8_t>(8193,call%2?0:call+1),"Queued temporary filter inputs did not survive");
    }
}
int main()try {
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    exercise(Device::CPU);exercise(Device::GPU);
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"HIP filter context retained resources");
    std::puts("PASS HIP selection: label modes/locks/categories, indexed duplicates, alias/stride/offset, live masks, node filters, shapes/inversion, three projections and queued temporary inputs");
    return 0;
}catch(const std::exception& error){std::fprintf(stderr,"FAIL %s\n",error.what());return 1;}
