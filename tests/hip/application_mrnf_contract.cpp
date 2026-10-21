/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor_backend.hpp>
#include <lfs/training/ops/registry.hpp>
#include <lfs/training/ops/mrnf.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <stdexcept>
using namespace lfs::core;
using namespace lfs::training;
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
static void close(double a,double b,double t=2e-5){require(std::isfinite(a)&&std::abs(a-b)<=t,"MRNF differs from CPU reference");}
int main()try {
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    {
        const auto* ops=training_ops(GpuBackend::HIP).mrnf;
        require(ops&&ops->noise&&ops->decay&&ops->percentile_bounds&&ops->gumbel&&ops->fold_error&&ops->compact_bool_indices&&ops->prune_bounds&&ops->replace_parent_weights,"Incomplete HIP MRNF family");
        constexpr size_t n=257;
        std::vector<float> means(3*n),opacity(n),visibility(n),scales(3*n),edge(n);
        std::vector<int> flags(n),active_flags(n);
        for(size_t i=0;i<n;++i) {
            opacity[i]=-10;visibility[i]=i%5?1.f:0.f;edge[i]=float(i%7)/7;flags[i]=i%11==0;active_flags[i]=i%3!=0;
            for(size_t j=0;j<3;++j){means[3*i+j]=float(int((i*31+j*17)%503)-251)*.01f;scales[3*i+j]=float(int((i+j)%9)-4)*.2f;}
        }
        auto mt=Tensor::from_vector(means,{n,3},Device::GPU),ot=Tensor::from_vector(opacity,{n},Device::GPU),vt=Tensor::from_vector(visibility,{n},Device::GPU),st=Tensor::from_vector(scales,{n,3},Device::GPU);
        auto frozen=Tensor::from_vector(flags,{n},Device::GPU).to(DataType::Bool),active=Tensor::from_vector(active_flags,{n},Device::GPU).to(DataType::Bool),et=Tensor::from_vector(edge,{n},Device::GPU);
        auto weights=Tensor::full({n},123.f,Device::GPU);
        std::puts("MRNF parent");std::fflush(stdout);
        ops->replace_parent_weights(ot,vt,active,frozen,et,weights);
        auto result=weights.cpu().to_vector();
        for(size_t i=0;i<n;++i)close(result[i],opacity[i]*(visibility[i]>0)*active_flags[i]*flags[i]*edge[i]);
        std::puts("MRNF noise");std::fflush(stdout);
        auto first=mt.clone(),second=mt.clone();
        lfs::gpu_ops::MrnfNoiseParams np{1234567,.03f,10.f,.2f};
        ops->noise(first,ot,vt,frozen,np);ops->noise(second,ot,vt,frozen,np);
        const auto noisy=first.cpu().to_vector(),repeated=second.cpu().to_vector();require(noisy==repeated,"MRNF seeded noise is not reproducible");
        bool changed=false;
        for(size_t i=0;i<n;++i)for(size_t j=0;j<3;++j) {
            const auto index=3*i+j;
            if(flags[i]||visibility[i]<=0)close(noisy[index],means[index],0);
            else {require(std::abs(noisy[index]-means[index])<=.200001f,"MRNF noise exceeded clamp");changed|=noisy[index]!=means[index];}
        }
        require(changed,"MRNF noise never changed a trainable visible mean");
        std::puts("MRNF decay");std::fflush(stdout);
        lfs::gpu_ops::DecayParams dp{.03f,.1f,.25f,vt};
        ops->decay(ot,st,frozen,dp);
        const auto decayed_opacity=ot.cpu().to_vector(),decayed_scales=st.cpu().to_vector();
        for(size_t i=0;i<n;++i) {
            double p=1/(1+std::exp(-double(opacity[i])));
            if(!flags[i]&&visibility[i]>0){p=std::clamp(p-.03*.75,1e-12,double(std::nextafter(1.f,0.f)));close(decayed_opacity[i],std::log(p/(1-p)));}
            else close(decayed_opacity[i],opacity[i],0);
            for(size_t j=0;j<3;++j)close(decayed_scales[3*i+j],flags[i]?scales[3*i+j]:std::log(std::max(std::exp(double(scales[3*i+j]))*(1-.1*.75),1e-12)));
        }
        std::puts("MRNF bounds");std::fflush(stdout);
        for(float percentile:{0.f,.5f,.97f,1.f}) {
            const auto bounds=ops->percentile_bounds(mt,percentile);
            const auto lo=size_t(((1.f-percentile)*.5f)*float(n-1)),hi=size_t((1.f-(1.f-percentile)*.5f)*float(n-1));std::array<float,3> ext{};
            for(size_t j=0;j<3;++j) {
                std::vector<float> axis(n);for(size_t i=0;i<n;++i)axis[i]=means[3*i+j];std::sort(axis.begin(),axis.end());
                close(bounds.center[j],(axis[lo]+axis[hi])*.5f,0);close(bounds.extent[j],(axis[hi]-axis[lo])*.5f,0);ext[j]=bounds.extent[j];
            }
            std::sort(ext.begin(),ext.end());close(bounds.median_size,2*ext[1],0);close(bounds.max_extent,ext[2],0);
        }
        std::puts("MRNF fold");std::fflush(stdout);
        auto maximum=Tensor::full({n},.5f,Device::GPU);
        std::vector<float> densification(2*n);for(size_t i=0;i<2*n;++i)densification[i]=float(i%13)*.1f;
        auto dens=Tensor::from_vector(densification,{2,n},Device::GPU);ops->fold_error(maximum,dens);
        const auto folded=maximum.cpu().to_vector(),cleared=dens.cpu().to_vector();
        for(size_t i=0;i<n;++i){close(folded[i],std::max(.5f,densification[n+i]),0);close(cleared[i],densification[i],0);close(cleared[n+i],0,0);}
        std::puts("MRNF compact");std::fflush(stdout);
        auto output=Tensor::full({7},-9.f,Device::GPU).to(DataType::Int64);
        const auto total=ops->compact_bool_indices(frozen,output,7);std::vector<int64_t> expected;
        for(size_t i=0;i<n;++i)if(flags[i])expected.push_back(i);
        require(total==expected.size(),"MRNF compaction total differs from CPU");
        const auto written=output.cpu().to_vector_int64();for(size_t i=0;i<7;++i)require(written[i]==expected[i],"MRNF bounded compaction differs from CPU");
        std::puts("MRNF prune");std::fflush(stdout);
        auto mask=active.clone();auto scale=Tensor::full({n},-2.f,Device::GPU);
        ops->prune_bounds(mt,scale,mask,{0,0,0},2.f,-1.f);
        const auto pruned=mask.cpu().to_vector_bool();for(size_t i=0;i<n;++i)require(pruned[i]==bool(active_flags[i]||std::max({std::abs(means[3*i]),std::abs(means[3*i+1]),std::abs(means[3*i+2])})>2.f),"MRNF prune differs from CPU");
        std::vector<float> sparse(n,0.f);for(size_t i=0;i<n;++i)if(i%5==0)sparse[i]=float(i+1);
        auto sw=Tensor::from_vector(sparse,{n},Device::GPU);
        std::puts("MRNF gumbel");std::fflush(stdout);
        for(bool compact:{false,true}) {
            auto indices=Tensor::empty({17},Device::GPU,DataType::Int64),again=Tensor::empty({17},Device::GPU,DataType::Int64);
            lfs::gpu_ops::GumbelParams gp{9876543,0,compact};
            ops->gumbel(nullptr,sw,indices,gp);ops->gumbel(nullptr,sw,again,gp);
            const auto chosen=indices.cpu().to_vector_int64();require(chosen==again.cpu().to_vector_int64(),"MRNF seeded Gumbel is not reproducible");
            require(std::set<int64_t>(chosen.begin(),chosen.end()).size()==chosen.size(),"MRNF Gumbel returned duplicate indices");
            for(auto index:chosen)require(index>=0&&index<int64_t(n)&&sparse[index]>0,"MRNF Gumbel selected a zero-weight row");
        }
        auto all=Tensor::empty({n},Device::GPU,DataType::Int64);ops->gumbel(nullptr,sw,all,{1,0,true});
        const auto identity=all.cpu().to_vector_int64();for(size_t i=0;i<n;++i)require(identity[i]==int64_t(i),"MRNF full Gumbel is not identity");
    }
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"MRNF contract retained native storage");
    std::puts("PASS HIP MRNF all callbacks: CPU bounds/decay/fold/parent/prune/compaction, deterministic noise and dense/sparse Gumbel");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());(void)shutdown_gpu_backend(GpuBackend::HIP);return 1;}
