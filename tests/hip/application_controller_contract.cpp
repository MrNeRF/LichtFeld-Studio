/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor_backend.hpp>
#include <lfs/training/ops/registry.hpp>
#include <lfs/training/ops/ppisp.hpp>
#include <cmath>
#include <cstdio>
#include <stdexcept>
using namespace lfs::core;
using namespace lfs::training;
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
static void close(double a,double b){require(std::isfinite(a)&&std::abs(a-b)<3e-5,"Controller differs from CPU reference");}
int main()try {
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    {
        const auto* ops=training_ops(GpuBackend::HIP).controller;
        require(ops&&ops->prepare_input&&ops->backward_layer,"Incomplete HIP Controller family");
        auto input=Tensor::full({1603},-.25f,Device::GPU);
        Tensor none;
        ops->prepare_input(none,input,.75f);
        auto result=input.cpu().to_vector();
        for(size_t i=0;i<result.size();++i)close(result[i],i==1600?.75:-.25);
        std::vector<float> features(1600);
        for(size_t i=0;i<features.size();++i)features[i]=float(i%31)/31;
        auto f=Tensor::from_vector(features,{1600},Device::GPU);
        ops->prepare_input(f,input,1.f);
        result=input.cpu().to_vector();
        for(size_t i=0;i<1600;++i)close(result[i],features[i]);
        close(result[1600],.75);close(result[1601],-.25);
        ops->prepare_input(f,input,1.25f);close(input.cpu().to_vector()[1600],1.25);
        for(size_t m:{1,17,33})for(size_t n:{1,19,67})for(bool input_grad:{false,true}) {
            std::vector<float> g(m),a(n),w(m*n);
            for(size_t i=0;i<m;++i)g[i]=float(int(i%7)-3)*.13f;
            for(size_t i=0;i<n;++i)a[i]=float(int(i%5)-2)*.2f;
            for(size_t i=0;i<m*n;++i)w[i]=float(int(i%13)-6)*.17f;
            auto gt=Tensor::from_vector(g,{m},Device::GPU),at=Tensor::from_vector(a,{n},Device::GPU),wt=Tensor::from_vector(w,{m,n},Device::GPU);
            auto wg=Tensor::full({m,n},.25f,Device::GPU),bg=Tensor::full({m},-.5f,Device::GPU);
            auto gi=input_grad?Tensor::full({n},123.f,Device::GPU):Tensor{};
            for(int repetition=1;repetition<=2;++repetition) {
                ops->backward_layer(gt,at,wt,wg,bg,gi);
                const auto actual_wg=wg.cpu().to_vector(),actual_bg=bg.cpu().to_vector();
                for(size_t i=0;i<m*n;++i)close(actual_wg[i],.25+repetition*double(g[i/n])*a[i%n]);
                for(size_t i=0;i<m;++i)close(actual_bg[i],-.5+repetition*double(g[i]));
                if(input_grad) {
                    const auto actual_gi=gi.cpu().to_vector();
                    for(size_t col=0;col<n;++col) {
                        double v=0;for(size_t row=0;row<m;++row)v+=double(g[row])*w[row*n+col];
                        close(actual_gi[col],a[col]>0?v:0);
                    }
                }
            }
        }
        auto short_input=Tensor::zeros({1600},Device::GPU);
        bool rejected=false;try{ops->prepare_input(f,short_input,1.f);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"Controller accepted undersized input");
    }
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"Controller contract retained native storage");
    std::puts("PASS HIP Controller CPU input/exposure contract, repeated gradient accumulation, ReLU, optional input gradient, odd dimensions");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
