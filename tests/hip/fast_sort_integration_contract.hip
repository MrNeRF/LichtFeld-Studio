/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Windows standalone native-module comparison; no Core, CUDA or Torch.
#define NOMINMAX
#include <hip/hip_runtime.h>
#include "../../src/training/hip/application_fast_api.h"
#include <windows.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <utility>
static void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
struct Module {
    HMODULE dll;
    decltype(&lfs_hip_application_fast_create) create;
    decltype(&lfs_hip_application_fast_forward) forward;
    decltype(&lfs_hip_application_fast_backward) backward;
    decltype(&lfs_hip_application_fast_info) info;
    decltype(&lfs_hip_application_fast_destroy) destroy;
    explicit Module(const char* path):dll(LoadLibraryA(path)){
        require(dll,"Cannot load training module");
#define LOAD(name) name=reinterpret_cast<decltype(name)>(GetProcAddress(dll,"lfs_hip_application_fast_" #name));require(name,"Missing native FastGS function")
        LOAD(create);LOAD(forward);LOAD(backward);LOAD(info);LOAD(destroy);
#undef LOAD
    }
    ~Module(){FreeLibrary(dll);}
};
struct Allocations {
    std::vector<void*> pointers;
    ~Allocations(){for(auto p:pointers)(void)hipFree(p);}
    uint64_t floats(size_t n,const float* input=nullptr){float* p;check(hipMalloc(&p,n*sizeof(float)));pointers.push_back(p);if(input)check(hipMemcpy(p,input,n*sizeof(float),hipMemcpyHostToDevice));else check(hipMemset(p,0,n*sizeof(float)));return reinterpret_cast<uint64_t>(p);}
};
int main(int argc,char** argv)try{
    require(argc==3 || (argc==4 && !std::strcmp(argv[3],"--profile")),
            "Pass absolute baseline and candidate training modules, optionally --profile");
    const bool compare_profiles=argc==4;
    check(hipSetDevice(0));hipStream_t stream;check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    Module baseline(argv[1]),candidate(argv[2]);
    constexpr unsigned attributes[]{3,3,4,1,3};
    struct Case { int width,height; unsigned count,bases; bool channels,empty; unsigned tight=0; };
    for(const auto fixture: {Case{1,1,40019,1,false,false}, Case{65,33,40019,1,false,false},
                            Case{193,129,40019,1,false,false}, Case{1,1,32767,1,false,false},
                            Case{1,1,32768,1,false,false}, Case{65,33,40019,16,false,false},
                            Case{65,33,40019,4,true,false}, Case{65,33,40019,1,false,true},
                            Case{193,129,1000000,1,false,false}, Case{1,1,40019,1,false,false,1},
                            Case{1,1,40019,1,false,false,2}}){
        // These admission assertions assume the original sort's workspace
        // layout. Kernel profiles can also use different radix scratch sizes.
        if(compare_profiles && fixture.tight)continue;
        const auto width=fixture.width,height=fixture.height;const auto n=fixture.count;Allocations memory;
        std::array<std::vector<float>,5> host;
        for(unsigned g=0;g<5;++g)host[g].resize(n*attributes[g]);
        for(unsigned i=0;i<n;++i){
            host[0][i*3]=width==1?0.f:float(int((i*37)%191)-95)*.009f;
            host[0][i*3+1]=height==1?0.f:float(int((i*71)%173)-86)*.008f;
            host[0][i*3+2]=(fixture.empty || (width>1 && i%19==0))?-3.f:(i%7==0?3.5f:3.f+float((i*1237)%997)*.001f);
            for(unsigned c=0;c<3;++c){host[1][i*3+c]=std::log(.16f+.013f*float((i+c)%11));host[4][i*3+c]=float(int((i*13+c*31)%97)-48)*.015f;}
            host[2][i*4]=1;host[3][i]=-4.f+float(i%7)*.03f;
        }
        std::array<uint64_t,5> parameters;for(unsigned g=0;g<5;++g)parameters[g]=memory.floats(host[g].size(),host[g].data());
        float identity[16]{};for(int i=0;i<4;++i)identity[i*5]=1;
        LfsHipFastForward f{};f.struct_bytes=sizeof(f);f.means=parameters[0];f.scales=parameters[1];f.rotations=parameters[2];f.opacities=parameters[3];f.sh0=parameters[4];
        f.view=memory.floats(16,identity);f.camera=memory.floats(3);f.count=n;f.active_bases=f.layout_bases=fixture.bases;f.width=width;f.height=height;f.fx=f.fy=64;f.cx=width*.5f;f.cy=height*.5f;f.dilation=1;
        const size_t sh_slots=((fixture.bases-1u)*3u+3u)/4u;
        const size_t sh_floats=((n+31u)/32u)*32u*sh_slots*4u;
        if(fixture.bases>1){std::vector<float> sh(sh_floats);for(size_t i=0;i<sh.size();++i)sh[i]=float(int(i%19)-9)*.001f;f.shN=memory.floats(sh.size(),sh.data());}
        const size_t pixels=size_t(width)*height;
        f.image=memory.floats(3*pixels);f.alpha=memory.floats(pixels);
        if(fixture.channels){f.depth=memory.floats(pixels);f.normal=memory.floats(3*pixels);}
        std::vector<float> gradient(3*pixels);for(size_t i=0;i<gradient.size();++i)gradient[i]=float(int(i%37)-18)*.001f;
        LfsHipFastBackward b{};b.struct_bytes=sizeof(b);b.grad_image=memory.floats(gradient.size(),gradient.data());b.beta1=.9f;b.beta2=.999f;b.eps=1e-8f;
        for(unsigned g=0;g<5;++g){auto& group=b.groups[g];group.parameter=parameters[g];group.gradient_out=memory.floats(n*attributes[g]);group.primitives=n;group.attributes=attributes[g];group.elements=n*attributes[g];group.joint_bits=16;}
        if(fixture.bases>1){auto& group=b.groups[5];group.parameter=f.shN;group.gradient_out=memory.floats(sh_floats);group.primitives=n;group.attributes=(fixture.bases-1)*3;group.elements=sh_floats;group.joint_bits=8;}
        if(fixture.channels){b.grad_depth=memory.floats(pixels,gradient.data());b.grad_normal=memory.floats(3*pixels,gradient.data());}
        std::vector<float> reference_image,reference_alpha,reference_depth,reference_normal;std::array<std::vector<float>,6> reference_gradients;
        uint64_t reference_peak=0;
        for(int variant=0;variant<2;++variant){auto& module=variant?candidate:baseline;LfsHipApplicationFast* state=nullptr;LfsHipError error{};
            auto invoke=[&](int code){if(code)throw std::runtime_error(error.message);};
            const uint64_t budget=fixture.tight && variant?reference_peak+65536+(fixture.tight==2?n*64ull:0):1024ull*1024*1024;
            invoke(module.create(0,reinterpret_cast<uint64_t>(stream),budget,&state,&error));
            invoke(module.forward(state,&f,&error));check(hipStreamSynchronize(stream));
            // The second forward checks that a workspace which fell back remains
            // reusable and does not retain the optional buffers or retry forever.
            if(fixture.tight){invoke(module.forward(state,&f,&error));check(hipStreamSynchronize(stream));}
            LfsHipFastInfo info{sizeof(info)};invoke(module.info(state,&info,&error));
            if(!variant)reference_peak=info.peak_bytes;
            require(info.live_bytes<=budget && info.peak_bytes<=budget,"Workspace exceeded its allocation budget");
            if(fixture.empty)require(info.visible==0,"Empty fixture unexpectedly visible");
            else if(n>=32768)require(info.visible>=32768,"Fixture does not exercise depth presort");
            else require(info.visible<32768,"Small fixture did not exercise fallback");
            auto download=[&](uint64_t address,size_t count){std::vector<float> data(count);check(hipMemcpy(data.data(),reinterpret_cast<void*>(address),count*sizeof(float),hipMemcpyDeviceToHost));return data;};
            double max_forward_error=0;
            auto matches=[&](const auto& actual,const auto& reference){
                if(!compare_profiles)return actual==reference;
                require(actual.size()==reference.size(),"Profile output shape changed");
                for(size_t i=0;i<actual.size();++i){
                    const double difference=std::abs(double(actual[i])-reference[i]);
                    max_forward_error=std::max(max_forward_error,difference);
                    if(!std::isfinite(actual[i]) || !std::isfinite(reference[i]) ||
                       difference>2e-5+2e-5*std::abs(reference[i]))return false;
                }
                return true;
            };
            auto image=download(f.image,3*pixels),alpha=download(f.alpha,pixels);
            if(!variant){reference_image=image;reference_alpha=alpha;}else require(matches(image,reference_image) && matches(alpha,reference_alpha),"Forward pixels/alpha differ beyond comparison policy");
            if(fixture.channels){auto depth=download(f.depth,pixels),normal=download(f.normal,3*pixels);if(!variant){reference_depth=depth;reference_normal=normal;}else require(matches(depth,reference_depth) && matches(normal,reference_normal),"Forward depth/normal differ beyond comparison policy");}
            if(fixture.tight){
                if(variant)require(info.live_bytes==reference_peak,"Fallback retained optional depth-sort workspace");
                for(unsigned g=0;g<5;++g)require(download(parameters[g],host[g].size())==host[g],"Forward/fallback modified model parameters");
            }
            if(fixture.tight==1){
                invoke(module.destroy(state,&error));
                std::printf("variant=%d tight_budget=%llu live=%llu peak=%llu repeated_exact_forward=1\n",variant,(unsigned long long)budget,(unsigned long long)info.live_bytes,(unsigned long long)info.peak_bytes);
                continue;
            }
            for(auto& group:b.groups)if(group.gradient_out)check(hipMemsetAsync(reinterpret_cast<void*>(group.gradient_out),0,group.elements*sizeof(float),stream));
            invoke(module.backward(state,&b,&error));check(hipStreamSynchronize(stream));
            if(fixture.tight==2){invoke(module.info(state,&info,&error));require(info.peak_bytes<=budget,"Backward admission exceeded budget");}
            double max_error=0;
            for(unsigned g=0;g<6;++g){if(!b.groups[g].gradient_out)continue;auto actual=download(b.groups[g].gradient_out,b.groups[g].elements);if(!variant)reference_gradients[g]=actual;else for(size_t i=0;i<actual.size();++i){double difference=std::abs(double(actual[i])-reference_gradients[g][i]);max_error=std::max(max_error,difference);require(std::isfinite(actual[i]) && difference<=2e-5+2e-5*std::abs(reference_gradients[g][i]),"Raw gradient changed beyond atomic accumulation tolerance");}}
            invoke(module.destroy(state,&error));
            std::printf("variant=%d image=%dx%d bases=%u channels=%d visible=%d instances=%d workspace_peak=%llu max_gradient_error=%.9g max_forward_error=%.9g exact_forward=%d\n",variant,width,height,fixture.bases,fixture.channels,info.visible,info.instances,(unsigned long long)info.peak_bytes,max_error,max_forward_error,!compare_profiles);std::fflush(stdout);
        }
    }
    check(hipStreamDestroy(stream));std::puts(compare_profiles?
        "PASS native FastGS profile comparison: bounded forward and gradient errors":
        "PASS large native FastGS sort integration: exact images/alpha and bounded gradients");
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
