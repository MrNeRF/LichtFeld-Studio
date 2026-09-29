// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/tensor_metal_reader.hpp"
#include "core/tensor_backend.hpp"
#include <array>
#include <cstdio>
#include <stdexcept>
using namespace lfs::core;
static void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
static void run() {
    const GpuBackendScope scope(GpuBackend::Metal);
    MetalTensorReader reader;
    constexpr size_t count=4097;
    for(int iteration=0;iteration<8;++iteration) {
        Tensor source=Tensor::full({count+2},float(iteration+1),Device::GPU);
        Tensor view=source.slice(0,1,count+1);
        std::array<const Tensor*,2> inputs{&view,nullptr};
        auto snapshot=[reader.device() newBufferWithLength:count*sizeof(float) options:MTLResourceStorageModeShared];
        auto command=reader.submit(inputs,[&](id<MTLCommandBuffer> command,std::span<const MetalTensorView> views) {
            require(views.size()==2 && !views[1].buffer,"Optional tensor view mismatch");
            require(views[0].bytes==count*sizeof(float),"Tensor slice byte count mismatch");
            auto encoder=[command blitCommandEncoder];
            [encoder copyFromBuffer:views[0].buffer sourceOffset:views[0].offset toBuffer:snapshot
                destinationOffset:0 size:views[0].bytes];
            [encoder endEncoding];
        });
        // This writes the SAME source storage immediately after submit. The
        // consumer must see the original producer values, never this mutation.
        source.add_(10.f);
        const auto cpu=source.to(Device::CPU);
        require(cpu.ptr<float>()[1]==float(iteration+11),"Tensor mutation did not finish");
        view=Tensor{};source=Tensor{};
        [command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,"Native tensor consumer failed");
        const auto* values=static_cast<const float*>(snapshot.contents);
        for(size_t i=0;i<count;++i)require(values[i]==float(iteration+1),"Producer/consumer ordering or slice offset mismatch");
    }
    Tensor cpu=Tensor::ones({3},Device::CPU);
    std::array<const Tensor*,1> inputs{&cpu};bool rejected=false;
    try{(void)reader.submit(inputs,[](auto,auto){});}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"CPU tensor was accepted as a native Metal buffer");
    Tensor gpu=Tensor::ones({2,3},Device::GPU);Tensor strided=gpu.transpose(0,1);inputs[0]=&strided;rejected=false;
    try{(void)reader.submit(inputs,[](auto,auto){});}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Noncontiguous tensor was accepted as a flat buffer");
    inputs[0]=&gpu;bool failed=false;
    try{(void)reader.submit(inputs,[](auto,auto){throw std::runtime_error("Encoding failure");});}
    catch(const std::runtime_error&){failed=true;}
    require(failed,"Encoding error was swallowed");
    auto command=reader.submit(inputs,[](auto,auto){});[command waitUntilCompleted];
    require(command.status==MTLCommandBufferStatusCompleted,"Encoding failure poisoned the next submission");
    std::puts("Metal tensor reader contracts passed: resident views, offsets, GPU ordering, mutations and encoding failure recovery.");
}
int main(){@autoreleasepool{
    if(!gpu_backend_available(GpuBackend::Metal)){
        std::puts("SKIP: resident tensor interop requires macOS 26 and Metal 4; native viewer tests remain separate.");return 77;
    }
    try{run();return 0;}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}}
