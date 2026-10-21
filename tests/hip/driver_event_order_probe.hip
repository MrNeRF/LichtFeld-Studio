/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Standalone diagnostic, no Torch. Exercises the ordering pattern discussed
// in AMD rocm-systems #12571/#12952; no runtime workaround is enabled here.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <windows.h>
#include "../../src/core/tensor/backend/hip/runtime_api.h"
static void check(hipError_t status) {
    if(status!=hipSuccess) { std::fprintf(stderr,"HIP error %s\n",hipGetErrorString(status));std::exit(2); }
}
struct Payload { unsigned* values; unsigned count; unsigned value; };
__global__ void delay() {
    const auto start=clock64();
    while(clock64()-start<2000000ull) {}
}
__global__ void write_struct(Payload payload) {
    const auto start=clock64();
    while(clock64()-start<2000000ull) {}
    for(unsigned i=threadIdx.x;i<payload.count;i+=blockDim.x) payload.values[i]=payload.value;
}
__global__ void write_pointer(unsigned* values,unsigned count,unsigned value) {
    const auto start=clock64();
    while(clock64()-start<2000000ull) {}
    for(unsigned i=threadIdx.x;i<count;i+=blockDim.x) values[i]=value;
}
struct ReaderPayload { const unsigned* input; unsigned* output; unsigned count; };
__global__ void read_struct(ReaderPayload payload) {
    for(unsigned i=threadIdx.x;i<payload.count;i+=blockDim.x) payload.output[i]=payload.input[i]+1;
}
int main(int argc,char** argv) {
    if(argc>2)return 2;
    HMODULE module=nullptr;
    LfsHipContext* context=nullptr;LfsHipBuffer* buffer=nullptr;LfsHipError error{};
    decltype(&lfs_hip_upload) upload_api=nullptr;
    decltype(&lfs_hip_download) download_api=nullptr;
    decltype(&lfs_hip_buffer_release) release_api=nullptr;
    decltype(&lfs_hip_context_destroy) destroy_api=nullptr;
    auto native_check=[&](int code) { if(code) {std::fprintf(stderr,"Runtime API error %s\n",error.message);std::exit(2);} };

    const unsigned count=std::getenv("LFS_HIP_PROBE_ELEMENTS") ? unsigned(std::strtoul(std::getenv("LFS_HIP_PROBE_ELEMENTS"),nullptr,10)) : 20480;
    if(!count || count>10u*1024u*1024u)return 2;
    constexpr unsigned repeats=16;
    unsigned* device=nullptr;unsigned* pinned=nullptr;
    check(hipMalloc(&device,count*sizeof(unsigned)));
    check(hipHostMalloc(&pinned,count*sizeof(unsigned)));
    std::vector<unsigned> pageable(count);
    hipStream_t explicit_queue;
    if(argc==2) {
        check(hipFree(device));device=nullptr;
        module=LoadLibraryA(argv[1]);if(!module)return 2;
        auto symbol=[&](const char* name) { auto value=GetProcAddress(module,name);if(!value)std::exit(2);return value; };
        auto create=reinterpret_cast<decltype(&lfs_hip_context_create)>(symbol("lfs_hip_context_create"));
        auto allocate=reinterpret_cast<decltype(&lfs_hip_buffer_allocate)>(symbol("lfs_hip_buffer_allocate"));
        auto address=reinterpret_cast<decltype(&lfs_hip_buffer_address)>(symbol("lfs_hip_buffer_address"));
        auto queue=reinterpret_cast<decltype(&lfs_hip_context_queue)>(symbol("lfs_hip_context_queue"));
        upload_api=reinterpret_cast<decltype(upload_api)>(symbol("lfs_hip_upload"));
        download_api=reinterpret_cast<decltype(download_api)>(symbol("lfs_hip_download"));
        release_api=reinterpret_cast<decltype(release_api)>(symbol("lfs_hip_buffer_release"));
        destroy_api=reinterpret_cast<decltype(destroy_api)>(symbol("lfs_hip_context_destroy"));
        native_check(create(0,&context,&error));native_check(allocate(context,count*sizeof(unsigned),&buffer,&error));
        uint64_t pointer=0,stream=0;int32_t ordinal=-1;
        native_check(address(context,buffer,&pointer,&error));native_check(queue(context,&stream,&ordinal,&error));
        device=reinterpret_cast<unsigned*>(uintptr_t(pointer));explicit_queue=reinterpret_cast<hipStream_t>(uintptr_t(stream));
    } else check(hipStreamCreateWithFlags(&explicit_queue,hipStreamNonBlocking));
    unsigned* reader_output=nullptr;check(hipMalloc(&reader_output,count*sizeof(unsigned)));
    const bool chain_reader=std::getenv("LFS_HIP_PROBE_CHAIN_READER")!=nullptr;
    unsigned total_failures=0;
    for(int upload=0;upload<2;++upload)
    for(int queue_kind=0;queue_kind<2;++queue_kind)
    for(int event_kind=0;event_kind<3;++event_kind)
    for(int pointer_kind=0;pointer_kind<2;++pointer_kind)
    for(int host_kind=0;host_kind<2;++host_kind) {
        if(context && !queue_kind)continue; // The native runtime owns its nonblocking queue.
        auto queue=queue_kind ? explicit_queue : nullptr;
        hipEvent_t event=nullptr;
        if(event_kind)check(hipEventCreateWithFlags(&event,event_kind==1 ? hipEventDefault : hipEventDisableTiming));
        unsigned failures=0;
        for(unsigned trial=0;trial<repeats;++trial) {
            check(hipMemsetAsync(device,0,count*sizeof(unsigned),queue));
            check(hipStreamSynchronize(queue));
            for(int j=0;j<8;++j)hipLaunchKernelGGL(delay,dim3(1),dim3(64),0,queue);
            const unsigned expected=0x12340000u+trial;
            if(pointer_kind)hipLaunchKernelGGL(write_pointer,dim3(1),dim3(256),0,queue,device,count,expected);
            else hipLaunchKernelGGL(write_struct,dim3(1),dim3(256),0,queue,(Payload{device,count,expected}));
            if(chain_reader)hipLaunchKernelGGL(read_struct,dim3(1),dim3(256),0,queue,(ReaderPayload{device,reader_output,count}));
            check(hipGetLastError());
            if(event)check(hipEventRecord(event,queue));
            unsigned* host=host_kind ? pinned : pageable.data();
            unsigned final_expected=expected;
            if(upload) {
                final_expected=0x98760000u+trial;
                for(unsigned i=0;i<count;++i)host[i]=final_expected;
                if(context)native_check(upload_api(context,buffer,0,host,count*sizeof(unsigned),&error));
                else check(hipMemcpyAsync(device,host,count*sizeof(unsigned),hipMemcpyHostToDevice,queue));
                check(hipStreamSynchronize(queue));
                // Verification is after full completion, so it cannot itself
                // trigger the event/copy race being measured.
                check(hipMemcpy(host,device,count*sizeof(unsigned),hipMemcpyDeviceToHost));
            } else {
                if(context)native_check(download_api(context,buffer,0,host,count*sizeof(unsigned),&error));
                else check(hipMemcpyAsync(host,device,count*sizeof(unsigned),hipMemcpyDeviceToHost,queue));
                check(hipStreamSynchronize(queue));
            }
            bool wrong=false;for(unsigned i=0;i<count;++i)wrong|=host[i]!=final_expected;
            failures+=wrong;
        }
        std::printf("direction=%s queue=%s event=%s arg=%s host=%s wrong=%u/%u\n",upload?"upload":"download",queue_kind?"nonblocking":"null",event_kind==0?"none":event_kind==1?"timed":"fence",pointer_kind?"pointer":"struct",host_kind?"pinned":"pageable",failures,repeats);
        std::fflush(stdout);total_failures+=failures;
        if(event)check(hipEventDestroy(event));
    }
    if(context) {native_check(release_api(context,buffer,&error));native_check(destroy_api(context,&error));FreeLibrary(module);}
    else {check(hipStreamDestroy(explicit_queue));check(hipFree(device));}
    check(hipHostFree(pinned));check(hipFree(reader_output));
    std::printf("%s HIP event/copy ordering failures=%u across %u trials\n",total_failures?"FAIL":"PASS",total_failures,argc==2?384u:768u);
    return total_failures ? 1 : 0;
}
