/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "../../src/core/tensor/backend/hip/runtime_api.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    template <class F>
    F symbol(HMODULE module, const char* name) {
        auto pointer = GetProcAddress(module, name);
        require(pointer != nullptr, name);
        return reinterpret_cast<F>(pointer);
    }
} // namespace

int main(int argc, char** argv) try {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    require(argc == 2, "Expected runtime module path");
    auto path = std::filesystem::absolute(argv[1]).make_preferred().wstring();
    const auto module = LoadLibraryExW(path.c_str(), nullptr,
                                       LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    require(module != nullptr, "Could not load HIP module");
#define LOAD(name) const auto name = symbol<decltype(&::name)>(module, #name)
    LOAD(lfs_hip_runtime_abi);
    LOAD(lfs_hip_device_count);
    LOAD(lfs_hip_device_info);
    LOAD(lfs_hip_context_create);
    LOAD(lfs_hip_context_destroy);
    LOAD(lfs_hip_context_synchronize);
    LOAD(lfs_hip_buffer_allocate);
    LOAD(lfs_hip_buffer_release);
    LOAD(lfs_hip_upload);
    LOAD(lfs_hip_download);
    LOAD(lfs_hip_readback_create);
    LOAD(lfs_hip_readback_enqueue);
    LOAD(lfs_hip_readback_wait);
    LOAD(lfs_hip_readback_poll);
    LOAD(lfs_hip_readback_destroy);

    LOAD(lfs_hip_fill_f32);
    LOAD(lfs_hip_affine_f32);
    LOAD(lfs_hip_warp_rgb);
    LOAD(lfs_hip_shared_image);
    LOAD(lfs_hip_shared_resize);
    LOAD(lfs_hip_cumsum);
    LOAD(lfs_hip_sort_pairs);
    LOAD(lfs_hip_select);
    LOAD(lfs_hip_copy_layouts);
    LOAD(lfs_hip_memory_info);
    LOAD(lfs_hip_trim_workspace);
    LOAD(lfs_hip_index_fill);
    LOAD(lfs_hip_environment_composite);
    LOAD(lfs_hip_pointwise_f16);
    LOAD(lfs_hip_masked_scatter);
#undef LOAD
    LfsHipError error{};
    const auto check = [&](int status) { require(status == 0, error.message); };
    const auto rejected = [&](int status) {
        require(status != 0 && error.message[0], "Invalid operation was not diagnosed");
    };
    require(lfs_hip_runtime_abi() == 1, "Unexpected runtime ABI");
    int32_t count = 0;
    check(lfs_hip_device_count(&count, &error));
    std::printf("HIP device count: %d\n", count);
    require(count > 0, "No HIP device; this validation requires real hardware");
    LfsHipDevice info{};
    check(lfs_hip_device_info(0, &info, &error));
    std::printf("HIP runtime contract: %s, %s, PCI %s, warp %d\n",
                info.name, info.architecture, info.pci_bus_id, info.warp_size);
    rejected(lfs_hip_device_info(count, &info, &error));
    rejected(lfs_hip_context_create(count, nullptr, &error));
    LfsHipContext* invalid = reinterpret_cast<LfsHipContext*>(uintptr_t(1));
    rejected(lfs_hip_context_create(count, &invalid, &error));
    require(!invalid, "Failed context creation published a handle");
    LfsHipContext *first = nullptr, *second = nullptr;
    check(lfs_hip_context_create(0, &first, &error));
    check(lfs_hip_context_create(0, &second, &error));
    constexpr size_t elements = 1048595;
    constexpr size_t bytes = elements * sizeof(float);
    LfsHipBuffer *input = nullptr, *output = nullptr;
    check(lfs_hip_buffer_allocate(first, bytes, &input, &error));
    check(lfs_hip_buffer_allocate(first, bytes, &output, &error));
    std::vector<float> source(elements), result(elements);
    for (size_t i = 0; i < elements; ++i)
        source[i] = float(int(i % 131) - 65) * 0.125f;
    check(lfs_hip_upload(first, input, 0, source.data(), bytes, &error));
    {
        LfsHipBuffer* flags = nullptr;
        check(lfs_hip_buffer_allocate(first, 6, &flags, &error));
        const uint8_t mask[] = {0, 1, 0, 1, 1, 0};
        const float replacements[] = {10.f, 20.f, 30.f};
        const float original[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
        float observed[6]{};
        check(lfs_hip_upload(first, flags, 0, mask, sizeof(mask), &error));
        check(lfs_hip_upload(first, input, 0, replacements, sizeof(replacements), &error));
        check(lfs_hip_upload(first, output, 0, original, sizeof(original), &error));
        rejected(lfs_hip_masked_scatter(first, {output, 0}, {flags, 0}, {input, 0},
                                        LFS_HIP_F32, 6, 2, &error));
        check(lfs_hip_download(first, output, 0, observed, sizeof(observed), &error));
        require(std::equal(std::begin(observed), std::end(observed), std::begin(original)),
                "Invalid HIP masked scatter partially changed output");
        check(lfs_hip_masked_scatter(first, {output, 0}, {flags, 0}, {input, 0},
                                     LFS_HIP_F32, 6, 3, &error));
        check(lfs_hip_download(first, output, 0, observed, sizeof(observed), &error));
        const float expected[] = {0.f, 10.f, 2.f, 20.f, 30.f, 5.f};
        require(std::equal(std::begin(observed), std::end(observed), std::begin(expected)),
                "HIP masked scatter changed stable source order");
        check(lfs_hip_buffer_release(first, flags, &error));
        check(lfs_hip_upload(first, input, 0, source.data(), bytes, &error));
        std::puts("PASS HIP masked scatter: stable order, source count guard, no partial write");
    }
    {
        const uint16_t lhs[] = {0xbc00, 0x0000, 0x3c00, 0x4000}; // -1, 0, 1, 2
        const uint16_t rhs[] = {0x4000, 0x4200, 0x4400, 0x4500}; // 2, 3, 4, 5
        uint16_t observed[4]{};
        check(lfs_hip_upload(first, input, 0, lhs, sizeof(lhs), &error));
        check(lfs_hip_upload(first, input, 16, rhs, sizeof(rhs), &error));
        check(lfs_hip_pointwise_f16(first, LFS_HIP_OP_AddTensor, {input, 0}, {input, 16},
                                    {output, 0}, 4, 0.f, &error));
        check(lfs_hip_download(first, output, 0, observed, sizeof(observed), &error));
        require(observed[0] == 0x3c00 && observed[1] == 0x4200 && observed[2] == 0x4500 && observed[3] == 0x4700,
                "HIP half tensor pointwise result differs");
        check(lfs_hip_pointwise_f16(first, LFS_HIP_OP_AddScalar, {input, 0}, {},
                                    {output, 0}, 4, 2.f, &error));
        check(lfs_hip_download(first, output, 0, observed, sizeof(observed), &error));
        require(observed[0] == 0x3c00 && observed[1] == 0x4000 && observed[2] == 0x4200 && observed[3] == 0x4400,
                "HIP half scalar pointwise result differs");
        uint8_t mask[4]{};
        check(lfs_hip_pointwise_f16(first, LFS_HIP_OP_LessScalar, {input, 0}, {},
                                    {output, 0}, 4, 0.f, &error));
        check(lfs_hip_download(first, output, 0, mask, sizeof(mask), &error));
        require(mask[0] == 1 && mask[1] == 0 && mask[2] == 0 && mask[3] == 0,
                "HIP half pointwise predicate differs");
        rejected(lfs_hip_pointwise_f16(first, LFS_HIP_OP_AddTensor, {input, bytes - 2}, {input, 16},
                                       {output, 0}, 4, 0.f, &error));
        check(lfs_hip_upload(first, input, 0, source.data(), bytes, &error));
        std::puts("PASS HIP half pointwise: tensor/scalar arithmetic, predicate, range rejection");
    }

    {
        constexpr size_t h=17,w=19,pixels=h*w;
        LfsHipSharedImage image{};image.struct_bytes=sizeof(image);image.height=h;image.width=w;image.channels=3;
        std::vector<uint8_t> u8(pixels*4);std::vector<uint16_t> u16(pixels*4);std::vector<float> floats(pixels*4);
        for(size_t i=0;i<u8.size();++i){u8[i]=uint8_t((i*71)%256);u16[i]=uint16_t((i*571)%65536);floats[i]=float(int((i*73)%313)-20)/271.f;}
        std::vector<float> converted(pixels*4);std::vector<uint8_t> byte_result(pixels*4);std::vector<uint16_t> short_result(pixels*4);
        float worst=0;
        for(unsigned op=0;op<16;++op){
            image.operation=op;image.channels=(op==11 || op==13)?1:((op==14 || op==15)?4:3);
            const size_t total=pixels*image.channels;
            const bool in16=op==1 || op==3 || op==7 || op==9 || op==15;
            const bool in32=op==2 || op==4 || op==5 || op==10 || op==12;
            check(lfs_hip_upload(first,input,0,in16?static_cast<void*>(u16.data()):in32?static_cast<void*>(floats.data()):static_cast<void*>(u8.data()),total*(in16?2:in32?4:1),&error));
            check(lfs_hip_shared_image(first,&image,{input,0},{output,0},{},&error));
            if(op==2){check(lfs_hip_download(first,output,0,short_result.data(),total*2,&error));for(size_t i=0;i<total;++i)require(short_result[i]==uint16_t(std::clamp(floats[i]*65535.f,0.f,65535.f)+.5f),"HIP u16 encoding differs");continue;}
            if(op==8 || op==9 || op==10){check(lfs_hip_download(first,output,0,byte_result.data(),total,&error));for(size_t i=0;i<total;++i){size_t from=(i%pixels)*image.channels+i/pixels;uint8_t expected=op==8?u8[from]:op==9?uint8_t((uint32_t(u16[from])*255+32767)/65535):uint8_t(std::clamp(floats[i],0.f,1.f)*255.f+.5f);require(byte_result[i]==expected,"HIP byte layout/conversion differs");}continue;}
            check(lfs_hip_download(first,output,0,converted.data(),total*4,&error));
            for(size_t i=0;i<total;++i){float expected=0;const size_t from=(i%pixels)*image.channels+i/pixels;
                if(op==0 || op==14)expected=(op==14 && i/pixels==3)?float(u8[from])/255.f:float(u8[from])*(1.f/255.f);
                if(op==1 || op==15)expected=(op==15 && i/pixels==3)?float(u16[from])/65535.f:float(u16[from])*(1.f/65535.f);
                if(op==3)expected=float(u16[i])*(1.f/65535.f);
                if(op==4)expected=std::clamp(floats[(i%3)*pixels+i/3]*.5f+.5f,0.f,1.f);
                if(op==5)expected=floats[from]*2.f-1.f;
                if(op==6)expected=float(u8[from])*(1.f/255.f)*2.f-1.f;
                if(op==7)expected=float(u16[from])*(1.f/65535.f)*2.f-1.f;
                if(op==12)expected=floats[from]*2.f-1.f;
                if(op==11)expected=float(u8[i])*(1.f/255.f);
                if(op==13)expected=float(u8[i])/255.f;
                worst=std::max(worst,std::abs(converted[i]-expected));require(std::abs(converted[i]-expected)<3e-7,"HIP float image conversion differs");
                if(op==13 || ((op==14 || op==15) && i/pixels==3))require(converted[i]==expected,"HIP exact mask alpha normalization differs");
            }
        }
        // Check every 16-bit alpha value, beyond the small non-aligned image case.
        image.operation=15;image.height=256;image.width=256;image.channels=4;
        std::vector<uint16_t> all_alpha(65536*4);for(size_t i=0;i<65536;++i)all_alpha[i*4+3]=uint16_t(i);
        check(lfs_hip_upload(first,input,0,all_alpha.data(),all_alpha.size()*2,&error));
        check(lfs_hip_shared_image(first,&image,{input,0},{output,0},{},&error));
        std::vector<float> alpha_plane(65536);check(lfs_hip_download(first,output,3*65536*4,alpha_plane.data(),alpha_plane.size()*4,&error));
        for(size_t i=0;i<65536;++i)require(alpha_plane[i]==float(i)/65535.f,"HIP 16-bit exact alpha differs from CPU rounded division");
        image.height=h;image.width=w;image.channels=1;image.operation=16;image.seed=0x12345678;
        check(lfs_hip_shared_image(first,&image,{}, {input,0},{},&error));
        std::vector<uint8_t> sentinel(pixels);check(lfs_hip_download(first,input,0,sentinel.data(),pixels,&error));
        for(size_t i=0;i<pixels;++i){uint32_t v=uint32_t(i)^image.seed;v^=v>>16;v*=0x7feb352du;v^=v>>15;v*=0x846ca68bu;v^=v>>16;require(sentinel[i]==uint8_t(v),"HIP sentinel hash differs");}
        uint32_t unchanged=1;check(lfs_hip_upload(first,output,0,&unchanged,4,&error));image.operation=17;
        check(lfs_hip_shared_image(first,&image,{input,0},{output,0},{},&error));check(lfs_hip_download(first,output,0,&unchanged,4,&error));require(unchanged==1,"HIP sentinel check rejected original");
        sentinel[111]^=1;check(lfs_hip_upload(first,input,0,sentinel.data(),pixels,&error));check(lfs_hip_shared_image(first,&image,{input,0},{output,0},{},&error));check(lfs_hip_download(first,output,0,&unchanged,4,&error));require(unchanged==0,"HIP sentinel check missed mutation");
        for(unsigned mode:{18u,19u}){
            image.operation=mode;image.channels=4;check(lfs_hip_upload(first,input,0,u8.data(),pixels*4,&error));
            check(lfs_hip_shared_image(first,&image,{input,0},{output,0},{output,pixels*3*4},&error));
            if(mode==18){check(lfs_hip_download(first,output,0,converted.data(),pixels*3*4,&error));for(size_t c=0;c<3;++c)for(size_t i=0;i<pixels;++i)require(converted[c*pixels+i]==float(u8[i*4+c])*(1.f/255.f),"HIP RGBA float RGB differs");}
            else{check(lfs_hip_download(first,output,0,byte_result.data(),pixels*3,&error));for(size_t c=0;c<3;++c)for(size_t i=0;i<pixels;++i)require(byte_result[c*pixels+i]==u8[i*4+c],"HIP RGBA byte RGB differs");}
            check(lfs_hip_download(first,output,pixels*3*4,converted.data(),pixels*4,&error));for(size_t i=0;i<pixels;++i)require(converted[i]==float(u8[i*4+3])/255.f,"HIP RGBA alpha differs");
        }
        image.channels=1;image.operation=20;check(lfs_hip_upload(first,output,0,floats.data(),pixels*4,&error));
        check(lfs_hip_shared_image(first,&image,{}, {output,0},{},&error));check(lfs_hip_download(first,output,0,converted.data(),pixels*4,&error));for(size_t i=0;i<pixels;++i)require(converted[i]==1.f-floats[i],"HIP mask invert differs");
        image.operation=21;image.threshold=.371f;check(lfs_hip_shared_image(first,&image,{}, {output,0},{},&error));check(lfs_hip_download(first,output,0,converted.data(),pixels*4,&error));for(size_t i=0;i<pixels;++i)require(converted[i]==float(1.f-floats[i]>=image.threshold),"HIP mask threshold differs");
        image.operation=22;image.threshold=0;check(lfs_hip_upload(first,output,0,floats.data(),pixels*4,&error));
        check(lfs_hip_shared_image(first,&image,{}, {output,0},{},&error));check(lfs_hip_download(first,output,0,converted.data(),pixels*4,&error));for(size_t i=0;i<pixels;++i){volatile float product=floats[i]*255.f;float expected=float(uint8_t(std::clamp(float(product),0.f,255.f)))/255.f;require(converted[i]==expected,"HIP mask byte-grid quantization differs");}
        image.operation=12;image.channels=3;image.srgb=1;image.flip_yz=1;image.world_to_camera=1;image.w2c[1]=-1;image.w2c[3]=1;image.w2c[8]=1;
        check(lfs_hip_upload(first,input,0,floats.data(),pixels*3*4,&error));check(lfs_hip_shared_image(first,&image,{input,0},{output,0},{},&error));check(lfs_hip_download(first,output,0,converted.data(),pixels*3*4,&error));
        for(size_t i=0;i<pixels;++i){double n[3];for(size_t c=0;c<3;++c){double v=floats[i*3+c];n[c]=2*(v<=.04045?v/12.92:std::pow((v+.055)/1.055,2.4))-1;}double expected[3]{n[1],n[0],-n[2]};for(size_t c=0;c<3;++c)require(std::abs(converted[c*pixels+i]-expected[c])<1.5e-6,"HIP transformed sRGB normals differ from CPU reference");}
        image.srgb=image.flip_yz=image.world_to_camera=0;
        image.height=h;image.width=w;image.channels=3;image.operation=0;
        rejected(lfs_hip_shared_image(first,&image,{input,0},{input,0},{},&error));
        rejected(lfs_hip_shared_image(second,&image,{input,0},{output,0},{},&error));
        rejected(lfs_hip_shared_image(first,&image,{input,0},{output,1},{},&error));
        image.width=SIZE_MAX;rejected(lfs_hip_shared_image(first,&image,{input,0},{output,0},{},&error));
        check(lfs_hip_upload(first,input,0,source.data(),bytes,&error));
        std::printf("PASS HIP core image conversions: all 16 modes, all 65536 exact alpha values, overlap/ownership/alignment/extents; max float error %.9g\n",worst);
    }
#include "shared_resize_contract.inc"
    {
        LfsHipReadback* staging = nullptr;
        check(lfs_hip_readback_create(first, &staging, &error));
        int32_t ready = -1;
        rejected(lfs_hip_readback_poll(second, staging, nullptr, &ready, &error));
        rejected(lfs_hip_context_destroy(first, &error));
        check(lfs_hip_readback_enqueue(first, staging, input, 16, bytes-16, &error));
        rejected(lfs_hip_readback_enqueue(first, staging, input, 0, 4, &error));
        check(lfs_hip_readback_wait(first, staging, &error));
        check(lfs_hip_readback_poll(first, staging, result.data(), &ready, &error));
        require(ready == 1 && std::equal(result.begin(), result.end()-4, source.begin()+4), "Native readback snapshot changed");
        check(lfs_hip_readback_poll(first, staging, nullptr, &ready, &error));
        require(ready == 0, "Readback consumed twice");
        rejected(lfs_hip_readback_enqueue(first, staging, input, bytes, 1, &error));
        check(lfs_hip_readback_enqueue(first, staging, nullptr, 0, 0, &error));
        check(lfs_hip_readback_destroy(first, staging, &error));
        rejected(lfs_hip_readback_destroy(first, staging, &error));
        std::puts("PASS native HIP readback ownership, pending drain, bounds and single-consumption");
    }
    rejected(lfs_hip_download(second, input, 0, result.data(), bytes, &error));
    rejected(lfs_hip_buffer_release(second, input, &error));
    rejected(lfs_hip_upload(first, input, SIZE_MAX, source.data(), 1, &error));
    rejected(lfs_hip_fill_f32(first, output, 1, 1, 0, &error));
    rejected(lfs_hip_fill_f32(first, output, 0, SIZE_MAX, 0, &error));
    rejected(lfs_hip_affine_f32(first, input, output, elements + 1, 1, 0, &error));
    check(lfs_hip_fill_f32(first, output, 0, elements, -999, &error));
    {
        LfsHipLayout layout{};
        layout.rank = 1;
        layout.dims[0] = elements;
        layout.strides[0] = 1;
        layout.elements = elements;
        const LfsHipView view{output, 0};
        rejected(lfs_hip_cumsum(second, view, LFS_HIP_F32, &layout, 0, &error));
        rejected(lfs_hip_cumsum(first, view, LFS_HIP_F32, &layout, 1, &error));
        rejected(lfs_hip_cumsum(first, view, LFS_HIP_F16, &layout, 0, &error));
        rejected(lfs_hip_cumsum(first, {output, 1}, LFS_HIP_F32, &layout, 0, &error));
        auto invalid_layout = layout;
        invalid_layout.rank = 2;
        invalid_layout.dims[0] = 2;
        invalid_layout.dims[1] = elements / 2;
        invalid_layout.strides[1] = 1;
        invalid_layout.elements = 2 * (elements / 2);
        rejected(lfs_hip_cumsum(first, view, LFS_HIP_F32, &invalid_layout, 0, &error));
        invalid_layout = layout;
        invalid_layout.dims[0] = invalid_layout.elements = UINT64_MAX;
        rejected(lfs_hip_cumsum(first, view, LFS_HIP_F32, &invalid_layout, 0, &error));
        check(lfs_hip_download(first, output, 0, result.data(), bytes, &error));
        require(std::all_of(result.begin(), result.end(), [](float value) { return value == -999; }),
                "Rejected scan changed data");
        check(lfs_hip_upload(first, output, 0, source.data(), bytes, &error));
        check(lfs_hip_cumsum(first, view, LFS_HIP_F32, &layout, 0, &error));
        check(lfs_hip_download(first, output, 0, result.data(), bytes, &error));
        float prefix = 0;
        for (size_t i = 0; i < elements; ++i) {
            prefix += source[i];
            require(result[i] == prefix, "Native scan differs from exact dyadic prefix");
        }
        LfsHipMemoryInfo memory{};
        check(lfs_hip_memory_info(first, &memory, &error));
        require(memory.workspace_bytes > 0 && memory.owned_bytes == 2 * bytes + memory.workspace_bytes,
                "Native scan workspace was not accounted");
        const auto workspace = memory.workspace_bytes;
        check(lfs_hip_upload(first, output, 0, source.data(), bytes, &error));
        check(lfs_hip_cumsum(first, view, LFS_HIP_F32, &layout, 0, &error));
        check(lfs_hip_memory_info(first, &memory, &error));
        require(memory.workspace_bytes == workspace, "Native scan did not reuse its workspace");
        // A strided native view exercises gaps that Tensor::cumsum's clone
        // normally makes contiguous. Each axis must preserve every gap byte.
        std::vector<float> strided(900, -999);
        for (size_t r = 0; r < 3; ++r)
            for (size_t c = 0; c < 263; ++c)
                strided[r * 300 + c] = float(int((r + c) % 7) - 3) * 0.25f;
        LfsHipLayout axes{};
        axes.rank = 2;
        axes.dims[0] = 3;
        axes.dims[1] = 263;
        axes.strides[0] = 300;
        axes.strides[1] = 1;
        axes.elements = 789;
        for (unsigned axis = 0; axis < 2; ++axis) {
            auto expected = strided;
            for (size_t r = 0; r < 3; ++r)
                for (size_t c = 0; c < 263; ++c) {
                    if (!axis && r)
                        expected[r * 300 + c] += expected[(r - 1) * 300 + c];
                    if (axis && c)
                        expected[r * 300 + c] += expected[r * 300 + c - 1];
                }
            check(lfs_hip_upload(first, output, 0, strided.data(), strided.size() * 4, &error));
            check(lfs_hip_cumsum(first, view, LFS_HIP_F32, &axes, axis, &error));
            std::vector<float> actual(strided.size());
            check(lfs_hip_download(first, output, 0, actual.data(), actual.size() * 4, &error));
            require(actual == expected, "Native axis scan changed a prefix or padding");
        }
        std::vector<uint64_t> wide(513, uint64_t(1) << 50);
        wide[0] = UINT64_MAX - 1024;
        layout.dims[0] = layout.elements = wide.size();
        for (unsigned dtype : {LFS_HIP_I64, LFS_HIP_U32}) {
            const size_t width = dtype == LFS_HIP_I64 ? 8 : 4;
            check(lfs_hip_upload(first, output, 0, wide.data(), wide.size() * width, &error));
            check(lfs_hip_cumsum(first, view, dtype, &layout, 0, &error));
            std::vector<uint64_t> actual(wide.size());
            check(lfs_hip_download(first, output, 0, actual.data(), actual.size() * width, &error));
            if (width == 8) {
                uint64_t prefix = 0;
                for (size_t i = 0; i < wide.size(); ++i) {
                    prefix += wide[i];
                    require(actual[i] == prefix, "Native Int64 scan lost bits or overflow semantics");
                }
            } else {
                std::vector<uint32_t> input(wide.size()), result(wide.size());
                std::memcpy(input.data(), wide.data(), input.size() * sizeof(uint32_t));
                std::memcpy(result.data(), actual.data(), result.size() * sizeof(uint32_t));
                uint32_t prefix = 0;
                for (size_t i = 0; i < wide.size(); ++i) {
                    prefix += input[i];
                    require(result[i] == prefix, "Native UInt32 scan lost overflow semantics");
                }
            }
        }
        layout.dims[0] = layout.elements = 0;
        check(lfs_hip_cumsum(first, {output, bytes}, LFS_HIP_F32, &layout, 0, &error));
    }
    {
        check(lfs_hip_fill_f32(first, output, 0, elements, -999, &error));
        rejected(lfs_hip_sort_pairs(second, {output, 0}, {input, 0}, 1, 7, 1, 0, &error));
        rejected(lfs_hip_sort_pairs(first, {output, 0}, {output, 0}, 1, 7, 1, 0, &error));
        rejected(lfs_hip_sort_pairs(first, {output, 1}, {input, 0}, 1, 7, 1, 0, &error));
        rejected(lfs_hip_sort_pairs(first, {output, 0}, {input, 4}, 1, 7, 1, 0, &error));
        rejected(lfs_hip_sort_pairs(first, {output, 0}, {input, 0}, 1, 7, 1, 2, &error));
        rejected(lfs_hip_sort_pairs(first, {output, 0}, {input, 0}, SIZE_MAX, 7, 1, 0, &error));
        check(lfs_hip_download(first, output, 0, result.data(), bytes, &error));
        require(std::all_of(result.begin(), result.end(), [](float v) { return v == -999; }), "Rejected sort changed values");
        std::vector<float> values(42), actual(42);
        std::vector<int64_t> order(42);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = float(int(i % 9) - 4);
        for (unsigned descending : {0u, 1u}) {
            check(lfs_hip_upload(first, output, 16, values.data(), values.size() * 4, &error));
            check(lfs_hip_sort_pairs(first, {output, 16}, {input, 32}, 2, 7, 3, descending, &error));
            check(lfs_hip_download(first, output, 16, actual.data(), actual.size() * 4, &error));
            check(lfs_hip_download(first, input, 32, order.data(), order.size() * 8, &error));
            for (size_t outer = 0; outer < 2; ++outer)
                for (size_t inner = 0; inner < 3; ++inner) {
                    std::vector<size_t> expected{0, 1, 2, 3, 4, 5, 6};
                    auto at = [&](size_t column) { return (outer * 7 + column) * 3 + inner; };
                    std::stable_sort(expected.begin(), expected.end(), [&](size_t a, size_t b) {
                        return descending ? values[at(a)] > values[at(b)] : values[at(a)] < values[at(b)];
                    });
                    for (size_t column = 0; column < 7; ++column)
                        require(actual[at(column)] == values[at(expected[column])] && order[at(column)] == int64_t(expected[column]),
                                "Native segmented sort differs from stable CPU reference");
                }
            check(lfs_hip_download(first, output, 0, result.data(), bytes, &error));
            for (size_t i = 0; i < elements; ++i)
                if (i < 4 || i >= 4 + values.size())
                    require(result[i] == -999, "Native sort wrote outside values range");
        }
        check(lfs_hip_sort_pairs(first, {output, bytes}, {input, 0}, 0, 7, 1, 0, &error));
        LfsHipMemoryInfo memory{};
        check(lfs_hip_memory_info(first, &memory, &error));
        require(memory.workspace_bytes > 0, "Sort workspace not accounted");
        check(lfs_hip_trim_workspace(first, &error));
        check(lfs_hip_memory_info(first, &memory, &error));
        require(memory.workspace_bytes == 0 && memory.owned_bytes == 2 * bytes, "Workspace trim did not release scratch");
        check(lfs_hip_trim_workspace(first, &error));
        check(lfs_hip_upload(first, output, 16, values.data(), values.size() * 4, &error));
        check(lfs_hip_sort_pairs(first, {output, 16}, {input, 32}, 2, 7, 3, 0, &error));
        check(lfs_hip_memory_info(first, &memory, &error));
        require(memory.workspace_bytes > 0, "Sort could not recreate trimmed workspace");
        check(lfs_hip_upload(first, input, 0, source.data(), bytes, &error));
    }
    {
        constexpr size_t length = 513;
        LfsHipBuffer* mask = nullptr;
        check(lfs_hip_buffer_allocate(first, length, &mask, &error));
        std::vector<float> values(length), expected;
        std::vector<uint8_t> flags(length);
        for (size_t i = 0; i < length; ++i) {
            values[i] = float(i);
            flags[i] = i % 3 == 0 ? 255 : 0;
            if (flags[i])
                expected.push_back(values[i]);
        }
        check(lfs_hip_upload(first, input, 0, values.data(), length * 4, &error));
        check(lfs_hip_upload(first, mask, 0, flags.data(), length, &error));
        check(lfs_hip_fill_f32(first, output, 0, elements, -999, &error));
        uint64_t selected = 99;
        rejected(lfs_hip_select(second, {input, 0}, {mask, 0}, {output, 16}, LFS_HIP_F32, 0, length, length, &selected, &error));
        rejected(lfs_hip_select(first, {input, 0}, {mask, 0}, {input, 0}, LFS_HIP_F32, 0, length, length, &selected, &error));
        rejected(lfs_hip_select(first, {input, 0}, {output, 16}, {output, 16}, LFS_HIP_F32, 0, length, length, &selected, &error));
        rejected(lfs_hip_select(first, {input, 0}, {mask, 0}, {output, 17}, LFS_HIP_F32, 0, length, length, &selected, &error));
        rejected(lfs_hip_select(first, {input, 0}, {mask, 0}, {output, 16}, 99, 0, length, length, &selected, &error));
        rejected(lfs_hip_select(first, {input, 0}, {mask, 0}, {output, 16}, LFS_HIP_F32, 0, SIZE_MAX, 0, &selected, &error));
        rejected(lfs_hip_select(first, {input, 0}, {mask, 0}, {output, 16}, LFS_HIP_F32, 0, length, length + 1, &selected, &error));
        rejected(lfs_hip_select(first, {input, 0}, {mask, 0}, {output, 16}, LFS_HIP_F32, 0, length, 2, &selected, &error));
        require(selected == 0, "Failed selection published a count");
        check(lfs_hip_download(first, output, 0, result.data(), bytes, &error));
        require(std::all_of(result.begin(), result.end(), [](float v) { return v == -999; }), "Rejected selection changed output");
        check(lfs_hip_select(first, {input, 0}, {mask, 0}, {output, 16}, LFS_HIP_F32, 0, length, expected.size(), &selected, &error));
        require(selected == expected.size(), "Masked selection returned incorrect count");
        check(lfs_hip_download(first, output, 0, result.data(), bytes, &error));
        for (size_t i = 0; i < elements; ++i)
            require(result[i] == (i >= 4 && i < 4 + expected.size() ? expected[i - 4] : -999), "Selection order or guards changed");
        check(lfs_hip_select(first, {input, 0}, {}, {output, 16}, LFS_HIP_F32, 1, length, length, &selected, &error));
        std::vector<int64_t> positions(selected);
        check(lfs_hip_download(first, output, 16, positions.data(), positions.size() * 8, &error));
        require(selected == length - 1, "Native nonzero returned incorrect count");
        for (size_t i = 0; i < positions.size(); ++i)
            require(positions[i] == int64_t(i + 1), "Native nonzero positions are not stable");
        check(lfs_hip_select(first, {input, bytes}, {}, {output, 0}, LFS_HIP_F32, 1, 0, 0, &selected, &error));
        require(selected == 0, "Empty selection returned elements");
        check(lfs_hip_trim_workspace(first, &error));
        check(lfs_hip_select(first, {input, 0}, {mask, 0}, {output, 16}, LFS_HIP_F32, 0, length, expected.size(), &selected, &error));
        require(selected == expected.size(), "Selection failed after workspace trim");
        check(lfs_hip_buffer_release(first, mask, &error));
        check(lfs_hip_upload(first, input, 0, source.data(), bytes, &error));
    }
    {
        LfsHipLayoutCopy copies[2]{};
        copies[0].input = {input, 48};
        copies[0].output = {output, 16};
        copies[0].element_bytes = 4;
        copies[0].input_layout.rank = copies[0].output_layout.rank = 2;
        copies[0].input_layout.elements = copies[0].output_layout.elements = 4;
        for (unsigned dim = 0; dim < 2; ++dim)
            copies[0].input_layout.dims[dim] = copies[0].output_layout.dims[dim] = 2;
        copies[0].input_layout.strides[0] = 2;
        copies[0].output_layout.strides[0] = 3;
        copies[0].input_layout.strides[1] = copies[0].output_layout.strides[1] = 1;
        copies[1] = copies[0];
        copies[1].input.offset = bytes - 1;
        check(lfs_hip_fill_f32(first, output, 0, elements, -999, &error));
        rejected(lfs_hip_copy_layouts(first, copies, 2, &error));
        check(lfs_hip_download(first, output, 0, result.data(), bytes, &error));
        require(std::all_of(result.begin(), result.end(), [](float v) { return v == -999; }),
                "Invalid later layout copy partially wrote an earlier request");
        rejected(lfs_hip_copy_layouts(second, copies, 1, &error));
        rejected(lfs_hip_copy_layouts(first, nullptr, 1, &error));
        rejected(lfs_hip_copy_layouts(first, copies, 4097, &error));
        const auto valid = copies[0];
        copies[0].element_bytes = 3;
        rejected(lfs_hip_copy_layouts(first, copies, 1, &error));
        copies[0] = valid;
        copies[0].output_layout.strides[0] = 1;
        rejected(lfs_hip_copy_layouts(first, copies, 1, &error));
        copies[0] = valid;
        copies[0].output = copies[0].input;
        rejected(lfs_hip_copy_layouts(first, copies, 1, &error));
        copies[0] = valid;
        copies[1] = valid;
        copies[1].input = {input, 128};
        copies[1].output = {output, 80};
        copies[1].input_layout = copies[1].output_layout = {};
        copies[1].input_layout.rank = copies[1].output_layout.rank = 1;
        copies[1].input_layout.elements = copies[1].output_layout.elements = 4;
        copies[1].input_layout.dims[0] = copies[1].output_layout.dims[0] = 4;
        copies[1].input_layout.strides[0] = copies[1].output_layout.strides[0] = 1;
        check(lfs_hip_copy_layouts(first, copies, 2, &error));
        check(lfs_hip_download(first, output, 0, result.data(), bytes, &error));
        for (size_t i = 0; i < elements; ++i) {
            float expected = -999;
            if (i == 4 || i == 5)
                expected = source[12 + i - 4];
            if (i == 7 || i == 8)
                expected = source[14 + i - 7];
            if (i >= 20 && i < 24)
                expected = source[32 + i - 20];
            require(result[i] == expected, "Layout copy values or padding differ");
        }
        check(lfs_hip_copy_layouts(first, nullptr, 0, &error));
    }
    // Ordered kernels, including in-place access, must be visible to download.
    check(lfs_hip_affine_f32(first, input, output, elements, 1.25f, 0.125f, &error));
    check(lfs_hip_affine_f32(first, output, output, elements, 2, -0.25f, &error));
    check(lfs_hip_download(first, output, 0, result.data(), bytes, &error));
    for (size_t i = 0; i < elements; ++i)
        require(result[i] == source[i] * 2.5f, "Queue ordering or affine mismatch");
    // Offset fill preserves prefix/suffix guard values.
    check(lfs_hip_fill_f32(first, output, 0, elements, -99, &error));
    check(lfs_hip_fill_f32(first, output, 13 * sizeof(float), elements - 26, 7, &error));
    check(lfs_hip_download(first, output, 0, result.data(), bytes, &error));
    for (size_t i = 0; i < elements; ++i)
        require(result[i] == (i < 13 || i >= elements - 13 ? -99 : 7), "Offset fill guard changed");
    check(lfs_hip_fill_f32(first, output, bytes, 0, 1, &error));
    check(lfs_hip_upload(first, output, bytes, nullptr, 0, &error));
    check(lfs_hip_download(first, output, bytes, nullptr, 0, &error));
    {
        LfsHipLens lens{60, 80, 8.5f, 64.51723f, 60.f * 11 / 17, 80.f * 83 / 129, 8.5f * 11 / 17, 64.51723f * 83 / 129, 17, 129, 11, 83, 0, {}, 1};
        lens.distortion[0] = 0.1f;
        constexpr size_t rgb_count = 3 * 17 * 129;
        std::vector<float> constant(rgb_count, 0.25f);
        check(lfs_hip_upload(first, input, 16, constant.data(), constant.size() * sizeof(float), &error));
        check(lfs_hip_fill_f32(first, output, 0, elements, -99, &error));
        check(lfs_hip_warp_rgb(first, {input, 16}, LFS_HIP_F32, {output, 32}, &lens, &error));
        check(lfs_hip_download(first, output, 0, result.data(), bytes, &error));
        for (size_t i = 0; i < elements; ++i) {
            const bool written = i >= 8 && i < 8 + 3 * 11 * 83;
            require(written ? std::abs(result[i] - 0.25f) < 1e-6f : result[i] == -99,
                    "RGB warp constant sampling or output guard changed");
        }
        rejected(lfs_hip_warp_rgb(second, {input, 16}, LFS_HIP_F32, {output, 32}, &lens, &error));
        rejected(lfs_hip_warp_rgb(first, {input, 16}, LFS_HIP_F32, {input, 32}, &lens, &error));
        rejected(lfs_hip_warp_rgb(first, {input, SIZE_MAX}, LFS_HIP_F32, {output, 32}, &lens, &error));
        rejected(lfs_hip_warp_rgb(first, {input, 1}, LFS_HIP_F32, {output, 32}, &lens, &error));
        rejected(lfs_hip_warp_rgb(first, {input, 16}, LFS_HIP_F32, {output, 2}, &lens, &error));
        rejected(lfs_hip_warp_rgb(first, {input, 16}, LFS_HIP_F16, {output, 32}, &lens, &error));
        rejected(lfs_hip_warp_rgb(first, {input, 16}, LFS_HIP_F32, {output, bytes - 4}, &lens, &error));
        auto invalid_lens = lens;
        invalid_lens.dst_fx = lens.src_fx / 128;
        rejected(lfs_hip_warp_rgb(first, {input, 16}, LFS_HIP_F32, {output, 32}, &invalid_lens, &error));
        invalid_lens = lens;
        invalid_lens.dst_fx = std::numeric_limits<float>::quiet_NaN();
        rejected(lfs_hip_warp_rgb(first, {input, 16}, LFS_HIP_F32, {output, 32}, &invalid_lens, &error));
        auto pathological_lens = lens;
        pathological_lens.distortion[0] = 1e38f;
        check(lfs_hip_warp_rgb(first, {input, 16}, LFS_HIP_F32, {output, 32}, &pathological_lens, &error));
        check(lfs_hip_download(first, output, 32, result.data(), 3 * 11 * 83 * sizeof(float), &error));
        for (size_t i = 0; i < 3 * 11 * 83; ++i)
            require(std::isfinite(result[i]) && result[i] >= 0 && result[i] <= 0.251f,
                    "Unrepresentable projected coordinates escaped the RGB warp bounds guard");
        // Exercise the C ABI identity path too; the Tensor frontend normally
        // short-circuits it before calling the native warp.
        lens.dst_width = lens.src_width;
        lens.dst_height = lens.src_height;
        lens.dst_fx = lens.src_fx;
        lens.dst_fy = lens.src_fy;
        lens.dst_cx = lens.src_cx;
        lens.dst_cy = lens.src_cy;
        lens.distortion[0] = 0;
        std::vector<uint8_t> rgb_bytes(rgb_count);
        for (size_t i = 0; i < rgb_bytes.size(); ++i)
            rgb_bytes[i] = uint8_t(i % 256);
        check(lfs_hip_upload(first, input, 1, rgb_bytes.data(), rgb_bytes.size(), &error));
        check(lfs_hip_warp_rgb(first, {input, 1}, LFS_HIP_U8, {output, 32}, &lens, &error));
        check(lfs_hip_download(first, output, 32, constant.data(), rgb_count * sizeof(float), &error));
        for (size_t i = 0; i < rgb_count; ++i)
            require(std::abs(constant[i] - float(rgb_bytes[i]) / 255.f) < 1e-7f,
                    "RGB byte identity normalization changed");
        // Exceed the runtime's 4096-block launch cap. Identity must use a
        // grid-stride loop too, including the final partial RGB rows.
        lens.src_width = lens.dst_width = 1;
        lens.src_height = lens.dst_height = int(elements / 3);
        const size_t large_rgb_count = size_t(lens.src_height) * 3;
        check(lfs_hip_upload(first, input, 0, source.data(), bytes, &error));
        check(lfs_hip_fill_f32(first, output, 0, elements, -99, &error));
        check(lfs_hip_warp_rgb(first, {input, 0}, LFS_HIP_F32, {output, 0}, &lens, &error));
        check(lfs_hip_download(first, output, 0, result.data(), bytes, &error));
        for (size_t i = 0; i < elements; ++i)
            require(result[i] == (i < large_rgb_count ? source[i] : -99), "Large float RGB identity lost its grid tail");
        rgb_bytes.resize(large_rgb_count);
        for (size_t i = 0; i < rgb_bytes.size(); ++i)
            rgb_bytes[i] = uint8_t(i % 256);
        check(lfs_hip_upload(first, input, 1, rgb_bytes.data(), rgb_bytes.size(), &error));
        check(lfs_hip_warp_rgb(first, {input, 1}, LFS_HIP_U8, {output, 0}, &lens, &error));
        check(lfs_hip_download(first, output, 0, result.data(), bytes, &error));
        for (size_t i = 0; i < elements; ++i)
            require(i < large_rgb_count ? std::abs(result[i] - float(rgb_bytes[i]) / 255.f) < 1e-7f : result[i] == -99,
                    "Large byte RGB identity lost its grid tail");
        std::puts("PASS RGB warp: context ownership, offsets, overlap, guards, bounded work, recovery and byte identity");
    }
    check(lfs_hip_context_synchronize(first, &error));
    {
        LfsHipBuffer* fill_indices = nullptr;
        check(lfs_hip_buffer_allocate(first, 16, &fill_indices, &error));
        LfsHipLayout layout{};
        layout.rank = 2;
        layout.dims[0] = 2;
        layout.dims[1] = 3;
        layout.strides[0] = 8;
        layout.strides[1] = 2;
        layout.elements = 6;
        std::vector<uint8_t> original(64, 0x35), expected = original, observed(64);
        int32_t positions[] = {2, 0};
        const float fill_value = -1.25f;
        uint32_t value_bits = 0;
        std::memcpy(&value_bits, &fill_value, sizeof(value_bits));
        for (size_t row = 0; row < 2; ++row)
            for (int32_t column : positions)
                std::memcpy(expected.data() + 8 + (row * 8 + size_t(column) * 2) * 4,
                            &value_bits, sizeof(value_bits));
        check(lfs_hip_upload(first, output, 0, original.data(), original.size(), &error));
        check(lfs_hip_upload(first, fill_indices, 4, positions, sizeof(positions), &error));
        check(lfs_hip_index_fill(first, {output, 8}, LFS_HIP_F32, {fill_indices, 4}, LFS_HIP_I32,
                                 &layout, 1, 2, value_bits, &error));
        check(lfs_hip_download(first, output, 0, observed.data(), observed.size(), &error));
        require(observed == expected, "HIP index fill changed strided padding or float bits");
        positions[1] = 3;
        check(lfs_hip_upload(first, fill_indices, 4, positions, sizeof(positions), &error));
        rejected(lfs_hip_index_fill(first, {output, 8}, LFS_HIP_F32, {fill_indices, 4}, LFS_HIP_I32,
                                    &layout, 1, 2, value_bits, &error));
        check(lfs_hip_download(first, output, 0, observed.data(), observed.size(), &error));
        require(observed == expected, "Invalid HIP index fill partially changed output");
        check(lfs_hip_buffer_release(first, fill_indices, &error));
    }

    {
        LfsHipBuffer *opacity = nullptr, *environment = nullptr;
        check(lfs_hip_buffer_allocate(first, 4 * sizeof(float), &opacity, &error));
        check(lfs_hip_buffer_allocate(first, 3 * sizeof(float), &environment, &error));
        const float rgb[] = {0.f, 0.5f, 1.f, 0.25f, 0.75f, 0.f,
                             0.f, 0.5f, 1.f, 0.25f, 0.75f, 0.f};
        const float alpha[] = {1.f, 1.f, 1.f, 1.f};
        const float env[] = {0.2f, 0.3f, 0.4f};
        std::vector<uint8_t> guard(64, 0x4b), observed(guard.size());
        check(lfs_hip_upload(first, input, 0, rgb, sizeof(rgb), &error));
        check(lfs_hip_upload(first, opacity, 0, alpha, sizeof(alpha), &error));
        check(lfs_hip_upload(first, environment, 0, env, sizeof(env), &error));
        check(lfs_hip_upload(first, output, 0, guard.data(), guard.size(), &error));
        LfsHipEnvironment params{};
        params.rotation[0] = params.rotation[4] = params.rotation[8] = 1.f;
        params.full_width = params.band_width = 2;
        params.full_height = params.band_height = 2;
        params.env_width = params.env_height = 1;
        params.focal_x = params.focal_y = 2.f;
        check(lfs_hip_environment_composite(first, {input, 0}, {opacity, 0}, {environment, 0},
                                             {output, 8}, &params, &error));
        check(lfs_hip_download(first, output, 0, observed.data(), observed.size(), &error));
        for (size_t i = 0; i < 12; ++i)
            require(observed[8 + i] == uint8_t(rgb[(i % 3) * 4 + i / 3] * 255.f + 0.5f),
                    "HIP environment alpha-one output differs from RGB");
        require(std::all_of(observed.begin(), observed.begin() + 8, [](uint8_t value) { return value == 0x4b; }),
                "HIP environment composite changed preceding guard bytes");
        const auto valid_output = observed;
        params.env_width = 0;
        rejected(lfs_hip_environment_composite(first, {input, 0}, {opacity, 0}, {environment, 0},
                                                {output, 8}, &params, &error));
        check(lfs_hip_download(first, output, 0, observed.data(), observed.size(), &error));
        require(observed == valid_output, "Invalid HIP environment dimensions partially changed output");
        check(lfs_hip_buffer_release(first, opacity, &error));
        check(lfs_hip_buffer_release(first, environment, &error));
    }

    check(lfs_hip_buffer_release(first, input, &error));
    // Leave output live, with queued work: context destruction owns its cleanup.
    check(lfs_hip_fill_f32(first, output, 0, elements, 42, &error));
    check(lfs_hip_context_destroy(first, &error));
    check(lfs_hip_context_destroy(second, &error));
    require(FreeLibrary(module), "Could not unload HIP module");
    std::puts("PASS: device errors, ownership, bounds, offsets, queue ordering and live-buffer teardown");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
