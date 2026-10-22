/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../src/core/tensor/backend/hip/hip_backend_ops.hpp"
#include "../../src/core/tensor/backend/readback_buffer.hpp"
#include "../../src/core/tensor/backend/export_pipeline.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <core/detail/fused_pointwise.hpp>
#include <core/detail/tensor_dtype_dispatch.hpp>
#include <core/failure_report.hpp>
#include <core/tensor.hpp>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <source_location>
#include <stdexcept>
#include <vector>
// This isolated executable links the real adapter without the rest of lfs_core.
// Tensor factory paths belong to integrated contracts and fail explicitly here.
// Diagnostic assertions throw locally without pulling in the application logger.
namespace lfs::core {
    Tensor::Tensor(Tensor&&) noexcept = default;
    Tensor& Tensor::operator=(Tensor&&) = default;
    Tensor::~Tensor() = default;
    Tensor Tensor::empty(TensorShape, Device, DataType, bool) { throw std::runtime_error("No Tensor factory linked in storage-only test"); }
    Tensor Tensor::contiguous() const { throw std::runtime_error("No Tensor layout factory linked in storage-only test"); }
    void Tensor::materialize_deferred_slow() { throw std::runtime_error("No deferred Tensor factory linked in storage-only test"); }
    GpuBackendScope::GpuBackendScope(GpuBackend) { throw std::runtime_error("No Tensor factory scope linked in isolated storage test"); }
    GpuBackendScope::~GpuBackendScope() = default;
    Tensor Tensor::from_external_owner(void*, TensorShape, Device, DataType, std::shared_ptr<void>, size_t, cudaStream_t, std::string) {
        throw std::runtime_error("No external Tensor factory linked in isolated storage test");
    }
    std::string capture_host_stacktrace(size_t) { return "isolated HIP storage test"; }
    namespace detail {
        [[noreturn]] void assertion_failed(std::string_view, std::string_view, std::string_view message, SourceSite) {
            throw std::runtime_error(std::string(message));
        }
    }
}
namespace lfs::core::internal {
    std::tuple<Tensor,Tensor> export_kmeans_sh(ExportKernels&,const Tensor&,int,int,int,int) { throw std::runtime_error("No export pipeline linked in storage-only test"); }
    void export_assign_sh3(ExportKernels&,const Tensor&,const Tensor&,const Tensor&,Tensor&,bool,bool) { throw std::runtime_error("No export pipeline linked in storage-only test"); }
    void export_decimate_candidates(ExportKernels&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,int,std::vector<uint32_t>&,std::vector<float>&) { throw std::runtime_error("No export pipeline linked in storage-only test"); }
    DecimateMerge export_decimate_merge(ExportKernels&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,int,const std::vector<int>&,const std::vector<uint32_t>&,const std::vector<uint32_t>&,const std::vector<uint32_t>&,size_t) { throw std::runtime_error("No export pipeline linked in storage-only test"); }
    GpuBackendOps::~GpuBackendOps() = default;
    void GpuBackendOps::push_range(const char*) { throw std::runtime_error("No ranges in isolated test"); }
    void GpuBackendOps::pop_range() { throw std::runtime_error("No ranges in isolated test"); }
    void GpuBackendOps::name_queue(ExecContext, const char*) {}
    Tensor GpuBackendOps::image_warp(const Tensor&, const UndistortParams&, int, bool, Tensor*, ExecContext) { throw std::runtime_error("No portable image warp linked in isolated test"); }
    // Actual Tensor image allocation is tested by core_tensor_training_smoke.
    Tensor HipBackendOps::image_undistort(const Tensor&, const UndistortParams&, bool, ExecContext) { throw std::runtime_error("No Tensor camera allocation linked in storage-only test"); }
    Tensor HipBackendOps::image_resize_prior(const Tensor&, int, int, bool, ExecContext) { throw std::runtime_error("No Tensor prior allocation linked in storage-only test"); }
    Tensor HipBackendOps::image_warp(const Tensor&, const UndistortParams&, int, bool, Tensor*, ExecContext) { throw std::runtime_error("No Tensor image allocation linked in storage-only test"); }

} // namespace lfs::core::internal
using namespace lfs::core;
using namespace lfs::core::internal;
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
template <class F>
void reject(F&& f, std::source_location site = std::source_location::current()) {
    bool rejected = false;
    try {
        f();
    } catch (const std::exception&) { rejected = true; }
    if (!rejected)
        throw std::runtime_error("Expected explicit HIP error at test line " + std::to_string(site.line()));
}
template <class T>
constexpr DataType type_id() {
    if constexpr (std::is_same_v<T, float>)
        return DataType::Float32;
    else if constexpr (std::is_same_v<T, detail::tensor_half_t>)
        return DataType::Float16;
    else if constexpr (std::is_same_v<T, int32_t>)
        return DataType::Int32;
    else if constexpr (std::is_same_v<T, int64_t>)
        return DataType::Int64;
    else if constexpr (std::is_same_v<T, uint8_t>)
        return DataType::UInt8;
    else if constexpr (std::is_same_v<T, bool>)
        return DataType::Bool;
    else
        return DataType::UInt32;
}
template <class To, class From>
To conversion_oracle(From value) {
    if constexpr (std::is_same_v<From, detail::tensor_half_t>)
        return conversion_oracle<To>(detail::tensor_half_to_float(value));
    else if constexpr (std::is_same_v<To, detail::tensor_half_t>)
        return detail::tensor_float_to_half(static_cast<float>(value));
    else if constexpr (std::is_same_v<To, uint8_t>)
        return detail::torch_uint8_cast(value);
    else if constexpr (std::is_integral_v<To> && !std::is_same_v<To, bool> && std::is_same_v<From, float>)
        return detail::saturating_float_cast<To>(value);
    else
        return static_cast<To>(value);
}
template <class From, class To>
void test_conversion(GpuBackendOps& gpu) {
    constexpr size_t count = 263;
    std::array<From, count> input{};
    const float values[] = {NAN, INFINITY, -INFINITY, -0.f, 0.f, -257.9f, -256.f, -255.9f, 255.9f, 256.f, 257.9f, 65504.f, 65520.f, 0x1p-24f, 0x1p-25f, 1.00048828125f, 0x1p31f, -0x1p31f, 0x1p32f, 0x1p63f, -0x1p63f, 0x1.fffffep30f};
    for (size_t i = 0; i < count; ++i) {
        if constexpr (std::is_same_v<From, float> || std::is_same_v<From, detail::tensor_half_t>)
            input[i] = conversion_oracle<From>(values[i % std::size(values)]);
        else if constexpr (std::is_same_v<From, bool>)
            input[i] = bool(i % 2);
        else {
            const From integers[] = {std::numeric_limits<From>::lowest(), std::numeric_limits<From>::max(), From(0), From(1), From(127), From(255), static_cast<From>(-1), static_cast<From>(16777217)};
            input[i] = integers[i % std::size(integers)];
        }
    }
    auto source = gpu.allocate(sizeof(input) + 32, 256, {}), target = gpu.allocate(count * sizeof(To) + 32, 256, {});
    auto in = source, out = target;
    in.byte_offset = out.byte_offset = 16;
    in.dtype = type_id<From>();
    out.dtype = type_id<To>();
    auto host = [](void* p) { return StorageRef{GpuBackend::HIP, p, 0, DataType::UInt8, nullptr, STORAGE_REF_HOST_MEMORY}; };
    gpu.copy_host_to_device({host(input.data()), in, sizeof(input)});
    gpu.memset({target, count * sizeof(To) + 32, 0x5a, true});
    gpu.convert_type(in, out, count, {});
    std::array<To, count> result{};
    gpu.copy_device_to_host({out, host(result.data()), sizeof(result)});
    for (size_t i = 0; i < count; ++i) {
        const auto expected = conversion_oracle<To>(input[i]);
        bool equal;
        if constexpr (std::is_same_v<To, float>)
            equal = std::isnan(expected) ? std::isnan(result[i]) : std::bit_cast<uint32_t>(expected) == std::bit_cast<uint32_t>(result[i]);
        else if constexpr (std::is_same_v<To, detail::tensor_half_t>)
            equal = std::isnan(float(expected)) ? std::isnan(float(result[i])) : std::memcmp(&expected, &result[i], sizeof(To)) == 0;
        else
            equal = result[i] == expected;
        if (!equal) {
            std::printf("conversion %u -> %u at %zu\n", unsigned(in.dtype), unsigned(out.dtype), i);
            throw std::runtime_error("HIP conversion differs from CPU contract");
        }
    }
    std::vector<uint8_t> guards(count * sizeof(To) + 32);
    gpu.copy_device_to_host({target, host(guards.data()), guards.size()});
    for (size_t i = 0; i < guards.size(); ++i)
        if (i < 16 || i >= 16 + sizeof(result))
            require(guards[i] == 0x5a, "HIP conversion overwrote output guards");
    gpu.deallocate(source, {});
    gpu.deallocate(target, {});
}
template <class From>
void test_conversion_destinations(GpuBackendOps& gpu) {
    test_conversion<From, float>(gpu);
    test_conversion<From, detail::tensor_half_t>(gpu);
    test_conversion<From, int32_t>(gpu);
    test_conversion<From, int64_t>(gpu);
    test_conversion<From, uint8_t>(gpu);
    test_conversion<From, bool>(gpu);
    test_conversion<From, uint32_t>(gpu);
}
void test_gathers(GpuBackendOps& gpu) {
    auto host = [](void* p) { return StorageRef{GpuBackend::HIP, p, 0, DataType::UInt8, nullptr, STORAGE_REF_HOST_MEMORY}; };
    auto offset = [](size_t i, const StridedLayout& layout) {size_t result=0;for(size_t r=layout.rank;r;--r){result+=(i%layout.dims[r-1])*layout.strides[r-1];i/=layout.dims[r-1];}return result; };
    for (auto dtype : {DataType::Float32, DataType::Float16, DataType::Int32, DataType::Int64, DataType::UInt8, DataType::Bool, DataType::UInt32})
        for (auto index_dtype : {DataType::Int32, DataType::Int64})
            for (unsigned dim = 0; dim < 3; ++dim)
                for (bool select : {false, true})
                    for (int boundary = 0; boundary < 3; ++boundary) {
                        StridedLayout source{};
                        source.rank = 3;
                        source.dims = {3, 5, 7};
                        source.strides = {1, 3, 15};
                        source.element_count = 105;
                        StridedLayout idx{};
                        idx.rank = select ? 1 : 3;
                        if (select) {
                            idx.dims[0] = 9;
                            idx.strides[0] = 1;
                            idx.element_count = 9;
                        } else {
                            idx.dims = source.dims;
                            idx.dims[dim] = 9;
                            idx.strides = {1, idx.dims[0], idx.dims[0] * idx.dims[1]};
                            idx.element_count = idx.dims[0] * idx.dims[1] * idx.dims[2];
                        }
                        const size_t extent = source.dims[dim], width = dtype_size(dtype), iw = dtype_size(index_dtype);
                        auto shape = source.dims;
                        shape[dim] = 9;
                        const size_t outputs = shape[0] * shape[1] * shape[2];
                        std::vector<uint8_t> values(105 * width);
                        for (size_t i = 0; i < values.size(); ++i)
                            values[i] = dtype == DataType::Bool ? uint8_t(i % 2) : uint8_t(i * 37 + 11);
                        std::vector<int64_t> indices(idx.element_count);
                        for (size_t i = 0; i < indices.size(); ++i)
                            indices[i] = boundary ? (i % 4 == 0 ? -1 : i % 4 == 1 ? int64_t(extent)
                                                                   : i % 4 == 2   ? (index_dtype == DataType::Int64 ? INT64_MIN : INT32_MIN)
                                                                                  : (index_dtype == DataType::Int64 ? INT64_MAX : INT32_MAX))
                                                  : int64_t(i % extent);
                        std::vector<int32_t> narrow(indices.size());
                        auto base_input = gpu.allocate(values.size() + 32, 256, {}), base_index = gpu.allocate(indices.size() * iw + 32, 256, {}), base_output = gpu.allocate(outputs * width + 32, 256, {});
                        auto in = base_input, ix = base_index, out = base_output;
                        in.dtype = out.dtype = dtype;
                        ix.dtype = index_dtype;
                        in.byte_offset = ix.byte_offset = out.byte_offset = 16;
                        gpu.copy_host_to_device({host(values.data()), in, values.size()});
                        auto upload = [&] {if(index_dtype==DataType::Int32){std::transform(indices.begin(),indices.end(),narrow.begin(),[](int64_t value){return static_cast<int32_t>(value);});gpu.copy_host_to_device({host(narrow.data()),ix,narrow.size()*4});}else gpu.copy_host_to_device({host(indices.data()),ix,indices.size()*8}); };
                        upload();
                        gpu.memset({base_output, outputs * width + 32, 0x5a, true});
                        IndexProgram program{};
                        program.dim = int(dim);
                        program.boundary_mode = boundary;
                        program.index_size = idx.element_count;
                        program.total_elements = outputs;
                        auto run = [&] {if(select)gpu.index_select(in,ix,out,source,program,{});else gpu.gather(in,ix,out,source,idx,program,{}); };
                        run();
                        std::vector<uint8_t> result(outputs * width + 32);
                        gpu.copy_device_to_host({base_output, host(result.data()), result.size()});
                        for (size_t i = 0; i < outputs; ++i) {
                            size_t remaining = i, src = 0, selected = 0;
                            for (size_t r = 3; r; --r) {
                                const size_t coordinate = remaining % shape[r - 1];
                                remaining /= shape[r - 1];
                                if (r - 1 == dim)
                                    selected = coordinate;
                                else
                                    src += coordinate * source.strides[r - 1];
                            }
                            int64_t index = indices[select ? selected : offset(i, idx)];
                            if (boundary == 1)
                                index = std::clamp(index, int64_t(0), int64_t(extent - 1));
                            if (boundary == 2) {
                                index %= int64_t(extent);
                                if (index < 0)
                                    index += int64_t(extent);
                            }
                            src += size_t(index) * source.strides[dim];
                            require(std::memcmp(result.data() + 16 + i * width, values.data() + src * width, width) == 0, "HIP gather/select changed an element or its bits");
                        }
                        for (size_t i = 0; i < result.size(); ++i)
                            if (i < 16 || i >= 16 + outputs * width)
                                require(result[i] == 0x5a, "HIP gather overwrote guards");
                        if (boundary == 0) {
                            indices.back() = -1;
                            upload();
                            gpu.memset({base_output, outputs * width + 32, 0x5a, true});
                            reject(run);
                            gpu.copy_device_to_host({base_output, host(result.data()), result.size()});
                            require(std::all_of(result.begin(), result.end(), [](auto x) { return x == 0x5a; }), "Invalid gather partially changed output");
                            indices.back() = 0;
                            upload();
                            run();
                            auto alias = ix;
                            alias.dtype = in.dtype;
                            reject([&] {if(select)gpu.index_select(in,ix,alias,source,program,{});else gpu.gather(in,ix,alias,source,idx,program,{}); });
                            program.dim = 3;
                            reject(run);
                        }
                        gpu.deallocate(base_input, {});
                        gpu.deallocate(base_index, {});
                        gpu.deallocate(base_output, {});
                    }
}
void test_index_updates(GpuBackendOps& gpu) {
    auto host = [](void* p) { return StorageRef{GpuBackend::HIP, p, 0, DataType::UInt8, nullptr, STORAGE_REF_HOST_MEMORY}; };
    for (auto dtype : {DataType::Float32, DataType::Float16, DataType::Int32, DataType::Int64, DataType::UInt8, DataType::Bool, DataType::UInt32})
        for (auto index_dtype : {DataType::Int32, DataType::Int64})
            for (unsigned axis = 0; axis < 3; ++axis)
                for (bool repeated : {false, true})
                    for (bool add : {false, true}) {
                        if (add && dtype != DataType::Float32 && dtype != DataType::Int32)
                            continue;
                        const size_t width = dtype_size(dtype), iw = dtype_size(index_dtype), index_count = repeated ? 263 : 9;
                        StridedLayout layout{};
                        layout.rank = 3;
                        layout.dims = {3, 5, 7};
                        layout.dims[axis] = 11;
                        layout.strides[2] = 2;
                        layout.strides[1] = layout.dims[2] * 2 + 3;
                        layout.strides[0] = layout.dims[1] * layout.strides[1] + 5;
                        layout.element_count = layout.dims[0] * layout.dims[1] * layout.dims[2];
                        const size_t span = (layout.dims[0] - 1) * layout.strides[0] + (layout.dims[1] - 1) * layout.strides[1] + (layout.dims[2] - 1) * layout.strides[2] + 1;
                        auto source_dims = layout.dims;
                        source_dims[axis] = index_count;
                        const size_t elements = source_dims[0] * source_dims[1] * source_dims[2];
                        std::vector<int64_t> indices(index_count);
                        std::vector<int32_t> narrow(index_count);
                        for (size_t i = 0; i < index_count; ++i)
                            indices[i] = int64_t((i * 7) % 11);
                        std::vector<uint8_t> initial(span * width + 32, 0x37), source_bytes(elements * width), expected;
                        if (add)
                            for (size_t a = 0; a < layout.dims[0]; ++a)
                                for (size_t b = 0; b < layout.dims[1]; ++b)
                                    for (size_t c = 0; c < layout.dims[2]; ++c) {
                                        const size_t pos = 16 + (a * layout.strides[0] + b * layout.strides[1] + c * layout.strides[2]) * width;
                                        if (dtype == DataType::Float32) {
                                            float v = 0.5f;
                                            std::memcpy(initial.data() + pos, &v, 4);
                                        } else {
                                            int32_t v = 1024;
                                            std::memcpy(initial.data() + pos, &v, 4);
                                        }
                                    }
                        expected = initial;
                        for (size_t i = 0; i < elements; ++i) {
                            size_t remainder = i, base = 0, selected = 0;
                            for (unsigned r = 3; r; --r) {
                                const size_t coordinate = remainder % source_dims[r - 1];
                                remainder /= source_dims[r - 1];
                                if (r - 1 == axis)
                                    selected = coordinate;
                                else
                                    base += coordinate * layout.strides[r - 1];
                            }
                            const size_t destination = 16 + (base + size_t(indices[selected]) * layout.strides[axis]) * width;
                            if (add) {
                                if (dtype == DataType::Float32) {
                                    float value = (int(i % 13) - 6) * 0.25f, acc;
                                    std::memcpy(source_bytes.data() + i * 4, &value, 4);
                                    std::memcpy(&acc, expected.data() + destination, 4);
                                    acc += value;
                                    std::memcpy(expected.data() + destination, &acc, 4);
                                } else {
                                    int32_t value = int(i % 13) - 6, acc;
                                    std::memcpy(source_bytes.data() + i * 4, &value, 4);
                                    std::memcpy(&acc, expected.data() + destination, 4);
                                    acc += value;
                                    std::memcpy(expected.data() + destination, &acc, 4);
                                }
                            } else {
                                // Duplicate copy order is unspecified; agreeing sources verify
                                // duplicate safety without claiming deterministic last-writer order.
                                for (size_t byte = 0; byte < width; ++byte)
                                    source_bytes[i * width + byte] = dtype == DataType::Bool ? uint8_t(base % 2) : uint8_t(base * 37 + byte * 11 + 9);
                                std::memcpy(expected.data() + destination, source_bytes.data() + i * width, width);
                            }
                        }
                        auto base_out = gpu.allocate(initial.size(), 256, {}), base_idx = gpu.allocate(index_count * iw + 32, 256, {}), base_src = gpu.allocate(source_bytes.size() + 32, 256, {});
                        auto out = base_out, idx = base_idx, src = base_src;
                        out.byte_offset = idx.byte_offset = src.byte_offset = 16;
                        out.dtype = src.dtype = dtype;
                        idx.dtype = index_dtype;
                        gpu.copy_host_to_device({host(initial.data()), base_out, initial.size()});
                        gpu.copy_host_to_device({host(source_bytes.data()), src, source_bytes.size()});
                        auto upload_indices = [&] {if(index_dtype==DataType::Int32){std::transform(indices.begin(),indices.end(),narrow.begin(),[](int64_t i){return static_cast<int32_t>(i);});gpu.copy_host_to_device({host(narrow.data()),idx,index_count*4});}else gpu.copy_host_to_device({host(indices.data()),idx,index_count*8}); };
                        upload_indices();
                        IndexProgram program{};
                        program.dim = int(axis);
                        program.index_size = index_count;
                        auto run = [&] {if(add)gpu.index_add(out,idx,src,layout,program,{});else gpu.index_copy(out,idx,src,layout,program,{}); };
                        run();
                        std::vector<uint8_t> result(initial.size());
                        gpu.copy_device_to_host({base_out, host(result.data()), result.size()});
                        require(result == expected, "HIP indexed update differs from CPU or changed guards/padding");
                        if (!add) {
                            StridedLayout source_layout{};
                            source_layout.rank = 3;
                            source_layout.dims = source_dims;
                            source_layout.strides = {source_dims[1] * source_dims[2], source_dims[2], 1};
                            source_layout.element_count = elements;
                            program.total_elements = elements;
                            gpu.copy_host_to_device({host(initial.data()), base_out, initial.size()});
                            gpu.scatter(out, idx, src, layout, source_layout, program, {});
                            gpu.copy_device_to_host({base_out, host(result.data()), result.size()});
                            require(result == expected, "HIP scatter differs from indexed update and CPU oracle");
                        }
                        for (auto invalid : {int64_t(-1), int64_t(11)}) {
                            indices.back() = invalid;
                            upload_indices();
                            gpu.copy_host_to_device({host(initial.data()), base_out, initial.size()});
                            reject(run);
                            gpu.copy_device_to_host({base_out, host(result.data()), result.size()});
                            require(result == initial, "Invalid indexed update partially changed model");
                        }
                        indices.back() = 0;
                        upload_indices();
                        run();
                        reject([&] {if(add)gpu.index_add(out,idx,out,layout,program,{});else gpu.index_copy(out,idx,out,layout,program,{}); });
                        auto index_alias = out;
                        index_alias.dtype = DataType::Int32;
                        reject([&] {if(add)gpu.index_add(out,index_alias,src,layout,program,{});else gpu.index_copy(out,index_alias,src,layout,program,{}); });
                        if (!add && dtype != DataType::Float32 && dtype != DataType::Int32)
                            reject([&] { gpu.index_add(out, idx, src, layout, program, {}); });
                        const auto valid_count = program.index_size;
                        program.index_size = SIZE_MAX;
                        reject(run);
                        program.index_size = valid_count;
                        auto overlapping = layout;
                        overlapping.strides[0] = 0;
                        reject([&] {if(add)gpu.index_add(out,idx,src,overlapping,program,{});else gpu.index_copy(out,idx,src,overlapping,program,{}); });
                        program.index_size = 0;
                        gpu.copy_host_to_device({host(initial.data()), base_out, initial.size()});
                        run();
                        gpu.copy_device_to_host({base_out, host(result.data()), result.size()});
                        require(result == initial, "Empty indexed update changed output");
                        program.dim = 3;
                        reject(run);
                        gpu.deallocate(base_out, {});
                        gpu.deallocate(base_idx, {});
                        gpu.deallocate(base_src, {});
                    }
}
void test_index_fill(GpuBackendOps& gpu) {
    auto host = [](void* p) { return StorageRef{GpuBackend::HIP, p, 0, DataType::UInt8, nullptr, STORAGE_REF_HOST_MEMORY}; };
    for (auto dtype : {DataType::Float32, DataType::Int32, DataType::UInt8, DataType::Bool}) {
        const size_t width = dtype == DataType::Float32 || dtype == DataType::Int32 ? 4 : 1;
        std::vector<uint8_t> initial(64, 0x35), expected = initial, result(initial.size());
        StridedLayout layout{};
        layout.rank = 2;
        layout.dims = {2, 3};
        layout.strides = {8, 2};
        layout.element_count = 6;
        int32_t host_indices[] = {2, 0};
        ScalarOperand value{};
        uint32_t bits = 0;
        if (dtype == DataType::Float32) {
            value.kind = ScalarKind::Float;
            value.value.float_value = -1.25f;
            std::memcpy(&bits, &value.value.float_value, sizeof(bits));
        } else if (dtype == DataType::Int32) {
            value.kind = ScalarKind::Int32;
            value.value.int32_value = -17;
            bits = static_cast<uint32_t>(-17);
        } else if (dtype == DataType::UInt8) {
            value.kind = ScalarKind::Int32;
            value.value.int32_value = 231;
            bits = 231;
        } else {
            value.kind = ScalarKind::Bool;
            value.value.bool_value = true;
            bits = 1;
        }
        for (size_t row = 0; row < 2; ++row)
            for (int column : host_indices)
                std::memcpy(expected.data() + 8 + (row * 8 + size_t(column) * 2) * width, &bits, width);
        auto output = gpu.allocate(initial.size(), 256, {}), indices = gpu.allocate(16, 256, {});
        auto base = output;
        output.dtype = dtype;
        output.byte_offset = 8;
        indices.dtype = DataType::Int32;
        indices.byte_offset = 4;
        gpu.copy_host_to_device({host(initial.data()), base, initial.size()});
        gpu.copy_host_to_device({host(host_indices), indices, sizeof(host_indices)});
        IndexProgram program{};
        program.dim = 1;
        program.index_size = 2;
        gpu.index_fill(output, indices, layout, program, value, {});
        gpu.copy_device_to_host({base, host(result.data()), result.size()});
        require(result == expected, "HIP index fill differs from strided CPU oracle or changed padding");
        host_indices[1] = 3;
        gpu.copy_host_to_device({host(host_indices), indices, sizeof(host_indices)});
        reject([&] { gpu.index_fill(output, indices, layout, program, value, {}); });
        gpu.copy_device_to_host({base, host(result.data()), result.size()});
        require(result == expected, "Invalid HIP index fill partially changed destination");
        gpu.deallocate(base, {});
        indices.byte_offset = 0;
        gpu.deallocate(indices, {});
    }
}
void test_strided_writes(GpuBackendOps& gpu) {
    auto host = [](void* p) { return StorageRef{GpuBackend::HIP, p, 0, DataType::UInt8, nullptr, STORAGE_REF_HOST_MEMORY}; };
    for (auto dtype : {DataType::Float32, DataType::Float16, DataType::Int32, DataType::Int64, DataType::UInt8, DataType::Bool, DataType::UInt32})
        for (unsigned mode = 0; mode < 3; ++mode) {
            if (mode == 2 && dtype != DataType::Float32)
                continue;
            const size_t width = dtype_size(dtype), count = 253, span = 275;
            auto base_in = gpu.allocate(count * width + 32, 256, {}), base_out = gpu.allocate(span * width + 32, 256, {});
            auto in = base_in, out = base_out;
            in.byte_offset = out.byte_offset = 16;
            in.dtype = out.dtype = dtype;
            if (mode == 2)
                in.dtype = DataType::Int32;
            std::vector<uint8_t> input(count * width), initial(span * width + 32, 0x37), expected = initial;
            for (size_t i = 0; i < count; ++i) {
                const size_t destination = 16 + (i / 23 + (i % 23) * 12) * width;
                if (mode == 2) {
                    const int32_t special[] = {0, INT32_MIN, INT32_MAX, 16777217, -16777217, 1, -1};
                    const auto value = special[i % 7];
                    const float result = float(value);
                    std::memcpy(input.data() + i * 4, &value, 4);
                    std::memcpy(expected.data() + destination, &result, 4);
                } else {
                    for (size_t b = 0; b < width; ++b)
                        input[i * width + b] = dtype == DataType::Bool ? uint8_t(i % 2) : uint8_t(i * 37 + b * 11 + 3);
                    std::memcpy(expected.data() + destination, input.data() + i * width, width);
                }
            }
            gpu.copy_host_to_device({host(input.data()), in, input.size()});
            gpu.copy_host_to_device({host(initial.data()), base_out, initial.size()});
            StridedLayout layout{};
            layout.rank = 2;
            layout.dims = {11, 23};
            layout.strides = {1, 12};
            layout.element_count = count;
            auto run = [&] {if(mode==0)gpu.strided_scatter(in,out,layout,{});else if(mode==1)gpu.strided_scatter_immediate(in,out,layout,{});else gpu.strided_scatter_int32_to_float32(in,out,layout,{}); };
            run();
            std::vector<uint8_t> result(initial.size());
            gpu.copy_device_to_host({base_out, host(result.data()), result.size()});
            require(result == expected, "HIP strided write changed values, padding or guards");
            layout.strides[1] = 1;
            reject(run);
            layout.strides[1] = SIZE_MAX;
            reject(run);
            layout.strides[1] = 12;
            layout.element_count = 254;
            reject(run);
            gpu.deallocate(base_in, {});
            gpu.deallocate(base_out, {});
        }
}
std::array<uint32_t, 4> cpu_philox(uint64_t counter, uint64_t seed) {
    std::array<uint32_t, 4> c{uint32_t(counter), uint32_t(counter >> 32), 0, 0};
    uint32_t k0 = uint32_t(seed), k1 = uint32_t(seed >> 32);
    for (unsigned r = 0; r < 10; ++r) {
        const uint64_t a = uint64_t(0xD2511F53u) * c[0], b = uint64_t(0xCD9E8D57u) * c[2];
        c = {uint32_t(b >> 32) ^ c[1] ^ k0, uint32_t(b), uint32_t(a >> 32) ^ c[3] ^ k1, uint32_t(a)};
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c;
}
void test_random(GpuBackendOps& gpu) {
    require(cpu_philox(0, 0) == std::array<uint32_t, 4>{0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}, "Philox CPU known-answer vector failed");
    constexpr size_t count = 131073;
    auto base = gpu.allocate(count * 4 + 32, 256, {});
    auto output = base;
    output.byte_offset = 16;
    auto host = [](void* p) { return StorageRef{GpuBackend::HIP, p, 0, DataType::UInt8, nullptr, STORAGE_REF_HOST_MEMORY}; };
    for (unsigned kind = 0; kind < 4; ++kind)
        for (uint64_t seed : {uint64_t(0), uint64_t(0x8000000000000011)}) {
            output.dtype = kind == 2 ? DataType::Int32 : DataType::Float32;
            RandomProgram p{};
            p.count = count;
            p.seed = seed;
            p.first = kind == 1 ? 0.37f : 0.f;
            p.second = 1;
            p.low = -7;
            p.high = 13;
            auto run = [&] {switch(kind){case 0:gpu.uniform(output,p,{});break;case 1:gpu.bernoulli(output,p,{});break;case 2:gpu.randint(output,p,{});break;case 3:gpu.normal(output,{},p,{});break;} };
            gpu.memset({base, count * 4 + 32, 0x5a, true});
            run();
            std::vector<uint8_t> bytes(count * 4 + 32);
            gpu.copy_device_to_host({base, host(bytes.data()), bytes.size()});
            for (size_t i = 0; i < bytes.size(); ++i)
                if (i < 16 || i >= 16 + count * 4)
                    require(bytes[i] == 0x5a, "HIP random wrote beyond output");
            double sum = 0, squares = 0, lag = 0;
            float previous = 0;
            std::array<size_t, 20> bins{};
            for (size_t i = 0; i < count; ++i) {
                if (kind == 2) {
                    int32_t value;
                    std::memcpy(&value, bytes.data() + 16 + i * 4, 4);
                    require(value >= p.low && value < p.high, "HIP randint outside half-open range");
                    ++bins[size_t(value - p.low)];
                } else {
                    float value;
                    std::memcpy(&value, bytes.data() + 16 + i * 4, 4);
                    require(std::isfinite(value), "HIP random produced nonfinite standard sample");
                    if (kind == 0)
                        require(value >= 0 && value < 1, "HIP uniform outside [0,1)");
                    if (kind == 1)
                        require(value == 0 || value == 1, "HIP Bernoulli produced nonbinary value");
                    sum += value;
                    squares += double(value) * value;
                    if (i)
                        lag += double(value) * previous;
                    previous = value;
                }
                if (kind < 3) {
                    const auto words = cpu_philox(i / 4, seed);
                    const auto word = words[i % 4];
                    if (kind == 2) {
                        int32_t actual;
                        std::memcpy(&actual, bytes.data() + 16 + i * 4, 4);
                        const auto expected = int32_t(int64_t(p.low) + int64_t((uint64_t(word) * 20) >> 32));
                        require(actual == expected, "HIP randint differs from CPU Philox mapping");
                    } else {
                        float actual;
                        std::memcpy(&actual, bytes.data() + 16 + i * 4, 4);
                        const float unit = float(word >> 8) * 0x1p-24f;
                        require(actual == (kind == 0 ? unit : (unit < p.first ? 1.f : 0.f)), "HIP uniform/Bernoulli differs from CPU Philox mapping");
                    }
                }
            }
            const double mean = sum / count, variance = squares / count - mean * mean;
            if (kind == 0)
                require(std::abs(mean - 0.5) < 0.005 && std::abs(variance - 1. / 12) < 0.003, "HIP uniform moments outside bounds");
            if (kind == 1)
                require(std::abs(mean - 0.37) < 0.01, "HIP Bernoulli frequency outside bounds");
            if (kind == 2) {
                double chi = 0, expected = double(count) / 20;
                for (auto n : bins)
                    chi += (double(n) - expected) * (double(n) - expected) / expected;
                require(chi < 80, "HIP randint frequency test failed");
            }
            if (kind == 3)
                require(std::abs(mean) < 0.015 && std::abs(variance - 1) < 0.03 && std::abs(lag / (count - 1)) < 0.02, "HIP normal moments/correlation outside bounds");
            run();
            std::vector<uint8_t> repeat(count * 4);
            gpu.copy_device_to_host({output, host(repeat.data()), repeat.size()});
            require(std::memcmp(repeat.data(), bytes.data() + 16, repeat.size()) == 0, "HIP random repeated seed is not reproducible");
            p.count = 263;
            run();
            gpu.copy_device_to_host({output, host(repeat.data()), 263 * 4});
            require(std::memcmp(repeat.data(), bytes.data() + 16, 263 * 4) == 0, "HIP random prefix changed with count");
            auto aligned = output;
            output.byte_offset = 4;
            p.count = 7;
            gpu.memset({base, count * 4 + 32, 0x5a, true});
            run();
            std::array<uint8_t, 48> unaligned{};
            gpu.copy_device_to_host({base, host(unaligned.data()), unaligned.size()});
            require(std::memcmp(unaligned.data() + 4, bytes.data() + 16, 28) == 0, "HIP unaligned random output changed sequence");
            for (size_t i = 0; i < unaligned.size(); ++i)
                if (i < 4 || i >= 32)
                    require(unaligned[i] == 0x5a, "HIP unaligned random output changed guards");
            output = aligned;
            p.count = 0;
            gpu.memset({base, count * 4 + 32, 0x5a, true});
            run();
            std::array<uint8_t, 32> empty{};
            gpu.copy_device_to_host({base, host(empty.data()), empty.size()});
            require(std::all_of(empty.begin(), empty.end(), [](auto v) { return v == 0x5a; }), "HIP empty random wrote output");
            p.count = SIZE_MAX;
            reject(run);
            p.count = 263;
            if (kind == 0) {
                p.first = 1;
                p.second = 0;
            } else if (kind == 1)
                p.first = NAN;
            else if (kind == 2)
                p.high = p.low;
            else
                p.second = -1;
            gpu.memset({base, count * 4 + 32, 0x5a, true});
            reject(run);
            std::array<uint8_t, 32> invalid{};
            gpu.copy_device_to_host({base, host(invalid.data()), invalid.size()});
            require(std::all_of(invalid.begin(), invalid.end(), [](auto v) { return v == 0x5a; }), "Rejected random parameters modified output");
        }
    RandomProgram p{};
    p.count = 263;
    p.first = -0.f;
    p.second = 0.f;
    output.dtype = DataType::Float32;
    gpu.normal(output, {}, p, {});
    std::array<float, 263> zeros{};
    gpu.copy_device_to_host({output, host(zeros.data()), sizeof(zeros)});
    for (auto v : zeros)
        require(v == 0 && std::signbit(v), "HIP zero-std normal lost signed mean");
    for (float probability : {0.f, 1.f}) {
        p.first = probability;
        gpu.bernoulli(output, p, {});
        gpu.copy_device_to_host({output, host(zeros.data()), sizeof(zeros)});
        for (float v : zeros)
            require(v == probability, "HIP endpoint Bernoulli is not constant");
    }
    p.first = p.second = -3.25f;
    gpu.uniform(output, p, {});
    gpu.copy_device_to_host({output, host(zeros.data()), sizeof(zeros)});
    for (float v : zeros)
        require(v == -3.25f, "HIP degenerate uniform changed value");
    p.first = 0;
    p.second = 1;
    gpu.normal(output, {}, p, {});
    gpu.copy_device_to_host({output, host(zeros.data()), sizeof(zeros)});
    p.first = 2;
    p.second = 3;
    gpu.normal(output, {}, p, {});
    std::array<float, 263> affine{};
    gpu.copy_device_to_host({output, host(affine.data()), sizeof(affine)});
    for (size_t i = 0; i < 263; ++i)
        require(std::abs(affine[i] - std::fma(3.f, zeros[i], 2.f)) < 2e-6f, "HIP normal ignored mean/std or changed seed stream");
    p.first = -std::numeric_limits<float>::max();
    p.second = std::numeric_limits<float>::max();
    gpu.uniform(output, p, {});
    gpu.copy_device_to_host({output, host(zeros.data()), sizeof(zeros)});
    for (auto v : zeros)
        require(std::isfinite(v) && v >= p.first && v < p.second, "HIP wide uniform overflowed");
    output.dtype = DataType::Int32;
    p.low = INT32_MIN;
    p.high = INT32_MAX;
    gpu.randint(output, p, {});
    std::array<int32_t, 263> wide{};
    gpu.copy_device_to_host({output, host(wide.data()), sizeof(wide)});
    for (size_t i = 0; i < wide.size(); ++i) {
        const auto w = cpu_philox(i / 4, p.seed)[i % 4];
        const auto expected = int32_t(int64_t(INT32_MIN) + int64_t((uint64_t(w) * uint64_t(4294967295)) >> 32));
        require(wide[i] == expected, "HIP full-width randint overflowed");
    }
    gpu.deallocate(base, {});
}
// Exercise the composed strided broadcast boundary, not just contiguous typed
// pointwise kernels. Integer values deliberately exceed Float32 exact range.
void test_integral_broadcast(GpuBackendOps& gpu) {
    const auto host = [](void* p) { return StorageRef{GpuBackend::HIP,p,0,DataType::UInt8,nullptr,STORAGE_REF_HOST_MEMORY}; };
    const auto run = [&]<class T>(DataType dtype) {
        constexpr size_t rows=3, columns=5, count=32;
        std::array<T,count> a{},b{};
        if constexpr(sizeof(T)==8) { a[0]=T((uint64_t{1}<<54)+17);a[2]=T(-((int64_t{1}<<54)+9));a[4]=0; }
        else if constexpr(sizeof(T)==4) { a[0]=T(16777217);a[2]=T(16777219);a[4]=T(7); }
        else { a[0]=T(1);a[2]=T(dtype==DataType::Bool?0:251);a[4]=T(0); }
        for(size_t c=0;c<columns;++c)b[c*3]=T(dtype==DataType::Bool?c%2:c);
        if constexpr(sizeof(T)>=4){b[3]=T(a[0]+1);b[6]=a[0];b[9]=T(a[2]-1);}
        auto left=gpu.allocate(count*sizeof(T)+32,16,{}),right=gpu.allocate(count*sizeof(T)+32,16,{});
        auto result=gpu.allocate(count*sizeof(T)+32,16,{});
        const auto base_left=left,base_right=right,base_result=result;
        left.byte_offset=right.byte_offset=result.byte_offset=16;left.dtype=right.dtype=dtype;
        gpu.copy_host_to_device({host(a.data()),left,sizeof(a)});gpu.copy_host_to_device({host(b.data()),right,sizeof(b)});
        StridedLayout al{},bl{},ol{};
        al.rank=bl.rank=ol.rank=2;al.dims={rows,1};al.strides={2,1};al.element_count=rows;
        bl.dims={1,columns};bl.strides={15,3};bl.element_count=columns;
        ol.dims={rows,columns};ol.strides={8,1};ol.element_count=rows*columns;
        const PointwiseOp predicates[]={PointwiseOp::EqualTensor,PointwiseOp::NotEqualTensor,PointwiseOp::LessTensor,
            PointwiseOp::LessEqualTensor,PointwiseOp::GreaterTensor,PointwiseOp::GreaterEqualTensor,
            PointwiseOp::LogicalAndTensor,PointwiseOp::LogicalOrTensor,PointwiseOp::LogicalXorTensor};
        for(auto op:predicates) {
            result.dtype=DataType::Bool;
            std::vector<uint8_t> actual(count*sizeof(T)+32,0xa5),expected=actual;
            gpu.copy_host_to_device({host(actual.data()),base_result,actual.size()});
            PointwiseProgram program{};program.op=op;program.in_dtype=dtype;program.out_dtype=DataType::Bool;
            gpu.broadcast_binary(program,left,al,right,bl,result,ol,{});
            gpu.copy_device_to_host({base_result,host(actual.data()),actual.size()});
            for(size_t row=0;row<rows;++row)for(size_t col=0;col<columns;++col){
                const T x=a[row*2],y=b[col*3];bool value=false;
                switch(op){
                case PointwiseOp::EqualTensor:value=x==y;break;case PointwiseOp::NotEqualTensor:value=x!=y;break;
                case PointwiseOp::LessTensor:value=x<y;break;case PointwiseOp::LessEqualTensor:value=x<=y;break;
                case PointwiseOp::GreaterTensor:value=x>y;break;case PointwiseOp::GreaterEqualTensor:value=x>=y;break;
                case PointwiseOp::LogicalAndTensor:value=bool(x)&&bool(y);break;case PointwiseOp::LogicalOrTensor:value=bool(x)||bool(y);break;
                case PointwiseOp::LogicalXorTensor:value=bool(x)!=bool(y);break;default:throw std::runtime_error("Invalid predicate fixture");}
                expected[16+row*8+col]=uint8_t(value);
            }
            require(actual==expected,"Typed broadcast predicate differs from exact CPU or overwrites guards");
        }
        if(dtype!=DataType::Bool) {
            const PointwiseOp arithmetic[]={PointwiseOp::AddTensor,PointwiseOp::SubTensor,PointwiseOp::MulTensor,
                PointwiseOp::DivTensor,PointwiseOp::ModTensor,PointwiseOp::PowTensor,PointwiseOp::MinimumTensor,PointwiseOp::MaximumTensor};
            for(auto op:arithmetic) {
                result.dtype=dtype;
                std::vector<uint8_t> actual(count*sizeof(T)+32,0xa5),expected=actual;
                gpu.copy_host_to_device({host(actual.data()),base_result,actual.size()});
                PointwiseProgram program{};program.op=op;program.in_dtype=program.out_dtype=dtype;
                gpu.broadcast_binary(program,left,al,right,bl,result,ol,{});
                gpu.copy_device_to_host({base_result,host(actual.data()),actual.size()});
                for(size_t row=0;row<rows;++row)for(size_t col=0;col<columns;++col){
                    using U=std::make_unsigned_t<T>;
                    const T x=a[row*2],y=b[col*3];T value{};
                    const auto wrap=[](U bits){return std::bit_cast<T>(bits);};
                    switch(op) {
                    case PointwiseOp::AddTensor:value=wrap(U(U(x)+U(y)));break;
                    case PointwiseOp::SubTensor:value=wrap(U(U(x)-U(y)));break;
                    case PointwiseOp::MulTensor:value=wrap(U(U(x)*U(y)));break;
                    case PointwiseOp::DivTensor:value=y?T(x/y):T{};break;
                    case PointwiseOp::ModTensor:value=y?T(x%y):T{};break;
                    case PointwiseOp::MinimumTensor:value=std::min(x,y);break;
                    case PointwiseOp::MaximumTensor:value=std::max(x,y);break;
                    case PointwiseOp::PowTensor:{
                        U exponent=U(y),factor=U(x),product=1;
                        if constexpr(std::is_signed_v<T>)if(y<0){value=x==1?T(1):x==T(-1)?(y%2?T(-1):T(1)):T(0);break;}
                        while(exponent){if(exponent&1)product=U(product*factor);factor=U(factor*factor);exponent>>=1;}
                        value=wrap(product);break;
                    }
                    default:throw std::runtime_error("Invalid integer broadcast fixture");
                    }
                    std::memcpy(expected.data()+16+(row*8+col)*sizeof(T),&value,sizeof(T));
                }
                require(actual==expected,"Integer broadcast arithmetic differs from exact CPU or overwrites guards");
            }
        }
        PointwiseProgram invalid{};
        invalid.op=PointwiseOp::EqualTensor;invalid.in_dtype=dtype;invalid.out_dtype=DataType::Bool;
        result.dtype=DataType::Bool;
        auto mixed=right;mixed.dtype=dtype==DataType::Int32?DataType::Int64:DataType::Int32;
        reject([&]{gpu.broadcast_binary(invalid,left,al,mixed,bl,result,ol,{});});
        auto bad_shape=ol;bad_shape.dims[1]=4;bad_shape.element_count=rows*4;
        reject([&]{gpu.broadcast_binary(invalid,left,al,right,bl,result,bad_shape,{});});
        auto alias=left;alias.dtype=DataType::Bool;
        reject([&]{gpu.broadcast_binary(invalid,left,al,right,bl,alias,ol,{});});
        if(dtype==DataType::Bool) {
            invalid.op=PointwiseOp::AddTensor;
            reject([&]{gpu.broadcast_binary(invalid,left,al,right,bl,result,ol,{});});
        }
        gpu.deallocate(base_left,{});gpu.deallocate(base_right,{});gpu.deallocate(base_result,{});
    };
    run.template operator()<uint8_t>(DataType::Bool);
    run.template operator()<int32_t>(DataType::Int32);
    run.template operator()<int64_t>(DataType::Int64);
    run.template operator()<uint32_t>(DataType::UInt32);
    run.template operator()<uint8_t>(DataType::UInt8);
    std::puts("PASS exact Bool/integer strided broadcasts and padding guards");
}

int main(int argc, char** argv) try {
    require(argc == 2, "Expected HIP runtime module");
    HipBackendOps native(std::filesystem::absolute(argv[1]), 0);
    GpuBackendOps& gpu = native;
    test_integral_broadcast(gpu);
    constexpr size_t count = 263;
    auto a = gpu.allocate((count + 16) * 4, 256, {});
    auto b = gpu.allocate((count + 16) * 4, 16, {});
    auto output = gpu.allocate((count + 16) * 4, 16, {});
    const auto base_a = a, base_b = b, base_output = output;
    a.byte_offset = 16;
    b.byte_offset = 32;
    output.byte_offset = 16;
    std::vector<float> x(count), y(count), actual(count), expected(count);
    const float special[] = {-std::numeric_limits<float>::infinity(), -4.f, -1.f, -0.f, 0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
    for (size_t i = 0; i < count; ++i) {
        x[i] = special[i % 12];
        y[i] = special[(i * 7 + 5) % 12];
    }
    auto host = [](void* data) { return StorageRef{GpuBackend::HIP, data, 0, DataType::Float32, nullptr, STORAGE_REF_HOST_MEMORY}; };
    auto upload = [&](StorageRef dst, const std::vector<float>& src) { gpu.copy_host_to_device({host(const_cast<float*>(src.data())), dst, src.size() * 4}); };
    {
        {
            auto upload_slot = native.create_upload_buffer();
            auto destination = gpu.allocate(32*4, 16, {});
            auto view = destination;
            view.byte_offset = 16;
            std::vector<float> input(13), copied(13);
            std::iota(input.begin(), input.end(), 1.f);
            upload_slot->enqueue(view, input.data(), input.size()*4, {});
            reject([&] { upload_slot->enqueue(view, input.data(), input.size()*4, {}); });
            std::fill(input.begin(), input.end(), -20.f);
            upload_slot->wait();
            require(upload_slot->poll() && !upload_slot->poll(), "HIP upload completion consumed twice");
            gpu.copy_device_to_host({view, host(copied.data()), copied.size()*4, true});
            for (size_t i=0; i<copied.size(); ++i) require(copied[i] == float(i+1), "Native HIP upload snapshot/offset failed");
            reject([&] { upload_slot->enqueue(view, nullptr, 4, {}); });
            auto foreign = view;
            foreign.backend = GpuBackend::CUDA;
            reject([&] { upload_slot->enqueue(foreign, input.data(), 4, {}); });
            gpu.deallocate(destination, {});
        }
        {
            auto marker = native.create_fence();
            require(marker->ready(), "Unrecorded HIP native fence is not ready");
            marker->record({});
            marker->wait_on({});
            marker->record({});
            marker->wait();
            require(marker->ready() && marker->ready(), "HIP native fence lost completion state");
            std::atomic<bool> callback{false};
            native.enqueue_host_callback([](void* user) { static_cast<std::atomic<bool>*>(user)->store(true, std::memory_order_release); }, &callback, {});
            marker->record({});
            marker->wait();
            require(callback.load(std::memory_order_acquire), "HIP native callback did not precede fence");
            require(native.queue_ready({}), "HIP native queue not ready after callback wait");
            auto slot = native.create_readback_slot(64);
            auto device = gpu.allocate(32, 16, {});
            upload(device, {1,2,3,4,5,6,7,8});
            slot->append(device, 16, 0);
            auto offset = device;
            offset.byte_offset = 16;
            slot->append(offset, 16, 32);
            marker->record({});
            marker->wait();
            auto* data = static_cast<float*>(slot->data());
            require(data[0]==1 && data[3]==4 && data[8]==5 && data[11]==8 && data[4]==0 && data[15]==0, "HIP packed readback padding or offsets failed");
            slot->release();
            reject([&] { slot->append(device, 32, 48); });
            gpu.deallocate(device, {});
            HipBackendOps leased(std::filesystem::absolute(argv[1]), 0);
            auto lease = leased.execution_lease({});
            require(lease.device == 0 && lease.native_queue && lease.keep_alive, "Invalid opaque HIP native queue lease");
            reject([&] { leased.shutdown(); });
            lease = {};
            leased.shutdown();
        }
        auto staging = gpu.create_readback_buffer();
        require(!staging->poll(nullptr), "Empty readback reported snapshot");
        staging->wait();
        auto source = gpu.allocate(1027 * 4, 16, {});
        std::vector<float> snapshot(1027), downloaded(1027, -7.f);
        for (size_t i = 0; i < snapshot.size(); ++i) snapshot[i] = float(i) * .25f;
        upload(source, snapshot);
        auto foreign = source;
        foreign.backend = GpuBackend::CUDA;
        reject([&] { staging->enqueue(foreign, snapshot.size()*4, {}); });
        auto offset = source;
        offset.byte_offset = 12;
        staging->enqueue(offset, 19*4, {});
        reject([&] { staging->enqueue(source, 4, {}); });
        gpu.load_fill(source, snapshot.size(), {}, {});
        staging->wait();
        reject([&] { staging->enqueue(source, 4, {}); });
        require(staging->poll(downloaded.data()), "Completed readback not ready");
        require(std::equal(downloaded.begin(), downloaded.begin()+19, snapshot.begin()+3), "Readback failed offset snapshot ordering");
        require(downloaded[19] == -7.f && !staging->poll(nullptr), "Readback overrun or double consumption");
        upload(source, snapshot);
        staging->enqueue(source, snapshot.size()*4, {});
        gpu.deallocate(source, {});
        staging->wait();
        require(staging->poll(downloaded.data()) && downloaded == snapshot, "Grown readback changed values");
        const StorageRef empty{GpuBackend::HIP, nullptr, 0, DataType::Float32, nullptr};
        staging->enqueue(empty, 0, {});
        staging->wait();
        require(staging->poll(nullptr), "Empty snapshot cannot be discarded");
        auto token = gpu.bridge({}, {});
        gpu.wait_for(token);
        auto invalid = token;
        invalid.backend = GpuBackend::CUDA;
        reject([&] { gpu.wait_for(invalid); });
        invalid = token;
        ++invalid.value;
        reject([&] { gpu.wait_for(invalid); });
        ExecContext cuda{};
        cuda.cuda_stream = reinterpret_cast<decltype(cuda.cuda_stream)>(uintptr_t(1));
        reject([&] { gpu.bridge(cuda, {}); });
        reject([&] { staging->enqueue(empty, 0, cuda); });
        std::unique_ptr<ReadbackBuffer> retained;
        {
            HipBackendOps temporary(std::filesystem::absolute(argv[1]), 0);
            auto storage = temporary.allocate(4, 16, {});
            temporary.load_fill(storage, 1, {}, {});
            reject([&] { gpu.wait_for(temporary.bridge({}, {})); });
            retained = temporary.create_readback_buffer();
            retained->enqueue(storage, 4, {});
        }
        float scalar = -1;
        retained->wait();
        require(retained->poll(&scalar) && scalar == 0, "Staging failed to retain context/DLL lifetime");
        retained.reset();
        std::puts("PASS HIP async staging: offsets, snapshot ordering, pending guard, grow/reuse, source release, discard, retained context and isolated sync tokens");
    }
    {
        // Offset views and poisoned padding detect out-of-range matrix writes.
        auto backing = gpu.allocate(20 * 4, 16, {});
        auto values = gpu.allocate(3 * 4, 16, {});
        upload(values, {-2.f, 0.f, 3.f});
        auto matrix = backing;
        matrix.byte_offset = 2 * 4;
        std::vector<float> matrix_host(20);
        auto download = [&] {
            gpu.copy_device_to_host({backing, host(matrix_host.data()), matrix_host.size() * 4, true});
        };
        upload(backing, std::vector<float>(20, 19.f));
        gpu.diag(values, matrix, 3, {});
        download();
        for (size_t i = 0; i < 20; ++i) {
            float expected = 19.f;
            if (i >= 2 && i < 11) expected = i == 2 ? -2.f : i == 10 ? 3.f : 0.f;
            require(matrix_host[i] == expected, "HIP diag offset or padding corruption");
        }
        gpu.eye(matrix, 2, 3, {});
        download();
        require(std::vector<float>(matrix_host.begin()+2, matrix_host.begin()+8) == std::vector<float>({1,0,0,0,1,0}), "HIP rectangular eye incorrect");
        const auto unchanged = matrix_host;
        auto foreign = values;
        foreign.backend = GpuBackend::CUDA;
        reject([&] { gpu.diag(foreign, matrix, 3, {}); });
        reject([&] { gpu.diag(matrix, matrix, 3, {}); });
        reject([&] { gpu.eye(matrix, SIZE_MAX, 2, {}); });
        reject([&] { gpu.eye(matrix, 5, 5, {}); });
        reject([&] { gpu.diag(values, matrix, SIZE_MAX, {}); });
        download();
        require(matrix_host == unchanged, "HIP rejected matrix operation modified output");
        gpu.dot_product(values, values, matrix, 3, {});
        download();
        require(matrix_host[2] == 13.f && matrix_host[1] == unchanged[1] && matrix_host[3] == unchanged[3], "HIP dot result or padding incorrect");
        const StorageRef empty{GpuBackend::HIP, nullptr, 0, DataType::Float32, nullptr};
        gpu.dot_product(empty, empty, matrix, 0, {});
        download();
        require(matrix_host[2] == 0.f, "HIP empty dot identity incorrect");
        gpu.eye(empty, 0, 3, {});
        gpu.diag(empty, empty, 0, {});
        const auto before_failure = matrix_host;
        reject([&] { gpu.dot_product(foreign, values, matrix, 3, {}); });
        reject([&] { gpu.dot_product(values, values, matrix, SIZE_MAX, {}); });
        download();
        require(matrix_host == before_failure, "HIP rejected dot modified output");
        gpu.deallocate(values, {});
        gpu.deallocate(backing, {});
        std::puts("PASS HIP matrix building blocks: rectangular eye, offset diag, dot, empty identity, overflow, foreign owner and overlap rejection");
    }

    {
        const StorageRef empty{GpuBackend::HIP, nullptr, 0, DataType::Float32, nullptr};
        gpu.memset({base_output, (count + 16) * 4, 0x5a, true});
        const auto live = gpu.stats().allocated_bytes;
        gpu.record_stream(empty, {});
        gpu.rehome_stream(empty, {});
        gpu.copy_device_to_device({empty, base_output, 0, true});
        gpu.copy_device_to_device({base_output, empty, 0, true});
        gpu.copy_device_to_device({empty, empty, 0, true});
        gpu.copy_host_to_device({host(nullptr), empty, 0, true});
        gpu.copy_device_to_host({empty, host(nullptr), 0, true});
        gpu.memset({empty, 0, 0, true});
        auto invalid = empty;
        invalid.backend = GpuBackend::CUDA;
        reject([&] { gpu.record_stream(invalid, {}); });
        reject([&] { gpu.copy_device_to_device({invalid, base_output, 0}); });
        invalid = empty;
        invalid.byte_offset = 4;
        reject([&] { gpu.record_stream(invalid, {}); });
        reject([&] { gpu.copy_device_to_device({empty, invalid, 0}); });
        invalid = empty;
        invalid.data = reinterpret_cast<void*>(uintptr_t(1));
        reject([&] { gpu.copy_device_to_device({invalid, base_output, 0}); });
        reject([&] { gpu.record_stream(host(nullptr), {}); });
        reject([&] { gpu.copy_device_to_device({empty, base_output, 4}); });
        reject([&] { gpu.copy_host_to_device({host(nullptr), empty, 4}); });
        std::vector<uint8_t> guard((count + 16) * 4);
        gpu.copy_device_to_host({base_output, host(guard.data()), guard.size()});
        require(gpu.stats().allocated_bytes == live && std::all_of(guard.begin(), guard.end(), [](auto value) { return value == 0x5a; }),
                "Empty HIP views allocated memory or changed a live buffer");
    }
    upload(a, x);
    upload(b, y);
    unsigned operations = 0;
    float worst = 0;
    auto run = [&](PointwiseOp id, bool binary, bool scalar_op, bool predicate, auto oracle) {
        gpu.memset({base_output, (count + 16) * 4, 0x5a, true});
        auto out = output;
        out.dtype = predicate ? DataType::Bool : DataType::Float32;
        PointwiseProgram program{};
        program.op = id;
        program.in_dtype = DataType::Float32;
        program.out_dtype = out.dtype;
        program.scalar.value.float_value = 0.7f;
        if (binary)
            gpu.binary(program, a, b, out, count, {});
        else if (scalar_op)
            gpu.scalar(program, a, out, count, {});
        else
            gpu.unary(program, a, out, count, {});
        if (predicate) {
            std::vector<uint8_t> values(count);
            gpu.copy_device_to_host({out, host(values.data()), count});
            for (size_t i = 0; i < count; ++i)
                require(values[i] == uint8_t(oracle(x[i], y[i], 0.7f)), "HIP predicate differs from native CPU functor");
        } else {
            gpu.copy_device_to_host({out, host(actual.data()), count * 4});
            for (size_t i = 0; i < count; ++i) {
                const float value = oracle(x[i], y[i], 0.7f), result = actual[i];
                if (std::isnan(value)) {
                    require(std::isnan(result), "HIP lost NaN");
                    continue;
                }
                if (std::isinf(value)) {
                    require(result == value, "HIP infinity differs");
                    continue;
                }
                require(std::isfinite(result), "HIP produced nonfinite finite result");
                const float relative = std::abs(result - value) / std::max(1.f, std::abs(value));
                worst = std::max(worst, relative);
                if (relative > 3e-5f) {
                    std::printf("op %u index %zu: %.9g vs %.9g\n", unsigned(id), i, result, value);
                    throw std::runtime_error("HIP pointwise differs from native CPU functor");
                }
                if (value == 0 && std::signbit(result) != std::signbit(value)) {
                    std::printf("signed zero op %u index %zu: %.9g vs %.9g\n", unsigned(id), i, result, value);
                    throw std::runtime_error("HIP signed zero differs");
                }
            }
        }
        std::vector<uint8_t> all((count + 16) * 4);
        gpu.copy_device_to_host({base_output, host(all.data()), all.size()});
        const size_t end = output.byte_offset + count * (predicate ? 1 : 4);
        for (size_t i = 0; i < all.size(); ++i)
            if (i < output.byte_offset || i >= end)
                require(all[i] == 0x5a, "HIP pointwise wrote beyond view");
        if (binary) {
            StridedLayout al{}, bl{}, ol{};
            al.rank = bl.rank = ol.rank = 2;
            al.dims[0] = 11;
            al.dims[1] = 1;
            al.strides[0] = 2;
            al.strides[1] = 0;
            al.element_count = 11;
            bl.dims[0] = 1;
            bl.dims[1] = 23;
            bl.strides[0] = 0;
            bl.strides[1] = 3;
            bl.element_count = 23;
            ol.dims[0] = 11;
            ol.dims[1] = 23;
            ol.strides[0] = 24;
            ol.strides[1] = 1;
            ol.element_count = 253;
            gpu.memset({base_output, (count + 16) * 4, 0x5a, true});
            gpu.broadcast_binary(program, a, al, b, bl, out, ol, {});
            gpu.copy_device_to_host({base_output, host(all.data()), all.size()});
            for (size_t row = 0; row < 11; ++row)
                for (size_t col = 0; col < 23; ++col) {
                    const size_t offset = out.byte_offset + (row * 24 + col) * (predicate ? 1 : 4);
                    const float wanted = oracle(x[row * 2], y[col * 3], 0.7f);
                    float value;
                    if (predicate)
                        value = float(all[offset]);
                    else
                        std::memcpy(&value, all.data() + offset, 4);
                    require(std::isnan(wanted) ? std::isnan(value) : (std::isinf(wanted) ? value == wanted : (std::isfinite(value) && std::abs(value - wanted) <= 3e-5f * std::max(1.f, std::abs(wanted)))),
                            "HIP strided broadcast differs from CPU functor");
                    if (wanted == 0)
                        require(std::signbit(value) == std::signbit(wanted), "HIP broadcast signed zero differs");
                }
            for (size_t offset = 0; offset < all.size(); ++offset) {
                const size_t width = predicate ? 1 : 4;
                const size_t local = offset >= out.byte_offset ? (offset - out.byte_offset) / width : SIZE_MAX;
                if (offset < out.byte_offset || local > 262 || local % 24 == 23)
                    require(all[offset] == 0x5a, "HIP broadcast overwrote padding");
            }
            auto bad = ol;
            bad.dims[1] = 22;
            bad.element_count = 242;
            reject([&] { gpu.broadcast_binary(program, a, al, b, bl, out, bad, {}); });
            bad = ol;
            bad.strides[0] = 1;
            reject([&] { gpu.broadcast_binary(program, a, al, b, bl, out, bad, {}); });
            bad = al;
            bad.element_count = 12;
            reject([&] { gpu.broadcast_binary(program, a, bad, b, bl, out, ol, {}); });
            reject([&] { gpu.broadcast_binary(program, a, al, b, bl, a, ol, {}); });
        }
        ++operations;
    };
    run(PointwiseOp::AddScalar, false, true, false, [](float x, float y, float scalar) { return float(ops::add_op{}(x, scalar)); });
    run(PointwiseOp::SubScalar, false, true, false, [](float x, float y, float scalar) { return float(ops::sub_op{}(x, scalar)); });
    run(PointwiseOp::MulScalar, false, true, false, [](float x, float y, float scalar) { return float(ops::mul_op{}(x, scalar)); });
    run(PointwiseOp::DivScalar, false, true, false, [](float x, float y, float scalar) { return float(ops::div_op{}(x, scalar)); });
    run(PointwiseOp::AddTensor, true, false, false, [](float x, float y, float scalar) { return float(ops::add_op{}(x, y)); });
    run(PointwiseOp::SubTensor, true, false, false, [](float x, float y, float scalar) { return float(ops::sub_op{}(x, y)); });
    run(PointwiseOp::MulTensor, true, false, false, [](float x, float y, float scalar) { return float(ops::mul_op{}(x, y)); });
    run(PointwiseOp::DivTensor, true, false, false, [](float x, float y, float scalar) { return float(ops::div_op{}(x, y)); });
    run(PointwiseOp::PowScalar, false, true, false, [](float x, float y, float scalar) { return float(ops::pow_op{}(x, scalar)); });
    run(PointwiseOp::ModScalar, false, true, false, [](float x, float y, float scalar) { return float(ops::mod_op{}(x, scalar)); });
    run(PointwiseOp::Abs, false, false, false, [](float x, float y, float scalar) { return float(ops::abs_op{}(x)); });
    run(PointwiseOp::Neg, false, false, false, [](float x, float y, float scalar) { return float(ops::neg_op{}(x)); });
    run(PointwiseOp::Exp, false, false, false, [](float x, float y, float scalar) { return float(ops::exp_op{}(x)); });
    run(PointwiseOp::Log, false, false, false, [](float x, float y, float scalar) { return float(ops::log_op{}(x)); });
    run(PointwiseOp::Sqrt, false, false, false, [](float x, float y, float scalar) { return float(ops::sqrt_op{}(x)); });
    run(PointwiseOp::Sigmoid, false, false, false, [](float x, float y, float scalar) { return float(ops::sigmoid_op{}(x)); });
    run(PointwiseOp::Relu, false, false, false, [](float x, float y, float scalar) { return float(ops::relu_op{}(x)); });
    run(PointwiseOp::Square, false, false, false, [](float x, float y, float scalar) { return float(ops::square_op{}(x)); });
    run(PointwiseOp::Tanh, false, false, false, [](float x, float y, float scalar) { return float(ops::tanh_op{}(x)); });
    run(PointwiseOp::Rsqrt, false, false, false, [](float x, float y, float scalar) { return float(ops::rsqrt_op{}(x)); });
    run(PointwiseOp::Sign, false, false, false, [](float x, float y, float scalar) { return float(ops::sign_op{}(x)); });
    run(PointwiseOp::Reciprocal, false, false, false, [](float x, float y, float scalar) { return float(ops::reciprocal_op{}(x)); });
    run(PointwiseOp::Floor, false, false, false, [](float x, float y, float scalar) { return float(ops::floor_op{}(x)); });
    run(PointwiseOp::Ceil, false, false, false, [](float x, float y, float scalar) { return float(ops::ceil_op{}(x)); });
    run(PointwiseOp::Round, false, false, false, [](float x, float y, float scalar) { return float(ops::round_op{}(x)); });
    run(PointwiseOp::PowTensor, true, false, false, [](float x, float y, float scalar) { return float(ops::pow_op{}(x, y)); });
    run(PointwiseOp::ModTensor, true, false, false, [](float x, float y, float scalar) { return float(ops::mod_op{}(x, y)); });
    run(PointwiseOp::MaximumTensor, true, false, false, [](float x, float y, float scalar) { return float(ops::maximum_op{}(x, y)); });
    run(PointwiseOp::MinimumTensor, true, false, false, [](float x, float y, float scalar) { return float(ops::minimum_op{}(x, y)); });
    run(PointwiseOp::EqualTensor, true, false, true, [](float x, float y, float scalar) { return float(ops::equal_op{}(x, y)); });
    run(PointwiseOp::NotEqualTensor, true, false, true, [](float x, float y, float scalar) { return float(ops::not_equal_op{}(x, y)); });
    run(PointwiseOp::LessTensor, true, false, true, [](float x, float y, float scalar) { return float(ops::less_op{}(x, y)); });
    run(PointwiseOp::LessEqualTensor, true, false, true, [](float x, float y, float scalar) { return float(ops::less_equal_op{}(x, y)); });
    run(PointwiseOp::GreaterTensor, true, false, true, [](float x, float y, float scalar) { return float(ops::greater_op{}(x, y)); });
    run(PointwiseOp::GreaterEqualTensor, true, false, true, [](float x, float y, float scalar) { return float(ops::greater_equal_op{}(x, y)); });
    run(PointwiseOp::LogicalAndTensor, true, false, true, [](float x, float y, float scalar) { return float(ops::logical_and_op{}(x, y)); });
    run(PointwiseOp::LogicalOrTensor, true, false, true, [](float x, float y, float scalar) { return float(ops::logical_or_op{}(x, y)); });
    run(PointwiseOp::LogicalXorTensor, true, false, true, [](float x, float y, float scalar) { return float(ops::logical_xor_op{}(x, y)); });
    run(PointwiseOp::Exp2, false, false, false, [](float x, float y, float scalar) { return float(ops::exp2_op{}(x)); });
    run(PointwiseOp::Log2, false, false, false, [](float x, float y, float scalar) { return float(ops::log2_op{}(x)); });
    run(PointwiseOp::Log10, false, false, false, [](float x, float y, float scalar) { return float(ops::log10_op{}(x)); });
    run(PointwiseOp::Log1p, false, false, false, [](float x, float y, float scalar) { return float(ops::log1p_op{}(x)); });
    run(PointwiseOp::Sin, false, false, false, [](float x, float y, float scalar) { return float(ops::sin_op{}(x)); });
    run(PointwiseOp::Cos, false, false, false, [](float x, float y, float scalar) { return float(ops::cos_op{}(x)); });
    run(PointwiseOp::Tan, false, false, false, [](float x, float y, float scalar) { return float(ops::tan_op{}(x)); });
    run(PointwiseOp::Asin, false, false, false, [](float x, float y, float scalar) { return float(ops::asin_op{}(x)); });
    run(PointwiseOp::Acos, false, false, false, [](float x, float y, float scalar) { return float(ops::acos_op{}(x)); });
    run(PointwiseOp::Atan, false, false, false, [](float x, float y, float scalar) { return float(ops::atan_op{}(x)); });
    run(PointwiseOp::Sinh, false, false, false, [](float x, float y, float scalar) { return float(ops::sinh_op{}(x)); });
    run(PointwiseOp::Cosh, false, false, false, [](float x, float y, float scalar) { return float(ops::cosh_op{}(x)); });
    run(PointwiseOp::Gelu, false, false, false, [](float x, float y, float scalar) { return float(ops::gelu_op{}(x)); });
    run(PointwiseOp::Swish, false, false, false, [](float x, float y, float scalar) { return float(ops::swish_op{}(x)); });
    run(PointwiseOp::Trunc, false, false, false, [](float x, float y, float scalar) { return float(ops::trunc_op{}(x)); });
    run(PointwiseOp::IsNan, false, false, true, [](float x, float y, float scalar) { return float(ops::isnan_op{}(x)); });
    run(PointwiseOp::IsInf, false, false, true, [](float x, float y, float scalar) { return float(ops::isinf_op{}(x)); });
    run(PointwiseOp::IsFinite, false, false, true, [](float x, float y, float scalar) { return float(ops::isfinite_op{}(x)); });
    run(PointwiseOp::LogicalNot, false, false, true, [](float x, float y, float scalar) { return float(ops::logical_not_op{}(x)); });
    run(PointwiseOp::MaximumScalar, false, true, false, [](float x, float y, float scalar) { return float(ops::maximum_op{}(x, scalar)); });
    run(PointwiseOp::MinimumScalar, false, true, false, [](float x, float y, float scalar) { return float(ops::minimum_op{}(x, scalar)); });
    run(PointwiseOp::EqualScalar, false, true, true, [](float x, float y, float scalar) { return float(ops::equal_op{}(x, scalar)); });
    run(PointwiseOp::NotEqualScalar, false, true, true, [](float x, float y, float scalar) { return float(ops::not_equal_op{}(x, scalar)); });
    run(PointwiseOp::LessScalar, false, true, true, [](float x, float y, float scalar) { return float(ops::less_op{}(x, scalar)); });
    run(PointwiseOp::LessEqualScalar, false, true, true, [](float x, float y, float scalar) { return float(ops::less_equal_op{}(x, scalar)); });
    run(PointwiseOp::GreaterScalar, false, true, true, [](float x, float y, float scalar) { return float(ops::greater_op{}(x, scalar)); });
    run(PointwiseOp::GreaterEqualScalar, false, true, true, [](float x, float y, float scalar) { return float(ops::greater_equal_op{}(x, scalar)); });
    run(PointwiseOp::ExpThenMulScalar, false, true, false, [](float x, float y, float scalar) { return float(ops::exp_op{}(x)*scalar); });
    run(PointwiseOp::MulScalarThenAbs, false, false, false, [](float x, float y, float scalar) { return float(ops::abs_op{}(x * scalar)); });
    run(PointwiseOp::MulScalarThenRelu, false, false, false, [](float x, float y, float scalar) { return float(ops::relu_op{}(x * scalar)); });
    require(gpu.has_nan(a, count, {}) && gpu.has_inf(a, count, {}), "HIP nonfinite reductions failed");
    require(std::isnan(gpu.min_scalar(a, count, {})) && std::isnan(gpu.max_scalar(a, count, {})), "HIP extrema lost NaN");
    auto large = gpu.allocate((1048595 + 8) * 4, 256, {});
    const auto large_base = large;
    large.byte_offset = 16;
    std::vector<float> numbers(1048595);
    double sum = 0;
    for (size_t i = 0; i < numbers.size(); ++i) {
        numbers[i] = float(int(i % 31) - 15) * 0.125f;
        sum += numbers[i];
    }
    upload(large, numbers);
    require(std::abs(gpu.sum_scalar(large, numbers.size(), {}) - sum) < 0.001, "HIP large sum/tail failed");
    require(std::abs(gpu.mean_scalar(large, numbers.size(), {}) - sum / numbers.size()) < 1e-7, "HIP large mean/tail failed");
    require(gpu.min_scalar(large, numbers.size(), {}) == -1.875f && gpu.max_scalar(large, numbers.size(), {}) == 1.875f, "HIP large extrema failed");
    require(!gpu.has_nan(large, numbers.size(), {}) && !gpu.has_inf(large, numbers.size(), {}), "HIP finite input flagged");
    require(gpu.sum_scalar(large, 0, {}) == 0 && std::isnan(gpu.mean_scalar(large, 0, {})), "HIP empty reduction identities failed");
    reject([&] { gpu.min_scalar(large, 0, {}); });
    const auto reduction_live = gpu.stats().allocated_bytes;
    (void)gpu.sum_scalar(large, numbers.size(), {});
    require(gpu.stats().allocated_bytes == reduction_live, "HIP reduction workspace was not reused");
    gpu.deallocate(large_base, {});
    // A chain exercises producer ordering without intermediate readbacks.
    for (size_t i = 0; i < count; ++i)
        x[i] = float(i) * 0.001f;
    upload(a, x);
    PointwiseProgram p{};
    p.in_dtype = p.out_dtype = DataType::Float32;
    p.op = PointwiseOp::Exp;
    gpu.unary(p, a, output, count, {});
    p.op = PointwiseOp::MulScalar;
    p.scalar.value.float_value = 0.5f;
    gpu.scalar(p, output, output, count, {});
    gpu.copy_device_to_host({output, host(actual.data()), count * 4});
    for (size_t i = 0; i < count; ++i)
        require(std::abs(actual[i] - std::exp(x[i]) * 0.5f) < 1e-6f, "HIP in-place dependent chain failed");
    for (size_t i = 0; i < count; ++i)
        y[i] = float(i % 13) * 0.02f;
    upload(b, y);
    tensor_ops::FusedPointwiseOpChain chain{};
    chain.num_ops = 3;
    chain.ops[0].kind = static_cast<uint8_t>(PointwiseOp::Exp);
    chain.ops[1].kind = static_cast<uint8_t>(PointwiseOp::MulScalar);
    chain.ops[1].scalar = 0.5f;
    chain.ops[2].kind = static_cast<uint8_t>(PointwiseOp::AddTensor);
    chain.ops[2].rhs = reinterpret_cast<const float*>(static_cast<const char*>(b.data) + b.byte_offset);
    const std::array rhs{b};
    gpu.fused_pointwise_chain(a, output, count, chain, rhs, {});
    gpu.copy_device_to_host({output, host(actual.data()), count * 4});
    for (size_t i = 0; i < count; ++i)
        require(std::abs(actual[i] - (std::exp(x[i]) * 0.5f + y[i])) < 2e-6f, "HIP fused Tensor rhs chain failed");
    for (unsigned i = 3; i < 16; ++i) {
        chain.ops[i].kind = static_cast<uint8_t>(PointwiseOp::AddScalar);
        chain.ops[i].scalar = 0.01f;
    }
    chain.num_ops = 16;
    gpu.fused_pointwise_chain(a, output, count, chain, rhs, {});
    gpu.copy_device_to_host({output, host(actual.data()), count * 4});
    for (size_t i = 0; i < count; ++i)
        require(std::abs(actual[i] - (std::exp(x[i]) * 0.5f + y[i] + 0.13f)) < 3e-6f, "HIP maximum chain length failed");
    reject([&] { gpu.fused_pointwise_chain(a, output, count, chain, {}, {}); });
    chain.num_ops = 17;
    reject([&] { gpu.fused_pointwise_chain(a, output, count, chain, rhs, {}); });
    chain.num_ops = 1;
    chain.ops[0].kind = static_cast<uint8_t>(PointwiseOp::EqualScalar);
    reject([&] { gpu.fused_pointwise_chain(a, output, count, chain, {}, {}); });
    StridedLayout layout{};
    layout.rank = 2;
    layout.dims[0] = 11;
    layout.dims[1] = 23;
    layout.strides[0] = 1;
    layout.strides[1] = 11;
    layout.element_count = 253;
    gpu.strided_copy(a, output, layout, {});
    gpu.copy_device_to_host({output, host(actual.data()), 253 * 4});
    for (size_t i = 0; i < 253; ++i)
        require(actual[i] == x[i / 23 + (i % 23) * 11], "HIP transposed contiguous copy failed");
    auto malformed = layout;
    malformed.strides[1] = SIZE_MAX;
    reject([&] { gpu.strided_copy(a, output, malformed, {}); });
    reject([&] { gpu.strided_copy(a, a, layout, {}); });
    gpu.memset({base_output, (count + 16) * 4, 0x5a, true});
    layout.dims[1] = 21;
    layout.strides[1] = 12;
    layout.element_count = 231;
    ScalarOperand fill{};
    fill.value.float_value = -0.25f;
    gpu.fill_strided(output, layout, fill, {});
    gpu.copy_device_to_host({output, host(actual.data()), count * 4});
    for (size_t i = 0; i < count; ++i) {
        const auto expected = (i < 251 && i % 12 < 11) ? std::bit_cast<uint32_t>(-0.25f) : 0x5a5a5a5au;
        require(std::bit_cast<uint32_t>(actual[i]) == expected, "HIP strided fill wrote into padding");
    }
    layout.strides[1] = 1;
    reject([&] { gpu.fill_strided(output, layout, fill, {}); });
    ScalarOperand minimum{}, maximum{};
    minimum.value.float_value = 0.05f;
    maximum.value.float_value = 0.15f;
    gpu.clamp_fused(a, output, minimum, maximum, count, {});
    gpu.copy_device_to_host({output, host(actual.data()), count * 4});
    for (size_t i = 0; i < count; ++i)
        require(actual[i] == std::clamp(x[i], 0.05f, 0.15f), "HIP clamp failed");
    auto foreign = a;
    foreign.backend = GpuBackend::Vulkan;
    reject([&] { gpu.scalar(p, foreign, output, count, {}); });
    auto unaligned = a;
    ++unaligned.byte_offset;
    reject([&] { gpu.scalar(p, unaligned, output, count, {}); });
    auto overlap = a;
    overlap.byte_offset += 4;
    reject([&] { gpu.scalar(p, a, overlap, count, {}); });
    reject([&] { gpu.scalar(p, a, output, SIZE_MAX, {}); });
    p.scalar.scalar_on_right = false;
    gpu.scalar(p, a, output, count, {});
    gpu.copy_device_to_host({output, host(actual.data()), count * 4});
    for(size_t i=0;i<count;++i)
        require(actual[i]==0.5f*x[i],"HIP scalar-left multiply differs from CPU");
    ExecContext cuda{};
    cuda.cuda_stream = reinterpret_cast<cudaStream_t>(uintptr_t(1));
    reject([&] { gpu.synchronize_stream(cuda); });
    reject([&] { gpu.shutdown(); });
    // Empty matrix multiplication is a supported no-op.
    std::vector<uint8_t> before_empty_gemm((count + 16) * 4), after_empty_gemm(before_empty_gemm.size());
    gpu.copy_device_to_host({base_output, host(before_empty_gemm.data()), before_empty_gemm.size()});
    gpu.sgemm(a, b, output, {}, {});
    gpu.copy_device_to_host({base_output, host(after_empty_gemm.data()), after_empty_gemm.size()});
    require(before_empty_gemm == after_empty_gemm, "Empty HIP GEMM changed output storage");
    gpu.copy_device_to_device({a, output, count * 4, true});
    gpu.copy_device_to_host({output, host(actual.data()), count * 4});
    require(actual == x, "HIP device copy failed");
    ScalarOperand start{}, step{};
    start.value.float_value = 1;
    step.value.float_value = 0.25f;
    gpu.load_arange(output, count, start, step, {});
    gpu.copy_device_to_host({output, host(actual.data()), count * 4});
    for (size_t i = 0; i < count; ++i)
        require(actual[i] == 1 + float(i) * 0.25f, "HIP arange failed");
    require(gpu.classify_pointer(static_cast<char*>(a.data) + a.byte_offset) == PointerClass::Device, "HIP pointer classification failed");
    for (const auto [outer, extent, inner] : std::array<std::array<size_t, 3>, 7>{{{7, 3, 4}, {2, 32, 3}, {3, 33, 2}, {1, 4096, 3}, {1, 4097, 3}, {1, 100003, 3}, {2, 263, 64}}}) {
        const size_t total = outer * extent * inner, outputs = outer * inner;
        auto source = gpu.allocate((total + 8) * 4, 256, {});
        auto target = gpu.allocate((outputs + 8) * 4, 256, {});
        auto source_view = source, target_view = target;
        source_view.byte_offset = target_view.byte_offset = 16;
        std::vector<float> values(total), result(outputs);
        for (size_t i = 0; i < total; ++i)
            values[i] = (int(i % 17) - 8) * 0.25f;
        gpu.copy_host_to_device({host(values.data()), source_view, total * 4});
        for (auto operation : {ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Min, ReduceOp::Max}) {
            ReduceProgram reduction{};
            reduction.op = operation;
            reduction.result_dtype = DataType::Float32;
            gpu.memset({target, (outputs + 8) * 4, 0x5a, true});
            gpu.strided_reduce(source_view, target_view, outer, extent, inner, reduction, {});
            gpu.copy_device_to_host({target_view, host(result.data()), outputs * 4});
            for (size_t o = 0; o < outputs; ++o) {
                double expected = operation == ReduceOp::Min ? INFINITY : operation == ReduceOp::Max ? -INFINITY
                                                                                                     : 0;
                for (size_t r = 0; r < extent; ++r) {
                    const float v = values[(o / inner * extent + r) * inner + o % inner];
                    if (operation == ReduceOp::Min)
                        expected = std::min(expected, double(v));
                    else if (operation == ReduceOp::Max)
                        expected = std::max(expected, double(v));
                    else
                        expected += v;
                }
                if (operation == ReduceOp::Mean)
                    expected /= extent;
                require(std::isfinite(result[o]) && std::abs(result[o] - expected) < 2e-5 * std::max(1., std::abs(expected)), "HIP axis reduction differs from double CPU oracle");
            }
            std::vector<uint8_t> guards((outputs + 8) * 4);
            gpu.copy_device_to_host({target, host(guards.data()), guards.size()});
            for (size_t i = 0; i < guards.size(); ++i)
                if (i < 16 || i >= 16 + outputs * 4)
                    require(guards[i] == 0x5a, "HIP reduction overwrote output guards");
            if (outer == 1) {
                gpu.column_reduce(source_view, target_view, extent, inner, reduction, {});
                std::vector<float> column(outputs);
                gpu.copy_device_to_host({target_view, host(column.data()), outputs * 4});
                require(result == column, "HIP column/strided reduction disagree");
            }
        }
        gpu.deallocate(source, {});
        gpu.deallocate(target, {});
    }
    {
        StridedLayout layout{};
        layout.rank = 3;
        layout.dims = {3, 5, 7};
        layout.strides = {35, 7, 1};
        layout.element_count = 105;
        ReduceProgram reduction{};
        reduction.op = ReduceOp::Sum;
        reduction.result_dtype = DataType::Float32;
        reduction.axis_count = 2;
        reduction.axes[0] = 0;
        reduction.axes[1] = 2;
        gpu.copy_host_to_device({host(x.data()), a, x.size() * 4});
        gpu.reduce(a, output, layout, reduction, {});
        std::array<float, 5> sums{};
        gpu.copy_device_to_host({output, host(sums.data()), sizeof(sums)});
        for (size_t j = 0; j < 5; ++j) {
            double expected = 0;
            for (size_t i = 0; i < 3; ++i)
                for (size_t k = 0; k < 7; ++k)
                    expected += x[i * 35 + j * 7 + k];
            require(std::abs(sums[j] - expected) < 1e-5, "HIP nonadjacent-axis reduction failed");
        }
        auto permuted = layout;
        permuted.dims = {7, 5, 3};
        permuted.strides = {1, 7, 35};
        gpu.reduce(a, output, permuted, reduction, {});
        std::array<float, 5> transposed{};
        gpu.copy_device_to_host({output, host(transposed.data()), sizeof(transposed)});
        for (size_t i = 0; i < 5; ++i)
            require(std::abs(transposed[i] - sums[i]) < 1e-5, "HIP transposed reduction failed");
        reduction.axes[1] = 0;
        reject([&] { gpu.reduce(a, output, layout, reduction, {}); });
        reduction.axes[1] = 3;
        reject([&] { gpu.reduce(a, output, layout, reduction, {}); });
        reduction.axis_count = 0;
        reduction.op = ReduceOp::Prod;
        reject([&] { gpu.reduce(a, output, layout, reduction, {}); });
        reduction.op = ReduceOp::Sum;
        reject([&] { gpu.reduce(a, a, layout, reduction, {}); });
        const float special_axes[] = {NAN, -0.f, INFINITY, 1.f, 0.f, -INFINITY, 2.f, -0.f, 3.f, 4.f, 0.f, 5.f};
        gpu.copy_host_to_device({host(const_cast<float*>(special_axes)), a, sizeof(special_axes)});
        for (auto op : {ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Min, ReduceOp::Max}) {
            reduction.op = op;
            gpu.strided_reduce(a, output, 1, 4, 3, reduction, {});
            std::array<float, 3> values{};
            gpu.copy_device_to_host({output, host(values.data()), sizeof(values)});
            for (size_t col = 0; col < 3; ++col) {
                float expected = op == ReduceOp::Min ? INFINITY : op == ReduceOp::Max ? -INFINITY
                                                                                      : 0;
                for (size_t row = 0; row < 4; ++row) {
                    const float v = special_axes[row * 3 + col];
                    if (op == ReduceOp::Min)
                        expected = ops::min_reduce_op{}(expected, v);
                    else if (op == ReduceOp::Max)
                        expected = ops::max_reduce_op{}(expected, v);
                    else
                        expected += v;
                }
                if (op == ReduceOp::Mean)
                    expected *= 0.25f;
                require(std::isnan(expected) ? std::isnan(values[col]) : values[col] == expected, "HIP axis reduction changed nonfinite values");
                if (expected == 0)
                    require(std::signbit(expected) == std::signbit(values[col]), "HIP axis extrema changed signed zero");
            }
        }
        reduction.op = ReduceOp::Mean;
        gpu.strided_reduce(a, output, 1, 0, 3, reduction, {});
        std::array<float, 3> empty{};
        gpu.copy_device_to_host({output, host(empty.data()), sizeof(empty)});
        require(std::all_of(empty.begin(), empty.end(), [](float v) { return std::isnan(v); }), "HIP empty axis mean is not NaN");
        reduction.op = ReduceOp::Sum;
        gpu.strided_reduce(a, output, 1, 0, 3, reduction, {});
        gpu.copy_device_to_host({output, host(empty.data()), sizeof(empty)});
        require(std::all_of(empty.begin(), empty.end(), [](float v) { return v == 0; }), "HIP empty axis sum is not zero");
        reduction.op = ReduceOp::Min;
        reject([&] { gpu.strided_reduce(a, output, 1, 0, 3, reduction, {}); });
        reject([&] { gpu.strided_reduce(a, output, SIZE_MAX, 2, 2, reduction, {}); });
    }
    for (size_t size : {size_t(0), size_t(1), size_t(263), size_t(1048595), size_t(16777219)}) {
        auto base = gpu.allocate((size + 8) * 4, 256, {});
        auto input = base;
        input.byte_offset = 16;
        std::vector<uint8_t> bytes(size);
        std::vector<float> floats(size);
        size_t expected_bytes = 0, expected_floats = 0;
        for (size_t i = 0; i < size; ++i) {
            bytes[i] = i % 3 == 0 ? 0 : i % 3 == 1 ? 1
                                                   : 255;
            if (size > 16777216)
                bytes[i] = 1;
            expected_bytes += bytes[i] != 0;
            const float special[] = {-0.f, 0.f, NAN, INFINITY, -INFINITY, 1.f, -1.f};
            floats[i] = special[i % 7];
            if (size > 16777216)
                floats[i] = 1.f;
            expected_floats += floats[i] != 0;
        }
        input.dtype = DataType::Bool;
        gpu.copy_host_to_device({host(bytes.data()), input, bytes.size()});
        require(gpu.count_nonzero_bool(input, size, {}) == expected_bytes, "HIP bool count differs from CPU");
        input.dtype = DataType::UInt8;
        require(gpu.count_nonzero_bool(input, size, {}) == expected_bytes, "HIP byte count differs from CPU");
        input.dtype = DataType::Float32;
        gpu.copy_host_to_device({host(floats.data()), input, floats.size() * 4});
        require(gpu.count_nonzero_float(input, size, {}) == expected_floats, "HIP float count differs from CPU");
        reject([&] { gpu.count_nonzero_bool(input, size, {}); });
        reject([&] { gpu.count_nonzero_float(input, SIZE_MAX, {}); });
        gpu.deallocate(base, {});
    }
    test_random(gpu);
    test_strided_writes(gpu);
    test_index_updates(gpu);
    test_index_fill(gpu);
    test_gathers(gpu);
    test_conversion_destinations<float>(gpu);
    test_conversion_destinations<detail::tensor_half_t>(gpu);
    test_conversion_destinations<int32_t>(gpu);
    test_conversion_destinations<int64_t>(gpu);
    test_conversion_destinations<uint8_t>(gpu);
    test_conversion_destinations<bool>(gpu);
    test_conversion_destinations<uint32_t>(gpu);
    {
        auto source = gpu.allocate((count + 8) * 8, 256, {}), target = gpu.allocate((count + 8) * 4, 256, {});
        auto in = source, out = target;
        in.dtype = DataType::Int64;
        out.dtype = DataType::Int32;
        in.byte_offset = out.byte_offset = 16;
        std::array<int64_t, count> indices{};
        for (size_t i = 0; i < count; ++i)
            indices[i] = int64_t(i % 17);
        gpu.copy_host_to_device({host(indices.data()), in, sizeof(indices)});
        gpu.index_cast(in, out, count, 17, {});
        std::array<int32_t, count> converted{};
        gpu.copy_device_to_host({out, host(converted.data()), sizeof(converted)});
        for (size_t i = 0; i < count; ++i)
            require(converted[i] == indices[i], "HIP index cast changed a valid index");
        for (auto invalid : {int64_t(-1), int64_t(17), INT64_MAX, INT64_MIN}) {
            indices[262] = invalid;
            gpu.copy_host_to_device({host(indices.data()), in, sizeof(indices)});
            gpu.memset({target, (count + 8) * 4, 0x5a, true});
            reject([&] { gpu.index_cast(in, out, count, 17, {}); });
            std::vector<uint8_t> all((count + 8) * 4);
            gpu.copy_device_to_host({target, host(all.data()), all.size()});
            require(std::all_of(all.begin(), all.end(), [](auto v) { return v == 0x5a; }), "Invalid index cast partially changed output");
        }
        indices.fill(0);
        gpu.copy_host_to_device({host(indices.data()), in, sizeof(indices)});
        gpu.index_cast(in, out, count, 1, {}); // Reusable after rejected indices.
        reject([&] { gpu.index_cast(in, out, count, 0, {}); });
        reject([&] { gpu.index_cast(in, out, count, SIZE_MAX, {}); });
        auto bad = in;
        bad.byte_offset = 17;
        reject([&] { gpu.index_cast(bad, out, count, 1, {}); });
        reject([&] { gpu.convert_type(in, out, SIZE_MAX, {}); });
        bad = out;
        bad.dtype = static_cast<DataType>(99);
        reject([&] { gpu.convert_type(in, bad, count, {}); });
        bad = in;
        bad.dtype = DataType::Float32;
        reject([&] { gpu.convert_type(in, bad, count, {}); });
        gpu.deallocate(source, {});
        gpu.deallocate(target, {});
    }
    require(gpu.stats().allocated_bytes == 3 * (count + 16) * 4 + 4097 * 4, "HIP accounting mismatch");
    gpu.deallocate(base_output, {});
    gpu.deallocate(base_b, {});
    gpu.deallocate(base_a, {});
    require(gpu.stats().allocated_bytes == 4097 * 4, "HIP Tensor user allocation leaked");
    require(gpu.stats().pool_reserved_current > gpu.stats().allocated_bytes, "HIP freed buffers were not cached");
    require(gpu.classify_pointer(base_a.data) == PointerClass::Unknown, "HIP cached pointer remained active");
    gpu.trim();
    require(gpu.stats().pool_reserved_current == 4097 * 4, "HIP trim retained cached buffers");
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        constexpr size_t length = 513;
        auto scan_values = gpu.allocate(length * 4, 256, {});
        auto sort_indices = gpu.allocate(length * 8, 256, {});
        scan_values.dtype = DataType::Float32;
        sort_indices.dtype = DataType::Int64;
        std::vector<float> values(length, 0.5f);
        gpu.copy_host_to_device({host(values.data()), scan_values, length * 4});
        StridedLayout scan_layout{};
        scan_layout.rank = 1;
        scan_layout.dims[0] = length;
        scan_layout.strides[0] = 1;
        scan_layout.element_count = length;
        gpu.cumsum(scan_values, scan_layout, 0, {});
        SortProgram sort_program{};
        sort_program.descending = true;
        gpu.sort_1d(scan_values, sort_indices, length, sort_program, {});
        gpu.copy_device_to_host({scan_values, host(values.data()), length * 4});
        std::vector<int64_t> indices(length);
        gpu.copy_device_to_host({sort_indices, host(indices.data()), length * 8});
        for (size_t i = 0; i < length; ++i)
            require(values[i] == 0.5f * float(length - i) && indices[i] == int64_t(length - i - 1),
                    "HIP scan/sort workspace reuse changed values or indices");
        require(gpu.sum_scalar(scan_values, length, {}) == float(length * (length + 1)) * 0.25f,
                "HIP reduction after scan/sort changed");
        const auto memory = gpu.stats();
        require(memory.allocated_bytes > length * 12 + 4097 * 4 && memory.pool_used_high >= memory.allocated_bytes,
                "HIP scan/sort scratch or peak was not accounted");
        gpu.deallocate(scan_values, {});
        gpu.deallocate(sort_indices, {});
        gpu.trim_if_reserved_unused_exceeds(0);
        require(gpu.stats().pool_reserved_current == 4097 * 4 && gpu.stats().allocated_bytes == 4097 * 4,
                "HIP trim did not release algorithm scratch or changed reduction retention");
    }
    // Reuse must respect pending GPU consumers without a host fence at release.
    auto producer = gpu.allocate(1024 * 1024, 256, {});
    auto consumer = gpu.allocate(1024 * 1024, 256, {});
    ScalarOperand seven{};
    seven.value.float_value = 7;
    gpu.load_fill(producer, 262144, seven, {});
    gpu.copy_device_to_device({producer, consumer, 1024 * 1024, true});
    gpu.deallocate(producer, {});
    auto reused = gpu.allocate(1024 * 1024, 256, {});
    require(reused.data == producer.data, "HIP exact-size cache did not reuse storage");
    seven.value.float_value = 9;
    gpu.load_fill(reused, 262144, seven, {});
    std::vector<float> pending(262144);
    gpu.copy_device_to_host({consumer, host(pending.data()), pending.size() * 4});
    require(std::all_of(pending.begin(), pending.end(), [](float v) { return v == 7; }), "HIP reuse raced a pending consumer");
    gpu.deallocate(reused, {});
    gpu.trim();
    gpu.copy_device_to_host({consumer, host(pending.data()), pending.size() * 4});
    require(pending.front() == 7 && pending.back() == 7, "HIP trim changed a live buffer");
    gpu.deallocate(consumer, {});
    gpu.trim();
    {
        HipBackendOps bounded(std::filesystem::absolute(argv[1]), 0, 1024);
        auto one = bounded.allocate(512, 256, {});
        auto two = bounded.allocate(512, 256, {});
        auto three = bounded.allocate(512, 256, {});
        bounded.deallocate(one, {});
        bounded.deallocate(two, {});
        bounded.deallocate(three, {});
        require(bounded.stats().allocated_bytes == 0 && bounded.stats().pool_reserved_current == 1024, "HIP cache exceeded configured budget");
        ExecContext direct{};
        direct.allocation_class = AllocationClass::Direct;
        auto uncached = bounded.allocate(512, 256, direct);
        require((uncached.flags & STORAGE_REF_DIRECT_ALLOCATION) != 0, "HIP direct allocation flag missing");
        bounded.deallocate(uncached, {});
        require(bounded.stats().pool_reserved_current == 1024, "HIP direct allocation was cached");
        bounded.trim_if_reserved_unused_exceeds(1024);
        require(bounded.stats().pool_reserved_current == 1024, "HIP cache threshold changed");
        bounded.trim_if_reserved_unused_exceeds(1023);
        require(bounded.stats().pool_reserved_current == 0, "HIP bounded cache did not trim");
        bounded.shutdown();
    }
    gpu.shutdown();
    gpu.shutdown();
    require(gpu.stats().allocated_bytes == 0, "HIP Tensor workspace leaked after shutdown");
    std::printf("PASS real GpuBackendOps HIP adapter: %u pointwise operations, offset/predicate guards, chain, invalid contracts; max relative %.9g\n", operations, worst);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "HIP CORE TEST FAILED: %s\n", error.what());
    return 1;
}
