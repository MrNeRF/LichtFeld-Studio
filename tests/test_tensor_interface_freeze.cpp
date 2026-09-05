/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/pinned_memory_allocator.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_debug.hpp"
#include "core/tensor_label.hpp"
#include "core/tensor_serialization_sink.hpp"
#include "core/tensor_trace.hpp"

#include <array>
#include <cstdio>
#include <istream>
#include <memory>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

    using namespace lfs::core;

    // The category counts are frozen to lane A section 2.1. Constructors,
    // destructors, deleted functions, internal orchestration, and macros are not
    // callables under that report's counting rule.
    // Creation/factory: 33
    // Shape/view/movement: 45
    // Pointwise unary: 42
    // Pointwise binary/broadcast: 51
    // Reductions/scans: 54
    // Matrix: 5
    // Indexing/masking: 34
    // Random: 18
    // NN: 19
    // Serialization: 8
    // Sync/stream/event: 19
    // Memory/allocator/pool stats: 45
    // Debug/trace/label: 61
    // Lazy/fused expression: 57
    // Row proxy/accessor: 38
    constexpr std::array<size_t, 15> kCategoryCounts{
        33, 45, 42, 51, 54, 5, 34, 18, 19, 8, 19, 45, 61, 57, 38};
    constexpr std::array<unsigned char, 529> kFrozenCallableSlots{};
    static_assert(kFrozenCallableSlots.size() == 529);
    static_assert([] {
        size_t total = 0;
        for (size_t count : kCategoryCounts)
            total += count;
        return total;
    }() == kFrozenCallableSlots.size());

    using T = Tensor;
    using S = TensorShape;
    using R = RankedDims;
    using RP = TensorRowProxy;

    // Exact signatures for non-overloaded callables. A requires expression only
    // proves a call is well-formed; it lets a changed return type, a widened
    // parameter or an added defaulted parameter through. decltype does not.
#define LFS_FREEZE(member, ...) \
    static_assert(std::is_same_v<decltype(&member), __VA_ARGS__>, "signature moved: " #member)
    LFS_FREEZE(T::exp, T (T::*)() const);
    LFS_FREEZE(T::log, T (T::*)() const);
    LFS_FREEZE(T::sqrt, T (T::*)() const);
    LFS_FREEZE(T::neg, T (T::*)() const);
    LFS_FREEZE(T::abs, T (T::*)() const);
    LFS_FREEZE(T::sigmoid, T (T::*)() const);
    LFS_FREEZE(T::tanh, T (T::*)() const);
    LFS_FREEZE(T::relu, T (T::*)() const);
    LFS_FREEZE(T::contiguous, T (T::*)() const);
    LFS_FREEZE(T::clone, T (T::*)() const);
    LFS_FREEZE(T::cpu, T (T::*)() const);
    LFS_FREEZE(T::count_nonzero, size_t (T::*)() const);
    LFS_FREEZE(T::has_nan, bool (T::*)() const);
    LFS_FREEZE(T::has_inf, bool (T::*)() const);
    LFS_FREEZE(T::sum_scalar, float (T::*)() const);
    LFS_FREEZE(T::mean_scalar, float (T::*)() const);
    LFS_FREEZE(T::max_scalar, float (T::*)() const);
    LFS_FREEZE(T::min_scalar, float (T::*)() const);
    LFS_FREEZE(T::to_vector, std::vector<float> (T::*)() const);
    LFS_FREEZE(T::to_vector_int, std::vector<int> (T::*)() const);
    LFS_FREEZE(T::to_vector_bool, std::vector<bool> (T::*)() const);
    LFS_FREEZE(T::stream, cudaStream_t (T::*)() const);
    LFS_FREEZE(T::set_stream, void (T::*)(cudaStream_t));
    LFS_FREEZE(T::record_stream, void (T::*)(cudaStream_t) const);
    LFS_FREEZE(T::sync_to_stream, void (T::*)(cudaStream_t) const);
    LFS_FREEZE(T::reserve, void (T::*)(size_t));
    LFS_FREEZE(T::numel, size_t (T::*)() const);
    LFS_FREEZE(T::ndim, size_t (T::*)() const);
    LFS_FREEZE(T::is_contiguous, bool (T::*)() const);
    LFS_FREEZE(T::is_valid, bool (T::*)() const);
    LFS_FREEZE(T::device, Device (T::*)() const);
    LFS_FREEZE(T::dtype, DataType (T::*)() const);
    LFS_FREEZE(T::bytes, size_t (T::*)() const);
    LFS_FREEZE(T::storage_offset, size_t (T::*)() const);
    // Selector exports: the additive public header must keep these shapes, and
    // taking their address ODR-uses them so the Windows import library is linked.
    LFS_FREEZE(default_gpu_backend, GpuBackend (*)());
    LFS_FREEZE(set_default_gpu_backend, lfs::Status (*)(GpuBackend));
    LFS_FREEZE(gpu_backend_available, bool (*)(GpuBackend));
    LFS_FREEZE(gpu_backend_memory_info, MemoryInfo (*)(GpuBackend));
    LFS_FREEZE(shutdown_gpu_backend, lfs::Status (*)(GpuBackend));
    LFS_FREEZE(gpu_backend_of, std::optional<GpuBackend> (*)(const T&));
#undef LFS_FREEZE
    constexpr size_t kExactSignatureCount = 40;
    [[maybe_unused]] constexpr auto kSelectorAnchors = std::tuple{
        &default_gpu_backend, &set_default_gpu_backend, &gpu_backend_available,
        &gpu_backend_memory_info, &shutdown_gpu_backend, &gpu_backend_of};
    [[gnu::used]] const void* const kSelectorAnchorAddress = &kSelectorAnchors;

    // Non-template overload sets use exact function-pointer types. Template
    // callables are checked below in requires expressions, which keeps them in an
    // unevaluated context while freezing their parameter and result contracts.
    [[maybe_unused]] constexpr auto kFactoryOverloads = std::tuple{
        static_cast<T (*)(float)>(&T::arange),
        static_cast<T (*)(float, float, float)>(&T::arange),
        static_cast<T (*)(size_t, Device)>(&T::eye),
        static_cast<T (*)(size_t, size_t, Device)>(&T::eye),
        static_cast<T (*)(const std::vector<float>&, S, Device)>(&T::from_vector),
        static_cast<T (*)(const std::vector<int>&, S, Device)>(&T::from_vector),
        static_cast<T (*)(const std::vector<bool>&, S, Device)>(&T::from_vector),
        static_cast<T (*)(std::initializer_list<float>, S, Device)>(&T::from_vector),
        static_cast<T (*)(std::initializer_list<int>, S, Device)>(&T::from_vector),
        static_cast<T (*)(std::initializer_list<bool>, S, Device)>(&T::from_vector),
        static_cast<T (*)(const T&)>(&T::ones_like),
        static_cast<T (*)(const T&, DataType)>(&T::ones_like)};

    [[maybe_unused]] constexpr auto kMovementOverloads = std::tuple{
        static_cast<T (T::*)(Device, cudaStream_t) const>(&T::to),
        static_cast<T (T::*)(DataType) const>(&T::to),
        static_cast<T (T::*)(std::span<const int>) const>(&T::reshape),
        static_cast<T (T::*)(std::initializer_list<int>) const>(&T::reshape),
        static_cast<T (T::*)(S) const>(&T::reshape),
        static_cast<T (T::*)(std::span<const int>) const>(&T::view),
        static_cast<T (T::*)(std::initializer_list<int>) const>(&T::view),
        static_cast<T (T::*)(S) const>(&T::view),
        static_cast<T (T::*)() const>(&T::squeeze),
        static_cast<T (T::*)(std::optional<int>) const>(&T::squeeze),
        static_cast<T (T::*)(int) const>(&T::squeeze),
        static_cast<T (T::*)(std::span<const int>) const>(&T::expand),
        static_cast<T (T::*)(std::initializer_list<int>) const>(&T::expand),
        static_cast<T (T::*)(const S&) const>(&T::expand),
        static_cast<T (T::*)(std::span<const int>) const>(&T::permute),
        static_cast<T (T::*)(std::initializer_list<int>) const>(&T::permute),
        static_cast<T (T::*)(std::span<const std::pair<int, int>>) const>(&T::slice),
        static_cast<T (T::*)(std::initializer_list<std::pair<int, int>>) const>(&T::slice),
        static_cast<T (T::*)(size_t, size_t, size_t) const>(&T::slice)};

    [[maybe_unused]] constexpr auto kReductionOverloads = std::tuple{
        static_cast<T (T::*)() const>(&T::sum),
        static_cast<T (T::*)(std::span<const int>, bool) const>(&T::sum),
        static_cast<T (T::*)(std::initializer_list<int>, bool) const>(&T::sum),
        static_cast<T (T::*)(int, bool) const>(&T::sum),
        static_cast<T (T::*)() const>(&T::mean),
        static_cast<T (T::*)(std::span<const int>, bool) const>(&T::mean),
        static_cast<T (T::*)(std::initializer_list<int>, bool) const>(&T::mean),
        static_cast<T (T::*)(int, bool) const>(&T::mean),
        static_cast<T (T::*)() const>(&T::max),
        static_cast<T (T::*)(std::span<const int>, bool) const>(&T::max),
        static_cast<T (T::*)(std::initializer_list<int>, bool) const>(&T::max),
        static_cast<T (T::*)(int, bool) const>(&T::max),
        static_cast<T (T::*)() const>(&T::min),
        static_cast<T (T::*)(std::span<const int>, bool) const>(&T::min),
        static_cast<T (T::*)(std::initializer_list<int>, bool) const>(&T::min),
        static_cast<T (T::*)(int, bool) const>(&T::min),
        static_cast<T (T::*)() const>(&T::prod),
        static_cast<T (T::*)(std::span<const int>, bool) const>(&T::prod),
        static_cast<T (T::*)(std::initializer_list<int>, bool) const>(&T::prod),
        static_cast<T (T::*)(int, bool) const>(&T::prod),
        static_cast<T (T::*)() const>(&T::std),
        static_cast<T (T::*)(std::span<const int>, bool, bool) const>(&T::std),
        static_cast<T (T::*)(std::initializer_list<int>, bool, bool) const>(&T::std),
        static_cast<T (T::*)(int, bool, bool) const>(&T::std),
        static_cast<T (T::*)() const>(&T::var),
        static_cast<T (T::*)(std::span<const int>, bool, bool) const>(&T::var),
        static_cast<T (T::*)(std::initializer_list<int>, bool, bool) const>(&T::var),
        static_cast<T (T::*)(int, bool, bool) const>(&T::var),
        static_cast<float (T::*)() const>(&T::item),
        static_cast<float (T::*)(float) const>(&T::norm),
        static_cast<T (T::*)(float, std::span<const int>, bool) const>(&T::norm),
        static_cast<T (T::*)(float, std::initializer_list<int>, bool) const>(&T::norm),
        static_cast<T (T::*)(float, int, bool) const>(&T::norm)};

    [[maybe_unused]] constexpr auto kIndexOverloads = std::tuple{
        static_cast<RP (T::*)(size_t)>(&T::operator[]),
        static_cast<const RP (T::*)(size_t) const>(&T::operator[]),
        static_cast<TensorIndexer (T::*)(const T&)>(&T::operator[]),
        static_cast<TensorIndexer (T::*)(const std::vector<T>&)>(&T::operator[]),
        static_cast<MaskedTensorProxy (T::*)(const T&) const>(&T::operator[]),
        static_cast<T (T::*)(int, const T&) const>(&T::index_select),
        static_cast<T (T::*)(int, const T&, BoundaryMode) const>(&T::index_select),
        static_cast<T (T::*)(int, const T&) const>(&T::gather),
        static_cast<T (T::*)(int, const T&, BoundaryMode) const>(&T::gather),
        static_cast<T& (T::*)(int, const T&, const T&, ScatterMode)>(&T::scatter_),
        static_cast<T& (T::*)(int, const T&, float, ScatterMode)>(&T::scatter_),
        static_cast<T& (T::*)(const T&, const T&)>(&T::index_put_),
        static_cast<T& (T::*)(const std::vector<T>&, const T&)>(&T::index_put_)};

    template <typename X>
    concept CreationFactorySurface = requires(const X& ct, S shape, Device device, DataType dtype,
                                              std::shared_ptr<void> owner, void* data,
                                              cudaStream_t stream) {
        { X::load(LoadOp::Const, LoadArgs{}) } -> std::same_as<X>;
        X::empty(shape, device, dtype, true);
        X::empty_pageable_host(shape, dtype);
        X::empty_unpinned(shape, dtype);
        X::zeros(shape, device, dtype);
        X::zeros_direct(shape, 8, device, dtype);
        X::ones(shape, device, dtype);
        X::full(shape, 1.0f, device, dtype);
        X::full_bool(shape, true, device);
        X::zeros_bool(shape, device);
        X::ones_bool(shape, device);
        X::arange(3.0f);
        X::arange(1.0f, 3.0f, 0.5f);
        X::linspace(0.0f, 1.0f, 3, device);
        X::eye(3, device);
        X::eye(3, 4, device);
        X::diag(ct);
        X::from_blob(data, shape, device, dtype, stream);
        X::from_external_owner(data, shape, device, dtype, owner);
        X::from_external_owner(data, shape, device, dtype, owner, 9);
        X::from_external_owner(data, shape, device, dtype, owner, 9, stream);
        X::from_external_owner(data, shape, device, dtype, owner, 9, stream, std::string{});
        X::zeros_like(ct);
        X::ones_like(ct);
        X::ones_like(ct, dtype);
        X::empty_like(ct);
        X::full_like(ct, 1.0f);
        X::cat(std::vector<X>{}, 0);
        X::stack(std::vector<X>{}, 0);
        X::where(ct, ct, ct);
    };

    template <typename X>
    concept MovementSurface = requires(X& t, const X& ct, MovementArgs args, S shape,
                                       std::span<const int> dims,
                                       std::span<const std::pair<int, int>> ranges,
                                       cudaStream_t stream) {
        ct.movement(MovementOp::Reshape, args);
        ct.clone();
        ct.contiguous();
        ct.to(Device::CPU, stream);
        ct.to_pageable_host(stream);
        ct.to(DataType::Float32);
        ct.cpu();
        ct.cuda();
        ct.reshape(dims);
        ct.reshape(shape);
        ct.view(dims);
        ct.view(shape);
        ct.squeeze();
        ct.squeeze(std::optional<int>{});
        ct.squeeze(0);
        ct.unsqueeze(0);
        ct.expand(dims);
        ct.expand(shape);
        ct.flatten(0, -1);
        ct.permute(dims);
        ct.transpose(0, 1);
        ct.t();
        ct.slice(ranges);
        ct.slice(0, 0, 1);
        ct.cat(ct, 0);
        ct.broadcast_to(shape);
        ct.can_broadcast_to(shape);
        ct.broadcast_shape(shape);
        t.reserve(8);
        ct.try_reshape(shape);
        X::split_batch(ct, 4);
    };

    template <typename Dims, typename Shape>
    concept ShapeValueSurface = requires(Dims& dims, const Dims& const_dims, Shape& shape,
                                         const Shape& const_shape, std::span<const size_t> span,
                                         std::vector<size_t> vector) {
        const_dims.size();
        const_dims.empty();
        dims[0];
        const_dims[0];
        dims.data();
        const_dims.data();
        dims.begin();
        dims.end();
        const_dims.begin();
        const_dims.end();
        const_dims.cbegin();
        const_dims.cend();
        dims.clear();
        dims.assign(span);
        dims.assign(vector);
        dims.assign({1, 2});
        const_dims == const_dims;
        const_dims != const_dims;
        const_dims == vector;
        const_dims != vector;
        const_shape.rank();
        const_shape[0];
        const_shape.elements();
        const_shape.dims();
        const_shape.strides();
        const_shape == const_shape;
        const_shape != const_shape;
        const_shape.str();
    };

    template <typename X>
    concept UnarySurface = requires(X& t, const X& ct) {
        ct.neg();
        ct.abs();
        ct.sign();
        ct.reciprocal();
        ct.exp();
        ct.exp2();
        ct.log();
        ct.log2();
        ct.log10();
        ct.log1p();
        ct.sqrt();
        ct.rsqrt();
        ct.square();
        ct.sin();
        ct.cos();
        ct.tan();
        ct.asin();
        ct.acos();
        ct.atan();
        ct.sinh();
        ct.cosh();
        ct.tanh();
        ct.sigmoid();
        ct.relu();
        ct.gelu();
        ct.swish();
        ct.floor();
        ct.ceil();
        ct.round();
        ct.trunc();
        ct.isnan();
        ct.isinf();
        ct.isfinite();
        ct.logical_not();
        ct.normalize(-1, 1e-12f);
        ct.logit(1e-7f);
        ct.clamp(-1.0f, 1.0f);
        ct.clamp_min(-1.0f);
        ct.clamp_max(1.0f);
        t.clamp_(-1.0f, 1.0f);
        t.clamp_min_(-1.0f);
        t.clamp_max_(1.0f);
        -ct;
        ~ct;
        !ct;
    };

    template <typename X>
    concept BinarySurface = requires(X& t, const X& ct) {
        ct.add(ct);
        ct.sub(ct);
        ct.mul(ct);
        ct.div(ct);
        ct.pow(ct);
        ct.mod(ct);
        ct.maximum(ct);
        ct.minimum(ct);
        ct.add(1.0f);
        ct.sub(1);
        ct.mul(1.0f);
        ct.div(1);
        ct.pow(2);
        ct.mod(2);
        ct.maximum(1.0f);
        ct.minimum(1.0f);
        ct.eq(ct);
        ct.ne(ct);
        ct.lt(ct);
        ct.le(ct);
        ct.gt(ct);
        ct.ge(ct);
        ct.eq(1.0f);
        ct.ne(1);
        ct.lt(1.0f);
        ct.le(1);
        ct.gt(1.0f);
        ct.ge(1);
        ct.logical_and(ct);
        ct.logical_or(ct);
        ct.logical_xor(ct);
        t.and_live_(ct);
        ct.where(ct, ct);
        t.add_(ct);
        t.add_(1.0f);
        t.sub_(ct);
        t.sub_(1.0f);
        t.mul_(ct);
        t.mul_(1.0f);
        t.div_(ct);
        t.div_(1.0f);
        ct + ct;
        ct - ct;
        ct* ct;
        ct / ct;
        ct % ct;
        ct == ct;
        ct != ct;
        ct < ct;
        ct <= ct;
        ct > ct;
        ct >= ct;
        ct&& ct;
        ct || ct;
        ct | ct;
    };

    template <typename X>
    concept ReductionSurface = requires(const X& ct, std::span<const int> dims) {
        ct.reduce(ReduceOp::Sum);
        ct.reduce(ReduceOp::Sum, ReduceArgs{});
        ct.sum();
        ct.sum(dims, true);
        ct.sum(0, true);
        ct.mean();
        ct.mean(dims, true);
        ct.mean(0, true);
        ct.max();
        ct.max(dims, true);
        ct.max(0, true);
        ct.min();
        ct.min(dims, true);
        ct.min(0, true);
        ct.prod();
        ct.prod(dims, true);
        ct.prod(0, true);
        ct.any();
        ct.any(dims, true);
        ct.any(0, true);
        ct.all();
        ct.all(dims, true);
        ct.all(0, true);
        ct.std();
        ct.std(dims, true, false);
        ct.std(0, true, false);
        ct.var();
        ct.var(dims, true, false);
        ct.var(0, true, false);
        ct.argmax();
        ct.argmax(dims, true);
        ct.argmin();
        ct.argmin(dims, true);
        ct.cumsum(0);
        ct.sum_scalar();
        ct.mean_scalar();
        ct.min_scalar();
        ct.max_scalar();
        ct.std_scalar(false);
        ct.var_scalar(false);
        ct.minmax();
        ct.norm(2.0f);
        ct.norm(2.0f, dims, true);
        ct.norm(2.0f, 0, true);
        ct.item();
        ct.template item<float>();
        ct.count_nonzero();
        ct.min_with_indices(0, true);
        ct.max_with_indices(0, true);
        ct.sort(0, false);
        ct.any_scalar();
    };

    template <typename X>
    concept MatrixSurface = requires(const X& ct) {
        ct.mm(ct);
        ct.bmm(ct);
        ct.matmul(ct);
        ct.dot(ct);
        ct.cdist(ct, 2.0f);
    };

    template <typename X>
    concept IndexSurface = requires(X& t, const X& ct, std::vector<X> tensors) {
        t[0];
        ct[0];
        t[ct];
        t[tensors];
        ct[ct];
        t.set_bool({0}, true);
        ct.get_bool({0});
        ct.masked_select(ct);
        t.masked_fill_(ct, 1.0f);
        ct.masked_fill(ct, 1.0f);
        ct.index_select(0, ct);
        ct.gather(0, ct);
        ct.take(ct);
        t.append_gather(ct);
        t.append_zeros(1);
        ct.gather_lazy(ct);
        ct.nonzero();
        ct.nonzero_split();
        t.scatter_(0, ct, ct, ScatterMode::None);
        t.scatter_(0, ct, 1.0f, ScatterMode::None);
        t.index_fill_(0, ct, 1.0f);
        t.index_copy_(0, ct, ct);
        t.index_add_(0, ct, ct);
        t.index_put_(ct, ct);
        t.index_put_(tensors, ct);
        ct.index_select(0, ct, BoundaryMode::Clamp);
        ct.index_select_into(t, 0, ct, BoundaryMode::Clamp);
        ct.gather(0, ct, BoundaryMode::Clamp);
        t.at({0});
        ct.at({0});
    };

    template <typename X>
    concept RandomSurface = requires(X& t, const X& ct, S shape, const RandomGenerator& const_generator,
                                     float* output, cudaStream_t stream) {
        X::rand(shape);
        X::randn(shape);
        X::uniform(shape);
        X::normal(shape);
        X::randint(shape, 0, 4);
        X::bernoulli(shape);
        X::multinomial(ct, 2, true);
        X::rand_like(ct);
        X::randn_like(ct);
        X::manual_seed(1);
        t.uniform_();
        t.normal_();
        RandomGenerator::instance();
        RandomGenerator::instance().manual_seed(1);
        RandomGenerator::instance().get_seed();
        RandomGenerator::instance().get_generator(Device::CPU);
        RandomGenerator::instance().get_next_cuda_seed();
        RandomGenerator::instance().get_impl();
        const_generator.get_impl();
        RandomGenerator::instance().generate_cuda_normal(output, 2, 0.0f, 1.0f, stream);
    };

    template <typename X>
    concept NnSurface = requires(X& t, const X& ct) {
        ct.sigmoid();
        ct.relu();
        ct.gelu();
        ct.swish();
        ct.normalize();
        ct.logit();
        ct.conv1x1(ct);
        ct.conv1x1(ct, ct);
        ct.max_pool2d(2, 2, 0);
        ct.adaptive_avg_pool2d(2, 2);
        ct.linear(ct);
        ct.linear(ct, ct);
        ct.conv1x1_bias_out(ct, ct, t);
        ct.conv1x1_bias_relu_out(ct, ct, t);
        ct.relu_out(t);
        ct.max_pool2d_out(2, 2, 0, t);
        ct.adaptive_avg_pool2d_out(2, 2, t);
        ct.linear_bias_relu_out(ct, ct, t);
        ct.linear_out(ct, ct, t);
    };

    template <typename X>
    concept SerializationSurface = requires(std::ostream& os, std::istream& is, const X& ct, X& t,
                                            TensorSerializationDescriptor descriptor,
                                            TensorSerializationSink& sink, std::string filename) {
        os << ct;
        is >> t;
        save_tensor(ct, filename);
        load_tensor(filename);
        descriptor.payload_bytes();
        current_tensor_serialization_sink();
        serialize_tensor_with_descriptor(os, ct, descriptor, nullptr);
        sink.write_tensor_payload(os, ct, nullptr, descriptor);
    };

    template <typename X>
    concept SyncSurface = requires(X& t, const X& ct, cudaStream_t stream) {
        ct.stream();
        t.set_stream(stream);
        ct.record_stream(stream);
        ct.sync_to_stream(stream);
        getCurrentCUDAStream();
        setCurrentCUDAStream(stream);
        waitForCUDAStream(stream, stream);
        prepare_inputs_for_stream({&ct}, stream);
        CudaEventPool::instance();
        CudaEventPool::instance().acquire();
        CudaEventPool::instance().release(nullptr);
        CudaEventPool::instance().shutdown();
        CudaEventPool::instance().stats();
        CudaEventPool::instance().pooled_count();
        set_cuda_event_acquire_failure_for_testing(false);
        bridgeStreams(stream, stream);
        retire_stream(stream);
        unretire_stream(stream);
        is_stream_retired(stream);
    };

    template <typename X>
    concept MemorySurface = requires(X& t, const X& ct, cudaStream_t stream, void* pointer) {
        t.template ptr<float>();
        ct.template ptr<float>();
        t.data_ptr();
        ct.data_ptr();
        t.storage_ptr();
        ct.storage_ptr();
        ct.shape();
        ct.device();
        ct.dtype();
        ct.owns_memory();
        ct.is_view();
        ct.is_external_storage();
        ct.is_empty();
        ct.is_valid();
        ct.numel();
        ct.bytes();
        ct.ndim();
        ct.size(0);
        ct.capacity();
        ct.logical_size();
        ct.external_storage_kind();
        ct.external_storage_owner();
        ct.is_contiguous();
        ct.strides();
        ct.stride(0);
        ct.storage_offset();
        ct.has_zero_stride();
        X::trim_memory_pool();
        X::trim_memory_pool_if_reserved_unused_exceeds(0);
        X::trim_device_memory_pool();
        X::shutdown_memory_pool();
        X::set_memory_pool_iteration(0);
        X::storage_memory_summary();
        X::log_storage_memory();
        X::log_storage_memory(std::string_view{});
        PinnedMemoryAllocator::instance();
        PinnedMemoryAllocator::instance().allocate(1);
        PinnedMemoryAllocator::instance().deallocate(pointer, stream);
        PinnedMemoryAllocator::instance().record_stream(pointer, stream);
        PinnedMemoryAllocator::instance().is_cuda_host_allocation(pointer);
        PinnedMemoryAllocator::instance().release_stream(stream);
        PinnedMemoryAllocator::instance().empty_cache();
        PinnedMemoryAllocator::instance().prewarm();
        PinnedMemoryAllocator::instance().shutdown();
        PinnedMemoryAllocator::instance().get_stats();
        PinnedMemoryAllocator::instance().reset_stats();
        PinnedMemoryAllocator::instance().set_enabled(true);
        PinnedMemoryAllocator::instance().is_enabled();
        PinnedMemoryAllocator::instance().set_force_fallback_for_testing(false);
        PinnedMemoryAllocator::instance().set_cache_limit_for_testing(1);
        PinnedMemoryAllocator::instance().cache_limit_bytes();
        MemoryInfo::cuda();
        MemoryInfo::cpu();
    };

    template <typename X>
    concept DebugSurface = requires(X& t, const X& ct, std::string text) {
        ct.has_lazy_expr();
        ct.is_deferred();
        ct.lazy_expr_id();
        ct.lazy_expr_info();
        ct.debug_id();
        ct.is_tracked();
        t.set_tracked();
        t.track();
        t.untrack();
        ct.name();
        t.set_name(text);
        t.set_exportable_provenance({}, 0, 0);
        ct.has_exportable_provenance();
        ct.exportable_control();
        ct.exportable_region();
        ct.exportable_bound_generation();
        t.assert_shape(S{});
        t.assert_shape(S{}, text);
        t.assert_device(Device::CPU);
        t.assert_dtype(DataType::Float32);
        t.assert_finite();
        ct.has_nan();
        ct.has_inf();
        ct.all_close(ct);
        ct.str();
        ct.to_vector();
        ct.to_vector_uint8();
        ct.to_vector_int64();
        ct.to_vector_int();
        ct.to_vector_bool();
        ct.debug_values();
        ct.print_formatted();
        ct.print_formatted(text, 4);
        ct.options();
        debug::TensorValidation{}.is_valid();
        debug::TensorValidation{}.to_string();
        debug::validate_tensor_cpu(ct);
        debug::validate_tensor_gpu(ct);
        debug::validate_tensor(ct);
        debug::log_tensor_validation(ct, "", "", 0);
        debug::TensorDiff{}.is_close();
        debug::TensorDiff{}.to_string();
        debug::diff_tensors(ct, ct);
        debug::log_tensor_diff(ct, ct, "");
        debug::TensorStats{}.to_string();
        debug::get_tensor_stats(ct);
        debug::log_tensor_info(ct, "");
        debug::TensorOpTracer::instance();
        debug::TensorOpTracer::instance().set_enabled(true);
        debug::TensorOpTracer::instance().is_enabled();
        debug::TensorOpTracer::instance().should_trace(ct);
        debug::TensorOpTracer::instance().should_trace(ct, ct);
        debug::TensorOpTracer::instance().push("", ct, SourceSite{"", 0, ""});
        debug::TensorOpTracer::instance().push("", ct, ct, SourceSite{"", 0, ""});
        debug::TensorOpTracer::instance().push("", S{}, SourceSite{"", 0, ""});
        debug::TensorOpTracer::instance().push("", S{}, S{}, SourceSite{"", 0, ""});
        debug::TensorOpTracer::instance().pop();
        debug::TensorOpTracer::instance().pop(S{});
        debug::TensorOpTracer::instance().print_stack();
        debug::TensorOpTracer::instance().print_history();
        debug::TensorOpTracer::instance().clear_history();
        debug::TensorOpTracer::instance().get_history();
        debug::OpTraceGuard("", ct, SourceSite{"", 0, ""}).set_output(S{});
        MemoryInfo{}.log();
    };

    template <typename E>
    concept ExprBaseSurface = requires(E& expression, const E& const_expression) {
        expression.derived();
        const_expression.derived();
        const_expression.eval();
        static_cast<T>(const_expression);
        const_expression.shape();
        const_expression.device();
        const_expression.dtype();
        const_expression.stream_hint();
        const_expression.snapshot();
    };

    template <typename X>
    concept LazySurface = requires(X& t, const X& ct, lfs::core::ops::abs_op operation) {
        X::enable_profiling(true);
        X::lazy_telemetry_snapshot();
        X::reset_lazy_telemetry();
        X::clear_lazy_ir_for_testing();
        ct.gather_lazy(ct).eval();
        ct.gather_lazy(ct).shape();
        ct.gather_lazy(ct).device();
        ct.gather_lazy(ct).dtype();
        ct.gather_lazy(ct).stream_hint();
        ct.gather_lazy(ct).snapshot();
        ct.gather_lazy(ct).map(operation);
        TensorLeaf(t).eval();
        TensorLeaf(t).shape();
        TensorLeaf(t).device();
        TensorLeaf(t).dtype();
        TensorLeaf(t).stream_hint();
        TensorLeaf(t).snapshot();
        TensorLeaf(t).map(operation);
        ct.template apply([](const X& value) { return value; });
        t.template inplace([](X&) {});
        ct.template timed("", [](const X& value) { return value; });
    };

    using LeafExpr = TensorLeaf;
    using UnaryExpression = UnaryExpr<LeafExpr, ops::abs_op>;
    using NestedUnaryExpression = UnaryExpr<UnaryExpression, ops::neg_op>;
    using BinaryExpression = BinaryExpr<LeafExpr, LeafExpr, ops::add_op>;
    using ScalarOperation = ops::scalar_right_op<ops::add_op, float>;
    using ScalarExpression = ScalarUnaryExpr<LeafExpr, ScalarOperation>;
    using PermutationExpression = PermutationExpr<LeafExpr, LeafExpr>;
    using GatherUnaryExpression = UnaryExpr<PermutationExpression, ops::abs_op>;

    template <typename X>
    concept ConcreteExprSurface = requires(const LeafExpr& leaf, const UnaryExpression& unary,
                                           const NestedUnaryExpression& nested,
                                           const BinaryExpression& binary,
                                           const ScalarExpression& scalar,
                                           const PermutationExpression& permutation,
                                           const GatherUnaryExpression& gather_unary,
                                           ops::neg_op operation) {
        leaf.eval_impl();
        leaf.snapshot_impl();
        leaf.shape_impl();
        leaf.device_impl();
        leaf.dtype_impl();
        leaf.stream_hint_impl();
        leaf.map(operation);
        unary.eval_impl();
        unary.snapshot_impl();
        unary.map(operation);
        unary.shape_impl();
        unary.device_impl();
        unary.dtype_impl();
        unary.stream_hint_impl();
        nested.eval_impl();
        nested.snapshot_impl();
        nested.map(operation);
        nested.shape_impl();
        nested.device_impl();
        nested.dtype_impl();
        nested.stream_hint_impl();
        binary.eval_impl();
        binary.snapshot_impl();
        binary.map(operation);
        binary.shape_impl();
        binary.device_impl();
        binary.dtype_impl();
        binary.stream_hint_impl();
        scalar.eval_impl();
        scalar.snapshot_impl();
        scalar.map(operation);
        scalar.shape_impl();
        scalar.device_impl();
        scalar.dtype_impl();
        scalar.stream_hint_impl();
        permutation.eval_impl();
        permutation.snapshot_impl();
        permutation.map(operation);
        permutation.shape_impl();
        permutation.device_impl();
        permutation.dtype_impl();
        permutation.stream_hint_impl();
        gather_unary.eval_impl();
        gather_unary.snapshot_impl();
        gather_unary.shape_impl();
        gather_unary.device_impl();
        gather_unary.dtype_impl();
        gather_unary.stream_hint_impl();
    };

    template <typename X>
    concept RowSurface = requires(X& t, const X& ct, RP& row, const RP& const_row,
                                  TensorIndexer& indexer, MaskedTensorProxy& masked) {
        t.template accessor<float, 1>();
        row[0];
        const_row[0];
        row.item();
        static_cast<float>(const_row);
        const_row.template item_as<float>();
        const_row.item_int();
        const_row.item_int64();
        static_cast<X>(const_row);
        row = const_row;
        row = ct;
        row = 1.0f;
        const_row - const_row;
        const_row + const_row;
        const_row* const_row;
        const_row / const_row;
        const_row - 1.0f;
        const_row + 1.0f;
        const_row * 1.0f;
        const_row / 1.0f;
        -const_row;
        const_row.pow(2.0f);
        const_row.sqrt();
        const_row.abs();
        const_row.neg();
        const_row.sum();
        const_row.mean();
        const_row.square();
        masked = 1.0f;
        masked = ct;
        static_cast<X>(masked);
        indexer = 1.0f;
        indexer = ct;
        static_cast<X>(indexer);
    };

    static_assert(CreationFactorySurface<T>);
    static_assert(MovementSurface<T>);
    static_assert(ShapeValueSurface<R, S>);
    static_assert(UnarySurface<T>);
    static_assert(BinarySurface<T>);
    static_assert(ReductionSurface<T>);
    static_assert(MatrixSurface<T>);
    static_assert(IndexSurface<T>);
    static_assert(RandomSurface<T>);
    static_assert(NnSurface<T>);
    static_assert(SerializationSurface<T>);
    static_assert(SyncSurface<T>);
    static_assert(MemorySurface<T>);
    static_assert(DebugSurface<T>);
    static_assert(LazySurface<T>);
    static_assert(ExprBaseSurface<LeafExpr>);
    static_assert(ExprBaseSurface<UnaryExpression>);
    static_assert(ExprBaseSurface<NestedUnaryExpression>);
    static_assert(ExprBaseSurface<BinaryExpression>);
    static_assert(ExprBaseSurface<ScalarExpression>);
    static_assert(ExprBaseSurface<PermutationExpression>);
    static_assert(ExprBaseSurface<GatherUnaryExpression>);
    static_assert(ConcreteExprSurface<T>);
    static_assert(RowSurface<T>);

} // namespace

int main() {
    constexpr size_t kExactCasts =
        std::tuple_size_v<decltype(kFactoryOverloads)> + std::tuple_size_v<decltype(kMovementOverloads)> +
        std::tuple_size_v<decltype(kReductionOverloads)> + std::tuple_size_v<decltype(kIndexOverloads)>;
    static_assert(kExactCasts + kExactSignatureCount <= kFrozenCallableSlots.size());
    std::printf("tensor_interface_freeze: covered %zu callables (%zu exact overload casts, "
                "%zu exact signatures, %zu well-formedness checks)\n",
                kFrozenCallableSlots.size(), kExactCasts, kExactSignatureCount,
                kFrozenCallableSlots.size() - kExactCasts - kExactSignatureCount);
    return 0;
}
