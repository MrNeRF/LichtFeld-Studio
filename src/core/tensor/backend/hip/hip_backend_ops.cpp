/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "hip_backend_ops.hpp"
#include "architecture_runtime.hpp"
#include "expression_api.h"
#include "runtime_api.h"
#include "runtime_context.hpp"
#ifdef LFS_HIP_EXPRESSION_JIT
#include "hip_expression_emitter.hpp"
#endif
#include "../../internal/expression_runtime.hpp"
#include "../../internal/tensor_impl.hpp"
#include "../export_pipeline.hpp"
#include "../readback_buffer.hpp"
#include "bias_api.h"
#include "gemm_api.h"
#include "inference_api.h"
#include "integer_reduce_api.h"
#include "inverse_warp_api.h"
#include "nn_api.h"
#include "point_filter_api.h"
#include "point_tree_api.h"
#include "predicate_api.h"
#include "sh_codec_api.h"
#include "timing_api.h"
#include "where_api.h"
#include <atomic>
#include <core/detail/fused_pointwise.hpp>
#include <core/detail/tensor_half.hpp>
#include <core/error.hpp>
#include <core/failure_report.hpp>
#include <core/logger.hpp>
#include <core/memory_pressure.hpp>
#include <core/splat_transform_math.hpp>
#include <core/tensor.hpp>
#include <core/tensor_environment.hpp>
#include <core/tensor_image.hpp>
#include <core/tensor_ppisp.hpp>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <source_location>
#include <stdexcept>
#include <unordered_map>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
namespace lfs::core::internal {
    int select_hip_device(const std::filesystem::path& module, int requested) {
        const auto count = hip_device_count(module);
        for (int ordinal = 0; ordinal < count; ++ordinal) {
            if (requested >= 0 && requested != ordinal)
                continue;
            const auto info = hip_device_info(module, ordinal);
            const auto identity = hip_device_identity(module, ordinal);
            if ((info.warp_size == 32 || info.warp_size == 64) &&
                lfs_hip_has_architecture(identity.compiled_architecture, info.architecture))
                return ordinal;
        }
        throw std::runtime_error("No compatible HIP GPU is available for this Tensor runtime build");
    }

    namespace {
        std::atomic<unsigned> live_contexts{0};
        std::atomic<uint64_t> next_context_identity{1};
        void require(bool value, const char* message) {
            if (!value)
                throw std::invalid_argument(message);
        }
        void check(int status, const LfsHipError& error) {
            if (status)
                throw HipBackendError(error.native_code, std::string("HIP Tensor: ") + error.message + " (" + std::to_string(error.native_code) + ")");
        }
        void queue(ExecContext context) {
            require(context.cuda_stream == nullptr, "HIP Tensor adapter accepts only its own default queue, never a CUDA stream");
        }
        float scalar_value(ScalarOperand value) {
            switch (value.kind) {
            case ScalarKind::Float: return value.value.float_value;
            case ScalarKind::Int32: return static_cast<float>(value.value.int32_value);
            case ScalarKind::Int64: return static_cast<float>(value.value.int64_value);
            case ScalarKind::Bool: return value.value.bool_value ? 1.f : 0.f;
            }
            throw std::invalid_argument("Unknown HIP scalar kind");
        }
        LfsHipLayout native_layout(const StridedLayout& source) {
            require(source.rank <= 8, "HIP Tensor rank exceeds eight dimensions");
            LfsHipLayout result{};
            result.rank = source.rank;
            result.elements = source.element_count;
            for (size_t i = 0; i < source.rank; ++i) {
                result.dims[i] = source.dims[i];
                result.strides[i] = source.strides[i];
            }
            return result;
        }
        uint64_t typed_scalar_bits(DataType type, ScalarOperand value) {
            switch (type) {
            case DataType::Float32: {
                require(value.kind == ScalarKind::Float, "HIP float fill requires a floating scalar");
                uint32_t bits = 0;
                std::memcpy(&bits, &value.value.float_value, sizeof(bits));
                return bits;
            }
            case DataType::Float16: {
                require(value.kind == ScalarKind::Float, "HIP half fill requires a floating scalar");
                const auto half = detail::tensor_float_to_half(value.value.float_value);
                static_assert(sizeof(half) == sizeof(uint16_t));
                uint16_t bits = 0;
                std::memcpy(&bits, &half, sizeof(bits));
                return bits;
            }
            case DataType::Int32:
                require(value.kind == ScalarKind::Int32, "HIP Int32 fill requires Int32 scalar");
                return static_cast<uint32_t>(value.value.int32_value);
            case DataType::Int64:
                require(value.kind == ScalarKind::Int64, "HIP Int64 fill requires Int64 scalar");
                return static_cast<uint64_t>(value.value.int64_value);
            case DataType::Bool:
                require(value.kind == ScalarKind::Bool, "HIP Bool fill requires Bool scalar");
                return value.value.bool_value ? 1 : 0;
            case DataType::UInt8:
                require(value.kind == ScalarKind::Int32 && value.value.int32_value >= 0 && value.value.int32_value <= 255, "HIP UInt8 fill scalar out of range");
                return static_cast<uint8_t>(value.value.int32_value);
            case DataType::UInt32:
                require(value.kind == ScalarKind::Int64 && value.value.int64_value >= 0 && uint64_t(value.value.int64_value) <= UINT32_MAX, "HIP UInt32 fill scalar out of range");
                return static_cast<uint32_t>(value.value.int64_value);
            default: throw std::invalid_argument("Unsupported HIP typed fill");
            }
        }
        size_t byte_count(size_t count) {
            require(count <= SIZE_MAX / 4, "HIP Tensor float count overflow");
            return count * 4;
        }
        size_t checked_product(size_t a, size_t b) {
            require(!b || a <= SIZE_MAX / b, "HIP concatenation extent overflow");
            return a * b;
        }
    } // namespace
    namespace {
        std::shared_ptr<HipRuntimeModule> load_tensor_module(const std::filesystem::path& path) {
#ifdef _WIN32
            return std::make_shared<HipRuntimeModule>(path, path.parent_path() / LFS_HIP_RTC_FILENAME);
#else
            return std::make_shared<HipRuntimeModule>(path);
#endif
        }
        std::shared_ptr<const HipRuntimeModule> inventory_module(const std::filesystem::path& path) {
            // Inventory is also read by UI/background samplers. Keep one native
            // module registration rather than concurrently registering and
            // unloading its device images on every name/budget query. No Tensor
            // context is created or retained by this discovery-only cache.
            static std::mutex mutex;
            static std::unordered_map<std::filesystem::path, std::shared_ptr<const HipRuntimeModule>> modules;
            std::lock_guard lock(mutex);
            const auto key = path.lexically_normal();
            if (const auto found = modules.find(key); found != modules.end())
                return found->second;
            auto module = load_tensor_module(path);
            modules.emplace(key, module);
            return module;
        }
    } // namespace
    int hip_device_count(const std::filesystem::path& path) {
        const auto module = inventory_module(path);
        int32_t count = 0;
        LfsHipError error{};
        check(module->api().device_count(&count, &error), error);
        require(count >= 0, "HIP hardware inventory returned a negative device count");
        return count;
    }
    LfsHipDevice hip_device_info(const std::filesystem::path& path, int ordinal) {
        require(ordinal >= 0, "HIP device ordinal must be nonnegative");
        const auto module = inventory_module(path);
        LfsHipDevice result{};
        LfsHipError error{};
        check(module->api().device_info(ordinal, &result, &error), error);
        result.name[sizeof(result.name) - 1] = 0;
        result.architecture[sizeof(result.architecture) - 1] = 0;
        result.pci_bus_id[sizeof(result.pci_bus_id) - 1] = 0;
        return result;
    }
    LfsHipDeviceIdentity hip_device_identity(const std::filesystem::path& path, int ordinal) {
        require(ordinal >= 0, "HIP device ordinal must be nonnegative");
        const auto module = inventory_module(path);
        LfsHipDeviceIdentity result{};
        result.struct_bytes = sizeof(result);
        LfsHipError error{};
        check(module->api().device_identity(ordinal, &result, &error), error);
        result.compiled_architecture[sizeof(result.compiled_architecture) - 1] = 0;
        require(result.struct_bytes == sizeof(result), "HIP device identity size mismatch");
        return result;
    }
    std::optional<GpuDeviceInfo> hip_gpu_device_info(const std::filesystem::path& path, int ordinal) {
        try {
            const auto info = hip_device_info(path, ordinal);
            const auto identity = hip_device_identity(path, ordinal);
            GpuDeviceInfo result{.name = info.name, .total_memory_bytes = info.total_bytes};
            std::copy_n(identity.uuid, result.uuid.size(), result.uuid.begin());
            return result;
        } catch (...) { /* LFS-CENSUS-OK(empty-catch): Inventory reports an unavailable ordinal without creating an execution context. */
            return std::nullopt;
        }
    }
    struct HipBackendOps::Impl {
        std::shared_ptr<HipRuntimeModule> module;
        std::unique_ptr<HipRuntimeContext> runtime;
        HipRuntimeApi api{};
        LfsHipContext* context = nullptr;
        bool counted = false;
        std::atomic<bool> active{false};
        int ordinal;
        uint64_t identity = next_context_identity.fetch_add(1, std::memory_order_relaxed);
        LfsHipDevice device{};
        struct Allocation {
            LfsHipBuffer* handle;
            size_t bytes;
            std::unique_ptr<StorageMeta> metadata;
            bool cacheable = true;
            bool retired = false;
            std::string diagnostic_origin;
        };
        std::unordered_map<void*, Allocation> allocations;
        std::vector<Allocation> cached;
        size_t cached_bytes = 0, cache_limit = 0, reserved_peak = 0;
        std::mutex mutex;
#ifdef LFS_HIP_EXPRESSION_JIT
        std::string expression_identity;
        ExpressionCache expressions;
        struct ExpressionKernel final : CompiledExpression {
            Impl* owner = nullptr;
            LfsHipExpression* module = nullptr;
            ~ExpressionKernel() override {
                if (module) {
                    LfsHipError error{};
                    if (owner->api.expression_destroy(owner->context, module, &error))
                        std::fprintf(stderr, "[error] HIP expression module retained until context teardown: %s\n", error.message);
                }
            }
        };
#endif
        size_t live = 0, peak = 0;
        size_t workspace_bytes = 0;
        size_t algorithm_workspace_bytes = 0;
        explicit Impl(std::filesystem::path path, int ordinal, size_t cache_limit) : ordinal(ordinal), cache_limit(cache_limit) {
            module = load_tensor_module(path);
            api = module->api();
            LfsHipError error{};
            check(api.device_info(ordinal, &device, &error), error);
            runtime = std::make_unique<HipRuntimeContext>(module, ordinal);
            context = runtime->lock().handle();
            live_contexts.fetch_add(1, std::memory_order_relaxed);
            counted = true;
            active.store(true, std::memory_order_release);
        }
        ~Impl() {
#ifdef LFS_HIP_EXPRESSION_JIT
            expressions.clear();
#endif
            // The owned runtime retains the loaded module on cleanup failure
            // and retries through the same retirement mechanism as host tests.
            runtime.reset();
            if (counted)
                live_contexts.fetch_sub(1, std::memory_order_relaxed);
        }
        LfsHipView view(StorageRef storage, size_t bytes = 0,
                        const std::source_location site = std::source_location::current()) {
            require(context && storage.backend == GpuBackend::HIP, "HIP Tensor received a foreign backend or closed context");
            const auto found = allocations.find(storage.data);
            require(found != allocations.end(), "HIP Tensor received unknown storage");
            require(!storage.meta || (storage.meta->backend == GpuBackend::HIP && storage.meta->gpu_descriptor.native_context == identity),
                    "HIP Tensor storage belongs to a different native context");
            if (storage.byte_offset > found->second.bytes || bytes > found->second.bytes - storage.byte_offset)
                throw std::invalid_argument(std::format(
                    "HIP Tensor view exceeds its allocation (offset={}, requested={}, allocation={}, dtype={}, operation={} at {}:{})",
                    storage.byte_offset, bytes, found->second.bytes, static_cast<unsigned>(storage.dtype),
                    site.function_name(), site.file_name(), site.line()));
            return {found->second.handle, storage.byte_offset};
        }
        void bias(StorageRef input, StorageRef bias, StorageRef output, int count, int channels, int spatial, bool relu) {
            require(count >= 0, "HIP bias count must be nonnegative");
            if (!count)
                return;
            require(channels > 0 && spatial > 0 && input.dtype == DataType::Float32 && bias.dtype == DataType::Float32 && output.dtype == DataType::Float32, "HIP bias requires Float32 operands and positive dimensions");
            const auto in = view(input, byte_count(count)), b = view(bias, byte_count(channels)), out = view(output, byte_count(count));
            LfsHipError error{};
            check(api.bias_f32(context, in, b, out, uint32_t(count), uint32_t(channels), uint32_t(spatial), relu ? 1 : 0, &error), error);
        }
        bool empty_view(StorageRef storage) const {
            if (storage.data)
                return false;
            require(context && storage.backend == GpuBackend::HIP && storage.byte_offset == 0 && !(storage.flags & STORAGE_REF_HOST_MEMORY), "Invalid HIP empty device view");
            return true;
        }
        void refresh_algorithm_workspace() {
            LfsHipMemoryInfo memory{};
            LfsHipError error{};
            check(api.memory_info(context, &memory, &error), error);
            workspace_bytes = algorithm_workspace_bytes = memory.workspace_bytes;
            peak = std::max(peak, live + workspace_bytes);
            reserved_peak = std::max(reserved_peak, live + cached_bytes + workspace_bytes);
        }
        void observed_workspace() { refresh_algorithm_workspace(); }
        void gather(StorageRef input, StorageRef indices, StorageRef output, const StridedLayout& input_layout, const StridedLayout& index_layout, const IndexProgram& program, bool select) {
            require(input.dtype == output.dtype, "HIP gather input/output dtype mismatch");
            require(program.dim >= 0 && size_t(program.dim) < input_layout.rank && program.boundary_mode >= 0 && program.boundary_mode <= 2, "Invalid HIP gather dimension or boundary");
            const auto source = native_layout(input_layout), idx = native_layout(index_layout);
            size_t count = 1;
            if (select) {
                for (size_t r = 0; r < input_layout.rank; ++r) {
                    const size_t dim = r == size_t(program.dim) ? index_layout.element_count : input_layout.dims[r];
                    if (!dim) {
                        count = 0;
                        break;
                    }
                }
                if (count)
                    for (size_t r = 0; r < input_layout.rank; ++r) {
                        const size_t dim = r == size_t(program.dim) ? index_layout.element_count : input_layout.dims[r];
                        require(dim <= SIZE_MAX / count, "HIP index_select shape overflow");
                        count *= dim;
                    }
            } else
                count = index_layout.element_count;
            require(select || count == program.total_elements, "HIP gather output count mismatch");
            const auto in = view(input), ix = view(indices), out = view(output);
            LfsHipError error{};
            const LfsHipGather descriptor{sizeof(LfsHipGather), uint32_t(input.dtype), uint32_t(indices.dtype), uint32_t(program.dim), uint32_t(program.boundary_mode), select ? 1u : 0u, source, idx};
            const int status = api.gather(context, in, ix, out, &descriptor, &error);
            check(status, error);
        }
        void index_update(StorageRef output, StorageRef indices, StorageRef source, const StridedLayout& layout, const IndexProgram& program, bool add) {
            require(source.dtype == output.dtype, "HIP indexed update source/output dtype mismatch");
            require(program.dim >= 0 && size_t(program.dim) < layout.rank && program.boundary_mode == 0, "Invalid HIP indexed update dimension or boundary");
            const auto destination = native_layout(layout);
            const auto out = view(output), idx = view(indices), src = view(source);
            LfsHipError error{};
            const LfsHipIndexUpdate descriptor{sizeof(LfsHipIndexUpdate), uint32_t(output.dtype), uint32_t(indices.dtype), uint32_t(program.dim), add ? 1u : 0u, program.index_size, destination};
            const int status = api.index_update(context, out, idx, src, &descriptor, &error);
            check(status, error);
        }
        void random(StorageRef output, const RandomProgram& program, uint32_t kind) {
            require(output.dtype == (kind == LFS_HIP_RANDOM_INTEGER ? DataType::Int32 : DataType::Float32), "HIP random output dtype mismatch");
            const auto out = view(output);
            const LfsHipRandom native{kind, program.seed, program.count, program.first, program.second, program.low, program.high};
            LfsHipError error{};
            check(api.random(context, out, &native, &error), error);
        }
        size_t count_nonzero(StorageRef input, size_t count) {
            const auto in = view(input);
            uint64_t result = 0;
            LfsHipError error{};
            check(api.count_nonzero(context, in, static_cast<uint32_t>(input.dtype), count, &result, &error), error);
            if (count)
                observed_workspace();
            require(result <= SIZE_MAX, "HIP nonzero count exceeds host size_t");
            return static_cast<size_t>(result);
        }
        void sync() {
            require(context, "HIP Tensor context is closed");
            LfsHipError error{};
            check(api.context_synchronize(context, &error), error);
        }
        float reduce(StorageRef input, size_t count, uint32_t mode) {
            require(input.dtype == DataType::Float32, "HIP reduction requires float32 input");
            const auto src = view(input, byte_count(count));
            LfsHipError error{};
            float result = 0;
            check(api.reduce_f32(context, src, count, mode, &result, &error), error);
            if (count)
                observed_workspace();
            peak = std::max(peak, live + workspace_bytes);
            reserved_peak = std::max(reserved_peak, live + cached_bytes + workspace_bytes);
            return result;
        }
        void trim_cache() {
            sync();
            while (!cached.empty()) {
                LfsHipError error{};
                check(api.buffer_release(context, cached.back().handle, &error), error);
                cached_bytes -= cached.back().bytes;
                cached.pop_back();
            }
        }
        void pointwise(const PointwiseProgram& program, StorageRef a, StorageRef b, StorageRef out, size_t count, bool binary) {
            if (!count) {
                require(program.op < PointwiseOp::Count && program.in_dtype == a.dtype && program.out_dtype == out.dtype && (!binary || b.dtype == a.dtype),
                        "Invalid HIP empty pointwise descriptor");
                for (const auto storage : {a, out}) {
                    if (storage.data)
                        (void)view(storage);
                    else
                        (void)empty_view(storage);
                }
                if (binary) {
                    if (b.data)
                        (void)view(b);
                    else
                        (void)empty_view(b);
                }
                return;
            }
            if ((a.dtype == DataType::Int32 || a.dtype == DataType::Int64 || a.dtype == DataType::UInt32 || a.dtype == DataType::UInt8) && out.dtype == a.dtype) {
                require(program.in_dtype == a.dtype && program.out_dtype == out.dtype && (!binary || b.dtype == a.dtype), "HIP integer arithmetic dtype mismatch");
                LfsHipIntegerPointwise native{};
                native.struct_bytes = sizeof(native);
                native.op = uint32_t(program.op);
                native.dtype = uint32_t(a.dtype);
                native.count = count;
                native.scalar_kind = uint32_t(program.scalar.kind);
                native.scalar_left = program.scalar.scalar_on_right ? 0 : 1;
                switch (program.scalar.kind) {
                case ScalarKind::Float: native.float_scalar = program.scalar.value.float_value; break;
                case ScalarKind::Int32: native.integer_scalar = program.scalar.value.int32_value; break;
                case ScalarKind::Int64: native.integer_scalar = program.scalar.value.int64_value; break;
                case ScalarKind::Bool: native.integer_scalar = program.scalar.value.bool_value; break;
                }
                const auto bytes = checked_product(count, dtype_size(a.dtype));
                const auto lhs = view(a, bytes), rhs = binary ? view(b, bytes) : LfsHipView{};
                const auto result = view(out, bytes);
                LfsHipError error{};
                check(api.integer_pointwise(context, lhs, rhs, result, &native, &error), error);
                return;
            }
            if (out.dtype == DataType::Bool && (a.dtype == DataType::Int32 || a.dtype == DataType::Int64 || a.dtype == DataType::UInt32 || a.dtype == DataType::Float16 || a.dtype == DataType::UInt8 || a.dtype == DataType::Bool)) {
                require(program.in_dtype == a.dtype && program.out_dtype == DataType::Bool && (!binary || b.dtype == a.dtype),
                        "HIP typed predicate dtype mismatch");
                LfsHipPredicate native{};
                native.struct_bytes = sizeof(native);
                native.op = uint32_t(program.op);
                native.dtype = uint32_t(a.dtype);
                native.count = count;
                native.scalar_kind = uint32_t(program.scalar.kind);
                native.scalar_left = program.scalar.scalar_on_right ? 0 : 1;
                switch (program.scalar.kind) {
                case ScalarKind::Float: native.float_scalar = program.scalar.value.float_value; break;
                case ScalarKind::Int32: native.integer_scalar = program.scalar.value.int32_value; break;
                case ScalarKind::Int64: native.integer_scalar = program.scalar.value.int64_value; break;
                case ScalarKind::Bool: native.integer_scalar = program.scalar.value.bool_value; break;
                }
                const auto bytes = checked_product(count, dtype_size(a.dtype));
                const auto lhs = view(a, bytes), rhs = binary ? view(b, bytes) : LfsHipView{};
                const auto result = view(out, count);
                LfsHipError error{};
                check(api.predicate_typed(context, lhs, rhs, result, &native, &error), error);
                return;
            }
            if (!binary && !program.scalar.scalar_on_right && (a.dtype == DataType::Float16 || a.dtype == DataType::Float32)) {
                bool scalar_op = false;
                switch (program.op) {
                case PointwiseOp::AddScalar:
                case PointwiseOp::SubScalar:
                case PointwiseOp::MulScalar:
                case PointwiseOp::DivScalar:
                case PointwiseOp::PowScalar:
                case PointwiseOp::ModScalar:
                case PointwiseOp::MinimumScalar:
                case PointwiseOp::MaximumScalar:
                case PointwiseOp::EqualScalar:
                case PointwiseOp::NotEqualScalar:
                case PointwiseOp::LessScalar:
                case PointwiseOp::LessEqualScalar:
                case PointwiseOp::GreaterScalar:
                case PointwiseOp::GreaterEqualScalar: scalar_op = true; break;
                default: break;
                }
                if (scalar_op) {
                    const bool predicate = program.op >= PointwiseOp::EqualScalar && program.op <= PointwiseOp::GreaterEqualScalar;
                    require(program.in_dtype == a.dtype && out.dtype == program.out_dtype && out.dtype == (predicate ? DataType::Bool : a.dtype), "HIP scalar-left dtype mismatch");
                    LfsHipError error{};
                    check(api.scalar_left_typed(context, uint32_t(program.op), view(a, checked_product(count, dtype_size(a.dtype))),
                                                view(out, checked_product(count, dtype_size(out.dtype))), count, scalar_value(program.scalar), uint32_t(a.dtype), &error),
                          error);
                    return;
                }
            }
            if (a.dtype == DataType::Float16) {
                const bool predicate = (program.op >= PointwiseOp::EqualTensor && program.op <= PointwiseOp::LogicalXorTensor) ||
                                       (program.op >= PointwiseOp::IsNan && program.op <= PointwiseOp::LogicalNot) ||
                                       (program.op >= PointwiseOp::EqualScalar && program.op <= PointwiseOp::GreaterEqualScalar);
                require(program.in_dtype == DataType::Float16 && out.dtype == program.out_dtype &&
                            out.dtype == (predicate ? DataType::Bool : DataType::Float16) &&
                            (!binary || b.dtype == DataType::Float16),
                        "HIP half pointwise dtype mismatch");
                const auto bytes = checked_product(count, dtype_size(DataType::Float16));
                const auto lhs = view(a, bytes), rhs = binary ? view(b, bytes) : LfsHipView{};
                const auto output = view(out, predicate ? count : bytes);
                LfsHipError error{};
                check(api.pointwise_f16(context, static_cast<uint32_t>(program.op), lhs, rhs, output,
                                        count, scalar_value(program.scalar), &error),
                      error);
                return;
            }

            if (!(program.in_dtype == DataType::Float32 && a.dtype == DataType::Float32 &&
                  out.dtype == program.out_dtype && (out.dtype == DataType::Float32 || out.dtype == DataType::Bool)))
                throw std::invalid_argument("Unsupported HIP pointwise dtype combination: op=" + std::to_string(unsigned(program.op)) +
                                            ", input=" + std::to_string(unsigned(a.dtype)) + ", declared_input=" + std::to_string(unsigned(program.in_dtype)) +
                                            ", output=" + std::to_string(unsigned(out.dtype)) + ", declared_output=" + std::to_string(unsigned(program.out_dtype)));
            const bool predicate = (program.op >= PointwiseOp::EqualTensor && program.op <= PointwiseOp::LogicalXorTensor) ||
                                   (program.op >= PointwiseOp::IsNan && program.op <= PointwiseOp::LogicalNot) ||
                                   (program.op >= PointwiseOp::EqualScalar && program.op <= PointwiseOp::GreaterEqualScalar);
            require((out.dtype == DataType::Bool) == predicate, "HIP pointwise output dtype does not match its operation");
            const auto lhs = view(a, byte_count(count));
            LfsHipView rhs{};
            if (binary) {
                require(b.dtype == DataType::Float32, "HIP binary rhs must be float32");
                rhs = view(b, byte_count(count));
            }
            const auto output = view(out, out.dtype == DataType::Bool ? count : byte_count(count));
            LfsHipError error{};
            check(api.pointwise_f32(context, static_cast<uint32_t>(program.op), lhs, rhs, output, count, scalar_value(program.scalar), &error), error);
        }
    };
    bool hip_backend_is_live() noexcept { return live_contexts.load(std::memory_order_relaxed) != 0; }
    HipBackendOps::HipBackendOps(const std::filesystem::path& path, int device, size_t max_cached_bytes) : impl_(std::make_shared<Impl>(path, device, max_cached_bytes)) {}
    HipBackendOps::~HipBackendOps() = default;
    void HipBackendOps::ensure_initialized() {
        if (impl_->active.load(std::memory_order_acquire))
            return;
        std::lock_guard lock(impl_->mutex);
        if (impl_->context)
            return;
        auto runtime = std::make_unique<HipRuntimeContext>(impl_->module, impl_->ordinal);
        auto* context = runtime->lock().handle();
        impl_->runtime = std::move(runtime);
        impl_->context = context;
        impl_->identity = next_context_identity.fetch_add(1, std::memory_order_relaxed);
        impl_->workspace_bytes = impl_->algorithm_workspace_bytes = 0;
        impl_->live = impl_->peak = impl_->reserved_peak = 0;
        live_contexts.fetch_add(1, std::memory_order_relaxed);
        impl_->counted = true;
        impl_->active.store(true, std::memory_order_release);
    }
    StorageRef HipBackendOps::allocate(size_t bytes, size_t alignment, ExecContext execution) {
        ensure_initialized();
        queue(execution);
        require(bytes > 0 && alignment && alignment <= 256 && (alignment & (alignment - 1)) == 0, "Invalid HIP allocation size/alignment");
        const auto allocation_failure = [&] {
            return AllocationFailure{.domain = MemoryDomain::HipDevice, .requested_bytes = bytes, .alignment = alignment, .device = impl_->ordinal, .label = execution.allocation_label, .operation = execution.allocation_operation, .native_error = 2};
        };
        if (MemoryPressureCoordinator::instance().probe_should_fail(MemoryDomain::HipDevice, bytes))
            throw MemoryAllocationError(allocation_failure());
        std::lock_guard lock(impl_->mutex);
        require(impl_->context, "HIP Tensor context is closed");
        Impl::Allocation allocation{};
        const bool cacheable = execution.allocation_class == AllocationClass::Pooled;
        auto found = std::find_if(impl_->cached.begin(), impl_->cached.end(), [&](const auto& entry) { return cacheable && entry.bytes == bytes; });
        LfsHipError error{};
        if (found != impl_->cached.end()) {
            allocation = std::move(*found);
            if (found != impl_->cached.end() - 1)
                *found = std::move(impl_->cached.back());
            impl_->cached.pop_back();
            impl_->cached_bytes -= bytes;
        } else {
            int status = impl_->api.buffer_allocate(impl_->context, bytes, &allocation.handle, &error);
            // An optional cache must not prevent an otherwise viable allocation.
            // Drain it and retry once; existing live tensors remain untouched.
            if (status && error.native_code == 2 && !impl_->cached.empty()) {
                impl_->trim_cache();
                status = impl_->api.buffer_allocate(impl_->context, bytes, &allocation.handle, &error);
            }
            if (status && error.native_code == 2)
                throw MemoryAllocationError(allocation_failure());
            check(status, error);
            allocation.bytes = bytes;
            allocation.cacheable = cacheable;
        }
        const auto handle = allocation.handle;
        if (bytes <= 512 && std::getenv("LFS_HIP_VALIDATION_ALLOCATION_TRACE"))
            allocation.diagnostic_origin = capture_host_stacktrace(1);
        uint64_t address = 0;
        try {
            if (allocation.metadata)
                address = allocation.metadata->gpu_descriptor.base_address;
            else {
                check(impl_->api.buffer_address(impl_->context, handle, &address, &error), error);
                allocation.metadata = std::make_unique<StorageMeta>();
                allocation.metadata->backend = GpuBackend::HIP;
                allocation.metadata->gpu_descriptor.base_address = address;
                allocation.metadata->gpu_descriptor.native_context = impl_->identity;
                allocation.metadata->gpu_descriptor.byte_size = bytes;
                allocation.metadata->gpu_descriptor.accounting_kind = StorageAccountingKind::HipOwned;
            }
            require(address && address % alignment == 0, "HIP allocation does not meet alignment");
            auto* pointer = reinterpret_cast<void*>(static_cast<uintptr_t>(address));
            require(!impl_->allocations.contains(pointer), "HIP allocator returned an already active address");
            impl_->allocations.emplace(pointer, std::move(allocation));
        } catch (...) {
            if (impl_->api.buffer_release(impl_->context, handle, &error))
                std::fprintf(stderr, "[error] HIP allocation cleanup failed; context retains allocation: %s\n", error.message);
            throw;
        }
        impl_->live += bytes;
        impl_->peak = std::max(impl_->peak, impl_->live + impl_->workspace_bytes);
        impl_->reserved_peak = std::max(impl_->reserved_peak, impl_->live + impl_->cached_bytes + impl_->workspace_bytes);
        auto* pointer = reinterpret_cast<void*>(static_cast<uintptr_t>(address));
        return {GpuBackend::HIP, pointer, 0, DataType::Float32, impl_->allocations.at(pointer).metadata.get(),
                cacheable ? 0u : STORAGE_REF_DIRECT_ALLOCATION};
    }
    void HipBackendOps::deallocate(StorageRef storage, ExecContext execution) noexcept {
        try {
            queue(execution);
            std::lock_guard lock(impl_->mutex);
            require(storage.byte_offset == 0, "HIP deallocation requires allocation base");
            const auto value = impl_->view(storage);
            auto found = impl_->allocations.find(storage.data);
            require(!found->second.retired, "HIP allocation was already retired");
            found->second.retired = true;
            const size_t bytes = found->second.bytes;
            if (found->second.cacheable && bytes <= impl_->cache_limit - impl_->cached_bytes) {
                // Only one native queue is admitted. Reusing this handle queues
                // later writes after every prior producer/consumer on that queue.
                impl_->cached.push_back(std::move(found->second));
                impl_->cached.back().retired = false;
                impl_->cached_bytes += bytes;
                impl_->live -= bytes;
                impl_->allocations.erase(found);
                return;
            }
            LfsHipError error{};
            check(impl_->api.buffer_release(impl_->context, value.buffer, &error), error);
            impl_->live -= bytes;
            impl_->allocations.erase(found);
        } catch (const std::exception& error) { /* LFS-CENSUS-OK(empty-catch): Report failed cleanup and retain GPU owners until safe teardown. */
            std::fprintf(stderr, "[error] HIP Tensor deallocation failed; allocation retained: %s\n", error.what());
        }
    }
    // These small matrix building blocks reuse the native queue and storage
    // validation; no host download or fallback to another GPU backend.
    void HipBackendOps::dot_product(StorageRef lhs, StorageRef rhs, StorageRef output, size_t count, ExecContext execution) {
        queue(execution);
        require(lhs.dtype == DataType::Float32 && rhs.dtype == DataType::Float32 && output.dtype == DataType::Float32,
                "HIP dot requires float32 operands");
        const auto bytes = byte_count(count);
        {
            std::lock_guard lock(impl_->mutex);
            (void)impl_->view(output, sizeof(float));
            if (count) {
                (void)impl_->view(lhs, bytes);
                (void)impl_->view(rhs, bytes);
            }
        }
        if (!count) {
            load_fill(output, 1, {}, execution);
            return;
        }
        auto product = allocate(bytes, 16, execution);
        try {
            binary({PointwiseOp::MulTensor, DataType::Float32, DataType::Float32, {}}, lhs, rhs, product, count, execution);
            StridedLayout layout{};
            layout.rank = 1;
            layout.dims[0] = layout.element_count = count;
            layout.strides[0] = 1;
            ReduceProgram sum{};
            sum.op = ReduceOp::Sum;
            sum.axis_count = 1;
            sum.result_dtype = DataType::Float32;
            reduce(product, output, layout, sum, execution);
        } catch (...) {
            deallocate(product, execution);
            throw;
        }
        deallocate(product, execution);
    }
    void HipBackendOps::diag(StorageRef diagonal, StorageRef output, size_t count, ExecContext execution) {
        queue(execution);
        require(diagonal.dtype == DataType::Float32 && output.dtype == DataType::Float32, "HIP diag requires float32");
        const size_t bytes = byte_count(checked_product(count, count));
        if (!count)
            return;
        std::lock_guard lock(impl_->mutex);
        const auto src = impl_->view(diagonal, byte_count(count)), dst = impl_->view(output, bytes);
        require(src.buffer != dst.buffer || src.offset >= dst.offset + bytes || dst.offset >= src.offset + byte_count(count),
                "HIP diag input and output must not overlap");
        LfsHipLayout layout{};
        layout.rank = 1;
        layout.dims[0] = layout.elements = count;
        layout.strides[0] = count + 1;
        LfsHipError error{};
        check(impl_->api.memset(impl_->context, dst, bytes, 0, &error), error);
        check(impl_->api.strided_scatter(impl_->context, src, dst, &layout, 4, 0, &error), error);
    }
    void HipBackendOps::eye(StorageRef output, size_t rows, size_t columns, ExecContext execution) {
        queue(execution);
        require(output.dtype == DataType::Float32, "HIP eye requires float32");
        const size_t elements = checked_product(rows, columns);
        const size_t bytes = byte_count(elements);
        if (!elements)
            return;
        // Check the complete allocation before initializing any element.
        {
            std::lock_guard lock(impl_->mutex);
            (void)impl_->view(output, bytes);
        }
        StridedLayout diagonal{};
        diagonal.rank = 1;
        diagonal.dims[0] = diagonal.element_count = std::min(rows, columns);
        diagonal.strides[0] = columns + 1;
        ScalarOperand one{};
        one.value.float_value = 1.f;
        memset({output, bytes, 0, false, execution});
        fill_strided(output, diagonal, one, execution);
    }
    void HipBackendOps::broadcast_binary(const PointwiseProgram& program, StorageRef lhs, const StridedLayout& lhs_layout, StorageRef rhs, const StridedLayout& rhs_layout, StorageRef output, const StridedLayout& output_layout, ExecContext execution) {
        queue(execution);
        require((lhs.dtype == DataType::Float32 || lhs.dtype == DataType::Float16 || lhs.dtype == DataType::Bool ||
                 lhs.dtype == DataType::UInt8 || lhs.dtype == DataType::Int32 || lhs.dtype == DataType::Int64 || lhs.dtype == DataType::UInt32) &&
                    program.in_dtype == lhs.dtype && rhs.dtype == lhs.dtype && output.dtype == program.out_dtype &&
                    (output.dtype == lhs.dtype || output.dtype == DataType::Bool),
                "HIP broadcast requires matching supported operands and matching/predicate output");
        const auto a = native_layout(lhs_layout), b = native_layout(rhs_layout), out = native_layout(output_layout);
        std::lock_guard lock(impl_->mutex);
        const auto av = impl_->view(lhs), bv = impl_->view(rhs), ov = impl_->view(output);
        LfsHipError error{};
        check(impl_->api.broadcast_typed(impl_->context, static_cast<uint32_t>(program.op), av, bv, ov, &a, &b, &out, uint32_t(lhs.dtype), uint32_t(dtype_size(output.dtype)), &error), error);
    }
    namespace {
        uint32_t native_reduce_op(ReduceOp operation) {
            switch (operation) {
            case ReduceOp::Sum: return LFS_HIP_REDUCE_SUM;
            case ReduceOp::Mean: return LFS_HIP_REDUCE_MEAN;
            case ReduceOp::Min: return LFS_HIP_REDUCE_MIN;
            case ReduceOp::Max: return LFS_HIP_REDUCE_MAX;
            case ReduceOp::Any: return LFS_HIP_REDUCE_ANY;
            case ReduceOp::All: return LFS_HIP_REDUCE_ALL;
            default: throw std::invalid_argument("Unsupported HIP reduction operation");
            }
        }
    } // namespace
    void HipBackendOps::reduce(StorageRef input, StorageRef output, const StridedLayout& layout, const ReduceProgram& program, ExecContext execution) {
        queue(execution);
        if (input.dtype != DataType::Float32 || program.op == ReduceOp::Any || program.op == ReduceOp::All) {
            require(output.dtype == program.result_dtype, "HIP reduction result dtype mismatch");
            const auto native = native_layout(layout);
            require(program.axis_count <= layout.rank, "HIP typed reduction axis count exceeds rank");
            uint32_t mask = 0;
            for (size_t i = 0; i < program.axis_count; ++i) {
                const int axis = program.axes[i];
                require(axis >= 0 && size_t(axis) < layout.rank && !(mask & (1u << axis)), "Invalid or duplicate HIP typed reduction axis");
                mask |= 1u << axis;
            }
            std::lock_guard lock(impl_->mutex);
            const auto in = impl_->view(input), out = impl_->view(output);
            LfsHipError error{};
            check(impl_->api.reduce_typed_layout(impl_->context, in, out, &native, mask, native_reduce_op(program.op), static_cast<uint32_t>(input.dtype), static_cast<uint32_t>(output.dtype), &error), error);
            if (input.dtype == DataType::Float16 && program.op != ReduceOp::Any && program.op != ReduceOp::All) {
                impl_->observed_workspace();
                impl_->peak = std::max(impl_->peak, impl_->live + impl_->workspace_bytes);
                impl_->reserved_peak = std::max(impl_->reserved_peak, impl_->live + impl_->cached_bytes + impl_->workspace_bytes);
            }
            return;
        }
        require(input.dtype == DataType::Float32 && output.dtype == DataType::Float32 && program.result_dtype == DataType::Float32,
                "HIP axis reductions require float32 input and output");
        uint32_t op;
        switch (program.op) {
        case ReduceOp::Sum: op = LFS_HIP_REDUCE_SUM; break;
        case ReduceOp::Mean: op = LFS_HIP_REDUCE_MEAN; break;
        case ReduceOp::Min: op = LFS_HIP_REDUCE_MIN; break;
        case ReduceOp::Max: op = LFS_HIP_REDUCE_MAX; break;
        default: throw std::invalid_argument("HIP axis reduction operation not implemented");
        }
        const auto native = native_layout(layout);
        require(program.axis_count <= layout.rank, "HIP reduction axis count exceeds rank");
        uint32_t mask = 0;
        for (size_t i = 0; i < program.axis_count; ++i) {
            const int axis = program.axes[i];
            require(axis >= 0 && size_t(axis) < layout.rank && !(mask & (1u << axis)), "Invalid or duplicate HIP reduction axis");
            mask |= 1u << axis;
        }
        std::lock_guard lock(impl_->mutex);
        const auto in = impl_->view(input), out = impl_->view(output);
        LfsHipError error{};
        check(impl_->api.reduce_layout_f32(impl_->context, in, out, &native, mask, op, &error), error);
        // Layout reductions use the same fixed workspace as scalar reductions.
        bool nonempty_output = true;
        for (size_t axis = 0; axis < layout.rank; ++axis)
            if (program.axis_count && !(mask & (1u << axis)) && !layout.dims[axis])
                nonempty_output = false;
        if (nonempty_output)
            impl_->observed_workspace();
        impl_->peak = std::max(impl_->peak, impl_->live + impl_->workspace_bytes);
        impl_->reserved_peak = std::max(impl_->reserved_peak, impl_->live + impl_->cached_bytes + impl_->workspace_bytes);
    }
    void HipBackendOps::strided_reduce(StorageRef input, StorageRef output, size_t outer, size_t reduced, size_t inner, const ReduceProgram& program, ExecContext execution) {
        require(!inner || reduced <= SIZE_MAX / inner, "HIP reduction extent overflow");
        const size_t stride = reduced * inner;
        require(!stride || outer <= SIZE_MAX / stride, "HIP reduction extent overflow");
        StridedLayout layout{};
        layout.rank = 3;
        layout.dims[0] = outer;
        layout.dims[1] = reduced;
        layout.dims[2] = inner;
        layout.strides[0] = stride;
        layout.strides[1] = inner;
        layout.strides[2] = 1;
        layout.element_count = outer * stride;
        auto axis = program;
        axis.axis_count = 1;
        axis.axes[0] = 1;
        reduce(input, output, layout, axis, execution);
    }
    bool HipBackendOps::arg_extreme(StorageRef input, StorageRef values, StorageRef indices, const ArgExtremeProgram& program, ExecContext execution) {
        queue(execution);
        require(input.dtype == DataType::Float32 && values.dtype == DataType::Float32 && indices.dtype == DataType::Int64,
                "HIP indexed extrema require Float32 input/values and Int64 indices");
        std::lock_guard lock(impl_->mutex);
        const auto in = impl_->view(input), val = impl_->view(values), idx = impl_->view(indices);
        LfsHipError error{};
        const int status = impl_->api.arg_extreme_f32(impl_->context, in, val, idx, program.outer, program.reduce,
                                                      program.inner, program.maximum ? 1u : 0u, &error);
        impl_->refresh_algorithm_workspace();
        check(status, error);
        return true;
    }
    void HipBackendOps::cdist(StorageRef lhs, StorageRef rhs, StorageRef output, size_t lhs_rows, size_t rhs_rows,
                              size_t columns, float p, ExecContext execution) {
        queue(execution);
        require(lhs.dtype == DataType::Float32 && rhs.dtype == DataType::Float32 && output.dtype == DataType::Float32,
                "HIP cdist requires Float32 inputs and output");
        require(!std::isnan(p) && p >= 0.f, "HIP cdist p must be nonnegative");
        const size_t count = checked_product(lhs_rows, rhs_rows);
        (void)byte_count(checked_product(lhs_rows, columns));
        (void)byte_count(checked_product(rhs_rows, columns));
        (void)byte_count(count);
        if (!count)
            return;
        if (!columns) {
            (void)impl_->empty_view(lhs);
            (void)impl_->empty_view(rhs);
            ScalarOperand zero{};
            load_fill(output, count, zero, execution);
            return;
        }
        std::lock_guard lock(impl_->mutex);
        const auto a = impl_->view(lhs), b = impl_->view(rhs), out = impl_->view(output);
        LfsHipError error{};
        check(impl_->api.cdist_f32(impl_->context, a, b, out, lhs_rows, rhs_rows, columns, p, &error), error);
    }
    void HipBackendOps::column_reduce(StorageRef input, StorageRef output, size_t rows, size_t columns, const ReduceProgram& program, ExecContext execution) {
        strided_reduce(input, output, 1, rows, columns, program, execution);
    }
    void HipBackendOps::convert_type(StorageRef input, StorageRef output, size_t count, ExecContext execution) {
        queue(execution);
        static_assert(static_cast<unsigned>(DataType::Float32) == LFS_HIP_F32 && static_cast<unsigned>(DataType::Float16) == LFS_HIP_F16 &&
                      static_cast<unsigned>(DataType::Int32) == LFS_HIP_I32 && static_cast<unsigned>(DataType::Int64) == LFS_HIP_I64 &&
                      static_cast<unsigned>(DataType::UInt8) == LFS_HIP_U8 && static_cast<unsigned>(DataType::Bool) == LFS_HIP_BOOL && static_cast<unsigned>(DataType::UInt32) == LFS_HIP_U32);
        std::lock_guard lock(impl_->mutex);
        const auto in = impl_->view(input), out = impl_->view(output);
        LfsHipError error{};
        if (output.dtype == DataType::Bool && input.dtype != DataType::Bool) {
            if (input.dtype == DataType::Float32) {
                check(impl_->api.pointwise_f32(impl_->context, LFS_HIP_OP_NotEqualScalar, in, {}, out, count, 0.f, &error), error);
            } else {
                const LfsHipPredicate predicate{sizeof(LfsHipPredicate), LFS_HIP_OP_NotEqualScalar, uint32_t(input.dtype), 0, count, 0, 0.f, 0};
                check(impl_->api.predicate_typed(impl_->context, in, {}, out, &predicate, &error), error);
            }
            return;
        }
        check(impl_->api.convert(impl_->context, in, static_cast<uint32_t>(input.dtype), out, static_cast<uint32_t>(output.dtype), count, &error), error);
    }
    void HipBackendOps::index_cast(StorageRef input, StorageRef output, size_t count, size_t extent, ExecContext execution) {
        queue(execution);
        require(input.dtype == DataType::Int64 && output.dtype == DataType::Int32, "HIP index cast requires int64 input and int32 output");
        std::lock_guard lock(impl_->mutex);
        const auto in = impl_->view(input), out = impl_->view(output);
        LfsHipError error{};
        const int status = impl_->api.index_cast(impl_->context, in, out, count, extent, &error);
        if (count && (status == 0 || error.native_code == LFS_HIP_ERROR_INVALID_INDEX)) {
            impl_->observed_workspace();
            impl_->peak = std::max(impl_->peak, impl_->live + impl_->workspace_bytes);
            impl_->reserved_peak = std::max(impl_->reserved_peak, impl_->live + impl_->cached_bytes + impl_->workspace_bytes);
        }
        check(status, error);
    }
    void HipBackendOps::gather(StorageRef input, StorageRef indices, StorageRef output, const StridedLayout& input_layout, const StridedLayout& index_layout, const IndexProgram& program, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        impl_->gather(input, indices, output, input_layout, index_layout, program, false);
    }
    void HipBackendOps::index_select(StorageRef input, StorageRef indices, StorageRef output, const StridedLayout& input_layout, const IndexProgram& program, ExecContext execution) {
        queue(execution);
        StridedLayout index_layout{};
        index_layout.rank = 1;
        index_layout.dims[0] = index_layout.element_count = program.index_size;
        index_layout.strides[0] = 1;
        std::lock_guard lock(impl_->mutex);
        impl_->gather(input, indices, output, input_layout, index_layout, program, true);
    }
    size_t HipBackendOps::count_nonzero_bool(StorageRef input, size_t count, ExecContext execution) {
        queue(execution);
        require(input.dtype == DataType::Bool || input.dtype == DataType::UInt8, "HIP byte count requires bool/uint8 input");
        std::lock_guard lock(impl_->mutex);
        return impl_->count_nonzero(input, count);
    }
    size_t HipBackendOps::count_nonzero_float(StorageRef input, size_t count, ExecContext execution) {
        queue(execution);
        require(input.dtype == DataType::Float32, "HIP float count requires float32 input");
        std::lock_guard lock(impl_->mutex);
        return impl_->count_nonzero(input, count);
    }
    void HipBackendOps::index_copy(StorageRef output, StorageRef indices, StorageRef source, const StridedLayout& layout, const IndexProgram& program, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        impl_->index_update(output, indices, source, layout, program, false);
    }
    void HipBackendOps::index_put(StorageRef output, StorageRef indices, StorageRef values, const IndexProgram& program, ExecContext execution) {
        queue(execution);
        require(output.dtype == DataType::Float32 && values.dtype == DataType::Float32,
                "HIP index_put requires Float32 output and values");
        std::lock_guard lock(impl_->mutex);
        const auto out = impl_->view(output), idx = impl_->view(indices), source = impl_->view(values);
        LfsHipError error{};
        check(impl_->api.index_put_f32(impl_->context, out, idx, static_cast<uint32_t>(indices.dtype),
                                       source, program.input_size, program.index_size, &error),
              error);
    }
    void HipBackendOps::gather_fused_unary(StorageRef input, StorageRef indices, StorageRef output,
                                           PointwiseOp unary, const IndexProgram& program, ExecContext execution) {
        queue(execution);
        require(input.dtype == DataType::Float32 && output.dtype == DataType::Float32,
                "HIP fused gather requires Float32 input and output");
        std::lock_guard lock(impl_->mutex);
        const auto in = impl_->view(input), idx = impl_->view(indices), out = impl_->view(output);
        LfsHipError error{};
        check(impl_->api.gather_unary_f32(impl_->context, in, idx, static_cast<uint32_t>(indices.dtype), out,
                                          program.input_size, program.index_size, static_cast<uint32_t>(unary), &error),
              error);
    }
    void HipBackendOps::scatter(StorageRef output, StorageRef indices, StorageRef source,
                                const StridedLayout& output_layout, const StridedLayout& source_layout,
                                const IndexProgram& program, ExecContext execution) {
        queue(execution);
        require(program.dim >= 0 && size_t(program.dim) < output_layout.rank &&
                    program.scatter_mode == 0 && output.dtype == source.dtype &&
                    source_layout.rank == output_layout.rank &&
                    source_layout.element_count == program.total_elements,
                "Invalid HIP scatter descriptor");
        for (size_t axis = 0; axis < output_layout.rank; ++axis)
            require(axis == size_t(program.dim) || source_layout.dims[axis] == output_layout.dims[axis],
                    "HIP scatter source shape mismatch");
        if (!program.total_elements)
            return;
        size_t stride = 1;
        for (size_t axis = source_layout.rank; axis; --axis) {
            require(source_layout.strides[axis - 1] == stride, "HIP scatter source must be contiguous");
            require(source_layout.dims[axis - 1] == 0 || stride <= SIZE_MAX / source_layout.dims[axis - 1],
                    "HIP scatter source shape overflow");
            stride *= source_layout.dims[axis - 1];
        }
        IndexProgram update = program;
        update.index_size = source_layout.dims[program.dim];
        std::lock_guard lock(impl_->mutex);
        impl_->index_update(output, indices, source, output_layout, update, false);
    }
    void HipBackendOps::index_add(StorageRef output, StorageRef indices, StorageRef source, const StridedLayout& layout, const IndexProgram& program, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        impl_->index_update(output, indices, source, layout, program, true);
    }
    void HipBackendOps::index_fill(StorageRef output, StorageRef indices, const StridedLayout& layout, const IndexProgram& program, ScalarOperand value, ExecContext execution) {
        queue(execution);
        require(program.dim >= 0 && size_t(program.dim) < layout.rank && program.boundary_mode == 0,
                "Invalid HIP index fill dimension or boundary");
        const auto destination = native_layout(layout);
        const auto bits = typed_scalar_bits(output.dtype, value);
        std::lock_guard lock(impl_->mutex);
        const auto out = impl_->view(output), idx = impl_->view(indices);
        LfsHipError error{};
        const LfsHipIndexFill descriptor{sizeof(LfsHipIndexFill), uint32_t(output.dtype), uint32_t(indices.dtype), uint32_t(program.dim), program.index_size, bits, destination};
        check(impl_->api.index_fill(impl_->context, out, idx, &descriptor, &error), error);
    }
    void HipBackendOps::environment_composite(StorageRef rgb, StorageRef alpha, StorageRef environment,
                                              StorageRef output, const EnvironmentCompositeParams& params,
                                              ExecContext execution) {
        queue(execution);
        require(rgb.dtype == DataType::Float32 && alpha.dtype == DataType::Float32 &&
                    environment.dtype == DataType::Float32 && output.dtype == DataType::UInt8,
                "HIP environment composite storage dtype mismatch");
        LfsHipEnvironment native{};
        std::copy_n(params.rotation, 9, native.rotation);
        native.full_width = params.full_width;
        native.full_height = params.full_height;
        native.band_width = params.band_width;
        native.band_height = params.band_height;
        native.y_offset = params.y_offset;
        native.focal_x = params.focal_x;
        native.focal_y = params.focal_y;
        native.center_x = params.center_x;
        native.center_y = params.center_y;
        native.equirect_view = params.equirect_view;
        native.exposure_factor = params.exposure_factor;
        native.env_rotation_radians = params.env_rotation_radians;
        native.env_width = params.env_width;
        native.env_height = params.env_height;
        std::lock_guard lock(impl_->mutex);
        LfsHipError error{};
        check(impl_->api.environment_composite(impl_->context, impl_->view(rgb), impl_->view(alpha),
                                               impl_->view(environment), impl_->view(output), &native, &error),
              error);
    }
    void HipBackendOps::strided_scatter(StorageRef input, StorageRef output, const StridedLayout& layout, ExecContext execution) {
        queue(execution);
        require(input.dtype == output.dtype && dtype_size(input.dtype) > 0, "HIP strided scatter dtype mismatch");
        const auto native = native_layout(layout);
        std::lock_guard lock(impl_->mutex);
        const auto in = impl_->view(input), out = impl_->view(output);
        LfsHipError error{};
        check(impl_->api.strided_scatter(impl_->context, in, out, &native, static_cast<uint32_t>(dtype_size(input.dtype)), 0, &error), error);
    }
    void HipBackendOps::strided_scatter_immediate(StorageRef input, StorageRef output, const StridedLayout& layout, ExecContext execution) {
        // Both native variants pass layout metadata by value to the GPU kernel.
        strided_scatter(input, output, layout, execution);
    }
    void HipBackendOps::strided_scatter_int32_to_float32(StorageRef input, StorageRef output, const StridedLayout& layout, ExecContext execution) {
        queue(execution);
        require(input.dtype == DataType::Int32 && output.dtype == DataType::Float32, "HIP converted strided scatter requires int32/float32");
        const auto native = native_layout(layout);
        std::lock_guard lock(impl_->mutex);
        const auto in = impl_->view(input), out = impl_->view(output);
        LfsHipError error{};
        check(impl_->api.strided_scatter(impl_->context, in, out, &native, 4, 1, &error), error);
    }
    void HipBackendOps::uniform(StorageRef output, const RandomProgram& program, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        impl_->random(output, program, LFS_HIP_RANDOM_UNIFORM);
    }
    void HipBackendOps::bernoulli(StorageRef output, const RandomProgram& program, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        impl_->random(output, program, LFS_HIP_RANDOM_BERNOULLI);
    }
    void HipBackendOps::randint(StorageRef output, const RandomProgram& program, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        impl_->random(output, program, LFS_HIP_RANDOM_INTEGER);
    }
    void HipBackendOps::normal(StorageRef output, const RandomProgram& program, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        impl_->random(output, program, LFS_HIP_RANDOM_NORMAL);
    }
    void HipBackendOps::unary(const PointwiseProgram& p, StorageRef in, StorageRef out, size_t count, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        impl_->pointwise(p, in, {}, out, count, false);
    }
    void HipBackendOps::bias_add(StorageRef input, StorageRef bias, StorageRef output, int count, int channels, int spatial, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        impl_->bias(input, bias, output, count, channels, spatial, false);
    }
    void HipBackendOps::bias_relu(StorageRef input, StorageRef bias, StorageRef output, int count, int channels, int spatial, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        impl_->bias(input, bias, output, count, channels, spatial, true);
    }
    void HipBackendOps::relu(StorageRef input, StorageRef output, int count, ExecContext execution) {
        require(count >= 0, "HIP ReLU count must be nonnegative");
        if (!count)
            return;
        unary(PointwiseProgram{PointwiseOp::Relu, DataType::Float32, DataType::Float32, {}}, input, output, size_t(count), execution);
    }
    void HipBackendOps::binary(const PointwiseProgram& p, StorageRef a, StorageRef b, StorageRef out, size_t count, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        impl_->pointwise(p, a, b, out, count, true);
    }
    void HipBackendOps::scalar(const PointwiseProgram& p, StorageRef in, StorageRef out, size_t count, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        auto program = p;
        // In-place Tensor scalar methods lower binary functors with the scalar
        // operand stored separately, as in the existing native backend contract.
        switch (program.op) {
        case PointwiseOp::AddTensor: program.op = PointwiseOp::AddScalar; break;
        case PointwiseOp::SubTensor: program.op = PointwiseOp::SubScalar; break;
        case PointwiseOp::MulTensor: program.op = PointwiseOp::MulScalar; break;
        case PointwiseOp::DivTensor: program.op = PointwiseOp::DivScalar; break;
        default: break;
        }
        impl_->pointwise(program, in, {}, out, count, false);
    }
    void HipBackendOps::load_fill(StorageRef out, size_t count, ScalarOperand value, ExecContext execution) {
        queue(execution);
        if (out.dtype != DataType::Float32) {
            StridedLayout layout{};
            layout.rank = 1;
            layout.dims[0] = layout.element_count = count;
            layout.strides[0] = 1;
            fill_strided(out, layout, value, execution);
            return;
        }
        std::lock_guard lock(impl_->mutex);
        const auto dst = impl_->view(out, byte_count(count));
        LfsHipError error{};
        check(impl_->api.fill_f32(impl_->context, dst.buffer, dst.offset, count, scalar_value(value), &error), error);
    }
    void HipBackendOps::load_arange(StorageRef out, size_t count, ScalarOperand start, ScalarOperand step, ExecContext execution) {
        queue(execution);
        require(out.dtype == DataType::Float32, "HIP arange requires float32");
        std::lock_guard lock(impl_->mutex);
        const auto dst = impl_->view(out, byte_count(count));
        LfsHipError error{};
        check(impl_->api.arange_f32(impl_->context, dst, count, scalar_value(start), scalar_value(step), &error), error);
    }
    void HipBackendOps::copy_host_to_device(const CopyRequest& r) {
        queue(r.context);
        require(r.src.data || !r.bytes, "Missing HIP upload source");
        std::lock_guard lock(impl_->mutex);
        if (!r.bytes && impl_->empty_view(r.dst)) {
            if (r.synchronous)
                impl_->sync();
            return;
        }
        const auto dst = impl_->view(r.dst, r.bytes);
        LfsHipError error{};
        const auto* host = r.src.data ? static_cast<const char*>(r.src.data) + r.src.byte_offset : nullptr;
        check(impl_->api.upload(impl_->context, dst.buffer, dst.offset, host, r.bytes, &error), error);
    }
    void HipBackendOps::copy_device_to_host(const CopyRequest& r) {
        queue(r.context);
        require(r.dst.data || !r.bytes, "Missing HIP download destination");
        std::lock_guard lock(impl_->mutex);
        if (!r.bytes && impl_->empty_view(r.src)) {
            if (r.synchronous)
                impl_->sync();
            return;
        }
        const auto src = impl_->view(r.src, r.bytes);
        LfsHipError error{};
        auto* host = r.dst.data ? static_cast<char*>(r.dst.data) + r.dst.byte_offset : nullptr;
        check(impl_->api.download(impl_->context, src.buffer, src.offset, host, r.bytes, &error), error);
    }
    void HipBackendOps::copy_device_to_device(const CopyRequest& r) {
        queue(r.context);
        std::lock_guard lock(impl_->mutex);
        if (!r.bytes) {
            if (!impl_->empty_view(r.src))
                (void)impl_->view(r.src);
            if (!impl_->empty_view(r.dst))
                (void)impl_->view(r.dst);
            if (r.synchronous)
                impl_->sync();
            return;
        }
        const auto src = impl_->view(r.src, r.bytes);
        const auto dst = impl_->view(r.dst, r.bytes);
        LfsHipError error{};
        check(impl_->api.copy(impl_->context, src, dst, r.bytes, &error), error);
        if (r.synchronous)
            impl_->sync();
    }
    void HipBackendOps::memset(const FillRequest& r) {
        queue(r.context);
        std::lock_guard lock(impl_->mutex);
        if (!r.bytes && impl_->empty_view(r.dst)) {
            if (r.synchronous)
                impl_->sync();
            return;
        }
        const auto dst = impl_->view(r.dst, r.bytes);
        LfsHipError error{};
        check(impl_->api.memset(impl_->context, dst, r.bytes, r.value, &error), error);
        if (r.synchronous)
            impl_->sync();
    }
    void HipBackendOps::record_stream(StorageRef storage, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        if (impl_->empty_view(storage))
            return;
        (void)impl_->view(storage);
    }
    void HipBackendOps::rehome_stream(StorageRef storage, ExecContext execution) { record_stream(storage, execution); }
    std::unique_ptr<ReadbackBuffer> HipBackendOps::create_readback_buffer() {
        ensure_initialized();
        class Staging final : public ReadbackBuffer {
            std::shared_ptr<Impl> owner_;
            LfsHipReadback* handle_ = nullptr;
            std::unique_ptr<HipRetiredResource> retirement_ = std::make_unique<HipRetiredResource>();
            size_t pending_bytes_ = 0;

        public:
            explicit Staging(std::shared_ptr<Impl> owner) : owner_(std::move(owner)) {
                std::lock_guard lock(owner_->mutex);
                LfsHipError error{};
                check(owner_->api.readback_create(owner_->context, &handle_, &error), error);
            }
            ~Staging() override {
                std::lock_guard lock(owner_->mutex);
                owner_->runtime->release_readback(handle_, std::move(retirement_));
            }
            void enqueue_tensor(const Tensor& source, size_t offset, size_t bytes, BackendQueueTarget target) override {
                require(target.handle == nullptr, "HIP readback requires the native default queue");
                auto storage = storage_ref(source);
                storage.byte_offset += offset;
                enqueue(storage, bytes, {});
            }
            void enqueue(StorageRef source, size_t bytes, ExecContext execution) override {
                queue(execution);
                std::lock_guard lock(owner_->mutex);
                const auto view = !bytes && owner_->empty_view(source) ? LfsHipView{} : owner_->view(source, bytes);
                LfsHipError error{};
                pending_bytes_ = bytes;
                check(owner_->api.readback_enqueue(owner_->context, handle_, view.buffer, view.offset, bytes, &error), error);
            }
            void wait() override {
                std::lock_guard lock(owner_->mutex);
                LfsHipError error{};
                check(owner_->api.readback_wait(owner_->context, handle_, &error), error);
            }
            bool poll(void* destination) override {
                std::lock_guard lock(owner_->mutex);
                LfsHipError error{};
                int32_t ready = 0;
                // A null destination explicitly discards a native failed
                // snapshot. Completion queries have no payload and must not
                // perform that discard while merely checking readiness.
                void* target = destination;
                if (!target && !pending_bytes_)
                    target = &ready;
                check(owner_->api.readback_poll(owner_->context, handle_, target, pending_bytes_, &ready, &error), error);
                return ready != 0;
            }
        };
        return std::make_unique<Staging>(impl_);
    }
    std::unique_ptr<BackendUploadBuffer> HipBackendOps::create_upload_buffer() {
        ensure_initialized();
        class Upload final : public BackendUploadBuffer {
            std::shared_ptr<Impl> owner_;
            LfsHipReadback* staging_ = nullptr;
            std::unique_ptr<HipRetiredResource> retirement_ = std::make_unique<HipRetiredResource>();
            std::optional<LfsHipError> failure_;

        public:
            explicit Upload(std::shared_ptr<Impl> owner) : owner_(std::move(owner)) {
                std::lock_guard lock(owner_->mutex);
                LfsHipError error{};
                check(owner_->api.readback_create(owner_->context, &staging_, &error), error);
            }
            ~Upload() override {
                std::lock_guard lock(owner_->mutex);
                owner_->runtime->release_readback(staging_, std::move(retirement_));
            }
            void enqueue_native(StorageRef destination, const void* source, size_t bytes, ExecContext execution) {
                queue(execution);
                std::lock_guard lock(owner_->mutex);
                const auto view = !bytes && owner_->empty_view(destination) ? LfsHipView{} : owner_->view(destination, bytes);
                LfsHipError error{};
                check(owner_->api.staged_upload(owner_->context, staging_, view.buffer, view.offset, source, bytes, &error), error);
                failure_.reset();
            }
            BackendUploadSubmission enqueue(Tensor& destination, const Tensor& source, BackendQueueTarget target, bool) override {
                if (!target.current && target.handle)
                    throw std::invalid_argument("HIP uploads require the native default queue");
                destination.set_stream(nullptr);
                enqueue_native(storage_ref(destination), source.data_ptr(), source.bytes(), {});
                return {};
            }
            BackendUploadSubmission enqueue_bytes(Tensor& destination, std::span<const std::byte> source, BackendQueueTarget target, bool) override {
                if (!target.current && target.handle)
                    throw std::invalid_argument("HIP uploads require the native default queue");
                destination.set_stream(nullptr);
                enqueue_native(storage_ref(destination), source.data(), source.size(), {});
                return {};
            }
            bool abandon_after_failure() noexcept override { return true; }
            void settle_enqueue_failure() override {
                std::lock_guard lock(owner_->mutex);
                owner_->sync();
            }
            void wait() override {
                std::lock_guard lock(owner_->mutex);
                if (failure_)
                    check(1, *failure_);
                LfsHipError error{};
                check(owner_->api.readback_wait(owner_->context, staging_, &error), error);
            }
            bool poll() override {
                std::lock_guard lock(owner_->mutex);
                if (failure_)
                    check(1, *failure_);
                LfsHipError error{};
                int32_t ready = 0;
                // Upload snapshots retain the payload size, so a non-null
                // zero-sized destination cannot query them. Preserve errors
                // locally when the native null poll consumes the snapshot.
                const int status = owner_->api.readback_poll(owner_->context, staging_, nullptr, 0, &ready, &error);
                if (status)
                    failure_ = error;
                check(status, error);
                return ready != 0;
            }
        };
        return std::make_unique<Upload>(impl_);
    }
    std::unique_ptr<BackendReadbackSlot> HipBackendOps::create_readback_slot(size_t bytes) {
        ensure_initialized();
        class Slot final : public BackendReadbackSlot {
            std::shared_ptr<Impl> owner_;
            LfsHipReadback* staging_ = nullptr;
            std::unique_ptr<HipRetiredResource> retirement_ = std::make_unique<HipRetiredResource>();
            void* host_ = nullptr;
            size_t capacity_ = 0;
            bool pending_ = false;

            bool ready_locked() {
                if (!pending_)
                    return true;
                LfsHipError error{};
                int32_t ready = 0;
                // Use the owned destination so a failed poll stays latched;
                // nullptr would explicitly discard the failed snapshot.
                check(owner_->api.readback_poll(owner_->context, staging_, host_, capacity_, &ready, &error), error);
                if (ready)
                    pending_ = false;
                return ready != 0;
            }

        public:
            Slot(std::shared_ptr<Impl> owner, size_t bytes) : owner_(std::move(owner)), capacity_(bytes) {
                std::lock_guard lock(owner_->mutex);
                LfsHipError error{};
                check(owner_->api.readback_create(owner_->context, &staging_, &error), error);
                try {
                    uint64_t address = 0;
                    check(owner_->api.readback_reserve(owner_->context, staging_, bytes, &address, &error), error);
                    host_ = reinterpret_cast<void*>(static_cast<uintptr_t>(address));
                } catch (...) {
                    owner_->runtime->release_readback(staging_, std::move(retirement_));
                    throw;
                }
            }
            ~Slot() override {
                std::lock_guard lock(owner_->mutex);
                owner_->runtime->release_readback(staging_, std::move(retirement_));
            }
            void* data() const override { return host_; }
            bool begin_append(const Tensor&, size_t, size_t, size_t, ExecContext execution) override {
                queue(execution);
                return false;
            }
            bool prefers_recycled_host_staging() const noexcept override { return false; }
            void append(StorageRef source, size_t bytes, size_t destination_offset, ExecContext execution) override {
                queue(execution);
                std::lock_guard lock(owner_->mutex);
                const auto input = owner_->view(source, bytes);
                LfsHipError error{};
                pending_ = true;
                check(owner_->api.readback_append(owner_->context, staging_, input.buffer, input.offset, bytes, destination_offset, &error), error);
            }
            void release() override {
                std::lock_guard lock(owner_->mutex);
                require(ready_locked(), "Cannot release incomplete HIP packed readback slot");
            }
            bool ready() override {
                std::lock_guard lock(owner_->mutex);
                return ready_locked();
            }
            void wait() override {
                std::lock_guard lock(owner_->mutex);
                if (!pending_)
                    return;
                LfsHipError error{};
                check(owner_->api.readback_wait(owner_->context, staging_, &error), error);
                require(ready_locked(), "HIP packed readback is not ready after waiting");
            }
        };
        return std::make_unique<Slot>(impl_, bytes);
    }
    StorageRef HipBackendOps::borrow_native_range(void* pointer, size_t bytes) {
        std::lock_guard lock(impl_->mutex);
        require(impl_->context != nullptr && pointer != nullptr, "HIP arena view needs a live native allocation");
        const auto address = reinterpret_cast<uintptr_t>(pointer);
        for (const auto& [base, allocation] : impl_->allocations) {
            const auto begin = reinterpret_cast<uintptr_t>(base);
            if (address < begin || address - begin > allocation.bytes || bytes > allocation.bytes - (address - begin))
                continue;
            return {GpuBackend::HIP, base, size_t(address - begin), DataType::Float32, allocation.metadata.get()};
        }
        throw std::invalid_argument("HIP arena view does not belong to a live native Tensor allocation");
    }

    void HipBackendOps::filter_points(StorageRef mask, const PointFilterProgram& p, ExecContext execution) {
        queue(execution);
        LfsHipPointFilter native{};
        native.struct_bytes = sizeof(native);
        native.count = p.count;
        native.transform_count = p.transform_count;
        native.allowed_count = p.allowed_count;
        native.flags = p.flags;
        native.model = static_cast<uint32_t>(p.window.projection.model);
        const auto& projection = p.window.projection;
        std::copy(projection.rotation.begin(), projection.rotation.end(), native.window);
        std::copy(projection.translation.begin(), projection.translation.end(), native.window + 9);
        const std::array<float, 13> tail{projection.focal_x, projection.focal_y, projection.center_x, projection.center_y,
                                         projection.ortho_scale, float(projection.width), float(projection.height), p.window.near_depth, p.window.far_depth,
                                         p.window.scale_x, p.window.scale_y, p.window.offset_x, p.window.offset_y};
        std::copy(tail.begin(), tail.end(), native.window + 12);
        std::lock_guard lock(impl_->mutex);
        for (size_t i = 0; i < p.inputs.size(); ++i)
            if (p.inputs[i])
                native.inputs[i] = impl_->view(*p.inputs[i], 0);
        LfsHipError error{};
        check(impl_->api.filter_points(impl_->context, impl_->view(mask, p.count), &native, &error), error);
    }
    void HipBackendOps::update_labels(StorageRef output, StorageRef selected, const LabelUpdateProgram& p, ExecContext execution) {
        queue(execution);
        require(p.count <= INT32_MAX && p.output_count <= INT32_MAX && p.allowed_count <= INT32_MAX, "HIP label count exceeds int32");
        LfsHipLabelUpdate native{};
        native.struct_bytes = sizeof(native);
        native.count = uint32_t(p.count);
        native.output_count = uint32_t(p.output_count);
        native.allowed_count = uint32_t(p.allowed_count);
        native.label = p.label;
        native.mode = p.mode;
        std::lock_guard lock(impl_->mutex);
        if (p.existing)
            native.existing = impl_->view(*p.existing, 0);
        if (p.locked)
            native.locked = impl_->view(*p.locked, 0);
        if (p.indices)
            native.indices = impl_->view(*p.indices, 0);
        if (p.categories)
            native.categories = impl_->view(*p.categories, 0);
        if (p.allowed)
            native.allowed = impl_->view(*p.allowed, 0);
        LfsHipError error{};
        check(impl_->api.update_labels(impl_->context, impl_->view(output, p.output_count),
                                       impl_->view(selected, p.count), &native, &error),
              error);
    }
    void HipBackendOps::and_live(StorageRef mask, StorageRef live, const MaskProgram& p, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        LfsHipError error{};
        check(impl_->api.and_live(impl_->context, impl_->view(mask, p.count), impl_->view(live, p.count), p.count, &error), error);
    }
    void HipBackendOps::histogram_u8(StorageRef values, StorageRef counts, size_t count, ExecContext execution) {
        queue(execution);
        require((values.dtype == DataType::UInt8 || values.dtype == DataType::Bool) && counts.dtype == DataType::Int32,
                "HIP label histogram requires byte input and Int32 bins");
        std::lock_guard lock(impl_->mutex);
        LfsHipError error{};
        check(impl_->api.histogram_u8(impl_->context, impl_->view(values, count), impl_->view(counts, 256 * sizeof(int32_t)), count, &error), error);
    }
    void HipBackendOps::project_points(StorageRef points, StorageRef output, size_t count, const PointProjection& p,
                                       const StorageRef* transforms, size_t transform_count, const StorageRef* indices, const StorageRef* visibility,
                                       size_t visibility_count, ExecContext execution) {
        queue(execution);
        require(points.dtype == DataType::Float32 && output.dtype == DataType::Float32 && count <= INT32_MAX &&
                    transform_count <= INT32_MAX && visibility_count <= INT32_MAX,
                "Invalid HIP point projection storage or count");
        require((!transforms || transforms->dtype == DataType::Float32) && (!indices || indices->dtype == DataType::Int32) &&
                    (!visibility || visibility->dtype == DataType::Bool || visibility->dtype == DataType::UInt8),
                "Invalid HIP point projection auxiliary dtype");
        LfsHipPointProject native{};
        native.struct_bytes = sizeof(native);
        native.count = uint32_t(count);
        native.model = uint32_t(p.model);
        native.width = uint32_t(p.width);
        native.height = uint32_t(p.height);
        native.transform_count = uint32_t(transform_count);
        native.visibility_count = uint32_t(visibility_count);
        std::copy(p.rotation.begin(), p.rotation.end(), native.frame);
        std::copy(p.translation.begin(), p.translation.end(), native.frame + 9);
        const std::array tail{p.focal_x, p.focal_y, p.center_x, p.center_y, p.ortho_scale, p.invalid_value};
        std::copy(tail.begin(), tail.end(), native.frame + 12);
        std::lock_guard lock(impl_->mutex);
        if (transforms)
            native.transforms = impl_->view(*transforms, 0);
        if (indices)
            native.indices = impl_->view(*indices, 0);
        if (visibility)
            native.visibility = impl_->view(*visibility, 0);
        LfsHipError error{};
        check(impl_->api.project_points(impl_->context, impl_->view(points, checked_product(count, 3 * sizeof(float))),
                                        impl_->view(output, checked_product(count, 2 * sizeof(float))), &native, &error),
              error);
    }
    void HipBackendOps::mark_points_2d(StorageRef mask, StorageRef points, size_t count, const PointRegion2D& p,
                                       const StorageRef* geometry, size_t geometry_count, ExecContext execution) {
        queue(execution);
        require((mask.dtype == DataType::UInt8 || mask.dtype == DataType::Bool) && points.dtype == DataType::Float32 &&
                    (!geometry || geometry->dtype == DataType::Float32) && count <= INT32_MAX && geometry_count <= INT32_MAX,
                "Invalid HIP point region storage or count");
        const LfsHipPointRegion native{sizeof(LfsHipPointRegion), uint32_t(count), uint32_t(p.kind), uint32_t(geometry_count),
                                       p.x0, p.y0, p.x1, p.y1, p.radius, p.minimum_coordinate};
        std::lock_guard lock(impl_->mutex);
        LfsHipError error{};
        check(impl_->api.mark_points_2d(impl_->context, impl_->view(mask, count), impl_->view(points, checked_product(count, 2 * sizeof(float))),
                                        geometry ? impl_->view(*geometry, checked_product(geometry_count, 2 * sizeof(float))) : LfsHipView{}, &native, &error),
              error);
    }
    uint64_t HipBackendOps::storage_address(StorageRef storage, size_t bytes) {
        std::lock_guard lock(impl_->mutex);
        (void)impl_->view(storage, bytes);
        return reinterpret_cast<uintptr_t>(storage.data) + storage.byte_offset;
    }
    bool HipBackendOps::queue_ready(ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        LfsHipError error{};
        int32_t ready = 0;
        check(impl_->api.context_ready(impl_->context, &ready, &error), error);
        return ready != 0;
    }
    void HipBackendOps::enqueue_host_callback(void (*callback)(void*), void* user, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        LfsHipError error{};
        check(impl_->api.host_callback(impl_->context, callback, user, &error), error);
    }
    BackendQueueLease HipBackendOps::execution_lease(ExecContext execution) {
        ensure_initialized();
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        BackendQueueLease lease{impl_};
        LfsHipError error{};
        check(impl_->api.context_queue(impl_->context, &lease.native_queue, &lease.device, &error), error);
        return lease;
    }
    std::unique_ptr<BackendTimestamps> HipBackendOps::create_timestamps(size_t count) {
        ensure_initialized();
        class Timestamps final : public BackendTimestamps {
            std::shared_ptr<Impl> owner_;
            LfsHipTiming* pool_ = nullptr;
            std::unique_ptr<HipRetiredResource> retirement_ = std::make_unique<HipRetiredResource>();
            std::vector<bool> recorded_;
            decltype(&lfs_hip_timing_mark) mark_;
            decltype(&lfs_hip_timing_wait) wait_;
            decltype(&lfs_hip_timing_elapsed) elapsed_;

        public:
            Timestamps(std::shared_ptr<Impl> owner, size_t size)
                : owner_(std::move(owner)), recorded_(size, false) {
                std::lock_guard lock(owner_->mutex);
                // Profiling uses the same validated, versioned dispatch table.
                const auto create = owner_->api.timing_create;
                mark_ = owner_->api.timing_mark;
                wait_ = owner_->api.timing_wait;
                elapsed_ = owner_->api.timing_elapsed;
                LfsHipError error{};
                check(create(owner_->context, size, &pool_, &error), error);
            }
            ~Timestamps() override {
                std::lock_guard lock(owner_->mutex);
                owner_->runtime->release_timing(pool_, std::move(retirement_));
            }
            bool mark(size_t index, void* target) override {
                if (target || index >= recorded_.size())
                    return false;
                std::lock_guard lock(owner_->mutex);
                LfsHipError error{};
                check(mark_(owner_->context, pool_, index, &error), error);
                recorded_[index] = true;
                return true;
            }
            bool wait_event(size_t index) override {
                std::lock_guard lock(owner_->mutex);
                if (index >= recorded_.size() || !recorded_[index])
                    return false;
                LfsHipError error{};
                check(wait_(owner_->context, pool_, index, &error), error);
                return true;
            }
            bool wait_queue(void* target) override {
                if (target)
                    return false;
                std::lock_guard lock(owner_->mutex);
                owner_->sync();
                return true;
            }
            std::optional<float> milliseconds(size_t begin, size_t end) const override {
                std::lock_guard lock(owner_->mutex);
                if (begin >= recorded_.size() || end >= recorded_.size())
                    return std::nullopt;
                LfsHipError error{};
                float value = 0.f;
                int32_t ready = 0;
                check(elapsed_(owner_->context, pool_, begin, end, &value, &ready, &error), error);
                return ready ? std::optional<float>(value) : std::nullopt;
            }
        };
        try {
            require(count > 0 && count <= 65536, "Invalid HIP timestamp pool size");
            return std::make_unique<Timestamps>(impl_, count);
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): Optional profiling must not disrupt execution; report unavailability.
            std::fprintf(stderr, "[warn] HIP GPU timestamps unavailable: %s\n", error.what());
            return {};
        }
    }

    std::unique_ptr<BackendFence> HipBackendOps::create_fence(std::optional<void*> imported) {
        ensure_initialized();
        if (imported)
            return {};
        class Fence final : public BackendFence {
            std::shared_ptr<Impl> owner_;
            LfsHipReadback* marker_ = nullptr;
            std::unique_ptr<HipRetiredResource> retirement_ = std::make_unique<HipRetiredResource>();
            mutable bool recorded_ = false, complete_ = true;

        public:
            explicit Fence(std::shared_ptr<Impl> owner) : owner_(std::move(owner)) {
                std::lock_guard lock(owner_->mutex);
                LfsHipError error{};
                check(owner_->api.readback_create(owner_->context, &marker_, &error), error);
            }
            ~Fence() override {
                std::lock_guard lock(owner_->mutex);
                owner_->runtime->release_readback(marker_, std::move(retirement_));
            }
            void record(ExecContext execution) override {
                queue(execution);
                std::lock_guard lock(owner_->mutex);
                recorded_ = true;
                complete_ = false;
                LfsHipError error{};
                check(owner_->api.marker_record(owner_->context, marker_, &error), error);
            }
            void wait_on(ExecContext execution) const override {
                queue(execution);
                // Only the same native queue is admitted: later submissions
                // already follow this marker, including after re-recording.
            }
            void wait() const override {
                std::lock_guard lock(owner_->mutex);
                if (!recorded_ || complete_)
                    return;
                LfsHipError error{};
                check(owner_->api.readback_wait(owner_->context, marker_, &error), error);
                int32_t ready = 0;
                check(owner_->api.readback_poll(owner_->context, marker_, nullptr, 0, &ready, &error), error);
                require(ready != 0, "HIP fence not ready after successful wait");
                complete_ = true;
            }
            bool ready() const override {
                std::lock_guard lock(owner_->mutex);
                if (!recorded_ || complete_)
                    return true;
                LfsHipError error{};
                int32_t ready = 0;
                // Markers have zero bytes; a non-null, zero-sized destination
                // keeps a failed snapshot latched across readiness queries.
                check(owner_->api.readback_poll(owner_->context, marker_, &ready, 0, &ready, &error), error);
                complete_ = ready != 0;
                return complete_;
            }
        };
        return std::make_unique<Fence>(impl_);
    }
    void HipBackendOps::wait_for(SyncToken token) {
        // The adapter admits exactly one queue. A token is scoped to this
        // context generation; native CUDA streams or foreign tokens are errors.
        require(token.backend == GpuBackend::HIP && token.value == impl_->identity && token.native == 0,
                "HIP synchronization received a foreign or expired context token");
        synchronize_stream({});
    }
    SyncToken HipBackendOps::bridge(ExecContext producer, ExecContext consumer) {
        queue(producer);
        queue(consumer);
        return {GpuBackend::HIP, impl_->identity, 0};
    }
    void HipBackendOps::release_stream(ExecContext execution) { synchronize_stream(execution); }
    void HipBackendOps::synchronize_stream(ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        impl_->sync();
    }
    bool HipBackendOps::drain_queue_for_destruction() noexcept {
        try {
            synchronize_stream({});
        } catch (const HipBackendError& error) {
            if (error.checked_execution_error()) {
                LOG_ERROR("HIP tensor queue completed with an execution error: {}", error.what());
                return true;
            }
            LOG_ERROR("HIP tensor queue retained after failed synchronization: {}", error.what());
            return false;
        } catch (const std::exception& error) {
            LOG_ERROR("HIP tensor queue retained after failed synchronization: {}", error.what());
            return false;
        }
        return true;
    }
    ExpressionCacheStats HipBackendOps::expression_cache_stats() {
#ifdef LFS_HIP_EXPRESSION_JIT
        std::lock_guard lock(impl_->mutex);
        return impl_->expressions.stats();
#else
        return {};
#endif
    }
    void HipBackendOps::synchronize_device() { synchronize_stream({}); }
    void HipBackendOps::device_barrier() { synchronize_stream({}); }
    void HipBackendOps::trim() {
        std::lock_guard lock(impl_->mutex);
        impl_->trim_cache();
        if (impl_->context) {
            LfsHipError error{};
            check(impl_->api.trim_workspace(impl_->context, &error), error);
            impl_->workspace_bytes -= impl_->algorithm_workspace_bytes;
            impl_->algorithm_workspace_bytes = 0;
        }
    }
    void HipBackendOps::trim_if_reserved_unused_exceeds(size_t threshold) {
        std::lock_guard lock(impl_->mutex);
        if (impl_->cached_bytes > threshold || impl_->algorithm_workspace_bytes > threshold - impl_->cached_bytes) {
            impl_->trim_cache();
            if (impl_->context) {
                LfsHipError error{};
                check(impl_->api.trim_workspace(impl_->context, &error), error);
                impl_->workspace_bytes -= impl_->algorithm_workspace_bytes;
                impl_->algorithm_workspace_bytes = 0;
            }
        }
    }
    void HipBackendOps::set_allocation_iteration(int) {}
    void HipBackendOps::record_tensor_allocation(StorageRef storage, const StridedLayout&, size_t bytes) {
        std::lock_guard lock(impl_->mutex);
        (void)impl_->view(storage, bytes);
    }
    MemoryInfo HipBackendOps::stats() {
        std::lock_guard lock(impl_->mutex);
        MemoryInfo info;
        info.device_id = impl_->ordinal;
        if (!impl_->context)
            return info;
        LfsHipMemoryInfo memory{};
        LfsHipError error{};
        check(impl_->api.memory_info(impl_->context, &memory, &error), error);
        impl_->reserved_peak = std::max<size_t>(impl_->reserved_peak, memory.owned_bytes);
        info.total_bytes = memory.total_bytes;
        info.free_bytes = memory.free_bytes;
        info.allocated_bytes = impl_->live + memory.workspace_bytes;
        info.pool_used_current = info.allocated_bytes;
        info.pool_reserved_current = memory.owned_bytes;
        info.pool_reserved_high = impl_->reserved_peak;
        impl_->peak = std::max(impl_->peak, info.allocated_bytes);
        info.pool_used_high = impl_->peak;
        return info;
    }
    void HipBackendOps::shutdown() {
        std::lock_guard lock(impl_->mutex);
        if (!impl_->allocations.empty() && std::getenv("LFS_HIP_VALIDATION_ALLOCATION_REPORT")) {
            std::fprintf(stderr, "[diagnostic] HIP shutdown: %zu live allocations, %zu live bytes\n", impl_->allocations.size(), impl_->live);
            unsigned shown = 0;
            for (const auto& [address, allocation] : impl_->allocations) {
                if (shown++ == 32)
                    break;
                std::fprintf(stderr, "[diagnostic] HIP retained allocation: bytes=%zu kind=%s\n", allocation.bytes, allocation.metadata ? allocation.metadata->external_kind.c_str() : "unknown");
                if (!allocation.diagnostic_origin.empty())
                    std::fprintf(stderr, "%s\n", allocation.diagnostic_origin.c_str());
            }
        }
        // A noexcept Tensor deleter cannot propagate a driver/host-allocation
        // failure. Retry only entries whose last Tensor lease was retired;
        // genuinely live Tensor storage remains protected by the check below.
        for (auto it = impl_->allocations.begin(); it != impl_->allocations.end();) {
            if (!it->second.retired) {
                ++it;
                continue;
            }
            LfsHipError error{};
            check(impl_->api.buffer_release(impl_->context, it->second.handle, &error), error);
            impl_->live -= it->second.bytes;
            it = impl_->allocations.erase(it);
        }
        require(impl_->allocations.empty(), "Cannot shut down HIP Tensor context with live allocations");
        require(impl_.use_count() == 1, "Cannot shut down HIP Tensor context with live queues, transfers or fences");
        if (!impl_->context)
            return;
#ifdef LFS_HIP_EXPRESSION_JIT
        impl_->expressions.clear();
#endif
        // Release cached handles individually before native close can partially
        // destroy its buffer registry and then fail on a later resource.
        impl_->trim_cache();
        impl_->runtime->close();
        impl_->context = nullptr;
        impl_->active.store(false, std::memory_order_release);
        impl_->cached.clear();
        impl_->cached_bytes = 0;
        if (impl_->counted) {
            live_contexts.fetch_sub(1, std::memory_order_relaxed);
            impl_->counted = false;
        }
    }
    PointerClass HipBackendOps::classify_pointer(const void* pointer) {
        std::lock_guard lock(impl_->mutex);
        const auto address = reinterpret_cast<uintptr_t>(pointer);
        for (const auto& [base, allocation] : impl_->allocations) {
            const auto begin = reinterpret_cast<uintptr_t>(base);
            if (address >= begin && address - begin < allocation.bytes)
                return PointerClass::Device;
        }
        return PointerClass::Unknown;
    }
    bool HipBackendOps::stream_is_capturing(ExecContext execution) {
        queue(execution);
        return false;
    }
    float HipBackendOps::sum_scalar(StorageRef input, size_t count, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        return impl_->reduce(input, count, LFS_HIP_REDUCE_SUM);
    }
    float HipBackendOps::mean_scalar(StorageRef input, size_t count, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        return impl_->reduce(input, count, LFS_HIP_REDUCE_MEAN);
    }
    float HipBackendOps::min_scalar(StorageRef input, size_t count, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        return impl_->reduce(input, count, LFS_HIP_REDUCE_MIN);
    }
    float HipBackendOps::max_scalar(StorageRef input, size_t count, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        return impl_->reduce(input, count, LFS_HIP_REDUCE_MAX);
    }
    bool HipBackendOps::has_nan(StorageRef input, size_t count, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        return impl_->reduce(input, count, LFS_HIP_REDUCE_HAS_NAN);
    }
    bool HipBackendOps::has_inf(StorageRef input, size_t count, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        return impl_->reduce(input, count, LFS_HIP_REDUCE_HAS_INF);
    }
    void HipBackendOps::fused_pointwise_chain(StorageRef input, StorageRef output, size_t count,
                                              const tensor_ops::FusedPointwiseOpChain& source,
                                              std::span<const StorageRef> rhs_storages, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        require(input.dtype == DataType::Float32 && output.dtype == DataType::Float32 && source.num_ops > 0 && source.num_ops <= 16,
                "Invalid HIP fused pointwise chain");
        const auto src = impl_->view(input, byte_count(count)), dst = impl_->view(output, byte_count(count));
        LfsHipChain chain{};
        chain.count = source.num_ops;
        for (unsigned i = 0; i < chain.count; ++i) {
            chain.ops[i].op = source.ops[i].kind;
            chain.ops[i].scalar = source.ops[i].scalar;
            if (source.ops[i].rhs) {
                const auto address = reinterpret_cast<uintptr_t>(source.ops[i].rhs);
                bool found = false;
                for (const auto& storage : rhs_storages) {
                    if (storage.dtype == DataType::Float32 && reinterpret_cast<uintptr_t>(storage.data) + storage.byte_offset == address) {
                        chain.ops[i].rhs = impl_->view(storage, byte_count(count));
                        found = true;
                        break;
                    }
                }
                require(found, "HIP fused chain rhs is not backed by its storage lease");
            }
        }
        LfsHipError error{};
        check(impl_->api.chain_f32(impl_->context, src, dst, count, &chain, &error), error);
    }
    void HipBackendOps::fused_transform_reduce(StorageRef input, StorageRef output, size_t count,
                                               const tensor_ops::FusedPointwiseOpChain& chain, const ReduceProgram& program,
                                               std::span<const StorageRef> rhs, ExecContext execution) {
        fused_segmented_transform_reduce(input, output, 1, count, chain, program, rhs, execution);
    }
    void HipBackendOps::fused_segmented_transform_reduce(StorageRef input, StorageRef output, size_t segments, size_t size,
                                                         const tensor_ops::FusedPointwiseOpChain& source, const ReduceProgram& program,
                                                         std::span<const StorageRef> rhs_storages, ExecContext execution) {
        queue(execution);
        require(!segments || size <= SIZE_MAX / segments, "HIP fused reduction extent overflow");
        require(input.dtype == DataType::Float32 && output.dtype == DataType::Float32 && program.result_dtype == DataType::Float32 &&
                    source.num_ops > 0 && source.num_ops <= 16,
                "HIP fused reductions require Float32 and a valid chain");
        uint32_t operation;
        switch (program.op) {
        case ReduceOp::Sum: operation = LFS_HIP_REDUCE_SUM; break;
        case ReduceOp::Mean: operation = LFS_HIP_REDUCE_MEAN; break;
        case ReduceOp::Min: operation = LFS_HIP_REDUCE_MIN; break;
        case ReduceOp::Max: operation = LFS_HIP_REDUCE_MAX; break;
        default: throw std::invalid_argument("Unsupported HIP fused reduction operation");
        }
        if (!segments || !size)
            return;
        const size_t bytes = byte_count(segments * size);
        std::lock_guard lock(impl_->mutex);
        const auto src = impl_->view(input, bytes), dst = impl_->view(output, byte_count(segments));
        LfsHipChain chain{};
        chain.count = source.num_ops;
        for (unsigned i = 0; i < chain.count; ++i) {
            chain.ops[i].op = source.ops[i].kind;
            chain.ops[i].scalar = source.ops[i].scalar;
            if (source.ops[i].rhs) {
                const auto address = reinterpret_cast<uintptr_t>(source.ops[i].rhs);
                bool found = false;
                for (const auto& storage : rhs_storages) {
                    if (storage.dtype == DataType::Float32 && reinterpret_cast<uintptr_t>(storage.data) + storage.byte_offset == address) {
                        chain.ops[i].rhs = impl_->view(storage, bytes);
                        found = true;
                        break;
                    }
                }
                require(found, "HIP fused reduction rhs is not backed by its storage lease");
            }
        }
        LfsHipError error{};
        check(impl_->api.chain_reduce_f32(impl_->context, src, dst, segments, size, &chain, operation, &error), error);
        if (segments < 64 && size > 4096)
            impl_->observed_workspace();
        impl_->peak = std::max(impl_->peak, impl_->live + impl_->workspace_bytes);
        impl_->reserved_peak = std::max(impl_->reserved_peak, impl_->live + impl_->cached_bytes + impl_->workspace_bytes);
    }
    void HipBackendOps::where(StorageRef condition, StorageRef x, StorageRef y, StorageRef output,
                              const StridedLayout& condition_layout, const StridedLayout& x_layout,
                              const StridedLayout& y_layout, const StridedLayout& output_layout, ExecContext execution) {
        queue(execution);
        require(condition.dtype == DataType::Bool && x.dtype == y.dtype && x.dtype == output.dtype, "HIP where dtype mismatch");
        if (!output_layout.element_count)
            return;
        LfsHipWhere p{};
        p.struct_bytes = sizeof(p);
        p.dtype = static_cast<uint32_t>(output.dtype);
        p.condition = native_layout(condition_layout);
        p.x = native_layout(x_layout);
        p.y = native_layout(y_layout);
        p.output = native_layout(output_layout);
        std::lock_guard lock(impl_->mutex);
        const auto mask = impl_->view(condition), a = impl_->view(x), b = impl_->view(y), out = impl_->view(output);
        LfsHipError error{};
        check(impl_->api.where_layout(impl_->context, mask, a, b, out, &p, &error), error);
    }
    void HipBackendOps::masked_fill(StorageRef output, StorageRef mask, const MaskProgram& program, ExecContext execution) {
        queue(execution);
        require(mask.dtype == DataType::Bool, "HIP masked fill requires a Bool predicate");
        if (!program.count)
            return;
        const auto bytes = checked_product(program.count, dtype_size(output.dtype));
        std::lock_guard lock(impl_->mutex);
        const auto m = impl_->view(mask, program.count), out = impl_->view(output, bytes);
        LfsHipError error{};
        check(impl_->api.masked_fill(impl_->context, m, out, program.count, static_cast<uint32_t>(output.dtype), scalar_value(program.value), &error), error);
    }
    void HipBackendOps::where_scalar(Tensor& destination, const Tensor& condition, float value, const Tensor& input) {
        pin_operands({&destination, &condition, &input});
        const auto mask = storage_ref(condition), source = storage_ref(input), output = storage_ref(destination);
        const auto count = destination.numel();
        const ExecContext execution{};
        queue(execution);
        require(mask.dtype == DataType::Bool && source.dtype == output.dtype &&
                    (source.dtype == DataType::Float32 || source.dtype == DataType::Float16),
                "Invalid HIP renderer selection operands");
        const size_t width = source.dtype == DataType::Float32 ? 4 : 2;
        require(count <= SIZE_MAX / width, "HIP renderer selection extent overflow");
        if (!count)
            return;
        std::lock_guard lock(impl_->mutex);
        const auto m = impl_->view(mask, count), src = impl_->view(source, count * width), dst = impl_->view(output, count * width);
        LfsHipError error{};
        check(impl_->api.where_scalar(impl_->context, m, src, dst, count,
                                      source.dtype == DataType::Float32 ? LFS_HIP_F32 : LFS_HIP_F16, value, &error),
              error);
    }
    void HipBackendOps::launch_gemm(StorageRef lhs, StorageRef rhs, StorageRef bias, StorageRef output, const GemmProgram& p, bool transpose, bool relu, ExecContext execution) {
        queue(execution);
        require(lhs.dtype == DataType::Float32 && rhs.dtype == DataType::Float32 && output.dtype == DataType::Float32 &&
                    (!bias.data || bias.dtype == DataType::Float32),
                "HIP GEMM requires Float32 storage");
        if (!p.batch || !p.m || !p.n)
            return;
        const auto lhs_bytes = byte_count(checked_product(p.batch, checked_product(p.m, p.k)));
        const auto rhs_bytes = byte_count(checked_product(p.batch, checked_product(p.k, p.n)));
        const auto out_bytes = byte_count(checked_product(p.batch, checked_product(p.m, p.n)));
        std::lock_guard lock(impl_->mutex);
        const auto a = lhs_bytes ? impl_->view(lhs, lhs_bytes) : LfsHipView{};
        const auto b = rhs_bytes ? impl_->view(rhs, rhs_bytes) : LfsHipView{};
        const auto c = impl_->view(output, out_bytes);
        const auto d = bias.data ? impl_->view(bias, byte_count(p.n)) : LfsHipView{};
        LfsHipGemm descriptor{sizeof(LfsHipGemm), uint32_t(transpose), uint32_t(relu), p.batch, p.m, p.n, p.k};
        LfsHipError error{};
        check(impl_->api.gemm_f32(impl_->context, a, b, d, c, &descriptor, &error), error);
    }
    void HipBackendOps::sgemm(StorageRef a, StorageRef b, StorageRef c, const GemmProgram& p, ExecContext execution) { launch_gemm(a, b, {}, c, p, false, false, execution); }
    void HipBackendOps::sgemm_tn(StorageRef a, StorageRef b, StorageRef c, const GemmProgram& p, ExecContext execution) { launch_gemm(a, b, {}, c, p, true, false, execution); }
    void HipBackendOps::sgemm_batched(StorageRef a, StorageRef b, StorageRef c, const GemmProgram& p, ExecContext execution) { launch_gemm(a, b, {}, c, p, false, false, execution); }
    void HipBackendOps::sgemm_bias_relu(StorageRef a, StorageRef b, StorageRef bias, StorageRef c, const GemmProgram& p, ExecContext execution) { launch_gemm(a, b, bias, c, p, false, true, execution); }
    void HipBackendOps::inference(StorageRef input, StorageRef output, const InferenceProgram& program, ExecContext execution) {
        queue(execution);
        require((output.dtype == DataType::Float32 || output.dtype == DataType::Float16) && (program.kernel == InferenceKernel::Grid || input.dtype == output.dtype), "HIP inference requires matching Float32 or Float16");
        const size_t width = output.dtype == DataType::Float32 ? 4 : 2;
        const auto& g = program.geometry;
        if (!program.count)
            return;
        if (program.kernel != InferenceKernel::Pool) {
            require(program.count <= INT_MAX, "HIP inference count exceeds int32");
            if (!program.count)
                return;
            LfsHipInference call{sizeof(LfsHipInference), uint32_t(program.kernel), {0, 0, uint32_t(program.count), 0, g.channels, g.height, g.width, g.out_height, g.out_width, g.kernel_h, g.kernel_w, g.stride_h, g.stride_w, g.pad_h, g.pad_w, g.dilation_h, g.dilation_w, g.offset, g.columns, g.mode, g.coord, g.include_pad, g.u0, g.u1, g.v0, g.v1}};
            std::lock_guard lock(impl_->mutex);
            const auto src = program.kernel == InferenceKernel::Grid ? LfsHipView{} : impl_->view(input);
            const auto dst = impl_->view(output, program.count * width);
            LfsHipError error{};
            check(impl_->api.inference_typed(impl_->context, src, dst, &call, uint32_t(output.dtype), &error), error);
            return;
        }
        require(g.mode == 0 || g.mode == 1, "Invalid HIP inference pooling mode");
        require(g.include_pad == 0 || g.include_pad == 1, "Invalid HIP inference pooling padding flag");
        LfsHipPool p{program.count, g.height, g.width, g.out_height, g.out_width, g.kernel_h, g.kernel_w,
                     g.stride_h, g.stride_w, g.pad_h, g.pad_w, uint32_t(g.mode), uint32_t(g.include_pad)};
        require(program.count <= SIZE_MAX / sizeof(float), "HIP pooling output size overflows");
        std::lock_guard lock(impl_->mutex);
        const auto src = impl_->view(input), dst = impl_->view(output, program.count * width);
        LfsHipError error{};
        check(impl_->api.pool_typed(impl_->context, src, dst, &p, uint32_t(output.dtype), &error), error);
    }
    void HipBackendOps::max_pool2d(StorageRef input, StorageRef output, const PoolProgram& p, ExecContext execution) {
        require(p.batch >= 0 && p.channels >= 0 && p.input_height > 0 && p.input_width > 0 && p.output_height > 0 && p.output_width > 0 && p.kernel_size > 0 && p.stride > 0 && p.padding >= 0, "Invalid HIP max-pooling geometry");
        const size_t count = checked_product(checked_product(size_t(p.batch), size_t(p.channels)), checked_product(size_t(p.output_height), size_t(p.output_width)));
        InferenceGeometry g{};
        g.channels = p.channels;
        g.height = p.input_height;
        g.width = p.input_width;
        g.out_height = p.output_height;
        g.out_width = p.output_width;
        g.kernel_h = g.kernel_w = p.kernel_size;
        g.stride_h = g.stride_w = p.stride;
        g.pad_h = g.pad_w = p.padding;
        g.mode = 0;
        inference(input, output, InferenceProgram{InferenceKernel::Pool, count, g}, execution);
    }
    void HipBackendOps::adaptive_avg_pool2d(StorageRef input, StorageRef output, const PoolProgram& p, ExecContext execution) {
        queue(execution);
        require(p.batch >= 0 && p.channels >= 0 && p.input_height > 0 && p.input_width > 0 && p.output_height > 0 && p.output_width > 0 && (input.dtype == DataType::Float32 || input.dtype == DataType::Float16) && output.dtype == input.dtype, "Invalid HIP adaptive pooling geometry/dtype");
        const size_t count = checked_product(checked_product(size_t(p.batch), size_t(p.channels)), checked_product(size_t(p.output_height), size_t(p.output_width)));
        const size_t width = input.dtype == DataType::Float32 ? 4 : 2;
        if (!count)
            return;
        require(count <= SIZE_MAX / width, "HIP adaptive pooling output overflows");
        LfsHipPool native{count, p.input_height, p.input_width, p.output_height, p.output_width, 1, 1, 1, 1, 0, 0, 2, 0};
        std::lock_guard lock(impl_->mutex);
        const auto in = impl_->view(input), out = impl_->view(output, count * width);
        LfsHipError error{};
        check(impl_->api.pool_typed(impl_->context, in, out, &native, uint32_t(output.dtype), &error), error);
    }
    void HipBackendOps::strided_copy(StorageRef input, StorageRef output, const StridedLayout& layout, ExecContext execution) {
        queue(execution);
        require(input.dtype == output.dtype, "HIP strided copy requires equal dtypes");
        std::lock_guard lock(impl_->mutex);
        const auto src = impl_->view(input), dst = impl_->view(output);
        const auto descriptor = native_layout(layout);
        LfsHipError error{};
        check(impl_->api.strided_copy(impl_->context, src, dst, &descriptor, dtype_size(input.dtype), &error), error);
    }
    void HipBackendOps::strided_copy_immediate(StorageRef input, StorageRef output, const StridedLayout& layout, ExecContext execution) {
        strided_copy(input, output, layout, execution);
        synchronize_stream(execution);
    }
    void HipBackendOps::fill_strided(StorageRef output, const StridedLayout& layout, ScalarOperand value, ExecContext execution) {
        queue(execution);

        std::lock_guard lock(impl_->mutex);
        const auto dst = impl_->view(output);
        const auto descriptor = native_layout(layout);
        LfsHipError error{};
        check(impl_->api.fill_layout_bits(impl_->context, dst, &descriptor, static_cast<uint32_t>(dtype_size(output.dtype)), typed_scalar_bits(output.dtype, value), &error), error);
    }
    void HipBackendOps::clamp_fused(StorageRef input, StorageRef output, ScalarOperand minimum, ScalarOperand maximum, size_t count, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        if (input.dtype == DataType::Int32) {
            require(output.dtype == DataType::Int32 && minimum.kind == ScalarKind::Int32 && maximum.kind == ScalarKind::Int32,
                    "HIP integer clamp requires Int32 operands");
            const auto src = impl_->view(input, byte_count(count)), dst = impl_->view(output, byte_count(count));
            LfsHipError error{};
            if (src.buffer != dst.buffer || src.offset != dst.offset)
                check(impl_->api.copy(impl_->context, src, dst, byte_count(count), &error), error);
            check(impl_->api.clamp_i32(impl_->context, dst, count, minimum.value.int32_value, maximum.value.int32_value, &error), error);
            return;
        }
        PointwiseProgram program{};
        program.in_dtype = program.out_dtype = DataType::Float32;
        program.op = PointwiseOp::MaximumScalar;
        program.scalar = minimum;
        impl_->pointwise(program, input, {}, output, count, false);
        program.op = PointwiseOp::MinimumScalar;
        program.scalar = maximum;
        impl_->pointwise(program, output, {}, output, count, false);
    }
    void HipBackendOps::clamp_scalar_int(StorageRef data, ScalarOperand minimum, ScalarOperand maximum, size_t count, ExecContext execution) {
        queue(execution);
        require(data.dtype == DataType::Int32 && minimum.kind == ScalarKind::Int32 && maximum.kind == ScalarKind::Int32,
                "HIP integer clamp requires Int32 operands");
        std::lock_guard lock(impl_->mutex);
        const auto view = impl_->view(data, byte_count(count));
        LfsHipError error{};
        check(impl_->api.clamp_i32(impl_->context, view, count, minimum.value.int32_value, maximum.value.int32_value, &error), error);
    }
    void HipBackendOps::clamp_scalar(StorageRef data, ScalarOperand minimum, ScalarOperand maximum, size_t count, ExecContext execution) {
        clamp_fused(data, data, minimum, maximum, count, execution);
    }
    void HipBackendOps::launch_inverse_lens(StorageRef output, const UndistortParams& p, ExecContext execution) {
        queue(execution);
        LfsHipLens lens{p.src_fx, p.src_fy, p.src_cx, p.src_cy, p.dst_fx, p.dst_fy, p.dst_cx, p.dst_cy, p.src_width, p.src_height, p.dst_width, p.dst_height, static_cast<int32_t>(p.model_type), {}, p.num_distortion};
        std::copy(std::begin(p.distortion), std::end(p.distortion), std::begin(lens.distortion));
        std::lock_guard lock(impl_->mutex);
        const auto dst = impl_->view(output, size_t(p.src_width) * p.src_height * 2 * sizeof(float));
        LfsHipError error{};
        check(impl_->api.inverse_lens(impl_->context, dst, &lens, &error), error);
    }
    void HipBackendOps::sh_codec(StorageRef source, StorageRef destination, const ShCodecProgram& program, ExecContext execution) {
        queue(execution);
        const auto& p = program.codec;
        if (!p.count || !p.destination_rest)
            return;
        require(source.dtype == (p.source_format == ShFormat::Float16 || p.source_format == ShFormat::Q16 ? DataType::Float16 : DataType::Float32) &&
                    destination.dtype == (p.destination_format == ShFormat::Float16 || p.destination_format == ShFormat::Q16 ? DataType::Float16 : DataType::Float32),
                "HIP SH codec dtype mismatch");
        LfsHipShCodec descriptor{};
        descriptor.struct_bytes = sizeof(descriptor);
        descriptor.source_format = uint32_t(p.source_format);
        descriptor.destination_format = uint32_t(p.destination_format);
        if (program.indices) {
            require(program.indices->dtype == DataType::Int32 || program.indices->dtype == DataType::Int64, "HIP SH codec requires integer indices");
            descriptor.index_type = program.indices->dtype == DataType::Int64 ? 2 : 1;
        }
        descriptor.scatter = p.scatter;
        descriptor.source_rest = p.source_rest;
        descriptor.destination_rest = p.destination_rest;
        descriptor.match_cpu_rounding = p.match_cpu_rounding;
        descriptor.source_rows = p.source_rows;
        descriptor.destination_rows = p.destination_rows;
        descriptor.count = p.count;
        descriptor.source_offset = p.source_offset;
        descriptor.destination_offset = p.destination_offset;
        std::lock_guard lock(impl_->mutex);
        const auto src = source.data ? impl_->view(source) : (impl_->empty_view(source), LfsHipView{});
        const auto dst = impl_->view(destination);
        auto optional = [&](const std::optional<StorageRef>& value) {if(!value)return LfsHipView{};require(value->dtype==DataType::Float32,"HIP SH bounds must be Float32");return impl_->view(*value); };
        const auto indices = program.indices ? impl_->view(*program.indices) : LfsHipView{};
        LfsHipError error{};
        check(impl_->api.sh_codec(impl_->context, &descriptor, src, dst, indices, optional(program.source_bounds), optional(program.destination_bounds), &error), error);
    }
    void HipBackendOps::launch_shared_resize(const LfsHipSharedResize& descriptor, StorageRef source, StorageRef destination, StorageRef coefficients_x, StorageRef coefficients_y, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        const auto src = impl_->view(source), dst = impl_->view(destination);
        const auto x = coefficients_x.data ? impl_->view(coefficients_x) : LfsHipView{};
        const auto y = coefficients_y.data ? impl_->view(coefficients_y) : LfsHipView{};
        LfsHipError error{};
        check(impl_->api.shared_resize(impl_->context, &descriptor, src, dst, x, y, &error), error);
    }
    void HipBackendOps::shared_image(const LfsHipSharedImage& descriptor, StorageRef source, StorageRef destination, StorageRef auxiliary, ExecContext execution) {
        queue(execution);
        std::lock_guard lock(impl_->mutex);
        const auto src = source.data ? impl_->view(source) : LfsHipView{};
        const auto dst = impl_->view(destination);
        const auto aux = auxiliary.data ? impl_->view(auxiliary) : LfsHipView{};
        LfsHipError error{};
        check(impl_->api.shared_image(impl_->context, &descriptor, src, dst, aux, &error), error);
    }
    void HipBackendOps::launch_inverse_warp(StorageRef input, StorageRef output, StorageRef validity, const UndistortParams& p, unsigned channels, unsigned mode, ExecContext execution) {
        queue(execution);
        LfsHipLens lens{p.src_fx, p.src_fy, p.src_cx, p.src_cy, p.dst_fx, p.dst_fy, p.dst_cx, p.dst_cy, p.src_width, p.src_height, p.dst_width, p.dst_height, static_cast<int32_t>(p.model_type), {}, p.num_distortion};
        std::copy(std::begin(p.distortion), std::end(p.distortion), std::begin(lens.distortion));
        std::lock_guard lock(impl_->mutex);
        const auto src = impl_->view(input, size_t(p.dst_width) * p.dst_height * channels * 4);
        const auto dst = impl_->view(output, size_t(p.src_width) * p.src_height * channels * 4);
        const auto valid = validity.data ? impl_->view(validity, size_t(p.src_width) * p.src_height) : LfsHipView{};
        LfsHipError error{};
        check(impl_->api.inverse_warp(impl_->context, src, dst, valid, channels, mode, &lens, &error), error);
    }
    void HipBackendOps::launch_shared_warp(StorageRef input, StorageRef output, const UndistortParams& p, unsigned channels, unsigned mode, ExecContext execution) {
        queue(execution);
        LfsHipLens lens{p.src_fx, p.src_fy, p.src_cx, p.src_cy, p.dst_fx, p.dst_fy, p.dst_cx, p.dst_cy, p.src_width, p.src_height, p.dst_width, p.dst_height, static_cast<int32_t>(p.model_type), {}, p.num_distortion};
        std::copy(std::begin(p.distortion), std::end(p.distortion), std::begin(lens.distortion));
        std::lock_guard lock(impl_->mutex);
        const auto src = impl_->view(input, size_t(p.src_width) * p.src_height * channels * (input.dtype == DataType::UInt8 ? 1 : 4));
        const auto dst = impl_->view(output, size_t(p.dst_width) * p.dst_height * channels * 4);
        LfsHipError error{};
        check(impl_->api.shared_warp(impl_->context, src, static_cast<uint32_t>(input.dtype), channels, mode, dst, &lens, &error), error);
    }
    void HipBackendOps::launch_rgb_warp(StorageRef input, StorageRef output, const UndistortParams& p, ExecContext execution) {
        queue(execution);
        LfsHipLens lens{p.src_fx, p.src_fy, p.src_cx, p.src_cy, p.dst_fx, p.dst_fy, p.dst_cx, p.dst_cy, p.src_width, p.src_height, p.dst_width, p.dst_height, static_cast<int32_t>(p.model_type), {}, p.num_distortion};
        std::copy(std::begin(p.distortion), std::end(p.distortion), std::begin(lens.distortion));
        std::lock_guard lock(impl_->mutex);
        const auto src = impl_->view(input, size_t(p.src_width) * p.src_height * 3 * (input.dtype == DataType::UInt8 ? 1 : 4));
        const auto dst = impl_->view(output, size_t(p.dst_width) * p.dst_height * 3 * sizeof(float));
        LfsHipError error{};
        check(impl_->api.warp_rgb(impl_->context, src, static_cast<uint32_t>(input.dtype), dst, &lens, &error), error);
    }
    void HipBackendOps::cumsum(StorageRef data, const StridedLayout& layout, int dim, ExecContext execution) {
        queue(execution);
        require(dim >= 0 && static_cast<unsigned>(dim) < layout.rank, "Invalid HIP scan axis");
        const auto native = native_layout(layout);
        require(data.dtype == DataType::Float32 || data.dtype == DataType::Int32 || data.dtype == DataType::Int64 || data.dtype == DataType::UInt32,
                "Unsupported HIP scan dtype");
        std::lock_guard lock(impl_->mutex);
        if (!layout.element_count) {
            bool empty = false;
            for (size_t d = 0; d < layout.rank; ++d)
                empty |= layout.dims[d] == 0;
            require(empty && data.byte_offset % dtype_size(data.dtype) == 0, "Invalid HIP empty scan layout");
            if (data.data)
                (void)impl_->view(data);
            else
                (void)impl_->empty_view(data);
            return;
        }
        const auto view = impl_->view(data, 0);
        LfsHipError error{};
        check(impl_->api.cumsum(impl_->context, view, static_cast<uint32_t>(data.dtype), &native, static_cast<uint32_t>(dim), &error), error);
        impl_->refresh_algorithm_workspace();
    }
    void HipBackendOps::sort_1d(StorageRef values, StorageRef indices, size_t count, const SortProgram& program, ExecContext execution) {
        SortProgram native = program;
        native.outer_size = 1;
        native.dim_size = count;
        native.inner_size = 1;
        sort_2d(values, indices, native, execution);
    }
    void HipBackendOps::sort_2d(StorageRef values, StorageRef indices, const SortProgram& program, ExecContext execution) {
        queue(execution);
        require(values.dtype == DataType::Float32 && indices.dtype == DataType::Int64, "HIP sort requires Float32 values and Int64 indices");
        std::lock_guard lock(impl_->mutex);
        const auto value_view = impl_->view(values), index_view = impl_->view(indices);
        LfsHipError error{};
        check(impl_->api.sort_pairs(impl_->context, value_view, index_view, program.outer_size, program.dim_size, program.inner_size, program.descending ? 1u : 0u, &error), error);
        impl_->refresh_algorithm_workspace();
    }
    size_t HipBackendOps::masked_select(StorageRef input, StorageRef mask, StorageRef output, const MaskProgram& program, ExecContext execution) {
        queue(execution);
        require(input.dtype == output.dtype && (mask.dtype == DataType::Bool || mask.dtype == DataType::UInt8), "HIP masked selection requires matching values and a byte mask");
        std::lock_guard lock(impl_->mutex);
        const auto source = impl_->view(input), flags = impl_->view(mask), destination = impl_->view(output);
        uint64_t selected = 0;
        LfsHipError error{};
        const LfsHipSelection descriptor{sizeof(LfsHipSelection), uint32_t(input.dtype), 0, 1, program.count, program.selected_count, program.selected_count};
        const int status = impl_->api.select(impl_->context, source, flags, destination, &descriptor, &selected, &error);
        impl_->refresh_algorithm_workspace();
        check(status, error);
        return static_cast<size_t>(selected);
    }
    void HipBackendOps::masked_scatter(StorageRef output, StorageRef mask, StorageRef source,
                                       const MaskProgram& program, ExecContext execution) {
        queue(execution);
        require(output.dtype == source.dtype &&
                    (mask.dtype == DataType::Bool || mask.dtype == DataType::UInt8),
                "HIP masked scatter requires matching values and a byte mask");
        std::lock_guard lock(impl_->mutex);
        const auto destination = impl_->view(output), flags = impl_->view(mask), values = impl_->view(source);
        LfsHipError error{};
        check(impl_->api.masked_scatter(impl_->context, destination, flags, values,
                                        static_cast<uint32_t>(output.dtype), program.count,
                                        program.selected_count, &error),
              error);
        impl_->refresh_algorithm_workspace();
    }
    size_t HipBackendOps::nonzero(StorageRef input, StorageRef output, const MaskProgram& program, ExecContext execution) {
        queue(execution);
        require(output.dtype == DataType::Int64 && uint32_t(input.dtype) <= uint32_t(DataType::UInt32), "Unsupported HIP nonzero value/index types");
        std::lock_guard lock(impl_->mutex);
        const auto source = impl_->view(input), destination = impl_->view(output);
        uint64_t selected = 0;
        LfsHipError error{};
        const LfsHipSelection descriptor{sizeof(LfsHipSelection), uint32_t(input.dtype), 1, 1, program.count, program.selected_count, program.selected_count};
        const int status = impl_->api.select(impl_->context, source, {}, destination, &descriptor, &selected, &error);
        impl_->refresh_algorithm_workspace();
        check(status, error);
        return static_cast<size_t>(selected);
    }
    size_t HipBackendOps::nonzero_bool(StorageRef input, StorageRef output, const MaskProgram& program, ExecContext execution) {
        require(input.dtype == DataType::Bool || input.dtype == DataType::UInt8, "HIP boolean nonzero requires byte values");
        return nonzero(input, output, program, execution);
    }
    void HipBackendOps::cat_last_dim(StorageRef output, std::span<const StorageRef> inputs, std::span<const StridedLayout> layouts, size_t rows, size_t row_size, size_t width, ExecContext execution) {
        require(!layouts.empty() && inputs.size() == layouts.size() && layouts[0].rank > 0 && layouts[0].rank <= 8, "Invalid HIP concatenation inputs");
        const unsigned axis = layouts[0].rank - 1;
        size_t columns = 0;
        for (const auto& layout : layouts) {
            require(layout.rank == layouts[0].rank && layout.dims[axis] <= SIZE_MAX - columns, "Invalid HIP concatenation rank or extent");
            columns += layout.dims[axis];
        }
        require(columns == row_size, "HIP concatenation row size differs");
        cat_middle_dim(output, inputs, layouts, rows, 1, static_cast<int>(axis), width, execution);
    }
    void HipBackendOps::cat_middle_dim(StorageRef output, std::span<const StorageRef> inputs, std::span<const StridedLayout> layouts, size_t outer, size_t inner, int dim, size_t width, ExecContext execution) {
        queue(execution);
        require(!layouts.empty() && inputs.size() == layouts.size() && inputs.size() <= 4096 && dim >= 0 && size_t(dim) < layouts[0].rank && layouts[0].rank <= 8 && width == dtype_size(output.dtype), "Invalid HIP concatenation descriptor");
        size_t actual_outer = 1, actual_inner = 1, total = 0;
        for (int d = 0; d < dim; ++d)
            actual_outer = checked_product(actual_outer, layouts[0].dims[d]);
        for (size_t d = dim + 1; d < layouts[0].rank; ++d)
            actual_inner = checked_product(actual_inner, layouts[0].dims[d]);
        require(actual_outer == outer && actual_inner == inner, "HIP concatenation outer/inner sizes differ");
        for (size_t i = 0; i < layouts.size(); ++i) {
            const auto& layout = layouts[i];
            require(layout.rank == layouts[0].rank && inputs[i].dtype == output.dtype, "HIP concatenation rank or value types differ");
            for (size_t d = 0; d < layout.rank; ++d)
                require(d == size_t(dim) || layout.dims[d] == layouts[0].dims[d], "HIP concatenation non-axis extents differ");
            require(layout.dims[dim] <= SIZE_MAX - total, "HIP concatenation axis overflow");
            total += layout.dims[dim];
            require(layout.element_count == checked_product(checked_product(outer, layout.dims[dim]), inner), "HIP concatenation element count differs");
        }
        const size_t output_bytes = checked_product(checked_product(checked_product(outer, total), inner), width);
        if (!output_bytes)
            return;
        std::lock_guard lock(impl_->mutex);
        const auto destination = impl_->view(output, output_bytes);
        std::vector<LfsHipLayoutCopy> copies;
        copies.reserve(inputs.size());
        size_t column = 0;
        for (size_t i = 0; i < layouts.size(); ++i) {
            const auto extent = layouts[i].dims[dim];
            if (layouts[i].element_count) {
                LfsHipLayout target{};
                target.rank = 3;
                target.dims[0] = outer;
                target.dims[1] = extent;
                target.dims[2] = inner;
                target.strides[0] = checked_product(total, inner);
                target.strides[1] = inner;
                target.strides[2] = 1;
                target.elements = layouts[i].element_count;
                auto view = destination;
                view.offset += checked_product(checked_product(column, inner), width);
                copies.push_back({impl_->view(inputs[i]), view, native_layout(layouts[i]), target, static_cast<uint32_t>(width)});
            }
            column += extent;
        }
        LfsHipError error{};
        check(impl_->api.copy_layouts(impl_->context, copies.data(), copies.size(), &error), error);
    }
    void HipBackendOps::push_range(const char*) {
        // Range annotations are optional instrumentation. The HIP runtime
        // boundary intentionally does not link CUDA/NVTX or a profiler SDK.
        static std::once_flag notice;
        std::call_once(notice, [] {
            std::fprintf(stderr, "[info] HIP GPU range annotations are unavailable in this runtime; profiling labels are disabled\n");
        });
    }
    void HipBackendOps::pop_range() {}
    bool HipBackendOps::nn_kernels() const { return true; }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    void HipBackendOps::launch_radius(StorageRef points, StorageRef references, StorageRef heads, StorageRef next,
                                      StorageRef output, size_t count, size_t buckets, float radius, unsigned mode,
                                      int32_t maximum, bool exclude_self, std::optional<StorageRef> queries,
                                      std::optional<StorageRef> values, std::optional<StorageRef> radii, ExecContext execution) {
        queue(execution);
        require(count <= INT32_MAX && buckets && !(buckets & (buckets - 1)) && buckets <= uint64_t(UINT32_MAX) + 1,
                "Invalid HIP radius query extent");
        require(std::isfinite(radius) && radius > 0 && mode <= 5 && (mode != 1 || maximum > 0),
                "Invalid HIP radius query parameters");
        require(points.dtype == DataType::Float32 && heads.dtype == DataType::Int32 && next.dtype == DataType::Int32,
                "Invalid HIP radius query input types");
        require(references.dtype == DataType::Bool || references.dtype == DataType::UInt8, "Invalid HIP radius reference mask");
        require(output.dtype == (mode == 0 ? DataType::Bool : mode == 1 || mode == 3 || mode == 5 ? DataType::Int32
                                                                                                  : DataType::Float32),
                "Invalid HIP radius query output type");
        if (queries)
            require(queries->dtype == DataType::Bool || queries->dtype == DataType::UInt8, "Invalid HIP query mask");
        if (values)
            require(values->dtype == output.dtype, "Invalid HIP neighbor minimum type");
        if (radii)
            require(radii->dtype == DataType::Float32, "Invalid HIP neighbor radii type");
        if (!count)
            return;
        std::lock_guard lock(impl_->mutex);
        const LfsHipView inputs[]{impl_->view(points), impl_->view(references), impl_->view(heads), impl_->view(next),
                                  queries ? impl_->view(*queries) : LfsHipView{}, values ? impl_->view(*values) : LfsHipView{},
                                  radii ? impl_->view(*radii) : LfsHipView{}};
        LfsHipError error{};
        check(impl_->api.radius_query(impl_->context, inputs, impl_->view(output), count, buckets,
                                      radius, mode, maximum, exclude_self, &error),
              error);
    }
    void HipBackendOps::radius_neighbors(StorageRef p, StorageRef r, StorageRef h, StorageRef n, StorageRef o,
                                         size_t count, size_t buckets, float radius, bool exclude, std::optional<StorageRef> q, ExecContext c) {
        launch_radius(p, r, h, n, o, count, buckets, radius, 0, 0, exclude, q, {}, {}, c);
    }
    void HipBackendOps::radius_neighbor_counts(StorageRef p, StorageRef r, StorageRef h, StorageRef n, StorageRef o,
                                               size_t count, size_t buckets, float radius, int32_t maximum, std::optional<StorageRef> q, ExecContext c) {
        launch_radius(p, r, h, n, o, count, buckets, radius, 1, maximum, true, q, {}, {}, c);
    }
    void HipBackendOps::point_neighbor_spacing(StorageRef p, StorageRef r, StorageRef h, StorageRef n, StorageRef o,
                                               size_t count, size_t buckets, float radius, ExecContext c) {
        launch_radius(p, r, h, n, o, count, buckets, radius, 2, 0, false, {}, {}, {}, c);
    }
    void HipBackendOps::radius_neighbor_min(StorageRef p, StorageRef v, StorageRef r, StorageRef h, StorageRef n,
                                            StorageRef o, size_t count, size_t buckets, float radius, std::optional<StorageRef> radii, ExecContext c) {
        launch_radius(p, r, h, n, o, count, buckets, radius, v.dtype == DataType::Int32 ? 3 : 4, 0, false, {}, v, radii, c);
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    void HipBackendOps::nearest_point_indices(StorageRef q, StorageRef t, StorageRef h, StorageRef n, StorageRef o,
                                              size_t nq, size_t nt, size_t buckets, float width, ExecContext execution) {
        queue(execution);
        require(q.dtype == DataType::Float32 && t.dtype == DataType::Float32 && h.dtype == DataType::Int32 &&
                    n.dtype == DataType::Int32 && o.dtype == DataType::Int32,
                "Invalid HIP nearest point types");
        require(nq <= INT32_MAX && nt <= INT32_MAX && std::isfinite(width) && width > 0 && buckets &&
                    !(buckets & (buckets - 1)) && buckets <= uint64_t(UINT32_MAX) + 1,
                "Invalid HIP nearest point extent");
        if (!nq)
            return;
        std::lock_guard lock(impl_->mutex);
        const LfsHipView inputs[]{impl_->view(q), nt ? impl_->view(t) : (impl_->empty_view(t), LfsHipView{}), impl_->view(h),
                                  nt ? impl_->view(n) : (impl_->empty_view(n), LfsHipView{})};
        LfsHipError error{};
        check(impl_->api.proximity(impl_->context, inputs, impl_->view(o), nq, nt, buckets, width, 0, &error), error);
    }
    void HipBackendOps::camera_frustum_counts(StorageRef p, StorageRef c, StorageRef o, size_t count,
                                              size_t cameras, float maximum, ExecContext execution) {
        queue(execution);
        require(p.dtype == DataType::Float32 && c.dtype == DataType::Float32 && o.dtype == DataType::Int32,
                "Invalid HIP frustum types");
        require(count <= INT32_MAX && cameras <= INT32_MAX && std::isfinite(maximum), "Invalid HIP frustum extent");
        if (!count)
            return;
        std::lock_guard lock(impl_->mutex);
        const LfsHipView inputs[]{impl_->view(p), cameras ? impl_->view(c) : (impl_->empty_view(c), LfsHipView{}), {}, {}};
        LfsHipError error{};
        check(impl_->api.proximity(impl_->context, inputs, impl_->view(o), count, cameras, 1, maximum, 1, &error), error);
    }
} // namespace lfs::core::internal

#ifdef LFS_HIP_EXPRESSION_JIT
namespace lfs::core::internal {
    void HipBackendOps::compiled_expression(const ExpressionLaunch& launch, ExecContext execution) {
        queue(execution);
        require(launch.program && launch.words && launch.words <= ExpressionLayout::max_words &&
                    launch.read_count <= launch.reads.size() && launch.write_count <= launch.writes.size(),
                "Invalid HIP expression descriptor");
        std::lock_guard lock(impl_->mutex);
        require(impl_->context, "HIP expression context is closed");
        for (size_t i = 0; i < launch.read_count; ++i)
            (void)impl_->view(launch.reads[i]);
        for (size_t i = 0; i < launch.write_count; ++i)
            (void)impl_->view(launch.writes[i]);
        LfsHipError error{};
        if (impl_->expression_identity.empty()) {
            char identity[256]{};
            check(impl_->api.expression_identity_for_device(impl_->ordinal, identity, sizeof(identity), &error), error);
            impl_->expression_identity = identity;
        }
        const auto key = impl_->expression_identity + "\n" + hip_expression_emitter_id + "\n" + expression_key(*launch.program, launch.signature);
        const auto kernel = impl_->expressions.get(key, [&] {
        const auto source=expression_hip(*launch.program,launch.signature);
        std::vector<char> binary(16*1024*1024);
        size_t size=0;char diagnostics[8192]{};
        const int status=impl_->api.expression_compile_for_device(impl_->ordinal,source.c_str(),binary.data(),binary.size(),&size,diagnostics,sizeof(diagnostics),&error);
        if(status) throw std::runtime_error(std::string(error.message)+"\n"+diagnostics);
        binary.resize(size);return binary; }, [&](std::span<const char> binary) -> std::shared_ptr<CompiledExpression> {
        auto compiled=std::make_shared<Impl::ExpressionKernel>();
        compiled->owner=impl_.get();compiled->bytes=binary.size();
        check(impl_->api.expression_load(impl_->context,binary.data(),binary.size(),&compiled->module,&error),error);
        return compiled; });
        if (launch.prepare_only || !launch.count)
            return;
        const auto work = (uint64_t(launch.count) + expression_packing(launch.signature) - 1) / expression_packing(launch.signature);
        check(impl_->api.expression_launch(impl_->context, static_cast<Impl::ExpressionKernel&>(*kernel).module,
                                           launch.arguments.data(), launch.words, work, &error),
              error);
    }
} // namespace lfs::core::internal
#endif

namespace lfs::core::internal {
    void HipBackendOps::launch_point_tree(StorageRef p, StorageRef sorted, StorageRef boxes, StorageRef visit,
                                          std::optional<StorageRef> radii, std::optional<StorageRef> queries, StorageRef output, const PointTreeProgram& program,
                                          bool spacing, ExecContext execution) {
        queue(execution);
        require(p.dtype == DataType::Float32 && sorted.dtype == DataType::Float32 && boxes.dtype == DataType::Float32 &&
                    visit.dtype == DataType::Int32 && output.dtype == (spacing ? DataType::Float32 : DataType::Int32),
                "Invalid HIP point-tree types");
        if (radii)
            require(radii->dtype == DataType::Float32, "Invalid HIP point-tree radii type");
        if (queries)
            require(queries->dtype == DataType::Bool || queries->dtype == DataType::UInt8, "Invalid HIP point-tree mask type");
        require(spacing || radii.has_value(), "Missing HIP point-tree radii");
        require(!program.points || !radii || radii->data, "Missing HIP point-tree radii storage");
        require(!program.points || !queries || queries->data, "Missing HIP point-tree query storage");
        std::lock_guard lock(impl_->mutex);
        LfsHipPointTree native{};
        native.points = program.points;
        native.references = program.references;
        native.levels = program.levels;
        static_assert(kPointTreeMaxLevels == 8 && kPointTreeFanout == 32);
        std::copy_n(program.level_offset, 8, native.level_offset);
        std::copy_n(program.level_count, 8, native.level_count);
        native.radius = program.radius;
        native.max_count = program.max_count;
        const auto view = [&](StorageRef storage) { return storage.data ? impl_->view(storage) : (impl_->empty_view(storage), LfsHipView{}); };
        const LfsHipView inputs[]{view(p), view(sorted), view(boxes), view(visit), radii ? view(*radii) : LfsHipView{}, queries ? view(*queries) : LfsHipView{}};
        LfsHipError error{};
        check(impl_->api.point_tree_query(impl_->context, inputs, view(output), &native, spacing, &error), error);
    }
    bool HipBackendOps::point_tree_counts(StorageRef p, StorageRef sorted, StorageRef boxes, StorageRef visit, StorageRef radii,
                                          std::optional<StorageRef> queries, StorageRef output, const PointTreeProgram& program, ExecContext execution) {
        launch_point_tree(p, sorted, boxes, visit, radii, queries, output, program, false, execution);
        return true;
    }
    bool HipBackendOps::point_tree_spacing(StorageRef p, StorageRef sorted, StorageRef boxes, StorageRef visit, StorageRef output,
                                           const PointTreeProgram& program, ExecContext execution) {
        launch_point_tree(p, sorted, boxes, visit, {}, {}, output, program, true, execution);
        return true;
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    bool HipBackendOps::radius_connected_components(StorageRef p, StorageRef r, StorageRef h, StorageRef n, StorageRef labels,
                                                    size_t count, size_t buckets, float radius, ExecContext execution) {
        launch_radius(p, r, h, n, labels, count, buckets, radius, 5, 0, false, {}, {}, {}, execution);
        return true;
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    bool HipBackendOps::point_tree_components(StorageRef p, StorageRef sorted, StorageRef boxes, StorageRef box_radii,
                                              StorageRef visit, StorageRef sorted_radii, StorageRef radii, StorageRef labels, const PointTreeProgram& program, ExecContext execution) {
        queue(execution);
        require(visit.dtype == DataType::Int32 && labels.dtype == DataType::Int32, "Invalid HIP component index types");
        for (const auto input : {p, sorted, boxes, box_radii, sorted_radii, radii})
            require(input.dtype == DataType::Float32, "HIP point-tree components require Float32 geometry/radii");
        std::lock_guard lock(impl_->mutex);
        LfsHipPointTree native{};
        native.points = program.points;
        native.references = program.references;
        native.levels = program.levels;
        std::copy_n(program.level_offset, 8, native.level_offset);
        std::copy_n(program.level_count, 8, native.level_count);
        native.radius = program.radius;
        native.max_count = program.max_count;
        const auto view = [&](StorageRef storage) { return storage.data ? impl_->view(storage) : (impl_->empty_view(storage), LfsHipView{}); };
        const LfsHipView inputs[]{view(p), view(sorted), view(boxes), view(box_radii), view(visit), view(sorted_radii), view(radii)};
        LfsHipError error{};
        check(impl_->api.point_tree_components(impl_->context, inputs, view(labels), &native, &error), error);
        return true;
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    void HipBackendOps::ppisp_apply(StorageRef input, StorageRef output, int width, int height, const PpispParams& params, ExecContext execution) {
        queue(execution);
        require(input.dtype == DataType::Float32 && output.dtype == DataType::Float32, "HIP PPISP requires Float32");
        static_assert(sizeof(PpispParams) == sizeof(LfsHipPpisp));
        static_assert(offsetof(PpispParams, x_offset) == offsetof(LfsHipPpisp, x_offset));
        LfsHipPpisp native{};
        std::memcpy(&native, &params, sizeof(native));
        std::lock_guard lock(impl_->mutex);
        LfsHipError error{};
        check(impl_->api.ppisp(impl_->context, impl_->view(input), impl_->view(output), width, height, &native, &error), error);
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    void HipBackendOps::affine_splat_geometry(StorageRef scales, StorageRef rotations, StorageRef out_scales, StorageRef out_rotations,
                                              const splat_transform::LinearTransform& matrix, size_t count, std::optional<StorageRef> matrices, ExecContext execution) {
        queue(execution);
        for (auto value : {scales, rotations, out_scales, out_rotations})
            require(value.dtype == DataType::Float32, "HIP affine geometry requires Float32");
        if (matrices)
            require(matrices->dtype == DataType::Float32, "HIP affine matrices require Float32");
        std::lock_guard lock(impl_->mutex);
        const auto view = [&](StorageRef value) { return value.data ? impl_->view(value) : (impl_->empty_view(value), LfsHipView{}); };
        const LfsHipView inputs[]{view(scales), view(rotations), matrices ? view(*matrices) : LfsHipView{}};
        const LfsHipView outputs[]{view(out_scales), view(out_rotations)};
        LfsHipError error{};
        check(impl_->api.affine(impl_->context, inputs, outputs, matrix.rows, count, &error), error);
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    Tensor HipBackendOps::morton_sort(const Tensor& positions, Tensor* sorted_keys, ExecContext execution) {
        queue(execution);
        const GpuBackendScope scope(GpuBackend::HIP);
        require(storage_ref(positions).backend == GpuBackend::HIP && positions.dtype() == DataType::Float32 && positions.ndim() == 2 && positions.size(1) == 3,
                "HIP Morton requires Float32 positions [N,3]");
        const auto count = positions.size(0);
        require(count <= INT32_MAX, "HIP Morton count exceeds indexing range");
        auto keys = Tensor::empty({count}, Device::GPU, DataType::Int64);
        auto indices = Tensor::empty({count}, Device::GPU, DataType::Int32);
        if (count) {
            auto input = positions.contiguous();
            std::lock_guard lock(impl_->mutex);
            LfsHipError error{};
            check(impl_->api.morton(impl_->context, impl_->view(storage_ref(input)), impl_->view(storage_ref(keys)),
                                    impl_->view(storage_ref(indices)), count, &error),
                  error);
            impl_->refresh_algorithm_workspace();
        }
        if (sorted_keys)
            *sorted_keys = std::move(keys);
        return indices;
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    void HipBackendOps::multinomial(StorageRef weights, StorageRef output, const RandomProgram& program, ExecContext execution) {
        queue(execution);
        // rocPRIM dispatch uses signed counts. Reject before storage validation
        // or dispatch with the same public error contract as other backends.
        if (program.count > size_t{std::numeric_limits<int>::max()} ||
            program.sample_count > size_t{std::numeric_limits<int>::max()}) {
            throw lfs::Exception(lfs::make_error({
                .code = lfs::ErrorCode::BoundsViolation,
                .domain = lfs::ErrorDomain::Tensor,
                .user_message = "HIP multinomial count exceeds safe indexing range",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
        require(weights.dtype == DataType::Float32 && output.dtype == DataType::Int64, "HIP multinomial requires Float32 weights and Int64 output");
        std::lock_guard lock(impl_->mutex);
        LfsHipError error{};
        const auto status = impl_->api.multinomial(impl_->context, impl_->view(weights), impl_->view(output), program.count,
                                                   program.sample_count, program.seed, program.replacement, &error);
        impl_->refresh_algorithm_workspace();
        check(status, error);
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    class HipExportKernels final : public ExportKernels {
        HipBackendOps& owner_;

    public:
        explicit HipExportKernels(HipBackendOps& owner) : owner_(owner) {}
        GpuBackend backend() const override { return GpuBackend::HIP; }
        bool float_atomics() const override { return true; }
        bool screened_assignment() const override { return false; }
        uint64_t address(const Tensor& tensor) const override {
            auto storage = storage_ref(tensor);
            require(storage.backend == GpuBackend::HIP, "Foreign HIP export tensor");
            std::lock_guard lock(owner_.impl_->mutex);
            const auto view = owner_.impl_->view(storage);
            LfsHipError error{};
            uint64_t address = 0;
            check(owner_.impl_->api.buffer_address(owner_.impl_->context, view.buffer, &address, &error), error);
            return address + view.offset;
        }
        void launch(const char* module, uint32_t phase, std::span<const std::byte> params, std::span<const StorageRef> reads,
                    std::span<const StorageRef> writes, size_t work) override { owner_.launch_export(module, phase, params, reads, writes, work); }
        void submit() override {} // Native kernels already enqueue on the owning stream.
    };
    void HipBackendOps::launch_export(const char* module, uint32_t phase, std::span<const std::byte> params,
                                      std::span<const StorageRef> reads, std::span<const StorageRef> writes, size_t work) {
        uint32_t id = 9;
        const char* names[]{"export_kmeans", "export_kmeans_accumulate", "export_kmeans_group", "export_radix", "export_morton", "export_decimate", "export_decimate_cost", "export_decimate_merge", "point_raster"};
        for (uint32_t i = 0; i < 9; ++i)
            if (!std::strcmp(module, names[i]))
                id = i;
        require(id < 9, "Unknown HIP export module");
        std::lock_guard lock(impl_->mutex);
        for (auto storage : reads) {
            if (storage.data)
                (void)impl_->view(storage);
            else
                impl_->empty_view(storage);
        }
        for (auto storage : writes)
            (void)impl_->view(storage);
        LfsHipError error{};
        check(impl_->api.export_dispatch(impl_->context, id, phase, params.data(), params.size(), work, &error), error);
    }
    std::tuple<Tensor, Tensor> HipBackendOps::kmeans_sh(const Tensor& sh, int points, int coeffs, int k, int iterations, bool, ExecContext execution) {
        queue(execution);
        HipExportKernels kernels(*this);
        return export_kmeans_sh(kernels, sh, points, coeffs, k, iterations);
    }
    void HipBackendOps::assign_sh3(const Tensor& sh, const Tensor& centroids, const Tensor& norms, Tensor& labels, bool fast, bool have_labels, ExecContext execution) {
        queue(execution);
        HipExportKernels kernels(*this);
        export_assign_sh3(kernels, sh, centroids, norms, labels, fast, have_labels);
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    void HipBackendOps::decimate_candidates(const Tensor& position, const Tensor& rotation, const Tensor& scale, const Tensor& opacity,
                                            const Tensor& dc, const Tensor& sh, int rest, std::vector<uint32_t>& idx, std::vector<float>& cost, ExecContext execution) {
        queue(execution);
        HipExportKernels kernels(*this);
        export_decimate_candidates(kernels, position, rotation, scale, opacity, dc, sh, rest, idx, cost);
    }
    DecimateMerge HipBackendOps::decimate_merge(const Tensor& position, const Tensor& rotation, const Tensor& scale, const Tensor& opacity,
                                                const Tensor& dc, const Tensor& sh, int rest, const std::vector<int>& member_group, const std::vector<uint32_t>& minimum,
                                                const std::vector<uint32_t>& members, const std::vector<uint32_t>& offsets, size_t removed, ExecContext execution) {
        queue(execution);
        HipExportKernels kernels(*this);
        return export_decimate_merge(kernels, position, rotation, scale, opacity, dc, sh, rest, member_group, minimum, members, offsets, removed);
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    void HipBackendOps::rasterize_points(const PointRasterProgram& p, ExecContext execution) {
        queue(execution);
        const size_t pixels = size_t(p.width) * p.height;
        require(p.width && p.height && pixels <= INT32_MAX / 2 && p.count <= INT32_MAX, "Invalid HIP point raster dimensions");
        require(p.positions.dtype == DataType::Float32 && p.colors.dtype == DataType::Float32 && p.parameters.dtype == DataType::Float32 &&
                    p.scratch.dtype == DataType::Int32 && p.image.dtype == DataType::Float32 && p.depth.dtype == DataType::Float32,
                "Invalid HIP point raster types");
        struct Push {
            uint64_t positions, colors, parameters, scratch, image, depth, transforms, indices, visibility, deleted, overlay;
            uint32_t count, width, height, transform_count, visibility_count, flags;
            float ortho_scale, focal_y, voxel_size, far_plane;
        };
        static_assert(sizeof(Push) == 128);
        std::vector<StorageRef> reads, writes{p.scratch, p.image, p.depth};
        std::vector<std::tuple<uint64_t, size_t, bool>> ranges;
        const auto address = [&](StorageRef storage, size_t bytes, bool write = false) -> uint64_t {
            std::lock_guard lock(impl_->mutex);
            if (!storage.data) {
                impl_->empty_view(storage);
                require(!bytes, "Missing HIP raster storage");
                return 0;
            }
            require(storage.byte_offset % ((storage.dtype == DataType::Float32 || storage.dtype == DataType::Int32) ? 4 : 1) == 0, "Misaligned HIP raster storage");
            auto view = impl_->view(storage, bytes);
            uint64_t pointer = 0;
            LfsHipError error{};
            check(impl_->api.buffer_address(impl_->context, view.buffer, &pointer, &error), error);
            pointer += view.offset;
            ranges.emplace_back(pointer, bytes, write);
            return pointer;
        };
        const auto input = [&](StorageRef storage, size_t bytes) {reads.push_back(storage);return address(storage,bytes); };
        const auto optional = [&](std::optional<StorageRef> storage, size_t bytes, DataType type) -> uint64_t {
            if (!storage)
                return 0;
            require(storage->dtype == type || (type == DataType::UInt8 && storage->dtype == DataType::Bool), "Invalid HIP raster optional input type");
            return input(*storage, bytes);
        };
        const Push push{input(p.positions, p.count * 12), input(p.colors, p.count * 12), input(p.parameters, (kPointRasterParameters + (p.overlay ? 258 * 4 : 0)) * 4),
                        address(p.scratch, pixels * 8, true), address(p.image, pixels * ((p.flags & kPointRasterTransparent) ? 16 : 12), true), address(p.depth, pixels * 4, true),
                        optional(p.transforms, size_t(p.transform_count) * 64, DataType::Float32), optional(p.indices, p.count * 4, DataType::Int32),
                        optional(p.visibility, p.visibility_count, DataType::UInt8), optional(p.deleted, p.count, DataType::UInt8), optional(p.overlay, p.count * 2, DataType::UInt8),
                        uint32_t(p.count), p.width, p.height, p.transform_count, p.visibility_count, p.flags, p.ortho_scale, p.focal_y, p.voxel_size, p.far_plane};
        for (size_t i = 0; i < ranges.size(); ++i)
            for (size_t j = 0; j < i; ++j) {
                const auto [a, an, aw] = ranges[i];
                const auto [b, bn, bw] = ranges[j];
                require(!an || !bn || (!aw && !bw) || a >= b + bn || b >= a + an, "Overlapping HIP raster output storage");
            }
        for (uint32_t phase = 0; phase < 4; ++phase) {
            const size_t work = (phase == 1 || phase == 2) ? p.count : pixels;
            if (work)
                launch_export("point_raster", phase, std::as_bytes(std::span(&push, 1)), reads, writes, work);
        }
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    void HipBackendOps::nn_norm(StorageRef input, StorageRef weight, std::optional<StorageRef> bias, StorageRef output, const NormProgram& p, ExecContext execution) {
        queue(execution);
        require((input.dtype == DataType::Float32 || input.dtype == DataType::Float16) && weight.dtype == input.dtype && output.dtype == input.dtype &&
                    (!bias || bias->dtype == input.dtype),
                "HIP NN norm requires matching Float32 or Float16 types");
        std::lock_guard lock(impl_->mutex);
        const auto view = [&](StorageRef storage) { return storage.data ? impl_->view(storage) : (impl_->empty_view(storage), LfsHipView{}); };
        const LfsHipNorm native{p.rows, p.cols, p.eps, uint32_t(input.dtype)};
        LfsHipError error{};
        check(impl_->api.nn_norm(impl_->context, view(input), view(weight), bias ? view(*bias) : LfsHipView{}, view(output), &native, &error), error);
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    void HipBackendOps::nn_linear(StorageRef a, StorageRef w, std::optional<StorageRef> bias, std::optional<StorageRef> scale,
                                  std::optional<StorageRef> residual, StorageRef output, const LinearProgram& p, ExecContext execution) {
        queue(execution);
        require((a.dtype == DataType::Float32 || a.dtype == DataType::Float16) && w.dtype == a.dtype && output.dtype == a.dtype, "HIP NN linear requires matching Float32 or Float16 types");
        for (const auto& item : {bias, scale, residual})
            if (item)
                require(item->dtype == a.dtype, "HIP NN linear affine input type differs");
        std::lock_guard lock(impl_->mutex);
        const auto view = [&](StorageRef storage) { return storage.data ? impl_->view(storage) : (impl_->empty_view(storage), LfsHipView{}); };
        const LfsHipView inputs[]{view(a), view(w), bias ? view(*bias) : LfsHipView{}, scale ? view(*scale) : LfsHipView{}, residual ? view(*residual) : LfsHipView{}};
        const LfsHipLinear native{p.batch, p.m, p.n, p.k, uint32_t(a.dtype), uint32_t(p.trans_b), uint32_t(p.batched_b), uint32_t(p.activation)};
        LfsHipError error{};
        check(impl_->api.nn_linear(impl_->context, inputs, view(output), &native, &error), error);
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    void HipBackendOps::nn_attention(StorageRef q, StorageRef k, StorageRef v, std::optional<StorageRef> mask, StorageRef output, const AttentionProgram& p, ExecContext execution) {
        queue(execution);
        require((q.dtype == DataType::Float32 || q.dtype == DataType::Float16) && k.dtype == q.dtype && v.dtype == q.dtype && output.dtype == q.dtype &&
                    (!mask || mask->dtype == q.dtype),
                "HIP NN attention requires matching Float32 or Float16 types");
        std::lock_guard lock(impl_->mutex);
        const auto view = [&](StorageRef storage) { return storage.data ? impl_->view(storage) : (impl_->empty_view(storage), LfsHipView{}); };
        const LfsHipView inputs[]{view(q), view(k), view(v), mask ? view(*mask) : LfsHipView{}};
        LfsHipAttention native{};
        native.groups = p.groups;
        native.heads = p.heads;
        native.queries = p.queries;
        native.keys = p.keys;
        native.dim = p.dim;
        std::copy_n(p.mask_strides.begin(), 4, native.strides);
        native.scale = p.scale;
        native.dtype = uint32_t(q.dtype);
        LfsHipError error{};
        const auto status = impl_->api.nn_attention(impl_->context, inputs, view(output), &native, &error);
        impl_->refresh_algorithm_workspace();
        check(status, error);
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    void HipBackendOps::nn_conv2d(StorageRef input, StorageRef weight, std::optional<StorageRef> bias, StorageRef output, const ConvProgram& p, ExecContext execution) {
        queue(execution);
        require((input.dtype == DataType::Float32 || input.dtype == DataType::Float16) && weight.dtype == input.dtype && output.dtype == input.dtype &&
                    (!bias || bias->dtype == input.dtype),
                "HIP NN convolution requires matching Float32 or Float16 types");
        const auto& g = p.geometry;
        LfsHipConv native{};
        native.batch = p.batch;
        native.out_channels = p.out_channels;
        native.groups = p.groups;
        native.dtype = uint32_t(input.dtype);
        native.transpose = p.transpose;
        native.activation = p.activation;
        auto& n = native.geometry;
        n.channels = g.channels;
        n.height = g.height;
        n.width = g.width;
        n.out_height = g.out_height;
        n.out_width = g.out_width;
        n.kernel_h = g.kernel_h;
        n.kernel_w = g.kernel_w;
        n.stride_h = g.stride_h;
        n.stride_w = g.stride_w;
        n.pad_h = g.pad_h;
        n.pad_w = g.pad_w;
        n.dilation_h = g.dilation_h;
        n.dilation_w = g.dilation_w;
        n.mode = g.mode;
        std::lock_guard lock(impl_->mutex);
        const auto view = [&](StorageRef storage) { return storage.data ? impl_->view(storage) : (impl_->empty_view(storage), LfsHipView{}); };
        const LfsHipView inputs[]{view(input), view(weight), bias ? view(*bias) : LfsHipView{}};
        LfsHipError error{};
        const auto status = impl_->api.nn_conv(impl_->context, inputs, view(output), &native, &error);
        impl_->refresh_algorithm_workspace();
        check(status, error);
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    bool HipBackendOps::triangle_tree_parity(StorageRef points, StorageRef visit, StorageRef triangles, StorageRef boxes, StorageRef output, const PointTreeProgram& p, ExecContext execution) {
        queue(execution);
        require(points.dtype == DataType::Float32 && triangles.dtype == DataType::Float32 && boxes.dtype == DataType::Float32 &&
                    visit.dtype == DataType::Int32 && output.dtype == DataType::Int32,
                "Invalid HIP triangle-tree dtypes");
        LfsHipPointTree native{};
        native.points = p.points;
        native.references = p.references;
        native.levels = p.levels;
        std::copy_n(p.level_offset, 8, native.level_offset);
        std::copy_n(p.level_count, 8, native.level_count);
        std::lock_guard lock(impl_->mutex);
        const auto view = [&](StorageRef value) { return value.data ? impl_->view(value) : (impl_->empty_view(value), LfsHipView{}); };
        const LfsHipView inputs[]{view(points), view(triangles), view(boxes), view(visit), {}, {}};
        LfsHipError error{};
        check(impl_->api.point_tree_query(impl_->context, inputs, view(output), &native, 2, &error), error);
        return true;
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    bool HipBackendOps::simplify_merge(const std::array<StorageRef, 5>& rows, StorageRef offsets, StorageRef members, const std::array<StorageRef, 5>& outputs, const SimplifyMergeProgram& p, ExecContext execution) {
        queue(execution);
        for (size_t i = 0; i < 5; ++i)
            require(rows[i].dtype == DataType::Float32 && outputs[i].dtype == DataType::Float32, "HIP simplify rows require Float32");
        require(offsets.dtype == DataType::Int32 && members.dtype == DataType::Int32, "HIP simplify grouping requires Int32");
        std::lock_guard lock(impl_->mutex);
        const auto view = [&](StorageRef value) { return value.data ? impl_->view(value) : (impl_->empty_view(value), LfsHipView{}); };
        LfsHipView inputs[7]{}, out[5]{};
        for (unsigned i = 0; i < 5; ++i) {
            inputs[i] = view(rows[i]);
            out[i] = view(outputs[i]);
        }
        inputs[5] = view(offsets);
        inputs[6] = view(members);
        LfsHipError error{};
        const auto status = impl_->api.simplify_merge(impl_->context, inputs, out, p.groups, p.app_dim, &error);
        impl_->refresh_algorithm_workspace();
        check(status, error);
        return true;
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    std::unique_ptr<BackendCompletion> HipBackendOps::create_completion(BackendCompletionTarget target) {
        if (target.external_device || target.external_point.semaphore)
            return {};
        ensure_initialized();
        class Completion final : public BackendCompletion {
            HipBackendOps& ops_;
            std::vector<std::pair<std::unique_ptr<ReadbackBuffer>, bool>> markers_;

        public:
            explicit Completion(HipBackendOps& ops) : ops_(ops) {}
            void capture(std::span<const Tensor* const>) override { record({}, 0); }
            void record(ExecContext execution, uint64_t) override {
                auto marker = ops_.create_readback_buffer();
                marker->enqueue({GpuBackend::HIP, nullptr, 0, DataType::Float32, nullptr}, 0, execution);
                markers_.emplace_back(std::move(marker), false);
            }
            VulkanTimelinePoint timeline() const override { return {}; }
            bool ready() override {
                for (auto& [marker, complete] : markers_) {
                    if (!complete)
                        complete = marker->poll(nullptr);
                    if (!complete)
                        return false;
                }
                return true;
            }
            void wait() override {
                std::exception_ptr failure;
                for (auto& [marker, complete] : markers_) {
                    if (complete)
                        continue;
                    try {
                        marker->wait();
                        complete = marker->poll(nullptr);
                    } catch (...) {
                        // LFS-CENSUS-OK(empty-catch): Settle every zero-byte error snapshot before propagating its first failure.
                        if (!failure)
                            failure = std::current_exception();
                    }
                }
                markers_.clear();
                if (failure)
                    std::rethrow_exception(failure);
            }
        };
        return std::make_unique<Completion>(*this);
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    void HipBackendOps::name_queue(ExecContext, const char*) {}
} // namespace lfs::core::internal
