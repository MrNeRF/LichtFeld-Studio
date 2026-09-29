/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/sh_value_quant_kernels.hpp"
#include "core/tensor.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/ops/sh_cuda.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    namespace ops = lfs::gpu_ops;
    namespace quant = lfs::core::sh_value_quant;

    constexpr size_t kN = 8;
    constexpr uint32_t kRest = 3;

    Tensor pattern(const lfs::core::TensorShape& shape, const float scale, const int seed) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i) {
            const auto k = static_cast<int>((i * 7919u + static_cast<size_t>(seed) * 104729u) % 2003u);
            values[i] = scale * (static_cast<float>(k) / 1001.0f - 1.0f);
        }
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    Tensor i32_rows(const std::vector<int>& values) {
        auto cpu = Tensor::empty({values.size()}, Device::CPU, DataType::Int32);
        for (size_t i = 0; i < values.size(); ++i) {
            cpu.ptr<int>()[i] = values[i];
        }
        return cpu.gpu();
    }

    Tensor i64_rows(const std::vector<int64_t>& values) {
        auto cpu = Tensor::empty({values.size()}, Device::CPU, DataType::Int64);
        for (size_t i = 0; i < values.size(); ++i) {
            cpu.ptr<int64_t>()[i] = values[i];
        }
        return cpu.gpu();
    }

    std::vector<uint8_t> bytes_of(const Tensor& tensor) {
        const auto cpu = tensor.cpu().contiguous();
        const auto* data = static_cast<const uint8_t*>(cpu.data_ptr());
        return {data, data + cpu.bytes()};
    }

    std::vector<uint8_t> device_bytes(const void* data, const size_t count) {
        std::vector<uint8_t> out(count);
        EXPECT_EQ(cudaMemcpy(out.data(), data, count, cudaMemcpyDeviceToHost), cudaSuccess);
        return out;
    }

    void expect_same(const Tensor& actual, const Tensor& expected, const std::string& what) {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(bytes_of(actual), bytes_of(expected)) << what;
    }

    void expect_changed(const Tensor& tensor, const std::vector<uint8_t>& before, const std::string& what) {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_NE(bytes_of(tensor), before) << what;
    }

    Tensor swizzled(const size_t n, const uint32_t rest, const int seed) {
        return pattern({lfs::core::sh_swizzled_float_count(n, rest)}, 0.25f, seed);
    }

    struct Q16 {
        Tensor codes;
        Tensor bounds;

        static Q16 make(const size_t n, const uint32_t rest) {
            return {
                Tensor::zeros({quant::sh_value_u16_count(n, rest)}, Device::GPU, DataType::Float16),
                Tensor::zeros({quant::n_bounds_for_prims(n) * 2}, Device::GPU),
            };
        }

        [[nodiscard]] Q16 clone() const { return {codes.clone(), bounds.clone()}; }

        void expect_same_q16(const Q16& expected, const std::string& what) const {
            expect_same(codes, expected.codes, what + " codes");
            expect_same(bounds, expected.bounds, what + " bounds");
        }
    };

    ops::ShRowsParams rows(const size_t n, const size_t count, const size_t offset, const uint32_t rest) {
        return {.source_rows = n,
                .count = count,
                .destination_offset = offset,
                .source_rest = rest,
                .destination_rest = rest};
    }

    void encode(Q16& dst, const Tensor& src, const size_t n, const uint32_t rest,
                const size_t code_offset, const size_t bounds_offset) {
        quant::encode_shN_float4_to_u16(
            src.ptr<float>(),
            reinterpret_cast<uint16_t*>(dst.codes.data_ptr()) + code_offset,
            dst.bounds.ptr<float>() + bounds_offset,
            n, rest, lfs::core::getCurrentCUDAStream());
    }

    const ops::ShOps& table() { return lfs::training::cuda_sh_ops(); }

} // namespace

using lfs::test::CudaBackendTest;

TEST_F(CudaBackendTest, ShOpsMatchDirectLaunchers) {
    const auto stream = lfs::core::getCurrentCUDAStream();
    const ops::ShOps& sh = table();
    const auto indices = i64_rows({1, 5, 2});
    const auto indices_i32 = i32_rows({1, 5, 2});

    {
        SCOPED_TRACE("encode/decode q16");
        const auto src = swizzled(kN, kRest, 3);
        Q16 expected = Q16::make(kN, kRest);
        Q16 actual = expected.clone();
        const auto before = bytes_of(actual.codes);
        encode(expected, src, kN, kRest, 0, 0);
        sh.encode_q16(src, actual.codes, actual.bounds, kN, kRest, 0, 0);
        expect_changed(actual.codes, before, "encode_q16");
        actual.expect_same_q16(expected, "encode_q16");

        auto decoded_expected = Tensor::zeros({lfs::core::sh_swizzled_float_count(kN, kRest)}, Device::GPU);
        auto decoded_actual = decoded_expected.clone();
        quant::decode_shN_u16_to_float4(
            reinterpret_cast<const uint16_t*>(expected.codes.data_ptr()),
            expected.bounds.ptr<float>(), decoded_expected.ptr<float>(), kN, kRest, stream);
        sh.decode_q16(actual.codes, actual.bounds, decoded_actual, kN, kRest);
        expect_changed(decoded_actual, bytes_of(Tensor::zeros(decoded_actual.shape(), Device::GPU)), "decode_q16");
        expect_same(decoded_actual, decoded_expected, "decode_q16");
    }

    {
        SCOPED_TRACE("encode_q16 offset");
        constexpr size_t kBlock = 256;
        constexpr size_t kChunk = 32;
        constexpr size_t kTotal = kBlock + kChunk;
        const auto src = swizzled(kChunk, kRest, 9);
        const size_t code_offset = quant::sh_value_u16_count(kBlock, kRest);
        const size_t bounds_offset = quant::n_bounds_for_prims(kBlock) * 2;
        Q16 expected = Q16::make(kTotal, kRest);
        expected.codes.copy_from(pattern(expected.codes.shape(), 3.0f, 11).to(DataType::Float16));
        expected.bounds.copy_from(pattern(expected.bounds.shape(), 0.5f, 12));
        Q16 actual = expected.clone();
        encode(expected, src, kChunk, kRest, code_offset, bounds_offset);
        sh.encode_q16(src, actual.codes, actual.bounds, kChunk, kRest, code_offset, bounds_offset);
        actual.expect_same_q16(expected, "encode_q16 offset");
    }

    {
        SCOPED_TRACE("block ids and runs");
        const auto dest = i64_rows({0, 256, 257, 512, 256});
        auto ids_expected = Tensor::zeros({dest.numel()}, Device::GPU);
        auto ids_actual = ids_expected.clone();
        quant::fill_quant_block_ids_f32(
            dest.ptr<int64_t>(), ids_expected.ptr<float>(), dest.numel(), stream);
        sh.block_ids(dest, ids_actual);
        expect_changed(ids_actual, bytes_of(Tensor::zeros(ids_actual.shape(), Device::GPU)), "block_ids");
        expect_same(ids_actual, ids_expected, "block_ids");

        auto sorted = ids_actual.sort(0, false).first.contiguous();
        const size_t n = sorted.numel();
        auto unique_expected = Tensor::empty({n}, Device::GPU, DataType::Int32);
        auto offsets_expected = Tensor::empty({n}, Device::GPU, DataType::Int32);
        auto runs_expected = Tensor::zeros({size_t{1}}, Device::GPU, DataType::Int32);
        auto unique_actual = unique_expected.clone();
        auto offsets_actual = offsets_expected.clone();
        auto runs_actual = runs_expected.clone();

        const size_t scan_bytes = quant::sorted_block_runs_scan_workspace_bytes(n, stream);
        Tensor flags = Tensor::empty({n}, Device::GPU, DataType::Int32);
        Tensor compact = Tensor::empty({n}, Device::GPU, DataType::Int32);
        Tensor scan = Tensor::empty({scan_bytes}, Device::GPU, DataType::UInt8);
        const quant::SortedBlockRunScratch scratch{
            .flags = flags.ptr<int32_t>(),
            .compact = compact.ptr<int32_t>(),
            .scan = scan.data_ptr(),
            .scan_bytes = scan_bytes,
        };
        quant::build_sorted_block_runs(
            sorted.ptr<float>(), unique_expected.ptr<int>(), offsets_expected.ptr<int>(),
            runs_expected.ptr<int>(), n, stream, &scratch);
        auto state = sh.create_run_scratch();
        sh.block_runs(*state, sorted, unique_actual, offsets_actual, runs_actual);
        expect_same(runs_actual, runs_expected, "block_runs count");
        const int n_runs = runs_expected.cpu().ptr<int>()[0];
        ASSERT_GT(n_runs, 0);
        expect_same(unique_actual.slice(0, 0, static_cast<size_t>(n_runs)),
                    unique_expected.slice(0, 0, static_cast<size_t>(n_runs)), "block_runs ids");
        expect_same(offsets_actual.slice(0, 0, static_cast<size_t>(n_runs)),
                    offsets_expected.slice(0, 0, static_cast<size_t>(n_runs)), "block_runs offsets");
    }

    Q16 stored = Q16::make(kN, kRest);
    sh.encode_q16(swizzled(kN, kRest, 4), stored.codes, stored.bounds, kN, kRest, 0, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    {
        SCOPED_TRACE("reencode_touched");
        auto canonical = pattern({indices.numel(), size_t{kRest}, size_t{3}}, 0.8f, 21);
        auto block_ids = Tensor::zeros({indices.numel()}, Device::GPU);
        sh.block_ids(indices, block_ids);
        auto sorted = block_ids.sort(0, false);
        Tensor order = std::move(sorted.second);
        Tensor sorted_ids = sorted.first.contiguous();
        Tensor sorted_dest = indices.index_select(0, order).contiguous();
        Tensor sorted_can = canonical.index_select(0, order).contiguous();
        auto unique = Tensor::empty({indices.numel()}, Device::GPU, DataType::Int32);
        auto offsets = Tensor::empty({indices.numel()}, Device::GPU, DataType::Int32);
        auto runs = Tensor::zeros({size_t{1}}, Device::GPU, DataType::Int32);
        auto state = sh.create_run_scratch();
        sh.block_runs(*state, sorted_ids, unique, offsets, runs);

        Q16 expected = stored.clone();
        Q16 actual = stored.clone();
        const auto before = bytes_of(actual.codes);
        const ops::Q16TouchParams touch{
            .sorted_count = indices.numel(),
            .primitives = kN,
            .decode_source_rows = kN,
            .rest = kRest};
        quant::reencode_touched_q16_blocks(
            reinterpret_cast<uint16_t*>(expected.codes.data_ptr()), expected.bounds.ptr<float>(),
            sorted_can.ptr<float>(), sorted_dest.ptr<int64_t>(), unique.ptr<int>(), offsets.ptr<int>(),
            runs.ptr<int>(), touch.sorted_count, touch.primitives, touch.decode_source_rows, touch.rest, stream);
        sh.reencode_touched(actual.codes, actual.bounds, sorted_can, sorted_dest, unique, offsets, runs, {}, touch);
        expect_changed(actual.codes, before, "reencode_touched");
        actual.expect_same_q16(expected, "reencode_touched");
    }

    {
        SCOPED_TRACE("decode_range");
        const auto sw = swizzled(kN, kRest, 6);
        constexpr uint64_t kCount = 6;
        auto q16_expected = Tensor::empty({size_t{kCount}}, Device::GPU);
        auto q16_actual = q16_expected.clone();
        const ops::ShRangeParams q16_range{
            .canonical_float_offset = 3,
            .float_count = kCount,
            .primitives = kN,
            .destination_rest = kRest,
            .layout_rest = kRest,
            .storage = ops::ShStorage::Q16};
        quant::decode_shN_u16_range_to_canonical(
            reinterpret_cast<const uint16_t*>(stored.codes.data_ptr()), stored.bounds.ptr<float>(),
            q16_expected.ptr<float>(), q16_range.canonical_float_offset, q16_range.float_count,
            q16_range.primitives, q16_range.destination_rest, q16_range.layout_rest, stream);
        sh.decode_range(stored.codes, stored.bounds, q16_actual, q16_range);
        expect_same(q16_actual, q16_expected, "decode_range q16");

        auto f32_expected = q16_expected.clone();
        auto f32_actual = q16_actual.clone();
        auto f32_range = q16_range;
        f32_range.storage = ops::ShStorage::Float32;
        lfs::core::undo_reorder_sh_range_from_swizzled(
            sw.ptr<float>(), f32_expected.ptr<float>(), f32_range.canonical_float_offset, f32_range.float_count,
            f32_range.primitives, f32_range.destination_rest, f32_range.layout_rest, stream);
        sh.decode_range(sw, Tensor{}, f32_actual, f32_range);
        expect_same(f32_actual, f32_expected, "decode_range f32");

        auto half = sw.to(DataType::Float16);
        auto f16_expected = q16_expected.clone();
        auto f16_actual = q16_actual.clone();
        auto f16_range = q16_range;
        f16_range.storage = ops::ShStorage::IeeeFloat16;
        quant::decode_shN_f16_range_to_canonical(
            static_cast<const uint16_t*>(half.data_ptr()), f16_expected.ptr<float>(),
            f16_range.canonical_float_offset, f16_range.float_count, f16_range.primitives,
            f16_range.destination_rest, f16_range.layout_rest, stream);
        sh.decode_range(half, Tensor{}, f16_actual, f16_range);
        expect_same(f16_actual, f16_expected, "decode_range f16");
    }

    {
        SCOPED_TRACE("swizzled row ops");
        const auto params = rows(kN, indices.numel(), 0, kRest);
        auto src = swizzled(kN, kRest, 15);

        auto zero_expected = src.clone();
        auto zero_actual = src.clone();
        const auto zero_before = bytes_of(zero_actual);
        lfs::core::shN_swizzled_zero_at_indices(
            zero_expected.ptr<float>(), indices_i32.ptr<int>(), indices_i32.numel(), kRest, stream);
        sh.zero_rows(zero_actual, indices_i32, kRest);
        expect_changed(zero_actual, zero_before, "zero_rows");
        expect_same(zero_actual, zero_expected, "zero_rows");

        auto gathered_expected = Tensor::zeros({lfs::core::sh_swizzled_float_count(indices.numel(), kRest)}, Device::GPU);
        auto gathered_actual = gathered_expected.clone();
        lfs::core::shN_swizzled_gather_self(
            src.ptr<float>(), gathered_expected.ptr<float>(), indices_i32.ptr<int>(),
            params.count, params.destination_offset, kRest, stream);
        sh.gather_swizzled(src, indices_i32, gathered_actual, params);
        expect_changed(gathered_actual, bytes_of(Tensor::zeros(gathered_actual.shape(), Device::GPU)), "gather_swizzled i32");
        expect_same(gathered_actual, gathered_expected, "gather_swizzled i32");

        auto i64_expected = gathered_expected.clone();
        auto i64_actual = gathered_actual.clone();
        lfs::core::shN_swizzled_gather_self_i64(
            src.ptr<float>(), i64_expected.ptr<float>(), indices.ptr<int64_t>(),
            params.count, 0, kRest, stream);
        sh.gather_swizzled(src, indices, i64_actual, params);
        expect_same(i64_actual, i64_expected, "gather_swizzled i64");

        auto bytes_src = pattern({src.numel()}, 40.0f, 16).abs().to(DataType::UInt8);
        auto u8_expected = Tensor::zeros(bytes_src.shape(), Device::GPU, DataType::UInt8);
        auto u8_actual = u8_expected.clone();
        lfs::core::shN_swizzled_gather_self_u8(
            bytes_src.ptr<uint8_t>(), u8_expected.ptr<uint8_t>(), indices_i32.ptr<int>(),
            params.count, 0, kRest, stream);
        sh.gather_swizzled(bytes_src, indices_i32, u8_actual, params);
        expect_same(u8_actual, u8_expected, "gather_swizzled u8");

        constexpr size_t kWide = 64;
        auto wide = swizzled(kWide, kRest, 18);
        auto wide_expected = wide.clone();
        auto wide_actual = wide.clone();
        const auto tail = rows(kWide, 2, 40, kRest);
        const auto head_i32 = i32_rows({0, 1});
        lfs::core::shN_swizzled_gather_self(
            wide_expected.ptr<float>(), wide_expected.ptr<float>(), head_i32.ptr<int>(),
            tail.count, tail.destination_offset, kRest, stream);
        sh.gather_swizzled(wide_actual, head_i32, wide_actual, tail);
        expect_same(wide_actual, wide_expected, "gather_swizzled in place");

        auto canonical_expected = Tensor::zeros({indices.numel(), size_t{kRest}, size_t{3}}, Device::GPU);
        auto canonical_actual = canonical_expected.clone();
        lfs::core::shN_swizzled_gather_to_linear_i64(
            src.ptr<float>(), indices.ptr<int64_t>(), canonical_expected.ptr<float>(),
            params.count, kRest, kRest, stream);
        sh.gather_canonical(src, indices, canonical_actual, params);
        expect_changed(canonical_actual, bytes_of(Tensor::zeros(canonical_actual.shape(), Device::GPU)), "gather_canonical");
        expect_same(canonical_actual, canonical_expected, "gather_canonical");

        auto appended_expected = Tensor::zeros({lfs::core::sh_swizzled_float_count(kN, kRest)}, Device::GPU);
        auto appended_actual = appended_expected.clone();
        const auto append_at = rows(indices.numel(), indices.numel(), 2, kRest);
        lfs::core::shN_swizzled_gather_from_linear(
            appended_expected.ptr<float>(), append_at.destination_offset, canonical_expected.ptr<float>(),
            append_at.count, kRest, kRest, stream);
        sh.append_canonical(canonical_actual, appended_actual, append_at);
        expect_same(appended_actual, appended_expected, "append_canonical");

        auto scattered_expected = src.clone();
        auto scattered_actual = src.clone();
        lfs::core::shN_swizzled_scatter_linear(
            scattered_expected.ptr<float>(), indices_i32.ptr<int>(), canonical_expected.ptr<float>(),
            params.count, kRest, kRest, stream);
        sh.scatter_canonical(canonical_actual, indices_i32, scattered_actual, params);
        expect_same(scattered_actual, scattered_expected, "scatter_canonical");
    }

    {
        SCOPED_TRACE("fill_bytes");
        constexpr size_t kLogical = 4;
        constexpr size_t kCapacity = 8;
        auto expected = Tensor::zeros_direct({kLogical}, kCapacity, Device::GPU, DataType::UInt8);
        auto actual = Tensor::zeros_direct({kLogical}, kCapacity, Device::GPU, DataType::UInt8);
        ASSERT_EQ(cudaMemsetAsync(expected.data_ptr(), 0xA5, kCapacity, stream), cudaSuccess);
        sh.fill_bytes(actual, kCapacity, 0xA5);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(device_bytes(actual.data_ptr(), kCapacity), device_bytes(expected.data_ptr(), kCapacity));
    }
}

TEST(ShOpsCapability, CompiledBackendsCarrySh) {
    EXPECT_EQ(lfs::training::training_ops(lfs::core::GpuBackend::CUDA).sh, &lfs::training::cuda_sh_ops());
#if defined(LFS_TEST_TENSOR_VULKAN)
    EXPECT_NE(lfs::training::training_ops(lfs::core::GpuBackend::Vulkan).sh, nullptr);
#else
    EXPECT_EQ(lfs::training::training_ops(lfs::core::GpuBackend::Vulkan).sh, nullptr);
#endif
    EXPECT_EQ(lfs::training::training_ops(lfs::core::GpuBackend::Metal).sh, nullptr);
    EXPECT_FALSE(lfs::training::unavailable_training_family(
        lfs::core::GpuBackend::CUDA, lfs::training::Family::Sh));
#if defined(LFS_TEST_TENSOR_VULKAN)
    EXPECT_FALSE(lfs::training::unavailable_training_family(lfs::core::GpuBackend::Vulkan, lfs::training::Family::Sh));
#else
    EXPECT_EQ(
        lfs::training::unavailable_training_family(lfs::core::GpuBackend::Vulkan, lfs::training::Family::Sh),
        "Vulkan training is unavailable for this configuration.\nMissing families: Sh.");
#endif
}
