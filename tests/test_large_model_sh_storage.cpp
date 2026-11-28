/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The training paths through the SH codec and the training SH kernels past the 32-bit wraps of SH degree 3, on a
// model of 95,444,224 Gaussians (large_model_test_support.hpp has the wrap rows and the blocks of rows checked):
// - the model-level helpers of sh_value_storage.cpp that densification, MRNF and MCMC call: the fp32 expansion
//   before a mutation and the commit after it, the compaction gather, the Q16 gather, the canonical gather, the
//   scatter, the append and a mutation batch;
// - the training ShOps (sh.metal and its twins): the Q16 encode and decode of a whole store and the canonical
//   gather;
// - Morton's Q16 reorder.
// Each result is compared with the CPU codec (64-bit offsets) on the compact store of the blocks; where a helper
// decodes, overwrites and re-encodes 256-row blocks, the reference does the same on the CPU.
//
// Tiers by device memory: see large_model_test_support.hpp. Each test's memory is in its skip message.

#include "large_model_test_support.hpp"

#include "core/splat_data.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_execution.hpp"
#include "lfs/training/idle_arena_scratch.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::ShFormat;
    using lfs::core::SplatData;
    using lfs::core::Tensor;
    namespace lm = lfs::test::large_model;
    namespace sh_value = lfs::training::sh_value;
    namespace ops = lfs::gpu_ops;

    // Means, sh0, scaling, rotation and opacity: 14 floats a Gaussian.
    size_t attribute_bytes(const size_t rows) { return rows * 14 * sizeof(float); }

    std::vector<size_t> shuffled_rows(const std::initializer_list<size_t> block_numbers, const unsigned seed) {
        std::vector<size_t> rows;
        for (const size_t b : block_numbers)
            for (size_t r = lm::kBlocks[b].begin; r < lm::kBlocks[b].end; ++r)
                rows.push_back(r);
        std::mt19937 rng(seed);
        std::shuffle(rows.begin(), rows.end(), rng);
        return rows;
    }

    std::vector<size_t> compact_rows(const std::vector<size_t>& rows) {
        std::vector<size_t> compact(rows.size());
        std::transform(rows.begin(), rows.end(), compact.begin(), lm::compact_row);
        return compact;
    }

    // A device canonical [rows.size(), 15, 3] tensor whose row i holds fill(rows[i], cell).
    Tensor canonical_tensor(const std::vector<size_t>& rows, float (*const fill)(size_t, uint32_t)) {
        Tensor host = Tensor::empty({rows.size(), size_t{lm::kRest}, size_t{3}}, Device::CPU, DataType::Float32);
        auto* const data = host.ptr<float>();
        for (size_t i = 0; i < rows.size(); ++i)
            for (uint32_t c = 0; c < lm::kCells; ++c)
                data[i * lm::kCells + c] = fill(rows[i], c);
        return host.gpu();
    }

    class LargeModelShStorage : public ::testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << lfs::core::gpu_backend_name(GetParam()) << " device unavailable";
            session_.emplace(GetParam());
            ASSERT_TRUE(session_->switched());
            const auto& table = lfs::training::training_ops(GetParam());
            sh_ = table.sh;
            morton_ = table.morton;
            sh_value::set_sh_value_quant_enabled_for_testing(true);
        }

        void TearDown() override {
            sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
            session_.reset();
        }

        // A zeroed device store, or nullopt with the reason in unavailable_.
        std::optional<lm::Store> device_store(const ShFormat format, const size_t rows) {
            auto store = lm::zero_store(format, rows, Device::GPU);
            if (!store)
                unavailable_ = std::format("this device can't allocate or clear {} rows of {} ({:.1f} GB)", rows,
                                           lm::format_name(format), lm::store_bytes(format, rows) / lm::kGB);
            return store;
        }

        // A device store of the blocks in `format`, for a model of `rows` Gaussians.
        std::optional<lm::Store> device_blocks(const ShFormat format) {
            auto store = device_store(format, lm::kRows);
            if (store)
                lm::place_blocks(*store, lm::compact_blocks(format));
            return store;
        }

        // A model of `rows` Gaussians at SH degree 3 with zeroed attributes and `store` as its SH rest (fp32 float4,
        // or Q16 codes with their bounds), or nullopt with the reason in unavailable_.
        std::optional<SplatData> model(lm::Store store, const size_t rows) {
            try {
                SplatData splat(3, Tensor::zeros({rows, 3}, Device::GPU), Tensor::zeros({rows, 1, 3}, Device::GPU),
                                store.values, Tensor::zeros({rows, 3}, Device::GPU),
                                Tensor::zeros({rows, 4}, Device::GPU), Tensor::zeros({rows, 1}, Device::GPU), 1.0f,
                                SplatData::ShNLayout::Swizzled);
                if (store.format == ShFormat::Q16)
                    splat.shN_value_bounds() = store.bounds;
                return splat;
            } catch (const std::exception& e) {
                unavailable_ = std::format("can't build a model of {} Gaussians: {}", rows, e.what());
                return std::nullopt;
            }
        }

        static lm::Store q16_of(const SplatData& splat, const size_t rows) {
            return {.format = ShFormat::Q16, .rows = rows, .values = splat.shN(), .bounds = splat.shN_value_bounds()};
        }

        std::optional<lfs::test::DefaultGpuBackendForTesting> session_;
        const ops::ShOps* sh_ = nullptr;
        const ops::MortonOps* morton_ = nullptr;
        std::string unavailable_;
    };

#define LFS_LARGE_MODEL_NEEDS(bytes)                         \
    do {                                                     \
        if (const auto reason = lm::skip_reason(bytes))      \
            GTEST_SKIP() << *reason;                         \
    } while (false)

#define LFS_LARGE_MODEL_OR_SKIP(optional) \
    do {                                  \
        if (!(optional))                  \
            GTEST_SKIP() << unavailable_; \
    } while (false)

    // ensure_shN_fp32_for_mutation (before a densification, a GUI transform bake, Morton's fp32 path) decodes the
    // whole Q16 store into fp32 float4 rows: the codec's destination enumeration and its float4 offsets pass 2^32,
    // and so do the Q16 source offsets.
    TEST_P(LargeModelShStorage, ExpandForMutationDecodesEveryRow) {
        const size_t fp32_bytes = lm::store_bytes(ShFormat::Float32, lm::kRows);
        LFS_LARGE_MODEL_NEEDS(lm::store_bytes(ShFormat::Q16, lm::kRows) + fp32_bytes + attribute_bytes(lm::kRows));
        // The expansion allocates and writes the fp32 store itself; a device that can't clear one that size (MoltenVK
        // stops at vkEndCommandBuffer) can't run it.
        if (!lm::zero_store(ShFormat::Float32, lm::kRows, Device::GPU))
            GTEST_SKIP() << std::format("this device can't clear a {:.1f} GB fp32 store, which the expansion writes",
                                        static_cast<double>(fp32_bytes) / lm::kGB);
        auto store = device_blocks(ShFormat::Q16);
        LFS_LARGE_MODEL_OR_SKIP(store);
        auto splat = model(std::move(*store), lm::kRows);
        LFS_LARGE_MODEL_OR_SKIP(splat);
        {
            lfs::training::LiveModelMutationGuard guard("ExpandForMutationDecodesEveryRow");
            ASSERT_TRUE(sh_value::ensure_shN_fp32_for_mutation(*splat));
        }
        ASSERT_EQ(splat->shN().dtype(), DataType::Float32);
        ASSERT_EQ(splat->shN().numel(), lm::value_elements(ShFormat::Float32, lm::kRows));
        lm::Store reference = lm::cpu_store(ShFormat::Float32, lm::kCompactRows);
        lm::convert(lm::compact_blocks(ShFormat::Q16), reference, {.count = lm::kCompactRows});
        lm::expect_blocks_match({.format = ShFormat::Float32, .rows = lm::kRows, .values = splat->shN()}, reference,
                                "ensure_shN_fp32_for_mutation");
    }

    // commit_shN_after_mutation (after every densification, MRNF's growth and retirement) encodes the whole fp32
    // store into Q16: float4 source offsets and Q16 destination offsets past 2^32.
    TEST_P(LargeModelShStorage, CommitAfterMutationEncodesEveryRow) {
        LFS_LARGE_MODEL_NEEDS(lm::store_bytes(ShFormat::Q16, lm::kRows) + lm::store_bytes(ShFormat::Float32, lm::kRows) +
                              attribute_bytes(lm::kRows));
        auto store = device_blocks(ShFormat::Float32);
        LFS_LARGE_MODEL_OR_SKIP(store);
        auto splat = model(std::move(*store), lm::kRows);
        LFS_LARGE_MODEL_OR_SKIP(splat);
        {
            lfs::training::LiveModelMutationGuard guard("CommitAfterMutationEncodesEveryRow");
            ASSERT_TRUE(sh_value::commit_shN_after_mutation(*splat));
        }
        ASSERT_TRUE(splat->shN_value_quantized());
        lm::Store reference = lm::cpu_store(ShFormat::Q16, lm::kCompactRows);
        lm::convert(lm::compact_blocks(ShFormat::Float32), reference, {.count = lm::kCompactRows});
        lm::expect_blocks_match(q16_of(*splat, lm::kRows), reference, "commit_shN_after_mutation");
    }

    // compact_shN_gather (MRNF and MCMC compaction, post-training compaction) re-encodes the kept rows Q16 to Q16.
    // Keeping every row from 256 on moves each block down by one Q16 block, so both the source and the destination
    // offsets pass 2^32.
    TEST_P(LargeModelShStorage, CompactGatherKeepsRowsPastTheWrap) {
        constexpr size_t kDropped = 256;
        constexpr size_t kKept = lm::kRows - kDropped;
        static_assert(kKept > lm::kQ16WrapRow + lm::kReorder);
        LFS_LARGE_MODEL_NEEDS(lm::store_bytes(ShFormat::Q16, lm::kRows) + lm::store_bytes(ShFormat::Q16, kKept) +
                              kKept * sizeof(int64_t) + attribute_bytes(lm::kRows));
        auto store = device_blocks(ShFormat::Q16);
        LFS_LARGE_MODEL_OR_SKIP(store);
        auto splat = model(std::move(*store), lm::kRows);
        LFS_LARGE_MODEL_OR_SKIP(splat);
        Tensor keep;
        {
            Tensor host = Tensor::empty({kKept}, Device::CPU, DataType::Int64);
            std::iota(host.ptr<int64_t>(), host.ptr<int64_t>() + kKept, static_cast<int64_t>(kDropped));
            keep = host.gpu();
        }
        {
            lfs::training::LiveModelMutationGuard guard("CompactGatherKeepsRowsPastTheWrap");
            sh_value::compact_shN_gather(*splat, keep, lm::kRows);
        }
        keep = Tensor{};
        ASSERT_TRUE(splat->shN_value_quantized());
        ASSERT_EQ(splat->shN().numel(), lm::value_elements(ShFormat::Q16, kKept));

        // The reference keeps the compact rows of Z, F and Q, which land at the kept rows' places.
        std::vector<size_t> compact(lm::kCompactRows - kDropped);
        std::iota(compact.begin(), compact.end(), kDropped);
        lm::Store reference = lm::cpu_store(ShFormat::Q16, compact.size());
        const Tensor indices = lm::index_tensor<int64_t>(compact, Device::CPU);
        lm::convert(lm::compact_blocks(ShFormat::Q16), reference, {.count = compact.size()}, &indices);
        const lm::Store actual = q16_of(*splat, kKept);
        for (size_t b = 1; b < lm::kBlocks.size(); ++b) {
            const auto& block = lm::kBlocks[b];
            lm::expect_rows_match(actual, block.begin - kDropped, reference, block.compact - kDropped, block.rows(),
                                  std::string("compact_shN_gather, block ") + block.name + " moved down by 256 rows");
        }
    }

    // gather_shN_q16 (MRNF's retirement, MCMC's relocation) gathers far rows Q16 to Q16 into a small store.
    TEST_P(LargeModelShStorage, GatherQ16ReadsRowsPastTheWrap) {
        LFS_LARGE_MODEL_NEEDS(lm::store_bytes(ShFormat::Q16, lm::kRows) + attribute_bytes(lm::kRows));
        auto store = device_blocks(ShFormat::Q16);
        LFS_LARGE_MODEL_OR_SKIP(store);
        auto splat = model(std::move(*store), lm::kRows);
        LFS_LARGE_MODEL_OR_SKIP(splat);
        const std::vector<size_t> rows = shuffled_rows({0, 2, 3}, 4204);
        std::pair<Tensor, Tensor> gathered;
        {
            lfs::training::LiveModelMutationGuard guard("GatherQ16ReadsRowsPastTheWrap");
            gathered = sh_value::gather_shN_q16(*splat, lm::index_tensor<int64_t>(rows, Device::GPU));
        }
        lm::Store reference = lm::cpu_store(ShFormat::Q16, rows.size());
        const Tensor indices = lm::index_tensor<int64_t>(compact_rows(rows), Device::CPU);
        lm::convert(lm::compact_blocks(ShFormat::Q16), reference, {.count = rows.size()}, &indices);
        lm::expect_store_matches(
            {.format = ShFormat::Q16, .rows = rows.size(), .values = gathered.first, .bounds = gathered.second},
            reference, "gather_shN_q16");
    }

    // gather_shN_to_canonical (the sources of a split or a clone) decodes far Q16 rows into canonical rows.
    TEST_P(LargeModelShStorage, GatherToCanonicalReadsRowsPastTheWrap) {
        LFS_LARGE_MODEL_NEEDS(lm::store_bytes(ShFormat::Q16, lm::kRows) + attribute_bytes(lm::kRows));
        auto store = device_blocks(ShFormat::Q16);
        LFS_LARGE_MODEL_OR_SKIP(store);
        auto splat = model(std::move(*store), lm::kRows);
        LFS_LARGE_MODEL_OR_SKIP(splat);
        const std::vector<size_t> rows = shuffled_rows({0, 2, 3}, 4205);
        Tensor canonical;
        {
            lfs::training::LiveModelMutationGuard guard("GatherToCanonicalReadsRowsPastTheWrap");
            sh_value::gather_shN_to_canonical(*splat, lm::index_tensor<int64_t>(rows, Device::GPU), canonical);
        }
        ASSERT_EQ(canonical.numel(), rows.size() * lm::kCells);
        lm::Store reference = lm::cpu_store(ShFormat::Canonical, rows.size());
        const Tensor indices = lm::index_tensor<int64_t>(compact_rows(rows), Device::CPU);
        lm::convert(lm::compact_blocks(ShFormat::Q16), reference, {.count = rows.size()}, &indices);
        lm::expect_store_matches({.format = ShFormat::Canonical, .rows = rows.size(), .values = canonical.flatten()},
                                 reference, "gather_shN_to_canonical");
    }

    // scatter_canonical_into_shN (a densification's in-place writes) re-encodes the 256-row blocks it touches with
    // the training kernel sh_reencode_touched, whose Q16 cell index passes 2^32 here.
    TEST_P(LargeModelShStorage, ScatterReencodesTouchedBlocksPastTheWrap) {
        LFS_LARGE_MODEL_NEEDS(lm::store_bytes(ShFormat::Q16, lm::kRows) + attribute_bytes(lm::kRows));
        auto store = device_blocks(ShFormat::Q16);
        LFS_LARGE_MODEL_OR_SKIP(store);
        auto splat = model(std::move(*store), lm::kRows);
        LFS_LARGE_MODEL_OR_SKIP(splat);
        // 100 rows across the float4 wrap block, 200 across the Q16 wrap row, in a shuffled order.
        std::vector<size_t> rows;
        for (size_t r = 89'478'450; r < 89'478'550; ++r)
            rows.push_back(r);
        for (size_t r = 95'443'650; r < 95'443'850; ++r)
            rows.push_back(r);
        std::mt19937 rng(4206);
        std::shuffle(rows.begin(), rows.end(), rng);
        {
            lfs::training::LiveModelMutationGuard guard("ScatterReencodesTouchedBlocksPastTheWrap");
            sh_value::scatter_canonical_into_shN(*splat, lm::index_tensor<int64_t>(rows, Device::GPU),
                                                 canonical_tensor(rows, lm::second_value));
        }
        std::vector<std::pair<size_t, lm::CanonicalRow>> writes;
        for (const size_t r : rows)
            writes.emplace_back(lm::compact_row(r), lm::row_of_values(r, lm::second_value));
        lm::expect_blocks_match(q16_of(*splat, lm::kRows), lm::reencoded(lm::compact_blocks(ShFormat::Q16), writes),
                                "scatter_canonical_into_shN");
    }

    // A model of kOld rows, its last Q16 block partly filled, and the device store of its blocks with room to grow
    // to kRows rows in place.
    constexpr size_t kOld = lm::kQ16WrapRow + 100;
    constexpr size_t kAppended = lm::kRows - kOld;

    std::optional<lm::Store> grown_store_of_blocks(const lm::Store& compact) {
        try {
            lm::Store store{.format = ShFormat::Q16, .rows = kOld};
            store.values = Tensor::zeros_direct({lm::value_elements(ShFormat::Q16, kOld)},
                                                lm::value_elements(ShFormat::Q16, lm::kRows), Device::GPU,
                                                DataType::Float16);
            store.bounds = Tensor::zeros_direct({lm::bound_elements(ShFormat::Q16, kOld)},
                                                lm::bound_elements(ShFormat::Q16, lm::kRows), Device::GPU,
                                                DataType::Float32);
            const Tensor values = compact.values.gpu(), bounds = compact.bounds.gpu();
            for (const auto& block : lm::kBlocks) {
                // The last block stops at the model's padded rows and its last bounds block.
                const size_t rows = std::min(block.end, lfs::core::sh_swizzled_padded_n(kOld)) - block.begin;
                lm::values_of_rows(store.values, ShFormat::Q16, block.begin, rows)
                    .copy_(lm::values_of_rows(values, ShFormat::Q16, block.compact, rows));
                const size_t bound_rows =
                    std::min(block.end, lfs::core::sh_value_quant::n_bounds_for_prims(kOld) * lm::kQuantBlock) -
                    block.begin;
                lm::bounds_of_rows(store.bounds, block.begin, bound_rows)
                    .copy_(lm::bounds_of_rows(bounds, block.compact, bound_rows));
            }
            (void)store.values.slice(0, store.values.numel() - 1, store.values.numel()).cpu();
            return store;
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }

    // append_canonical_to_shN (densification's appends, MRNF's growth) at row 95,443,812, in the middle of the Q16
    // block after the wrap: it decodes the partial block through the codec from a far source offset, overlays the
    // new rows and encodes the block at a far code offset.
    TEST_P(LargeModelShStorage, AppendPastTheWrapKeepsEveryRow) {
        LFS_LARGE_MODEL_NEEDS(lm::store_bytes(ShFormat::Q16, lm::kRows) + attribute_bytes(kOld));
        const lm::Store compact = lm::compact_blocks(ShFormat::Q16, kOld);
        auto store = grown_store_of_blocks(compact);
        if (!store)
            GTEST_SKIP() << "this device can't allocate or clear a Q16 store of " << lm::kRows << " rows";
        auto splat = model(std::move(*store), kOld);
        LFS_LARGE_MODEL_OR_SKIP(splat);
        std::vector<size_t> rows(kAppended);
        std::iota(rows.begin(), rows.end(), kOld);
        {
            lfs::training::LiveModelMutationGuard guard("AppendPastTheWrapKeepsEveryRow");
            sh_value::append_canonical_to_shN(*splat, canonical_tensor(rows, lm::second_value), kOld);
        }
        ASSERT_EQ(splat->shN().numel(), lm::value_elements(ShFormat::Q16, lm::kRows));
        std::vector<std::pair<size_t, lm::CanonicalRow>> writes;
        for (const size_t r : rows)
            writes.emplace_back(lm::compact_row(r), lm::row_of_values(r, lm::second_value));
        lm::expect_blocks_match(q16_of(*splat, lm::kRows), lm::reencoded(compact, writes), "append_canonical_to_shN");
    }

    // A mutation batch (one refine event: scatters, zeros and an append) flushes through one re-encode of the
    // touched blocks, with the new rows decoded as zeros.
    TEST_P(LargeModelShStorage, MutationBatchPastTheWrapKeepsEveryRow) {
        LFS_LARGE_MODEL_NEEDS(lm::store_bytes(ShFormat::Q16, lm::kRows) + attribute_bytes(kOld));
        const lm::Store compact = lm::compact_blocks(ShFormat::Q16, kOld);
        auto store = grown_store_of_blocks(compact);
        if (!store)
            GTEST_SKIP() << "this device can't allocate or clear a Q16 store of " << lm::kRows << " rows";
        auto splat = model(std::move(*store), kOld);
        LFS_LARGE_MODEL_OR_SKIP(splat);
        std::vector<size_t> scattered, zeroed, appended(kAppended);
        for (size_t r = 89'478'450; r < 89'478'550; ++r)
            scattered.push_back(r);
        for (size_t r = 95'443'500; r < 95'443'550; ++r)
            zeroed.push_back(r);
        std::iota(appended.begin(), appended.end(), kOld);
        {
            lfs::training::LiveModelMutationGuard guard("MutationBatchPastTheWrapKeepsEveryRow");
            sh_value::ShNMutationBatch batch(*splat);
            batch.scatter(lm::index_tensor<int64_t>(scattered, Device::GPU), canonical_tensor(scattered, lm::second_value));
            batch.zero(lm::index_tensor<int64_t>(zeroed, Device::GPU));
            batch.append(canonical_tensor(appended, lm::second_value), kOld);
            batch.flush();
        }
        ASSERT_EQ(splat->shN().numel(), lm::value_elements(ShFormat::Q16, lm::kRows));
        std::vector<std::pair<size_t, lm::CanonicalRow>> writes;
        for (const size_t r : scattered)
            writes.emplace_back(lm::compact_row(r), lm::row_of_values(r, lm::second_value));
        for (const size_t r : zeroed)
            writes.emplace_back(lm::compact_row(r), lm::CanonicalRow{});
        for (const size_t r : appended)
            writes.emplace_back(lm::compact_row(r), lm::row_of_values(r, lm::second_value));
        lm::expect_blocks_match(q16_of(*splat, lm::kRows), lm::reencoded(compact, writes), "ShNMutationBatch");
    }

    // The training ShOps encode a whole fp32 float4 store into Q16 (sh_encode_q16: Q16 cell index past 2^32).
    TEST_P(LargeModelShStorage, TrainingEncodeQ16PastTheWrap) {
        LFS_LARGE_MODEL_NEEDS(lm::store_bytes(ShFormat::Q16, lm::kRows) + lm::store_bytes(ShFormat::Float32, lm::kRows));
        if (sh_ == nullptr)
            GTEST_SKIP() << "Sh slot is empty";
        auto source = device_blocks(ShFormat::Float32);
        LFS_LARGE_MODEL_OR_SKIP(source);
        auto codes = device_store(ShFormat::Q16, lm::kRows);
        LFS_LARGE_MODEL_OR_SKIP(codes);
        sh_->encode_q16(source->values, codes->values, codes->bounds, lm::kRows, lm::kRest, 0, 0);
        lm::Store reference = lm::cpu_store(ShFormat::Q16, lm::kCompactRows);
        lm::convert(lm::compact_blocks(ShFormat::Float32), reference, {.count = lm::kCompactRows});
        lm::expect_blocks_match(*codes, reference, "ShOps::encode_q16");
    }

    // The training ShOps decode a whole Q16 store into fp32 float4 rows (sh_decode_q16).
    TEST_P(LargeModelShStorage, TrainingDecodeQ16PastTheWrap) {
        LFS_LARGE_MODEL_NEEDS(lm::store_bytes(ShFormat::Q16, lm::kRows) + lm::store_bytes(ShFormat::Float32, lm::kRows));
        if (sh_ == nullptr)
            GTEST_SKIP() << "Sh slot is empty";
        auto codes = device_blocks(ShFormat::Q16);
        LFS_LARGE_MODEL_OR_SKIP(codes);
        auto decoded = device_store(ShFormat::Float32, lm::kRows);
        LFS_LARGE_MODEL_OR_SKIP(decoded);
        sh_->decode_q16(codes->values, codes->bounds, decoded->values, lm::kRows, lm::kRest);
        lm::Store reference = lm::cpu_store(ShFormat::Float32, lm::kCompactRows);
        lm::convert(lm::compact_blocks(ShFormat::Q16), reference, {.count = lm::kCompactRows});
        lm::expect_blocks_match(*decoded, reference, "ShOps::decode_q16");
    }

    // The training ShOps gather far fp32 float4 rows into canonical rows (sh_gather_canonical).
    TEST_P(LargeModelShStorage, TrainingGatherCanonicalPastTheFloatWrap) {
        LFS_LARGE_MODEL_NEEDS(lm::store_bytes(ShFormat::Float32, lm::kRows));
        if (sh_ == nullptr)
            GTEST_SKIP() << "Sh slot is empty";
        auto source = device_blocks(ShFormat::Float32);
        LFS_LARGE_MODEL_OR_SKIP(source);
        const std::vector<size_t> rows = shuffled_rows({0, 2, 3}, 4207);
        Tensor canonical = Tensor::zeros({rows.size() * lm::kCells}, Device::GPU);
        sh_->gather_canonical(source->values, lm::index_tensor<int64_t>(rows, Device::GPU), canonical,
                              {.source_rows = lm::kRows, .count = rows.size(), .source_rest = lm::kRest,
                               .destination_rest = lm::kRest});
        lm::Store reference = lm::cpu_store(ShFormat::Canonical, rows.size());
        const Tensor indices = lm::index_tensor<int64_t>(compact_rows(rows), Device::CPU);
        lm::convert(lm::compact_blocks(ShFormat::Float32), reference, {.count = rows.size()}, &indices);
        lm::expect_store_matches({.format = ShFormat::Canonical, .rows = rows.size(), .values = canonical}, reference,
                                 "ShOps::gather_canonical");
    }

    // Morton's Q16 reorder re-encodes every destination block from the decoded source rows. The permutation swaps
    // H with the last 256 rows and reverses F; every other row stays.
    TEST_P(LargeModelShStorage, MortonPermuteQ16PastTheWrap) {
        // The store, the permutation, one cell column of the reorder's group buffer and the model's attributes.
        // Vulkan's permute_sh_q16 goes through fp32: it expands the store into fp32 float4 rows and gathers them
        // into a second fp32 store before it encodes again, two more stores of 18.3 GB here.
        const bool through_fp32 = GetParam() == GpuBackend::Vulkan;
        const size_t fp32_bytes = lm::store_bytes(ShFormat::Float32, lm::kRows);
        LFS_LARGE_MODEL_NEEDS(lm::store_bytes(ShFormat::Q16, lm::kRows) + lm::kRows * sizeof(int64_t) +
                              lm::kRows * sizeof(uint16_t) + attribute_bytes(lm::kRows) +
                              (through_fp32 ? 2 * fp32_bytes : 0));
        if (morton_ == nullptr || morton_->permute_sh_q16 == nullptr)
            GTEST_SKIP() << "Morton slot is empty";
        if (through_fp32 && !lm::zero_store(ShFormat::Float32, lm::kRows, Device::GPU))
            GTEST_SKIP() << std::format("this device can't clear a {:.1f} GB fp32 store, and Vulkan's permute_sh_q16 "
                                        "expands the Q16 store into two of them",
                                        static_cast<double>(fp32_bytes) / lm::kGB);
        auto store = device_blocks(ShFormat::Q16);
        LFS_LARGE_MODEL_OR_SKIP(store);
        auto splat = model(std::move(*store), lm::kRows);
        LFS_LARGE_MODEL_OR_SKIP(splat);
        const auto source_of = [](const size_t row) -> size_t {
            constexpr size_t tail = lm::kRows - 256;
            const auto& f = lm::kBlocks[2];
            if (row < 256)
                return tail + row;
            if (row >= tail)
                return row - tail;
            if (f.contains(row))
                return f.end - 1 - (row - f.begin);
            return row;
        };
        Tensor permutation;
        {
            Tensor host = Tensor::empty({lm::kRows}, Device::CPU, DataType::Int64);
            auto* const data = host.ptr<int64_t>();
            std::iota(data, data + lm::kRows, int64_t{0});
            for (const auto& block : lm::kBlocks)
                for (size_t r = block.begin; r < block.end; ++r)
                    data[r] = static_cast<int64_t>(source_of(r));
            permutation = host.gpu();
        }
        {
            const auto target = lfs::core::TensorExecutionTarget::current();
            const lfs::training::IdleArenaScratch arena(8u << 20, 0, target);
            lfs::training::LiveModelMutationGuard guard("MortonPermuteQ16PastTheWrap");
            morton_->permute_sh_q16(*splat, permutation, target, arena);
        }
        permutation = Tensor{};
        std::vector<size_t> compact(lm::kCompactRows);
        for (const auto& block : lm::kBlocks)
            for (size_t r = block.begin; r < block.end; ++r)
                compact[lm::compact_row(r)] = lm::compact_row(source_of(r));
        lm::Store reference = lm::cpu_store(ShFormat::Q16, lm::kCompactRows);
        const Tensor indices = lm::index_tensor<int64_t>(compact, Device::CPU);
        lm::convert(lm::compact_blocks(ShFormat::Q16), reference, {.count = lm::kCompactRows}, &indices);
        lm::expect_blocks_match(q16_of(*splat, lm::kRows), reference, "MortonOps::permute_sh_q16");
    }

    // A small gradient a function of the row and cell: (value - 1024) / 8192, exact in fp32.
    float gradient_value(const size_t row, const uint32_t cell) { return (lm::value(row, cell) - 1024.0f) / 8192.0f; }

    // Rows [first, first + rows) of two flat tensors with `width` elements a row must hold the same bytes.
    void expect_bytes_match(const Tensor& actual, const size_t actual_first, const Tensor& expected,
                            const size_t expected_first, const size_t rows, const size_t width,
                            const std::string& what) {
        const auto a = lm::host_copy<uint8_t>(actual.flatten().slice(0, actual_first * width, (actual_first + rows) * width));
        const auto e =
            lm::host_copy<uint8_t>(expected.flatten().slice(0, expected_first * width, (expected_first + rows) * width));
        ASSERT_EQ(a.size(), e.size()) << what;
        size_t wrong = 0, first = 0;
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i] != e[i] && wrong++ == 0)
                first = i;
        EXPECT_EQ(wrong, 0u) << what << ": " << wrong << " of " << a.size() << " bytes differ; first in row "
                             << actual_first + first / (a.size() / rows);
    }

    // One SH Adam step (adam_step_sh: the 8-bit joint moments, and the Q16 values re-encoded under new block bounds)
    // over a Q16 model whose gradients are nonzero only in the blocks. A row's step depends only on its 256-row block,
    // so the same kernel on the compact store of the blocks, where nothing wraps, is the reference, byte for byte.
    TEST_P(LargeModelShStorage, AdamStepQ16PastTheWrap) {
        namespace joint = lfs::training::joint_adam;
        const auto moment_bytes = [](const size_t rows) { return lm::value_elements(ShFormat::Float32, rows) * 2; };
        LFS_LARGE_MODEL_NEEDS(lm::store_bytes(ShFormat::Q16, lm::kRows) + lm::store_bytes(ShFormat::Float32, lm::kRows) +
                              moment_bytes(lm::kRows));
        const ops::AdamOps* const adam = lfs::training::training_ops(GetParam()).adam;
        if (adam == nullptr)
            GTEST_SKIP() << "Adam slot is empty";

        const lm::Store compact_values = lm::compact_blocks(ShFormat::Q16);
        lm::Store compact_gradient = lm::cpu_store(ShFormat::Float32, lm::kCompactRows);
        {
            const lm::Store canonical = lm::canonical_rows(
                lm::kCompactRows,
                [](const size_t compact) -> std::optional<size_t> {
                    for (const auto& block : lm::kBlocks)
                        if (compact >= block.compact && compact < block.compact + block.rows())
                            return block.holds_values ? std::optional(block.begin + (compact - block.compact))
                                                      : std::nullopt;
                    return std::nullopt;
                },
                gradient_value);
            lm::convert(canonical, compact_gradient, {.count = lm::kCompactRows});
        }

        const Tensor none;
        const ops::AdamMasks masks{none, none, none};
        const auto step = [&](lm::Store& values, const Tensor& gradient, const size_t rows, Tensor& packed,
                              Tensor& bounds) {
            packed = Tensor::zeros({moment_bytes(rows)}, Device::GPU, DataType::UInt8);
            bounds = Tensor::zeros({joint::n_bounds_for_prims(rows), 4}, Device::GPU);
            const ops::ShStepParams params{.primitives = static_cast<int>(rows),
                                           .layout_slots = static_cast<int>(lm::kSlots),
                                           .active_bases = 16,
                                           .value_bits = 16,
                                           .value_cells = static_cast<int>(lm::kCells),
                                           .step_size = 0.05f,
                                           .bc2_sqrt_rcp = static_cast<float>(1.0 / std::sqrt(0.001))};
            adam->step_sh(values.values, packed, bounds, values.bounds, gradient, masks, ops::AdamHyper{},
                          ops::AdamModifiers{}, params);
        };

        // The reference: the compact stores on the device.
        lm::Store small{.format = ShFormat::Q16, .rows = lm::kCompactRows, .values = compact_values.values.gpu(),
                        .bounds = compact_values.bounds.gpu()};
        Tensor small_packed, small_bounds;
        step(small, compact_gradient.values.gpu(), lm::kCompactRows, small_packed, small_bounds);

        auto values = device_blocks(ShFormat::Q16);
        LFS_LARGE_MODEL_OR_SKIP(values);
        Tensor packed, bounds;
        {
            auto gradient = device_store(ShFormat::Float32, lm::kRows);
            LFS_LARGE_MODEL_OR_SKIP(gradient);
            lm::place_blocks(*gradient, compact_gradient);
            step(*values, gradient->values, lm::kRows, packed, bounds);
        }
        lm::expect_blocks_match(*values, small, "adam_step_sh values", 0);
        for (const auto& block : lm::kBlocks) {
            const std::string where = std::string("adam_step_sh, block ") + block.name;
            // 8-bit joint moments: two bytes a cell, 48 cells a row; bounds: four floats a 256-row block.
            expect_bytes_match(packed, block.begin, small_packed, block.compact, block.rows(), lm::kFloatWidth * 2,
                               where + ", moments");
            expect_bytes_match(bounds, block.begin / lm::kQuantBlock, small_bounds, block.compact / lm::kQuantBlock,
                               block.rows() / lm::kQuantBlock, 4, where + ", moment bounds");
        }
    }

#undef LFS_LARGE_MODEL_NEEDS
#undef LFS_LARGE_MODEL_OR_SKIP

    INSTANTIATE_TEST_SUITE_P(Backends, LargeModelShStorage,
                             testing::Values(GpuBackend::CUDA, GpuBackend::Metal, GpuBackend::Vulkan),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });

} // namespace
