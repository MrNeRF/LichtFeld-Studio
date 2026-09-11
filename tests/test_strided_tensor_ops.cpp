/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * Training-reachable strided-operation correctness.
 *
 * Ops under test (materialize-firewall / stride-aware):
 *   - masked_select / masked_fill_ on strided views
 *   - index_put_ on view destinations (offset + non-contiguous)
 *   - densification_info.index_select(dim=1) on [2, N]
 *   - Tensor::multinomial with non-contiguous (strided column) weights
 *
 * Reference for every test: explicit host CPU loops over logical indices
 * (NOT LibTorch — the contract is logical layout semantics).
 */

#include "core/tensor.hpp"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace lfs::core;

namespace {

    // Logical row-major read of a (possibly strided) Float32 tensor via to_vector
    // materialization — used only as a convenience; primary oracles are hand loops
    // over known base storage patterns below.

    std::vector<float> host_f32(const Tensor& t) {
        return t.cpu().contiguous().to_vector();
    }

    Tensor bool_mask_from(const std::vector<bool>& bits, TensorShape shape, Device device) {
        return Tensor::from_vector(bits, shape, device);
    }

    Tensor i32_tensor_from(const std::vector<int>& vals, Device device) {
        return Tensor::from_vector(vals, {vals.size()}, device);
    }

    // ----- CPU reference: masked_select on row-major logical layout -----
    std::vector<float> ref_masked_select(const std::vector<float>& logical_in,
                                         const std::vector<bool>& logical_mask) {
        EXPECT_EQ(logical_in.size(), logical_mask.size());
        std::vector<float> out;
        out.reserve(logical_in.size());
        for (size_t i = 0; i < logical_in.size(); ++i) {
            if (logical_mask[i]) {
                out.push_back(logical_in[i]);
            }
        }
        return out;
    }

    // ----- CPU reference: index_select dim=1 on [2, N] densification_info -----
    // info row-major: row0[0..N), row1[0..N)
    std::vector<float> ref_index_select_dim1_2xN(const std::vector<float>& info_row_major,
                                                 size_t N,
                                                 const std::vector<int>& indices) {
        EXPECT_EQ(info_row_major.size(), 2 * N);
        const size_t K = indices.size();
        std::vector<float> out(2 * K);
        for (size_t o = 0; o < 2; ++o) {
            for (size_t k = 0; k < K; ++k) {
                const int sel = indices[k];
                EXPECT_GE(sel, 0);
                EXPECT_LT(static_cast<size_t>(sel), N);
                out[o * K + k] = info_row_major[o * N + static_cast<size_t>(sel)];
            }
        }
        return out;
    }

    // ----- CPU reference: flat index_put_ on a 1D logical view -----
    void ref_index_put_1d_view(std::vector<float>& base,
                               size_t view_start,
                               size_t view_len,
                               const std::vector<int>& idx,
                               const std::vector<float>& vals) {
        EXPECT_EQ(idx.size(), vals.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            const int local = idx[i];
            EXPECT_GE(local, 0);
            EXPECT_LT(static_cast<size_t>(local), view_len);
            base[view_start + static_cast<size_t>(local)] = vals[i];
        }
    }

    void expect_vec_eq(const std::vector<float>& got,
                       const std::vector<float>& ref,
                       const std::string& label) {
        ASSERT_EQ(got.size(), ref.size()) << label << ": size mismatch";
        for (size_t i = 0; i < ref.size(); ++i) {
            EXPECT_FLOAT_EQ(got[i], ref[i]) << label << " mismatch at " << i
                                            << " got=" << got[i] << " ref=" << ref[i];
        }
    }

} // namespace

class StridedTensorOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
    }
};

// =============================================================================
// =============================================================================
//
// base [2,3] = [[1,2,3],[4,5,6]]
// view = base.t() → [3,2] logical [[1,4],[2,5],[3,6]]  (non-contiguous)
// mask true at logical positions that select {1, 2, 6}
// Without materialize firewall, linear scan of physical storage yields {1,2,3}.

TEST_F(StridedTensorOpsTest, MaskedSelect_TransposedView_CPU) {
    auto base = Tensor::from_vector({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3}, Device::CPU);
    auto view = base.transpose(0, 1); // [3,2]
    ASSERT_FALSE(view.is_contiguous());
    ASSERT_EQ(view.shape()[0], 3u);
    ASSERT_EQ(view.shape()[1], 2u);

    // Logical values of view in row-major order: 1,4, 2,5, 3,6
    const std::vector<float> logical_in = {1.f, 4.f, 2.f, 5.f, 3.f, 6.f};
    // Mask: pick positions 0,2,5 → values 1, 2, 6
    const std::vector<bool> logical_mask = {true, false, true, false, false, true};
    auto mask = bool_mask_from(logical_mask, {3, 2}, Device::CPU);

    auto selected = view.masked_select(mask);
    const auto got = host_f32(selected);
    const auto ref = ref_masked_select(logical_in, logical_mask);
    expect_vec_eq(got, ref, "MaskedSelect_TransposedView_CPU");
}

TEST_F(StridedTensorOpsTest, MaskedSelect_TransposedView_CUDA) {
    auto base = Tensor::from_vector({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3}, Device::CUDA);
    auto view = base.transpose(0, 1);
    ASSERT_FALSE(view.is_contiguous());

    const std::vector<float> logical_in = {1.f, 4.f, 2.f, 5.f, 3.f, 6.f};
    const std::vector<bool> logical_mask = {true, false, true, false, false, true};
    auto mask = bool_mask_from(logical_mask, {3, 2}, Device::CUDA);

    auto selected = view.masked_select(mask);
    expect_vec_eq(host_f32(selected), ref_masked_select(logical_in, logical_mask),
                  "MaskedSelect_TransposedView_CUDA");
}

// =============================================================================
//    corruption of physical storage outside the view mapping
// =============================================================================
//
// base [2,3] = [[1,2,3],[4,5,6]]
// view = transpose [3,2]
// fill logical pos 0 (value 1 → base[0,0]) and pos 3 (value 5 → base[1,1]) with -1
// Expected base: [[-1,2,3],[4,-1,6]]
// Wrong linear fill would clobber physical slots 0 and 3 → base becomes
// [[-1,2,3],[-1,5,6]] (different).

TEST_F(StridedTensorOpsTest, MaskedFill_TransposedView_NoSiblingCorruption_CUDA) {
    auto base = Tensor::from_vector({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3}, Device::CUDA);
    auto view = base.transpose(0, 1); // [3,2]
    ASSERT_FALSE(view.is_contiguous());

    // Logical mask over [3,2]: positions 0 and 3
    const std::vector<bool> logical_mask = {true, false, false, true, false, false};
    auto mask = bool_mask_from(logical_mask, {3, 2}, Device::CUDA);

    view.masked_fill_(mask, -1.0f);

    // Reference: apply fill in logical view space, map back to base layout
    // logical view: [1,4, 2,5, 3,6] → fill pos 0,3 → [-1,4, 2,-1, 3,6]
    // base row-major for transpose view writeback:
    //   view[r,c] lives at base[c,r]
    //   view[0,0]=-1 → base[0,0]; view[1,1]=-1 → base[1,1]
    const std::vector<float> ref_base = {-1.f, 2.f, 3.f, 4.f, -1.f, 6.f};
    expect_vec_eq(host_f32(base), ref_base, "MaskedFill_TransposedView_base");

    // View logical content
    const std::vector<float> ref_view = {-1.f, 4.f, 2.f, -1.f, 3.f, 6.f};
    expect_vec_eq(host_f32(view), ref_view, "MaskedFill_TransposedView_view");
}

TEST_F(StridedTensorOpsTest, MaskedFill_TransposedView_CPU) {
    auto base = Tensor::from_vector({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3}, Device::CPU);
    auto view = base.transpose(0, 1);
    ASSERT_FALSE(view.is_contiguous());

    const std::vector<bool> logical_mask = {true, false, false, true, false, false};
    auto mask = bool_mask_from(logical_mask, {3, 2}, Device::CPU);

    view.masked_fill_(mask, -1.0f);
    expect_vec_eq(host_f32(base), {-1.f, 2.f, 3.f, 4.f, -1.f, 6.f},
                  "MaskedFill_TransposedView_CPU");
}

// =============================================================================
// =============================================================================
//
// base = [0,1,2,3,4,5,6,7]
// v = base.slice(0, 2, 5) → [2,3,4], storage_offset=2, is_contiguous=true
// v.index_put_([1], [9]) must set base[3]=9, NOT base[1].

TEST_F(StridedTensorOpsTest, IndexPut_OffsetContiguousView_CUDA) {
    auto base = Tensor::from_vector(
        {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}, {8}, Device::CUDA);
    auto v = base.slice(0, 2, 5); // [2,3,4]
    ASSERT_EQ(v.numel(), 3u);
    ASSERT_TRUE(v.is_contiguous());
    ASSERT_EQ(v.storage_offset(), 2u);

    auto idx = i32_tensor_from({1}, Device::CUDA);
    auto vals = Tensor::from_vector(std::vector<float>{9.f}, {1}, Device::CUDA);
    v.index_put_(idx, vals);

    std::vector<float> ref = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    ref_index_put_1d_view(ref, /*view_start=*/2, /*view_len=*/3, {1}, {9.f});
    expect_vec_eq(host_f32(base), ref, "IndexPut_OffsetContiguousView_CUDA");
    // Explicit: base[1] must remain 1; base[3] must be 9
    EXPECT_FLOAT_EQ(host_f32(base)[1], 1.f);
    EXPECT_FLOAT_EQ(host_f32(base)[3], 9.f);
}

TEST_F(StridedTensorOpsTest, IndexPut_OffsetContiguousView_CPU) {
    auto base = Tensor::from_vector(
        {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}, {8}, Device::CPU);
    auto v = base.slice(0, 2, 5);
    auto idx = i32_tensor_from({1}, Device::CPU);
    auto vals = Tensor::from_vector(std::vector<float>{9.f}, {1}, Device::CPU);
    v.index_put_(idx, vals);

    std::vector<float> ref = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    ref_index_put_1d_view(ref, 2, 3, {1}, {9.f});
    expect_vec_eq(host_f32(base), ref, "IndexPut_OffsetContiguousView_CPU");
}

// index_put_ on a non-contiguous 2D slice destination (row-assignment path)
// base [4,4] sequential; slice [0:3, 0:3]; put row 1 of the slice with [100,200,300]
// Must write base row1 cols 0..2, leave base[1,3] untouched.

TEST_F(StridedTensorOpsTest, IndexPut_NonContiguousSlice_RowAssign_CUDA) {
    std::vector<float> data(16);
    for (int i = 0; i < 16; ++i)
        data[i] = static_cast<float>(i + 1);
    auto base = Tensor::from_vector(data, {4, 4}, Device::CUDA);
    auto slice = base.slice(0, 0, 3).slice(1, 0, 3); // 3x3, non-contiguous
    ASSERT_FALSE(slice.is_contiguous());

    auto row_idx = i32_tensor_from({1}, Device::CUDA);
    auto vals = Tensor::from_vector(std::vector<float>{100.f, 200.f, 300.f}, {1, 3}, Device::CUDA);
    slice.index_put_(row_idx, vals);

    auto got = host_f32(base);
    // base row1 (physical indices 4,5,6,7): 5,6,7,8 → 100,200,300,8
    EXPECT_FLOAT_EQ(got[4], 100.f);
    EXPECT_FLOAT_EQ(got[5], 200.f);
    EXPECT_FLOAT_EQ(got[6], 300.f);
    EXPECT_FLOAT_EQ(got[7], 8.f) << "sibling column outside slice must be untouched";
    // other rows intact
    EXPECT_FLOAT_EQ(got[0], 1.f);
    EXPECT_FLOAT_EQ(got[15], 16.f);
}

// =============================================================================
// 4) densification_info.index_select(dim=1) — MRNF compact_splats semantics
// =============================================================================
//
// densification_info is [2, N]:
//   row0[i] = float(i)           (e.g. vis counts)
//   row1[i] = 100.f + float(i)   (e.g. grad norms)
// valid_indices = [1, 3, 5]
// result must be [2, 3] with columns {1,3,5}.

TEST_F(StridedTensorOpsTest, DensificationInfo_IndexSelectDim1_CUDA) {
    constexpr size_t N = 8;
    std::vector<float> info_data(2 * N);
    for (size_t i = 0; i < N; ++i) {
        info_data[i] = static_cast<float>(i);             // row 0
        info_data[N + i] = 100.f + static_cast<float>(i); // row 1
    }
    auto info = Tensor::from_vector(info_data, {2, N}, Device::CUDA);
    ASSERT_TRUE(info.is_contiguous());

    const std::vector<int> valid = {1, 3, 5};
    auto idx = i32_tensor_from(valid, Device::CUDA);

    auto gathered = info.index_select(1, idx);
    ASSERT_EQ(gathered.ndim(), 2u);
    ASSERT_EQ(gathered.shape()[0], 2u);
    ASSERT_EQ(gathered.shape()[1], valid.size());

    const auto ref = ref_index_select_dim1_2xN(info_data, N, valid);
    expect_vec_eq(host_f32(gathered), ref, "DensificationInfo_IndexSelectDim1");

    // Also verify index_select_into (compact_splats path)
    auto dest = Tensor::zeros({2, valid.size()}, Device::CUDA);
    info.index_select_into(dest, 1, idx, BoundaryMode::Assert);
    expect_vec_eq(host_f32(dest), ref, "DensificationInfo_IndexSelectIntoDim1");
}

TEST_F(StridedTensorOpsTest, DensificationInfo_IndexSelectDim1_CPU) {
    constexpr size_t N = 6;
    std::vector<float> info_data(2 * N);
    for (size_t i = 0; i < N; ++i) {
        info_data[i] = static_cast<float>(i);
        info_data[N + i] = 50.f + static_cast<float>(i);
    }
    auto info = Tensor::from_vector(info_data, {2, N}, Device::CPU);
    const std::vector<int> valid = {0, 2, 4};
    auto idx = i32_tensor_from(valid, Device::CPU);
    auto gathered = info.index_select(1, idx);
    expect_vec_eq(host_f32(gathered), ref_index_select_dim1_2xN(info_data, N, valid),
                  "DensificationInfo_IndexSelectDim1_CPU");
}

// Non-contiguous [2,N] source: take a column-range slice of a wider buffer then
// index_select dim 1 — exercises materialize-then-gather firewall.
TEST_F(StridedTensorOpsTest, DensificationInfo_IndexSelectDim1_StridedSource_CUDA) {
    // Wide [2, 10]; logical densify window is columns [2, 8) → shape [2,6] strided?
    // slice dim1 keeps row stride = 10 → non-contiguous.
    std::vector<float> wide(2 * 10);
    for (size_t i = 0; i < 10; ++i) {
        wide[i] = static_cast<float>(i);
        wide[10 + i] = 100.f + static_cast<float>(i);
    }
    auto wide_t = Tensor::from_vector(wide, {2, 10}, Device::CUDA);
    auto info = wide_t.slice(1, 2, 8); // [2, 6], columns 2..7 of wide
    ASSERT_EQ(info.shape()[0], 2u);
    ASSERT_EQ(info.shape()[1], 6u);
    ASSERT_FALSE(info.is_contiguous());

    // Logical info_data for the window (cols 2..7):
    std::vector<float> logical(2 * 6);
    for (size_t i = 0; i < 6; ++i) {
        logical[i] = static_cast<float>(i + 2);
        logical[6 + i] = 100.f + static_cast<float>(i + 2);
    }
    const std::vector<int> valid = {1, 3, 5}; // within window → wide cols 3,5,7
    auto idx = i32_tensor_from(valid, Device::CUDA);
    auto gathered = info.index_select(1, idx);
    expect_vec_eq(host_f32(gathered), ref_index_select_dim1_2xN(logical, 6, valid),
                  "DensificationInfo_IndexSelectDim1_StridedSource");
}

// =============================================================================
// =============================================================================
//
// weights = column 1 of [[1,0,0],[0,100,0],[0,0,1]] → logical [0, 100, 0]
// With replacement, 200 samples must almost always hit index 1.
// Physical linear scan of the column-strided storage would read a different
// probability mass → wrong mode.

TEST_F(StridedTensorOpsTest, Multinomial_StridedColumnWeights_CUDA) {
    // own [3,2] = [[0, 0],[100,0],[0,0]]; column 0 is logical weights [0,100,0]
    // with physical stride 2 (non-contiguous rank-1 after squeeze).
    auto own = Tensor::from_vector(
        {0.f, 0.f,
         100.f, 0.f,
         0.f, 0.f},
        {3, 2}, Device::CUDA);
    auto strided_w = own.slice(1, 0, 1).squeeze(); // [3]
    ASSERT_EQ(strided_w.ndim(), 1u);
    ASSERT_EQ(strided_w.numel(), 3u);
    {
        const auto w = host_f32(strided_w);
        ASSERT_EQ(w.size(), 3u);
        EXPECT_FLOAT_EQ(w[0], 0.f);
        EXPECT_FLOAT_EQ(w[1], 100.f);
        EXPECT_FLOAT_EQ(w[2], 0.f);
    }
    // Prefer non-contiguous input; if squeeze densified it, still check the
    // logical distribution (contiguous firewall must preserve values either way).
    if (!strided_w.is_contiguous()) {
        GTEST_LOG_(INFO) << "multinomial input is non-contiguous";
    }

    constexpr int N = 200;
    auto samples = Tensor::multinomial(strided_w, N, /*replacement=*/true);
    ASSERT_EQ(samples.numel(), static_cast<size_t>(N));
    auto s = samples.cpu().to_vector_int64();
    int count1 = 0;
    for (int64_t v : s) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 3);
        if (v == 1)
            ++count1;
    }
    // With weights [0,100,0], every sample must be index 1
    EXPECT_EQ(count1, N) << "multinomial on strided weights must sample logical "
                            "probabilities (mode=1); got count1="
                         << count1;
}

TEST_F(StridedTensorOpsTest, Multinomial_StridedColumnWeights_CPU) {
    auto own = Tensor::from_vector(
        {0.f, 0.f,
         100.f, 0.f,
         0.f, 0.f},
        {3, 2}, Device::CPU);
    auto strided_w = own.slice(1, 0, 1).squeeze();
    const auto w = host_f32(strided_w);
    ASSERT_EQ(w.size(), 3u);
    EXPECT_FLOAT_EQ(w[1], 100.f);

    constexpr int N = 100;
    auto samples = Tensor::multinomial(strided_w, N, true);
    auto s = samples.to_vector_int64();
    for (int64_t v : s) {
        EXPECT_EQ(v, 1) << "CPU multinomial must honor logical strided weights";
    }
}

// Sanity: contiguous densification_info dim0 row extract still works (control)
TEST_F(StridedTensorOpsTest, DensificationInfo_RowExtractDim0_Control) {
    constexpr size_t N = 5;
    std::vector<float> info_data(2 * N);
    for (size_t i = 0; i < N; ++i) {
        info_data[i] = static_cast<float>(i);
        info_data[N + i] = 10.f * static_cast<float>(i);
    }
    auto info = Tensor::from_vector(info_data, {2, N}, Device::CUDA);
    auto idx = i32_tensor_from({1}, Device::CUDA);
    auto row1 = info.index_select(0, idx); // [1, N]
    ASSERT_EQ(row1.shape()[0], 1u);
    ASSERT_EQ(row1.shape()[1], N);
    expect_vec_eq(host_f32(row1),
                  std::vector<float>(info_data.begin() + static_cast<long>(N), info_data.end()),
                  "row extract dim0 control");
}
