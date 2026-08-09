/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/tensor_zero_stride.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace lfs::core;

namespace {

    void cuda_ok() {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    }

    Tensor cpu_f32(std::vector<float> v, std::vector<size_t> shape) {
        auto t = Tensor::from_vector(v, TensorShape(shape), Device::CPU);
        EXPECT_TRUE(t.is_valid());
        return t;
    }

    Tensor cuda_f32(std::vector<float> v, std::vector<size_t> shape) {
        return cpu_f32(std::move(v), std::move(shape)).cuda();
    }

    void expect_allclose(const Tensor& a, const Tensor& b, float atol = 1e-5f,
                         const std::string& ctx = "") {
        auto av = a.cpu().contiguous().to_vector();
        auto bv = b.cpu().contiguous().to_vector();
        ASSERT_EQ(av.size(), bv.size()) << ctx;
        for (size_t i = 0; i < av.size(); ++i) {
            EXPECT_NEAR(av[i], bv[i], atol) << ctx << " idx=" << i;
        }
    }

} // namespace

// ---------------------------------------------------------------------------
// View metadata / alloc
// ---------------------------------------------------------------------------

TEST(ZeroStrideExpand, BroadcastToSameShapeDoesNotClone) {
    auto base = cuda_f32({1, 2, 3, 4}, {2, 2});
    const auto snap = alloc_counter::snapshot();
    auto same = base.broadcast_to(TensorShape({2, 2}));
    cuda_ok();
    EXPECT_EQ(alloc_counter::delta_since(snap), 0u)
        << "same-shape broadcast_to must not allocate";
    EXPECT_EQ(same.data_ptr(), base.data_ptr());
    EXPECT_EQ(same.shape(), base.shape());
}

TEST(ZeroStrideExpand, ExpandIsViewWithZeroStrideNoAlloc) {
    // [1,4] -> [256,4]: expanded size 4 KiB floats if materialized would miss tiny slabs.
    // Use large expand so a materializing path would need a real driver alloc on cold pool.
    constexpr size_t rows = 4096;
    auto base = Tensor::full({1, 64}, 3.25f, Device::CUDA);
    cuda_ok();

    // Allocate and free a different size so the expand target is cold.
    {
        auto poison = Tensor::empty({rows, 64}, Device::CUDA);
        (void)poison;
    }
    cuda_ok();

    const auto snap = alloc_counter::snapshot();
    auto exp = base.expand(TensorShape({rows, 64}));
    cuda_ok();

    EXPECT_EQ(alloc_counter::delta_since(snap), 0u)
        << "expand must be metadata-only (zero-stride view)";
    EXPECT_TRUE(exp.is_view()) << "expand must produce a view";
    EXPECT_FALSE(exp.is_contiguous()) << "expanded dim must break contiguity";
    EXPECT_TRUE(exp.has_zero_stride()) << "API has_zero_stride() required";
    EXPECT_EQ(exp.shape(), TensorShape({rows, 64}));
    ASSERT_EQ(exp.strides().size(), 2u);
    EXPECT_EQ(exp.strides()[0], 0u) << "broadcast dim stride must be 0";
    EXPECT_EQ(exp.strides()[1], 1u);
    // storage_ptr() observes the shared allocation without materializing the
    // zero-stride view; data_ptr() and ptr() are flat-buffer escapes.
    EXPECT_EQ(exp.storage_ptr(), base.storage_ptr());
}

TEST(ZeroStrideExpand, BroadcastToIsViewWithZeroStride) {
    auto base = cuda_f32({10.f, 20.f, 30.f}, {1, 3});
    auto b = base.broadcast_to(TensorShape({4, 3}));
    cuda_ok();
    EXPECT_TRUE(b.has_zero_stride());
    EXPECT_TRUE(b.is_view());
    EXPECT_FALSE(b.is_contiguous());
    EXPECT_EQ(b.strides()[0], 0u);
    EXPECT_EQ(b.numel(), 12u);
    auto vals = b.cpu().contiguous().to_vector();
    ASSERT_EQ(vals.size(), 12u);
    for (size_t r = 0; r < 4; ++r) {
        EXPECT_FLOAT_EQ(vals[r * 3 + 0], 10.f);
        EXPECT_FLOAT_EQ(vals[r * 3 + 1], 20.f);
        EXPECT_FLOAT_EQ(vals[r * 3 + 2], 30.f);
    }
}

TEST(ZeroStrideExpand, ContiguousMaterializesExpandedStorage) {
    auto base = cuda_f32({1.f, 2.f}, {1, 2});
    auto exp = base.expand(TensorShape({3, 2}));
    auto dense = exp.contiguous();
    cuda_ok();
    EXPECT_TRUE(dense.is_contiguous());
    EXPECT_FALSE(dense.has_zero_stride());
    EXPECT_EQ(dense.numel(), 6u);
    expect_allclose(dense, cpu_f32({1, 2, 1, 2, 1, 2}, {3, 2}).cuda());
}

// ---------------------------------------------------------------------------
// Allowlisted consumers — view path == materialized path
// ---------------------------------------------------------------------------

TEST(ZeroStrideExpand, AllowlistedBinaryAddMatchesMaterialized) {
    auto a = cuda_f32({1.f, 2.f, 3.f}, {1, 3});
    auto b = cuda_f32({10.f, 20.f, 30.f, 40.f, 50.f, 60.f, 70.f, 80.f, 90.f}, {3, 3});
    auto a_view = a.expand(TensorShape({3, 3}));
    ASSERT_TRUE(a_view.has_zero_stride());

    auto view_sum = a_view.add(b);
    auto mat_sum = a_view.contiguous().add(b);
    cuda_ok();
    expect_allclose(view_sum, mat_sum, 1e-5f, "binary add view vs mat");
    EXPECT_TRUE(zero_stride::is_allowlisted(zero_stride::ConsumerKind::ElementwiseFirewall));
}

TEST(ZeroStrideExpand, AllowlistedMulScalarMatchesMaterialized) {
    auto a = cuda_f32({2.f, 4.f}, {2, 1});
    auto v = a.expand(TensorShape({2, 5}));
    auto view_r = v.mul(3.0f);
    auto mat_r = v.contiguous().mul(3.0f);
    cuda_ok();
    expect_allclose(view_r, mat_r, 1e-5f, "scalar mul");
}

TEST(ZeroStrideExpand, Stride0TimesStridedMix) {
    // left: expand view; right: transpose (strided non-zero)
    auto left = cuda_f32({1.f, 2.f, 3.f}, {1, 3}).expand(TensorShape({3, 3}));
    auto right_base = cuda_f32({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3});
    auto right = right_base.transpose(0, 1); // strided
    ASSERT_TRUE(left.has_zero_stride());
    ASSERT_FALSE(right.is_contiguous());
    ASSERT_FALSE(right.has_zero_stride());

    auto view_r = left.add(right);
    auto mat_r = left.contiguous().add(right.contiguous());
    cuda_ok();
    expect_allclose(view_r, mat_r, 1e-5f, "stride0 x strided mix");
}

TEST(ZeroStrideExpand, AllowlistedBroadcastBinaryShapeIndexed) {
    // No explicit expand: binary broadcast kernel (shape-indexed) is allowlisted.
    auto a = cuda_f32({1.f, 2.f, 3.f}, {1, 3});
    auto b = cuda_f32({10.f, 20.f, 30.f,
                       40.f, 50.f, 60.f,
                       70.f, 80.f, 90.f},
                      {3, 3});
    auto r = a.add(b); // internal broadcast
    cuda_ok();
    auto expected = cpu_f32({11, 22, 33, 41, 52, 63, 71, 82, 93}, {3, 3}).cuda();
    expect_allclose(r, expected, 1e-5f, "broadcast binary");
    EXPECT_TRUE(zero_stride::is_allowlisted(zero_stride::ConsumerKind::BroadcastBinary));
}

// ---------------------------------------------------------------------------
// In-place rejection
// ---------------------------------------------------------------------------

TEST(ZeroStrideExpand, InPlaceAddOnExpandViewThrows) {
    auto base = cuda_f32({1.f, 2.f, 3.f}, {1, 3});
    auto v = base.expand(TensorShape({2, 3}));
    ASSERT_TRUE(v.has_zero_stride());
    EXPECT_THROW(v.add_(1.0f), std::runtime_error);
}

TEST(ZeroStrideExpand, InPlaceBinaryOnExpandViewThrows) {
    auto base = cuda_f32({1.f, 2.f}, {1, 2});
    auto v = base.expand(TensorShape({2, 2}));
    auto other = cuda_f32({1, 1, 1, 1}, {2, 2});
    EXPECT_THROW(v.add_(other), std::runtime_error);
}

TEST(ZeroStrideExpand, InPlaceZeroOnExpandViewThrows) {
    auto base = cuda_f32({1.f, 2.f}, {1, 2});
    auto v = base.expand(TensorShape({3, 2}));
    EXPECT_THROW(v.zero_(), std::runtime_error);
}

TEST(ZeroStrideExpand, NonAllowlistedCatMaterializesWithoutCorruption) {
    auto row = cuda_f32({1.f, 2.f, 3.f}, {1, 3});
    auto a = row.expand(TensorShape({2, 3})); // [[1,2,3],[1,2,3]]
    auto b = cuda_f32({4.f, 5.f, 6.f, 7.f, 8.f, 9.f}, {2, 3});
    ASSERT_TRUE(a.has_zero_stride());
    ASSERT_FALSE(zero_stride::is_allowlisted(zero_stride::ConsumerKind::Cat));

    auto cat = Tensor::cat({a, b}, /*dim=*/0);
    cuda_ok();
    // Expected: 4x3 = [1,2,3, 1,2,3, 4,5,6, 7,8,9]
    auto expected = cpu_f32({1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {4, 3}).cuda();
    expect_allclose(cat, expected, 1e-5f, "cat canary");
}

TEST(ZeroStrideExpand, NonAllowlistedMaskedSelectMaterializes) {
    auto base = cuda_f32({1.f, 2.f, 3.f}, {1, 3});
    auto v = base.expand(TensorShape({2, 3}));
    auto mask = Tensor::from_vector(
                    std::vector<bool>{true, false, true, false, true, false},
                    TensorShape({2, 3}), Device::CPU)
                    .cuda();
    ASSERT_TRUE(v.has_zero_stride());
    ASSERT_FALSE(zero_stride::is_allowlisted(zero_stride::ConsumerKind::MaskedSelect));

    auto selected = v.masked_select(mask);
    cuda_ok();
    // Logical values at True positions: (0,0)=1, (0,2)=3, (1,1)=2 → [1,3,2]
    auto vals = selected.cpu().to_vector();
    ASSERT_EQ(vals.size(), 3u);
    EXPECT_FLOAT_EQ(vals[0], 1.f);
    EXPECT_FLOAT_EQ(vals[1], 3.f);
    EXPECT_FLOAT_EQ(vals[2], 2.f);
}

TEST(ZeroStrideExpand, NonAllowlistedCloneMaterializesDense) {
    auto base = cuda_f32({5.f, 6.f}, {1, 2});
    auto v = base.expand(TensorShape({2, 2}));
    auto c = v.clone();
    cuda_ok();
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_FALSE(c.has_zero_stride());
    expect_allclose(c, cpu_f32({5, 6, 5, 6}, {2, 2}).cuda());
}

// ---------------------------------------------------------------------------
// Allowlist registry itself
// ---------------------------------------------------------------------------

TEST(ZeroStrideExpand, AllowlistRegistryCoversSafeConsumers) {
    using K = zero_stride::ConsumerKind;
    EXPECT_TRUE(zero_stride::is_allowlisted(K::Contiguous));
    EXPECT_TRUE(zero_stride::is_allowlisted(K::Clone));
    EXPECT_TRUE(zero_stride::is_allowlisted(K::ElementwiseFirewall));
    EXPECT_TRUE(zero_stride::is_allowlisted(K::BroadcastBinary));
    EXPECT_FALSE(zero_stride::is_allowlisted(K::Cat));
    EXPECT_FALSE(zero_stride::is_allowlisted(K::MaskedSelect));
    EXPECT_FALSE(zero_stride::is_allowlisted(K::InPlaceMutate));
}
