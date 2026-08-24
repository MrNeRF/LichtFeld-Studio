/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn.hpp"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

    constexpr float kF32Rtol = 1e-5f;
    constexpr float kF32Atol = 1e-5f;
    constexpr float kF16Rtol = 2e-2f;
    constexpr float kF16Atol = 2e-2f;

    std::vector<float> host_f32(const lfs::core::Tensor& t) {
        return t.to(lfs::core::DataType::Float32).to(lfs::core::Device::CPU).contiguous().to_vector();
    }

    lfs::core::Tensor upload(const std::vector<float>& data, const std::vector<std::size_t>& shape,
                             lfs::core::DataType dtype) {
        auto t = lfs::core::Tensor::from_vector(data, lfs::core::TensorShape(shape),
                                                lfs::core::Device::CUDA);
        if (dtype == lfs::core::DataType::Float16) {
            return t.to(lfs::core::DataType::Float16);
        }
        return t;
    }

    bool all_close(const std::vector<float>& a, const std::vector<float>& b, float rtol, float atol) {
        if (a.size() != b.size()) {
            ADD_FAILURE() << "size mismatch " << a.size() << " vs " << b.size();
            return false;
        }
        int mismatches = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const float diff = std::abs(a[i] - b[i]);
            const float tol = atol + rtol * std::abs(b[i]);
            if (diff > tol) {
                if (mismatches < 8) {
                    ADD_FAILURE() << "idx " << i << " got " << a[i] << " expected " << b[i]
                                  << " diff " << diff;
                }
                ++mismatches;
            }
        }
        if (mismatches > 0) {
            ADD_FAILURE() << mismatches << " / " << a.size() << " mismatches";
            return false;
        }
        return true;
    }

    std::vector<float> cpu_gemm(const std::vector<float>& a, const std::vector<float>& b, int m,
                                int n, int k, bool trans_b, const float* bias, int act) {
        std::vector<float> c(static_cast<std::size_t>(m) * n);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int t = 0; t < k; ++t) {
                    const float av = a[static_cast<std::size_t>(i) * k + t];
                    const float bv = trans_b ? b[static_cast<std::size_t>(j) * k + t]
                                             : b[static_cast<std::size_t>(t) * n + j];
                    sum += av * bv;
                }
                if (bias) {
                    sum += bias[j];
                }
                if (act == 1) {
                    sum = std::max(sum, 0.0f);
                } else if (act == 2) {
                    const float inner = 0.7978845608028654f * (sum + 0.044715f * sum * sum * sum);
                    sum = 0.5f * sum * (1.0f + std::tanh(inner));
                } else if (act == 3) {
                    sum = 0.5f * sum * (1.0f + std::erf(sum * 0.7071067811865476f));
                } else if (act == 4) {
                    sum = sum / (1.0f + std::exp(-sum));
                }
                c[static_cast<std::size_t>(i) * n + j] = sum;
            }
        }
        return c;
    }

    std::vector<float> cpu_layer_norm(const std::vector<float>& x, const std::vector<float>& w,
                                      const std::vector<float>& b, int rows, int cols, float eps) {
        std::vector<float> y(x.size());
        for (int r = 0; r < rows; ++r) {
            float mean = 0.0f;
            for (int c = 0; c < cols; ++c) {
                mean += x[r * cols + c];
            }
            mean /= static_cast<float>(cols);
            float var = 0.0f;
            for (int c = 0; c < cols; ++c) {
                const float d = x[r * cols + c] - mean;
                var += d * d;
            }
            var /= static_cast<float>(cols);
            const float inv = 1.0f / std::sqrt(var + eps);
            for (int c = 0; c < cols; ++c) {
                y[r * cols + c] = (x[r * cols + c] - mean) * inv * w[c] + b[c];
            }
        }
        return y;
    }

    std::vector<float> cpu_softmax(const std::vector<float>& x, int rows, int cols) {
        std::vector<float> y(x.size());
        for (int r = 0; r < rows; ++r) {
            float m = x[r * cols];
            for (int c = 1; c < cols; ++c) {
                m = std::max(m, x[r * cols + c]);
            }
            float s = 0.0f;
            for (int c = 0; c < cols; ++c) {
                y[r * cols + c] = std::exp(x[r * cols + c] - m);
                s += y[r * cols + c];
            }
            for (int c = 0; c < cols; ++c) {
                y[r * cols + c] /= s;
            }
        }
        return y;
    }

    std::vector<float> cpu_attention(const std::vector<float>& q, const std::vector<float>& k,
                                     const std::vector<float>& v, int b, int h, int n, int d) {
        const float scale = 1.0f / std::sqrt(static_cast<float>(d));
        std::vector<float> o(q.size(), 0.0f);
        std::vector<float> scores(static_cast<std::size_t>(n) * n);
        for (int bh = 0; bh < b * h; ++bh) {
            const float* qh = q.data() + static_cast<std::size_t>(bh) * n * d;
            const float* kh = k.data() + static_cast<std::size_t>(bh) * n * d;
            const float* vh = v.data() + static_cast<std::size_t>(bh) * n * d;
            float* oh = o.data() + static_cast<std::size_t>(bh) * n * d;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    float dot = 0.0f;
                    for (int t = 0; t < d; ++t) {
                        dot += qh[i * d + t] * kh[j * d + t];
                    }
                    scores[i * n + j] = dot * scale;
                }
            }
            auto p = cpu_softmax(scores, n, n);
            for (int i = 0; i < n; ++i) {
                for (int t = 0; t < d; ++t) {
                    float sum = 0.0f;
                    for (int j = 0; j < n; ++j) {
                        sum += p[i * n + j] * vh[j * d + t];
                    }
                    oh[i * d + t] = sum;
                }
            }
        }
        return o;
    }

    std::vector<float> cpu_conv2d(const std::vector<float>& in, const std::vector<float>& w,
                                  const std::vector<float>* bias, int n, int cin, int h, int wi,
                                  int cout, int kh, int kw, int sh, int sw, int ph, int pw, int dh,
                                  int dw, int groups, int pad_mode = 0) {
        const int cin_g = cin / groups;
        const int cout_g = cout / groups;
        const int oh = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
        const int ow = (wi + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
        std::vector<float> out(static_cast<std::size_t>(n) * cout * oh * ow, 0.0f);
        for (int ni = 0; ni < n; ++ni) {
            for (int g = 0; g < groups; ++g) {
                for (int oc = 0; oc < cout_g; ++oc) {
                    const int oc_g = g * cout_g + oc;
                    for (int y = 0; y < oh; ++y) {
                        for (int x = 0; x < ow; ++x) {
                            float sum = bias ? (*bias)[oc_g] : 0.0f;
                            for (int ic = 0; ic < cin_g; ++ic) {
                                const int ic_g = g * cin_g + ic;
                                for (int ky = 0; ky < kh; ++ky) {
                                    const int iy0 = y * sh - ph + ky * dh;
                                    for (int kx = 0; kx < kw; ++kx) {
                                        const int ix0 = x * sw - pw + kx * dw;
                                        int iy = iy0;
                                        int ix = ix0;
                                        if (pad_mode == 1) {
                                            iy = std::clamp(iy, 0, h - 1);
                                            ix = std::clamp(ix, 0, wi - 1);
                                        } else if (iy < 0 || iy >= h || ix < 0 || ix >= wi) {
                                            continue;
                                        }
                                        const float iv =
                                            in[(((ni * cin + ic_g) * h + iy) * wi + ix)];
                                        const float wv = w[((((oc_g * cin_g) + ic) * kh + ky) * kw + kx)];
                                        sum += iv * wv;
                                    }
                                }
                            }
                            out[(((ni * cout + oc_g) * oh + y) * ow + x)] = sum;
                        }
                    }
                }
            }
        }
        return out;
    }

    std::string project_root() {
        return PROJECT_ROOT_PATH;
    }

    std::string vcpkg_python() {
        return project_root() + "/build/vcpkg_installed/x64-linux/tools/python3/python3";
    }

} // namespace

class NnOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        int devices = 0;
        ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
        ASSERT_GT(devices, 0);
    }
};

TEST_F(NnOpsTest, GemmOddShapesFp32AndFp16) {
    const int shapes[][3] = {{17, 19, 13}, {1, 8, 3}, {64, 64, 32}, {137, 65, 17}};
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (const auto& mnk : shapes) {
        const int m = mnk[0], n = mnk[1], k = mnk[2];
        std::vector<float> a(m * k), b(n * k), bias(n);
        for (auto& v : a) {
            v = dist(rng);
        }
        for (auto& v : b) {
            v = dist(rng);
        }
        for (auto& v : bias) {
            v = dist(rng);
        }
        const auto ref = cpu_gemm(a, b, m, n, k, true, bias.data(), 0);
        for (const auto dtype : {lfs::core::DataType::Float32, lfs::core::DataType::Float16}) {
            auto A = upload(a, {static_cast<std::size_t>(m), static_cast<std::size_t>(k)}, dtype);
            auto B = upload(b, {static_cast<std::size_t>(n), static_cast<std::size_t>(k)}, dtype);
            auto Bs = upload(bias, {static_cast<std::size_t>(n)}, dtype);
            auto C = lfs::core::nn::gemm(A, B, false, true, &Bs, lfs::core::nn::Activation::None);
            const float rtol = dtype == lfs::core::DataType::Float16 ? kF16Rtol : kF32Rtol;
            const float atol = dtype == lfs::core::DataType::Float16 ? kF16Atol : kF32Atol;
            EXPECT_TRUE(all_close(host_f32(C), ref, rtol, atol)) << "m=" << m << " n=" << n << " k=" << k;
        }
    }
}

TEST_F(NnOpsTest, GemmNnAndReluEpilogue) {
    const int m = 21, n = 18, k = 11;
    std::vector<float> a(m * k, 0.25f), b(k * n);
    for (int i = 0; i < k * n; ++i) {
        b[i] = (i % 7) * 0.05f - 0.1f;
    }
    const auto ref = cpu_gemm(a, b, m, n, k, false, nullptr, 1);
    auto A = upload(a, {static_cast<std::size_t>(m), static_cast<std::size_t>(k)},
                    lfs::core::DataType::Float32);
    auto B = upload(b, {static_cast<std::size_t>(k), static_cast<std::size_t>(n)},
                    lfs::core::DataType::Float32);
    auto C = lfs::core::nn::gemm(A, B, false, false, nullptr, lfs::core::nn::Activation::Relu);
    EXPECT_TRUE(all_close(host_f32(C), ref, kF32Rtol, kF32Atol));
}

TEST_F(NnOpsTest, LinearMatchesGemmNT) {
    std::vector<float> x(8 * 5), w(3 * 5), bias(3, 0.1f);
    std::iota(x.begin(), x.end(), 0.0f);
    std::iota(w.begin(), w.end(), -2.0f);
    auto X = upload(x, {8, 5}, lfs::core::DataType::Float32);
    auto W = upload(w, {3, 5}, lfs::core::DataType::Float32);
    auto B = upload(bias, {3}, lfs::core::DataType::Float32);
    auto y = lfs::core::nn::linear(X, W, &B, lfs::core::nn::Activation::None);
    const auto ref = cpu_gemm(x, w, 8, 3, 5, true, bias.data(), 0);
    EXPECT_TRUE(all_close(host_f32(y), ref, kF32Rtol, kF32Atol));
}

TEST_F(NnOpsTest, LayerNormAndRmsNorm) {
    const int rows = 6, cols = 17;
    std::vector<float> x(rows * cols), w(cols, 1.2f), b(cols, -0.3f);
    for (int i = 0; i < rows * cols; ++i) {
        x[i] = std::sin(0.1f * i);
    }
    auto X = upload(x, {static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)},
                    lfs::core::DataType::Float32);
    auto W = upload(w, {static_cast<std::size_t>(cols)}, lfs::core::DataType::Float32);
    auto B = upload(b, {static_cast<std::size_t>(cols)}, lfs::core::DataType::Float32);
    auto y = lfs::core::nn::layer_norm(X, W, B, 1e-5f);
    EXPECT_TRUE(all_close(host_f32(y), cpu_layer_norm(x, w, b, rows, cols, 1e-5f), kF32Rtol, kF32Atol));

    std::vector<float> rms_ref(x.size());
    for (int r = 0; r < rows; ++r) {
        float ss = 0.0f;
        for (int c = 0; c < cols; ++c) {
            ss += x[r * cols + c] * x[r * cols + c];
        }
        const float inv = 1.0f / std::sqrt(ss / cols + 1e-6f);
        for (int c = 0; c < cols; ++c) {
            rms_ref[r * cols + c] = x[r * cols + c] * inv * w[c];
        }
    }
    auto yr = lfs::core::nn::rms_norm(X, W, 1e-6f);
    EXPECT_TRUE(all_close(host_f32(yr), rms_ref, kF32Rtol, kF32Atol));
}

TEST_F(NnOpsTest, SoftmaxLastDim) {
    std::vector<float> x = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 5.0f};
    auto X = upload(x, {2, 3}, lfs::core::DataType::Float32);
    auto y = lfs::core::nn::softmax(X);
    EXPECT_TRUE(all_close(host_f32(y), cpu_softmax(x, 2, 3), kF32Rtol, kF32Atol));
}

TEST_F(NnOpsTest, AttentionWmmaTileParity) {
    const int b = 1, h = 1, n = 17, d = 64;
    std::vector<float> q(b * h * n * d), k(q.size()), v(q.size());
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-0.6f, 0.6f);
    for (auto& val : q) {
        val = dist(rng);
    }
    for (auto& val : k) {
        val = dist(rng);
    }
    for (auto& val : v) {
        val = dist(rng);
    }
    const auto ref = cpu_attention(q, k, v, b, h, n, d);
    auto shape = std::vector<std::size_t>{1, 1, static_cast<std::size_t>(n),
                                          static_cast<std::size_t>(d)};
    auto Q = upload(q, shape, lfs::core::DataType::Float16);
    auto K = upload(k, shape, lfs::core::DataType::Float16);
    auto V = upload(v, shape, lfs::core::DataType::Float16);
    auto O = lfs::core::nn::attention(Q, K, V);
    EXPECT_TRUE(all_close(host_f32(O), ref, kF16Rtol, kF16Atol));
}

TEST_F(NnOpsTest, AttentionVsExplicitSoftmax) {
    const int b = 1, h = 2, n = 9, d = 8;
    std::vector<float> q(b * h * n * d), k(q.size()), v(q.size());
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-0.8f, 0.8f);
    for (auto& val : q) {
        val = dist(rng);
    }
    for (auto& val : k) {
        val = dist(rng);
    }
    for (auto& val : v) {
        val = dist(rng);
    }
    const auto ref = cpu_attention(q, k, v, b, h, n, d);
    auto shape = std::vector<std::size_t>{static_cast<std::size_t>(b), static_cast<std::size_t>(h),
                                          static_cast<std::size_t>(n), static_cast<std::size_t>(d)};
    for (const auto dtype : {lfs::core::DataType::Float32, lfs::core::DataType::Float16}) {
        auto Q = upload(q, shape, dtype);
        auto K = upload(k, shape, dtype);
        auto V = upload(v, shape, dtype);
        auto O = lfs::core::nn::attention(Q, K, V);
        const float rtol = dtype == lfs::core::DataType::Float16 ? kF16Rtol : 2e-4f;
        const float atol = dtype == lfs::core::DataType::Float16 ? kF16Atol : 2e-4f;
        EXPECT_TRUE(all_close(host_f32(O), ref, rtol, atol));
    }
}

TEST_F(NnOpsTest, WindowedAttentionMatchesPerWindow) {
    const int b = 1, h = 1, n = 10, d = 4, win = 4;
    std::vector<float> q(b * h * n * d);
    for (int i = 0; i < static_cast<int>(q.size()); ++i) {
        q[i] = 0.05f * (i - 7);
    }
    auto shape = std::vector<std::size_t>{1, 1, static_cast<std::size_t>(n), static_cast<std::size_t>(d)};
    auto Q = upload(q, shape, lfs::core::DataType::Float32);
    auto Qw = lfs::core::nn::window_partition(Q, win);
    auto Ow = lfs::core::nn::attention(Qw, Qw, Qw);
    auto O = lfs::core::nn::window_unpartition(Ow, win, n);
    const int n_win = (n + win - 1) / win;
    std::vector<float> ref(q.size(), 0.0f);
    for (int w = 0; w < n_win; ++w) {
        const int start = w * win;
        const int len = std::min(win, n - start);
        std::vector<float> qw(static_cast<std::size_t>(win) * d, 0.0f);
        for (int i = 0; i < len; ++i) {
            for (int t = 0; t < d; ++t) {
                qw[i * d + t] = q[(start + i) * d + t];
            }
        }
        auto local = cpu_attention(qw, qw, qw, 1, 1, win, d);
        for (int i = 0; i < len; ++i) {
            for (int t = 0; t < d; ++t) {
                ref[(start + i) * d + t] = local[i * d + t];
            }
        }
    }
    EXPECT_TRUE(all_close(host_f32(O), ref, 2e-4f, 2e-4f));
}

TEST_F(NnOpsTest, Conv2dMatchesSevenLoop) {
    const int n = 1, cin = 4, h = 9, w = 11, cout = 6, kh = 3, kw = 3;
    std::vector<float> in(n * cin * h * w), wt(cout * (cin / 2) * kh * kw), bias(cout);
    for (int i = 0; i < static_cast<int>(in.size()); ++i) {
        in[i] = std::sin(0.07f * i);
    }
    for (int i = 0; i < static_cast<int>(wt.size()); ++i) {
        wt[i] = std::cos(0.03f * i) * 0.2f;
    }
    for (int i = 0; i < cout; ++i) {
        bias[i] = 0.01f * i;
    }
    lfs::core::nn::Conv2dParams p;
    p.stride_h = 2;
    p.stride_w = 1;
    p.pad_h = 1;
    p.pad_w = 2;
    p.dilation_h = 1;
    p.dilation_w = 2;
    p.groups = 2;
    const auto ref = cpu_conv2d(in, wt, &bias, n, cin, h, w, cout, kh, kw, p.stride_h, p.stride_w,
                                p.pad_h, p.pad_w, p.dilation_h, p.dilation_w, p.groups);
    auto In = upload(in, {1, 4, 9, 11}, lfs::core::DataType::Float32);
    auto W = upload(wt, {6, 2, 3, 3}, lfs::core::DataType::Float32);
    auto B = upload(bias, {6}, lfs::core::DataType::Float32);
    auto Out = lfs::core::nn::conv2d(In, W, &B, p);
    EXPECT_TRUE(all_close(host_f32(Out), ref, 2e-5f, 2e-5f));

    lfs::core::nn::Conv2dParams p1;
    p1.stride_h = 1;
    p1.stride_w = 1;
    p1.pad_h = 1;
    p1.pad_w = 1;
    std::vector<float> w1(cout * cin * kh * kw);
    for (int i = 0; i < static_cast<int>(w1.size()); ++i) {
        w1[i] = std::cos(0.03f * i) * 0.2f;
    }
    const auto ref1 = cpu_conv2d(in, w1, &bias, n, cin, h, w, cout, kh, kw, 1, 1, 1, 1, 1, 1, 1);
    auto W1 = upload(w1, {6, 4, 3, 3}, lfs::core::DataType::Float32);
    auto Out1 = lfs::core::nn::conv2d(In, W1, &B, p1);
    EXPECT_TRUE(all_close(host_f32(Out1), ref1, 2e-5f, 2e-5f));
}

TEST_F(NnOpsTest, Conv2dDirect1x1And3x3) {
    const int n = 1, cin = 4, h = 7, w = 9, cout = 5;
    std::vector<float> in(n * cin * h * w), w1(cout * cin), b(cout);
    std::vector<float> w3(cout * cin * 9);
    for (int i = 0; i < static_cast<int>(in.size()); ++i) {
        in[i] = std::sin(0.11f * i);
    }
    for (int i = 0; i < static_cast<int>(w1.size()); ++i) {
        w1[i] = std::cos(0.07f * i) * 0.15f;
    }
    for (int i = 0; i < static_cast<int>(w3.size()); ++i) {
        w3[i] = std::cos(0.05f * i) * 0.12f;
    }
    for (int i = 0; i < cout; ++i) {
        b[i] = 0.02f * i;
    }
    const auto ref1 = cpu_conv2d(in, w1, &b, n, cin, h, w, cout, 1, 1, 1, 1, 0, 0, 1, 1, 1);
    const auto ref3 = cpu_conv2d(in, w3, &b, n, cin, h, w, cout, 3, 3, 1, 1, 1, 1, 1, 1, 1);
    for (const auto dtype : {lfs::core::DataType::Float32, lfs::core::DataType::Float16}) {
        auto In = upload(in, {1, 4, 7, 9}, dtype);
        auto W1 = upload(w1, {5, 4, 1, 1}, dtype);
        auto W3 = upload(w3, {5, 4, 3, 3}, dtype);
        auto B = upload(b, {5}, dtype);
        lfs::core::nn::Conv2dParams p1;
        auto Out1 = lfs::core::nn::conv2d(In, W1, &B, p1);
        const float rtol = dtype == lfs::core::DataType::Float16 ? kF16Rtol : 2e-5f;
        const float atol = dtype == lfs::core::DataType::Float16 ? kF16Atol : 2e-5f;
        EXPECT_TRUE(all_close(host_f32(Out1), ref1, rtol, atol)) << "1x1 " << static_cast<int>(dtype);
        lfs::core::nn::Conv2dParams p3;
        p3.pad_h = 1;
        p3.pad_w = 1;
        auto Out3 = lfs::core::nn::conv2d(In, W3, &B, p3);
        EXPECT_TRUE(all_close(host_f32(Out3), ref3, rtol, atol)) << "3x3 " << static_cast<int>(dtype);
        p3.pad_mode = lfs::core::nn::ConvPadMode::Replicate;
        auto OutR = lfs::core::nn::conv2d(In, W3, &B, p3);
        EXPECT_EQ(OutR.shape()[2], 7u);
        EXPECT_EQ(OutR.shape()[3], 9u);
    }
}

TEST_F(NnOpsTest, Conv2dFp16Large3x3MatchesRef) {
    const int n = 1, cin = 32, h = 16, w = 20, cout = 64;
    std::vector<float> in(n * cin * h * w), wt(cout * cin * 9), bias(cout);
    for (int i = 0; i < static_cast<int>(in.size()); ++i) {
        in[i] = std::sin(0.013f * i);
    }
    for (int i = 0; i < static_cast<int>(wt.size()); ++i) {
        wt[i] = std::cos(0.017f * i) * 0.08f;
    }
    for (int i = 0; i < cout; ++i) {
        bias[i] = 0.01f * static_cast<float>(i);
    }
    const auto refz = cpu_conv2d(in, wt, &bias, n, cin, h, w, cout, 3, 3, 1, 1, 1, 1, 1, 1, 1);
    auto In = upload(in, {1, 32, 16, 20}, lfs::core::DataType::Float16);
    auto W = upload(wt, {64, 32, 3, 3}, lfs::core::DataType::Float16);
    auto B = upload(bias, {64}, lfs::core::DataType::Float16);
    lfs::core::nn::Conv2dParams p;
    p.pad_h = 1;
    p.pad_w = 1;
    auto OutZ = lfs::core::nn::conv2d(In, W, &B, p);
    EXPECT_TRUE(all_close(host_f32(OutZ), refz, kF16Rtol, kF16Atol));
    p.pad_mode = lfs::core::nn::ConvPadMode::Replicate;
    const auto refr = cpu_conv2d(in, wt, &bias, n, cin, h, w, cout, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
    auto OutR = lfs::core::nn::conv2d(In, W, &B, p);
    EXPECT_TRUE(all_close(host_f32(OutR), refr, kF16Rtol, kF16Atol)) << "3x3 replicate";
}

TEST_F(NnOpsTest, SplitQkvMergeHeadsAndResidual) {
    const int b = 1, s = 5, h = 2, d = 4;
    std::vector<float> qkv(b * s * 3 * h * d);
    for (int i = 0; i < static_cast<int>(qkv.size()); ++i) {
        qkv[i] = 0.01f * (i - 20);
    }
    auto QKV = upload(qkv, {1, 5, 3, 2, 4}, lfs::core::DataType::Float32);
    auto split = lfs::core::nn::split_qkv(QKV, h);
    auto merged = lfs::core::nn::merge_heads(split[0]);
    EXPECT_EQ(merged.shape()[0], 1u);
    EXPECT_EQ(merged.shape()[1], 5u);
    EXPECT_EQ(merged.shape()[2], 8u);
    std::vector<float> x(s * 8, 0.5f), hid(s * 8), gamma(8, 1.25f);
    for (int i = 0; i < static_cast<int>(hid.size()); ++i) {
        hid[i] = 0.1f * i;
    }
    auto X = upload(x, {5, 8}, lfs::core::DataType::Float32);
    auto H = upload(hid, {5, 8}, lfs::core::DataType::Float32);
    auto G = upload(gamma, {8}, lfs::core::DataType::Float32);
    auto Y = lfs::core::nn::residual_scale(X, H, G);
    std::vector<float> ref(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        ref[i] = x[i] + hid[i] * gamma[i % 8];
    }
    EXPECT_TRUE(all_close(host_f32(Y), ref, kF32Rtol, kF32Atol));
}

TEST_F(NnOpsTest, UvGridMatchesCpuFormula) {
    const int height = 3, width = 5;
    const float aspect = 1.5f;
    auto t = lfs::core::nn::uv_grid(height, width, aspect, lfs::core::DataType::Float32,
                                    lfs::core::Device::CUDA, nullptr);
    const auto got = host_f32(t);
    const float span_x = aspect / std::sqrt(1.0f + aspect * aspect);
    const float span_y = 1.0f / std::sqrt(1.0f + aspect * aspect);
    const float u0 = -span_x * static_cast<float>(width - 1) / static_cast<float>(width);
    const float u1 = span_x * static_cast<float>(width - 1) / static_cast<float>(width);
    const float v0 = -span_y * static_cast<float>(height - 1) / static_cast<float>(height);
    const float v1 = span_y * static_cast<float>(height - 1) / static_cast<float>(height);
    std::vector<float> ref(static_cast<std::size_t>(2) * height * width);
    for (int y = 0; y < height; ++y) {
        const float vv = v0 + (v1 - v0) * static_cast<float>(y) / static_cast<float>(height - 1);
        for (int x = 0; x < width; ++x) {
            const float uu = u0 + (u1 - u0) * static_cast<float>(x) / static_cast<float>(width - 1);
            const std::size_t pix = static_cast<std::size_t>(y) * width + x;
            ref[pix] = uu;
            ref[static_cast<std::size_t>(height) * width + pix] = vv;
        }
    }
    EXPECT_TRUE(all_close(got, ref, kF32Rtol, kF32Atol));
}

TEST_F(NnOpsTest, ConvTranspose2dFp16Scatter) {
    std::vector<float> in(1 * 2 * 3 * 3, 1.0f);
    std::vector<float> wt(2 * 3 * 2 * 2, 0.25f);
    auto In = upload(in, {1, 2, 3, 3}, lfs::core::DataType::Float16);
    auto W = upload(wt, {2, 3, 2, 2}, lfs::core::DataType::Float16);
    lfs::core::nn::Conv2dParams p;
    p.stride_h = 2;
    p.stride_w = 2;
    auto Out = lfs::core::nn::conv_transpose2d(In, W, nullptr, p);
    EXPECT_EQ(Out.shape()[1], 3u);
    const auto got = host_f32(Out);
    const float expected = 0.5f;
    int mismatches = 0;
    for (float v : got) {
        if (std::abs(v - expected) > 2e-3f) {
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0);
}

TEST_F(NnOpsTest, ConvTranspose2dBasic) {
    std::vector<float> in(1 * 2 * 3 * 3, 1.0f);
    std::vector<float> wt(2 * 3 * 2 * 2, 0.25f);
    auto In = upload(in, {1, 2, 3, 3}, lfs::core::DataType::Float32);
    auto W = upload(wt, {2, 3, 2, 2}, lfs::core::DataType::Float32);
    lfs::core::nn::Conv2dParams p;
    p.stride_h = 2;
    p.stride_w = 2;
    auto Out = lfs::core::nn::conv_transpose2d(In, W, nullptr, p);
    EXPECT_EQ(Out.shape()[1], 3u);
    EXPECT_GT(Out.shape()[2], 3u);
    EXPECT_GT(Out.shape()[3], 3u);

    // ONNX ConvTranspose: each input channel contributes W[cin, cout, kh, kw]
    // at out[y*s+kh, x*s+kw]. With ones input and 0.25 weights, overlapped
    // 2x2 kernels on stride 2 do not overlap, so every spatial output is 0.5.
    const auto got = host_f32(Out);
    const float expected = 0.5f; // 2 input channels * 0.25
    int mismatches = 0;
    for (float v : got) {
        if (std::abs(v - expected) > 1e-5f) {
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0);
}

TEST_F(NnOpsTest, ResizeMatchesNumpyFixture) {
    const std::string dir = project_root() + "/tests/data/nn/";
    struct Case {
        const char* file;
        lfs::core::nn::ResizeMode mode;
        lfs::core::nn::CoordTransform coord;
    };
    const Case cases[] = {
        {"resize_nearest_asymmetric.json", lfs::core::nn::ResizeMode::Nearest,
         lfs::core::nn::CoordTransform::Asymmetric},
        {"resize_bilinear_half_pixel.json", lfs::core::nn::ResizeMode::Bilinear,
         lfs::core::nn::CoordTransform::HalfPixel},
        {"resize_align_corners.json", lfs::core::nn::ResizeMode::Bilinear,
         lfs::core::nn::CoordTransform::AlignCorners},
    };
    for (const auto& c : cases) {
        std::ifstream in(dir + c.file);
        if (!in) {
            GTEST_SKIP() << "missing fixture " << c.file;
        }
        const auto body = nlohmann::json::parse(in);
        const auto input = body["input"].get<std::vector<float>>();
        const auto output = body["output"].get<std::vector<float>>();
        const auto in_shape = body["input_shape"].get<std::vector<std::size_t>>();
        const auto out_shape = body["output_shape"].get<std::vector<std::size_t>>();
        auto X = upload(input, in_shape, lfs::core::DataType::Float32);
        auto Y = lfs::core::nn::resize2d(X, static_cast<int>(out_shape[2]),
                                         static_cast<int>(out_shape[3]), c.mode, c.coord);
        EXPECT_TRUE(all_close(host_f32(Y), output, kF32Rtol, kF32Atol)) << c.file;
    }
}

TEST_F(NnOpsTest, GeluSiluReluAndCast) {
    std::vector<float> x = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f, 3.0f};
    auto X = upload(x, {6}, lfs::core::DataType::Float32);
    auto erf = lfs::core::nn::gelu(X, lfs::core::nn::GELUApprox::Erf);
    auto tanh = lfs::core::nn::gelu(X, lfs::core::nn::GELUApprox::Tanh);
    auto sl = lfs::core::nn::silu(X);
    auto rl = lfs::core::nn::relu(X);
    std::vector<float> erf_ref(x.size()), tanh_ref(x.size()), sl_ref(x.size()), rl_ref(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        erf_ref[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] * 0.7071067811865476f));
        const float inner = 0.7978845608028654f * (x[i] + 0.044715f * x[i] * x[i] * x[i]);
        tanh_ref[i] = 0.5f * x[i] * (1.0f + std::tanh(inner));
        sl_ref[i] = x[i] / (1.0f + std::exp(-x[i]));
        rl_ref[i] = std::max(x[i], 0.0f);
    }
    EXPECT_TRUE(all_close(host_f32(erf), erf_ref, kF32Rtol, kF32Atol));
    EXPECT_TRUE(all_close(host_f32(tanh), tanh_ref, kF32Rtol, kF32Atol));
    EXPECT_TRUE(all_close(host_f32(sl), sl_ref, kF32Rtol, kF32Atol));
    EXPECT_TRUE(all_close(host_f32(rl), rl_ref, kF32Rtol, kF32Atol));
    auto h = lfs::core::nn::cast(X, lfs::core::DataType::Float16);
    auto back = lfs::core::nn::cast(h, lfs::core::DataType::Float32);
    EXPECT_TRUE(all_close(host_f32(back), x, kF16Rtol, kF16Atol));
}

TEST_F(NnOpsTest, AvgAndMaxPool) {
    std::vector<float> x(1 * 1 * 4 * 4);
    std::iota(x.begin(), x.end(), 1.0f);
    auto X = upload(x, {1, 1, 4, 4}, lfs::core::DataType::Float32);
    auto mx = lfs::core::nn::max_pool2d(X, 2, 2, 2, 2, 0, 0);
    auto av = lfs::core::nn::avg_pool2d(X, 2, 2, 2, 2, 0, 0, true);
    EXPECT_TRUE(all_close(host_f32(mx), {6.0f, 8.0f, 14.0f, 16.0f}, kF32Rtol, kF32Atol));
    EXPECT_TRUE(all_close(host_f32(av), {3.5f, 5.5f, 11.5f, 13.5f}, kF32Rtol, kF32Atol));
}

TEST_F(NnOpsTest, WeightFilePythonRoundTrip) {
    const std::string py = vcpkg_python();
    const std::string script = project_root() + "/tools/nn_export/export_onnx_weights.py";
    const std::string out = project_root() + "/tests/data/nn/roundtrip.lfw";
    const std::string cmd = py + " " + script + " --write-test-lfw " + out;
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        GTEST_SKIP() << "python writer unavailable: " << cmd;
    }
    auto file = lfs::core::nn::WeightFile::open(out);
    ASSERT_TRUE(file.has_value()) << std::string(file.error().detail());
    auto w = file->load("w", lfs::core::Device::CUDA);
    ASSERT_TRUE(w.has_value()) << std::string(w.error().detail());
    EXPECT_EQ(w->shape()[0], 3u);
    EXPECT_EQ(w->dtype(), lfs::core::DataType::Float32);
    auto h = file->load("h", lfs::core::Device::CPU, lfs::core::DataType::Float32);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(h->numel(), 8u);
    auto b = file->load("b", lfs::core::Device::CUDA);
    ASSERT_TRUE(b.has_value());
    EXPECT_TRUE(all_close(host_f32(*b), {0.5f, -1.25f, 3.0f}, kF32Rtol, kF32Atol));
}

TEST_F(NnOpsTest, MogeFixtureIsOptionalAndSmall) {
    const std::string path = project_root() + "/tests/data/nn/moge2_ref_fixture.json";
    std::ifstream in(path);
    if (!in) {
        GTEST_SKIP() << "MoGe fixture not generated";
    }
    std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_LT(body.size(), 2u * 1024u * 1024u);
    EXPECT_NE(body.find("\"normal\""), std::string::npos);
}
