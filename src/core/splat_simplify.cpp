/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_simplify.hpp"

#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "nanoflann.hpp"

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lfs::core {

    namespace {

        constexpr float kTwoPiPow1p5 = 0x1.f7fccep+3f;
        // Match `np.log(2.0 * np.pi).astype(np.float32)` exactly.
        constexpr float kLog2Pi = 0x1.d67f1cp+0f;
        constexpr float kEpsCov = 1e-8f;
        constexpr float kMinScale = 1e-12f;
        constexpr float kMinQuatNorm = 1e-12f;
        constexpr float kMinProb = 1e-6f;
        constexpr float kMinDet = 1e-30f;
        constexpr float kMinEval = 1e-18f;
        constexpr int kJacobiIterations = 32;

        struct PointCloudAdaptor {
            const float* points = nullptr;
            size_t num_points = 0;

            [[nodiscard]] inline size_t kdtree_get_point_count() const { return num_points; }
            [[nodiscard]] inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
                return static_cast<double>(points[idx * 3 + dim]);
            }
            template <class BBOX>
            bool kdtree_get_bbox(BBOX&) const { return false; }
        };

        using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor>,
            PointCloudAdaptor,
            3>;

        struct SplatSimplifyWorkset {
            Tensor means;
            Tensor scaling;
            Tensor rotation;
            Tensor opacity;
            Tensor appearance;
            int max_sh_degree = 0;
            int active_sh_degree = 0;
            int shn_coeffs = 0;
            float scene_scale = 1.0f;

            [[nodiscard]] int size() const { return means.is_valid() ? static_cast<int>(means.size(0)) : 0; }
        };

        struct NativeRows {
            int count = 0;
            int app_dim = 0;
            std::vector<float> means;
            std::vector<float> scales;
            std::vector<float> rotation;
            std::vector<float> opacity;
            std::vector<float> appearance;
        };

        struct CacheEntry {
            std::array<float, 9> R{};
            std::array<float, 9> Rt{};
            std::array<float, 9> sigma{};
            std::array<float, 3> variance{};
            std::array<float, 3> invdiag{};
            float logdet = 0.0f;
            float mass = 0.0f;
        };

        struct Eigen3x3 {
            std::array<float, 3> values{};
            std::array<float, 9> vectors{};
        };

        struct SimplifyScratch {
            std::vector<CacheEntry> cache;
            std::vector<float> costs;
            std::vector<size_t> order;
            std::vector<uint8_t> used_rows;
            std::vector<int> keep_idx;
            std::vector<std::pair<int, int>> pairs;
        };

        [[nodiscard]] Tensor flatten_sh_like_ply(const Tensor& sh) {
            if (!sh.is_valid())
                return Tensor{};
            if (sh.ndim() == 3) {
                const auto transposed = sh.transpose(1, 2).contiguous();
                return transposed.reshape({static_cast<int>(sh.size(0)), static_cast<int>(sh.size(1) * sh.size(2))})
                    .contiguous();
            }
            return sh.contiguous();
        }

        [[nodiscard]] Tensor unflatten_sh_like_ply(const Tensor& flat, const int coeff_count) {
            if (!flat.is_valid() || coeff_count <= 0)
                return Tensor{};
            const auto reshaped = flat.reshape({static_cast<int>(flat.size(0)), 3, coeff_count}).contiguous();
            return reshaped.transpose(1, 2).contiguous();
        }

        [[nodiscard]] SplatSimplifyWorkset make_workset_from_input(const SplatData& input, const Device device) {
            const bool has_deleted = input.has_deleted_mask() && input.deleted().count_nonzero() > 0;
            const Tensor keep_mask = has_deleted ? input.deleted().logical_not() : Tensor{};

            const auto select_or_clone = [&](const Tensor& tensor) -> Tensor {
                if (!tensor.is_valid())
                    return Tensor{};
                if (has_deleted)
                    return tensor.index_select(0, keep_mask).contiguous();
                return tensor;
            };

            const auto means = select_or_clone(input.means_raw()).to(device).contiguous();
            const auto sh0 = select_or_clone(input.sh0_raw()).to(device).contiguous();
            const auto shN = input.shN_raw().is_valid() ? select_or_clone(input.shN_raw()).to(device).contiguous() : Tensor{};
            const auto scaling = select_or_clone(input.scaling_raw()).to(device).contiguous();
            const auto rotation = select_or_clone(input.rotation_raw()).to(device).contiguous();
            const auto opacity = select_or_clone(input.opacity_raw()).to(device).contiguous();

            const int n = static_cast<int>(means.size(0));
            auto sh0_flat = flatten_sh_like_ply(sh0).reshape({n, 3}).contiguous();
            Tensor appearance = sh0_flat;
            int shn_coeffs = 0;
            if (shN.is_valid()) {
                shn_coeffs = static_cast<int>(shN.size(1));
                auto shn_flat = flatten_sh_like_ply(shN).reshape({n, shn_coeffs * 3}).contiguous();
                appearance = Tensor::cat({sh0_flat, shn_flat}, 1).contiguous();
            }

            SplatSimplifyWorkset workset;
            workset.means = means;
            workset.scaling = scaling;
            workset.rotation = rotation;
            workset.opacity = opacity;
            workset.appearance = appearance;
            workset.max_sh_degree = input.get_max_sh_degree();
            workset.active_sh_degree = input.get_active_sh_degree();
            workset.shn_coeffs = shn_coeffs;
            workset.scene_scale = input.get_scene_scale();
            return workset;
        }

        [[nodiscard]] std::unique_ptr<SplatData> make_splat_from_workset(const SplatSimplifyWorkset& workset, const Device device) {
            const auto sh0 = unflatten_sh_like_ply(workset.appearance.slice(1, 0, 3).contiguous(), 1).to(device).contiguous();
            Tensor shN;
            if (workset.shn_coeffs > 0) {
                shN = unflatten_sh_like_ply(
                          workset.appearance.slice(1, 3, 3 + workset.shn_coeffs * 3).contiguous(),
                          workset.shn_coeffs)
                          .to(device)
                          .contiguous();
            }

            auto result = std::make_unique<SplatData>(
                workset.max_sh_degree,
                workset.means.to(device).contiguous(),
                sh0,
                shN,
                workset.scaling.to(device).contiguous(),
                workset.rotation.to(device).contiguous(),
                workset.opacity.to(device).contiguous(),
                workset.scene_scale);
            result->set_active_sh_degree(workset.active_sh_degree);
            result->set_max_sh_degree(workset.max_sh_degree);
            return result;
        }

        [[nodiscard]] bool report_progress(const SplatSimplifyProgressCallback& progress,
                                           const float value,
                                           const std::string& stage) {
            if (!progress)
                return true;
            return progress(std::clamp(value, 0.0f, 1.0f), stage);
        }

        [[nodiscard]] float sigmoid(const float x) {
            if (x >= 0.0f) {
                const float z = std::exp(-x);
                return 1.0f / (1.0f + z);
            }
            const float z = std::exp(x);
            return z / (1.0f + z);
        }

        [[nodiscard]] float clamp_prob(const float p) {
            return std::clamp(p, kMinProb, 1.0f - kMinProb);
        }

        [[nodiscard]] float logit_from_alpha(const float alpha) {
            const float q = clamp_prob(alpha);
            return std::log(q / (1.0f - q));
        }

        [[nodiscard]] float clamp_scale_raw(const float raw) {
            return std::clamp(raw, -30.0f, 30.0f);
        }

        [[nodiscard]] float activated_scale(const float raw) {
            return std::max(std::exp(clamp_scale_raw(raw)), kMinScale);
        }

        [[nodiscard]] float strict_mul(const float a, const float b) {
            volatile float out = a * b;
            return out;
        }

        [[nodiscard]] float strict_add(const float a, const float b) {
            volatile float out = a + b;
            return out;
        }

        [[nodiscard]] float strict_sub(const float a, const float b) {
            volatile float out = a - b;
            return out;
        }

        [[nodiscard]] float strict_sum3_prod(const float a0,
                                             const float b0,
                                             const float a1,
                                             const float b1,
                                             const float a2,
                                             const float b2) {
            const float t0 = strict_mul(a0, b0);
            const float t1 = strict_mul(a1, b1);
            const float t2 = strict_mul(a2, b2);
            return strict_add(strict_add(t0, t1), t2);
        }

        [[nodiscard]] float strict_prod3(const float a, const float b, const float c) {
            return strict_mul(strict_mul(a, b), c);
        }

        [[nodiscard]] float fma_dot3(const float a0,
                                     const float b0,
                                     const float a1,
                                     const float b1,
                                     const float a2,
                                     const float b2) {
            float sum = 0.0f;
            sum = std::fma(a0, b0, sum);
            sum = std::fma(a1, b1, sum);
            sum = std::fma(a2, b2, sum);
            return sum;
        }

        [[nodiscard]] std::array<float, 3> make_cost_sample() {
            return {0.12573022f, -0.13210486f, 0.64042264f};
        }

        void quat_to_rotmat(const float qw, const float qx, const float qy, const float qz, std::array<float, 9>& out) {
            const float xx = qx * qx;
            const float yy = qy * qy;
            const float zz = qz * qz;
            const float wx = qw * qx;
            const float wy = qw * qy;
            const float wz = qw * qz;
            const float xy = qx * qy;
            const float xz = qx * qz;
            const float yz = qy * qz;

            out[0] = 1.0f - 2.0f * (yy + zz);
            out[1] = 2.0f * (xy - wz);
            out[2] = 2.0f * (xz + wy);
            out[3] = 2.0f * (xy + wz);
            out[4] = 1.0f - 2.0f * (xx + zz);
            out[5] = 2.0f * (yz - wx);
            out[6] = 2.0f * (xz - wy);
            out[7] = 2.0f * (yz + wx);
            out[8] = 1.0f - 2.0f * (xx + yy);
        }

        [[nodiscard]] std::array<float, 9> transpose3(const std::array<float, 9>& src) {
            return {
                src[0],
                src[3],
                src[6],
                src[1],
                src[4],
                src[7],
                src[2],
                src[5],
                src[8],
            };
        }

        void sigma_from_rot_var(const std::array<float, 9>& R,
                                const float vx,
                                const float vy,
                                const float vz,
                                std::array<float, 9>& out) {
            // Match NumPy's `np.matmul(R * v[None, :], R.T)` by first scaling
            // columns, then multiplying by the transposed rotation.
            const std::array<float, 3> variance = {vx, vy, vz};
            std::array<float, 9> scaled{};
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    const size_t idx = static_cast<size_t>(row * 3 + col);
                    scaled[idx] = strict_mul(R[idx], variance[static_cast<size_t>(col)]);
                }
            }
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    out[static_cast<size_t>(row * 3 + col)] = fma_dot3(
                        scaled[static_cast<size_t>(row * 3 + 0)],
                        R[static_cast<size_t>(col * 3 + 0)],
                        scaled[static_cast<size_t>(row * 3 + 1)],
                        R[static_cast<size_t>(col * 3 + 1)],
                        scaled[static_cast<size_t>(row * 3 + 2)],
                        R[static_cast<size_t>(col * 3 + 2)]);
                }
            }
        }

        void matmul_rowvec3_mat3(const std::array<float, 3>& lhs,
                                 const std::array<float, 9>& rhs,
                                 std::array<float, 3>& out) {
            out[0] = strict_sum3_prod(lhs[0], rhs[0], lhs[1], rhs[3], lhs[2], rhs[6]);
            out[1] = strict_sum3_prod(lhs[0], rhs[1], lhs[1], rhs[4], lhs[2], rhs[7]);
            out[2] = strict_sum3_prod(lhs[0], rhs[2], lhs[1], rhs[5], lhs[2], rhs[8]);
        }

        [[nodiscard]] float det3(const std::array<float, 9>& A) {
            return A[0] * (A[4] * A[8] - A[5] * A[7]) -
                   A[1] * (A[3] * A[8] - A[5] * A[6]) +
                   A[2] * (A[3] * A[7] - A[4] * A[6]);
        }

        template <typename T>
        [[nodiscard]] float logabsdet3_lu(const std::array<T, 9>& A) {
            std::array<double, 9> lu{};
            for (size_t i = 0; i < lu.size(); ++i)
                lu[i] = static_cast<double>(A[i]);
            double log_abs_det = 0.0;
            for (int k = 0; k < 3; ++k) {
                int pivot = k;
                double pivot_abs = std::abs(lu[static_cast<size_t>(k * 3 + k)]);
                for (int row = k + 1; row < 3; ++row) {
                    const double candidate_abs = std::abs(lu[static_cast<size_t>(row * 3 + k)]);
                    if (candidate_abs > pivot_abs) {
                        pivot = row;
                        pivot_abs = candidate_abs;
                    }
                }

                if (pivot_abs <= 0.0)
                    return static_cast<float>(std::log(kMinDet));

                if (pivot != k) {
                    for (int col = 0; col < 3; ++col) {
                        std::swap(
                            lu[static_cast<size_t>(k * 3 + col)],
                            lu[static_cast<size_t>(pivot * 3 + col)]);
                    }
                }

                const double pivot_value = lu[static_cast<size_t>(k * 3 + k)];
                log_abs_det += std::log(std::max(std::abs(pivot_value), static_cast<double>(kMinDet)));
                for (int row = k + 1; row < 3; ++row) {
                    const double factor = lu[static_cast<size_t>(row * 3 + k)] / pivot_value;
                    lu[static_cast<size_t>(row * 3 + k)] = factor;
                    for (int col = k + 1; col < 3; ++col) {
                        lu[static_cast<size_t>(row * 3 + col)] -=
                            factor * lu[static_cast<size_t>(k * 3 + col)];
                    }
                }
            }
            return static_cast<float>(std::max(log_abs_det, std::log(static_cast<double>(kMinDet))));
        }

        [[nodiscard]] float gauss_logpdf_diagrot(const float x,
                                                 const float y,
                                                 const float z,
                                                 const float mx,
                                                 const float my,
                                                 const float mz,
                                                 const std::array<float, 9>& R,
                                                 const std::array<float, 3>& invdiag,
                                                 const float logdet) {
            const float dx = x - mx;
            const float dy = y - my;
            const float dz = z - mz;
            // NumPy's `np.matmul(d, R)` for these 1x3 row-vector products rounds like
            // ordinary float32 mul/add, not a fused FMA chain.
            const float y0 = strict_sum3_prod(dx, R[0], dy, R[3], dz, R[6]);
            const float y1 = strict_sum3_prod(dx, R[1], dy, R[4], dz, R[7]);
            const float y2 = strict_sum3_prod(dx, R[2], dy, R[5], dz, R[8]);
            const float quad = strict_add(
                strict_add(
                    strict_mul(strict_mul(y0, y0), invdiag[0]),
                    strict_mul(strict_mul(y1, y1), invdiag[1])),
                strict_mul(strict_mul(y2, y2), invdiag[2]));
            const float term = strict_add(strict_add(strict_mul(3.0f, kLog2Pi), logdet), quad);
            return strict_mul(-0.5f, term);
        }

        [[nodiscard]] float log_add_exp(const float a, const float b) {
            if (a == -std::numeric_limits<float>::infinity())
                return b;
            if (b == -std::numeric_limits<float>::infinity())
                return a;
            const double da = static_cast<double>(a);
            const double db = static_cast<double>(b);
            const double m = std::max(da, db);
            return static_cast<float>(m + std::log1p(std::exp(-std::abs(da - db))));
        }

        [[nodiscard]] float pairwise_sum_inplace(float* values, int count) {
            if (count <= 0)
                return 0.0f;
            int active = count;
            while (active > 1) {
                int write = 0;
                int read = 0;
                for (; read + 1 < active; read += 2)
                    values[write++] = strict_add(values[read], values[read + 1]);
                if (read < active)
                    values[write++] = values[read];
                active = write;
            }
            return values[0];
        }

        [[nodiscard]] NativeRows rows_from_workset(const SplatSimplifyWorkset& workset) {
            NativeRows rows;
            rows.count = workset.size();
            rows.app_dim = rows.count > 0 ? static_cast<int>(workset.appearance.size(1)) : 0;
            rows.means = workset.means.cpu().contiguous().to_vector();
            rows.scales = workset.scaling.cpu().contiguous().to_vector();
            rows.rotation = workset.rotation.cpu().contiguous().to_vector();
            rows.opacity = workset.opacity.reshape({rows.count}).cpu().contiguous().to_vector();
            rows.appearance = workset.appearance.cpu().contiguous().to_vector();

            for (size_t i = 0; i < rows.scales.size(); ++i)
                rows.scales[i] = activated_scale(rows.scales[i]);

            for (int i = 0; i < rows.count; ++i) {
                rows.opacity[static_cast<size_t>(i)] = sigmoid(rows.opacity[static_cast<size_t>(i)]);

                const size_t i4 = static_cast<size_t>(i) * 4;
                float qw = rows.rotation[i4 + 0];
                float qx = rows.rotation[i4 + 1];
                float qy = rows.rotation[i4 + 2];
                float qz = rows.rotation[i4 + 3];
                const float inv_q = 1.0f / std::max(std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz), kMinQuatNorm);
                rows.rotation[i4 + 0] = qw * inv_q;
                rows.rotation[i4 + 1] = qx * inv_q;
                rows.rotation[i4 + 2] = qy * inv_q;
                rows.rotation[i4 + 3] = qz * inv_q;
            }
            return rows;
        }

        [[nodiscard]] SplatSimplifyWorkset workset_from_rows(const NativeRows& rows, const SplatSimplifyWorkset& template_workset) {
            SplatSimplifyWorkset out = template_workset;
            out.means = Tensor::from_vector(rows.means, {static_cast<size_t>(rows.count), size_t{3}}, Device::CPU);
            std::vector<float> scaling_raw(rows.scales.size());
            for (size_t i = 0; i < rows.scales.size(); ++i)
                scaling_raw[i] = std::log(std::max(rows.scales[i], kMinScale));

            std::vector<float> opacity_raw(rows.opacity.size());
            for (size_t i = 0; i < rows.opacity.size(); ++i)
                opacity_raw[i] = logit_from_alpha(rows.opacity[i]);

            out.scaling = Tensor::from_vector(scaling_raw, {static_cast<size_t>(rows.count), size_t{3}}, Device::CPU);
            out.rotation = Tensor::from_vector(rows.rotation, {static_cast<size_t>(rows.count), size_t{4}}, Device::CPU);
            out.opacity = Tensor::from_vector(opacity_raw, {static_cast<size_t>(rows.count), size_t{1}}, Device::CPU);
            out.appearance = Tensor::from_vector(
                rows.appearance,
                {static_cast<size_t>(rows.count), static_cast<size_t>(rows.app_dim)},
                Device::CPU);
            return out;
        }

        void copy_row(const NativeRows& src, const int src_row, NativeRows& dst, const int dst_row) {
            std::copy_n(src.means.begin() + static_cast<ptrdiff_t>(src_row * 3), 3, dst.means.begin() + static_cast<ptrdiff_t>(dst_row * 3));
            std::copy_n(src.scales.begin() + static_cast<ptrdiff_t>(src_row * 3), 3, dst.scales.begin() + static_cast<ptrdiff_t>(dst_row * 3));
            std::copy_n(src.rotation.begin() + static_cast<ptrdiff_t>(src_row * 4), 4, dst.rotation.begin() + static_cast<ptrdiff_t>(dst_row * 4));
            dst.opacity[static_cast<size_t>(dst_row)] = src.opacity[static_cast<size_t>(src_row)];
            if (src.app_dim > 0) {
                std::copy_n(src.appearance.begin() + static_cast<ptrdiff_t>(src_row * src.app_dim),
                            src.app_dim,
                            dst.appearance.begin() + static_cast<ptrdiff_t>(dst_row * src.app_dim));
            }
        }

        [[nodiscard]] float median_of(std::vector<float> values) {
            if (values.empty())
                return 0.0f;
            std::sort(values.begin(), values.end());
            const size_t mid = values.size() / 2;
            if ((values.size() & 1U) != 0U)
                return values[mid];
            return 0.5f * (values[mid - 1] + values[mid]);
        }

        [[nodiscard]] NativeRows prune_by_opacity(const NativeRows& input, const float requested_threshold) {
            if (input.count == 0)
                return input;

            const float median_alpha = median_of(input.opacity);
            const float threshold = std::min(requested_threshold, median_alpha);

            std::vector<int> keep_idx;
            keep_idx.reserve(static_cast<size_t>(input.count));
            for (int i = 0; i < input.count; ++i) {
                if (input.opacity[static_cast<size_t>(i)] >= threshold)
                    keep_idx.push_back(i);
            }

            NativeRows out;
            out.count = static_cast<int>(keep_idx.size());
            out.app_dim = input.app_dim;
            out.means.resize(static_cast<size_t>(out.count) * 3);
            out.scales.resize(static_cast<size_t>(out.count) * 3);
            out.rotation.resize(static_cast<size_t>(out.count) * 4);
            out.opacity.resize(static_cast<size_t>(out.count));
            out.appearance.resize(static_cast<size_t>(out.count) * static_cast<size_t>(out.app_dim));

            for (int dst_row = 0; dst_row < out.count; ++dst_row)
                copy_row(input, keep_idx[static_cast<size_t>(dst_row)], out, dst_row);
            return out;
        }

        void build_cache(const NativeRows& rows, std::vector<CacheEntry>& cache) {
            cache.resize(static_cast<size_t>(rows.count));
            tbb::parallel_for(tbb::blocked_range<int>(0, rows.count), [&](const tbb::blocked_range<int>& range) {
                for (int i = range.begin(); i != range.end(); ++i) {
                    auto& entry = cache[static_cast<size_t>(i)];
                    const size_t i3 = static_cast<size_t>(i) * 3;
                    const size_t i4 = static_cast<size_t>(i) * 4;

                    const float sx = std::max(rows.scales[i3 + 0], kMinScale);
                    const float sy = std::max(rows.scales[i3 + 1], kMinScale);
                    const float sz = std::max(rows.scales[i3 + 2], kMinScale);
                    entry.variance = {
                        sx * sx + kEpsCov,
                        sy * sy + kEpsCov,
                        sz * sz + kEpsCov,
                    };
                    entry.invdiag = {
                        1.0f / std::max(entry.variance[0], kMinDet),
                        1.0f / std::max(entry.variance[1], kMinDet),
                        1.0f / std::max(entry.variance[2], kMinDet),
                    };
                    const double variance_prod =
                        static_cast<double>(std::max(entry.variance[0], kMinDet)) *
                        static_cast<double>(std::max(entry.variance[1], kMinDet)) *
                        static_cast<double>(std::max(entry.variance[2], kMinDet));
                    entry.logdet = static_cast<float>(std::log(std::max(variance_prod, static_cast<double>(kMinDet))));

                    const float qw = rows.rotation[i4 + 0];
                    const float qx = rows.rotation[i4 + 1];
                    const float qy = rows.rotation[i4 + 2];
                    const float qz = rows.rotation[i4 + 3];
                    quat_to_rotmat(qw, qx, qy, qz, entry.R);
                    entry.Rt = transpose3(entry.R);
                    sigma_from_rot_var(entry.R, entry.variance[0], entry.variance[1], entry.variance[2], entry.sigma);

                    const float alpha = rows.opacity[static_cast<size_t>(i)];
                    const float scale_prod = strict_prod3(sx, sy, sz);
                    entry.mass = strict_add(strict_mul(strict_mul(kTwoPiPow1p5, alpha), scale_prod), 1e-12f);
                }
            });
        }

        [[nodiscard]] float compute_edge_cost(const NativeRows& rows,
                                              const std::vector<CacheEntry>& cache,
                                              const int i,
                                              const int j,
                                              const std::array<float, 3>& z) {
            const size_t i3 = static_cast<size_t>(i) * 3;
            const size_t j3 = static_cast<size_t>(j) * 3;

            const float mux = rows.means[i3 + 0];
            const float muy = rows.means[i3 + 1];
            const float muz = rows.means[i3 + 2];
            const float mvx = rows.means[j3 + 0];
            const float mvy = rows.means[j3 + 1];
            const float mvz = rows.means[j3 + 2];

            const float wi = cache[static_cast<size_t>(i)].mass;
            const float wj = cache[static_cast<size_t>(j)].mass;
            const float W = wi + wj;
            const float Wsafe = W > 0.0f ? W : 1.0f;
            const float pi = std::clamp(wi / Wsafe, 1e-12f, 1.0f - 1e-12f);
            const float pj = 1.0f - pi;
            const float log_pi = std::log(pi);
            const float log_pj = std::log(pj);

            const std::array<float, 3> mu_m = {
                strict_add(strict_mul(pi, mux), strict_mul(pj, mvx)),
                strict_add(strict_mul(pi, muy), strict_mul(pj, mvy)),
                strict_add(strict_mul(pi, muz), strict_mul(pj, mvz)),
            };
            const std::array<float, 3> d_i = {
                mux - mu_m[0],
                muy - mu_m[1],
                muz - mu_m[2],
            };
            const std::array<float, 3> d_j = {
                mvx - mu_m[0],
                mvy - mu_m[1],
                mvz - mu_m[2],
            };

            std::array<float, 9> sigma_i = cache[static_cast<size_t>(i)].sigma;
            std::array<float, 9> sigma_j = cache[static_cast<size_t>(j)].sigma;
            sigma_i[0] = strict_add(sigma_i[0], strict_mul(d_i[0], d_i[0]));
            sigma_i[1] = strict_add(sigma_i[1], strict_mul(d_i[0], d_i[1]));
            sigma_i[2] = strict_add(sigma_i[2], strict_mul(d_i[0], d_i[2]));
            sigma_i[3] = strict_add(sigma_i[3], strict_mul(d_i[1], d_i[0]));
            sigma_i[4] = strict_add(sigma_i[4], strict_mul(d_i[1], d_i[1]));
            sigma_i[5] = strict_add(sigma_i[5], strict_mul(d_i[1], d_i[2]));
            sigma_i[6] = strict_add(sigma_i[6], strict_mul(d_i[2], d_i[0]));
            sigma_i[7] = strict_add(sigma_i[7], strict_mul(d_i[2], d_i[1]));
            sigma_i[8] = strict_add(sigma_i[8], strict_mul(d_i[2], d_i[2]));
            sigma_j[0] = strict_add(sigma_j[0], strict_mul(d_j[0], d_j[0]));
            sigma_j[1] = strict_add(sigma_j[1], strict_mul(d_j[0], d_j[1]));
            sigma_j[2] = strict_add(sigma_j[2], strict_mul(d_j[0], d_j[2]));
            sigma_j[3] = strict_add(sigma_j[3], strict_mul(d_j[1], d_j[0]));
            sigma_j[4] = strict_add(sigma_j[4], strict_mul(d_j[1], d_j[1]));
            sigma_j[5] = strict_add(sigma_j[5], strict_mul(d_j[1], d_j[2]));
            sigma_j[6] = strict_add(sigma_j[6], strict_mul(d_j[2], d_j[0]));
            sigma_j[7] = strict_add(sigma_j[7], strict_mul(d_j[2], d_j[1]));
            sigma_j[8] = strict_add(sigma_j[8], strict_mul(d_j[2], d_j[2]));

            std::array<float, 9> sigma_m{};
            for (int a = 0; a < 9; ++a) {
                sigma_m[static_cast<size_t>(a)] = strict_add(
                    strict_mul(pi, sigma_i[static_cast<size_t>(a)]),
                    strict_mul(pj, sigma_j[static_cast<size_t>(a)]));
            }
            sigma_m[1] = sigma_m[3] = strict_mul(0.5f, strict_add(sigma_m[1], sigma_m[3]));
            sigma_m[2] = sigma_m[6] = strict_mul(0.5f, strict_add(sigma_m[2], sigma_m[6]));
            sigma_m[5] = sigma_m[7] = strict_mul(0.5f, strict_add(sigma_m[5], sigma_m[7]));
            sigma_m[0] = strict_add(sigma_m[0], kEpsCov);
            sigma_m[4] = strict_add(sigma_m[4], kEpsCov);
            sigma_m[8] = strict_add(sigma_m[8], kEpsCov);

            const float logdet_m = logabsdet3_lu(sigma_m);
            const float ep_neg_log_q = 0.5f * (3.0f * kLog2Pi + logdet_m + 3.0f);

            const float stdix = std::sqrt(std::max(cache[static_cast<size_t>(i)].variance[0], 0.0f));
            const float stdiy = std::sqrt(std::max(cache[static_cast<size_t>(i)].variance[1], 0.0f));
            const float stdiz = std::sqrt(std::max(cache[static_cast<size_t>(i)].variance[2], 0.0f));
            const float stdjx = std::sqrt(std::max(cache[static_cast<size_t>(j)].variance[0], 0.0f));
            const float stdjy = std::sqrt(std::max(cache[static_cast<size_t>(j)].variance[1], 0.0f));
            const float stdjz = std::sqrt(std::max(cache[static_cast<size_t>(j)].variance[2], 0.0f));

            const auto& Rt_i = cache[static_cast<size_t>(i)].Rt;
            const auto& Rt_j = cache[static_cast<size_t>(j)].Rt;
            const auto& R_i = cache[static_cast<size_t>(i)].R;
            const auto& R_j = cache[static_cast<size_t>(j)].R;

            const float zi0 = strict_mul(z[0], stdix);
            const float zi1 = strict_mul(z[1], stdiy);
            const float zi2 = strict_mul(z[2], stdiz);
            const float zj0 = strict_mul(z[0], stdjx);
            const float zj1 = strict_mul(z[1], stdjy);
            const float zj2 = strict_mul(z[2], stdjz);
            std::array<float, 3> zi = {zi0, zi1, zi2};
            std::array<float, 3> zj = {zj0, zj1, zj2};
            std::array<float, 3> offset_i{};
            std::array<float, 3> offset_j{};
            matmul_rowvec3_mat3(zi, Rt_i, offset_i);
            matmul_rowvec3_mat3(zj, Rt_j, offset_j);
            const std::array<float, 3> x_i = {
                strict_add(mux, offset_i[0]),
                strict_add(muy, offset_i[1]),
                strict_add(muz, offset_i[2]),
            };
            const std::array<float, 3> x_j = {
                strict_add(mvx, offset_j[0]),
                strict_add(mvy, offset_j[1]),
                strict_add(mvz, offset_j[2]),
            };

            const float log_ni_on_i = gauss_logpdf_diagrot(
                x_i[0], x_i[1], x_i[2], mux, muy, muz, R_i, cache[static_cast<size_t>(i)].invdiag, cache[static_cast<size_t>(i)].logdet);
            const float log_nj_on_i = gauss_logpdf_diagrot(
                x_i[0], x_i[1], x_i[2], mvx, mvy, mvz, R_j, cache[static_cast<size_t>(j)].invdiag, cache[static_cast<size_t>(j)].logdet);
            const float log_ni_on_j = gauss_logpdf_diagrot(
                x_j[0], x_j[1], x_j[2], mux, muy, muz, R_i, cache[static_cast<size_t>(i)].invdiag, cache[static_cast<size_t>(i)].logdet);
            const float log_nj_on_j = gauss_logpdf_diagrot(
                x_j[0], x_j[1], x_j[2], mvx, mvy, mvz, R_j, cache[static_cast<size_t>(j)].invdiag, cache[static_cast<size_t>(j)].logdet);

            const float ei = log_add_exp(log_pi + log_ni_on_i, log_pj + log_nj_on_i);
            const float ej = log_add_exp(log_pi + log_ni_on_j, log_pj + log_nj_on_j);
            float cost = strict_add(
                strict_add(
                    strict_mul(pi, ei),
                    strict_mul(pj, ej)),
                ep_neg_log_q);

            const size_t ai = static_cast<size_t>(i) * static_cast<size_t>(rows.app_dim);
            const size_t aj = static_cast<size_t>(j) * static_cast<size_t>(rows.app_dim);
            if (rows.app_dim > 0) {
                std::array<float, 64> stack_sq{};
                std::vector<float> heap_sq;
                float* sq = nullptr;
                if (rows.app_dim <= static_cast<int>(stack_sq.size())) {
                    sq = stack_sq.data();
                } else {
                    heap_sq.resize(static_cast<size_t>(rows.app_dim));
                    sq = heap_sq.data();
                }
                for (int k = 0; k < rows.app_dim; ++k) {
                    const float d = strict_sub(
                        rows.appearance[ai + static_cast<size_t>(k)],
                        rows.appearance[aj + static_cast<size_t>(k)]);
                    sq[k] = strict_mul(d, d);
                }
                cost = strict_add(cost, pairwise_sum_inplace(sq, rows.app_dim));
            }
            return cost;
        }

        [[nodiscard]] Eigen3x3 sort_eigendecomposition(const Eigen3x3& out) {
            std::array<int, 3> order = {0, 1, 2};
            std::sort(order.begin(), order.end(), [&](const int lhs, const int rhs) {
                if (out.values[static_cast<size_t>(lhs)] != out.values[static_cast<size_t>(rhs)])
                    return out.values[static_cast<size_t>(lhs)] > out.values[static_cast<size_t>(rhs)];
                return lhs < rhs;
            });

            Eigen3x3 sorted;
            for (int col = 0; col < 3; ++col) {
                const int src_col = order[static_cast<size_t>(col)];
                sorted.values[static_cast<size_t>(col)] = out.values[static_cast<size_t>(src_col)];
                for (int row = 0; row < 3; ++row)
                    sorted.vectors[static_cast<size_t>(row * 3 + col)] = out.vectors[static_cast<size_t>(row * 3 + src_col)];
            }

            if (det3(sorted.vectors) < 0.0f) {
                sorted.vectors[2] *= -1.0f;
                sorted.vectors[5] *= -1.0f;
                sorted.vectors[8] *= -1.0f;
            }
            return sorted;
        }

        [[nodiscard]] Eigen3x3 eigen_symmetric_3x3_jacobi(const std::array<float, 9>& Ain) {
            std::array<float, 9> A = Ain;
            std::array<float, 9> V = {
                1.0f,
                0.0f,
                0.0f,
                0.0f,
                1.0f,
                0.0f,
                0.0f,
                0.0f,
                1.0f,
            };

            for (int iter = 0; iter < kJacobiIterations; ++iter) {
                int p = 0;
                int q = 1;
                float max_abs = std::abs(A[1]);
                if (std::abs(A[2]) > max_abs) {
                    p = 0;
                    q = 2;
                    max_abs = std::abs(A[2]);
                }
                if (std::abs(A[5]) > max_abs) {
                    p = 1;
                    q = 2;
                    max_abs = std::abs(A[5]);
                }
                if (max_abs < 1e-12f)
                    break;

                const int pp = 3 * p + p;
                const int qq = 3 * q + q;
                const int pq = 3 * p + q;
                const float app = A[static_cast<size_t>(pp)];
                const float aqq = A[static_cast<size_t>(qq)];
                const float apq = A[static_cast<size_t>(pq)];
                const float tau = (aqq - app) / (2.0f * apq);
                const float t = std::copysign(1.0f, tau) / (std::abs(tau) + std::sqrt(1.0f + tau * tau));
                const float c = 1.0f / std::sqrt(1.0f + t * t);
                const float s = t * c;

                for (int k = 0; k < 3; ++k) {
                    if (k == p || k == q)
                        continue;
                    const int kp = 3 * k + p;
                    const int kq = 3 * k + q;
                    const float akp = A[static_cast<size_t>(kp)];
                    const float akq = A[static_cast<size_t>(kq)];
                    A[static_cast<size_t>(kp)] = c * akp - s * akq;
                    A[static_cast<size_t>(3 * p + k)] = A[static_cast<size_t>(kp)];
                    A[static_cast<size_t>(kq)] = s * akp + c * akq;
                    A[static_cast<size_t>(3 * q + k)] = A[static_cast<size_t>(kq)];
                }

                A[static_cast<size_t>(pp)] = c * c * app - 2.0f * s * c * apq + s * s * aqq;
                A[static_cast<size_t>(qq)] = s * s * app + 2.0f * s * c * apq + c * c * aqq;
                A[static_cast<size_t>(pq)] = 0.0f;
                A[static_cast<size_t>(3 * q + p)] = 0.0f;

                for (int k = 0; k < 3; ++k) {
                    const int kp = 3 * k + p;
                    const int kq = 3 * k + q;
                    const float vkp = V[static_cast<size_t>(kp)];
                    const float vkq = V[static_cast<size_t>(kq)];
                    V[static_cast<size_t>(kp)] = c * vkp - s * vkq;
                    V[static_cast<size_t>(kq)] = s * vkp + c * vkq;
                }
            }

            Eigen3x3 out;
            out.values = {A[0], A[4], A[8]};
            out.vectors = V;
            return sort_eigendecomposition(out);
        }

        [[nodiscard]] Eigen3x3 eigen_symmetric_3x3(const std::array<float, 9>& Ain) {
            return eigen_symmetric_3x3_jacobi(Ain);
        }

        void rotmat_to_quat(const std::array<float, 9>& R, std::array<float, 4>& out) {
            const float m00 = R[0];
            const float m11 = R[4];
            const float m22 = R[8];
            const float tr = m00 + m11 + m22;
            float qw = 0.0f;
            float qx = 0.0f;
            float qy = 0.0f;
            float qz = 0.0f;

            if (tr > 0.0f) {
                const float S = std::sqrt(tr + 1.0f) * 2.0f;
                qw = 0.25f * S;
                qx = (R[7] - R[5]) / S;
                qy = (R[2] - R[6]) / S;
                qz = (R[3] - R[1]) / S;
            } else if (R[0] > R[4] && R[0] > R[8]) {
                const float S = std::sqrt(1.0f + R[0] - R[4] - R[8]) * 2.0f;
                qw = (R[7] - R[5]) / S;
                qx = 0.25f * S;
                qy = (R[1] + R[3]) / S;
                qz = (R[2] + R[6]) / S;
            } else if (R[4] > R[8]) {
                const float S = std::sqrt(1.0f + R[4] - R[0] - R[8]) * 2.0f;
                qw = (R[2] - R[6]) / S;
                qx = (R[1] + R[3]) / S;
                qy = 0.25f * S;
                qz = (R[5] + R[7]) / S;
            } else {
                const float S = std::sqrt(1.0f + R[8] - R[0] - R[4]) * 2.0f;
                qw = (R[3] - R[1]) / S;
                qx = (R[2] + R[6]) / S;
                qy = (R[5] + R[7]) / S;
                qz = 0.25f * S;
            }

            const float inv_n = 1.0f / std::max(std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz), kMinQuatNorm);
            out[0] = qw * inv_n;
            out[1] = qx * inv_n;
            out[2] = qy * inv_n;
            out[3] = qz * inv_n;
        }

        void decompose_sigma_to_raw_scale_quat(const std::array<float, 9>& sigma,
                                               std::array<float, 3>& scaling_raw,
                                               std::array<float, 4>& rotation_raw) {
            const auto eig = eigen_symmetric_3x3(sigma);
            std::array<float, 3> evals = {
                std::max(eig.values[0], kMinEval),
                std::max(eig.values[1], kMinEval),
                std::max(eig.values[2], kMinEval),
            };

            scaling_raw[0] = std::log(std::max(std::sqrt(evals[0]), kMinScale));
            scaling_raw[1] = std::log(std::max(std::sqrt(evals[1]), kMinScale));
            scaling_raw[2] = std::log(std::max(std::sqrt(evals[2]), kMinScale));
            rotmat_to_quat(eig.vectors, rotation_raw);
        }

        [[nodiscard]] std::vector<std::pair<int, int>> build_knn_union_edges(const NativeRows& rows, const int knn_k) {
            if (rows.count <= 1 || knn_k <= 0)
                return {};

            const int k_eff = std::min(std::max(1, knn_k), std::max(1, rows.count - 1));
            PointCloudAdaptor cloud{rows.means.data(), static_cast<size_t>(rows.count)};
            KDTree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
            index.buildIndex();

            tbb::enumerable_thread_specific<std::vector<std::uint64_t>> local_edge_keys;

            tbb::parallel_for(tbb::blocked_range<int>(0, rows.count), [&](const tbb::blocked_range<int>& range) {
                std::vector<size_t> ret_indices;
                std::vector<double> out_dists_sqr;
                auto& edge_keys = local_edge_keys.local();
                if (edge_keys.empty())
                    edge_keys.reserve(static_cast<size_t>(range.size()) * static_cast<size_t>(k_eff));

                for (int i = range.begin(); i != range.end(); ++i) {
                    const size_t query_count = static_cast<size_t>(std::min(rows.count, k_eff + 1));
                    ret_indices.assign(query_count, 0);
                    out_dists_sqr.assign(query_count, 0.0);
                    nanoflann::KNNResultSet<double> result_set(query_count);
                    result_set.init(ret_indices.data(), out_dists_sqr.data());
                    const double query[3] = {
                        static_cast<double>(rows.means[static_cast<size_t>(i) * 3 + 0]),
                        static_cast<double>(rows.means[static_cast<size_t>(i) * 3 + 1]),
                        static_cast<double>(rows.means[static_cast<size_t>(i) * 3 + 2]),
                    };
                    index.findNeighbors(result_set, query, nanoflann::SearchParameters(0.0f, true));

                    const size_t take = std::min(static_cast<size_t>(k_eff), result_set.size() > 0 ? result_set.size() - 1 : size_t{0});
                    for (size_t j = 0; j < take; ++j) {
                        const int neighbor = static_cast<int>(ret_indices[j + 1]);
                        if (neighbor < 0 || neighbor == i)
                            continue;
                        const int u = std::min(i, neighbor);
                        const int v = std::max(i, neighbor);
                        edge_keys.push_back(
                            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32) |
                            static_cast<std::uint32_t>(v));
                    }
                }
            });

            size_t total_edge_keys = 0;
            for (const auto& local : local_edge_keys)
                total_edge_keys += local.size();

            std::vector<std::uint64_t> edge_keys;
            edge_keys.reserve(total_edge_keys);
            for (const auto& local : local_edge_keys)
                edge_keys.insert(edge_keys.end(), local.begin(), local.end());

            std::sort(edge_keys.begin(), edge_keys.end());
            edge_keys.erase(std::unique(edge_keys.begin(), edge_keys.end()), edge_keys.end());

            std::vector<std::pair<int, int>> edges;
            edges.reserve(edge_keys.size());
            for (const std::uint64_t key : edge_keys) {
                const int u = static_cast<int>(key >> 32);
                const int v = static_cast<int>(key & 0xffffffffU);
                edges.emplace_back(u, v);
            }
            return edges;
        }

        void compute_edge_costs(const NativeRows& rows,
                                const std::vector<CacheEntry>& cache,
                                const std::vector<std::pair<int, int>>& edges,
                                std::vector<float>& costs) {
            const auto z = make_cost_sample();
            costs.assign(edges.size(), std::numeric_limits<float>::infinity());

            tbb::parallel_for(tbb::blocked_range<size_t>(0, edges.size()), [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    const auto [u, v] = edges[i];
                    costs[i] = compute_edge_cost(rows, cache, u, v, z);
                }
            });
        }

        void greedy_pairs_from_edges(const std::vector<std::pair<int, int>>& edges,
                                     const std::vector<float>& costs,
                                     const int count,
                                     const int max_pairs,
                                     std::vector<size_t>& order,
                                     std::vector<uint8_t>& used,
                                     std::vector<std::pair<int, int>>& pairs) {
            order.clear();
            order.reserve(edges.size());
            for (size_t i = 0; i < costs.size(); ++i) {
                if (std::isfinite(costs[i]))
                    order.push_back(i);
            }
            std::stable_sort(order.begin(), order.end(), [&](const size_t lhs, const size_t rhs) {
                return costs[lhs] < costs[rhs];
            });

            used.assign(static_cast<size_t>(count), uint8_t{0});
            pairs.clear();
            pairs.reserve(static_cast<size_t>(std::max(0, max_pairs)));
            for (const size_t edge_idx : order) {
                const auto [u, v] = edges[edge_idx];
                if (used[static_cast<size_t>(u)] || used[static_cast<size_t>(v)])
                    continue;
                used[static_cast<size_t>(u)] = 1;
                used[static_cast<size_t>(v)] = 1;
                pairs.emplace_back(u, v);
                if (max_pairs > 0 && static_cast<int>(pairs.size()) >= max_pairs)
                    break;
            }
        }

        [[nodiscard]] NativeRows merge_pairs(const NativeRows& input,
                                             const std::vector<CacheEntry>& cache,
                                             const std::vector<std::pair<int, int>>& pairs,
                                             std::vector<uint8_t>& used,
                                             std::vector<int>& keep_idx) {
            if (pairs.empty())
                return input;

            used.assign(static_cast<size_t>(input.count), uint8_t{0});
            for (const auto [u, v] : pairs) {
                used[static_cast<size_t>(u)] = 1;
                used[static_cast<size_t>(v)] = 1;
            }

            keep_idx.clear();
            keep_idx.reserve(static_cast<size_t>(input.count));
            for (int i = 0; i < input.count; ++i) {
                if (!used[static_cast<size_t>(i)])
                    keep_idx.push_back(i);
            }

            NativeRows out;
            out.count = static_cast<int>(keep_idx.size() + pairs.size());
            out.app_dim = input.app_dim;
            out.means.resize(static_cast<size_t>(out.count) * 3);
            out.scales.resize(static_cast<size_t>(out.count) * 3);
            out.rotation.resize(static_cast<size_t>(out.count) * 4);
            out.opacity.resize(static_cast<size_t>(out.count));
            out.appearance.resize(static_cast<size_t>(out.count) * static_cast<size_t>(out.app_dim));

            tbb::parallel_for(tbb::blocked_range<int>(0, static_cast<int>(keep_idx.size())), [&](const tbb::blocked_range<int>& range) {
                for (int dst_row = range.begin(); dst_row != range.end(); ++dst_row)
                    copy_row(input, keep_idx[static_cast<size_t>(dst_row)], out, dst_row);
            });

            tbb::parallel_for(tbb::blocked_range<int>(0, static_cast<int>(pairs.size())), [&](const tbb::blocked_range<int>& range) {
                for (int pair_idx = range.begin(); pair_idx != range.end(); ++pair_idx) {
                    const auto [i, j] = pairs[static_cast<size_t>(pair_idx)];
                    const auto& cache_i = cache[static_cast<size_t>(i)];
                    const auto& cache_j = cache[static_cast<size_t>(j)];
                    const size_t i3 = static_cast<size_t>(i) * 3;
                    const size_t j3 = static_cast<size_t>(j) * 3;

                    const float sxi = std::max(input.scales[i3 + 0], kMinScale);
                    const float syi = std::max(input.scales[i3 + 1], kMinScale);
                    const float szi = std::max(input.scales[i3 + 2], kMinScale);
                    const float sxj = std::max(input.scales[j3 + 0], kMinScale);
                    const float syj = std::max(input.scales[j3 + 1], kMinScale);
                    const float szj = std::max(input.scales[j3 + 2], kMinScale);

                    const float alpha_i = input.opacity[static_cast<size_t>(i)];
                    const float alpha_j = input.opacity[static_cast<size_t>(j)];
                    const float wi = cache_i.mass;
                    const float wj = cache_j.mass;
                    const float W = std::max(wi + wj, 1e-12f);

                    const int out_row = static_cast<int>(keep_idx.size()) + pair_idx;
                    const size_t o3 = static_cast<size_t>(out_row) * 3;
                    const size_t o4 = static_cast<size_t>(out_row) * 4;

                    out.means[o3 + 0] = (wi * input.means[i3 + 0] + wj * input.means[j3 + 0]) / W;
                    out.means[o3 + 1] = (wi * input.means[i3 + 1] + wj * input.means[j3 + 1]) / W;
                    out.means[o3 + 2] = (wi * input.means[i3 + 2] + wj * input.means[j3 + 2]) / W;

                    std::array<float, 9> sig_i{};
                    std::array<float, 9> sig_j{};
                    sigma_from_rot_var(cache_i.R, sxi * sxi, syi * syi, szi * szi, sig_i);
                    sigma_from_rot_var(cache_j.R, sxj * sxj, syj * syj, szj * szj, sig_j);

                    const float dix = input.means[i3 + 0] - out.means[o3 + 0];
                    const float diy = input.means[i3 + 1] - out.means[o3 + 1];
                    const float diz = input.means[i3 + 2] - out.means[o3 + 2];
                    const float djx = input.means[j3 + 0] - out.means[o3 + 0];
                    const float djy = input.means[j3 + 1] - out.means[o3 + 1];
                    const float djz = input.means[j3 + 2] - out.means[o3 + 2];

                    sig_i[0] += dix * dix;
                    sig_i[1] += dix * diy;
                    sig_i[2] += dix * diz;
                    sig_i[3] += diy * dix;
                    sig_i[4] += diy * diy;
                    sig_i[5] += diy * diz;
                    sig_i[6] += diz * dix;
                    sig_i[7] += diz * diy;
                    sig_i[8] += diz * diz;
                    sig_j[0] += djx * djx;
                    sig_j[1] += djx * djy;
                    sig_j[2] += djx * djz;
                    sig_j[3] += djy * djx;
                    sig_j[4] += djy * djy;
                    sig_j[5] += djy * djz;
                    sig_j[6] += djz * djx;
                    sig_j[7] += djz * djy;
                    sig_j[8] += djz * djz;

                    std::array<float, 9> sigma{};
                    for (int a = 0; a < 9; ++a) {
                        sigma[static_cast<size_t>(a)] =
                            (wi * sig_i[static_cast<size_t>(a)] + wj * sig_j[static_cast<size_t>(a)]) / W;
                    }
                    sigma[1] = sigma[3] = 0.5f * (sigma[1] + sigma[3]);
                    sigma[2] = sigma[6] = 0.5f * (sigma[2] + sigma[6]);
                    sigma[5] = sigma[7] = 0.5f * (sigma[5] + sigma[7]);
                    sigma[0] += kEpsCov;
                    sigma[4] += kEpsCov;
                    sigma[8] += kEpsCov;

                    std::array<float, 3> scaling_raw{};
                    std::array<float, 4> rotation{};
                    decompose_sigma_to_raw_scale_quat(sigma, scaling_raw, rotation);

                    out.scales[o3 + 0] = activated_scale(scaling_raw[0]);
                    out.scales[o3 + 1] = activated_scale(scaling_raw[1]);
                    out.scales[o3 + 2] = activated_scale(scaling_raw[2]);
                    out.rotation[o4 + 0] = rotation[0];
                    out.rotation[o4 + 1] = rotation[1];
                    out.rotation[o4 + 2] = rotation[2];
                    out.rotation[o4 + 3] = rotation[3];
                    out.opacity[static_cast<size_t>(out_row)] = alpha_i + alpha_j - alpha_i * alpha_j;

                    const size_t ai = static_cast<size_t>(i) * static_cast<size_t>(input.app_dim);
                    const size_t aj = static_cast<size_t>(j) * static_cast<size_t>(input.app_dim);
                    const size_t ao = static_cast<size_t>(out_row) * static_cast<size_t>(input.app_dim);
                    for (int k = 0; k < input.app_dim; ++k) {
                        out.appearance[ao + static_cast<size_t>(k)] =
                            (wi * input.appearance[ai + static_cast<size_t>(k)] +
                             wj * input.appearance[aj + static_cast<size_t>(k)]) /
                            W;
                    }
                }
            });

            return out;
        }

        [[nodiscard]] int target_count_for(const int input_count, const double ratio) {
            const double clamped_ratio = std::clamp(ratio, 0.0, 1.0);
            return std::clamp(
                static_cast<int>(std::ceil(static_cast<double>(input_count) * clamped_ratio)),
                1,
                std::max(1, input_count));
        }

        [[nodiscard]] int pass_merge_cap_for(const int input_count, const double merge_cap) {
            const double clamped_merge_cap = std::clamp(merge_cap, 0.01, 0.5);
            return std::max(1, static_cast<int>(clamped_merge_cap * static_cast<double>(input_count)));
        }

        [[nodiscard]] float progress_for_count(const int input_count, const int target_count, const int current_count) {
            if (input_count <= target_count)
                return 0.95f;
            const float denom = static_cast<float>(std::max(1, input_count - target_count));
            const float numer = static_cast<float>(std::clamp(input_count - current_count, 0, input_count - target_count));
            return 0.10f + 0.85f * (numer / denom);
        }

        [[nodiscard]] std::expected<SplatSimplifyWorkset, std::string> simplify_workset(
            const SplatSimplifyWorkset& input,
            const SplatSimplifyOptions& options,
            SplatSimplifyProgressCallback progress) {
            try {
                NativeRows current = rows_from_workset(input);
                if (current.count == 0)
                    return std::unexpected("Splat simplify: input splat is empty");

                const int input_count = current.count;
                const int target_count = target_count_for(input_count, options.ratio);
                const int pass_merge_cap = pass_merge_cap_for(input_count, options.merge_cap);
                SimplifyScratch scratch;

                if (!report_progress(progress, 0.0f, "Pruning opacity"))
                    return std::unexpected("Cancelled");
                current = prune_by_opacity(current, options.opacity_prune_threshold);

                if (current.count == 0)
                    return std::unexpected("Splat simplify: input has no visible gaussians");
                if (current.count <= target_count) {
                    (void)report_progress(progress, 1.0f, "Complete");
                    return workset_from_rows(current, input);
                }

                int pass = 0;
                while (current.count > target_count) {
                    const float pass_progress = progress_for_count(input_count, target_count, current.count);
                    const std::string pass_prefix = "Pass " + std::to_string(pass + 1) + ": ";

                    if (!report_progress(progress, pass_progress, pass_prefix + "building kNN graph"))
                        return std::unexpected("Cancelled");
                    const auto edges = build_knn_union_edges(current, options.knn_k);
                    if (edges.empty())
                        return std::unexpected(
                            "Splat simplify stalled at " + std::to_string(current.count) +
                            " gaussians (target " + std::to_string(target_count) + ")");

                    if (!report_progress(progress, pass_progress + 0.01f, pass_prefix + "computing edge costs"))
                        return std::unexpected("Cancelled");
                    build_cache(current, scratch.cache);
                    compute_edge_costs(current, scratch.cache, edges, scratch.costs);

                    if (!report_progress(progress, pass_progress + 0.02f, pass_prefix + "selecting pairs"))
                        return std::unexpected("Cancelled");
                    const int merges_needed = current.count - target_count;
                    const int max_pairs_this_pass = merges_needed > 0 ? std::min(merges_needed, pass_merge_cap) : 0;
                    greedy_pairs_from_edges(
                        edges,
                        scratch.costs,
                        current.count,
                        max_pairs_this_pass,
                        scratch.order,
                        scratch.used_rows,
                        scratch.pairs);
                    if (scratch.pairs.empty()) {
                        return std::unexpected(
                            "Splat simplify stalled at " + std::to_string(current.count) +
                            " gaussians (target " + std::to_string(target_count) + ")");
                    }

                    if (!report_progress(progress,
                                         pass_progress + 0.03f,
                                         pass_prefix + "merging " + std::to_string(scratch.pairs.size()) + " pairs"))
                        return std::unexpected("Cancelled");
                    current = merge_pairs(current, scratch.cache, scratch.pairs, scratch.used_rows, scratch.keep_idx);
                    ++pass;
                }

                (void)report_progress(progress, 1.0f, "Complete");
                return workset_from_rows(current, input);
            } catch (const std::exception& e) {
                return std::unexpected(e.what());
            }
        }

    } // namespace

    std::expected<std::unique_ptr<SplatData>, std::string> simplify_splats(
        const SplatData& input,
        const SplatSimplifyOptions& options,
        SplatSimplifyProgressCallback progress) {
        try {
            if (!input.means_raw().is_valid() || input.size() == 0)
                return std::unexpected("Splat simplify: input splat is empty");

            auto workset = make_workset_from_input(input, Device::CPU);
            auto result = simplify_workset(workset, options, std::move(progress));
            if (!result)
                return std::unexpected(result.error());
            return make_splat_from_workset(*result, Device::CUDA);
        } catch (const std::exception& e) {
            LOG_ERROR("simplify_splats failed: {}", e.what());
            return std::unexpected(e.what());
        }
    }

} // namespace lfs::core
