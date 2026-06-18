/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_simplify.hpp"
#include "core/splat_simplify_history.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/logger.hpp"
#include "core/splat_data.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lfs::core {

    namespace {

        constexpr float kTwoPiPow1p5 = 0x1.f7fccep+3f;
        constexpr float kEpsCov = 1e-8f;
        constexpr float kMinScale = 1e-12f;
        constexpr float kMinQuatNorm = 1e-12f;
        constexpr float kMinProb = 1e-6f;
        constexpr float kMinEval = 1e-18f;
        constexpr int kJacobiIterations = 32;
        constexpr float kEllipsoidAreaP = 1.6075f;

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

        struct Eigen3x3 {
            std::array<float, 3> values{};
            std::array<float, 9> vectors{};
        };

        struct SimplifyHistoryState {
            SplatSimplifyMergeTree tree;
            std::vector<int32_t> current_node_ids;
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
            Tensor shN;
            if (input.shN_raw().is_valid() && input.shN_raw().numel() > 0 &&
                input.max_sh_coeffs_rest() > 0) {
                if (has_deleted) {
                    auto keep_indices = keep_mask.nonzero();
                    if (keep_indices.ndim() == 2)
                        keep_indices = keep_indices.squeeze(1);
                    const size_t keep_count = static_cast<size_t>(keep_indices.numel());
                    const size_t layout_rest = input.max_sh_coeffs_rest();
                    shN = Tensor::empty({keep_count, layout_rest, 3}, input.shN_raw().device());
                    if (keep_indices.dtype() == DataType::Int64) {
                        shN_swizzled_gather_to_linear_i64(
                            input.shN_raw().ptr<float>(),
                            keep_indices.ptr<int64_t>(),
                            shN.ptr<float>(),
                            keep_count,
                            static_cast<uint32_t>(layout_rest),
                            static_cast<uint32_t>(layout_rest));
                    } else {
                        auto keep_i32 = keep_indices.dtype() == DataType::Int32
                                            ? keep_indices
                                            : keep_indices.to(DataType::Int32);
                        shN_swizzled_gather_to_linear(
                            input.shN_raw().ptr<float>(),
                            keep_i32.ptr<int>(),
                            shN.ptr<float>(),
                            keep_count,
                            static_cast<uint32_t>(layout_rest),
                            static_cast<uint32_t>(layout_rest));
                    }
                    shN = shN.to(device).contiguous();
                } else {
                    shN = input.shN_canonical().to(device).contiguous();
                }
            }
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

        [[nodiscard]] SimplifyHistoryState make_history_state(const SplatSimplifyWorkset& input,
                                                              const SplatSimplifyOptions& options,
                                                              const int target_count) {
            SimplifyHistoryState history;
            history.tree.source_means = input.means.contiguous();
            history.tree.source_sh0 = unflatten_sh_like_ply(input.appearance.slice(1, 0, 3).contiguous(), 1);
            if (input.shn_coeffs > 0) {
                history.tree.source_shN = unflatten_sh_like_ply(
                    input.appearance.slice(1, 3, 3 + input.shn_coeffs * 3).contiguous(),
                    input.shn_coeffs);
            }
            history.tree.source_scaling = input.scaling.contiguous();
            history.tree.source_rotation = input.rotation.contiguous();
            history.tree.source_opacity = input.opacity.contiguous();
            history.tree.source_active_sh_degree = input.active_sh_degree;
            history.tree.source_max_sh_degree = input.max_sh_degree;
            history.tree.source_scene_scale = input.scene_scale;
            history.tree.target_count = target_count;
            history.tree.requested_ratio = options.ratio;
            history.tree.requested_lod_base = options.lod_base;
            history.tree.requested_opacity_prune_threshold = options.opacity_prune_threshold;

            history.current_node_ids.resize(static_cast<size_t>(input.size()));
            for (int i = 0; i < input.size(); ++i)
                history.current_node_ids[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            return history;
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

        void sigma_from_rot_var(const std::array<float, 9>& R,
                                const float vx,
                                const float vy,
                                const float vz,
                                std::array<float, 9>& out) {
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

        [[nodiscard]] float det3(const std::array<float, 9>& A) {
            return A[0] * (A[4] * A[8] - A[5] * A[7]) -
                   A[1] * (A[3] * A[8] - A[5] * A[6]) +
                   A[2] * (A[3] * A[7] - A[4] * A[6]);
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
                            dst.appearance.begin() + static_cast<ptrdiff_t>(dst_row * dst.app_dim));
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

        [[nodiscard]] NativeRows prune_by_opacity(const NativeRows& input,
                                                  const float requested_threshold,
                                                  std::vector<int>* keep_idx_out = nullptr) {
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
            if (keep_idx_out)
                *keep_idx_out = keep_idx;

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

        [[nodiscard]] float ellipsoid_area(const float sx, const float sy, const float sz) {
            const float t1 = std::pow(sx * sy, kEllipsoidAreaP);
            const float t2 = std::pow(sx * sz, kEllipsoidAreaP);
            const float t3 = std::pow(sy * sz, kEllipsoidAreaP);
            return 4.0f * static_cast<float>(M_PI) * std::pow((t1 + t2 + t3) / 3.0f, 1.0f / kEllipsoidAreaP);
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

        void compute_bounds(const NativeRows& rows, float out_min[3], float out_max[3]) {
            if (rows.count == 0) {
                for (int i = 0; i < 3; ++i) {
                    out_min[i] = 0.0f;
                    out_max[i] = 0.0f;
                }
                return;
            }
            for (int i = 0; i < 3; ++i) {
                out_min[i] = rows.means[static_cast<size_t>(i)];
                out_max[i] = rows.means[static_cast<size_t>(i)];
            }
            for (int r = 1; r < rows.count; ++r) {
                const size_t r3 = static_cast<size_t>(r) * 3;
                for (int i = 0; i < 3; ++i) {
                    out_min[i] = std::min(out_min[i], rows.means[r3 + i]);
                    out_max[i] = std::max(out_max[i], rows.means[r3 + i]);
                }
            }
        }

        [[nodiscard]] float compute_voxel_size(const NativeRows& rows, int target_count) {
            float min[3], max[3];
            compute_bounds(rows, min, max);
            float volume = 1.0f;
            int active_dims = 0;
            for (int axis = 0; axis < 3; ++axis) {
                const float extent = max[axis] - min[axis];
                if (extent > 1e-6f) {
                    volume *= extent;
                    ++active_dims;
                }
            }
            if (active_dims == 0)
                return 1.0f;
            return std::pow(volume / std::max(1, target_count), 1.0f / static_cast<float>(active_dims)) * 1.2f;
        }

        [[nodiscard]] int pass_target_count_for(const int current_count,
                                                const int final_target_count,
                                                const float lod_base) {
            const float base = std::max(lod_base, 1.01f);
            const int lod_target = static_cast<int>(std::ceil(static_cast<float>(current_count) / base));
            return std::clamp(std::max(final_target_count, lod_target), 1, std::max(1, current_count - 1));
        }

        struct VoxelKey {
            int64_t x, y, z;
            bool operator==(const VoxelKey& other) const {
                return x == other.x && y == other.y && z == other.z;
            }
        };

        struct VoxelKeyHash {
            std::size_t operator()(const VoxelKey& k) const noexcept {
                // Simple hash combining
                std::size_t h = static_cast<std::size_t>(k.x);
                h = h * 31 + static_cast<std::size_t>(k.y);
                h = h * 31 + static_cast<std::size_t>(k.z);
                return h;
            }
        };

        [[nodiscard]] std::vector<std::vector<int>> group_into_voxels(
            const NativeRows& rows,
            float voxel_size,
            const float bounds_min[3]) {
            std::vector<std::vector<int>> groups;
            if (voxel_size <= 0.0f || rows.count == 0)
                return groups;

            std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> cells;
            cells.reserve(static_cast<size_t>(rows.count));

            const float inv_size = 1.0f / voxel_size;
            for (int i = 0; i < rows.count; ++i) {
                const size_t i3 = static_cast<size_t>(i) * 3;
                VoxelKey key;
                key.x = static_cast<int64_t>(std::floor((rows.means[i3 + 0] - bounds_min[0]) * inv_size));
                key.y = static_cast<int64_t>(std::floor((rows.means[i3 + 1] - bounds_min[1]) * inv_size));
                key.z = static_cast<int64_t>(std::floor((rows.means[i3 + 2] - bounds_min[2]) * inv_size));
                cells[key].push_back(i);
            }

            groups.reserve(cells.size());
            for (auto& [key, indices] : cells) {
                std::sort(indices.begin(), indices.end());
                groups.push_back(std::move(indices));
            }
            std::sort(groups.begin(), groups.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.front() < rhs.front();
            });
            return groups;
        }

        [[nodiscard]] NativeRows merge_voxel_groups(
            const NativeRows& input,
            const std::vector<std::vector<int>>& groups,
            std::vector<int>& keep_idx,
            SimplifyHistoryState* history,
            int pass_index) {
            keep_idx.clear();
            keep_idx.reserve(static_cast<size_t>(input.count));

            // Count output rows
            int out_count = 0;
            for (const auto& group : groups) {
                if (group.size() == 1) {
                    keep_idx.push_back(group[0]);
                    ++out_count;
                } else if (group.size() > 1) {
                    ++out_count;
                }
            }

            NativeRows out;
            out.count = out_count;
            out.app_dim = input.app_dim;
            out.means.resize(static_cast<size_t>(out.count) * 3);
            out.scales.resize(static_cast<size_t>(out.count) * 3);
            out.rotation.resize(static_cast<size_t>(out.count) * 4);
            out.opacity.resize(static_cast<size_t>(out.count));
            out.appearance.resize(static_cast<size_t>(out.count) * static_cast<size_t>(out.app_dim));
            // Rebuild current_node_ids from the output of this pass
            std::vector<int32_t> next_node_ids;
            if (history)
                next_node_ids.reserve(static_cast<size_t>(out_count));

            int out_row = 0;
            for (const auto& group : groups) {
                if (group.empty())
                    continue;

                if (group.size() == 1) {
                    copy_row(input, group[0], out, out_row);
                    if (history) {
                        next_node_ids.push_back(history->current_node_ids[static_cast<size_t>(group[0])]);
                    }
                    ++out_row;
                    continue;
                }

                // Compute weights and total weight (volume-based, not area-based)
                std::vector<float> weights;
                weights.reserve(group.size());
                float total_weight = 0.0f;
                for (int idx : group) {
                    const size_t idx3 = static_cast<size_t>(idx) * 3;
                    const float sx = std::max(input.scales[idx3 + 0], kMinScale);
                    const float sy = std::max(input.scales[idx3 + 1], kMinScale);
                    const float sz = std::max(input.scales[idx3 + 2], kMinScale);
                    const float alpha = input.opacity[static_cast<size_t>(idx)];
                    const float volume = sx * sy * sz;
                    const float w = volume * alpha;
                    weights.push_back(w);
                    total_weight += w;
                }
                if (total_weight < 1e-30f)
                    total_weight = 1e-30f;
                for (float& w : weights)
                    w /= total_weight;

                // Compute weighted center
                const size_t o3 = static_cast<size_t>(out_row) * 3;
                float cx = 0.0f, cy = 0.0f, cz = 0.0f;
                for (size_t g = 0; g < group.size(); ++g) {
                    const int idx = group[g];
                    const size_t idx3 = static_cast<size_t>(idx) * 3;
                    cx += weights[g] * input.means[idx3 + 0];
                    cy += weights[g] * input.means[idx3 + 1];
                    cz += weights[g] * input.means[idx3 + 2];
                }
                out.means[o3 + 0] = cx;
                out.means[o3 + 1] = cy;
                out.means[o3 + 2] = cz;

                // Compute blended covariance
                std::array<float, 9> sigma{};
                for (size_t g = 0; g < group.size(); ++g) {
                    const int idx = group[g];
                    const size_t idx3 = static_cast<size_t>(idx) * 3;
                    const size_t idx4 = static_cast<size_t>(idx) * 4;

                    const float sx = std::max(input.scales[idx3 + 0], kMinScale);
                    const float sy = std::max(input.scales[idx3 + 1], kMinScale);
                    const float sz = std::max(input.scales[idx3 + 2], kMinScale);

                    float qw = input.rotation[idx4 + 0];
                    float qx = input.rotation[idx4 + 1];
                    float qy = input.rotation[idx4 + 2];
                    float qz = input.rotation[idx4 + 3];
                    std::array<float, 9> R{};
                    quat_to_rotmat(qw, qx, qy, qz, R);

                    std::array<float, 9> sig{};
                    sigma_from_rot_var(R, sx * sx, sy * sy, sz * sz, sig);

                    // Add delta outer product
                    const float dx = input.means[idx3 + 0] - cx;
                    const float dy = input.means[idx3 + 1] - cy;
                    const float dz = input.means[idx3 + 2] - cz;
                    sig[0] += dx * dx;
                    sig[1] += dx * dy;
                    sig[2] += dx * dz;
                    sig[3] += dy * dx;
                    sig[4] += dy * dy;
                    sig[5] += dy * dz;
                    sig[6] += dz * dx;
                    sig[7] += dz * dy;
                    sig[8] += dz * dz;

                    // Accumulate weighted
                    for (int a = 0; a < 9; ++a)
                        sigma[static_cast<size_t>(a)] += weights[g] * sig[static_cast<size_t>(a)];
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
                const size_t o4 = static_cast<size_t>(out_row) * 4;
                out.rotation[o4 + 0] = rotation[0];
                out.rotation[o4 + 1] = rotation[1];
                out.rotation[o4 + 2] = rotation[2];
                out.rotation[o4 + 3] = rotation[3];

                // Opacity: union of coverage for independent Gaussians
                float merged_opacity = 1.0f;
                for (int idx : group) {
                    merged_opacity *= (1.0f - input.opacity[static_cast<size_t>(idx)]);
                }
                merged_opacity = 1.0f - merged_opacity;
                out.opacity[static_cast<size_t>(out_row)] = std::clamp(merged_opacity, 0.0f, 1.0f);

                // Appearance weighted average
                const size_t ao = static_cast<size_t>(out_row) * static_cast<size_t>(input.app_dim);
                for (int k = 0; k < input.app_dim; ++k)
                    out.appearance[ao + static_cast<size_t>(k)] = 0.0f;
                for (size_t g = 0; g < group.size(); ++g) {
                    const int idx = group[g];
                    const size_t ai = static_cast<size_t>(idx) * static_cast<size_t>(input.app_dim);
                    for (int k = 0; k < input.app_dim; ++k)
                        out.appearance[ao + static_cast<size_t>(k)] += weights[g] * input.appearance[ai + static_cast<size_t>(k)];
                }

                // History tracking: decompose N-way merge into sequential binary merges
                if (history) {
                    int current_node = history->current_node_ids[static_cast<size_t>(group[0])];
                    for (size_t g = 1; g < group.size(); ++g) {
                        const int next_node = history->current_node_ids[static_cast<size_t>(group[g])];
                        history->tree.merge_left.push_back(current_node);
                        history->tree.merge_right.push_back(next_node);
                        history->tree.merge_pass.push_back(pass_index);
                        const int merged_node = static_cast<int>(history->tree.leaf_count() + history->tree.merge_count() - 1);
                        current_node = merged_node;
                    }
                    next_node_ids.push_back(current_node);
                }

                ++out_row;
            }

            if (history)
                history->current_node_ids = std::move(next_node_ids);

            return out;
        }

        [[nodiscard]] int target_count_for(const int input_count, const double ratio) {
            const double clamped_ratio = std::clamp(ratio, 0.0, 1.0);
            return std::clamp(
                static_cast<int>(std::ceil(static_cast<double>(input_count) * clamped_ratio)),
                1,
                std::max(1, input_count));
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
            SplatSimplifyProgressCallback progress,
            SimplifyHistoryState* history = nullptr) {
            try {
                NativeRows current = rows_from_workset(input);
                if (current.count == 0)
                    return std::unexpected("Splat simplify: input splat is empty");

                const int input_count = current.count;
                const int target_count = target_count_for(input_count, options.ratio);
                std::vector<int> keep_idx;
                if (history)
                    *history = make_history_state(input, options, target_count);

                if (!report_progress(progress, 0.0f, "Pruning opacity"))
                    return std::unexpected("Cancelled");
                current = prune_by_opacity(
                    current,
                    options.opacity_prune_threshold,
                    history ? &keep_idx : nullptr);

                if (current.count == 0)
                    return std::unexpected("Splat simplify: input has no visible gaussians");
                if (history) {
                    std::vector<int32_t> kept_ids;
                    std::vector<int32_t> pruned_ids;
                    kept_ids.reserve(static_cast<size_t>(current.count));
                    pruned_ids.reserve(history->current_node_ids.size());

                    std::vector<uint8_t> kept_mask(history->current_node_ids.size(), uint8_t{0});
                    for (const int idx : keep_idx) {
                        if (idx >= 0 && static_cast<size_t>(idx) < kept_mask.size())
                            kept_mask[static_cast<size_t>(idx)] = 1;
                    }
                    for (size_t i = 0; i < history->current_node_ids.size(); ++i) {
                        const int32_t node_id = history->current_node_ids[i];
                        if (i < kept_mask.size() && kept_mask[i]) {
                            kept_ids.push_back(node_id);
                        } else if (node_id >= 0) {
                            pruned_ids.push_back(node_id);
                        }
                    }

                    history->tree.post_prune_count = current.count;
                    history->tree.pruned_leaf_ids = std::move(pruned_ids);
                    history->current_node_ids = std::move(kept_ids);
                }
                if (current.count <= target_count) {
                    if (history)
                        history->tree.final_roots = history->current_node_ids;
                    (void)report_progress(progress, 1.0f, "Complete");
                    return workset_from_rows(current, input);
                }

                int pass = 0;
                while (current.count > target_count) {
                    const float pass_progress = progress_for_count(input_count, target_count, current.count);
                    const std::string pass_prefix = "Pass " + std::to_string(pass + 1) + ": ";

                    if (!report_progress(progress, pass_progress, pass_prefix + "building voxel grid"))
                        return std::unexpected("Cancelled");

                    float bounds_min[3], bounds_max[3];
                    compute_bounds(current, bounds_min, bounds_max);
                    const int pass_target_count = pass_target_count_for(
                        current.count,
                        target_count,
                        options.lod_base);
                    float voxel_size = compute_voxel_size(current, pass_target_count);

                    // If we're not reducing enough, increase voxel size
                    bool reduced = false;
                    for (int attempt = 0; attempt < 10 && !reduced; ++attempt) {
                        auto groups = group_into_voxels(current, voxel_size, bounds_min);

                        int merge_count = 0;
                        for (const auto& g : groups)
                            if (g.size() > 1)
                                ++merge_count;

                        if (merge_count == 0) {
                            // No merges possible with this voxel size, increase it
                            voxel_size *= 1.5f;
                            continue;
                        }

                        if (!report_progress(progress,
                                             pass_progress + 0.02f,
                                             pass_prefix + "merging " + std::to_string(merge_count) + " voxels"))
                            return std::unexpected("Cancelled");

                        current = merge_voxel_groups(current, groups, keep_idx, history, pass);
                        reduced = true;
                    }

                    if (!reduced) {
                        return std::unexpected(
                            "Splat simplify stalled at " + std::to_string(current.count) +
                            " gaussians (target " + std::to_string(target_count) + ")");
                    }

                    ++pass;
                }

                if (history)
                    history->tree.final_roots = history->current_node_ids;
                (void)report_progress(progress, 1.0f, "Complete");
                return workset_from_rows(current, input);
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Splat simplify failed: ") + e.what());
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

    std::expected<SplatSimplifyResult, std::string> simplify_splats_with_history(
        const SplatData& input,
        const SplatSimplifyOptions& options,
        SplatSimplifyProgressCallback progress) {
        try {
            if (!input.means_raw().is_valid() || input.size() == 0)
                return std::unexpected("Splat simplify: input splat is empty");

            auto workset = make_workset_from_input(input, Device::CPU);
            SimplifyHistoryState history;
            auto result = simplify_workset(workset, options, std::move(progress), &history);
            if (!result)
                return std::unexpected(result.error());

            SplatSimplifyResult out;
            out.splat = make_splat_from_workset(*result, Device::CUDA);
            out.merge_tree = std::move(history.tree);
            return out;
        } catch (const std::exception& e) {
            LOG_ERROR("simplify_splats_with_history failed: {}", e.what());
            return std::unexpected(e.what());
        }
    }

} // namespace lfs::core
