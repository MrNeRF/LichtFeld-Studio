/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_simplify.hpp"
#include "core/error.hpp"
#include "core/splat_simplify_history.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_fused.hpp"
#include "core/tensor_sh.hpp"
#include "core/tensor_simplify.hpp"

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
                    shN = Tensor::empty_like(input.shN_raw(), {keep_count, layout_rest, 3}, DataType::Float32);
                    sh_codec(input.shN_raw(), shN,
                             {.source_format = sh_storage_format(input.shN_raw(), input.shN_value_bounds()),
                              .destination_format = ShFormat::Canonical,
                              .source_rows = size_t(input.size()),
                              .destination_rows = keep_count,
                              .count = keep_count,
                              .source_rest = uint32_t(layout_rest),
                              .destination_rest = uint32_t(layout_rest)},
                             &keep_indices, input.shN_value_quantized() ? &input.shN_value_bounds() : nullptr);
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
            if (raw == -std::numeric_limits<float>::infinity())
                return 0.0f;
            return std::max(std::exp(clamp_scale_raw(raw)), kMinScale);
        }

        [[nodiscard]] float strict_mul(const float a, const float b) {
            volatile float out = a * b;
            return out;
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
                scaling_raw[i] = rows.scales[i] == 0.0f ? -std::numeric_limits<float>::infinity()
                                                        : std::log(std::max(rows.scales[i], kMinScale));

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
                                               std::array<float, 4>& rotation_raw,
                                               const bool allow_collapsed) {
            const auto eig = eigen_symmetric_3x3(sigma);
            // A rotated rank-deficient covariance can acquire tiny positive
            // eigenvalues from Float32 roundoff. Do not give its null space
            // thickness; retain the ordinary regularization for volume splats.
            const float zero_tolerance = allow_collapsed
                                             ? std::max({eig.values[0], eig.values[1], eig.values[2], 0.0f}) *
                                                   (8.0f * std::numeric_limits<float>::epsilon())
                                             : 0.0f;
            for (size_t axis = 0; axis < 3; ++axis) {
                const float variance = std::max(eig.values[axis], allow_collapsed ? 0.0f : kMinEval);
                scaling_raw[axis] = variance <= zero_tolerance ? -std::numeric_limits<float>::infinity()
                                                               : std::log(std::max(std::sqrt(variance), kMinScale));
            }
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

        [[nodiscard]] float voxel_size_for_bounds(const float min[3], const float max[3], int target_count) {
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

        [[nodiscard]] float compute_voxel_size(const NativeRows& rows, int target_count) {
            float min[3], max[3];
            compute_bounds(rows, min, max);
            return voxel_size_for_bounds(min, max, target_count);
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

        void split_group_by_extent(
            const NativeRows& rows,
            std::vector<int> indices,
            const int max_group_size,
            std::vector<std::vector<int>>& out_groups) {
            if (indices.empty())
                return;
            if (static_cast<int>(indices.size()) <= max_group_size) {
                std::sort(indices.begin(), indices.end());
                out_groups.push_back(std::move(indices));
                return;
            }

            float min[3], max[3];
            const size_t first3 = static_cast<size_t>(indices.front()) * 3;
            for (int axis = 0; axis < 3; ++axis)
                min[axis] = max[axis] = rows.means[first3 + static_cast<size_t>(axis)];
            for (const int idx : indices) {
                const size_t idx3 = static_cast<size_t>(idx) * 3;
                for (int axis = 0; axis < 3; ++axis) {
                    const float value = rows.means[idx3 + static_cast<size_t>(axis)];
                    min[axis] = std::min(min[axis], value);
                    max[axis] = std::max(max[axis], value);
                }
            }

            int split_axis = 0;
            float max_extent = max[0] - min[0];
            for (int axis = 1; axis < 3; ++axis) {
                const float extent = max[axis] - min[axis];
                if (extent > max_extent) {
                    max_extent = extent;
                    split_axis = axis;
                }
            }

            if (max_extent <= 1e-6f) {
                std::sort(indices.begin(), indices.end());
                for (size_t begin = 0; begin < indices.size(); begin += static_cast<size_t>(max_group_size)) {
                    const size_t end = std::min(indices.size(), begin + static_cast<size_t>(max_group_size));
                    out_groups.emplace_back(indices.begin() + static_cast<std::ptrdiff_t>(begin),
                                            indices.begin() + static_cast<std::ptrdiff_t>(end));
                }
                return;
            }

            std::sort(indices.begin(), indices.end(), [&](const int lhs, const int rhs) {
                const float lhs_value = rows.means[static_cast<size_t>(lhs) * 3 + static_cast<size_t>(split_axis)];
                const float rhs_value = rows.means[static_cast<size_t>(rhs) * 3 + static_cast<size_t>(split_axis)];
                if (lhs_value == rhs_value)
                    return lhs < rhs;
                return lhs_value < rhs_value;
            });

            const size_t mid = indices.size() / 2;
            std::vector<int> left(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(mid));
            std::vector<int> right(indices.begin() + static_cast<std::ptrdiff_t>(mid), indices.end());
            split_group_by_extent(rows, std::move(left), max_group_size, out_groups);
            split_group_by_extent(rows, std::move(right), max_group_size, out_groups);
        }

        [[nodiscard]] std::vector<std::vector<int>> cap_voxel_group_sizes(
            const NativeRows& rows,
            std::vector<std::vector<int>> groups,
            const int max_group_size) {
            if (max_group_size <= 1)
                return groups;

            std::vector<std::vector<int>> capped_groups;
            capped_groups.reserve(groups.size());
            for (auto& group : groups) {
                if (static_cast<int>(group.size()) > max_group_size) {
                    split_group_by_extent(rows, std::move(group), max_group_size, capped_groups);
                } else {
                    capped_groups.push_back(std::move(group));
                }
            }

            std::sort(capped_groups.begin(), capped_groups.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.front() < rhs.front();
            });
            return capped_groups;
        }

        [[nodiscard]] std::vector<std::vector<int>> limit_groups_to_target(
            const std::vector<std::vector<int>>& groups,
            const int current_count,
            const int target_count) {
            int remaining_savings = current_count - target_count;
            if (remaining_savings <= 0) {
                std::vector<std::vector<int>> singletons;
                singletons.reserve(static_cast<size_t>(current_count));
                for (const auto& group : groups) {
                    for (const int idx : group)
                        singletons.push_back({idx});
                }
                return singletons;
            }

            int possible_savings = 0;
            for (const auto& group : groups)
                possible_savings += std::max(0, static_cast<int>(group.size()) - 1);
            if (possible_savings <= remaining_savings)
                return groups;

            std::vector<size_t> order;
            order.reserve(groups.size());
            for (size_t i = 0; i < groups.size(); ++i)
                order.push_back(i);
            std::sort(order.begin(), order.end(), [&](const size_t lhs, const size_t rhs) {
                const int lhs_savings = std::max(0, static_cast<int>(groups[lhs].size()) - 1);
                const int rhs_savings = std::max(0, static_cast<int>(groups[rhs].size()) - 1);
                if (lhs_savings != rhs_savings)
                    return lhs_savings < rhs_savings;
                return groups[lhs].front() < groups[rhs].front();
            });

            std::vector<uint8_t> consumed(groups.size(), uint8_t{0});
            std::vector<std::vector<int>> adjusted_groups;
            adjusted_groups.reserve(static_cast<size_t>(current_count));
            for (const size_t group_index : order) {
                const auto& group = groups[group_index];
                const int savings = static_cast<int>(group.size()) - 1;
                if (savings <= 0 || remaining_savings <= 0)
                    continue;

                std::vector<int> sorted_group = group;
                std::sort(sorted_group.begin(), sorted_group.end());
                consumed[group_index] = 1;
                if (savings <= remaining_savings) {
                    adjusted_groups.push_back(std::move(sorted_group));
                    remaining_savings -= savings;
                    continue;
                }

                const auto merge_size = static_cast<size_t>(remaining_savings + 1);
                adjusted_groups.emplace_back(sorted_group.begin(),
                                             sorted_group.begin() + static_cast<std::ptrdiff_t>(merge_size));
                for (size_t i = merge_size; i < sorted_group.size(); ++i)
                    adjusted_groups.push_back({sorted_group[i]});
                remaining_savings = 0;
            }

            for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
                if (consumed[group_index])
                    continue;
                for (const int idx : groups[group_index])
                    adjusted_groups.push_back({idx});
            }

            std::sort(adjusted_groups.begin(), adjusted_groups.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.front() < rhs.front();
            });
            return adjusted_groups;
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
                bool has_collapsed = false;
                for (int idx : group) {
                    const size_t idx3 = static_cast<size_t>(idx) * 3;
                    has_collapsed |= input.scales[idx3] == 0 || input.scales[idx3 + 1] == 0 || input.scales[idx3 + 2] == 0;
                    const float sx = std::max(input.scales[idx3 + 0], kMinScale);
                    const float sy = std::max(input.scales[idx3 + 1], kMinScale);
                    const float sz = std::max(input.scales[idx3 + 2], kMinScale);
                    const float alpha = input.opacity[static_cast<size_t>(idx)];
                    const float volume = sx * sy * sz;
                    const float w = volume * alpha;
                    weights.push_back(w);
                    total_weight += w;
                }
                if (total_weight < 1e-30f && !has_collapsed) {
                    total_weight = 1e-30f;
                } else if (total_weight < 1e-30f) {
                    // Point/line splats have tiny regularized mass. Normalize
                    // relative weights instead of moving their centre to zero.
                    const float largest_weight = *std::max_element(weights.begin(), weights.end());
                    total_weight = 0.0f;
                    for (float& w : weights) {
                        w = largest_weight > 0 ? w / largest_weight : 1.0f;
                        total_weight += w;
                    }
                }
                for (float& w : weights)
                    w /= total_weight;

                // Compute weighted center
                const size_t o3 = static_cast<size_t>(out_row) * 3;
                const size_t origin_row = static_cast<size_t>(group.front()) * 3;
                const float ox = has_collapsed ? input.means[origin_row] : 0.0f;
                const float oy = has_collapsed ? input.means[origin_row + 1] : 0.0f;
                const float oz = has_collapsed ? input.means[origin_row + 2] : 0.0f;
                float cx = 0.0f, cy = 0.0f, cz = 0.0f;
                for (size_t g = 0; g < group.size(); ++g) {
                    const int idx = group[g];
                    const size_t idx3 = static_cast<size_t>(idx) * 3;
                    cx += weights[g] * (input.means[idx3 + 0] - ox);
                    cy += weights[g] * (input.means[idx3 + 1] - oy);
                    cz += weights[g] * (input.means[idx3 + 2] - oz);
                }
                // Coincident collapsed splats must stay exactly coincident:
                // rounding the centre would introduce a spurious covariance.
                cx += ox;
                cy += oy;
                cz += oz;
                out.means[o3 + 0] = cx;
                out.means[o3 + 1] = cy;
                out.means[o3 + 2] = cz;

                // Compute blended covariance
                std::array<float, 9> sigma{};
                for (size_t g = 0; g < group.size(); ++g) {
                    const int idx = group[g];
                    const size_t idx3 = static_cast<size_t>(idx) * 3;
                    const size_t idx4 = static_cast<size_t>(idx) * 4;

                    // The mass floor above is not geometric thickness.
                    const float sx = input.scales[idx3 + 0];
                    const float sy = input.scales[idx3 + 1];
                    const float sz = input.scales[idx3 + 2];

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
                if (!has_collapsed) {
                    sigma[0] += kEpsCov;
                    sigma[4] += kEpsCov;
                    sigma[8] += kEpsCov;
                }

                std::array<float, 3> scaling_raw{};
                std::array<float, 4> rotation{};
                decompose_sigma_to_raw_scale_quat(sigma, scaling_raw, rotation, has_collapsed);

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
                        const int max_group_size = std::max(
                            2,
                            static_cast<int>(std::ceil(static_cast<double>(current.count) /
                                                       static_cast<double>(pass_target_count))) +
                                1);
                        groups = cap_voxel_group_sizes(current, std::move(groups), max_group_size);
                        groups = limit_groups_to_target(groups, current.count, pass_target_count);

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

        // ----- Device passes: simplify_workset on one GPU backend, the same groups in the same order -----
        //
        // Every row carries a group label, any value shared by exactly its group. A stable sort by label
        // lays each group out contiguously with its members ascending; relabelled by their first member,
        // the groups also come out in the order simplify_workset gives them. Integer arithmetic runs in
        // fused kernels, since tensor scalar operations compute in Float32.
        namespace device {
            namespace f = fused;

            constexpr int32_t kSortBias = 0x00800000;

            Tensor int_scalar(const int value) {
                return Tensor::from_vector(std::vector<int>{value}, {1}, Device::CPU);
            }

            // Non-negative Int32 values as Float32 keys in the same order: biased into the normal floats.
            Tensor sort_keys(const Tensor& values) {
                static const f::Kernel kernel = [] {
                    f::Builder builder(1);
                    builder.output(builder.input(DataType::Int32, 1).load() + kSortBias, DataType::Int32);
                    return f::Kernel(builder);
                }();
                return kernel({values.numel()}, {values})[0].view_as(DataType::Float32);
            }

            // Positions that sort keys ascending, ties in their current order.
            Tensor stable_order(const Tensor& keys) {
                return keys.sort(0).second.to(DataType::Int32);
            }

            // Int32 0..count-1 on the backend of like.
            Tensor iota(const size_t count, const Tensor& like) {
                static const f::Kernel kernel = [] {
                    f::Builder builder(1);
                    const auto anchor = builder.input(DataType::Int32, 1).at({0});
                    builder.output(builder.iota(0) + anchor * 0, DataType::Int32);
                    return f::Kernel(builder);
                }();
                return kernel({count}, {Tensor::zeros({1}, like.device(), DataType::Int32)})[0];
            }

            // Runs of equal values in a sorted Int32 sequence: for each position its run's start and size
            // and its offset in the run.
            struct Runs {
                Tensor start;
                Tensor size;
                Tensor offset;
                Tensor starts; // Int32 [runs]
            };

            Runs runs(const Tensor& sorted) {
                static const f::Kernel heads = [] {
                    f::Builder builder(1);
                    const auto value = builder.input(DataType::Int32, 1);
                    const auto position = builder.iota(0);
                    builder.output(f::where(position == 0 || value.load() != value.gather({position - 1}, f::Bounds::Clamp), 1, 0),
                                   DataType::Int32);
                    return f::Kernel(builder);
                }();
                static const f::Kernel spans = [] {
                    f::Builder builder(1);
                    const auto run = builder.input(DataType::Int32, 1).load() - 1;
                    const auto starts = builder.input(DataType::Int32, 1);
                    const auto count = builder.input(DataType::Int32, 1).at({0});
                    const auto total = builder.input(DataType::Int32, 1).at({0});
                    const auto start = starts.gather({run});
                    const auto end = f::where(run + 1 < total, starts.gather({run + 1}, f::Bounds::Clamp), count);
                    builder.output(start, DataType::Int32);
                    builder.output(end - start, DataType::Int32);
                    builder.output(builder.iota(0) - start, DataType::Int32);
                    return f::Kernel(builder);
                }();
                const size_t count = sorted.numel();
                const auto head = heads({count}, {sorted})[0];
                Runs result;
                result.starts = head.ne(0).nonzero().reshape({-1}).to(DataType::Int32).contiguous();
                auto spanned = spans({count}, {head.cumsum(0), result.starts, int_scalar(static_cast<int>(count)),
                                               int_scalar(static_cast<int>(result.starts.numel()))});
                result.start = std::move(spanned[0]);
                result.size = std::move(spanned[1]);
                result.offset = std::move(spanned[2]);
                return result;
            }

            // Rows in label order, ascending within each label, with the runs of that order.
            struct Grouping {
                Tensor order; // Int32 [n] rows
                Runs runs;
            };

            Grouping group_by(const Tensor& labels) {
                Grouping result;
                result.order = stable_order(sort_keys(labels));
                result.runs = runs(labels.index_select(0, result.order).contiguous());
                return result;
            }

            // Each row labelled with the first member of its group, which is the group's smallest row.
            Tensor first_member_labels(const Tensor& labels) {
                const auto grouping = group_by(labels);
                auto result = Tensor::zeros({labels.numel()}, labels.device(), DataType::Int32);
                result.index_copy_(0, grouping.order, grouping.order.index_select(0, grouping.runs.start));
                return result;
            }

            // group_into_voxels: rows of one voxel share a label.
            Tensor voxel_labels(const Tensor& means, const float voxel_size, const float bounds_min[3]) {
                const size_t count = means.size(0);
                const auto low = Tensor::from_vector(std::vector<float>{bounds_min[0], bounds_min[1], bounds_min[2]},
                                                     {1, 3}, Device::CPU)
                                     .to(means.device());
                const auto cells = ((means - low) * (1.0f / voxel_size)).floor().contiguous();
                auto order = iota(count, means);
                for (const int axis : {2, 1, 0})
                    order = order.index_select(
                        0, stable_order(cells.slice(1, axis, axis + 1).squeeze(1).index_select(0, order)));
                // Equal cells are neighbours in this order; number the runs.
                static const f::Kernel heads = [] {
                    f::Builder builder(1);
                    const auto cells = builder.input(DataType::Float32, 2);
                    const auto position = builder.iota(0);
                    auto differs = position == 0;
                    for (int32_t axis = 0; axis < 3; ++axis)
                        differs = differs || cells.gather({position, builder.constant(axis)}) !=
                                                 cells.gather({position - 1, builder.constant(axis)}, f::Bounds::Clamp);
                    builder.output(f::where(differs, 1, 0), DataType::Int32);
                    return f::Kernel(builder);
                }();
                const auto voxel = heads({count}, {cells.index_select(0, order).contiguous()})[0].cumsum(0);
                auto labels = Tensor::zeros({count}, means.device(), DataType::Int32);
                labels.index_copy_(0, order, voxel);
                return labels;
            }

            // cap_voxel_group_sizes: a group above max_size splits at the median of its widest axis (ties by
            // row) until no part is above it; a group without extent is cut into runs of max_size rows.
            Tensor cap_groups(const Tensor& means, Tensor labels, const int max_size) {
                const size_t count = means.size(0);
                // Running min (channels 0-2) and max (3-5) over a group's positions, doubling the reach.
                static const f::Kernel extent_step = [] {
                    f::Builder builder(2);
                    const auto values = builder.input(DataType::Float32, 2);
                    const auto offset = builder.input(DataType::Int32, 1).load({0});
                    const auto reach = builder.input(DataType::Int32, 1).at({0});
                    const auto position = builder.iota(0);
                    const auto channel = builder.iota(1);
                    const auto own = values.load();
                    const auto other = values.gather({position - reach, channel}, f::Bounds::Clamp);
                    const auto combined = f::where(channel < 3, f::min(own, other), f::max(own, other));
                    builder.output(f::where(offset >= reach, combined, own), DataType::Float32);
                    return f::Kernel(builder);
                }();
                // Each position's coordinate on its group's widest axis (first of equal extents), and
                // whether the group has no extent.
                static const f::Kernel split_keys = [] {
                    f::Builder builder(1);
                    const auto extents = builder.input(DataType::Float32, 2);
                    const auto points = builder.input(DataType::Float32, 2);
                    const auto order = builder.input(DataType::Int32, 1).load();
                    const auto start = builder.input(DataType::Int32, 1).load();
                    const auto size = builder.input(DataType::Int32, 1).load();
                    const auto last = start + size - 1;
                    const auto extent = [&](const int32_t axis) {
                        return extents.gather({last, builder.constant(axis + 3)}) -
                               extents.gather({last, builder.constant(axis)});
                    };
                    const auto e0 = extent(0), e1 = extent(1), e2 = extent(2);
                    const auto axis1 = e1 > e0;
                    const auto best01 = f::where(axis1, e1, e0);
                    const auto axis2 = e2 > best01;
                    const auto axis = f::where(axis2, 2, f::where(axis1, 1, 0));
                    const auto flat = f::where(axis2, e2, best01) <= 1e-6f;
                    builder.output(f::where(flat, 0.0f, points.gather({order, axis})), DataType::Float32);
                    builder.output(f::where(flat, 1, 0), DataType::Int32);
                    return f::Kernel(builder);
                }();
                // A row's part of its split group, as an offset from the group start: the second half
                // starts at size / 2; without extent, runs of max_size.
                static const f::Kernel parts = [] {
                    f::Builder builder(1);
                    const auto offset = builder.input(DataType::Int32, 1).load();
                    const auto size = builder.input(DataType::Int32, 1).load();
                    const auto flat = builder.input(DataType::Int32, 1).load();
                    const auto max_size = builder.input(DataType::Int32, 1).at({0});
                    const auto half = size / 2;
                    builder.output(f::where(flat != 0, (offset / max_size) * max_size, f::where(offset >= half, half, 0)),
                                   DataType::Int32);
                    return f::Kernel(builder);
                }();
                static const f::Kernel add = [] {
                    f::Builder builder(1);
                    builder.output(builder.input(DataType::Int32, 1).load() + builder.input(DataType::Int32, 1).load(),
                                   DataType::Int32);
                    return f::Kernel(builder);
                }();
                for (;;) {
                    const auto grouping = group_by(labels);
                    const auto& group = grouping.runs;
                    const auto over = group.size.gt(static_cast<float>(max_size));
                    if (!over.any().item<bool>())
                        break;
                    auto extents = Tensor::cat({means, means}, 1).index_select(0, grouping.order).contiguous();
                    const int largest = group.size.max().item<int>();
                    for (int reach = 1; reach < largest; reach *= 2)
                        extents = extent_step({count, 6}, {extents, group.offset, int_scalar(reach)})[0];
                    const auto keyed = split_keys({count}, {extents, means, grouping.order, group.start, group.size});
                    // Positions of oversized groups in order of (group, key, row): positions already run by
                    // row within a group.
                    const auto positions = over.nonzero().reshape({-1}).to(DataType::Int32);
                    auto split = positions.index_select(0, stable_order(keyed[0].index_select(0, positions)));
                    split = split.index_select(0, stable_order(sort_keys(group.start.index_select(0, split))));
                    const auto split_runs = runs(group.start.index_select(0, split).contiguous());
                    const auto part = parts({split.numel()}, {split_runs.offset, group.size.index_select(0, split),
                                                              keyed[1].index_select(0, split), int_scalar(max_size)})[0];
                    // A group label is any position in it: its start, or its start plus the part offset.
                    auto next = Tensor::zeros({count}, means.device(), DataType::Int32);
                    next.index_copy_(0, grouping.order, group.start);
                    next.index_copy_(0, grouping.order.index_select(0, split),
                                     add({split.numel()}, {group.start.index_select(0, split), part})[0]);
                    labels = std::move(next);
                }
                return labels;
            }

            // limit_groups_to_target: when merging every group would undershoot the target, merge the
            // groups with the fewest savings first (ties by first member), part of the group that reaches
            // the target, and leave the rest as single rows.
            Tensor limit_groups(Tensor labels, const int current_count, const int target_count) {
                const int remaining = current_count - target_count;
                const auto grouping = group_by(labels);
                const auto& group = grouping.runs;
                const size_t groups = group.starts.numel();
                const auto sizes = group.size.index_select(0, group.starts).contiguous();
                static const f::Kernel minus_one = [] {
                    f::Builder builder(1);
                    builder.output(builder.input(DataType::Int32, 1).load() - 1, DataType::Int32);
                    return f::Kernel(builder);
                }();
                const auto savings = minus_one({groups}, {sizes})[0];
                if (remaining <= 0) {
                    // Every row on its own: no merges, and the caller widens the voxels.
                    return iota(labels.numel(), labels);
                }
                const auto possible = savings.cumsum(0).slice(0, groups - 1, groups).item<int>();
                if (possible <= remaining)
                    return labels;
                const auto firsts = grouping.order.index_select(0, group.starts).contiguous();
                auto order = stable_order(sort_keys(firsts));
                order = order.index_select(0, stable_order(sort_keys(savings.index_select(0, order))));
                const auto ordered_savings = savings.index_select(0, order).contiguous();
                const auto prefix = ordered_savings.cumsum(0);
                // Rows kept together in each group, in that order.
                static const f::Kernel kept = [] {
                    f::Builder builder(1);
                    const auto saving = builder.input(DataType::Int32, 1).load();
                    const auto prefix = builder.input(DataType::Int32, 1).load();
                    const auto remaining = builder.input(DataType::Int32, 1).at({0});
                    const auto before = prefix - saving;
                    const auto whole = prefix <= remaining;
                    const auto partial = before < remaining;
                    builder.output(f::where(saving <= 0, 1, f::where(whole, saving + 1, f::where(partial, remaining - before + 1, 1))),
                                   DataType::Int32);
                    return f::Kernel(builder);
                }();
                const auto ordered_kept = kept({groups}, {ordered_savings, prefix, int_scalar(remaining)})[0];
                auto group_kept = Tensor::zeros({groups}, labels.device(), DataType::Int32);
                group_kept.index_copy_(0, order, ordered_kept);
                // Positions past the kept rows become their own groups.
                static const f::Kernel relabel = [] {
                    f::Builder builder(1);
                    const auto start = builder.input(DataType::Int32, 1).load();
                    const auto offset = builder.input(DataType::Int32, 1).load();
                    const auto run = builder.input(DataType::Int32, 1).load() - 1;
                    const auto kept = builder.input(DataType::Int32, 1);
                    builder.output(f::where(offset < kept.gather({run}), start, start + offset), DataType::Int32);
                    return f::Kernel(builder);
                }();
                const auto runs_index = [&] {
                    // Run number of each position: the number of starts at or before it.
                    static const f::Kernel heads = [] {
                        f::Builder builder(1);
                        builder.output(f::where(builder.input(DataType::Int32, 1).load() == 0, 1, 0), DataType::Int32);
                        return f::Kernel(builder);
                    }();
                    return heads({labels.numel()}, {group.offset})[0].cumsum(0);
                }();
                auto next = Tensor::zeros({labels.numel()}, labels.device(), DataType::Int32);
                next.index_copy_(0, grouping.order,
                                 relabel({labels.numel()}, {group.start, group.offset, runs_index, group_kept})[0]);
                return next;
            }

            SimplifyRows activated_rows(const SplatSimplifyWorkset& workset) {
                const auto raw_scales = workset.scaling.contiguous();
                const auto scales = Tensor::where(raw_scales.eq(-std::numeric_limits<float>::infinity()),
                                                  Tensor::zeros_like(raw_scales),
                                                  raw_scales.clamp(-30.0f, 30.0f).exp().maximum(kMinScale));
                const auto rotation = workset.rotation.contiguous();
                const auto norm = (rotation * rotation).sum(1, true).sqrt().maximum(kMinQuatNorm);
                return {workset.means.contiguous(), scales.contiguous(), (rotation / norm).contiguous(),
                        workset.opacity.reshape({-1}).sigmoid().contiguous(), workset.appearance.contiguous()};
            }

            SplatSimplifyWorkset raw_workset(const SimplifyRows& rows, const SplatSimplifyWorkset& like) {
                SplatSimplifyWorkset out = like;
                out.means = rows.means;
                out.scaling = Tensor::where(rows.scales.eq(0.0f),
                                            Tensor::full_like(rows.scales, -std::numeric_limits<float>::infinity()),
                                            rows.scales.maximum(kMinScale).log())
                                  .contiguous();
                out.rotation = rows.rotation;
                const auto q = rows.opacity.clamp(kMinProb, 1.0f - kMinProb);
                out.opacity = (q / (q.neg() + 1.0f)).log().reshape({-1, 1}).contiguous();
                out.appearance = rows.appearance;
                return out;
            }

            SimplifyRows select_rows(const SimplifyRows& rows, const Tensor& keep) {
                return {rows.means.index_select(0, keep).contiguous(), rows.scales.index_select(0, keep).contiguous(),
                        rows.rotation.index_select(0, keep).contiguous(), rows.opacity.index_select(0, keep).contiguous(),
                        rows.appearance.index_select(0, keep).contiguous()};
            }

            struct Unsupported {};

            // simplify_workset on the rows' GPU backend. Throws Unsupported, before any work, when the
            // backend has no merge kernel.
            lfs::Error failure(const lfs::ErrorCode code, std::string message,
                               const SourceSite detection = LFS_SOURCE_SITE_CURRENT()) {
                return lfs::make_error({.code = code,
                                        .domain = lfs::ErrorDomain::Core,
                                        .user_message = std::move(message),
                                        .detection = detection});
            }

            lfs::Result<SplatSimplifyWorkset> simplify(const SplatSimplifyWorkset& input,
                                                       const SplatSimplifyOptions& options,
                                                       const SplatSimplifyProgressCallback& progress) {
                GpuBackendScope scope(gpu_backend_of(input.means).value());
                {
                    // Probe the merge kernel with an empty grouping.
                    const auto none = Tensor::zeros({1}, Device::GPU, DataType::Int32);
                    const auto empty = SimplifyRows{Tensor::zeros({0, 3}, Device::GPU), Tensor::zeros({0, 3}, Device::GPU),
                                                    Tensor::zeros({0, 4}, Device::GPU), Tensor::zeros({0}, Device::GPU),
                                                    Tensor::zeros({0, size_t(input.appearance.size(1))}, Device::GPU)};
                    if (!simplify_merge_groups(empty, none, Tensor::zeros({0}, Device::GPU, DataType::Int32)))
                        throw Unsupported{};
                }
                auto current = activated_rows(input);
                const int input_count = static_cast<int>(current.means.size(0));
                if (input_count == 0)
                    return failure(lfs::ErrorCode::InvalidArgument, "Splat simplify: input splat is empty");
                const int target_count = target_count_for(input_count, options.ratio);

                if (!report_progress(progress, 0.0f, "Pruning opacity"))
                    return failure(lfs::ErrorCode::Cancelled, "Cancelled");
                {
                    // prune_by_opacity: the threshold never exceeds the median opacity.
                    const auto sorted = current.opacity.sort(0).first;
                    const size_t mid = static_cast<size_t>(input_count) / 2;
                    const auto middle = sorted.slice(0, input_count % 2 ? mid : mid - 1, mid + 1).cpu().to_vector();
                    const float median = middle.size() == 1 ? middle[0] : 0.5f * (middle[0] + middle[1]);
                    const float threshold = std::min(options.opacity_prune_threshold, median);
                    current = select_rows(current, current.opacity.ge(threshold).nonzero().reshape({-1}).to(DataType::Int32));
                }
                if (current.means.size(0) == 0)
                    return failure(lfs::ErrorCode::InvalidArgument, "Splat simplify: input has no visible gaussians");

                int pass = 0;
                while (static_cast<int>(current.means.size(0)) > target_count) {
                    const int count = static_cast<int>(current.means.size(0));
                    const float pass_progress = progress_for_count(input_count, target_count, count);
                    const std::string pass_prefix = "Pass " + std::to_string(pass + 1) + ": ";
                    if (!report_progress(progress, pass_progress, pass_prefix + "building voxel grid"))
                        return failure(lfs::ErrorCode::Cancelled, "Cancelled");
                    const auto bounds = Tensor::cat({current.means.min(0), current.means.max(0)}, 0).cpu().to_vector();
                    const float bounds_min[3] = {bounds[0], bounds[1], bounds[2]};
                    const float bounds_max[3] = {bounds[3], bounds[4], bounds[5]};
                    const int pass_target_count = pass_target_count_for(count, target_count, options.lod_base);
                    float voxel_size = voxel_size_for_bounds(bounds_min, bounds_max, pass_target_count);
                    bool reduced = false;
                    for (int attempt = 0; attempt < 10 && !reduced; ++attempt) {
                        const int max_group_size = std::max(
                            2, static_cast<int>(std::ceil(static_cast<double>(count) / static_cast<double>(pass_target_count))) + 1);
                        auto labels = voxel_labels(current.means, voxel_size, bounds_min);
                        labels = cap_groups(current.means, std::move(labels), max_group_size);
                        labels = limit_groups(std::move(labels), count, pass_target_count);
                        const auto grouping = group_by(first_member_labels(labels));
                        const auto sizes = grouping.runs.size.index_select(0, grouping.runs.starts);
                        const int merge_count = static_cast<int>(sizes.gt(1.0f).to(DataType::Float32).sum().item<float>());
                        if (merge_count == 0) {
                            voxel_size *= 1.5f;
                            continue;
                        }
                        if (!report_progress(progress, pass_progress + 0.02f,
                                             pass_prefix + "merging " + std::to_string(merge_count) + " voxels"))
                            return failure(lfs::ErrorCode::Cancelled, "Cancelled");
                        const auto offsets = Tensor::cat({grouping.runs.starts, int_scalar(count).to(Device::GPU)}, 0).contiguous();
                        auto merged = simplify_merge_groups(current, offsets, grouping.order);
                        if (!merged)
                            throw Unsupported{};
                        current = std::move(*merged);
                        reduced = true;
                    }
                    if (!reduced)
                        return failure(lfs::ErrorCode::Internal, "Splat simplify stalled at " + std::to_string(count) +
                                                                     " gaussians (target " + std::to_string(target_count) + ")");
                    ++pass;
                }
                (void)report_progress(progress, 1.0f, "Complete");
                return raw_workset(current, input);
            }
        } // namespace device

    } // namespace

    std::expected<std::unique_ptr<SplatData>, std::string> simplify_splats(
        const SplatData& input,
        const SplatSimplifyOptions& options,
        SplatSimplifyProgressCallback progress) {
        try {
            if (!input.means_raw().is_valid() || input.size() == 0)
                return std::unexpected("Splat simplify: input splat is empty");

            if (input.means_raw().device() == Device::GPU) {
                try {
                    auto result = device::simplify(make_workset_from_input(input, Device::GPU), options, progress);
                    if (!result)
                        return std::unexpected(std::string(result.error().user_message()));
                    return make_splat_from_workset(*result, Device::GPU);
                } catch (const device::Unsupported&) {
                    // No merge kernel on this backend: the CPU passes below.
                }
            }
            auto workset = make_workset_from_input(input, Device::CPU);
            auto result = simplify_workset(workset, options, std::move(progress));
            if (!result)
                return std::unexpected(result.error());
            return make_splat_from_workset(*result, input.means_raw().device());
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
            out.splat = make_splat_from_workset(*result, Device::GPU);
            out.merge_tree = std::move(history.tree);
            return out;
        } catch (const std::exception& e) {
            LOG_ERROR("simplify_splats_with_history failed: {}", e.what());
            return std::unexpected(e.what());
        }
    }

} // namespace lfs::core
