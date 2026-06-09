/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/bhatt_lod.hpp"
#include "core/splat_data.hpp"
#include "core/logger.hpp"
#include "core/splat_data_transform.hpp"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lfs::core {

namespace {

constexpr float kMinScale = 1e-12f;

// The 26 neighboring cells around a center cell (excluding (0,0,0))
constexpr std::array<std::array<int8_t, 3>, 26> kNeighborOffsets = {{
    {{-1, -1, -1}}, {{-1, -1, 0}}, {{-1, -1, 1}},
    {{-1,  0, -1}}, {{-1,  0, 0}}, {{-1,  0, 1}},
    {{-1,  1, -1}}, {{-1,  1, 0}}, {{-1,  1, 1}},
    {{ 0, -1, -1}}, {{ 0, -1, 0}}, {{ 0, -1, 1}},
    {{ 0,  0, -1}},                  {{ 0,  0, 1}},
    {{ 0,  1, -1}}, {{ 0,  1, 0}}, {{ 0,  1, 1}},
    {{ 1, -1, -1}}, {{ 1, -1, 0}}, {{ 1, -1, 1}},
    {{ 1,  0, -1}}, {{ 1,  0, 0}}, {{ 1,  0, 1}},
    {{ 1,  1, -1}}, {{ 1,  1, 0}}, {{ 1,  1, 1}}
}};
constexpr float kMinQuatNorm = 1e-12f;
constexpr float kEllipsoidAreaP = 1.6075f;
constexpr int kJacobiIterations = 10;
constexpr float kMinEval = 1e-18f;
constexpr float kEpsCov = 1e-8f;
constexpr float SH_C0 = 0.28209479177387814f;

[[nodiscard]] float sigmoid(const float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

[[nodiscard]] float clamp_scale_raw(const float raw) {
    return std::clamp(raw, -30.0f, 30.0f);
}

[[nodiscard]] float activated_scale(const float raw) {
    return std::max(std::exp(clamp_scale_raw(raw)), kMinScale);
}

[[nodiscard]] float ellipsoid_area(const float sx, const float sy, const float sz) {
    const float t1 = std::pow(sx * sy, kEllipsoidAreaP);
    const float t2 = std::pow(sx * sz, kEllipsoidAreaP);
    const float t3 = std::pow(sy * sz, kEllipsoidAreaP);
    return 4.0f * static_cast<float>(M_PI) * std::pow((t1 + t2 + t3) / 3.0f, 1.0f / kEllipsoidAreaP);
}

[[nodiscard]] float lod_opacity(const float opacity) {
    if (opacity > 1.0f) {
        constexpr float kE = 2.718281828459045f;
        return std::sqrt(1.0f + kE * std::log(opacity));
    }
    return 1.0f;
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
            scaled[idx] = R[idx] * variance[static_cast<size_t>(col)];
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            out[static_cast<size_t>(row * 3 + col)] =
                scaled[static_cast<size_t>(row * 3 + 0)] * R[static_cast<size_t>(col * 3 + 0)] +
                scaled[static_cast<size_t>(row * 3 + 1)] * R[static_cast<size_t>(col * 3 + 1)] +
                scaled[static_cast<size_t>(row * 3 + 2)] * R[static_cast<size_t>(col * 3 + 2)];
        }
    }
}

[[nodiscard]] float det3(const std::array<float, 9>& A) {
    return A[0] * (A[4] * A[8] - A[5] * A[7]) -
           A[1] * (A[3] * A[8] - A[5] * A[6]) +
           A[2] * (A[3] * A[7] - A[4] * A[6]);
}

struct Eigen3x3 {
    std::array<float, 3> values{};
    std::array<float, 9> vectors{};
};

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
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
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

// Compute symmetric 3x3 covariance from scale + quaternion and store 6 unique elements + det
inline void compute_covariance_from_scale_quat(
    float sx, float sy, float sz,
    float qw, float qx, float qy, float qz,
    float& out_xx, float& out_xy, float& out_xz,
    float& out_yy, float& out_yz, float& out_zz,
    float& out_det) {

    std::array<float, 9> R;
    quat_to_rotmat(qw, qx, qy, qz, R);
    const float sx2 = sx * sx;
    const float sy2 = sy * sy;
    const float sz2 = sz * sz;
    std::array<float, 9> cov;
    sigma_from_rot_var(R, sx2, sy2, sz2, cov);

    out_xx = cov[0];
    out_xy = cov[1];
    out_xz = cov[2];
    out_yy = cov[4];
    out_yz = cov[5];
    out_zz = cov[8];
    out_det = det3(cov);
}

[[nodiscard]] BhattLodWorkset make_workset_from_splatdata(const SplatData& input) {
    const auto t0 = std::chrono::high_resolution_clock::now();

    const auto means_cpu = input.means_raw().cpu().contiguous();
    const auto scaling_cpu = input.scaling_raw().cpu().contiguous();
    const auto rotation_cpu = input.rotation_raw().cpu().contiguous();
    const auto opacity_cpu = input.opacity_raw().cpu().contiguous();
    const auto sh0_cpu = input.sh0_raw().cpu().contiguous();

    const float* means_ptr = means_cpu.ptr<float>();
    const float* scaling_ptr = scaling_cpu.ptr<float>();
    const float* rotation_ptr = rotation_cpu.ptr<float>();
    const float* opacity_ptr = opacity_cpu.ptr<float>();
    const float* sh0_ptr = sh0_cpu.ptr<float>();

    const size_t total_count = input.size();
    const bool has_deleted = input.has_deleted_mask() && input.deleted().count_nonzero() > 0;

    const auto deleted_cpu = has_deleted ? input.deleted().cpu().contiguous() : Tensor{};
    const uint8_t* deleted_ptr = has_deleted ? deleted_cpu.ptr<uint8_t>() : nullptr;

    LOG_DEBUG("make_workset_from_splatdata: total_count={}, has_deleted={}", total_count, has_deleted);

    // Step 1: Count visible splats per chunk in parallel
    const size_t grain_size = std::max(size_t(4096), total_count / 16);
    const size_t num_chunks = (total_count + grain_size - 1) / grain_size;
    std::vector<size_t> chunk_counts(num_chunks);

    const auto t_count_start = std::chrono::high_resolution_clock::now();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_chunks),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
                const size_t start = chunk_idx * grain_size;
                const size_t end = std::min(start + grain_size, total_count);
                size_t count = 0;
                for (size_t i = start; i < end; ++i) {
                    if (!has_deleted || (deleted_ptr && deleted_ptr[i] == 0)) {
                        ++count;
                    }
                }
                chunk_counts[chunk_idx] = count;
            }
        });
    const auto t_count_end = std::chrono::high_resolution_clock::now();
    const auto count_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_count_end - t_count_start).count();

    // Step 2: Prefix sum to compute chunk offsets
    std::vector<size_t> chunk_offsets(num_chunks);
    size_t visible_count = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        chunk_offsets[i] = visible_count;
        visible_count += chunk_counts[i];
    }

    LOG_DEBUG("make_workset_from_splatdata: counting done in {} ms, visible_count={}, grain_size={}, num_chunks={}",
             count_ms, visible_count, grain_size, num_chunks);

    BhattLodWorkset workset;
    workset.reserve(visible_count, input.get_max_sh_degree());

    Tensor shN_canonical;
    const float* shN_ptr = nullptr;
    int shN_rest_coeffs = 0;
    if (input.shN_raw().is_valid() && input.shN_raw().numel() > 0 && input.max_sh_coeffs_rest() > 0) {
        shN_canonical = input.shN_canonical_cpu();
        shN_ptr = shN_canonical.ptr<float>();
        shN_rest_coeffs = static_cast<int>(input.max_sh_coeffs_rest());
    }

    // Step 3: Parallel extraction into pre-allocated arrays
    const auto t_extract_start = std::chrono::high_resolution_clock::now();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_chunks),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
                const size_t start = chunk_idx * grain_size;
                const size_t end = std::min(start + grain_size, total_count);
                size_t out_idx = chunk_offsets[chunk_idx];

                for (size_t i = start; i < end; ++i) {
                    if (has_deleted && deleted_ptr && deleted_ptr[i] != 0) {
                        continue;
                    }

                    const size_t i3 = i * 3;
                    const size_t i4 = i * 4;

                    const float cx = means_ptr[i3 + 0];
                    const float cy = means_ptr[i3 + 1];
                    const float cz = means_ptr[i3 + 2];

                    const float sx = activated_scale(scaling_ptr[i3 + 0]);
                    const float sy = activated_scale(scaling_ptr[i3 + 1]);
                    const float sz = activated_scale(scaling_ptr[i3 + 2]);

                    float qw = rotation_ptr[i4 + 0];
                    float qx = rotation_ptr[i4 + 1];
                    float qy = rotation_ptr[i4 + 2];
                    float qz = rotation_ptr[i4 + 3];
                    const float inv_q = 1.0f / std::max(std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz), kMinQuatNorm);
                    qw *= inv_q;
                    qx *= inv_q;
                    qy *= inv_q;
                    qz *= inv_q;

                    const float op = sigmoid(opacity_ptr[i]);

                    const float r_ = 0.5f + SH_C0 * sh0_ptr[i3 + 0];
                    const float g_ = 0.5f + SH_C0 * sh0_ptr[i3 + 1];
                    const float b_ = 0.5f + SH_C0 * sh0_ptr[i3 + 2];

                    workset.center_x[out_idx] = cx;
                    workset.center_y[out_idx] = cy;
                    workset.center_z[out_idx] = cz;
                    workset.scale_x[out_idx] = sx;
                    workset.scale_y[out_idx] = sy;
                    workset.scale_z[out_idx] = sz;
                    workset.qw[out_idx] = qw;
                    workset.qx[out_idx] = qx;
                    workset.qy[out_idx] = qy;
                    workset.qz[out_idx] = qz;
                    workset.opacity[out_idx] = op;
                    workset.r[out_idx] = r_;
                    workset.g[out_idx] = g_;
                    workset.b[out_idx] = b_;

                    compute_covariance_from_scale_quat(
                        sx, sy, sz, qw, qx, qy, qz,
                        workset.cov_xx[out_idx], workset.cov_xy[out_idx], workset.cov_xz[out_idx],
                        workset.cov_yy[out_idx], workset.cov_yz[out_idx], workset.cov_zz[out_idx],
                        workset.cov_det[out_idx]);

                    if (workset.max_sh_degree >= 1) {
                        const float* sh1_src = nullptr;
                        const float* sh2_src = nullptr;
                        const float* sh3_src = nullptr;

                        if (shN_ptr && shN_rest_coeffs > 0) {
                            const float* rest_ptr = shN_ptr + i * shN_rest_coeffs * 3;
                            if (input.get_max_sh_degree() >= 1 && shN_rest_coeffs >= 3) {
                                sh1_src = rest_ptr;
                            }
                            if (input.get_max_sh_degree() >= 2 && shN_rest_coeffs >= 8) {
                                sh2_src = rest_ptr + 3 * 3;
                            }
                            if (input.get_max_sh_degree() >= 3 && shN_rest_coeffs >= 15) {
                                sh3_src = rest_ptr + 8 * 3;
                            }
                        }

                        if (sh1_src) {
                            std::copy_n(sh1_src, 9, workset.sh1.data() + out_idx * 9);
                        }
                        if (sh2_src) {
                            std::copy_n(sh2_src, 15, workset.sh2.data() + out_idx * 15);
                        }
                        if (sh3_src) {
                            std::copy_n(sh3_src, 21, workset.sh3.data() + out_idx * 21);
                        }
                    }

                    workset.child_a[out_idx] = -1;
                    workset.child_b[out_idx] = -1;
                    workset.is_active[out_idx] = 1;

                    ++out_idx;
                }
            }
        });
    const auto t_extract_end = std::chrono::high_resolution_clock::now();
    const auto extract_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_extract_end - t_extract_start).count();

    workset.current_count = visible_count;

    const auto t_total = std::chrono::high_resolution_clock::now();
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_total - t0).count();
    LOG_DEBUG("make_workset_from_splatdata: extraction done in {} ms, total time {} ms", extract_ms, total_ms);

    return workset;
}

constexpr float MERGE_BASE = 2.00f;

static uint64_t hash_cell(int64_t x, int64_t y, int64_t z) {
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    h ^= static_cast<uint64_t>(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= static_cast<uint64_t>(y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= static_cast<uint64_t>(z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

struct ActiveEntry {
    float neg_feature_size = 0.0f;
    uint32_t index = 0;

    bool operator<(const ActiveEntry& other) const {
        if (neg_feature_size != other.neg_feature_size) {
            return neg_feature_size < other.neg_feature_size;
        }
        return index < other.index;
    }
};

struct CellKey {
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;

    bool operator==(const CellKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct CellKeyHash {
    size_t operator()(const CellKey& key) const {
        return static_cast<size_t>(hash_cell(key.x, key.y, key.z));
    }
};

using CellMap = std::unordered_map<CellKey, std::vector<uint32_t>, CellKeyHash>;

[[nodiscard]] CellKey cell_for(const BhattLodWorkset& ws, const size_t idx, const float step) {
    CellKey key;
    ws.grid_cell(idx, step, key.x, key.y, key.z);
    return key;
}

void erase_from_cell(CellMap& cells, const CellKey& key, const uint32_t idx) {
    auto it = cells.find(key);
    if (it == cells.end()) {
        return;
    }
    auto& entries = it->second;
    entries.erase(std::remove(entries.begin(), entries.end(), idx), entries.end());
    if (entries.empty()) {
        cells.erase(it);
    }
}

} // namespace

// Workset methods
void BhattLodWorkset::reserve(size_t initial_count, int max_sh_degree) {
    this->initial_count = initial_count;
    this->max_sh_degree = max_sh_degree;
    this->capacity = initial_count * 3 / 2;
    this->current_count = 0;

    center_x.resize(capacity);
    center_y.resize(capacity);
    center_z.resize(capacity);
    scale_x.resize(capacity);
    scale_y.resize(capacity);
    scale_z.resize(capacity);
    qw.resize(capacity);
    qx.resize(capacity);
    qy.resize(capacity);
    qz.resize(capacity);
    opacity.resize(capacity);
    r.resize(capacity);
    g.resize(capacity);
    b.resize(capacity);
    feature_size.resize(capacity);
    area.resize(capacity);
    cov_xx.resize(capacity);
    cov_xy.resize(capacity);
    cov_xz.resize(capacity);
    cov_yy.resize(capacity);
    cov_yz.resize(capacity);
    cov_zz.resize(capacity);
    cov_det.resize(capacity);
    child_a.resize(capacity);
    child_b.resize(capacity);
    is_active.resize(capacity);

    if (max_sh_degree >= 1) {
        sh1.resize(capacity * 9);
    }
    if (max_sh_degree >= 2) {
        sh2.resize(capacity * 15);
    }
    if (max_sh_degree >= 3) {
        sh3.resize(capacity * 21);
    }

    std::fill(child_a.begin(), child_a.end(), -1);
    std::fill(child_b.begin(), child_b.end(), -1);
    std::fill(is_active.begin(), is_active.end(), 0);
}

size_t BhattLodWorkset::add_node(
    float cx, float cy, float cz,
    float sx, float sy, float sz,
    float qw_, float qx_, float qy_, float qz_,
    float op,
    float r_, float g_, float b_,
    const float* sh1_ptr, const float* sh2_ptr, const float* sh3_ptr) {

    const size_t idx = current_count;
    if (idx >= capacity) {
        // Grow arrays by 50% if capacity is exceeded
        const size_t new_capacity = capacity * 3 / 2 + 1;
        center_x.resize(new_capacity);
        center_y.resize(new_capacity);
        center_z.resize(new_capacity);
        scale_x.resize(new_capacity);
        scale_y.resize(new_capacity);
        scale_z.resize(new_capacity);
        qw.resize(new_capacity);
        qx.resize(new_capacity);
        qy.resize(new_capacity);
        qz.resize(new_capacity);
        opacity.resize(new_capacity);
        r.resize(new_capacity);
        g.resize(new_capacity);
        b.resize(new_capacity);
        feature_size.resize(new_capacity);
        area.resize(new_capacity);
        cov_xx.resize(new_capacity);
        cov_xy.resize(new_capacity);
        cov_xz.resize(new_capacity);
        cov_yy.resize(new_capacity);
        cov_yz.resize(new_capacity);
        cov_zz.resize(new_capacity);
        cov_det.resize(new_capacity);
        child_a.resize(new_capacity);
        child_b.resize(new_capacity);
        is_active.resize(new_capacity);
        if (max_sh_degree >= 1) {
            sh1.resize(new_capacity * 9);
        }
        if (max_sh_degree >= 2) {
            sh2.resize(new_capacity * 15);
        }
        if (max_sh_degree >= 3) {
            sh3.resize(new_capacity * 21);
        }
        capacity = new_capacity;
    }

    center_x[idx] = cx;
    center_y[idx] = cy;
    center_z[idx] = cz;
    scale_x[idx] = sx;
    scale_y[idx] = sy;
    scale_z[idx] = sz;
    qw[idx] = qw_;
    qx[idx] = qx_;
    qy[idx] = qy_;
    qz[idx] = qz_;
    opacity[idx] = op;
    r[idx] = r_;
    g[idx] = g_;
    b[idx] = b_;

    compute_covariance_from_scale_quat(
        sx, sy, sz, qw_, qx_, qy_, qz_,
        cov_xx[idx], cov_xy[idx], cov_xz[idx],
        cov_yy[idx], cov_yz[idx], cov_zz[idx],
        cov_det[idx]);

    if (max_sh_degree >= 1 && sh1_ptr) {
        std::copy_n(sh1_ptr, 9, sh1.data() + idx * 9);
    }
    if (max_sh_degree >= 2 && sh2_ptr) {
        std::copy_n(sh2_ptr, 15, sh2.data() + idx * 15);
    }
    if (max_sh_degree >= 3 && sh3_ptr) {
        std::copy_n(sh3_ptr, 21, sh3.data() + idx * 21);
    }

    child_a[idx] = -1;
    child_b[idx] = -1;
    is_active[idx] = 1;

    ++current_count;
    return idx;
}

float BhattLodWorkset::compute_feature_size(size_t idx) const {
    const float max_s = std::max({scale_x[idx], scale_y[idx], scale_z[idx]});
    return 2.0f * max_s * lod_opacity(opacity[idx]);
}

float BhattLodWorkset::compute_area(size_t idx) const {
    return ellipsoid_area(scale_x[idx], scale_y[idx], scale_z[idx]);
}

void BhattLodWorkset::grid_cell(size_t idx, float step, int64_t& gx, int64_t& gy, int64_t& gz) const {
    gx = static_cast<int64_t>(std::floor(center_x[idx] / step));
    gy = static_cast<int64_t>(std::floor(center_y[idx] / step));
    gz = static_cast<int64_t>(std::floor(center_z[idx] / step));
}

float BhattLodWorkset::similarity(size_t a, size_t b) const {
    // Average cached covariances
    const float m00 = 0.5f * (cov_xx[a] + cov_xx[b]);
    const float m01 = 0.5f * (cov_xy[a] + cov_xy[b]);
    const float m02 = 0.5f * (cov_xz[a] + cov_xz[b]);
    const float m11 = 0.5f * (cov_yy[a] + cov_yy[b]);
    const float m12 = 0.5f * (cov_yz[a] + cov_yz[b]);
    const float m22 = 0.5f * (cov_zz[a] + cov_zz[b]);

    const float det_a = cov_det[a];
    const float det_b = cov_det[b];

    const float C00 = m11 * m22 - m12 * m12;
    const float C01 = m02 * m12 - m01 * m22;
    const float C02 = m01 * m12 - m02 * m11;
    const float C11 = m00 * m22 - m02 * m02;
    const float C12 = m01 * m02 - m00 * m12;
    const float C22 = m00 * m11 - m01 * m01;

    const float det = m00 * C00 + m01 * C01 + m02 * C02;
    const float det_sigma = det;

    if (det_sigma <= kEpsCov || det_a <= kEpsCov || det_b <= kEpsCov ||
        !std::isfinite(det_sigma) || !std::isfinite(det_a) || !std::isfinite(det_b)) {
        return 0.0f;
    }

    if (std::abs(det) < kEpsCov) {
        return 0.0f;
    }

    const float inv_det = 1.0f / det;
    const float inv_xx = C00 * inv_det;
    const float inv_yy = C11 * inv_det;
    const float inv_zz = C22 * inv_det;
    const float inv_xy = C01 * inv_det;
    const float inv_xz = C02 * inv_det;
    const float inv_yz = C12 * inv_det;

    const float dx = center_x[b] - center_x[a];
    const float dy = center_y[b] - center_y[a];
    const float dz = center_z[b] - center_z[a];

    const float quad = inv_xx * dx * dx
                     + inv_yy * dy * dy
                     + inv_zz * dz * dz
                     + 2.0f * inv_xy * dx * dy
                     + 2.0f * inv_xz * dx * dz
                     + 2.0f * inv_yz * dy * dz;

    const float term1 = 0.125f * quad;
    const float term2 = 0.5f * std::log(det_sigma / std::sqrt(det_a * det_b));
    const float distance = term1 + term2;
    const float spatial = std::exp(-distance);

    const float dr = r[a] - r[b];
    const float dg = g[a] - g[b];
    const float db = this->b[a] - this->b[b];
    const float color_delta2 = dr * dr + dg * dg + db * db;

    const float metric = spatial * std::exp(-color_delta2);
    if (std::isnan(metric) || !std::isfinite(metric)) {
        return 0.0f;
    }
    return metric;
}

size_t BhattLodWorkset::merge_nodes(size_t a, size_t node_b, float filter_size) {
    const size_t new_idx = current_count;
    if (new_idx >= capacity) {
        // Grow arrays by 50% if capacity is exceeded
        const size_t new_capacity = capacity * 3 / 2 + 1;
        center_x.resize(new_capacity);
        center_y.resize(new_capacity);
        center_z.resize(new_capacity);
        scale_x.resize(new_capacity);
        scale_y.resize(new_capacity);
        scale_z.resize(new_capacity);
        qw.resize(new_capacity);
        qx.resize(new_capacity);
        qy.resize(new_capacity);
        qz.resize(new_capacity);
        opacity.resize(new_capacity);
        r.resize(new_capacity);
        g.resize(new_capacity);
        b.resize(new_capacity);
        feature_size.resize(new_capacity);
        area.resize(new_capacity);
        cov_xx.resize(new_capacity);
        cov_xy.resize(new_capacity);
        cov_xz.resize(new_capacity);
        cov_yy.resize(new_capacity);
        cov_yz.resize(new_capacity);
        cov_zz.resize(new_capacity);
        cov_det.resize(new_capacity);
        child_a.resize(new_capacity);
        child_b.resize(new_capacity);
        is_active.resize(new_capacity);
        if (max_sh_degree >= 1) {
            sh1.resize(new_capacity * 9);
        }
        if (max_sh_degree >= 2) {
            sh2.resize(new_capacity * 15);
        }
        if (max_sh_degree >= 3) {
            sh3.resize(new_capacity * 21);
        }
        capacity = new_capacity;
    }

    // Compute weights
    float wa = area[a] * opacity[a];
    float wb = area[node_b] * opacity[node_b];
    float total_weight = wa + wb;
    if (total_weight < 1e-30f) total_weight = 1e-30f;
    wa /= total_weight;
    wb /= total_weight;

    // Weighted center
    center_x[new_idx] = wa * center_x[a] + wb * center_x[node_b];
    center_y[new_idx] = wa * center_y[a] + wb * center_y[node_b];
    center_z[new_idx] = wa * center_z[a] + wb * center_z[node_b];

    // Weighted color
    r[new_idx] = wa * r[a] + wb * r[node_b];
    g[new_idx] = wa * g[a] + wb * g[node_b];
    this->b[new_idx] = wa * this->b[a] + wb * this->b[node_b];

    // Weighted covariance
    std::array<float, 9> total_cov{};
    for (int i = 0; i < 9; ++i) total_cov[i] = 0.0f;

    const float filter2 = (0.5f * filter_size) * (0.5f * filter_size);

    auto add_cov = [&](size_t idx, float weight) {
        const float dx = center_x[idx] - center_x[new_idx];
        const float dy = center_y[idx] - center_y[new_idx];
        const float dz = center_z[idx] - center_z[new_idx];
        total_cov[0] += weight * (cov_xx[idx] + dx * dx + filter2);
        total_cov[1] += weight * (cov_xy[idx] + dx * dy);
        total_cov[2] += weight * (cov_xz[idx] + dx * dz);
        total_cov[3] += weight * (cov_xy[idx] + dx * dy);
        total_cov[4] += weight * (cov_yy[idx] + dy * dy + filter2);
        total_cov[5] += weight * (cov_yz[idx] + dy * dz);
        total_cov[6] += weight * (cov_xz[idx] + dx * dz);
        total_cov[7] += weight * (cov_yz[idx] + dy * dz);
        total_cov[8] += weight * (cov_zz[idx] + dz * dz + filter2);
    };

    add_cov(a, wa);
    add_cov(node_b, wb);

    // Symmetrize and regularize
    total_cov[1] = total_cov[3] = 0.5f * (total_cov[1] + total_cov[3]);
    total_cov[2] = total_cov[6] = 0.5f * (total_cov[2] + total_cov[6]);
    total_cov[5] = total_cov[7] = 0.5f * (total_cov[5] + total_cov[7]);
    total_cov[0] += kEpsCov;
    total_cov[4] += kEpsCov;
    total_cov[8] += kEpsCov;

    // Eigendecompose
    const auto eig = eigen_symmetric_3x3_jacobi(total_cov);
    std::array<float, 3> evals = {
        std::max(eig.values[0], kMinEval),
        std::max(eig.values[1], kMinEval),
        std::max(eig.values[2], kMinEval),
    };

    scale_x[new_idx] = std::sqrt(evals[0]);
    scale_y[new_idx] = std::sqrt(evals[1]);
    scale_z[new_idx] = std::sqrt(evals[2]);

    std::array<float, 4> quat;
    rotmat_to_quat(eig.vectors, quat);
    qw[new_idx] = quat[0];
    qx[new_idx] = quat[1];
    qy[new_idx] = quat[2];
    qz[new_idx] = quat[3];

    // Compute covariance directly from eigendecomposition (R * diag(evals) * R^T)
    {
        const float e0 = evals[0];
        const float e1 = evals[1];
        const float e2 = evals[2];
        const float* R = eig.vectors.data();
        cov_xx[new_idx] = R[0]*R[0]*e0 + R[1]*R[1]*e1 + R[2]*R[2]*e2;
        cov_xy[new_idx] = R[0]*R[3]*e0 + R[1]*R[4]*e1 + R[2]*R[5]*e2;
        cov_xz[new_idx] = R[0]*R[6]*e0 + R[1]*R[7]*e1 + R[2]*R[8]*e2;
        cov_yy[new_idx] = R[3]*R[3]*e0 + R[4]*R[4]*e1 + R[5]*R[5]*e2;
        cov_yz[new_idx] = R[3]*R[6]*e0 + R[4]*R[7]*e1 + R[5]*R[8]*e2;
        cov_zz[new_idx] = R[6]*R[6]*e0 + R[7]*R[7]*e1 + R[8]*R[8]*e2;
        cov_det[new_idx] = e0 * e1 * e2;
    }

    // Opacity
    float new_area = ellipsoid_area(scale_x[new_idx], scale_y[new_idx], scale_z[new_idx]);
    if (new_area < 1e-30f) new_area = 1e-30f;
    float new_opacity = total_weight / new_area;
    new_opacity = std::clamp(new_opacity, 0.000001f, 1000.0f);
    opacity[new_idx] = new_opacity;

    // SH blending
    if (max_sh_degree >= 1) {
        float* sh1_new = sh1.data() + new_idx * 9;
        const float* sh1_a = sh1.data() + a * 9;
        const float* sh1_b = sh1.data() + node_b * 9;
        for (int i = 0; i < 9; ++i) {
            sh1_new[i] = wa * sh1_a[i] + wb * sh1_b[i];
        }
    }
    if (max_sh_degree >= 2) {
        float* sh2_new = sh2.data() + new_idx * 15;
        const float* sh2_a = sh2.data() + a * 15;
        const float* sh2_b = sh2.data() + node_b * 15;
        for (int i = 0; i < 15; ++i) {
            sh2_new[i] = wa * sh2_a[i] + wb * sh2_b[i];
        }
    }
    if (max_sh_degree >= 3) {
        float* sh3_new = sh3.data() + new_idx * 21;
        const float* sh3_a = sh3.data() + a * 21;
        const float* sh3_b = sh3.data() + node_b * 21;
        for (int i = 0; i < 21; ++i) {
            sh3_new[i] = wa * sh3_a[i] + wb * sh3_b[i];
        }
    }

    // Set children
    child_a[new_idx] = static_cast<int32_t>(a);
    child_b[new_idx] = static_cast<int32_t>(node_b);
    is_active[new_idx] = 1;

    // Compute feature_size and area for the new node
    feature_size[new_idx] = compute_feature_size(new_idx);
    area[new_idx] = compute_area(new_idx);

    ++current_count;
    return new_idx;
}

// Main entry point
std::expected<std::unique_ptr<SplatData>, std::string> build_bhatt_lod(
    const SplatData& input,
    float lod_base,
    SplatSimplifyProgressCallback progress) {
    try {
        const auto t_total_start = std::chrono::high_resolution_clock::now();
        LOG_DEBUG("build_bhatt_lod: starting, input_count={}, lod_base={}", input.size(), lod_base);

        if (!input.means_raw().is_valid() || input.size() == 0) {
            return std::unexpected("build_bhatt_lod: input splat is empty");
        }

        const auto t_workset_start = std::chrono::high_resolution_clock::now();
        auto workset = make_workset_from_splatdata(input);
        const auto t_workset_end = std::chrono::high_resolution_clock::now();
        const auto workset_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_workset_end - t_workset_start).count();
        LOG_DEBUG("build_bhatt_lod: workset creation took {} ms, visible_count={}", workset_ms, workset.current_count);

        if (workset.current_count == 0) {
            return std::unexpected("build_bhatt_lod: no visible gaussians");
        }

        if (progress && !progress(0.05f, "Preparing LOD workset")) {
            return std::unexpected("build_bhatt_lod: cancelled by user");
        }

        // Compute feature_size and area for all initial nodes in parallel
        const auto t_feature_start = std::chrono::high_resolution_clock::now();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, workset.current_count),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    workset.feature_size[i] = workset.compute_feature_size(i);
                    workset.area[i] = workset.compute_area(i);
                }
            });
        const auto t_feature_end = std::chrono::high_resolution_clock::now();
        const auto feature_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_feature_end - t_feature_start).count();
        LOG_DEBUG("build_bhatt_lod: feature_size/area computation took {} ms", feature_ms);

        if (progress && !progress(0.10f, "Computing feature sizes")) {
            return std::unexpected("build_bhatt_lod: cancelled by user");
        }

        // Sort by feature_size (ascending) using indirect sort
        std::vector<size_t> order(workset.current_count);
        for (size_t i = 0; i < workset.current_count; ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return workset.feature_size[a] < workset.feature_size[b];
        });

        const float min_feature_size = workset.feature_size[order[0]];
        const int16_t level_min = static_cast<int16_t>(std::ceil(std::log(std::max(min_feature_size, 1e-12f)) / std::log(MERGE_BASE)));

        std::vector<ActiveEntry> active_entries;
        active_entries.reserve(workset.capacity);

        const auto t_merge_start = std::chrono::high_resolution_clock::now();

        size_t frontier = 0;
        int level = level_min;
        size_t total_merges = 0;
        for (; ; ++level) {
            const float step = std::pow(MERGE_BASE, level);

            // Add new splats to active set
            while (frontier < order.size() && workset.feature_size[order[frontier]] <= step) {
                const auto idx = static_cast<uint32_t>(order[frontier]);
                active_entries.push_back({-workset.feature_size[idx], idx});
                ++frontier;
            }

            std::priority_queue<ActiveEntry> active;
            for (const auto& entry : active_entries) {
                active.push(entry);
            }

            CellMap cells;
            cells.reserve(active_entries.size() * 2 + 1);
            for (const auto& entry : active_entries) {
                if (workset.is_active[entry.index] != 0) {
                    cells[cell_for(workset, entry.index, step)].push_back(entry.index);
                }
            }

            std::vector<ActiveEntry> next_active;
            next_active.reserve(active_entries.size());

            // Pre-select 8 random neighbor offsets for this level
            std::array<int, 26> neighbor_indices;
            std::iota(neighbor_indices.begin(), neighbor_indices.end(), 0);
            std::mt19937 rng(static_cast<uint32_t>(level + 12345)); // deterministic seed per level
            std::shuffle(neighbor_indices.begin(), neighbor_indices.end(), rng);

            size_t merges_this_level = 0;
            size_t loop_iterations = 0;
            LOG_DEBUG("build_bhatt_lod: level={}, active_entries={}, step={}", level, active_entries.size(), step);

            if (progress) {
                const float level_progress = std::min(0.80f, 0.15f + 0.65f * static_cast<float>(total_merges) / static_cast<float>(workset.initial_count));
                if (!progress(level_progress, std::format("Starting LOD level {} ({} active nodes)", level, active_entries.size()))) {
                    return std::unexpected("build_bhatt_lod: cancelled by user");
                }
            }

            while (!active.empty()) {
                const ActiveEntry entry = active.top();
                active.pop();

                const uint32_t idx = entry.index;
                if (workset.is_active[idx] == 0) {
                    continue;
                }

                const CellKey grid = cell_for(workset, idx, step);
                uint32_t best_neighbor = std::numeric_limits<uint32_t>::max();
                CellKey best_cell{};
                float best_similarity = -std::numeric_limits<float>::infinity();

                // 1. Always check own cell
                {
                    auto it = cells.find(grid);
                    if (it != cells.end()) {
                        for (const uint32_t neighbor : it->second) {
                            if (neighbor == idx || workset.is_active[neighbor] == 0) {
                                continue;
                            }
                            const float similarity = workset.similarity(idx, neighbor);
                            if (similarity > best_similarity) {
                                best_similarity = similarity;
                                best_neighbor = neighbor;
                                best_cell = grid;
                            }
                        }
                    }
                }

                // 2. Check 8 randomly selected neighboring cells
                for (int i = 0; i < 8; ++i) {
                    const auto& offset = kNeighborOffsets[neighbor_indices[i]];
                    const CellKey neighbor_cell{grid.x + offset[0], grid.y + offset[1], grid.z + offset[2]};
                    auto it = cells.find(neighbor_cell);
                    if (it == cells.end()) {
                        continue;
                    }
                    for (const uint32_t neighbor : it->second) {
                        if (neighbor == idx || workset.is_active[neighbor] == 0) {
                            continue;
                        }
                        const float similarity = workset.similarity(idx, neighbor);
                        if (similarity > best_similarity) {
                            best_similarity = similarity;
                            best_neighbor = neighbor;
                            best_cell = neighbor_cell;
                        }
                    }
                }

                if (best_neighbor == std::numeric_limits<uint32_t>::max()) {
                    next_active.push_back(entry);
                    continue;
                }

                const auto merged = static_cast<uint32_t>(workset.merge_nodes(idx, best_neighbor, 0.0f));
                workset.is_active[idx] = 0;
                erase_from_cell(cells, grid, idx);
                workset.is_active[best_neighbor] = 0;
                erase_from_cell(cells, best_cell, best_neighbor);
                ++merges_this_level;

                const float feature_size = workset.feature_size[merged];
                const ActiveEntry merged_entry{-feature_size, merged};
                if (feature_size > step) {
                    next_active.push_back(merged_entry);
                } else {
                    cells[cell_for(workset, merged, step)].push_back(merged);
                    active.push(merged_entry);
                }

                ++loop_iterations;
                if (merges_this_level % 10000 == 0) {
                    LOG_DEBUG("build_bhatt_lod: level={}, merges={}/{}, queue_size={}, nodes={}",
                             level, merges_this_level, loop_iterations, active.size(), workset.current_count);
                    if (progress) {
                        const float merge_progress = std::min(0.80f, 0.15f + 0.65f * static_cast<float>(total_merges + merges_this_level) / static_cast<float>(workset.initial_count));
                        if (!progress(merge_progress, std::format("Merging LOD level {} ({} merges)", level, total_merges + merges_this_level))) {
                            return std::unexpected("build_bhatt_lod: cancelled by user");
                        }
                    }
                }
            }

            total_merges += merges_this_level;
            LOG_DEBUG("build_bhatt_lod: level={} complete, merges_this_level={}, next_active={}, total_merges={}",
                     level, merges_this_level, next_active.size(), total_merges);
            active_entries = std::move(next_active);
            if (frontier >= order.size() && active_entries.size() <= 1) break;
        }

        const auto t_merge_end = std::chrono::high_resolution_clock::now();
        const auto merge_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_merge_end - t_merge_start).count();
        LOG_DEBUG("build_bhatt_lod: merge loop took {} ms, levels={}, total_merges={}, nodes_after_merge={}",
                 merge_ms, level - level_min + 1, total_merges, workset.current_count);

        if (progress && !progress(0.85f, "Merging complete")) {
            return std::unexpected("build_bhatt_lod: cancelled by user");
        }

        // Pruning and SplatData construction
        if (progress && !progress(0.90f, "Pruning LOD tree")) {
            return std::unexpected("build_bhatt_lod: cancelled by user");
        }

        // Step 1: Post-order pruning to select output nodes
        const auto t_prune_start = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> to_output(workset.current_count, 0);
        for (size_t i = 0; i < workset.initial_count; ++i) {
            to_output[i] = 1;
        }
        std::vector<std::vector<uint32_t>> output_children(workset.current_count);

        struct Pruner {
            const BhattLodWorkset& ws;
            float lod_base;
            std::vector<uint8_t>& to_output;
            std::vector<std::vector<uint32_t>>& output_children;

            float run(size_t idx) {
                const float my_size = ws.area[idx] * ws.opacity[idx];

                std::vector<size_t> children;
                if (ws.child_a[idx] >= 0) children.push_back(ws.child_a[idx]);
                if (ws.child_b[idx] >= 0) children.push_back(ws.child_b[idx]);

                if (children.empty()) {
                    to_output[idx] = 1;
                    return my_size;
                }

                float max_child_size = -std::numeric_limits<float>::infinity();
                std::vector<uint32_t> all_outputs;

                for (size_t child : children) {
                    float child_size = run(child);
                    max_child_size = std::max(max_child_size, child_size);

                    if (to_output[child]) {
                        all_outputs.push_back(static_cast<uint32_t>(child));
                    } else {
                        all_outputs.insert(all_outputs.end(),
                            output_children[child].begin(), output_children[child].end());
                    }
                }

                if (my_size >= max_child_size * lod_base) {
                    to_output[idx] = 1;
                    output_children[idx] = std::move(all_outputs);
                    return my_size;
                } else {
                    output_children[idx] = std::move(all_outputs);
                    return max_child_size;
                }
            }
        };

        const size_t root_index = workset.current_count - 1;
        to_output[root_index] = 1; // Force root to be output before pruning, matching Spark.
        Pruner pruner{workset, lod_base, to_output, output_children};
        pruner.run(root_index);
        const auto t_prune_end = std::chrono::high_resolution_clock::now();
        const auto prune_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_prune_end - t_prune_start).count();
        LOG_DEBUG("build_bhatt_lod: pruning took {} ms", prune_ms);

        // Step 2: Build a flat buffer where every node's direct children occupy
        // the contiguous range [child_start, child_start + child_count).
        std::vector<uint32_t> old_indices;
        std::vector<uint16_t> out_child_count;
        std::vector<uint32_t> out_child_start;
        std::vector<uint8_t> out_lod_level;

        struct FlatTreeBuilder {
            std::vector<uint32_t>& old_indices;
            std::vector<uint16_t>& out_child_count;
            std::vector<uint32_t>& out_child_start;
            std::vector<uint8_t>& out_lod_level;
            const std::vector<uint8_t>& to_output;
            const std::vector<std::vector<uint32_t>>& output_children;

            size_t append_node(size_t old_idx, uint8_t level) {
                if (old_idx >= to_output.size() || !to_output[old_idx]) {
                    throw std::runtime_error("build_bhatt_lod: attempted to emit non-output LOD node");
                }

                const size_t new_idx = old_indices.size();
                old_indices.push_back(static_cast<uint32_t>(old_idx));
                out_child_count.push_back(0);
                out_child_start.push_back(0);
                out_lod_level.push_back(level);
                return new_idx;
            }

            void expand_node(size_t new_idx, size_t old_idx, uint8_t level) {
                const auto& children = output_children[old_idx];
                if (children.empty()) {
                    return;
                }
                if (children.size() > std::numeric_limits<uint16_t>::max()) {
                    throw std::runtime_error(
                        "build_bhatt_lod: LOD node has too many children: old_idx=" +
                        std::to_string(old_idx) +
                        " level=" +
                        std::to_string(level) +
                        " child_count=" +
                        std::to_string(children.size()));
                }

                out_child_start[new_idx] = static_cast<uint32_t>(old_indices.size());
                out_child_count[new_idx] = static_cast<uint16_t>(children.size());

                std::vector<size_t> child_new_indices;
                child_new_indices.reserve(children.size());
                for (uint32_t child_old : children) {
                    child_new_indices.push_back(append_node(child_old, static_cast<uint8_t>(level + 1)));
                }

                for (size_t i = 0; i < children.size(); ++i) {
                    expand_node(child_new_indices[i], children[i], static_cast<uint8_t>(level + 1));
                }
            }

            void run(size_t root_old_idx) {
                const size_t root_new_idx = append_node(root_old_idx, 0);
                expand_node(root_new_idx, root_old_idx, 0);
            }
        };

        FlatTreeBuilder flat_tree{
            old_indices,
            out_child_count,
            out_child_start,
            out_lod_level,
            to_output,
            output_children,
        };
        const auto t_flat_start = std::chrono::high_resolution_clock::now();
        flat_tree.run(root_index);
        const auto t_flat_end = std::chrono::high_resolution_clock::now();
        const auto flat_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_flat_end - t_flat_start).count();
        LOG_DEBUG("build_bhatt_lod: flat tree building took {} ms, output_count={}", flat_ms, old_indices.size());

        // Step 3: Build SplatData
        const size_t output_count = old_indices.size();
        const int max_sh = input.get_max_sh_degree();
        const int shN_coeffs = static_cast<int>(input.max_sh_coeffs_rest());

        std::vector<float> means_vec(output_count * 3);
        std::vector<float> opacity_vec(output_count);
        std::vector<float> sh0_vec(output_count * 3); // [N, 1, 3] flattened
        std::vector<float> scaling_vec(output_count * 3);
        std::vector<float> rotation_vec(output_count * 4);
        std::vector<float> shN_vec;
        if (shN_coeffs > 0) {
            shN_vec.resize(output_count * shN_coeffs * 3);
        }

        tbb::parallel_for(tbb::blocked_range<size_t>(0, output_count),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    size_t old_idx = old_indices[i];

                    means_vec[i * 3 + 0] = workset.center_x[old_idx];
                    means_vec[i * 3 + 1] = workset.center_y[old_idx];
                    means_vec[i * 3 + 2] = workset.center_z[old_idx];

                    // Spark LOD encoding: display-space alpha directly (can exceed 1.0)
                    opacity_vec[i] = std::max(workset.opacity[old_idx], 0.0f);

                    // SH0: convert display-space back to optimizer-domain
                    sh0_vec[i * 3 + 0] = (workset.r[old_idx] - 0.5f) / SH_C0;
                    sh0_vec[i * 3 + 1] = (workset.g[old_idx] - 0.5f) / SH_C0;
                    sh0_vec[i * 3 + 2] = (workset.b[old_idx] - 0.5f) / SH_C0;

                    // Scaling: log-space
                    scaling_vec[i * 3 + 0] = std::log(std::max(workset.scale_x[old_idx], 1e-8f));
                    scaling_vec[i * 3 + 1] = std::log(std::max(workset.scale_y[old_idx], 1e-8f));
                    scaling_vec[i * 3 + 2] = std::log(std::max(workset.scale_z[old_idx], 1e-8f));

                    // Rotation: already normalized
                    rotation_vec[i * 4 + 0] = workset.qw[old_idx];
                    rotation_vec[i * 4 + 1] = workset.qx[old_idx];
                    rotation_vec[i * 4 + 2] = workset.qy[old_idx];
                    rotation_vec[i * 4 + 3] = workset.qz[old_idx];

                    // SH rest coefficients
                    if (shN_coeffs > 0) {
                        float* dst = shN_vec.data() + i * shN_coeffs * 3;
                        if (max_sh >= 1 && shN_coeffs >= 3) {
                            const float* src = workset.sh1.data() + old_idx * 9;
                            for (int k = 0; k < 3; ++k) {
                                dst[k * 3 + 0] = src[k * 3 + 0];
                                dst[k * 3 + 1] = src[k * 3 + 1];
                                dst[k * 3 + 2] = src[k * 3 + 2];
                            }
                        }
                        if (max_sh >= 2 && shN_coeffs >= 8) {
                            const float* src = workset.sh2.data() + old_idx * 15;
                            for (int k = 0; k < 5; ++k) {
                                dst[(3 + k) * 3 + 0] = src[k * 3 + 0];
                                dst[(3 + k) * 3 + 1] = src[k * 3 + 1];
                                dst[(3 + k) * 3 + 2] = src[k * 3 + 2];
                            }
                        }
                        if (max_sh >= 3 && shN_coeffs >= 15) {
                            const float* src = workset.sh3.data() + old_idx * 21;
                            for (int k = 0; k < 7; ++k) {
                                dst[(8 + k) * 3 + 0] = src[k * 3 + 0];
                                dst[(8 + k) * 3 + 1] = src[k * 3 + 1];
                                dst[(8 + k) * 3 + 2] = src[k * 3 + 2];
                            }
                        }
                    }
                }
            });

        const auto t_splatdata_start = std::chrono::high_resolution_clock::now();

        Tensor means_tensor = Tensor::from_vector(means_vec, {output_count, 3}, Device::CPU);
        Tensor opacity_tensor = Tensor::from_vector(opacity_vec, {output_count, 1}, Device::CPU);
        Tensor sh0_tensor = Tensor::from_vector(sh0_vec, {output_count, 1, 3}, Device::CPU);
        Tensor scaling_tensor = Tensor::from_vector(scaling_vec, {output_count, 3}, Device::CPU);
        Tensor rotation_tensor = Tensor::from_vector(rotation_vec, {output_count, 4}, Device::CPU);

        Tensor shN_tensor;
        if (shN_coeffs > 0) {
            shN_tensor = Tensor::from_vector(shN_vec, {output_count, static_cast<size_t>(shN_coeffs), 3}, Device::CPU);
        }

        auto result = std::make_unique<SplatData>(
            max_sh,
            std::move(means_tensor),
            std::move(sh0_tensor),
            std::move(shN_tensor),
            std::move(scaling_tensor),
            std::move(rotation_tensor),
            std::move(opacity_tensor),
            1.0f // scene_scale
        );

        // Populate SplatLodTree
        auto lod_tree = std::make_unique<SplatLodTree>();
        lod_tree->child_count = std::move(out_child_count);
        lod_tree->child_start = std::move(out_child_start);
        lod_tree->lod_level = std::move(out_lod_level);
        const size_t chunk_count =
            (output_count + SplatLodTree::kChunkSplats - 1) / SplatLodTree::kChunkSplats;
        lod_tree->chunk_to_page.resize(chunk_count);
        lod_tree->page_to_chunk.resize(chunk_count);
        std::iota(lod_tree->chunk_to_page.begin(), lod_tree->chunk_to_page.end(), 0u);
        std::iota(lod_tree->page_to_chunk.begin(), lod_tree->page_to_chunk.end(), 0u);
        lod_tree->centers.reserve(output_count);
        lod_tree->sizes.reserve(output_count);

        for (size_t i = 0; i < output_count; ++i) {
            size_t old_idx = old_indices[i];
            lod_tree->centers.emplace_back(
                workset.center_x[old_idx],
                workset.center_y[old_idx],
                workset.center_z[old_idx]
            );

            const float sx = workset.scale_x[old_idx];
            const float sy = workset.scale_y[old_idx];
            const float sz = workset.scale_z[old_idx];
            const float avg_scale = (sx + sy + sz) / 3.0f;
            float expansion = 1.0f;
            const float lod_alpha = std::max(workset.opacity[old_idx], 0.0f);
            if (lod_alpha > 1.0f) {
                const float spark_lod_opacity = std::min(lod_alpha * 4.0f - 3.0f, 5.0f);
                expansion = 1.0f + 0.7f * (spark_lod_opacity - 1.0f);
            }
            const float size = 2.0f * expansion * avg_scale;
            lod_tree->sizes.push_back(size);
        }

        lod_tree->lod_opacity_encoded = true;
        result->lod_tree = std::move(lod_tree);

        const auto t_splatdata_end = std::chrono::high_resolution_clock::now();
        const auto splatdata_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_splatdata_end - t_splatdata_start).count();
        LOG_DEBUG("build_bhatt_lod: SplatData construction took {} ms", splatdata_ms);

        const auto t_total_end = std::chrono::high_resolution_clock::now();
        const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_total_end - t_total_start).count();
        LOG_DEBUG("build_bhatt_lod: complete, total time {} ms, output_count={}", total_ms, output_count);

        if (progress && !progress(1.0f, "LOD tree complete")) {
            return std::unexpected("build_bhatt_lod: cancelled by user");
        }

        return result;
    } catch (const std::exception& e) {
        LOG_ERROR("build_bhatt_lod failed: {}", e.what());
        return std::unexpected(e.what());
    }
}

} // namespace lfs::core
