/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_ops.hpp"
#include <cuda_runtime.h>

namespace lfs::core::tensor_ops {

    namespace {
        constexpr size_t MAX_GRID_Y_DIM = 65535;
    }

    // Transpose kernel using shared memory
    template <int TILE_DIM, int BLOCK_ROWS>
    __global__ void transpose_kernel(const float* input, float* output, size_t rows, size_t cols) {
        __shared__ float tile[TILE_DIM][TILE_DIM + 1];

        int x = blockIdx.x * TILE_DIM + threadIdx.x;
        int y = blockIdx.y * TILE_DIM + threadIdx.y;
        int width = cols;

        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            if (x < cols && (y + j) < rows) {
                tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
            }
        }

        __syncthreads();

        x = blockIdx.y * TILE_DIM + threadIdx.x;
        y = blockIdx.x * TILE_DIM + threadIdx.y;
        width = rows;

        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            if (x < rows && (y + j) < cols) {
                output[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }

    // Register-tiled sgemm: C = A @ B
    template <int BM, int BN, int BK, int TM, int TN>
    __global__ void sgemm_optimized_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int m, int n, int k) {
        __shared__ float As[BM][BK];
        __shared__ float Bs[BK][BN];

        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int tid = ty * blockDim.x + tx;
        const int num_threads = blockDim.x * blockDim.y;

        const int block_row = blockIdx.y * BM;
        const int block_col = blockIdx.x * BN;
        const int thread_row = ty * TM;
        const int thread_col = tx * TN;

        float acc[TM][TN] = {{0.0f}};
        float a_frag[TM];
        float b_frag[TN];

        for (int tile = 0; tile < k; tile += BK) {
            for (int i = tid; i < BM * BK; i += num_threads) {
                int r = i / BK, c = i % BK;
                int gr = block_row + r, gc = tile + c;
                As[r][c] = (gr < m && gc < k) ? __ldg(&A[gr * k + gc]) : 0.0f;
            }
            for (int i = tid; i < BK * BN; i += num_threads) {
                int r = i / BN, c = i % BN;
                int gr = tile + r, gc = block_col + c;
                Bs[r][c] = (gr < k && gc < n) ? __ldg(&B[gr * n + gc]) : 0.0f;
            }
            __syncthreads();

#pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
                for (int i = 0; i < TM; ++i)
                    a_frag[i] = As[thread_row + i][kk];
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    b_frag[j] = Bs[kk][thread_col + j];
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] += a_frag[i] * b_frag[j];
            }
            __syncthreads();
        }

#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                int gr = block_row + thread_row + i;
                int gc = block_col + thread_col + j;
                if (gr < m && gc < n)
                    C[gr * n + gc] = acc[i][j];
            }
        }
    }

    // Tiled sgemm with double buffering: C = A @ B
    template <int TILE_SIZE>
    __global__ void sgemm_tiled_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int m, int n, int k) {
        __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];
        __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];

        const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
        const int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

        float sum = 0.0f;
        int curr = 0;

        {
            int a_col = threadIdx.x, b_row = threadIdx.y;
            As[0][threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? __ldg(&A[row * k + a_col]) : 0.0f;
            Bs[0][threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? __ldg(&B[b_row * n + col]) : 0.0f;
        }
        __syncthreads();

        for (int t = 0; t < num_tiles; ++t) {
            int next = 1 - curr;
            if (t + 1 < num_tiles) {
                int a_col = (t + 1) * TILE_SIZE + threadIdx.x;
                int b_row = (t + 1) * TILE_SIZE + threadIdx.y;
                As[next][threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? __ldg(&A[row * k + a_col]) : 0.0f;
                Bs[next][threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? __ldg(&B[b_row * n + col]) : 0.0f;
            }
#pragma unroll
            for (int i = 0; i < TILE_SIZE; ++i)
                sum += As[curr][threadIdx.y][i] * Bs[curr][i][threadIdx.x];
            __syncthreads();
            curr = next;
        }

        if (row < m && col < n)
            C[row * n + col] = sum;
    }

    // Fused sgemm + bias + relu for conv1x1
    template <int TILE_SIZE>
    __global__ void sgemm_bias_relu_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           const float* __restrict__ bias,
                                           float* __restrict__ C,
                                           int m, int n, int k) {
        __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];
        __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];

        const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
        const int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

        float sum = 0.0f;
        int curr = 0;

        {
            int a_col = threadIdx.x, b_row = threadIdx.y;
            As[0][threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? __ldg(&A[row * k + a_col]) : 0.0f;
            Bs[0][threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? __ldg(&B[b_row * n + col]) : 0.0f;
        }
        __syncthreads();

        for (int t = 0; t < num_tiles; ++t) {
            int next = 1 - curr;
            if (t + 1 < num_tiles) {
                int a_col = (t + 1) * TILE_SIZE + threadIdx.x;
                int b_row = (t + 1) * TILE_SIZE + threadIdx.y;
                As[next][threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? __ldg(&A[row * k + a_col]) : 0.0f;
                Bs[next][threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? __ldg(&B[b_row * n + col]) : 0.0f;
            }
#pragma unroll
            for (int i = 0; i < TILE_SIZE; ++i)
                sum += As[curr][threadIdx.y][i] * Bs[curr][i][threadIdx.x];
            __syncthreads();
            curr = next;
        }

        if (row < m && col < n)
            C[row * n + col] = fmaxf(sum + __ldg(&bias[row]), 0.0f);
    }

    // Register-tiled sgemm: C = A @ B^T
    template <int BM, int BN, int BK, int TM, int TN>
    __global__ void sgemm_tn_optimized_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int m, int n, int k) {
        __shared__ float As[BM][BK];
        __shared__ float Bs[BK][BN];

        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int tid = ty * blockDim.x + tx;
        const int num_threads = blockDim.x * blockDim.y;

        const int block_row = blockIdx.y * BM;
        const int block_col = blockIdx.x * BN;
        const int thread_row = ty * TM;
        const int thread_col = tx * TN;

        float acc[TM][TN] = {{0.0f}};
        float a_frag[TM];
        float b_frag[TN];

        for (int tile = 0; tile < k; tile += BK) {
            for (int i = tid; i < BM * BK; i += num_threads) {
                int r = i / BK, c = i % BK;
                int gr = block_row + r, gc = tile + c;
                As[r][c] = (gr < m && gc < k) ? __ldg(&A[gr * k + gc]) : 0.0f;
            }
            for (int i = tid; i < BK * BN; i += num_threads) {
                int r = i / BN, c = i % BN;
                int gk = tile + r, gn = block_col + c;
                Bs[r][c] = (gk < k && gn < n) ? __ldg(&B[gn * k + gk]) : 0.0f;
            }
            __syncthreads();

#pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
                for (int i = 0; i < TM; ++i)
                    a_frag[i] = As[thread_row + i][kk];
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    b_frag[j] = Bs[kk][thread_col + j];
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] += a_frag[i] * b_frag[j];
            }
            __syncthreads();
        }

#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                int gr = block_row + thread_row + i;
                int gc = block_col + thread_col + j;
                if (gr < m && gc < n)
                    C[gr * n + gc] = acc[i][j];
            }
        }
    }

    // Tiled sgemm with double buffering: C = A @ B^T
    template <int TILE_SIZE>
    __global__ void sgemm_tn_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int m, int n, int k) {
        __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];
        __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];

        const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
        const int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

        float sum = 0.0f;
        int curr = 0;

        {
            int a_col = threadIdx.x, b_col = threadIdx.y;
            As[0][threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? __ldg(&A[row * k + a_col]) : 0.0f;
            Bs[0][threadIdx.y][threadIdx.x] = (col < n && b_col < k) ? __ldg(&B[col * k + b_col]) : 0.0f;
        }
        __syncthreads();

        for (int t = 0; t < num_tiles; ++t) {
            int next = 1 - curr;
            if (t + 1 < num_tiles) {
                int a_col = (t + 1) * TILE_SIZE + threadIdx.x;
                int b_col = (t + 1) * TILE_SIZE + threadIdx.y;
                As[next][threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? __ldg(&A[row * k + a_col]) : 0.0f;
                Bs[next][threadIdx.y][threadIdx.x] = (col < n && b_col < k) ? __ldg(&B[col * k + b_col]) : 0.0f;
            }
#pragma unroll
            for (int i = 0; i < TILE_SIZE; ++i)
                sum += As[curr][threadIdx.y][i] * Bs[curr][i][threadIdx.x];
            __syncthreads();
            curr = next;
        }

        if (row < m && col < n)
            C[row * n + col] = sum;
    }

    // Batched tiled sgemm
    template <int TILE_SIZE>
    __global__ void sgemm_batched_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int m, int n, int k,
                                         long long stride_a, long long stride_b, long long stride_c) {
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        const int batch = blockIdx.z;
        const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

        const float* A_b = A + batch * stride_a;
        const float* B_b = B + batch * stride_b;
        float* C_b = C + batch * stride_c;

        float sum = 0.0f;
        for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
            int a_col = t * TILE_SIZE + threadIdx.x;
            int b_row = t * TILE_SIZE + threadIdx.y;
            As[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? A_b[row * k + a_col] : 0.0f;
            Bs[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? B_b[b_row * n + col] : 0.0f;
            __syncthreads();
#pragma unroll
            for (int i = 0; i < TILE_SIZE; ++i)
                sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
            __syncthreads();
        }
        if (row < m && col < n)
            C_b[row * n + col] = sum;
    }

    // Utility kernels
    __global__ void eye_kernel(float* data, size_t m, size_t n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < m * n)
            data[idx] = ((idx / n) == (idx % n)) ? 1.0f : 0.0f;
    }

    __global__ void diag_kernel(const float* diagonal, float* matrix, size_t n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n * n)
            matrix[idx] = ((idx / n) == (idx % n)) ? diagonal[idx / n] : 0.0f;
    }

    __global__ void extract_diag_kernel(const float* matrix, float* diagonal, size_t n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            diagonal[idx] = matrix[idx * n + idx];
    }

    // Launch functions
    void launch_eye(float* data, size_t m, size_t n, cudaStream_t stream) {
        int bs = 256;
        eye_kernel<<<(m * n + bs - 1) / bs, bs, 0, stream>>>(data, m, n);
    }

    void launch_diag(const float* diagonal, float* matrix, size_t n, cudaStream_t stream) {
        int bs = 256;
        diag_kernel<<<(n * n + bs - 1) / bs, bs, 0, stream>>>(diagonal, matrix, n);
    }

    void launch_extract_diag(const float* matrix, float* diagonal, size_t n, cudaStream_t stream) {
        int bs = 256;
        extract_diag_kernel<<<(n + bs - 1) / bs, bs, 0, stream>>>(matrix, diagonal, n);
    }

    void launch_sgemm(const float* a, const float* b, float* c, size_t m, size_t n, size_t k, cudaStream_t stream) {
        if (m >= 16 && n >= 64 && k >= 8) {
            constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
            dim3 block(BN / TN, BM / TM);
            const size_t max_rows_per_launch = MAX_GRID_Y_DIM * static_cast<size_t>(BM);
            for (size_t row_offset = 0; row_offset < m; row_offset += max_rows_per_launch) {
                const size_t rows_this_launch = std::min(max_rows_per_launch, m - row_offset);
                dim3 grid((n + BN - 1) / BN, (rows_this_launch + BM - 1) / BM);
                sgemm_optimized_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
                    a + row_offset * k,
                    b,
                    c + row_offset * n,
                    rows_this_launch,
                    n,
                    k);
            }
        } else {
            constexpr int T = 16;
            dim3 block(T, T);
            const size_t max_rows_per_launch = MAX_GRID_Y_DIM * static_cast<size_t>(T);
            for (size_t row_offset = 0; row_offset < m; row_offset += max_rows_per_launch) {
                const size_t rows_this_launch = std::min(max_rows_per_launch, m - row_offset);
                dim3 grid((n + T - 1) / T, (rows_this_launch + T - 1) / T);
                sgemm_tiled_kernel<T><<<grid, block, 0, stream>>>(
                    a + row_offset * k,
                    b,
                    c + row_offset * n,
                    rows_this_launch,
                    n,
                    k);
            }
        }
    }

    void launch_sgemm_tn(const float* a, const float* b, float* c, size_t m, size_t n, size_t k, cudaStream_t stream) {
        if (m >= 16 && n >= 64 && k >= 8) {
            constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
            dim3 block(BN / TN, BM / TM);
            const size_t max_rows_per_launch = MAX_GRID_Y_DIM * static_cast<size_t>(BM);
            for (size_t row_offset = 0; row_offset < m; row_offset += max_rows_per_launch) {
                const size_t rows_this_launch = std::min(max_rows_per_launch, m - row_offset);
                dim3 grid((n + BN - 1) / BN, (rows_this_launch + BM - 1) / BM);
                sgemm_tn_optimized_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
                    a + row_offset * k,
                    b,
                    c + row_offset * n,
                    rows_this_launch,
                    n,
                    k);
            }
        } else {
            constexpr int T = 16;
            dim3 block(T, T);
            const size_t max_rows_per_launch = MAX_GRID_Y_DIM * static_cast<size_t>(T);
            for (size_t row_offset = 0; row_offset < m; row_offset += max_rows_per_launch) {
                const size_t rows_this_launch = std::min(max_rows_per_launch, m - row_offset);
                dim3 grid((n + T - 1) / T, (rows_this_launch + T - 1) / T);
                sgemm_tn_kernel<T><<<grid, block, 0, stream>>>(
                    a + row_offset * k,
                    b,
                    c + row_offset * n,
                    rows_this_launch,
                    n,
                    k);
            }
        }
    }

    void launch_sgemm_batched(const float* a, const float* b, float* c,
                              size_t batch, size_t m, size_t n, size_t k, cudaStream_t stream) {
        constexpr int T = 16;
        dim3 block(T, T);
        const size_t max_rows_per_launch = MAX_GRID_Y_DIM * static_cast<size_t>(T);
        for (size_t row_offset = 0; row_offset < m; row_offset += max_rows_per_launch) {
            const size_t rows_this_launch = std::min(max_rows_per_launch, m - row_offset);
            dim3 grid((n + T - 1) / T, (rows_this_launch + T - 1) / T, batch);
            sgemm_batched_kernel<T><<<grid, block, 0, stream>>>(
                a + row_offset * k,
                b,
                c + row_offset * n,
                rows_this_launch,
                n,
                k,
                m * k,
                k * n,
                m * n);
        }
    }

    void launch_sgemm_bias_relu(const float* a, const float* b, const float* bias, float* c,
                                size_t m, size_t n, size_t k, cudaStream_t stream) {
        constexpr int T = 16;
        dim3 block(T, T);
        const size_t max_rows_per_launch = MAX_GRID_Y_DIM * static_cast<size_t>(T);
        for (size_t row_offset = 0; row_offset < m; row_offset += max_rows_per_launch) {
            const size_t rows_this_launch = std::min(max_rows_per_launch, m - row_offset);
            dim3 grid((n + T - 1) / T, (rows_this_launch + T - 1) / T);
            sgemm_bias_relu_kernel<T><<<grid, block, 0, stream>>>(
                a + row_offset * k,
                b,
                bias + row_offset,
                c + row_offset * n,
                rows_this_launch,
                n,
                k);
        }
    }

} // namespace lfs::core::tensor_ops
