/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                              \
    do {                                              \
        cudaError_t error = call;                     \
        if (error != cudaSuccess) {                   \
            LOG_ERROR("CUDA error at {}:{} - {}: {}", \
                      __FILE__, __LINE__,             \
                      cudaGetErrorName(error),        \
                      cudaGetErrorString(error));     \
        }                                             \
    } while (0)

#define CHECK_CUBLAS(call)                             \
    do {                                               \
        cublasStatus_t error = call;                   \
        if (error != CUBLAS_STATUS_SUCCESS) {          \
            LOG_ERROR("cuBLAS error at {}:{} - {}",    \
                      __FILE__, __LINE__, (int)error); \
        }                                              \
    } while (0)

namespace lfs::core {

    // Global cuBLAS handle (should be properly managed in production)
    static cublasHandle_t& get_cublas_handle() {
        static cublasHandle_t handle = nullptr;
        if (!handle) {
            CHECK_CUBLAS(cublasCreate(&handle));
        }
        return handle;
    }

    // ============= Matrix Operations =============
    Tensor Tensor::mm(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensors for matrix multiplication");
            return Tensor();
        }

        if (shape_.rank() != 2 || other.shape_.rank() != 2) {
            LOG_ERROR("Matrix multiplication requires 2D tensors");
            return Tensor();
        }

        if (shape_[1] != other.shape_[0]) {
            LOG_ERROR("Matrix dimensions don't match: {}x{} @ {}x{}",
                      shape_[0], shape_[1], other.shape_[0], other.shape_[1]);
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("Matrix multiplication requires tensors on same device");
            return Tensor();
        }

        // Ensure contiguous for cuBLAS/CPU operations
        const Tensor& a = is_contiguous() ? *this : contiguous();
        const Tensor& b = other.is_contiguous() ? other : other.contiguous();

        size_t m = shape_[0];
        size_t k = shape_[1];
        size_t n = other.shape_[1];

        auto result = empty({m, n}, device_, dtype_);

        if (device_ == Device::CUDA) {
            // Use cuBLAS for GPU
            cublasHandle_t handle = get_cublas_handle();

            const float alpha = 1.0f;
            const float beta = 0.0f;

            // cuBLAS uses column-major, we have row-major, so we compute C = B^T @ A^T
            CHECK_CUBLAS(cublasSgemm(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                b.ptr<float>(), n,
                a.ptr<float>(), k,
                &beta,
                result.ptr<float>(), n));
        } else {
            // CPU implementation
            const float* a_ptr = a.ptr<float>();
            const float* b_ptr = b.ptr<float>();
            float* c = result.ptr<float>();

            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < k; ++l) {
                        sum += a_ptr[i * k + l] * b_ptr[l * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
        }

        return result;
    }

    Tensor Tensor::bmm(const Tensor& other) const {
        // Batch matrix multiply
        // Input shapes: [B, M, K] @ [B, K, N] -> [B, M, N]

        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensors for bmm");
            return Tensor();
        }

        if (shape_.rank() != 3 || other.shape_.rank() != 3) {
            LOG_ERROR("BMM requires 3D tensors");
            return Tensor();
        }

        if (shape_[0] != other.shape_[0]) {
            LOG_ERROR("Batch dimensions must match for bmm");
            return Tensor();
        }

        if (shape_[2] != other.shape_[1]) {
            LOG_ERROR("Matrix dimensions incompatible for bmm: {}x{} @ {}x{}",
                      shape_[1], shape_[2], other.shape_[1], other.shape_[2]);
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("BMM requires tensors on same device");
            return Tensor();
        }

        // Ensure contiguous for cuBLAS/CPU operations
        const Tensor& a = is_contiguous() ? *this : contiguous();
        const Tensor& b = other.is_contiguous() ? other : other.contiguous();

        size_t batch_size = shape_[0];
        size_t m = shape_[1];
        size_t k = shape_[2];
        size_t n = other.shape_[2];

        auto result = empty({batch_size, m, n}, device_, dtype_);

        if (device_ == Device::CUDA) {
            // Use cuBLAS batched GEMM
            cublasHandle_t handle = get_cublas_handle();

            const float alpha = 1.0f;
            const float beta = 0.0f;

            // For batched GEMM, we need arrays of pointers
            std::vector<const float*> a_ptrs(batch_size);
            std::vector<const float*> b_ptrs(batch_size);
            std::vector<float*> c_ptrs(batch_size);

            size_t a_stride = m * k;
            size_t b_stride = k * n;
            size_t c_stride = m * n;

            for (size_t i = 0; i < batch_size; ++i) {
                a_ptrs[i] = a.ptr<float>() + i * a_stride;
                b_ptrs[i] = b.ptr<float>() + i * b_stride;
                c_ptrs[i] = result.ptr<float>() + i * c_stride;
            }

            // Allocate device memory for pointer arrays
            const float** d_a_ptrs;
            const float** d_b_ptrs;
            float** d_c_ptrs;

            CHECK_CUDA(cudaMalloc(&d_a_ptrs, batch_size * sizeof(float*)));
            CHECK_CUDA(cudaMalloc(&d_b_ptrs, batch_size * sizeof(float*)));
            CHECK_CUDA(cudaMalloc(&d_c_ptrs, batch_size * sizeof(float*)));

            CHECK_CUDA(cudaMemcpy(d_a_ptrs, a_ptrs.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_b_ptrs, b_ptrs.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_c_ptrs, c_ptrs.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));

            // Perform batched GEMM
            CHECK_CUBLAS(cublasSgemmBatched(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                d_b_ptrs, n,
                d_a_ptrs, k,
                &beta,
                d_c_ptrs, n,
                batch_size));

            // Clean up
            CHECK_CUDA(cudaFree(d_a_ptrs));
            CHECK_CUDA(cudaFree(d_b_ptrs));
            CHECK_CUDA(cudaFree(d_c_ptrs));

        } else {
            // CPU implementation
            const float* a_data = a.ptr<float>();
            const float* b_data = b.ptr<float>();
            float* c_data = result.ptr<float>();

            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        float sum = 0.0f;
                        for (size_t l = 0; l < k; ++l) {
                            sum += a_data[b * m * k + i * k + l] *
                                   b_data[b * k * n + l * n + j];
                        }
                        c_data[b * m * n + i * n + j] = sum;
                    }
                }
            }
        }

        return result;
    }

    Tensor Tensor::matmul(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensors for matmul");
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("Matmul requires tensors on same device");
            return Tensor();
        }

        // Ensure contiguous for cuBLAS/CPU operations
        const Tensor& a = is_contiguous() ? *this : contiguous();
        const Tensor& b = other.is_contiguous() ? other : other.contiguous();

        // Determine dimensions for matrix multiplication
        size_t m, k, n;
        std::vector<size_t> result_shape;

        // Handle different dimension cases
        if (a.shape_.rank() == 1 && b.shape_.rank() == 1) {
            // Vector dot product - returns scalar (rank 0)
            if (a.shape_[0] != b.shape_[0]) {
                LOG_ERROR("Vector dimensions don't match for dot product");
                return Tensor();
            }
            return a.dot(b);
        } else if (a.shape_.rank() == 1 && b.shape_.rank() == 2) {
            // Vector-matrix multiplication (1D @ 2D)
            if (a.shape_[0] != b.shape_[0]) {
                LOG_ERROR("Dimension mismatch for vector-matrix multiplication");
                return Tensor();
            }
            m = 1;
            k = a.shape_[0];
            n = b.shape_[1];
            result_shape = {n};
        } else if (a.shape_.rank() == 2 && b.shape_.rank() == 1) {
            // Matrix-vector multiplication (2D @ 1D)
            if (a.shape_[1] != b.shape_[0]) {
                LOG_ERROR("Dimension mismatch for matrix-vector multiplication");
                return Tensor();
            }
            m = a.shape_[0];
            k = a.shape_[1];
            n = 1;
            result_shape = {m};
        } else if (a.shape_.rank() == 2 && b.shape_.rank() == 2) {
            // Matrix-matrix multiplication (2D @ 2D)
            if (a.shape_[1] != b.shape_[0]) {
                LOG_ERROR("Matrix dimensions don't match for multiplication: {}x{} @ {}x{}",
                          a.shape_[0], a.shape_[1], b.shape_[0], b.shape_[1]);
                return Tensor();
            }
            m = a.shape_[0];
            k = a.shape_[1];
            n = b.shape_[1];
            result_shape = {m, n};
        } else if (a.shape_.rank() == 3 && b.shape_.rank() == 3) {
            // Batch matrix multiplication
            if (a.shape_[0] != b.shape_[0]) {
                LOG_ERROR("Batch dimensions must match for matmul");
                return Tensor();
            }
            return a.bmm(b);
        } else if (a.shape_.rank() == 2 && b.shape_.rank() == 3) {
            // 2D @ 3D: broadcast 2D to match batch dimension
            // (M, K) @ (B, K, N) -> (B, M, N)
            if (a.shape_[1] != b.shape_[1]) {
                LOG_ERROR("Dimension mismatch for 2D @ 3D matmul: {}x{} @ {}x{}x{}",
                          a.shape_[0], a.shape_[1], b.shape_[0], b.shape_[1], b.shape_[2]);
                return Tensor();
            }

            size_t batch_size = b.shape_[0];
            size_t M = a.shape_[0];
            size_t K = a.shape_[1];
            size_t N = b.shape_[2];

            // Expand 2D tensor to 3D by adding batch dimension
            auto expanded_a = a.unsqueeze(0).expand({static_cast<int>(batch_size),
                                                     static_cast<int>(M),
                                                     static_cast<int>(K)});
            return expanded_a.bmm(b);

        } else if (a.shape_.rank() == 3 && b.shape_.rank() == 2) {
            // 3D @ 2D: broadcast 2D to match batch dimension
            // (B, M, K) @ (K, N) -> (B, M, N)
            if (a.shape_[2] != b.shape_[0]) {
                LOG_ERROR("Dimension mismatch for 3D @ 2D matmul: {}x{}x{} @ {}x{}",
                          a.shape_[0], a.shape_[1], a.shape_[2], b.shape_[0], b.shape_[1]);
                return Tensor();
            }

            size_t batch_size = a.shape_[0];
            size_t M = a.shape_[1];
            size_t K = a.shape_[2];
            size_t N = b.shape_[1];

            // Expand 2D tensor to 3D by adding batch dimension
            auto expanded_b = b.unsqueeze(0).expand({static_cast<int>(batch_size),
                                                     static_cast<int>(K),
                                                     static_cast<int>(N)});
            return a.bmm(expanded_b);

        } else {
            LOG_ERROR("MatMul not implemented for {}D @ {}D", a.shape_.rank(), b.shape_.rank());
            return Tensor();
        }

        auto result = empty(TensorShape(result_shape), device_, dtype_);

        if (device_ == Device::CUDA) {
            // Use cuBLAS for GPU
            cublasHandle_t handle = get_cublas_handle();

            const float alpha = 1.0f;
            const float beta = 0.0f;

            // Create temporary reshaped views for matrix multiplication
            Tensor a_2d, b_2d;

            if (a.shape_.rank() == 1) {
                a_2d = a.view({1, static_cast<int>(a.shape_[0])});
            } else {
                a_2d = a.view(a.shape_);
            }

            if (b.shape_.rank() == 1) {
                b_2d = b.view({static_cast<int>(b.shape_[0]), 1});
            } else {
                b_2d = b.view(b.shape_);
            }

            // cuBLAS uses column-major, we have row-major
            CHECK_CUBLAS(cublasSgemm(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                b_2d.ptr<float>(), (b.shape_.rank() == 1) ? 1 : n,
                a_2d.ptr<float>(), k,
                &beta,
                result.ptr<float>(), (result_shape.size() == 1 && a.shape_.rank() == 2) ? 1 : n));
        } else {
            // CPU implementation
            const float* a_data = a.ptr<float>();
            const float* b_data = b.ptr<float>();
            float* c_data = result.ptr<float>();

            if (a.shape_.rank() == 1 && b.shape_.rank() == 2) {
                // Vector-matrix: [k] @ [k, n] -> [n]
                for (size_t j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (size_t i = 0; i < k; ++i) {
                        sum += a_data[i] * b_data[i * n + j];
                    }
                    c_data[j] = sum;
                }
            } else if (a.shape_.rank() == 2 && b.shape_.rank() == 1) {
                // Matrix-vector: [m, k] @ [k] -> [m]
                for (size_t i = 0; i < m; ++i) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < k; ++j) {
                        sum += a_data[i * k + j] * b_data[j];
                    }
                    c_data[i] = sum;
                }
            } else {
                // Matrix-matrix: [m, k] @ [k, n] -> [m, n]
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        float sum = 0.0f;
                        for (size_t l = 0; l < k; ++l) {
                            sum += a_data[i * k + l] * b_data[l * n + j];
                        }
                        c_data[i * n + j] = sum;
                    }
                }
            }
        }

        return result;
    }

    Tensor Tensor::dot(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensors for dot product");
            return Tensor();
        }

        if (shape_.rank() != 1 || other.shape_.rank() != 1) {
            LOG_ERROR("Dot product requires 1D tensors");
            return Tensor();
        }

        if (shape_[0] != other.shape_[0]) {
            LOG_ERROR("Vector dimensions don't match for dot product");
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("Dot product requires tensors on same device");
            return Tensor();
        }

        // Ensure contiguous for cuBLAS/CPU operations
        const Tensor& a = is_contiguous() ? *this : contiguous();
        const Tensor& b = other.is_contiguous() ? other : other.contiguous();

        // Create result with shape {1} first
        auto result = empty({1}, device_, dtype_);

        if (device_ == Device::CUDA) {
            cublasHandle_t handle = get_cublas_handle();

            CHECK_CUBLAS(cublasSdot(
                handle,
                a.shape_[0],
                a.ptr<float>(), 1,
                b.ptr<float>(), 1,
                result.ptr<float>()));
        } else {
            const float* a_data = a.ptr<float>();
            const float* b_data = b.ptr<float>();
            float sum = 0.0f;

            for (size_t i = 0; i < a.shape_[0]; ++i) {
                sum += a_data[i] * b_data[i];
            }

            *result.ptr<float>() = sum;
        }

        // Convert to scalar (rank-0) by creating a view with empty shape
        // This shares memory with the rank-1 result
        Tensor scalar(result.raw_ptr(), TensorShape(std::vector<size_t>{}), device_, dtype_);
        scalar.data_owner_ = result.data_owner_;
        scalar.is_view_ = true;
        return scalar;
    }

#undef CHECK_CUDA
#undef CHECK_CUBLAS

} // namespace lfs::core