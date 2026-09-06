/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Packed128 - 128-bit vectorized memory operations
 * Inspired by llm.c/llmc/cuda_utils.cuh
 *
 * This provides:
 * - 128-bit aligned loads/stores for any data type
 * - Streaming cache hints (__ldcs, __stcs) for better cache utilization
 * - Cache-global stores (__stcg) for gradients/intermediate values
 */

#pragma once

#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lfs::core {

    // ============================================================================
    // Packed128: Force 128-bit (16-byte) aligned loads/stores
    // ============================================================================

    /**
     * @brief 128-bit vectorized data structure
     *
     * This forces the compiler to use 128-bit load/store instructions (LDG.128, STS.128)
     * on GPUs that support it. Similar to float4, but works for any type and size.
     *
     * Key benefits:
     * - 4× memory bandwidth for float (4 floats per load)
     * - 8× memory bandwidth for half/bfloat16 (8 values per load)
     * - Better coalescing and cache line utilization
     * - Supports streaming and cache-global hints
     */
    template <typename ElementType>
    struct alignas(16) Packed128 {
        static constexpr size_t size = sizeof(int4) / sizeof(ElementType);
        ElementType payload[size];

        // Constructors
        // Note: Explicitly empty instead of = default to avoid nvcc 12.8 ICE
        __device__ Packed128() {}

        __device__ explicit Packed128(int4 bits) {
            static_assert(sizeof(bits) == sizeof(payload), "Size mismatch");
            memcpy(&payload, &bits, sizeof(bits));
        }

        // Factory methods
        __device__ static Packed128 constant(ElementType value) {
            Packed128 result;
#pragma unroll
            for (int k = 0; k < size; ++k) {
                result.payload[k] = value;
            }
            return result;
        }

        __device__ static Packed128 zeros() {
            return constant(ElementType(0));
        }

        __device__ static Packed128 ones() {
            return constant(ElementType(1));
        }

        // Element access
        __device__ ElementType& operator[](int index) {
            return payload[index];
        }

        __device__ const ElementType& operator[](int index) const {
            return payload[index];
        }

        // Conversion to int4 (for PTX instructions)
        __device__ int4 get_bits() const {
            int4 bits;
            static_assert(sizeof(bits) == sizeof(payload), "Size mismatch");
            memcpy(&bits, &payload, sizeof(bits));
            return bits;
        }
    };

    // ============================================================================
    // Load/Store Functions with Cache Hints
    // ============================================================================

    /**
     * @brief Standard 128-bit load (cached in L1 and L2)
     *
     * Use this for:
     * - Data that will be reused soon (e.g., weights, biases)
     * - Small tensors that fit in cache
     */
    template <typename ElementType>
    __device__ inline Packed128<ElementType> load128(const ElementType* address) {
        return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
    }

    /**
     * @brief Streaming load (bypass L1 cache, only cache in L2)
     *
     * Use this for:
     * - Large activations that won't be reused (forward pass)
     * - Streaming data that would thrash L1 cache
     * - Read-once patterns
     *
     * This is the __ldcs instruction: "load streaming"
     */
    template <typename ElementType>
    __device__ inline Packed128<ElementType> load128cs(const ElementType* address) {
        return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
    }

    /**
     * @brief Standard 128-bit store (write-back to L1 and L2)
     *
     * Use this for:
     * - Data that will be read back soon
     * - Small result tensors
     */
    template <typename ElementType>
    __device__ inline void store128(ElementType* target, Packed128<ElementType> value) {
        *reinterpret_cast<int4*>(target) = value.get_bits();
    }

    /**
     * @brief Streaming store (bypass L1 cache, only write to L2)
     *
     * Use this for:
     * - Large activation outputs (forward pass)
     * - Data that won't be read back soon
     * - Write-once patterns
     *
     * This is the __stcs instruction: "store streaming"
     */
    template <typename ElementType>
    __device__ inline void store128cs(ElementType* target, Packed128<ElementType> value) {
        __stcs(reinterpret_cast<int4*>(target), value.get_bits());
    }

    /**
     * @brief Cache-global store (bypass L1, write to L2 only)
     *
     * Use this for:
     * - Gradients (computed once, read once in backward pass)
     * - Intermediate results that won't be reused immediately
     * - Reduces L1 cache pressure
     *
     * This is the __stcg instruction: "store cache global" (L2 only)
     */
    template <typename ElementType>
    __device__ inline void store128cg(ElementType* target, Packed128<ElementType> value) {
        __stcg(reinterpret_cast<int4*>(target), value.get_bits());
    }

    // ============================================================================
    // Convenient Type Aliases
    // ============================================================================

    using f128 = Packed128<float>;          // 4 floats (16 bytes)
    using half128 = Packed128<half>;        // 8 halves (16 bytes)
    using bf128 = Packed128<__nv_bfloat16>; // 8 bfloat16s (16 bytes)
    using i128 = Packed128<int>;            // 4 ints (16 bytes)
    using u128 = Packed128<unsigned int>;   // 4 uints (16 bytes)

    // ============================================================================
    // Compatibility Shims for Older CUDA Versions
    // ============================================================================

#if defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && \
    !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))

    // Older CUDA doesn't provide __ldcs/__stcs for bfloat16
    // We implement them by casting to unsigned short

    __device__ inline __nv_bfloat16 __ldcs(const __nv_bfloat16* address) {
        unsigned short bf = __ldcs(reinterpret_cast<const unsigned short*>(address));
        return __nv_bfloat16_raw{bf};
    }

    __device__ inline void __stcs(__nv_bfloat16* address, __nv_bfloat16 value) {
        __stcs(reinterpret_cast<unsigned short*>(address),
               reinterpret_cast<const __nv_bfloat16_raw&>(value).x);
    }

#endif // CUDA < 12 && arch < SM80

    // ============================================================================
    // Helper Functions
    // ============================================================================

    /**
     * @brief Check if pointer is aligned for Packed128 access
     */
    template <typename T>
    __host__ __device__ inline bool is_aligned_128(const T* ptr) {
        return (reinterpret_cast<uintptr_t>(ptr) % 16) == 0;
    }

    /**
     * @brief Check if size is a multiple of Packed128::size
     */
    template <typename T>
    __host__ __device__ inline bool is_size_aligned_128(size_t n) {
        return (n % Packed128<T>::size) == 0;
    }

} // namespace lfs::core
