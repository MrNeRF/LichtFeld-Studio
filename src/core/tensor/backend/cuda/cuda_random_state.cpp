/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../random_state.hpp"
#include "core/assert.hpp"
#include <curand.h>
#include <limits>
#include <string>
#define CHECK_CURAND(call)                                                   \
    do {                                                                     \
        const curandStatus_t error = (call);                                 \
        LFS_ASSERT_MSG(error == CURAND_STATUS_SUCCESS,                       \
                       std::string("CURAND operation failed with status ") + \
                           std::to_string(static_cast<int>(error)));         \
    } while (0)

namespace lfs::core::internal {
    namespace {
        class CudaRandomState final : public NativeRandomState {
            curandGenerator_t generator_ = nullptr;
            uint64_t offset_ = 0;
            uint64_t seed_;

            void ensure_generator() {
                if (generator_)
                    return;
                CHECK_CURAND(curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_PHILOX4_32_10));
                const auto status = curandSetPseudoRandomGeneratorSeed(generator_, seed_);
                if (status != CURAND_STATUS_SUCCESS) {
                    curandDestroyGenerator(generator_);
                    generator_ = nullptr;
                    CHECK_CURAND(status);
                }
            }

        public:
            explicit CudaRandomState(uint64_t seed) : seed_(seed) {}
            ~CudaRandomState() override {
                if (generator_)
                    curandDestroyGenerator(generator_);
            }
            void reseed(uint64_t seed) override {
                seed_ = seed;
                offset_ = 0;
                if (generator_) {
                    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(generator_, seed));
                    CHECK_CURAND(curandSetGeneratorOffset(generator_, 0));
                }
            }
            void normal(float* output, size_t count, float mean, float deviation, void* stream) override {
                LFS_ASSERT_MSG(output != nullptr, "CUDA normal generation requires a valid output pointer");
                LFS_ASSERT_MSG(count > 0 && count % 2 == 0, "CUDA normal generation requires a positive even value count");
                LFS_ASSERT_MSG(count <= std::numeric_limits<uint64_t>::max() - offset_, "CUDA random generator offset overflow");
                ensure_generator();
                offset_ += count;
                CHECK_CURAND(curandSetStream(generator_, static_cast<cudaStream_t>(stream)));
                CHECK_CURAND(curandGenerateNormal(generator_, output, count, mean, deviation));
            }
            void* handle_address() override {
                ensure_generator();
                return &generator_;
            }
        };
    } // namespace
    std::unique_ptr<NativeRandomState> make_native_random_state(uint64_t seed, const char*) { return std::make_unique<CudaRandomState>(seed); }
} // namespace lfs::core::internal
