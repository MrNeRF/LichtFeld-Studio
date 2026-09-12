/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_backend.hpp"

#include "core/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <format>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace lfs::core {
    namespace {

        constexpr size_t kN = 4096;
        constexpr size_t kMat = 64;
        constexpr float kAbsTol = 1e-4f;
        constexpr float kRelTol = 1e-4f;
        constexpr float kMatmulTol = 2e-3f;

        lfs::Status fail_step(const char* step, const std::string& detail) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Tensor,
                .user_message = std::format("{}: {}", step, detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }

        bool close(const float actual, const float expected, const float atol, const float rtol) {
            if (std::isnan(actual) || std::isnan(expected)) {
                return std::isnan(actual) && std::isnan(expected);
            }
            return std::abs(actual - expected) <= atol + rtol * std::abs(expected);
        }

        lfs::Status check_vector(const char* step,
                                 const std::vector<float>& actual,
                                 const std::vector<float>& expected,
                                 const float atol,
                                 const float rtol = kRelTol) {
            if (actual.size() != expected.size()) {
                return fail_step(step, std::format("size {} vs {}", actual.size(), expected.size()));
            }
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!close(actual[i], expected[i], atol, rtol)) {
                    return fail_step(step, std::format("index {} got {} expected {}",
                                                       i, actual[i], expected[i]));
                }
            }
            return {};
        }

        lfs::Status check_scalar(const char* step,
                                 const float actual,
                                 const float expected,
                                 const float atol,
                                 const float rtol = kRelTol) {
            if (!close(actual, expected, atol, rtol)) {
                return fail_step(step, std::format("got {} expected {}", actual, expected));
            }
            return {};
        }

        std::vector<float> make_pattern(const size_t n) {
            std::vector<float> values(n);
            for (size_t i = 0; i < n; ++i) {
                values[i] = static_cast<float>(i) * 0.01f + 0.5f;
            }
            return values;
        }

        lfs::Status run_pointwise_and_reduction(const std::vector<float>& host) {
            const Tensor a = Tensor::from_vector(host, {host.size()}, Device::GPU);
            const Tensor pointwise = (a * 2.0f + 1.0f).sqrt();

            std::vector<float> expected_pointwise(host.size());
            float expected_sum = 0.0f;
            float expected_max = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < host.size(); ++i) {
                expected_pointwise[i] = std::sqrt(host[i] * 2.0f + 1.0f);
                expected_sum += host[i];
                expected_max = std::max(expected_max, host[i]);
            }

            if (auto status = check_vector("pointwise", pointwise.to_vector(),
                                           expected_pointwise, kAbsTol);
                !status) {
                return status;
            }
            if (auto status = check_scalar("sum_scalar", a.sum_scalar(), expected_sum, 1e-3f, 1e-4f);
                !status) {
                return status;
            }
            return check_scalar("max_scalar", a.max_scalar(), expected_max, kAbsTol);
        }

        lfs::Status run_full_corpus(const std::vector<float>& host) {
            if (auto status = run_pointwise_and_reduction(host); !status) {
                return status;
            }

            const Tensor a = Tensor::from_vector(host, {host.size()}, Device::GPU);

            const auto [sorted_values, sorted_indices] = a.sort(0, false);
            std::vector<size_t> order(host.size());
            for (size_t i = 0; i < order.size(); ++i) {
                order[i] = i;
            }
            std::stable_sort(order.begin(), order.end(), [&](const size_t lhs, const size_t rhs) {
                return host[lhs] < host[rhs];
            });
            std::vector<float> expected_sorted(host.size());
            for (size_t i = 0; i < order.size(); ++i) {
                expected_sorted[i] = host[order[i]];
            }
            if (auto status = check_vector("sort", sorted_values.to_vector(), expected_sorted, kAbsTol);
                !status) {
                return status;
            }
            const std::vector<int64_t> got_indices = sorted_indices.to_vector_int64();
            if (got_indices.size() != order.size()) {
                return fail_step("sort", std::format("index size {} vs {}",
                                                     got_indices.size(), order.size()));
            }
            for (size_t i = 0; i < order.size(); ++i) {
                if (got_indices[i] != static_cast<int64_t>(order[i])) {
                    return fail_step("sort", std::format("index {} got {} expected {}",
                                                         i, got_indices[i], order[i]));
                }
            }

            const Tensor matrix = a.reshape({kMat, kMat});
            const Tensor product = matrix.matmul(matrix);
            std::vector<float> expected_product(kMat * kMat, 0.0f);
            for (size_t i = 0; i < kMat; ++i) {
                for (size_t k = 0; k < kMat; ++k) {
                    const float aik = host[i * kMat + k];
                    for (size_t j = 0; j < kMat; ++j) {
                        expected_product[i * kMat + j] += aik * host[k * kMat + j];
                    }
                }
            }
            if (auto status = check_vector("matmul", product.to_vector(), expected_product,
                                           kMatmulTol, kRelTol);
                !status) {
                return status;
            }

            const std::vector<int> index_host{0, 1, 17, static_cast<int>(kN - 1)};
            const Tensor indices = Tensor::from_vector(index_host, {index_host.size()}, Device::GPU);
            const Tensor selected = a.index_select(0, indices);
            std::vector<float> expected_selected;
            expected_selected.reserve(index_host.size());
            for (const int index : index_host) {
                expected_selected.push_back(host[static_cast<size_t>(index)]);
            }
            if (auto status = check_vector("index_select", selected.to_vector(),
                                           expected_selected, kAbsTol);
                !status) {
                return status;
            }

            const Tensor mask = a.gt(2.0f);
            const Tensor masked = a.masked_select(mask);
            std::vector<float> expected_masked;
            size_t expected_nonzero = 0;
            for (const float value : host) {
                if (value > 2.0f) {
                    expected_masked.push_back(value);
                    ++expected_nonzero;
                }
            }
            if (auto status = check_vector("masked_select", masked.to_vector(),
                                           expected_masked, kAbsTol);
                !status) {
                return status;
            }

            const size_t got_nonzero = mask.count_nonzero();
            if (got_nonzero != expected_nonzero) {
                return fail_step("count_nonzero", std::format("got {} expected {}",
                                                              got_nonzero, expected_nonzero));
            }
            return {};
        }

    } // namespace

    lfs::Status tensor_backend_selftest(const GpuBackend backend) {
        if (!gpu_backend_available(backend)) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Unavailable,
                .domain = backend == GpuBackend::Vulkan ? lfs::ErrorDomain::Vulkan
                                                        : lfs::ErrorDomain::CUDA,
                .user_message = "backend unavailable",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }

        try {
            GpuBackendScope scope(backend);
            const std::vector<float> host = make_pattern(kN);

            {
                if (auto status = run_full_corpus(host); !status) {
                    return status;
                }
            }

            if (auto status = shutdown_gpu_backend(backend); !status) {
                return status;
            }

            return run_pointwise_and_reduction(host);
        } catch (const lfs::Exception& exception) {
            return lfs::Status::failure(exception.error());
        } catch (const std::exception& exception) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Tensor,
                .user_message = exception.what(),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
    }

} // namespace lfs::core
