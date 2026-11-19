/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <nvimgcodec.h>
#include <atomic>
#include <future>
#include <memory>
#include <stdexcept>
#include <utility>
#include <list>
#include <vector>

namespace nvimgcodec {

/**
 * @brief Results of a processing operation.
 */
struct ProcessingResult
{
    nvimgcodecProcessingStatus_t status_ = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    std::exception_ptr exception_ = nullptr;

    bool isSuccess() const {
      return status_ == NVIMGCODEC_PROCESSING_STATUS_SUCCESS && exception_ == nullptr;
    }

    static ProcessingResult success() { return {NVIMGCODEC_PROCESSING_STATUS_SUCCESS, {}}; }
    static ProcessingResult failure(nvimgcodecProcessingStatus_t status) { return {status, {}}; }
    static ProcessingResult failure(std::exception_ptr exception) { return {NVIMGCODEC_PROCESSING_STATUS_FAIL, std::move(exception)}; }
};

/**
 * @brief A promise object for processing results.
 */
class ProcessingResultsPromise
{
  public:
    using ResultsImpl = std::vector<nvimgcodecProcessingStatus_t>;
    using PromiseImpl = std::promise<ResultsImpl>;
    using FutureImpl = std::future<ResultsImpl>;

    explicit ProcessingResultsPromise(int num_samples);
    ~ProcessingResultsPromise() = default;

    ProcessingResultsPromise(const ProcessingResultsPromise& other) = delete;
    ProcessingResultsPromise(ProcessingResultsPromise&&) = default;
    ProcessingResultsPromise& operator=(const ProcessingResultsPromise&) = delete;
    ProcessingResultsPromise& operator=(ProcessingResultsPromise&&) = default;

    /**
   * @brief Obtains a future object for the caller/consume
   */
    FutureImpl getFuture();

    /**
   * @brief The number of samples in this promise
   */
    int getNumSamples() const;

    /**
   * @brief Sets the result for a specific sample
   */
    void set(int index, ProcessingResult res);

    /**
     * @brief Sets the result for multiple samples at once
     */
    void set(const std::vector<std::pair<int, nvimgcodecProcessingStatus_t>>& results);

    /** 
    * @brief Sets all results at once with the same result
    */
    void setAll(ProcessingResult res);

    /**
     * @brief Checks whether a given element is already set
     */
    bool isSet(int index) const;

    /**
     * @brief Checks whether all elements are already set
     */
    bool isAllSet() const;

  private:
    std::vector<nvimgcodecProcessingStatus_t> results_;
    std::vector<std::atomic<bool>> is_set_;
    std::atomic<bool> is_all_set_;
    std::atomic<size_t> pending_;
    PromiseImpl promise_impl_;
};

} // namespace nvimgcodec
