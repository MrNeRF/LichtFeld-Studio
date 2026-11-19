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

#include "processing_results.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>

namespace nvimgcodec {

void ProcessingResultsPromise::set(int index, ProcessingResult res) {
  if (is_set_[index].exchange(true))
    throw std::runtime_error("Processing results for sample " + std::to_string(index) + " was already set.");
  results_[index] = res.status_;
  if (pending_.fetch_sub(1) == 1) { // last one
    is_all_set_.store(true);
    promise_impl_.set_value(results_);
  }
}

void ProcessingResultsPromise::set(const std::vector<std::pair<int, nvimgcodecProcessingStatus_t>>& results) {
  if (results.empty()) {
    return;  // nothing to do
  }
  for (auto& result : results) {
    int sample_idx = result.first;
    auto status = result.second;
    if (is_set_[sample_idx].exchange(true))
      throw std::runtime_error("Processing results for sample " + std::to_string(sample_idx) + " was already set.");
    results_[sample_idx] = status;
  }
  if (pending_.fetch_sub(results.size()) == results.size()) { // last elements
      is_all_set_.store(true);
      promise_impl_.set_value(results_);
  }
}

void ProcessingResultsPromise::setAll(ProcessingResult res)
{
  size_t size = getNumSamples();
  for (size_t index = 0; index < size; index++)
    set(index, res);
}

bool ProcessingResultsPromise::isSet(int index) const
{
  return is_set_[index].load();
}

bool ProcessingResultsPromise::isAllSet() const
{
  return is_all_set_.load();
}

int ProcessingResultsPromise::getNumSamples() const {
  return results_.size();
}

ProcessingResultsPromise::FutureImpl ProcessingResultsPromise::getFuture() {
  return promise_impl_.get_future();
}

ProcessingResultsPromise::ProcessingResultsPromise(int num_samples)
    : results_(num_samples)
    , is_set_(num_samples)
    , pending_(num_samples)
{
  for (auto& elem: is_set_)
    elem.store(false);
  is_all_set_.store(false);

  if (num_samples == 0) {
    is_all_set_.store(true);
    promise_impl_.set_value(results_);
  }
}

} // namespace nvimgcodec
