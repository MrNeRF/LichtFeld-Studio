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
#include "default_executor.h"
#include "imgproc/exception.h"
#include "log.h"
#include <cassert>
#include <thread>

namespace nvimgcodec {

    DefaultExecutor::DefaultExecutor(ILogger* logger, int num_threads)
        : logger_(logger), desc_{NVIMGCODEC_STRUCTURE_TYPE_EXECUTOR_DESC, sizeof(nvimgcodecExecutorDesc_t), nullptr, this, &static_schedule, &static_run, &static_wait, &static_get_num_threads}, num_threads_(num_threads) {
        const int cpu_cores_count = std::thread::hardware_concurrency();
        if (cpu_cores_count > 0 && num_threads_ > cpu_cores_count - 1) { // -1 because we also use the main thread.
            NVIMGCODEC_LOG_WARNING(logger_, "Requested " << num_threads_ << " threads but there are only " << cpu_cores_count
                                                         << " CPU cores available. Will limit to " << cpu_cores_count - 1
                                                         << " threads to maximize performance");
            num_threads_ = cpu_cores_count - 1;
        } else if (num_threads_ == 0) {
            // Limiting the number of threads, as it typically hurts performance
            // Only setting a higher number when explicitly requested.
            num_threads_ = std::max(1, cpu_cores_count - 1);
        }
        NVIMGCODEC_LOG_INFO(logger_,
                            "Requested num_threads=" << num_threads << ", cpu_cores_count=" << cpu_cores_count << ", selected num_threads=" << num_threads_);
    }

    DefaultExecutor::~DefaultExecutor() {
    }

    nvimgcodecExecutorDesc_t* DefaultExecutor::getExecutorDesc() {
        return &desc_;
    }

    nvimgcodecStatus_t DefaultExecutor::schedule_impl(int device_id, int sample_idx, void* task_context, bool start_immediately,
                                                      void (*task)(int thread_id, int sample_idx, void* task_context)) {
        try {
            std::stringstream ss;
            ss << "Executor-" << device_id;
            auto it = device_id2thread_pool_.try_emplace(device_id, num_threads_, device_id, false, ss.str());

            auto& thread_pool = it.first->second;
            auto task_wrapper = [task_context, sample_idx, task](int thread_id) { task(thread_id, sample_idx, task_context); };
            thread_pool.addWork(task_wrapper);
            if (start_immediately)
                thread_pool.run();
        } catch (const std::runtime_error& e) {
            NVIMGCODEC_LOG_ERROR(logger_, e.what());
            return NVIMGCODEC_STATUS_INTERNAL_ERROR;
        }

        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t DefaultExecutor::run(int device_id) {
        auto it = device_id2thread_pool_.find(device_id);
        if (it == device_id2thread_pool_.end()) {
            return NVIMGCODEC_STATUS_INTERNAL_ERROR;
        }
        it->second.run();
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t DefaultExecutor::wait(int device_id) {
        auto it = device_id2thread_pool_.find(device_id);
        if (it == device_id2thread_pool_.end()) {
            return NVIMGCODEC_STATUS_INTERNAL_ERROR;
        }
        it->second.wait();
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    int DefaultExecutor::get_num_threads() const {
        return num_threads_;
    }

    nvimgcodecStatus_t DefaultExecutor::static_schedule(
        void* instance, int device_id, int sample_idx, void* task_context, void (*task)(int thread_id, int sample_idx, void* task_context)) {
        try {
            CHECK_NULL(instance);
            DefaultExecutor* handle = reinterpret_cast<DefaultExecutor*>(instance);
            return handle->schedule_impl(device_id, sample_idx, task_context, false, task);
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t DefaultExecutor::static_run(void* instance, int device_id) {
        try {
            CHECK_NULL(instance);
            DefaultExecutor* handle = reinterpret_cast<DefaultExecutor*>(instance);
            return handle->run(device_id);
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t DefaultExecutor::static_wait(void* instance, int device_id) {
        try {
            CHECK_NULL(instance);
            DefaultExecutor* handle = reinterpret_cast<DefaultExecutor*>(instance);
            return handle->wait(device_id);
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    int DefaultExecutor::static_get_num_threads(void* instance) {
        try {
            CHECK_NULL(instance);
            DefaultExecutor* handle = reinterpret_cast<DefaultExecutor*>(instance);
            return handle->get_num_threads();
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

} // namespace nvimgcodec