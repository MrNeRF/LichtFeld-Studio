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
#include <map>
#include "iexecutor.h"
#include "thread_pool.h"

namespace nvimgcodec {

 class ILogger; 

class DefaultExecutor : public IExecutor
{
  public:
    explicit DefaultExecutor(ILogger* logger, int num_threads);
    ~DefaultExecutor() override;
    nvimgcodecExecutorDesc_t* getExecutorDesc() override;

  private:
    nvimgcodecStatus_t schedule_impl(int device_id, int sample_idx, void* task_context, bool start_immediately,
        void (*task)(int thread_id, int sample_idx, void* task_context));
    nvimgcodecStatus_t run(int device_id);
    nvimgcodecStatus_t wait(int device_id);
    int get_num_threads() const;

    static nvimgcodecStatus_t static_schedule(
        void* instance, int device_id, int sample_idx, void* task_context,
        void (*task)(int thread_id, int sample_idx, void* task_context));
    static nvimgcodecStatus_t static_run(void* instance, int device_id);
    static nvimgcodecStatus_t static_wait(void* instance, int device_id);
    static int static_get_num_threads(void* instance);

    ILogger* logger_;
    nvimgcodecExecutorDesc_t desc_;
    int num_threads_;
    std::map<int, ThreadPool> device_id2thread_pool_;
};

} // namespace nvimgcodec