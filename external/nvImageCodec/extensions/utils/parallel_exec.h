/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include <future>
#include "nvimgcodec.h"


struct BlockParallelCtx {
    void (*task)(int thread_id, int sample_idx, void* task_context);
    void *task_ctx;
    int num_tasks;
    int num_blocks;
    std::vector<std::promise<void>> promise;
};

static void BlockParallelExec(void* task_ctx, void (*task)(int thread_id, int sample_idx, void* task_context), int num_tasks,
    const nvimgcodecExecutionParams_t* exec_params)
{
    auto executor = exec_params->executor;
    int num_threads = executor->getNumThreads(executor->instance);

    if (num_tasks < (num_threads + 1) || num_threads < 2) {  // not worth parallelizing
        for (int i = 0; i < num_tasks; i++)
            task(-1, i, task_ctx);
    } else {
        // Divide `num_tasks` tasks into `num_blocks` blocks
        int num_blocks = num_threads + 1; // the last block is processed in the current thread
        BlockParallelCtx block_ctx{task, task_ctx, num_tasks, num_blocks};
        block_ctx.promise.resize(num_threads);
        std::vector<std::future<void>> fut;
        fut.reserve(num_threads);
        for (auto& pr : block_ctx.promise)
            fut.push_back(pr.get_future());

        auto block_task = [](int tid, int block_idx, void* context) -> void {
            auto* block_ctx = reinterpret_cast<BlockParallelCtx*>(context);
            int64_t i_start = block_ctx->num_tasks * block_idx / block_ctx->num_blocks;
            int64_t i_end = block_ctx->num_tasks * (block_idx + 1) / block_ctx->num_blocks;
            for (int i = i_start; i < i_end; i++)
                block_ctx->task(tid, i, block_ctx->task_ctx);
            if (block_idx < static_cast<int>(block_ctx->promise.size()))
                block_ctx->promise[block_idx].set_value();
        };
        int block_idx = 0;
        for (; block_idx < num_threads; ++block_idx) {
            executor->schedule(executor->instance, exec_params->device_id, block_idx, -1, &block_ctx, block_task);
        }
        executor->run(executor->instance, exec_params->device_id);
        block_task(-1, block_idx, &block_ctx);

        // wait for it to finish
        for (auto& f : fut)
            f.wait();
    }
}
