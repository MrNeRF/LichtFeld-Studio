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

#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <list>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace nvimgcodec {

    // arbitrary value for the CPU device ID
    constexpr int CPU_ONLY_DEVICE_ID = -99999;

    class ThreadPool {
    public:
        // Basic unit of work that our threads do
        typedef std::function<void(int)> Work;

        ThreadPool(int num_thread, int device_id, bool set_affinity, const char* name);

        ThreadPool(int num_thread, int device_id, bool set_affinity, const std::string& name)
            : ThreadPool(num_thread, device_id, set_affinity, name.c_str()) {
        }

        ~ThreadPool();

        /**
         * @brief Adds work to the queue with optional priority, and optionally starts processing
         * @param work Work to be executed
         * @param start_immediately Whether we should start the thread pool execution (if idle) on this call
         *
         * @brief The jobs are queued and are picked by the threads in FIFO order.
         */
        void addWork(Work work);

        /**
         * @brief Wakes up all the threads to complete all the queued work
         */
        void run();

        /**
         * @brief Blocks until all work issued to the thread pool is complete
         */
        void wait(bool checkForErrors = true);

        int getThreadsNum() const;

        std::vector<std::thread::id> getThreadIds() const;

        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;
        ThreadPool(ThreadPool&&) = delete;
        ThreadPool& operator=(ThreadPool&&) = delete;

    private:
        void threadMain(int thread_id, int device_id, bool set_affinity, const std::string& name);

        std::vector<std::thread> threads_;
        std::list<Work> work_queue_;

        bool running_;
        bool work_complete_;
        bool started_;
        int active_threads_;
        std::mutex mutex_;
        std::condition_variable condition_;
        std::condition_variable completed_;

        //  Stored error strings for each thread
        std::vector<std::queue<std::string>> tl_errors_;
    };

} // namespace nvimgcodec
