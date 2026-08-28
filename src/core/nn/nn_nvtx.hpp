/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/alloc_counter.hpp"
#include "core/cuda_error.hpp"

#include <nvtx3/nvToolsExt.h>

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace lfs::core::nn {

    struct NvtxRange {
        explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
        ~NvtxRange() { nvtxRangePop(); }
        NvtxRange(const NvtxRange&) = delete;
        NvtxRange& operator=(const NvtxRange&) = delete;
    };

    // CUDA-event stage table. Enabled with LFS_NN_PROFILE=1. Records are
    // ordered on `stream`; dump() synchronizes once at the end.
    class StageProfile {
    public:
        explicit StageProfile(cudaStream_t stream) : stream_(stream) {
            const char* env = std::getenv("LFS_NN_PROFILE");
            on_ = env != nullptr && env[0] != '\0' && env[0] != '0';
            alloc0_ = alloc_counter::snapshot();
            if (on_) {
                mark("start");
            }
        }

        ~StageProfile() {
            for (auto& e : events_) {
                LFS_CUDA_CHECK(cudaEventDestroy(e));
            }
        }

        StageProfile(const StageProfile&) = delete;
        StageProfile& operator=(const StageProfile&) = delete;

        [[nodiscard]] bool enabled() const { return on_; }

        void mark(const char* name) {
            if (!on_) {
                return;
            }
            cudaEvent_t ev = nullptr;
            LFS_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDefault));
            LFS_CUDA_CHECK(cudaEventRecord(ev, stream_));
            names_.emplace_back(name);
            events_.push_back(ev);
        }

        void dump() {
            const auto allocs = alloc_counter::delta_since(alloc0_);
            if (!on_ || events_.size() < 2) {
                if (on_) {
                    std::fprintf(stderr, "[nn.profile] driver_allocs=%llu\n",
                                 static_cast<unsigned long long>(allocs));
                }
                return;
            }
            LFS_CUDA_CHECK(cudaEventSynchronize(events_.back()));
            std::size_t free_b = 0;
            std::size_t total_b = 0;
            LFS_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
            const std::size_t used = total_b >= free_b ? total_b - free_b : 0;
            std::fprintf(stderr,
                         "[nn.profile] stages (GPU ms)  driver_allocs=%llu  cuda_used=%.2f MiB\n",
                         static_cast<unsigned long long>(allocs),
                         static_cast<double>(used) / (1024.0 * 1024.0));
            float total = 0.0f;
            for (std::size_t i = 0; i + 1 < events_.size(); ++i) {
                float ms = 0.0f;
                LFS_CUDA_CHECK(cudaEventElapsedTime(&ms, events_[i], events_[i + 1]));
                total += ms;
                std::fprintf(stderr, "  %7.3f  %s\n", static_cast<double>(ms), names_[i].c_str());
            }
            std::fprintf(stderr, "  %7.3f  TOTAL\n", static_cast<double>(total));
        }

    private:
        bool on_ = false;
        cudaStream_t stream_ = nullptr;
        alloc_counter::Snapshot alloc0_ = 0;
        std::vector<std::string> names_;
        std::vector<cudaEvent_t> events_;
    };

} // namespace lfs::core::nn
