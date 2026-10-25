/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

namespace {
    void release_timing_native(LfsHipContext* context, LfsHipTiming* pool) {
        check(hipStreamSynchronize(context->stream));
        for (auto& event : pool->events) {
            if (event)
                check(hipEventDestroy(event));
            event = nullptr;
        }
        context->timings.erase(pool);
        delete pool;
    }
} // namespace
int lfs_hip_timing_create(LfsHipContext* context, size_t count, LfsHipTiming** output, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && output && count > 0 && count <= 65536, "Invalid HIP timing pool size or output");
        *output = nullptr;
        DeviceScope device(context->device);
        auto pool = std::make_unique<LfsHipTiming>();
        pool->events.resize(count, nullptr);
        pool->sequence.resize(count, 0);
        context->timings.insert(pool.get()); // Establish ownership before any event is acquired.
        try {
            for (auto& event : pool->events)
                check(hipEventCreate(&event));
        } catch (...) {
            pool->retired = true;
            auto* retained = pool.release();
            LfsHipError cleanup{};
            if (boundary(&cleanup, [&] { release_timing_native(context, retained); }))
                std::fprintf(stderr, "[error] HIP timing creation cleanup retained by context: %s (%d)\n", cleanup.message, cleanup.native_code);
            throw;
        }
        *output = pool.release();
    });
}

namespace {
    void validate_timing(LfsHipContext* context, LfsHipTiming* pool, size_t index) {
        require(context && pool && context->timings.contains(pool), "Foreign HIP timing pool");
        require(index < pool->events.size(), "HIP timing event index out of range");
    }
} // namespace

int lfs_hip_timing_mark(LfsHipContext* context, LfsHipTiming* pool, size_t index, LfsHipError* error) {
    return boundary(error, [&] {
        validate_timing(context, pool, index);
        DeviceScope device(context->device);
        check(hipEventRecord(pool->events[index], context->stream));
        pool->sequence[index] = ++pool->next_sequence;
    });
}

int lfs_hip_timing_wait(LfsHipContext* context, LfsHipTiming* pool, size_t index, LfsHipError* error) {
    return boundary(error, [&] {
        validate_timing(context, pool, index);
        require(pool->sequence[index] != 0, "HIP timing event was not recorded");
        DeviceScope device(context->device);
        check(hipEventSynchronize(pool->events[index]));
    });
}

int lfs_hip_timing_elapsed(LfsHipContext* context, LfsHipTiming* pool, size_t begin, size_t end,
                           float* milliseconds, int32_t* ready, LfsHipError* error) {
    return boundary(error, [&] {
        require(milliseconds && ready, "Missing HIP timing result");
        *milliseconds = 0.f;
        *ready = 0;
        validate_timing(context, pool, begin);
        validate_timing(context, pool, end);
        if (!pool->sequence[begin] || !pool->sequence[end] || pool->sequence[end] < pool->sequence[begin])
            return;
        DeviceScope device(context->device);
        for (const auto index : {begin, end}) {
            const auto status = hipEventQuery(pool->events[index]);
            if (status == hipErrorNotReady)
                return;
            check(status);
        }
        float elapsed = 0.f;
        const auto status = hipEventElapsedTime(&elapsed, pool->events[begin], pool->events[end]);
        if (status == hipErrorNotReady)
            return;
        check(status);
        if (!std::isfinite(elapsed) || elapsed < 0.f)
            return;
        *milliseconds = elapsed;
        *ready = 1;
    });
}

int lfs_hip_timing_destroy(LfsHipContext* context, LfsHipTiming* pool, LfsHipError* error) {
    return boundary(error, [&] {
        validate_timing(context, pool, 0);
        DeviceScope device(context->device);
        release_timing_native(context, pool);
    });
}
