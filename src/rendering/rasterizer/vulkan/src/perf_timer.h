#pragma once

#include "gs_pipeline.h"

#include <chrono>
#include <map>
#include <mutex>

namespace PerfTimer {

#define PERF_TIMER_TRAIN_STAGES   \
    _(ProjectionForward)          \
    _(GenerateKeys)               \
    _(ComputeTileRanges)          \
    _(RasterizeForward)           \
    _(_Cumsum)                    \
    _(CalculateIndexBufferOffset) \
    _(SortRTS)

#define _(name) name,
    enum TrainStage {
        PERF_TIMER_TRAIN_STAGES
            END
    };
#undef _

    void reset();

    void hostTic();
    void hostToc();

    template <TrainStage stage>
    struct Timer {
        VulkanGSPipeline* module;
        std::chrono::time_point<std::chrono::high_resolution_clock> then;
        // static std::mutex mutex;

        Timer(VulkanGSPipeline* module);
        ~Timer();
    };

    void pushMarker(VulkanGSPipeline* module);
    void popMarkers(VulkanGSPipeline* module);

    std::vector<std::pair<size_t, double>> update(std::vector<double> times);

    std::map<std::string, std::tuple<size_t, double>> get_summary();

} // namespace PerfTimer
