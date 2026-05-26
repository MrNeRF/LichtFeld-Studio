#pragma once

#include "gs_pipeline.h"

#include <chrono>
#include <cstddef>
#include <map>
#include <mutex>
#include <vector>

namespace PerfTimer {

#define PERF_TIMER_TRAIN_STAGES   \
    _(ProjectionForward)          \
    _(GenerateKeys)               \
    _(ComputeTileRanges)          \
    _(RasterizeForward)           \
    _(_Cumsum)                    \
    _(CalculateIndexBufferOffset) \
    _(SortPrimitivesByDepth)      \
    _(BuildVisibleFlags)          \
    _(VisiblePrefix)              \
    _(PrepareVisibleSort)         \
    _(CompactVisiblePrimitives)   \
    _(SortVisiblePrimitives)      \
    _(CopyPrimitiveSortIndices)   \
    _(ApplyDepthOrdering)         \
    _(PrepareTileSort)            \
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

    using Marker = std::pair<int, int>;

    std::vector<Marker> takeMarkers();
    std::vector<std::pair<size_t, double>> update(std::vector<double> times);
    std::vector<std::pair<size_t, double>> update(std::vector<double> times,
                                                  const std::vector<Marker>& batch_marks);

    std::map<std::string, std::tuple<size_t, double>> get_summary();
    const char* stage_name(size_t stage);
    size_t stage_count();

} // namespace PerfTimer
