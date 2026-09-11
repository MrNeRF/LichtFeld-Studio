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
    _(CullSplats)                 \
    _(ProjectionSurvivors)

#define _(name) name,
    enum TrainStage {
        PERF_TIMER_TRAIN_STAGES
            END
    };
#undef _

    void hostTic();
    void hostToc();

    template <TrainStage stage>
    struct Timer {
        VulkanGSPipeline* module;
        std::chrono::time_point<std::chrono::high_resolution_clock> then;

        Timer(VulkanGSPipeline* module);
        ~Timer();
    };

    void pushMarker(VulkanGSPipeline* module);
    void popMarkers(VulkanGSPipeline* module);

    using Marker = std::pair<int, int>;

    std::vector<Marker> takeMarkers();
    void discardMarkers() noexcept;
    std::vector<std::pair<size_t, double>> update(std::vector<double> times,
                                                  const std::vector<Marker>& batch_marks);

    const char* stage_name(size_t stage);
    size_t stage_count();

} // namespace PerfTimer
