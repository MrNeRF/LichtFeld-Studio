#include "perf_timer.h"

#include "diagnostics/vram_profiler.hpp"

#include <stack>

namespace PerfTimer {

// names of train stages as strings
#define _(name) #name,
    static std::string _TrainStageNames[TrainStage::END] = {
        PERF_TIMER_TRAIN_STAGES};
#undef _

    static struct _TimerObject {
        size_t count = 0;
        double total_time = 0.0;
    } stages[TrainStage::END];

    const char* diagnosticStageScope(const TrainStage stage) {
        switch (stage) {
        case ProjectionForward: return "vksplat.shaders.slang.spirv.projection_forward";
        case RasterizeForward: return "vksplat.shaders.slang.spirv.rasterize_forward";
        case _Cumsum: return "vksplat.shaders.slang.spirv.cumsum";
        case CalculateIndexBufferOffset: return "vksplat.shaders.slang.spirv.index_buffer_offset";
        case SortPrimitivesByDepth: return "vksplat.shaders.slang.spirv.sort_primitives_by_depth";
        case BuildVisibleFlags: return "vksplat.shaders.slang.spirv.visible_flags";
        case VisiblePrefix: return "vksplat.shaders.slang.spirv.visible_prefix";
        case PrepareVisibleSort: return "vksplat.shaders.slang.spirv.prepare_visible_sort";
        case CompactVisiblePrimitives: return "vksplat.shaders.slang.spirv.compact_visible_primitives";
        case SortVisiblePrimitives: return "vksplat.shaders.glsl.spirv.radix_sort_visible";
        case CopyPrimitiveSortIndices: return "vksplat.shaders.slang.spirv.copy_primitive_sort_indices";
        case ApplyDepthOrdering: return "vksplat.shaders.slang.spirv.apply_depth_ordering";
        case PrepareTileSort: return "vksplat.shaders.slang.spirv.prepare_tile_sort";
        case END: break;
        }
        return nullptr;
    }

    std::vector<Marker> marks;
    std::vector<TrainStage> pushedMarks;

    bool hostHold = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> hostStartTime;
    double hostTimeDelta = -1.0;

    void hostTic() {
        if (hostHold) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                    "PerfTimer::hostTic cannot start an already-running host interval (host_hold={}, accumulated_seconds={}, marker_count={}, pushed_marker_count={})",
                    hostHold,
                    hostTimeDelta,
                    marks.size(),
                    pushedMarks.size()),
                LFS_SOURCE_SITE_CURRENT());
        }
        hostHold = true;
        hostStartTime = std::chrono::high_resolution_clock::now();
    }

    void hostToc() {
        if (hostTimeDelta < 0.0) {
            hostTimeDelta = 0.0;
            return;
        }
        if (!hostHold) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                    "PerfTimer::hostToc requires a running host interval (host_hold={}, accumulated_seconds={}, marker_count={}, pushed_marker_count={})",
                    hostHold,
                    hostTimeDelta,
                    marks.size(),
                    pushedMarks.size()),
                LFS_SOURCE_SITE_CURRENT());
        }
        hostHold = false;
        auto hostEndTime = std::chrono::high_resolution_clock::now();
        hostTimeDelta += std::chrono::duration<double>(hostEndTime - hostStartTime).count();
    }

    template <TrainStage stage>
    Timer<stage>::Timer(VulkanGSPipeline* module) : module(module) {
        then = std::chrono::high_resolution_clock::now();
        if (module->writeTimestamp(1))
            marks.emplace_back(stage, 1);
    }

    template <TrainStage stage>
    Timer<stage>::~Timer() {
        PerfTimer::stages[int(stage)].count += 1;

        if (module->writeTimestampNoExcept(-1))
            marks.emplace_back(stage, -1);
    }

    void pushMarker(VulkanGSPipeline* module) {

        if (!module->writeTimestamp(-1)) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                    "PerfTimer::pushMarker could not write an exit timestamp (module={:#x}, marker_count={}, pushed_marker_count={})",
                    lfs::rendering::vkHandleValue(module),
                    marks.size(),
                    pushedMarks.size()),
                LFS_SOURCE_SITE_CURRENT());
        }

        int depth = 1;
        for (int i = (int)marks.size() - 1; i >= 0; --i) {
            auto [stage, delta] = marks[i];
            depth -= delta;
            if (depth == 0) {
                pushedMarks.push_back(static_cast<TrainStage>(stage));
                marks.emplace_back(static_cast<int>(stage), -1);
                return;
            }
        }
        lfs::rendering::throw_renderer_contract(
            std::format(
                "PerfTimer::pushMarker found no matching open marker (module={:#x}, marker_count={}, pushed_marker_count={}, search_depth={})",
                lfs::rendering::vkHandleValue(module),
                marks.size(),
                pushedMarks.size(),
                depth),
            LFS_SOURCE_SITE_CURRENT());
    }

    void popMarkers(VulkanGSPipeline* module) {
        while (!pushedMarks.empty()) {
            auto stage = pushedMarks.back();
            pushedMarks.pop_back();
            PerfTimer::stages[int(stage)].total_time += hostTimeDelta;
            if (!module->writeTimestamp(1)) {
                lfs::rendering::throw_renderer_contract(
                    std::format(
                        "PerfTimer::popMarkers could not reopen a paused marker (module={:#x}, stage={}, remaining_pushed={}, marker_count={})",
                        lfs::rendering::vkHandleValue(module),
                        static_cast<int>(stage),
                        pushedMarks.size(),
                        marks.size()),
                    LFS_SOURCE_SITE_CURRENT());
            }
            marks.emplace_back(static_cast<int>(stage), 1);
        }
        hostTimeDelta = 0.0;
    }

    std::vector<Marker> takeMarkers() {
        std::vector<Marker> result;
        result.swap(marks);
        return result;
    }

    void discardMarkers() noexcept {
        marks.clear();
        pushedMarks.clear();
    }

    std::vector<std::pair<size_t, double>> update(std::vector<double> times,
                                                  const std::vector<Marker>& batch_marks) {
        if (times.size() != batch_marks.size()) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                    "PerfTimer batch update requires one marker per timestamp (timestamp_count={}, marker_count={})",
                    times.size(),
                    batch_marks.size()),
                LFS_SOURCE_SITE_CURRENT());
        }
        std::vector<std::pair<size_t, double>> results(TrainStage::END, {0, 0.0});
        std::vector<std::pair<TrainStage, double>> stack;
        for (size_t i = 0; i < times.size(); i++) {
            auto [stage_int, delta] = batch_marks[i];
            LFS_VK_DEBUG_ASSERT(
                stage_int >= 0 && stage_int < static_cast<int>(TrainStage::END),
                "PerfTimer marker stage is outside the stage enum (index={}, stage={}, delta={}, stage_count={}, stack_depth={})",
                i,
                stage_int,
                delta,
                static_cast<int>(TrainStage::END),
                stack.size());
            const auto stage = static_cast<TrainStage>(stage_int);
            if (delta == 1) {
                stack.emplace_back(stage, times[i]);
            } else {
                LFS_VK_DEBUG_ASSERT(
                    !stack.empty() && stack.back().first == stage,
                    "PerfTimer exit marker does not match the open marker stack (index={}, exit_stage={}, delta={}, stack_depth={}, open_stage={})",
                    i,
                    stage_int,
                    delta,
                    stack.size(),
                    stack.empty() ? -1 : static_cast<int>(stack.back().first));
                double dt = times[i] - stack.back().second;
                PerfTimer::stages[int(stage)].total_time += dt;
                stack.pop_back();
                results[stage].first += 1;
                results[stage].second += dt;
            }
        }
        if (!stack.empty()) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                    "PerfTimer batch ended with unclosed markers (remaining_depth={}, top_stage={}, timestamp_count={}, marker_count={})",
                    stack.size(),
                    static_cast<int>(stack.back().first),
                    times.size(),
                    batch_marks.size()),
                LFS_SOURCE_SITE_CURRENT());
        }
        for (int stage = 0; stage < int(TrainStage::END); ++stage) {
            const auto [count, elapsed_seconds] = results[stage];
            const char* const scope = count > 0
                                          ? diagnosticStageScope(static_cast<TrainStage>(stage))
                                          : nullptr;
            if (!scope) {
                continue;
            }
            lfs::diagnostics::VramProfiler::instance().recordTimerSample(
                scope,
                elapsed_seconds * 1000.0);
        }
        return results;
    }

    const char* stage_name(const size_t stage) {
        if (stage >= static_cast<size_t>(TrainStage::END))
            return "Unknown";
        return _TrainStageNames[stage].c_str();
    }

    size_t stage_count() {
        return static_cast<size_t>(TrainStage::END);
    }

// template instantiation of timers
#define _(name) template struct Timer<name>;
    PERF_TIMER_TRAIN_STAGES
#undef _

}; // namespace PerfTimer
