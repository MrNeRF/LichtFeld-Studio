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
        case GenerateKeys: return "vksplat.shaders.slang.spirv.generate_keys";
        case ComputeTileRanges: return "vksplat.shaders.slang.spirv.compute_tile_ranges";
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
        case SortRTS: return "vksplat.shaders.glsl.spirv.radix_sort";
        case END: break;
        }
        return nullptr;
    }

    std::vector<Marker> marks;
    std::vector<TrainStage> pushedMarks;

    bool hostHold = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> hostStartTime;
    double hostTimeDelta = -1.0;

    // reset timing and counts
    void reset() {
        for (int i = 0; i < int(TrainStage::END); i++) {
            stages[i] = {0, 0.0};
        }
        marks.clear();
    }

    void hostTic() {
        if (hostHold)
            _THROW_ERROR("hostTic");
        hostHold = true;
        hostStartTime = std::chrono::high_resolution_clock::now();
    }

    void hostToc() {
        if (hostTimeDelta < 0.0) {
            hostTimeDelta = 0.0;
            return;
        }
        if (!hostHold)
            _THROW_ERROR("hostToc");
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
        // std::lock_guard<std::mutex> lock(mutex);
        PerfTimer::stages[int(stage)].count += 1;

        if (module->writeTimestampNoExcept(-1))
            marks.emplace_back(stage, -1);

        // auto now = std::chrono::high_resolution_clock::now();
        // double dt = std::chrono::duration<double>(now - then).count();

        // PerfTimer::stages[int(stage)].total_time += dt;
    }

    void pushMarker(VulkanGSPipeline* module) {

        if (!module->writeTimestamp(-1))
            _THROW_ERROR("Failed to write exit timestamp in pushMarker");

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
        _THROW_ERROR("Empty stack in pushMarker");
    }

    void popMarkers(VulkanGSPipeline* module) {
        while (!pushedMarks.empty()) {
            auto stage = pushedMarks.back();
            pushedMarks.pop_back();
            PerfTimer::stages[int(stage)].total_time += hostTimeDelta;
            if (!module->writeTimestamp(1))
                _THROW_ERROR("Failed to write enter timestamp in popMarkers");
            marks.emplace_back(static_cast<int>(stage), 1);
        }
        hostTimeDelta = 0.0;
    }

    std::vector<Marker> takeMarkers() {
        std::vector<Marker> result;
        result.swap(marks);
        return result;
    }

    std::vector<std::pair<size_t, double>> update(std::vector<double> times) {
        auto batch_marks = takeMarkers();
        return update(std::move(times), batch_marks);
    }

    std::vector<std::pair<size_t, double>> update(std::vector<double> times,
                                                  const std::vector<Marker>& batch_marks) {
        if (times.size() != batch_marks.size())
            _THROW_ERROR(
                "Number of timestamps (" + std::to_string(times.size()) +
                ") and number of marks (" + std::to_string(batch_marks.size()) +
                ") mismatch in batch time update");
        std::vector<std::pair<size_t, double>> results(TrainStage::END, {0, 0.0});
        std::vector<std::pair<TrainStage, double>> stack;
        for (size_t i = 0; i < times.size(); i++) {
            auto [stage_int, delta] = batch_marks[i];
            if (stage_int < 0 || stage_int >= static_cast<int>(TrainStage::END))
                _THROW_ERROR("Invalid timer stage in batch time update");
            const auto stage = static_cast<TrainStage>(stage_int);
            if (delta == 1) {
                stack.emplace_back(stage, times[i]);
            } else {
                if (stack.empty() || stack.back().first != stage)
                    _THROW_ERROR("Unbalanced timer markers in batch time update");
                double dt = times[i] - stack.back().second;
                PerfTimer::stages[int(stage)].total_time += dt;
                stack.pop_back();
                results[stage].first += 1;
                results[stage].second += dt;
            }
        }
        if (!stack.empty())
            _THROW_ERROR("Unclosed timer markers in batch time update");
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

    // return a summary of timing
    std::map<std::string, std::tuple<size_t, double>> get_summary() {
        std::map<std::string, std::tuple<size_t, double>> summary;
        for (int i = 0; i < int(TrainStage::END); i++) {
            std::string name = _TrainStageNames[i];
            summary[name] = {stages[i].count, stages[i].total_time};
        }
        return summary;
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
    // template <TrainStage name> Timer<name>::Timer();
    // template <TrainStage name> Timer<name>::~Timer();

}; // namespace PerfTimer
