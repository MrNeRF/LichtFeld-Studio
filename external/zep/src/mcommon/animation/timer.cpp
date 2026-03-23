#include <chrono> // Timing
#include <iomanip>

#include "zep/mcommon/logger.h"

#include "zep/mcommon/animation/timer.h"

namespace Zep {

    struct TimedSection {
        uint64_t elapsed = 0;
        uint64_t count;
    };

    timer globalTimer;
    profile_data globalProfiler;

    uint64_t timer_get_time_now() {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    void timer_start(timer& timer) {
        timer_restart(timer);
    }

    void timer_restart(timer& timer) {
        timer.startTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    uint64_t timer_get_elapsed(const timer& timer) {
        auto now = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        return now - timer.startTime;
    }

    double timer_get_elapsed_seconds(const timer& timer) {
        return timer_to_seconds(timer_get_elapsed(timer));
    }

    double timer_to_seconds(uint64_t value) {
        return double(value / 1000000.0);
    }

    double timer_to_ms(uint64_t value) {
        return double(value / 1000.0);
    }

    ProfileBlock::ProfileBlock(const char* timer)
        : strTimer(timer) {
        timer_start(blockTimer);
        if (globalProfiler.timerData.find(timer) == globalProfiler.timerData.end()) {
            globalProfiler.timerData[timer] = profile_value{};
        }
    }

    ProfileBlock::~ProfileBlock() {
        elapsed = timer_get_elapsed(blockTimer);
        profile_add_value(globalProfiler.timerData.find(strTimer)->second, double(elapsed));
    }

    void profile_add_value(profile_value& val, double av) {
        val.count++;
        val.average = val.average * (val.count - 1) / val.count + av / val.count;
        val.current = av;
    }

} // namespace Zep
