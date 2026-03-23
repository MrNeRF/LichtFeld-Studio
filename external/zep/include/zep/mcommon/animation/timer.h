#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace Zep {

    enum class TimerSample : uint32_t {
        None,
        Restart
    };

    struct timer {
        int64_t startTime = 0;
    };
    extern timer globalTimer;

    uint64_t timer_get_time_now();
    void timer_restart(timer& timer);
    void timer_start(timer& timer);
    uint64_t timer_get_elapsed(const timer& timer);
    double timer_get_elapsed_seconds(const timer& timer);
    double timer_to_seconds(uint64_t value);
    double timer_to_ms(uint64_t value);

    struct profile_value {
        double average = 0;
        double current = 0;
        uint64_t count = 0;
    };

    struct profile_data {
        std::unordered_map<const char*, profile_value> timerData;
    };
    extern profile_data globalProfiler;

    void profile_add_value(profile_value& val, double av);

    class ProfileBlock {
    public:
        const char* strTimer;
        timer blockTimer;
        uint64_t elapsed = 0;

        ProfileBlock(const char* timer);
        ~ProfileBlock();
    };

#define TIME_SCOPE(name) ProfileBlock name##_timer_block(#name);

} // namespace Zep
