#pragma once

#include <chrono>
#include <future>
#include <thread>

namespace Zep {
    template <typename R>
    bool is_future_ready(std::future<R> const& f) {
        return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    template <typename T>
    std::future<T> make_ready_future(T val) {
        std::promise<T> promise;
        promise.set_value(val);
        return promise.get_future();
    }

} // namespace Zep
