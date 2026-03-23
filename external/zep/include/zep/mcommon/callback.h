#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>
/**
 * Basic std::function callback example. Templated on a function signature
 * that takes in a uint32_t and returns void.
 */
typedef std::function<void(uint32_t)> cb_t;

/**
 * Here's an example type that stores a std::function and an argument
 * to pass with that callback.  Passing a uint32_t input to your callback
 * is a pretty common implementation. Also useful if you need to store a
 * pointer for use with a BOUNCE function.
 */
struct cb_arg_t {
    // the callback - takes a uint32_t input.
    std::function<void(uint32_t)> cb;
    // value to return with the callback.
    uint32_t arg;
};

/**
 * Alternative storage implementation.  Perhaps you want to store
 * callbacks for different event types?
 */
/*
enum my_events_t
{
    VIDEO_STOP = 0,
    VIDEO_START,
    EVENT_MAX
};
*/

struct cb_event_t {
    std::function<void(uint32_t)> cb;
    my_events_t ev;
};
