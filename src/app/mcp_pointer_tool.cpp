/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/mcp_screen_tools.hpp"

#include "app/mcp_app_utils.hpp"
#include "input/injected_pointer.hpp"
#include "input/sdl_coordinate_utils.hpp"
#include "mcp/mcp_tools.hpp"
#include "visualizer/visualizer.hpp"
#include "visualizer/visualizer_impl.hpp"

#include <SDL3/SDL.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <expected>
#include <glm/vec2.hpp>
#include <string>
#include <thread>

namespace lfs::app {
    namespace {
        using json = nlohmann::json;

        struct ArgumentError {
            std::string message;
        };

        std::expected<glm::vec2, ArgumentError> vec2Arg(const json& args, const char* key) {
            if (!args.contains(key) || !args[key].is_array() || args[key].size() != 2 ||
                !args[key][0].is_number() || !args[key][1].is_number())
                return std::unexpected(ArgumentError{"must be a 2-element number array"});
            const glm::vec2 result(args[key][0].get<float>(), args[key][1].get<float>());
            if (!std::isfinite(result.x) || !std::isfinite(result.y))
                return std::unexpected(ArgumentError{"must contain finite numbers"});
            return result;
        }

        std::expected<int, ArgumentError> pointerButton(const json& args) {
            const std::string button = args.value("button", std::string("left"));
            if (button == "left")
                return SDL_BUTTON_LEFT;
            if (button == "right")
                return SDL_BUTTON_RIGHT;
            if (button == "middle")
                return SDL_BUTTON_MIDDLE;
            return std::unexpected(ArgumentError{"button must be 'left', 'right', or 'middle'"});
        }

        SDL_Keymod pointerModifiers(const json& args) {
            SDL_Keymod result = SDL_KMOD_NONE;
            for (const auto& item : args.value("modifiers", json::array())) {
                const std::string modifier = item.get<std::string>();
                if (modifier == "ctrl")
                    result = static_cast<SDL_Keymod>(result | SDL_KMOD_CTRL);
                else if (modifier == "shift")
                    result = static_cast<SDL_Keymod>(result | SDL_KMOD_SHIFT);
                else if (modifier == "alt")
                    result = static_cast<SDL_Keymod>(result | SDL_KMOD_ALT);
                else if (modifier == "super")
                    result = static_cast<SDL_Keymod>(result | SDL_KMOD_GUI);
            }
            return result;
        }

        void pushMotion(SDL_Window* window, const glm::vec2 pixels, const glm::vec2 previous) {
            const auto scale = vis::input::windowPixelScale(window);
            SDL_Event event{};
            event.type = SDL_EVENT_MOUSE_MOTION;
            event.motion.windowID = SDL_GetWindowID(window);
            event.motion.x = pixels.x / scale.x;
            event.motion.y = pixels.y / scale.y;
            event.motion.xrel = (pixels.x - previous.x) / scale.x;
            event.motion.yrel = (pixels.y - previous.y) / scale.y;
            vis::input::pushInjectedPointerEvent(event);
        }

        void pushButton(SDL_Window* window, const glm::vec2 pixels, const int button,
                        const bool down) {
            const auto scale = vis::input::windowPixelScale(window);
            SDL_Event event{};
            event.type = down ? SDL_EVENT_MOUSE_BUTTON_DOWN : SDL_EVENT_MOUSE_BUTTON_UP;
            event.button.windowID = SDL_GetWindowID(window);
            event.button.button = static_cast<Uint8>(button);
            event.button.down = down;
            event.button.clicks = 1;
            event.button.x = pixels.x / scale.x;
            event.button.y = pixels.y / scale.y;
            vis::input::pushInjectedPointerEvent(event);
        }

        json injectPointer(vis::VisualizerImpl& impl, const json& args) {
            SDL_Window* const window = impl.getWindow();
            if (!window)
                return json{{"error", "The application window is not available"}};
            const std::string action = args.value("action", "");
            const auto current = vis::input::lastInjectedPointer();
            glm::vec2 position = current ? glm::vec2(current->x, current->y)
                                         : vis::input::wheelPointerInPixels(window);
            if (action == "move" || action == "down" || action == "up") {
                glm::vec2 next = position;
                if (action == "move" || args.contains("x") || args.contains("y")) {
                    if (!args.contains("x") || !args.contains("y") || !args["x"].is_number() ||
                        !args["y"].is_number())
                        return mcp::invalid_argument_result("x and y must both be numbers", "x");
                    next = {args["x"].get<float>(), args["y"].get<float>()};
                }
                if (!std::isfinite(next.x) || !std::isfinite(next.y))
                    return mcp::invalid_argument_result("x and y must be finite", "x");
                vis::input::injectPointerModifiers(pointerModifiers(args));
                vis::input::injectPointerMove(next.x, next.y);
                pushMotion(window, next, position);
                if (action != "move") {
                    const auto button = pointerButton(args);
                    if (!button)
                        return mcp::invalid_argument_result(button.error().message, "button");
                    const bool down = action == "down";
                    vis::input::injectPointerButton(*button, down);
                    pushButton(window, next, *button, down);
                }
                return json{{"success", true}, {"x", next.x}, {"y", next.y}};
            }
            if (action == "wheel" || action == "pinch") {
                const float dx = args.value("dx", 0.0f);
                const float dy = args.value("dy", 0.0f);
                if (!std::isfinite(dx) || !std::isfinite(dy))
                    return mcp::invalid_argument_result("dx and dy must be finite", "dx");
                if (args.contains("x") || args.contains("y")) {
                    if (!args.contains("x") || !args.contains("y") || !args["x"].is_number() || !args["y"].is_number())
                        return mcp::invalid_argument_result("x and y must both be numbers", "x");
                    const glm::vec2 next(args["x"].get<float>(), args["y"].get<float>());
                    if (!std::isfinite(next.x) || !std::isfinite(next.y))
                        return mcp::invalid_argument_result("x and y must be finite", "x");
                    vis::input::injectPointerMove(next.x, next.y);
                    pushMotion(window, next, position);
                    position = next;
                }
                const auto scale = vis::input::windowPixelScale(window);
                vis::input::injectPointerModifiers(pointerModifiers(args));
                SDL_Event event{};
                if (action == "pinch") {
                    const float pinch_scale = args.value("scale", 1.0f);
                    if (!std::isfinite(pinch_scale) || pinch_scale <= 0.0f)
                        return mcp::invalid_argument_result("scale must be a finite positive number", "scale");
                    event.type = SDL_EVENT_PINCH_UPDATE;
                    event.pinch.scale = pinch_scale;
                    vis::input::injectPointerWheel();
                    event.pinch.windowID = SDL_GetWindowID(window);
                    vis::input::pushInjectedPointerEvent(event);
                    return json{{"success", true}};
                }
                event.type = SDL_EVENT_MOUSE_WHEEL;
                event.wheel.windowID = SDL_GetWindowID(window);
                event.wheel.x = dx;
                event.wheel.y = dy;
                event.wheel.mouse_x = position.x / scale.x;
                event.wheel.mouse_y = position.y / scale.y;
                vis::input::injectPointerWheel();
                vis::input::pushInjectedPointerEvent(event);
                return json{{"success", true}};
            }
            return mcp::invalid_argument_result("Unknown pointer action", "action");
        }

        json injectDragOverFrames(vis::VisualizerImpl* impl, const json& args) {
            const auto from = vec2Arg(args, "from");
            const auto to = vec2Arg(args, "to");
            const auto button = pointerButton(args);
            if (!from)
                return mcp::invalid_argument_result(from.error().message, "from");
            if (!to)
                return mcp::invalid_argument_result(to.error().message, "to");
            if (!button)
                return mcp::invalid_argument_result(button.error().message, "button");

            const int steps = std::clamp(args.value("steps", 12), 1, 120);
            const SDL_Keymod modifiers = pointerModifiers(args);
            auto result = post_and_wait(impl, [impl, start = *from, button = *button, modifiers]() {
                SDL_Window* const window = impl->getWindow();
                if (!window)
                    return json{{"error", "The application window is not available"}};
                const auto current = vis::input::injectedPointer();
                const glm::vec2 previous = current ? glm::vec2(current->x, current->y)
                                                   : vis::input::wheelPointerInPixels(window);
                vis::input::injectPointerModifiers(modifiers);
                vis::input::injectPointerMove(start.x, start.y);
                pushMotion(window, start, previous);
                vis::input::injectPointerButton(button, true);
                pushButton(window, start, button, true);
                return json{{"success", true}};
            });
            if (result.contains("error"))
                return result;

            constexpr auto frame_interval = std::chrono::milliseconds(16);
            std::this_thread::sleep_for(frame_interval);
            glm::vec2 previous = *from;
            for (int step = 1; step <= steps; ++step) {
                const float t = static_cast<float>(step) / static_cast<float>(steps);
                const glm::vec2 next = *from + (*to - *from) * t;
                result = post_and_wait(impl, [impl, next, previous]() {
                    SDL_Window* const window = impl->getWindow();
                    if (!window)
                        return json{{"error", "The application window is not available"}};
                    vis::input::injectPointerMove(next.x, next.y);
                    pushMotion(window, next, previous);
                    return json{{"success", true}};
                });
                if (result.contains("error"))
                    return result;
                previous = next;
                std::this_thread::sleep_for(frame_interval);
            }

            result = post_and_wait(impl, [impl, end = *to, button = *button]() {
                SDL_Window* const window = impl->getWindow();
                if (!window)
                    return json{{"error", "The application window is not available"}};
                vis::input::injectPointerButton(button, false);
                pushButton(window, end, button, false);
                return json{{"success", true}};
            });
            if (result.contains("error"))
                return result;
            return json{{"success", true}, {"steps", steps}};
        }
    } // namespace

    void register_pointer_tool(mcp::ToolRegistry& registry, vis::Visualizer* viewer) {
        auto* const impl = dynamic_cast<vis::VisualizerImpl*>(viewer);
        if (!impl)
            return;
        registry.register_tool(
            mcp::McpTool{
                .name = "ui.pointer",
                .description = "Inject pointer input through the real SDL and polled-state path",
                .input_schema =
                    {.type = "object",
                     .properties = json{
                         {"action", json{{"type", "string"},
                                         {"enum", json::array({"move", "down", "up", "wheel", "pinch", "drag"})}}},
                         {"scale", {{"type", "number"}, {"exclusiveMinimum", 0}}},
                         {"x", json{{"type", "number"}}},
                         {"y", json{{"type", "number"}}},
                         {"dx", json{{"type", "number"}}},
                         {"dy", json{{"type", "number"}}},
                         {"button", json{{"type", "string"},
                                         {"enum", json::array({"left", "right", "middle"})}}},
                         {"from", json{{"type", "array"}, {"minItems", 2}, {"maxItems", 2}}},
                         {"to", json{{"type", "array"}, {"minItems", 2}, {"maxItems", 2}}},
                         {"steps", json{{"type", "integer"}, {"minimum", 1}, {"maximum", 120}}},
                         {"modifiers", json{{"type", "array"},
                                            {"items", json{{"type", "string"},
                                                           {"enum", json::array({"ctrl", "shift", "alt", "super"})}}}}}},
                     .required = {"action"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [impl](const json& args) -> json {
                if (args.value("action", "") == "drag")
                    return injectDragOverFrames(impl, args);
                return post_and_wait(impl, [impl, args]() { return injectPointer(*impl, args); });
            });
    }

} // namespace lfs::app
