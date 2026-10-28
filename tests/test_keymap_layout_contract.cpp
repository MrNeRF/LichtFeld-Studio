#include "input/input_bindings.hpp"
#include "input/sdl_key_mapping.hpp"
#include "keymap_history_helpers.hpp"
#include <SDL3/SDL_keyboard.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <tuple>

namespace lfs::vis::input {
    namespace {
        using KeymapTest = test::KeymapTest;
        struct LayoutKey {
            SDL_Scancode scancode = SDL_SCANCODE_UNKNOWN;
            SDL_Keycode unshifted = SDLK_UNKNOWN;
            SDL_Keycode shifted = SDLK_UNKNOWN;
        };

        std::vector<LayoutKey> readLayout(const std::filesystem::path& path) {
            const auto table = nlohmann::json::parse(std::ifstream(path));
            std::vector<LayoutKey> keys;
            for (const auto& row : table.at("rows")) {
                keys.push_back({static_cast<SDL_Scancode>(row.at("scancode").get<int>()),
                                static_cast<SDL_Keycode>(row.at("unshifted").get<int>()),
                                static_cast<SDL_Keycode>(row.at("shifted").get<int>())});
            }
            return keys;
        }

        SDL_Keycode keycodeAt(const std::vector<LayoutKey>& layout, SDL_Scancode scancode,
                              bool shifted) {
            for (const auto& row : layout) {
                if (row.scancode == scancode)
                    return shifted ? row.shifted : row.unshifted;
            }
            return SDL_GetKeyFromScancode(scancode,
                                          shifted ? SDL_KMOD_SHIFT : SDL_KMOD_NONE, false);
        }

        SDL_Scancode positionForDefaultKey(const std::vector<LayoutKey>& layout, int key) {
            // Top-row digits represent physical positions, including layouts
            // where the unshifted glyph is punctuation and Shift types a digit.
            if (key >= KEY_0 && key <= KEY_9)
                return appKeyToSdlScancode(key);

            // Letters and punctuation use the unshifted layout symbol when
            // available. For scripts with no Latin symbol, use the physical
            // position as the shortcut fallback.
            if ((key >= KEY_A && key <= KEY_Z) ||
                (key >= KEY_APOSTROPHE && key <= KEY_GRAVE_ACCENT)) {
                const auto symbol = static_cast<SDL_Keycode>(
                    key >= KEY_A && key <= KEY_Z ? key + ('a' - 'A') : key);
                for (const auto& row : layout) {
                    // The top number row is reserved for physical digit
                    // shortcuts, even if it prints punctuation here.
                    if (row.unshifted == symbol &&
                        !(row.scancode >= SDL_SCANCODE_1 &&
                          row.scancode <= SDL_SCANCODE_0))
                        return row.scancode;
                }
            }
            return appKeyToSdlScancode(key);
        }

        bool isPunctuationKey(const int key) {
            switch (key) {
            case KEY_APOSTROPHE:
            case KEY_COMMA:
            case KEY_MINUS:
            case KEY_PERIOD:
            case KEY_SLASH:
            case KEY_SEMICOLON:
            case KEY_EQUAL:
            case KEY_LEFT_BRACKET:
            case KEY_BACKSLASH:
            case KEY_RIGHT_BRACKET:
            case KEY_GRAVE_ACCENT: return true;
            default: return false;
            }
        }

        TEST_F(KeymapTest, EveryDefaultKeyDispatchesAcrossSavedLayouts) {
            const std::filesystem::path root =
                std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/keymap_layouts";
            const InputBindings bindings;
            const auto defaults = InputBindings::createDefaultProfile();
            size_t layouts = 0;
            for (const auto& entry : std::filesystem::directory_iterator(root)) {
                if (entry.path().extension() != ".json")
                    continue;
                ++layouts;
                const auto layout = readLayout(entry.path());
                ASSERT_EQ(layout.size(), 49u) << entry.path();
                for (const auto& binding : defaults.bindings) {
                    const auto* trigger = std::get_if<KeyTrigger>(&binding.trigger);
                    if (!trigger)
                        continue;
                    const auto scancode = positionForDefaultKey(layout, trigger->key);
                    ASSERT_NE(scancode, SDL_SCANCODE_UNKNOWN)
                        << entry.path() << " " << getActionName(binding.action);
                    const auto keycode = keycodeAt(
                        layout, scancode, (trigger->modifiers & MODIFIER_SHIFT) != 0);
                    const int logical = sdlShortcutKeyToAppKey(
                        scancode, keycode, keycodeAt(layout, scancode, false));
                    const int physical = sdlScancodeToAppKey(scancode);
                    if (trigger->key >= KEY_0 && trigger->key <= KEY_9) {
                        EXPECT_EQ(physical, trigger->key)
                            << "Number-row shortcuts use the physical digit position";
                        EXPECT_EQ(logical, trigger->key)
                            << "A shifted layout glyph must not change the digit shortcut";
                    } else {
                        EXPECT_TRUE(logical == trigger->key || physical == trigger->key)
                            << "Letters and punctuation must reach the layout symbol or "
                               "the physical fallback";
                    }
                    for (const auto mode : kAllToolModes) {
                        if (binding.mode != mode &&
                            (binding.mode != ToolMode::GLOBAL ||
                             !describe(binding.action).inherits_from_global))
                            continue;
                        SCOPED_TRACE(entry.path().filename().string() + " " +
                                     getActionName(binding.action) + " mode " +
                                     std::to_string(static_cast<int>(mode)));
                        // A mode binding with the same key replaces the inherited action.
                        const auto local = std::ranges::find_if(defaults.bindings, [&](const Binding& b) {
                            const auto* key = std::get_if<KeyTrigger>(&b.trigger);
                            return b.mode == mode && key && key->key == trigger->key &&
                                   key->modifiers == trigger->modifiers;
                        });
                        EXPECT_EQ(bindings.getActionForShortcut(mode, physical, logical,
                                                                trigger->modifiers),
                                  local != defaults.bindings.end() ? local->action : binding.action);
                    }
                }
            }
            EXPECT_EQ(layouts, 19u);
        }

        TEST_F(KeymapTest, PrintedCommaAndEqualReachTheirNamedActions) {
            const auto root = std::filesystem::path(PROJECT_ROOT_PATH) /
                              "tests/data/keymap_layouts";
            InputBindings bindings;
            for (const auto& [layout_name, symbol, action] : {
                     std::tuple{"ru", KEY_COMMA, Action::OPEN_PREFERENCES},
                     std::tuple{"ua", KEY_COMMA, Action::OPEN_PREFERENCES},
                     std::tuple{"de", KEY_EQUAL, Action::ZOOM_SPEED_UP},
                     std::tuple{"es", KEY_EQUAL, Action::ZOOM_SPEED_UP},
                 }) {
                SCOPED_TRACE(layout_name);
                const auto layout = readLayout(root / (std::string(layout_name) + ".json"));
                const auto row = std::ranges::find_if(layout, [&](const LayoutKey& key) {
                    return key.shifted == static_cast<SDL_Keycode>(symbol) &&
                           key.unshifted != static_cast<SDL_Keycode>(symbol);
                });
                ASSERT_NE(row, layout.end()) << "The printed symbol must require Shift";
                const int physical = sdlScancodeToAppKey(row->scancode);
                const int logical = sdlShortcutKeyToAppKey(
                    row->scancode, row->shifted, row->unshifted);
                const int printed = sdlPrintedPunctuationToAppKey(row->shifted);
                ASSERT_EQ(printed, symbol);
                EXPECT_EQ(bindings.getActionForShortcut(
                              ToolMode::GLOBAL, physical, logical,
                              MODIFIER_CTRL | MODIFIER_SHIFT, printed, true),
                          action);
            }
        }

        TEST_F(KeymapTest, PieMenuUsesPhysicalTopLeftKeyAcrossLayouts) {
            const auto root = std::filesystem::path(PROJECT_ROOT_PATH) /
                              "tests/data/keymap_layouts";
            const InputBindings bindings;
            for (const auto* layout_name : {"it", "pt", "cz", "ua"}) {
                SCOPED_TRACE(layout_name);
                const auto layout = readLayout(root / (std::string(layout_name) + ".json"));
                const auto row = std::ranges::find_if(layout, [](const LayoutKey& key) {
                    return key.scancode == SDL_SCANCODE_GRAVE;
                });
                ASSERT_NE(row, layout.end());
                const int physical = sdlScancodeToAppKey(row->scancode);
                const int logical = sdlShortcutKeyToAppKey(
                    row->scancode, row->unshifted, row->unshifted);
                const int printed = sdlPrintedPunctuationToAppKey(row->unshifted);
                EXPECT_EQ(physical, KEY_GRAVE_ACCENT);
                EXPECT_EQ(bindings.getActionForShortcut(
                              ToolMode::GLOBAL, physical, logical, MODIFIER_NONE, printed),
                          Action::PIE_MENU);
            }
        }

        TEST_F(KeymapTest, PrintedSymbolsOnDigitRowsBeatPhysicalFallback) {
            InputBindings bindings;
            const auto defaults = InputBindings::createDefaultProfile();
            const auto root = std::filesystem::path(PROJECT_ROOT_PATH) /
                              "tests/data/keymap_layouts";

            for (const auto* layout_name : {"fr", "be", "de", "es", "us"}) {
                const auto layout = readLayout(root / (std::string(layout_name) + ".json"));
                SCOPED_TRACE(layout_name);
                size_t checked = 0;
                for (const auto& binding : defaults.bindings) {
                    const auto* trigger = std::get_if<KeyTrigger>(&binding.trigger);
                    if (!trigger)
                        continue;
                    const bool punctuation = isPunctuationKey(trigger->key);
                    const bool digit = trigger->key >= KEY_0 && trigger->key <= KEY_9;
                    if (!punctuation && !digit)
                        continue;

                    for (const auto& row : layout) {
                        const bool number_row = row.scancode >= SDL_SCANCODE_1 &&
                                                row.scancode <= SDL_SCANCODE_0;
                        if (digit && !number_row)
                            continue;
                        for (const bool shifted : {false, true}) {
                            const auto keycode = shifted ? row.shifted : row.unshifted;
                            const int printed = sdlPrintedKeyToAppKey(keycode);
                            if (printed != trigger->key)
                                continue;
                            const int physical = sdlScancodeToAppKey(row.scancode);
                            const int modifiers = trigger->modifiers |
                                                  (shifted ? MODIFIER_SHIFT : MODIFIER_NONE);
                            const bool shift_needed = shifted && row.shifted != row.unshifted;
                            Action expected = binding.action;
                            // On DE/ES, Shift+0 produces '='. The explicit
                            // Ctrl+Shift+= binding wins over Ctrl+= because its
                            // complete held modifier chord matches exactly.
                            if (layout_name != std::string_view("fr") &&
                                layout_name != std::string_view("be") &&
                                layout_name != std::string_view("us") &&
                                binding.action == Action::CAMERA_SPEED_UP &&
                                row.scancode == SDL_SCANCODE_0 && shifted) {
                                expected = Action::ZOOM_SPEED_UP;
                            }
                            EXPECT_EQ(bindings.getActionForShortcut(
                                          binding.mode, physical, physical, modifiers, printed,
                                          shift_needed),
                                      expected)
                                << getActionName(binding.action) << " on scancode "
                                << static_cast<int>(row.scancode)
                                << (shifted ? " shifted" : " unshifted");
                            ++checked;
                        }
                    }
                }
                EXPECT_GT(checked, 0u);
            }

            for (const auto* layout_name : {"fr", "be"}) {
                const auto layout = readLayout(root / (std::string(layout_name) + ".json"));
                const auto row = std::ranges::find_if(layout, [](const LayoutKey& key) {
                    return key.scancode == SDL_SCANCODE_1;
                });
                ASSERT_NE(row, layout.end());
                const int printed = sdlPrintedKeyToAppKey(row->unshifted);
                ASSERT_NE(printed, KEY_1) << layout_name;
                EXPECT_EQ(bindings.getActionForShortcut(
                              ToolMode::SELECTION, KEY_1, KEY_1, MODIFIER_CTRL, printed, false),
                          Action::SELECT_MODE_CENTERS)
                    << layout_name << " unshifted first-row key falls back to physical Ctrl+1";
            }
        }
    } // namespace
} // namespace lfs::vis::input
