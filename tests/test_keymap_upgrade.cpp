#include "core/error_bus.hpp"
#include "input/input_bindings.hpp"
#include "input/keymap_profile_overrides.hpp"
#include "keymap_history_helpers.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iterator>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string_view>

namespace lfs::vis::input {
    namespace {
        using KeymapTest = test::KeymapTest;
        namespace fs = std::filesystem;
        using json = nlohmann::json;
        std::string bindingDigest(const json& bindings) {
            std::uint64_t digest = 14695981039346656037ull;
            for (const unsigned char byte : bindings.dump())
                digest = (digest ^ byte) * 1099511628211ull;
            std::ostringstream output;
            output << std::hex << std::setfill('0') << std::setw(16) << digest;
            return output.str();
        }

        TEST_F(KeymapTest, CurrentDefaultsHaveARecordedVersion) {
            InputBindings current;
            const auto saved = fs::temp_directory_path() / "lfs_keymap_recorded_defaults.json";
            ASSERT_TRUE(current.saveProfileToFile(saved));
            const auto actual = json::parse(std::ifstream(saved));
            fs::remove(saved);
            const fs::path baseline = fs::path(PROJECT_ROOT_PATH) /
                                      "tests/data/keymap_contract" /
                                      ("current_v" + std::to_string(actual.at("version").get<int>()) +
                                       ".digest");
            ASSERT_TRUE(fs::exists(baseline))
                << "A new keymap profile version needs the previous defaults. Run: "
                   "python3 tools/generate_keymap_defaults.py --record-previous HEAD";
            std::ifstream digest_file(baseline);
            std::string version_label;
            int recorded_version = -1;
            std::string digest_label;
            std::string recorded_digest;
            digest_file >> version_label >> recorded_version >> digest_label >> recorded_digest;
            EXPECT_EQ(version_label, "version");
            EXPECT_EQ(digest_label, "fnv1a64");
            EXPECT_EQ(actual.at("version"), recorded_version)
                << "A new keymap profile version needs the previous defaults. Run: "
                   "python3 tools/generate_keymap_defaults.py --record-previous HEAD";
            EXPECT_EQ(bindingDigest(actual.at("bindings")), recorded_digest)
                << "Default bindings changed without recording the previous profile version. "
                   "Bump PROFILE_VERSION, then run: "
                   "python3 tools/generate_keymap_defaults.py --record-previous HEAD";
        }

        TEST_F(KeymapTest, SidebarCategoriesUpgradeAndPreserveRebindingAndRemoval) {
            const auto path = fs::temp_directory_path() / "lfs_keymap_sidebar_upgrade.json";
            auto old = test::historicalSavedProfile("v33");
            std::erase_if(old["bindings"].get_ref<json::array_t&>(), [](const json& binding) {
                return binding.value("action", -1) == static_cast<int>(Action::TOGGLE_UI);
            });
            test::writeProfile(path, old);
            InputBindings upgraded;
            ASSERT_TRUE(upgraded.loadProfileFromFile(path));
            EXPECT_EQ(upgraded.getActionForKey(ToolMode::GLOBAL, KEY_TAB, MODIFIER_CTRL),
                      Action::NEXT_SIDEBAR_CATEGORY);
            EXPECT_EQ(upgraded.getActionForKey(ToolMode::GLOBAL, KEY_TAB, MODIFIER_CTRL | MODIFIER_SHIFT),
                      Action::PREVIOUS_SIDEBAR_CATEGORY);
            EXPECT_EQ(upgraded.getActionForScroll(ToolMode::GLOBAL, MODIFIER_CTRL),
                      Action::HISTOGRAM_ZOOM_MARKED);
            EXPECT_EQ(upgraded.getActionForScroll(ToolMode::SELECTION, MODIFIER_CTRL),
                      Action::BRUSH_RESIZE);
            upgraded.setBinding(ToolMode::GLOBAL, Action::NEXT_SIDEBAR_CATEGORY, KeyTrigger{KEY_F8});
            upgraded.clearBinding(ToolMode::GLOBAL, Action::PREVIOUS_SIDEBAR_CATEGORY);
            ASSERT_TRUE(upgraded.saveProfileToFile(path));
            InputBindings reloaded;
            ASSERT_TRUE(reloaded.loadProfileFromFile(path));
            EXPECT_EQ(reloaded.getActionForKey(ToolMode::GLOBAL, KEY_F8, 0),
                      Action::NEXT_SIDEBAR_CATEGORY);
            EXPECT_EQ(reloaded.getActionForKey(ToolMode::GLOBAL, KEY_TAB, MODIFIER_CTRL), Action::NONE);
            EXPECT_EQ(reloaded.getActionForKey(ToolMode::GLOBAL, KEY_TAB, MODIFIER_CTRL | MODIFIER_SHIFT),
                      Action::NONE);
            EXPECT_FALSE(reloaded.getTriggerForAction(Action::TOGGLE_UI));
            fs::remove(path);
        }

        TEST_F(KeymapTest, AlignPointControlsUpgradeAndPreserveRebinding) {
            const auto path = fs::temp_directory_path() / "lfs_keymap_align_controls_upgrade.json";
            test::writeProfile(path, test::historicalSavedProfile("v34"));
            InputBindings upgraded;
            ASSERT_TRUE(upgraded.loadProfileFromFile(path));
            EXPECT_EQ(upgraded.getActionForKey(ToolMode::ALIGN, KEY_BACKSPACE, 0), Action::REMOVE_LAST_POINT);
            EXPECT_EQ(upgraded.getActionForKey(ToolMode::ALIGN, KEY_ENTER, 0), Action::APPLY_ALIGNMENT);
            EXPECT_EQ(upgraded.getActionForKey(ToolMode::ALIGN, KEY_KP_ENTER, 0), Action::APPLY_ALIGNMENT);
            EXPECT_EQ(upgraded.getActionForKey(ToolMode::ALIGN, KEY_DELETE, 0), Action::DELETE_SELECTED);
            EXPECT_EQ(upgraded.getActionForKey(ToolMode::ALIGN, KEY_ESCAPE, 0), Action::CANCEL_POLYGON);
            EXPECT_EQ(upgraded.getActionForKey(ToolMode::SELECTION, KEY_ENTER, 0), Action::CONFIRM_POLYGON);
            EXPECT_EQ(upgraded.getActionForKey(ToolMode::CROP_BOX, KEY_ENTER, 0), Action::APPLY_CROP_BOX);
            upgraded.setBinding(ToolMode::ALIGN, Action::REMOVE_LAST_POINT, KeyTrigger{KEY_F9});
            ASSERT_TRUE(upgraded.saveProfileToFile(path));
            InputBindings reloaded;
            ASSERT_TRUE(reloaded.loadProfileFromFile(path));
            EXPECT_EQ(reloaded.getActionForKey(ToolMode::ALIGN, KEY_F9, 0), Action::REMOVE_LAST_POINT);
            EXPECT_EQ(reloaded.getActionForKey(ToolMode::ALIGN, KEY_BACKSPACE, 0), Action::NONE);
            fs::remove(path);
        }

        TEST_F(KeymapTest, CropPolygonVertexClicksUpgradeAndPreserveRebinding) {
            InputBindings defaults;
            EXPECT_EQ(defaults.getActionForMouseButton(ToolMode::CROP_BOX, MouseButton::LEFT, MODIFIER_SHIFT),
                      Action::INSERT_CROP_POLYGON_VERTEX);
            EXPECT_EQ(defaults.getActionForMouseButton(ToolMode::CROP_BOX, MouseButton::LEFT, MODIFIER_CTRL),
                      Action::REMOVE_CROP_POLYGON_VERTEX);

            const auto path = fs::temp_directory_path() / "lfs_keymap_crop_polygon_vertex_upgrade.json";
            test::writeProfile(path, test::historicalSavedProfile("v35"));
            InputBindings upgraded;
            ASSERT_TRUE(upgraded.loadProfileFromFile(path));
            EXPECT_EQ(upgraded.getActionForMouseButton(ToolMode::CROP_BOX, MouseButton::LEFT, MODIFIER_SHIFT),
                      Action::INSERT_CROP_POLYGON_VERTEX);
            EXPECT_EQ(upgraded.getActionForMouseButton(ToolMode::CROP_BOX, MouseButton::LEFT, MODIFIER_CTRL),
                      Action::REMOVE_CROP_POLYGON_VERTEX);
            EXPECT_EQ(upgraded.getActionForKey(ToolMode::CROP_BOX, KEY_ENTER, 0), Action::APPLY_CROP_BOX);
            EXPECT_EQ(upgraded.getActionForDrag(ToolMode::SELECTION, MouseButton::LEFT, MODIFIER_SHIFT),
                      Action::SELECTION_ADD);
            upgraded.setBinding(ToolMode::CROP_BOX, Action::REMOVE_CROP_POLYGON_VERTEX,
                                MouseButtonTrigger{MouseButton::LEFT, MODIFIER_ALT});
            ASSERT_TRUE(upgraded.saveProfileToFile(path));
            InputBindings reloaded;
            ASSERT_TRUE(reloaded.loadProfileFromFile(path));
            EXPECT_EQ(reloaded.getActionForMouseButton(ToolMode::CROP_BOX, MouseButton::LEFT, MODIFIER_ALT),
                      Action::REMOVE_CROP_POLYGON_VERTEX);
            EXPECT_EQ(reloaded.getActionForMouseButton(ToolMode::CROP_BOX, MouseButton::LEFT, MODIFIER_CTRL),
                      Action::NONE);
            EXPECT_EQ(reloaded.getActionForMouseButton(ToolMode::CROP_BOX, MouseButton::LEFT, MODIFIER_SHIFT),
                      Action::INSERT_CROP_POLYGON_VERTEX);
            fs::remove(path);
        }

        TEST_F(KeymapTest, EditorCommandsUpgradeFromHistoricalDefaultsAndKeepOverrides) {
            const auto path = fs::temp_directory_path() / "lfs_keymap_editor_upgrade.json";
            auto old = test::historicalSavedProfile("v39");
            test::writeProfile(path, old);
            InputBindings upgraded;
            ASSERT_TRUE(upgraded.loadProfileFromFile(path));
            EXPECT_EQ(upgraded.getActionForKey(ToolMode::EDITOR, KEY_F5, 0), Action::EDITOR_RUN);
            EXPECT_EQ(upgraded.getActionForKey(ToolMode::EDITOR, KEY_ENTER, MODIFIER_CTRL), Action::EDITOR_RUN);
            EXPECT_EQ(upgraded.getActionForKey(ToolMode::EDITOR, KEY_C, MODIFIER_CTRL), Action::EDITOR_STOP);
            EXPECT_FALSE(upgraded.getTriggerForAction(Action::EDITOR_TOGGLE_KIND, ToolMode::EDITOR));
            ASSERT_TRUE(upgraded.saveProfileToFile(path));
            InputBindings current;
            ASSERT_TRUE(current.loadProfileFromFile(path));
            EXPECT_EQ(current.getActionForKey(ToolMode::EDITOR, KEY_F5, 0), Action::EDITOR_RUN);
            EXPECT_EQ(current.getActionForKey(ToolMode::EDITOR, KEY_ENTER, MODIFIER_CTRL), Action::EDITOR_RUN);
            EXPECT_EQ(current.getActionForKey(ToolMode::EDITOR, KEY_KP_ENTER, MODIFIER_CTRL), Action::EDITOR_RUN);
            EXPECT_EQ(current.getActionForKey(ToolMode::EDITOR, KEY_C, MODIFIER_CTRL), Action::EDITOR_STOP);
            upgraded.setBinding(ToolMode::EDITOR, Action::EDITOR_RUN, KeyTrigger{KEY_F9});
            upgraded.setBinding(ToolMode::EDITOR, Action::EDITOR_TOGGLE_KIND, KeyTrigger{KEY_F8});
            upgraded.clearBinding(ToolMode::EDITOR, Action::EDITOR_STOP);
            ASSERT_TRUE(upgraded.saveProfileToFile(path));
            InputBindings reloaded;
            ASSERT_TRUE(reloaded.loadProfileFromFile(path));
            EXPECT_EQ(reloaded.getActionForKey(ToolMode::EDITOR, KEY_F9, 0), Action::EDITOR_RUN);
            EXPECT_EQ(reloaded.getActionForKey(ToolMode::EDITOR, KEY_F8, 0), Action::EDITOR_TOGGLE_KIND);
            EXPECT_NE(reloaded.getActionForKey(ToolMode::EDITOR, KEY_ENTER, MODIFIER_CTRL), Action::EDITOR_RUN);
            EXPECT_FALSE(reloaded.getTriggerForAction(Action::EDITOR_STOP, ToolMode::EDITOR));
            EXPECT_EQ(reloaded.getActionForKey(ToolMode::GLOBAL, KEY_F5, 0), Action::ASSET_REFRESH);
            fs::remove(path);
        }

        TEST_F(KeymapTest, ScaleDistanceUpgradeAndPreserveRebindingAndRemoval) {
            const auto path = fs::temp_directory_path() / "lfs_keymap_scale_distance_upgrade.json";
            auto old = test::historicalSavedProfile("v37");
            std::erase_if(old["bindings"].get_ref<json::array_t&>(), [](const json& binding) {
                return binding.value("action", -1) == static_cast<int>(Action::TOGGLE_UI);
            });
            test::writeProfile(path, old);
            InputBindings upgraded;
            ASSERT_TRUE(upgraded.loadProfileFromFile(path));
            EXPECT_EQ(upgraded.getActionForKey(ToolMode::GLOBAL, KEY_7, 0),
                      Action::TOOL_SCALE_DISTANCE);
            upgraded.setBinding(ToolMode::GLOBAL, Action::TOOL_SCALE_DISTANCE, KeyTrigger{KEY_F9});
            ASSERT_TRUE(upgraded.saveProfileToFile(path));
            InputBindings reloaded;
            ASSERT_TRUE(reloaded.loadProfileFromFile(path));
            EXPECT_EQ(reloaded.getActionForKey(ToolMode::GLOBAL, KEY_F9, 0),
                      Action::TOOL_SCALE_DISTANCE);
            EXPECT_EQ(reloaded.getActionForKey(ToolMode::GLOBAL, KEY_7, 0), Action::NONE);
            EXPECT_FALSE(reloaded.getTriggerForAction(Action::TOGGLE_UI));
            fs::remove(path);
        }

        TEST_F(KeymapTest, StaleFullListDoesNotOverrideFutureDefaults) {
            const fs::path path = fs::temp_directory_path() / "lfs_keymap_future_defaults.json";
            InputBindings current;
            ASSERT_TRUE(current.saveProfileToFile(path));
            auto saved = json::parse(std::ifstream(path));
            for (auto& binding : saved["bindings"]) {
                if (binding.value("action", -1) == static_cast<int>(Action::CAMERA_ORBIT)) {
                    binding["button"] = static_cast<int>(MouseButton::RIGHT);
                }
            }
            test::writeProfile(path, saved);
            InputBindings loaded;
            ASSERT_TRUE(loaded.loadProfileFromFile(path));
            EXPECT_EQ(loaded.getActionForDrag(ToolMode::SELECTION, MouseButton::MIDDLE, 0),
                      Action::CAMERA_ORBIT);
            EXPECT_EQ(loaded.getActionForDrag(ToolMode::SELECTION, MouseButton::RIGHT, 0),
                      Action::CAMERA_PAN);
            fs::remove(path);
        }

        TEST_F(KeymapTest, RemovingOneParallelDefaultKeepsItsNeighborInherited) {
            const fs::path path = fs::temp_directory_path() / "lfs_keymap_parallel_default.json";
            auto saved = test::historicalSavedProfile("v0.5.3");
            bool removed_shift = false;
            for (auto it = saved["bindings"].begin(); it != saved["bindings"].end();) {
                if (it->value("description", "") == "Brush size" &&
                    it->value("trigger_type", "") == "scroll" &&
                    it->value("modifiers", -1) == MODIFIER_SHIFT) {
                    it = saved["bindings"].erase(it);
                    removed_shift = true;
                } else {
                    ++it;
                }
            }
            ASSERT_TRUE(removed_shift);
            test::writeProfile(path, saved);
            InputBindings loaded;
            ASSERT_TRUE(loaded.loadProfileFromFile(path));
            ASSERT_TRUE(loaded.saveProfileToFile(path));
            const auto upgraded = json::parse(std::ifstream(path));
            EXPECT_EQ(upgraded.at("version"), test::currentProfileVersion());
            size_t brush_additions = 0;
            size_t brush_removals = 0;
            for (const auto& addition : upgraded["overrides"]["bindings"])
                brush_additions += addition.value("action", -1) ==
                                   static_cast<int>(Action::BRUSH_RESIZE);
            for (const auto& removal : upgraded["overrides"]["removed"])
                brush_removals += removal.value("action", -1) ==
                                  static_cast<int>(Action::BRUSH_RESIZE);
            EXPECT_EQ(brush_additions, 0u);
            EXPECT_EQ(brush_removals, 1u);
            InputBindings reloaded;
            ASSERT_TRUE(reloaded.loadProfileFromFile(path));
            EXPECT_EQ(reloaded.getActionForScroll(ToolMode::SELECTION, MODIFIER_CTRL),
                      Action::BRUSH_RESIZE);
            EXPECT_EQ(reloaded.getActionForScroll(ToolMode::SELECTION, MODIFIER_SHIFT),
                      Action::NONE);
            fs::remove(path);
        }

        TEST_F(KeymapTest, ClearingOneDefaultWritesAnExplicitTombstone) {
            const fs::path path = fs::temp_directory_path() / "lfs_keymap_clear_tombstone.json";
            InputBindings bindings;
            bindings.clearBinding(ToolMode::GLOBAL, Action::TOOL_SCALE);
            ASSERT_TRUE(bindings.saveProfileToFile(path));
            auto saved = json::parse(std::ifstream(path));
            const auto& removed = saved.at("overrides").at("removed");
            EXPECT_TRUE(std::ranges::any_of(removed, [](const json& item) {
                return item.value("mode", -1) == static_cast<int>(ToolMode::GLOBAL) &&
                       item.value("action", -1) == static_cast<int>(Action::TOOL_SCALE);
            }));
            // The override record, rather than the compatibility full list,
            // defines the effective binding when this file is loaded again.
            saved["bindings"].push_back({{"mode", static_cast<int>(ToolMode::GLOBAL)},
                                         {"action", static_cast<int>(Action::TOOL_SCALE)},
                                         {"description", "Scale"},
                                         {"trigger_type", "key"},
                                         {"key", KEY_4},
                                         {"modifiers", 0}});
            test::writeProfile(path, saved);
            InputBindings reloaded;
            ASSERT_TRUE(reloaded.loadProfileFromFile(path));
            EXPECT_EQ(reloaded.getActionForKey(ToolMode::GLOBAL, KEY_4, 0), Action::NONE);
            fs::remove(path);
        }

        TEST_F(KeymapTest, HistoricalRemovalAndInsertedDefaultSurviveLoadSaveReload) {
            const fs::path path = fs::temp_directory_path() /
                                  "lfs_keymap_historical_simulated_defaults.json";
            auto old_profile = test::historicalSavedProfile("v0.5.4");
            const auto removed_old_default = std::ranges::find_if(
                old_profile["bindings"], [](const json& binding) {
                    return binding.value("description", "") == "Scale" &&
                           binding.value("trigger_type", "") == "key" &&
                           binding.value("key", -1) == KEY_4;
                });
            ASSERT_NE(removed_old_default, old_profile["bindings"].end());
            old_profile["bindings"].erase(removed_old_default);
            test::writeProfile(path, old_profile);

            auto simulated_defaults = InputBindings::createDefaultProfile();
            const Binding inserted{ToolMode::GLOBAL, KeyTrigger{KEY_F13},
                                   Action::TOOL_SCALE, getActionName(Action::TOOL_SCALE)};
            const auto original_scale = std::ranges::find_if(
                simulated_defaults.bindings, [](const Binding& binding) {
                    const auto* key = std::get_if<KeyTrigger>(&binding.trigger);
                    return binding.mode == ToolMode::GLOBAL &&
                           binding.action == Action::TOOL_SCALE && key && key->key == KEY_4;
                });
            ASSERT_NE(original_scale, simulated_defaults.bindings.end());
            simulated_defaults.bindings.insert(original_scale, inserted);

            const auto expect_upgraded_profile = [&](InputBindings& bindings) {
                EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_F13, 0),
                          Action::TOOL_SCALE);
                EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_4, 0), Action::NONE);
            };
            {
                detail::ScopedDefaultProfileOverride defaults(simulated_defaults);
                InputBindings loaded;
                ASSERT_TRUE(loaded.loadProfileFromFile(path));
                expect_upgraded_profile(loaded);
                ASSERT_TRUE(loaded.saveProfileToFile(path));

                InputBindings reloaded;
                ASSERT_TRUE(reloaded.loadProfileFromFile(path));
                expect_upgraded_profile(reloaded);
            }
            fs::remove(path);
        }

        class NewerProfileWarning final : public lfs::NativeErrorConsumer {
        public:
            void on_error(const lfs::ErrorNotification& notification,
                          const lfs::ErrorDeliveryInfo&) noexcept override {
                if (notification.error.code() == lfs::ErrorCode::Unsupported &&
                    notification.error.domain() == lfs::ErrorDomain::App)
                    count.fetch_add(1, std::memory_order_relaxed);
            }
            std::atomic<int> count{0};
        };

        TEST_F(KeymapTest, UnknownNewerProfileStaysUntouched) {
            NewerProfileWarning warnings;
            auto warning_subscription = lfs::ErrorBus::instance().subscribe(warnings);
            namespace fs = std::filesystem;
            const fs::path dir = fs::temp_directory_path() / "lfs_future_keymap_guard";
            fs::remove_all(dir);
            fs::create_directories(dir);
            const fs::path path = dir / "Default.json";
            const fs::path other = dir / "Export.json";
            for (const std::string_view version : {"999", "2147483648"}) {
                const std::string original = "{\"name\":\"From a newer build\",\"version\":" +
                                             std::string(version) + R"(,"future_field":{"keep":true},"bindings":[
                    {"mode":0,"action":987654,"description":"Future action",
                    "trigger_type":"key","key":83,"modifiers":0,"on_repeat":false}]})";
                {
                    std::ofstream output(path);
                    output << original;
                }
                InputBindings bindings;
                EXPECT_FALSE(bindings.loadProfileFromFile(path)) << version;
                EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_4, MODIFIER_NONE),
                          Action::TOOL_SCALE);
                EXPECT_EQ(bindings.getActionForDrag(ToolMode::SELECTION, MouseButton::MIDDLE, 0),
                          Action::CAMERA_ORBIT);
                EXPECT_FALSE(bindings.saveProfileToFile(path)) << version;
                EXPECT_TRUE(bindings.saveProfileToFile(other));
                std::ifstream input(path);
                const std::string actual{std::istreambuf_iterator<char>(input),
                                         std::istreambuf_iterator<char>()};
                EXPECT_EQ(actual, original) << version;
            }
            EXPECT_EQ(warnings.count.load(std::memory_order_relaxed), 4);
            for (const std::string_view version : {"1e3", "-1"}) {
                const std::string original = "{\"name\":\"Malformed version\",\"version\":" +
                                             std::string(version) + ",\"bindings\":[]}";
                {
                    std::ofstream output(path);
                    output << original;
                }
                InputBindings bindings;
                EXPECT_FALSE(bindings.loadProfileFromFile(path)) << version;
                ASSERT_TRUE(bindings.saveProfileToFile(path)) << version;
                bool preserved = false;
                for (const auto& entry : fs::directory_iterator(dir)) {
                    if (entry.path().filename().string().starts_with("Default.json.damaged-")) {
                        std::ifstream backup(entry.path());
                        const std::string actual{std::istreambuf_iterator<char>(backup),
                                                 std::istreambuf_iterator<char>()};
                        preserved = actual == original;
                        backup.close();
                        fs::remove(entry.path());
                    }
                }
                EXPECT_TRUE(preserved) << version;
            }
            EXPECT_EQ(warnings.count.load(std::memory_order_relaxed), 4);
            fs::remove_all(dir);
        }

        TEST_F(KeymapTest, EarlyReleaseScaleAndHomeKeepTheirMeaningAfterRebind) {
            namespace fs = std::filesystem;
            const fs::path path = fs::temp_directory_path() / "lfs_early_keymap_rebind.json";
            struct RebindCase {
                std::string_view description;
                int old_key;
                int new_key;
                Action action;
            };
            constexpr std::array rebinds = {
                RebindCase{"Scale", KEY_4, KEY_F9, Action::TOOL_SCALE},
                RebindCase{"Home", KEY_H, KEY_F8, Action::CAMERA_RESET_HOME},
            };
            for (const std::string_view release : {"v0.5.0", "v0.5.1", "v0.5.2"}) {
                SCOPED_TRACE(std::string(release));
                auto saved = test::historicalSavedProfile(release);
                for (const auto& rebind : rebinds) {
                    const auto binding = std::ranges::find_if(
                        saved["bindings"], [&](const json& entry) {
                            return entry.value("description", "") == rebind.description &&
                                   entry.value("trigger_type", "") == "key" &&
                                   entry.value("key", -1) == rebind.old_key &&
                                   entry.value("modifiers", -1) == 0;
                        });
                    ASSERT_NE(binding, saved["bindings"].end()) << rebind.description;
                    (*binding)["key"] = rebind.new_key;
                }
                for (auto& binding : saved["bindings"])
                    binding.erase("action_symbol");
                test::writeProfile(path, saved);
                InputBindings bindings;
                ASSERT_TRUE(bindings.loadProfileFromFile(path));
                for (const auto& rebind : rebinds)
                    EXPECT_EQ(bindings.getActionForKey(ToolMode::SELECTION, rebind.new_key, 0),
                              rebind.action)
                        << rebind.description;
                ASSERT_TRUE(bindings.saveProfileToFile(path));
                InputBindings reloaded;
                ASSERT_TRUE(reloaded.loadProfileFromFile(path));
                for (const auto& rebind : rebinds)
                    EXPECT_EQ(reloaded.getActionForKey(ToolMode::SELECTION, rebind.new_key, 0),
                              rebind.action)
                        << rebind.description;
            }
            fs::remove(path);
        }

        TEST_F(KeymapTest, OldActionIdTableAlsoCoversUserAddedNonDefaultAction) {
            namespace fs = std::filesystem;
            const fs::path path = fs::temp_directory_path() / "lfs_early_nondefault_action.json";
            auto saved = test::historicalSavedProfile("v0.5.0");
            saved["bindings"].push_back({{"mode", static_cast<int>(ToolMode::SELECTION)},
                                         {"action", 37}, // BRUSH_RESIZE in this release.
                                         {"description", "Resize Brush"},
                                         {"trigger_type", "scroll"},
                                         {"modifiers", MODIFIER_CTRL | MODIFIER_SHIFT | MODIFIER_ALT}});
            test::writeProfile(path, saved);
            InputBindings bindings;
            ASSERT_TRUE(bindings.loadProfileFromFile(path));
            EXPECT_EQ(bindings.getActionForScroll(
                          ToolMode::SELECTION,
                          MODIFIER_CTRL | MODIFIER_SHIFT | MODIFIER_ALT),
                      Action::BRUSH_RESIZE);
            fs::remove(path);
        }

        void expectHistoricalCoreBindings(const InputBindings& loaded) {
            for (const auto mode : kAllToolModes)
                EXPECT_EQ(loaded.getActionForDrag(mode, MouseButton::MIDDLE, 0),
                          Action::CAMERA_ORBIT);
            EXPECT_EQ(loaded.getActionForDrag(ToolMode::SELECTION, MouseButton::LEFT, 0),
                      Action::SELECTION_REPLACE);
            EXPECT_EQ(loaded.getActionForScroll(ToolMode::SELECTION, MODIFIER_CTRL),
                      Action::BRUSH_RESIZE);
            EXPECT_EQ(loaded.getActionForKey(ToolMode::SELECTION, KEY_DELETE, 0),
                      Action::DELETE_SELECTED);
            for (const auto mode : {ToolMode::TRANSLATE, ToolMode::ROTATE, ToolMode::SCALE}) {
                EXPECT_EQ(loaded.getActionForMouseButton(mode, MouseButton::LEFT, 0, false),
                          Action::NODE_PICK);
                EXPECT_EQ(loaded.getActionForDrag(mode, MouseButton::LEFT, 0),
                          Action::NODE_RECT_SELECT);
            }
        }

        TEST_F(KeymapTest, HistoricalDefaultsAndUserChangesSurvive) {
            const auto root = std::filesystem::temp_directory_path() / "lfs_keymap_history_regression";
            test::ScopedKeymapHome environment(root);
            const auto output = root / "defaults.json";
            InputBindings defaults;
            ASSERT_TRUE(defaults.saveProfileToFile(output));
            const auto expected = nlohmann::json::parse(std::ifstream(output))["bindings"];
            for (const auto& snapshot : test::historicalProfiles()) {
                SCOPED_TRACE(snapshot.at("name").get<std::string>());
                const auto path = root / "historical.json";
                test::writeProfile(
                    path, test::historicalSavedProfile(snapshot.at("name").get<std::string>()));
                InputBindings loaded;
                ASSERT_TRUE(loaded.loadProfileFromFile(path));
                ASSERT_TRUE(loaded.saveProfileToFile(output));
                EXPECT_EQ(nlohmann::json::parse(std::ifstream(output))["bindings"], expected);
                expectHistoricalCoreBindings(loaded);
            }

            const auto path = root / "customized.json";
            auto saved = test::historicalSavedProfile("v0.5.4");
            bool rebound = false;
            for (auto it = saved["bindings"].begin(); it != saved["bindings"].end();) {
                if (it->value("mode", -1) == static_cast<int>(ToolMode::SELECTION) &&
                    it->value("action", -1) == static_cast<int>(Action::DELETE_SELECTED)) {
                    it = saved["bindings"].erase(it);
                    continue;
                }
                if (it->value("mode", -1) == static_cast<int>(ToolMode::GLOBAL) &&
                    it->value("action", -1) == static_cast<int>(Action::CAMERA_ORBIT)) {
                    (*it)["modifiers"] = MODIFIER_ALT;
                    rebound = true;
                }
                ++it;
            }
            ASSERT_TRUE(rebound);
            saved["bindings"].push_back({{"mode", static_cast<int>(ToolMode::SELECTION)},
                                         {"action", static_cast<int>(Action::TOGGLE_UI)},
                                         {"description", getActionName(Action::TOGGLE_UI)},
                                         {"trigger_type", "key"},
                                         {"key", KEY_F9},
                                         {"modifiers", 0},
                                         {"on_repeat", false}});
            test::writeProfile(path, saved);
            InputBindings loaded;
            ASSERT_TRUE(loaded.loadProfileFromFile(path));
            EXPECT_EQ(loaded.getActionForDrag(ToolMode::SELECTION, MouseButton::MIDDLE, 0),
                      Action::NONE);
            EXPECT_EQ(loaded.getActionForDrag(ToolMode::SELECTION, MouseButton::MIDDLE, MODIFIER_ALT),
                      Action::CAMERA_ORBIT);
            EXPECT_EQ(loaded.getActionForKey(ToolMode::SELECTION, KEY_DELETE, 0), Action::NONE);
            EXPECT_EQ(loaded.getActionForKey(ToolMode::SELECTION, KEY_F9, 0), Action::TOGGLE_UI);
        }

        TEST_F(KeymapTest, EarlyReleaseRebindKeepsItsActionIdentity) {
            const auto root = std::filesystem::temp_directory_path() / "lfs_keymap_early_rebind";
            test::ScopedKeymapHome environment(root);
            const auto path = root / "profile.json";
            auto saved = test::historicalSavedProfile("v0.5.0");
            bool changed = false;
            for (auto& binding : saved["bindings"]) {
                if (binding.value("mode", -1) == static_cast<int>(ToolMode::SELECTION) &&
                    binding.value("description", "") == "Select" &&
                    binding.value("trigger_type", "") == "drag" &&
                    binding.value("button", -1) == static_cast<int>(MouseButton::LEFT) &&
                    binding.value("modifiers", -1) == 0) {
                    binding["modifiers"] = MODIFIER_SUPER;
                    changed = true;
                }
            }
            ASSERT_TRUE(changed);
            test::writeProfile(path, saved);
            InputBindings loaded;
            ASSERT_TRUE(loaded.loadProfileFromFile(path));
            EXPECT_EQ(loaded.getActionForDrag(ToolMode::SELECTION, MouseButton::LEFT, MODIFIER_SUPER),
                      Action::SELECTION_REPLACE);
        }

        TEST_F(KeymapTest, ResavedProfileDropsEarlierDefaultResidue) {
            const auto root = std::filesystem::temp_directory_path() / "lfs_keymap_resaved_defaults";
            test::ScopedKeymapHome environment(root);
            const auto path = root / "profile.json";
            auto saved = test::historicalSavedProfile("v0.5.4");
            const auto prior = test::historicalSavedProfile("v0.5.0");
            for (const auto& old : prior["bindings"]) {
                const auto description = old.value("description", "");
                if (old.value("mode", -1) != static_cast<int>(ToolMode::SELECTION) ||
                    (description != "Select" && description != "Delete Gaussians" && description != "Orbit") ||
                    (description == "Select" && old.value("trigger_type", "") != "drag"))
                    continue;
                for (auto it = saved["bindings"].begin(); it != saved["bindings"].end();) {
                    if (it->value("mode", -1) == old.value("mode", -1) &&
                        it->value("trigger_type", "") == old.value("trigger_type", "") &&
                        it->value("button", -1) == old.value("button", -1) &&
                        it->value("key", -1) == old.value("key", -1) &&
                        it->value("modifiers", -1) == old.value("modifiers", -1))
                        it = saved["bindings"].erase(it);
                    else
                        ++it;
                }
                saved["bindings"].push_back(old);
            }
            test::writeProfile(path, saved);
            InputBindings loaded;
            ASSERT_TRUE(loaded.loadProfileFromFile(path));
            EXPECT_EQ(loaded.getActionForDrag(ToolMode::SELECTION, MouseButton::MIDDLE, 0),
                      Action::CAMERA_ORBIT);
        }
    } // namespace
} // namespace lfs::vis::input
