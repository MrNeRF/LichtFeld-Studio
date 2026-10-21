#include "core/error_bus.hpp"
#include "core/user_paths.hpp"
#include "input/input_bindings.hpp"
#include "input/sdl_key_mapping.hpp"
#include "keymap_history_helpers.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <string_view>

namespace lfs::vis::input {
    namespace {
        using KeymapTest = test::KeymapTest;
        namespace fs = std::filesystem;
        class DamagedProfileWarning final : public lfs::NativeErrorConsumer {
        public:
            void on_error(const lfs::ErrorNotification& notification,
                          const lfs::ErrorDeliveryInfo&) noexcept override {
                if (notification.error.code() == lfs::ErrorCode::InvalidArgument &&
                    notification.error.domain() == lfs::ErrorDomain::App)
                    count.fetch_add(1, std::memory_order_relaxed);
            }
            std::atomic<int> count{0};
        };

        std::string readBytes(const fs::path& path) {
            std::ifstream input(path, std::ios::binary);
            return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        }

        TEST_F(KeymapTest, KeepsOriginalBeforeReplacingDamagedFile) {
            const auto dir = fs::temp_directory_path() / "lfs_damaged_keymap_guard";
            fs::remove_all(dir);
            fs::create_directories(dir);
            DamagedProfileWarning warnings;
            auto subscription = lfs::ErrorBus::instance().subscribe(warnings);
            for (const std::string_view damaged : {
                     R"({"name":"Default","version":29,"bindings":[)",
                     R"({"name":"Default","version":29,"bindings":[{"mode":0,"trigger_type":"key","key":74}]})",
                 }) {
                SCOPED_TRACE(std::string(damaged));
                const auto path = dir / "Default.json";
                {
                    std::ofstream output(path, std::ios::binary);
                    output << damaged;
                }
                InputBindings bindings;
                const bool loaded = bindings.loadProfileFromFile(path);
                if (damaged.ends_with("["))
                    EXPECT_FALSE(loaded);
                else
                    EXPECT_TRUE(loaded);
                ASSERT_TRUE(bindings.saveProfileToFile(path));
                EXPECT_NE(readBytes(path), damaged);
                int copies = 0;
                for (const auto& entry : fs::directory_iterator(dir)) {
                    if (entry.path().extension() == ".bak") {
                        EXPECT_EQ(readBytes(entry.path()), damaged);
                        ++copies;
                    }
                }
                EXPECT_GE(copies, 1);
                for (const auto& entry : fs::directory_iterator(dir)) {
                    if (entry.path().extension() == ".bak")
                        fs::remove(entry.path());
                }
            }
            EXPECT_EQ(warnings.count.load(std::memory_order_relaxed), 3);
            fs::remove_all(dir);
        }

        TEST_F(KeymapTest, RefusesToOverwriteSymbolicLink) {
            const auto dir = fs::temp_directory_path() / "lfs_keymap_symlink_guard";
            fs::remove_all(dir);
            fs::create_directories(dir);
            const auto target = dir / "sentinel.txt";
            {
                std::ofstream output(target);
                output << "sentinel";
            }
            const auto link = dir / "Default.json";
            std::error_code error;
            fs::create_symlink(target, link, error);
            if (error) {
                GTEST_SKIP() << error.message();
            }
            InputBindings bindings;
            EXPECT_FALSE(bindings.saveProfileToFile(link));
            EXPECT_EQ(readBytes(target), "sentinel");
            fs::remove_all(dir);
        }

        TEST_F(KeymapTest, BacksUpParseableProfileWithSemanticInvalidity) {
            const auto dir = fs::temp_directory_path() / "lfs_semantically_damaged_keymap";
            fs::remove_all(dir);
            fs::create_directories(dir);
            const auto path = dir / "Default.json";
            InputBindings defaults;
            ASSERT_TRUE(defaults.saveProfileToFile(path));
            auto stored = nlohmann::json::parse(std::ifstream(path));
            const auto invalid = nlohmann::json{
                {"mode", static_cast<int>(ToolMode::GLOBAL)},
                {"action", static_cast<int>(Action::TOOL_SCALE)},
                {"description", "Invalid modifier mask"},
                {"trigger_type", "key"},
                {"key", KEY_F23},
                {"modifiers", 64}};
            stored["bindings"].push_back(invalid);
            stored["overrides"]["bindings"].push_back(invalid);
            const auto original = stored.dump(2) + '\n';
            {
                std::ofstream output(path, std::ios::binary);
                output << original;
            }
            InputBindings loaded;
            ASSERT_TRUE(loaded.loadProfileFromFile(path));
            ASSERT_TRUE(loaded.saveProfileToFile(path));
            bool backed_up_original = false;
            for (const auto& entry : fs::directory_iterator(dir)) {
                if (entry.path().extension() == ".bak")
                    backed_up_original |= readBytes(entry.path()) == original;
            }
            EXPECT_TRUE(backed_up_original);
            fs::remove_all(dir);
        }

        TEST_F(KeymapTest, FunctionKeysF13ThroughF24StayDistinct) {
            InputBindings bindings;
            bindings.startCapture(ToolMode::GLOBAL, Action::TOOL_SCALE);
            bindings.captureKey(KEY_UNKNOWN, KEY_UNKNOWN, 0);
            EXPECT_TRUE(bindings.isCapturing());
            EXPECT_FALSE(bindings.getAndClearCaptured());
            bindings.cancelCapture();
            bindings.setBinding(ToolMode::GLOBAL, Action::TOOL_SCALE,
                                KeyTrigger{KEY_UNKNOWN});
            EXPECT_EQ(bindings.getActionForShortcut(ToolMode::SELECTION, KEY_UNKNOWN,
                                                    KEY_UNKNOWN, 0),
                      Action::NONE);
            std::set<int> app_keys;
            for (int i = 0; i < 12; ++i) {
                const auto scancode = static_cast<SDL_Scancode>(SDL_SCANCODE_F13 + i);
                const auto keycode = static_cast<SDL_Keycode>(SDLK_F13 + i);
                const int physical = sdlScancodeToAppKey(scancode);
                const int logical = sdlKeycodeToAppKey(keycode);
                SCOPED_TRACE("F" + std::to_string(13 + i));
                ASSERT_NE(physical, KEY_UNKNOWN);
                EXPECT_EQ(logical, physical);
                EXPECT_TRUE(app_keys.insert(physical).second);
                EXPECT_EQ(appKeyToSdlScancode(physical), scancode);
                EXPECT_EQ(getKeyName(physical), "F" + std::to_string(13 + i));
                bindings.startCapture(ToolMode::GLOBAL, Action::TOOL_SCALE);
                bindings.captureKey(physical, logical, 0);
                const auto captured = bindings.getAndClearCaptured();
                ASSERT_TRUE(captured);
                ASSERT_TRUE(std::holds_alternative<KeyTrigger>(*captured));
                EXPECT_EQ(std::get<KeyTrigger>(*captured).key, physical);
                bindings.setBinding(ToolMode::GLOBAL, Action::TOOL_SCALE, *captured);
                EXPECT_EQ(bindings.getActionForShortcut(ToolMode::SELECTION, physical,
                                                        logical, 0),
                          Action::TOOL_SCALE);
            }
            EXPECT_EQ(app_keys.size(), 12u);
            for (const int key : app_keys) {
                bindings.setBinding(ToolMode::GLOBAL, Action::TOOL_SCALE,
                                    KeyTrigger{key});
                for (const int other : app_keys) {
                    if (other != key)
                        EXPECT_NE(bindings.getActionForKey(ToolMode::SELECTION, other, 0),
                                  Action::TOOL_SCALE);
                }
            }
        }

        namespace fs = std::filesystem;
        using json = nlohmann::json;
        class InvalidEntryWarnings final : public lfs::NativeErrorConsumer {
        public:
            void on_error(const lfs::ErrorNotification& notification,
                          const lfs::ErrorDeliveryInfo&) noexcept override {
                if (notification.error.code() == lfs::ErrorCode::InvalidArgument &&
                    notification.error.domain() == lfs::ErrorDomain::App) {
                    count.fetch_add(1, std::memory_order_relaxed);
                    message = notification.error.user_message();
                }
            }
            std::atomic<int> count{0};
            std::string message;
        };

        TEST_F(KeymapTest, KeepsValidBindingsAndPreservesEverySkippedEntry) {
            const fs::path source = fs::temp_directory_path() / "lfs_invalid_keymap_source.json";
            const fs::path saved = fs::temp_directory_path() / "lfs_invalid_keymap_saved.json";
            InputBindings defaults;
            ASSERT_TRUE(defaults.saveProfileToFile(source));
            auto profile = json::parse(std::ifstream(source));
            auto& entries = profile["overrides"]["bindings"];
            entries.push_back({{"mode", 0}, {"action", static_cast<int>(Action::TOOL_SCALE)}, {"description", "bad modifiers"}, {"trigger_type", "key"}, {"key", KEY_8}, {"modifiers", 64}});
            entries.push_back({{"mode", 0}, {"action", static_cast<int>(Action::CAMERA_MOVE_FORWARD)}, {"description", "modified movement"}, {"trigger_type", "key"}, {"key", KEY_W}, {"modifiers", MODIFIER_CTRL}});
            entries.push_back({{"mode", 0}, {"action", static_cast<int>(Action::TOOL_SCALE)}, {"description", "unknown trigger"}, {"trigger_type", "future_trigger"}, {"code", 7}});
            entries.push_back({{"mode", 999}, {"action", static_cast<int>(Action::TOOL_SCALE)}, {"description", "unknown mode"}, {"trigger_type", "key"}, {"key", KEY_8}, {"modifiers", 0}});
            entries.push_back({{"mode", 0}, {"action", 99999}, {"description", "unknown action"}, {"trigger_type", "key"}, {"key", KEY_8}, {"modifiers", 0}});
            entries.push_back({{"mode", 0}, {"action", static_cast<int>(Action::TOOL_SCALE)}, {"description", "out of range"}, {"trigger_type", "key"}, {"key", 10000}, {"modifiers", 0}});
            entries.push_back({{"mode", 0}, {"action", static_cast<int>(Action::CAMERA_FOCUS_SELECTION)}, {"description", "Focus Selection"}, {"trigger_type", "key"}, {"key", KEY_F23}, {"modifiers", 0}});
            entries.push_back({{"mode", 0}, {"action", static_cast<int>(Action::TOOL_SCALE)}, {"description", "Scale"}, {"trigger_type", "key"}, {"key", KEY_F24}, {"modifiers", 0}});
            entries.push_back({{"mode", 0}, {"action", static_cast<int>(Action::TOOL_ROTATE)}, {"description", "Rotate"}, {"trigger_type", "key"}, {"key", KEY_F24}, {"modifiers", 0}});
            const json raw_entry = {{"mode", 0}, {"action", 99999}, {"trigger_type", "future_trigger"}, {"opaque", json::array({1, 2, 3})}};
            entries.push_back(raw_entry);
            test::writeProfile(source, profile);

            InvalidEntryWarnings warnings;
            auto subscription = lfs::ErrorBus::instance().subscribe(warnings);
            InputBindings loaded;
            ASSERT_TRUE(loaded.loadProfileFromFile(source));
            EXPECT_EQ(loaded.getActionForKey(ToolMode::GLOBAL, KEY_F23, 0),
                      Action::CAMERA_FOCUS_SELECTION);
            EXPECT_EQ(loaded.getActionForKey(ToolMode::GLOBAL, KEY_F24, 0), Action::TOOL_SCALE);
            EXPECT_EQ(loaded.getActionForKey(ToolMode::GLOBAL, KEY_4, 0), Action::TOOL_SCALE);
            EXPECT_EQ(loaded.getActionForKey(ToolMode::GLOBAL, 10000, 0), Action::NONE);
            EXPECT_EQ(warnings.count.load(std::memory_order_relaxed), 1);
            EXPECT_FALSE(warnings.message.empty());
            EXPECT_NE(warnings.message.find("1, 2"), std::string::npos);

            ASSERT_TRUE(loaded.saveProfileToFile(saved));
            const auto round_trip = json::parse(std::ifstream(saved));
            EXPECT_NE(std::ranges::find(round_trip["bindings"], raw_entry),
                      round_trip["bindings"].end());
            EXPECT_NE(std::ranges::find(round_trip["overrides"]["bindings"], raw_entry),
                      round_trip["overrides"]["bindings"].end());

            const fs::path extensions = fs::temp_directory_path() /
                                        "lfs_keymap_extension_identity.json";
            for (const auto& entry : fs::directory_iterator(extensions.parent_path())) {
                if (entry.path().filename().string().starts_with(extensions.filename().string()))
                    fs::remove(entry.path());
            }
            InputBindings extension_defaults;
            ASSERT_TRUE(extension_defaults.saveProfileToFile(extensions));
            auto extension_profile = json::parse(std::ifstream(extensions));
            extension_profile["bindings"].push_back(
                {{"mode", 0}, {"action", static_cast<int>(Action::TOOL_SCALE)}, {"trigger_type", "key"}, {"key", KEY_F12}, {"modifiers", 0}, {"extension_tag", "scale"}});
            extension_profile["bindings"].push_back(
                {{"mode", 0}, {"action", static_cast<int>(Action::TOOL_ROTATE)}, {"trigger_type", "key"}, {"key", KEY_F12}, {"modifiers", 0}, {"extension_tag", "rotate"}});
            test::writeProfile(extensions, extension_profile);
            DamagedProfileWarning damaged_warnings;
            auto damaged_subscription = lfs::ErrorBus::instance().subscribe(damaged_warnings);
            InputBindings extension_writer;
            ASSERT_TRUE(extension_writer.saveProfileToFile(extensions));
            EXPECT_EQ(damaged_warnings.count.load(std::memory_order_relaxed), 0);
            EXPECT_FALSE(std::ranges::any_of(fs::directory_iterator(extensions.parent_path()),
                                             [&](const auto& entry) {
                                                 return entry.path().filename().string().starts_with(
                                                     extensions.filename().string() + ".damaged-");
                                             }));
            fs::remove(extensions);
            fs::remove(source);
            fs::remove(saved);
        }

        TEST_F(KeymapTest, UnknownTriggerOnHistoricalScalePreservesItsDefault) {
            const fs::path source = fs::temp_directory_path() / "lfs_unknown_scale_trigger.json";
            const fs::path saved = fs::temp_directory_path() / "lfs_unknown_scale_trigger_saved.json";
            auto profile = test::historicalSavedProfile("v0.5.4");
            auto scale = std::ranges::find_if(profile["bindings"], [](const json& entry) {
                return entry.value("mode", -1) == static_cast<int>(ToolMode::GLOBAL) &&
                       entry.value("description", "") == "Scale" &&
                       entry.value("trigger_type", "") == "key" &&
                       entry.value("key", -1) == KEY_4;
            });
            ASSERT_NE(scale, profile["bindings"].end());
            scale->erase("key");
            (*scale)["trigger_type"] = "key_future_2031";
            const json invalid_scale = *scale;
            test::writeProfile(source, profile);

            InvalidEntryWarnings warnings;
            auto subscription = lfs::ErrorBus::instance().subscribe(warnings);
            InputBindings loaded;
            ASSERT_TRUE(loaded.loadProfileFromFile(source));
            EXPECT_EQ(warnings.count.load(std::memory_order_relaxed), 1);
            EXPECT_EQ(loaded.getActionForKey(ToolMode::GLOBAL, KEY_4, 0), Action::TOOL_SCALE);
            EXPECT_EQ(loaded.getActionForKey(ToolMode::GLOBAL, KEY_0, 0), Action::NONE);
            ASSERT_TRUE(loaded.saveProfileToFile(saved));
            const auto saved_profile = json::parse(std::ifstream(saved));
            EXPECT_NE(std::ranges::find(saved_profile["bindings"], invalid_scale),
                      saved_profile["bindings"].end());
            EXPECT_NE(std::ranges::find(saved_profile["overrides"]["bindings"], invalid_scale),
                      saved_profile["overrides"]["bindings"].end());
            InputBindings reloaded;
            ASSERT_TRUE(reloaded.loadProfileFromFile(saved));
            EXPECT_EQ(reloaded.getActionForKey(ToolMode::GLOBAL, KEY_4, 0), Action::TOOL_SCALE);
            fs::remove(source);
            fs::remove(saved);
        }

        TEST_F(KeymapTest, LargeProfileSkipsOutOfRangeCodesInLinearLoad) {
            const fs::path path = fs::temp_directory_path() / "lfs_large_keymap_profile.json";
            InputBindings defaults;
            ASSERT_TRUE(defaults.saveProfileToFile(path));
            auto profile = json::parse(std::ifstream(path));
            auto& additions = profile["overrides"]["bindings"];
            for (int i = 0; i < 100000; ++i) {
                additions.push_back({{"mode", static_cast<int>(ToolMode::GLOBAL)},
                                     {"action", static_cast<int>(Action::TOOL_SCALE)},
                                     {"description", "Scale"},
                                     {"trigger_type", "key"},
                                     {"key", 10000 + i},
                                     {"modifiers", 0},
                                     {"on_repeat", false}});
            }
            {
                std::ofstream output(path);
                output << profile.dump();
            }
            InvalidEntryWarnings warnings;
            auto subscription = lfs::ErrorBus::instance().subscribe(warnings);
            InputBindings loaded;
            const auto begin = std::chrono::steady_clock::now();
            ASSERT_TRUE(loaded.loadProfileFromFile(path));
            const auto elapsed = std::chrono::steady_clock::now() - begin;
            EXPECT_LT(elapsed, std::chrono::seconds(10));
            EXPECT_EQ(loaded.getActionForKey(ToolMode::GLOBAL, KEY_4, 0), Action::TOOL_SCALE);
            EXPECT_EQ(loaded.getActionForKey(ToolMode::GLOBAL, 10000, 0), Action::NONE);
            EXPECT_EQ(warnings.count.load(std::memory_order_relaxed), 1);
            fs::remove(path);
        }

        using ScopedProfileHome = test::ScopedKeymapHome;
        TEST_F(KeymapTest, NamedSaveWritesAndLoadsItsOwnActiveName) {
            namespace fs = std::filesystem;
            const fs::path root = fs::temp_directory_path() / "lfs_keymap_named_save";
            ScopedProfileHome home(root);
            const auto paths = core::UserPaths::resolve();
            ASSERT_TRUE(paths);
            ASSERT_TRUE(paths->ensureDirectories());
            InputBindings bindings;
            bindings.setBinding(ToolMode::GLOBAL, Action::TOOL_SCALE, KeyTrigger{KEY_J, 0});
            ASSERT_TRUE(bindings.saveProfile("Named"));
            InputBindings restarted;
            EXPECT_EQ(restarted.getCurrentProfileName(), "Named");
            EXPECT_EQ(restarted.getActionForKey(ToolMode::GLOBAL, KEY_J, 0), Action::TOOL_SCALE);
            EXPECT_EQ(restarted.getActionForKey(ToolMode::GLOBAL, KEY_4, 0), Action::NONE);
            restarted.loadProfile("Default");
            InputBindings default_restart;
            EXPECT_EQ(default_restart.getCurrentProfileName(), "Default");
            EXPECT_EQ(default_restart.getActionForKey(ToolMode::GLOBAL, KEY_4, 0), Action::TOOL_SCALE);
            ASSERT_TRUE(bindings.saveProfile("FirstName"));
            const auto first = nlohmann::json::parse(
                std::ifstream(paths->keymapDir() / "FirstName.json"));
            EXPECT_EQ(first.at("name"), "FirstName");
            bindings.loadProfile("FirstName");
            EXPECT_EQ(bindings.getCurrentProfileName(), "FirstName");
            ASSERT_TRUE(bindings.saveProfile("SecondName"));
            const auto second = nlohmann::json::parse(
                std::ifstream(paths->keymapDir() / "SecondName.json"));
            EXPECT_EQ(second.at("name"), "SecondName");
            bindings.loadProfile("SecondName");
            EXPECT_EQ(bindings.getCurrentProfileName(), "SecondName");
        }

        using ScopedProfileHome = test::ScopedKeymapHome;
        TEST_F(KeymapTest, PartialImportKeepsValidBindingsAndSkipsBrokenEntry) {
            namespace fs = std::filesystem;
            const fs::path root = fs::temp_directory_path() / "lfs_keymap_rejected_import";
            ScopedProfileHome home(root);
            InputBindings bindings;
            bindings.setBinding(ToolMode::GLOBAL, Action::TOOL_SCALE, KeyTrigger{KEY_F9});
            const fs::path valid = root / "valid.json";
            ASSERT_TRUE(bindings.saveProfileToFile(valid));
            ASSERT_TRUE(bindings.loadProfileFromFile(valid));
            const auto before_name = bindings.getCurrentProfileName();
            const auto before_revision = bindings.getBindingsRevision();
            const auto before_display = bindings.getBindingsForMode(ToolMode::GLOBAL);
            ASSERT_EQ(bindings.getActionForKey(ToolMode::SELECTION, KEY_F9, 0),
                      Action::TOOL_SCALE);

            const fs::path invalid = root / "invalid.json";
            const std::string raw_invalid_entry =
                R"({ "mode" : 0, "description" : "Missing action", "trigger_type" : "key", "key" : 53, "modifiers" : 0, "opaque" : [1, 2] })";
            {
                std::ofstream output(invalid);
                output << "{\"name\":\"Broken\",\"version\":" << test::currentProfileVersion()
                       << ",\"bindings\":[],\"overrides\":{\"bindings\":["
                       << R"({"mode":0,"action":62,"description":"Translate","trigger_type":"key","key":52,"modifiers":0},)"
                       << raw_invalid_entry << "],\"removed\":[],\"disabled\":[]}}";
            }
            ASSERT_TRUE(bindings.loadProfileFromFile(invalid));
            EXPECT_EQ(bindings.getCurrentProfileName(), "Broken");
            EXPECT_GT(bindings.getBindingsRevision(), before_revision);
            EXPECT_NE(bindings.getBindingsForMode(ToolMode::GLOBAL), before_display);
            EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_4, 0), Action::TOOL_TRANSLATE);
            EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_5, 0), Action::TOOL_MIRROR);
            EXPECT_EQ(bindings.getActionForKey(ToolMode::SELECTION, KEY_F9, 0), Action::NONE);
            const fs::path roundtrip = root / "roundtrip.json";
            ASSERT_TRUE(bindings.saveProfileToFile(roundtrip));
            std::ifstream saved(roundtrip, std::ios::binary);
            const std::string roundtrip_bytes((std::istreambuf_iterator<char>(saved)), {});
            const auto roundtrip_json = nlohmann::json::parse(roundtrip_bytes);
            const auto invalid_entry = nlohmann::json::parse(raw_invalid_entry);
            EXPECT_NE(std::ranges::find(roundtrip_json["bindings"], invalid_entry),
                      roundtrip_json["bindings"].end());
            EXPECT_NE(std::ranges::find(roundtrip_json["overrides"]["bindings"], invalid_entry),
                      roundtrip_json["overrides"]["bindings"].end());
        }

        TEST_F(KeymapTest, SideMouseButtonsHaveDifferentReadableNames) {
            const auto back = static_cast<MouseButton>(3);
            const auto forward = static_cast<MouseButton>(4);
            EXPECT_NE(getMouseButtonName(back), getMouseButtonName(forward));
            EXPECT_NE(getMouseButtonName(back), "Mouse?");
            EXPECT_NE(getMouseButtonName(forward), "Mouse?");
            InputBindings bindings;
            bindings.setBinding(ToolMode::GLOBAL, Action::CAMERA_SET_PIVOT,
                                MouseButtonTrigger{back});
            const auto back_label = bindings.getTriggerDescription(
                Action::CAMERA_SET_PIVOT, ToolMode::GLOBAL);
            bindings.setBinding(ToolMode::GLOBAL, Action::CAMERA_SET_PIVOT,
                                MouseButtonTrigger{forward});
            const auto forward_label = bindings.getTriggerDescription(
                Action::CAMERA_SET_PIVOT, ToolMode::GLOBAL);
            EXPECT_NE(back_label, forward_label);
        }

        TEST_F(KeymapTest, RejectsMovementModifiersAtCaptureAndImport) {
            InputBindings bindings;
            for (const int forbidden : {MODIFIER_CTRL, MODIFIER_ALT, MODIFIER_SUPER}) {
                SCOPED_TRACE(forbidden);
                EXPECT_FALSE(InputBindings::isValidTriggerForAction(
                    Action::CAMERA_MOVE_FORWARD, KeyTrigger{KEY_K, forbidden}));
                bindings.startCapture(ToolMode::GLOBAL, Action::CAMERA_MOVE_FORWARD);
                bindings.captureKey(KEY_K, KEY_K, forbidden);
                EXPECT_TRUE(bindings.isCapturing());
                EXPECT_FALSE(bindings.getAndClearCaptured());
                bindings.cancelCapture();
                bindings.setBinding(ToolMode::GLOBAL, Action::CAMERA_MOVE_FORWARD,
                                    KeyTrigger{KEY_K, forbidden});
                EXPECT_NE(bindings.getKeyForAction(Action::CAMERA_MOVE_FORWARD), KEY_K);
            }
            bindings.startCapture(ToolMode::GLOBAL, Action::CAMERA_MOVE_FORWARD);
            bindings.captureKey(KEY_K, KEY_K, MODIFIER_SHIFT);
            EXPECT_FALSE(bindings.isCapturing());
            EXPECT_EQ(bindings.getKeyForAction(Action::CAMERA_MOVE_FORWARD), KEY_K);

            const auto path = std::filesystem::temp_directory_path() /
                              "lfs_invalid_movement_keymap.json";
            {
                std::ofstream output(path);
                output << R"({"name":"Broken","version":)" << test::currentProfileVersion()
                       << R"(,"bindings":[],"overrides":{"bindings":[
                {"mode":0,"action":64,"description":"Scale","trigger_type":"key","key":297,"modifiers":0},
                {"mode":0,"action":5,"description":"Forward","trigger_type":"key","key":75,"modifiers":2}],
                "removed":[],"disabled":[]}})";
            }
            ASSERT_TRUE(bindings.loadProfileFromFile(path));
            EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_F8, 0), Action::TOOL_SCALE);
            EXPECT_NE(bindings.getKeyForAction(Action::CAMERA_MOVE_FORWARD), KEY_K);
            std::filesystem::remove(path);
        }

        TEST_F(KeymapTest, RejectsImpossibleModifierMasks) {
            InputBindings bindings;
            for (const int invalid : {16, 32, -1}) {
                SCOPED_TRACE(invalid);
                const KeyTrigger key{KEY_4, invalid};
                EXPECT_FALSE(InputBindings::isValidTriggerForAction(Action::TOOL_SCALE, key));
                bindings.setBinding(ToolMode::GLOBAL, Action::TOOL_SCALE, key);
                EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_4, 0),
                          Action::TOOL_SCALE);
            }
            EXPECT_TRUE(InputBindings::isValidTriggerForAction(
                Action::TOOL_SCALE, KeyTrigger{KEY_4, MODIFIER_SHIFT | MODIFIER_CTRL}));
        }

        TEST_F(KeymapTest, CapturedInsertShowsItsNameAndDispatches) {
            InputBindings bindings;
            const int physical = sdlScancodeToAppKey(SDL_SCANCODE_INSERT);
            const int logical = sdlKeycodeToAppKey(SDLK_INSERT);
            EXPECT_EQ(physical, KEY_INSERT);
            EXPECT_EQ(logical, KEY_INSERT);
            bindings.startCapture(ToolMode::GLOBAL, Action::TOOL_SCALE);
            bindings.captureKey(physical, logical, 0);
            const auto captured = bindings.getAndClearCaptured();
            ASSERT_TRUE(captured);
            EXPECT_EQ(getKeyName(KEY_INSERT), "Insert");
            EXPECT_EQ(bindings.getTriggerDescription(Action::TOOL_SCALE), "Insert");
            EXPECT_EQ(bindings.getActionForShortcut(ToolMode::GLOBAL, physical, logical, 0),
                      Action::TOOL_SCALE);
        }

        TEST_F(KeymapTest, ProfileNamesCannotEscapeTheKeymapDirectory) {
            const auto root = fs::temp_directory_path() / "lfs_keymap_name_boundary";
            test::ScopedKeymapHome home(root);
            const auto keymaps = root / "config" / "keymaps";
            const auto outside = root.parent_path() / "lfs_keymap_escape_probe.json";
            {
                std::ofstream file(outside);
                file << "sentinel";
            }
            InputBindings bindings;
            EXPECT_FALSE(bindings.saveProfile("../../../lfs_keymap_escape_probe"));
            EXPECT_FALSE(bindings.saveProfile("CON"));
            EXPECT_FALSE(bindings.saveProfile("parent\\child"));
            std::string contents;
            {
                std::ifstream file(outside);
                std::getline(file, contents);
            }
            EXPECT_EQ(contents, "sentinel");
            EXPECT_FALSE(fs::exists(keymaps / "CON.json"));
            EXPECT_FALSE(fs::exists(keymaps / "parent\\child.json"));
            fs::remove(outside);
        }

        TEST_F(KeymapTest, ProfileNamesRejectReservedAndPathNames) {
            for (const std::string_view name : {
                     "", ".", "..", "my..profile", "parent/child", "parent\\child",
                     "/absolute", "C:\\absolute", "CON", "con.txt", "CON .txt", "PRN",
                     "AUX", "NUL", "COM1", "COM¹", "LPT9", "LPT²", "name:",
                     "name?", "name*", "end.", "end "}) {
                SCOPED_TRACE(std::string(name));
                EXPECT_FALSE(InputBindings::isValidProfileName(name));
            }
            for (const std::string_view name : {"Default", "My Map", "custom_v2", "scene-keys"}) {
                SCOPED_TRACE(std::string(name));
                EXPECT_TRUE(InputBindings::isValidProfileName(name));
            }
        }
    } // namespace
} // namespace lfs::vis::input
