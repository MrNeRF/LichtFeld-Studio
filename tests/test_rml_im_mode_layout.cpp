/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "python/gil.hpp"
#include "python/lfs/rml_im_mode_layout.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/ElementText.h>

#include <cstdlib>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lfs::python {

    std::string rml_src_for_dynamic_texture(uint64_t, int, int) {
        return {};
    }

    class RmlImModeLayoutTestAccess {
    public:
        using PathDialogCallback = RmlImModeLayout::PathDialogCallback;

        static void setPathDialogCallback(PathDialogCallback callback) {
            RmlImModeLayout::set_path_dialog_callback_for_testing(
                std::move(callback));
        }

        static void resetPathDialogCallback() {
            RmlImModeLayout::reset_path_dialog_callback_for_testing();
        }

        static void clickBrowse(RmlImModeLayout& layout) {
            for (auto& slot : layout.containers_.front().slots) {
                if (slot.type == SlotType::SmallButton) {
                    slot.events->clicked = true;
                    return;
                }
            }
            FAIL() << "path_input browse slot was not created";
        }

        static void setPendingInput(RmlImModeLayout& layout,
                                    const std::string& value) {
            auto& slot = inputSlot(layout);
            slot.events->string_value = value;
            slot.events->changed = true;
        }

        static std::string inputAttribute(const RmlImModeLayout& layout) {
            const auto& slot = inputSlot(layout);
            auto* const input = directInput(slot.element);
            EXPECT_NE(input, nullptr);
            return input
                       ? input->GetAttribute<Rml::String>("value", "")
                       : std::string{};
        }

        static std::string inputEventValue(const RmlImModeLayout& layout) {
            return inputSlot(layout).events->string_value;
        }

        static bool inputHasPendingChange(const RmlImModeLayout& layout) {
            return inputSlot(layout).events->changed;
        }

    private:
        static Slot& inputSlot(RmlImModeLayout& layout) {
            for (auto& slot : layout.containers_.front().slots) {
                if (slot.type == SlotType::InputText)
                    return slot;
            }
            std::abort();
        }

        static const Slot& inputSlot(const RmlImModeLayout& layout) {
            for (const auto& slot : layout.containers_.front().slots) {
                if (slot.type == SlotType::InputText)
                    return slot;
            }
            std::abort();
        }

        static Rml::Element* directInput(Rml::Element* parent) {
            if (!parent)
                return nullptr;
            for (int i = 0; i < parent->GetNumChildren(); ++i) {
                auto* const child = parent->GetChild(i);
                if (child && child->GetTagName() == "input")
                    return child;
            }
            return nullptr;
        }
    };

} // namespace lfs::python

namespace {

    void prependBuiltPythonModulePath() {
        const auto module_dir =
            std::filesystem::path(PROJECT_ROOT_PATH) / "build/src/python";
        const std::string value = module_dir.string();
        const char* const existing = std::getenv("PYTHONPATH");
#ifdef _WIN32
        const char separator = ';';
#else
        const char separator = ':';
#endif
        const std::string combined =
            existing && *existing ? value + separator + existing : value;
#ifdef _WIN32
        _putenv_s("PYTHONPATH", combined.c_str());
#else
        setenv("PYTHONPATH", combined.c_str(), 1);
#endif
    }

    Rml::Element* findByClass(Rml::Element* root, const Rml::String& class_name) {
        if (!root)
            return nullptr;
        if (root->IsClassSet(class_name))
            return root;
        for (int i = 0; i < root->GetNumChildren(); ++i) {
            if (auto* const result = findByClass(root->GetChild(i), class_name))
                return result;
        }
        return nullptr;
    }

    class RmlImModeLayoutTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
        }

        static void TearDownTestSuite() {
            Rml::Shutdown();
        }

        void SetUp() override {
            auto root = document_.CreateElement("div");
            ASSERT_TRUE(root);
            root->SetId("im-root");
            root_ = document_.AppendChild(std::move(root));
            ASSERT_TRUE(root_);
        }

        void TearDown() override {
            layout_.release_elements();
            lfs::python::RmlImModeLayoutTestAccess::resetPathDialogCallback();
        }

        Rml::ElementDocument document_{"body"};
        lfs::python::RmlImModeLayout layout_;
        Rml::Element* root_ = nullptr;
    };

    TEST_F(RmlImModeLayoutTest, PathInputRoutesDialogAndSynchronizesLiveControl) {
        prependBuiltPythonModulePath();
        ASSERT_TRUE(lfs::python::ensure_initialized());

        struct DialogCall {
            bool folder_mode = false;
            std::filesystem::path default_path;
            std::optional<std::string> title;
            bool gil_released = false;
        };
        std::vector<DialogCall> calls;
        lfs::python::RmlImModeLayoutTestAccess::setPathDialogCallback(
            [&](const bool folder_mode,
                const std::filesystem::path& default_path,
                const std::optional<std::string>& title) {
                calls.push_back({
                    .folder_mode = folder_mode,
                    .default_path = default_path,
                    .title = title,
                    .gil_released = PyGILState_Check() == 0,
                });
                return calls.size() == 1
                           ? std::filesystem::path("/tmp/folder-picked")
                           : std::filesystem::path("/tmp/file-picked");
            });

        const lfs::python::GilAcquire gil;
        layout_.begin_frame(&document_);
        EXPECT_EQ(layout_.path_input("Output", "/tmp/start", true, ""),
                  std::make_tuple(false, std::string("/tmp/start")));
        layout_.end_frame();

        lfs::python::RmlImModeLayoutTestAccess::setPendingInput(
            layout_, "/tmp/pending");
        lfs::python::RmlImModeLayoutTestAccess::clickBrowse(layout_);
        layout_.begin_frame(&document_);
        EXPECT_EQ(layout_.path_input("Output", "/tmp/start", true, ""),
                  std::make_tuple(true, std::string("/tmp/folder-picked")));
        EXPECT_EQ(
            lfs::python::RmlImModeLayoutTestAccess::inputAttribute(layout_),
            "/tmp/folder-picked");
        EXPECT_EQ(
            lfs::python::RmlImModeLayoutTestAccess::inputEventValue(layout_),
            "/tmp/folder-picked");
        EXPECT_FALSE(
            lfs::python::RmlImModeLayoutTestAccess::inputHasPendingChange(layout_));
        layout_.end_frame();

        lfs::python::RmlImModeLayoutTestAccess::clickBrowse(layout_);
        layout_.begin_frame(&document_);
        EXPECT_EQ(
            layout_.path_input(
                "Output", "/tmp/folder-picked", false, "Choose output"),
            std::make_tuple(true, std::string("/tmp/file-picked")));
        layout_.end_frame();

        ASSERT_EQ(calls.size(), 2u);
        EXPECT_TRUE(calls[0].folder_mode);
        EXPECT_EQ(calls[0].default_path, "/tmp/pending");
        EXPECT_FALSE(calls[0].title.has_value());
        EXPECT_TRUE(calls[0].gil_released);
        EXPECT_FALSE(calls[1].folder_mode);
        EXPECT_EQ(calls[1].default_path, "/tmp/folder-picked");
        ASSERT_TRUE(calls[1].title.has_value());
        EXPECT_EQ(*calls[1].title, "Choose output");
        EXPECT_TRUE(calls[1].gil_released);
    }

    TEST_F(RmlImModeLayoutTest, SplitAndGridFlowEmitStructuralFlexState) {
        layout_.begin_frame(&document_);
        lfs::python::RmlSubLayout split(
            &layout_, lfs::python::RmlLayoutType::Split, 0.3f);
        split.enter();
        split.label("First");
        split.label("Second");
        split.exit();
        layout_.end_frame();

        auto* const split_element = findByClass(root_, "im-split");
        ASSERT_TRUE(split_element);
        ASSERT_EQ(split_element->GetNumChildren(), 2);
        EXPECT_EQ(
            split_element->GetChild(0)->GetAttribute<Rml::String>(
                "data-im-flex-basis", ""),
            "30%");
        EXPECT_EQ(
            split_element->GetChild(1)->GetAttribute<Rml::String>(
                "data-im-flex-basis", ""),
            "70%");
        ASSERT_TRUE(split_element->GetChild(0)->GetProperty("flex-basis"));
        EXPECT_EQ(
            split_element->GetChild(0)->GetProperty("flex-basis")->ToString(),
            "30%");

        layout_.begin_frame(&document_);
        lfs::python::RmlSubLayout grid(
            &layout_, lfs::python::RmlLayoutType::GridFlow, 0.5f, 3);
        grid.enter();
        grid.label("One");
        grid.label("Two");
        grid.label("Three");
        grid.exit();
        layout_.end_frame();

        auto* const grid_element = findByClass(root_, "im-grid-flow");
        ASSERT_TRUE(grid_element);
        ASSERT_EQ(grid_element->GetNumChildren(), 3);
        for (int i = 0; i < grid_element->GetNumChildren(); ++i) {
            auto* const child = grid_element->GetChild(i);
            EXPECT_EQ(
                child->GetAttribute<Rml::String>("data-im-flex-basis", ""),
                "33.33%");
            ASSERT_TRUE(child->GetProperty("flex-basis"));
            EXPECT_EQ(child->GetProperty("flex-basis")->ToString(), "33.33%");
        }

        layout_.begin_frame(&document_);
        lfs::python::RmlSubLayout uneven_grid(
            &layout_, lfs::python::RmlLayoutType::GridFlow, 0.5f, 3, false,
            false);
        uneven_grid.enter();
        uneven_grid.label("Natural one");
        uneven_grid.label("Natural two");
        uneven_grid.exit();
        layout_.end_frame();

        auto* const uneven_grid_element = findByClass(root_, "im-grid-flow");
        ASSERT_TRUE(uneven_grid_element);
        ASSERT_EQ(uneven_grid_element->GetNumChildren(), 2);
        for (int i = 0; i < uneven_grid_element->GetNumChildren(); ++i) {
            auto* const child = uneven_grid_element->GetChild(i);
            EXPECT_EQ(
                child->GetAttribute<Rml::String>("data-im-flex-basis", ""),
                "auto");
            EXPECT_EQ(
                child->GetAttribute<Rml::String>("data-im-flex-grow", ""),
                "0");
            EXPECT_EQ(
                child->GetAttribute<Rml::String>("data-im-align-self", ""),
                "flex-start");
        }
    }

    TEST_F(RmlImModeLayoutTest, PlainLabelTextDoesNotParseAsRml) {
        layout_.begin_frame(&document_);
        layout_.label("Literal <tag> & value");
        layout_.end_frame();

        auto* const label = findByClass(root_, "im-label");
        ASSERT_TRUE(label);
        ASSERT_EQ(label->GetNumChildren(), 1);
        auto* const text = dynamic_cast<Rml::ElementText*>(label->GetFirstChild());
        ASSERT_TRUE(text);
        EXPECT_EQ(text->GetText(), "Literal <tag> & value");
    }

} // namespace
