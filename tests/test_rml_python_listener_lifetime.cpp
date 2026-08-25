/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "python/gil.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/RenderInterface.h>

#include <nanobind/eval.h>
#include <nanobind/nanobind.h>

#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>

namespace nb = nanobind;

namespace {

    class StubRenderInterface final : public Rml::RenderInterface {
    public:
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>,
                                                    Rml::Span<const int>) override {
            return 1;
        }

        void RenderGeometry(Rml::CompiledGeometryHandle,
                            Rml::Vector2f,
                            Rml::TextureHandle) override {}

        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}

        Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions,
                                       const Rml::String&) override {
            dimensions = {16, 16};
            return 1;
        }

        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>,
                                           Rml::Vector2i) override {
            return 1;
        }

        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

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

    constexpr const char* kDocumentRml = R"RML(
<rml>
    <head>
        <title>listener-lifetime</title>
    </head>
    <body id="root">
        <div id="target"/>
    </body>
</rml>
)RML";

    class RmlPythonListenerLifetimeTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            prependBuiltPythonModulePath();
            ASSERT_TRUE(Rml::Initialise());
        }

        static void TearDownTestSuite() {
            Rml::Shutdown();
        }

        void SetUp() override {
            context_ = Rml::CreateContext("python_listener_lifetime", {320, 240},
                                          &render_interface_);
            ASSERT_TRUE(context_);
            document_ = context_->LoadDocumentFromMemory(kDocumentRml);
            ASSERT_TRUE(document_);
            document_->Show();
            context_->Update();
        }

        void TearDown() override {
            if (document_ && context_) {
                if (lfs::python::can_acquire_gil()) {
                    const lfs::python::GilAcquire gil;
                    lfs::python::unregister_rml_document(kDocName);
                } else {
                    lfs::python::unregister_rml_document(kDocName);
                }
                context_->UnloadDocument(document_);
                context_->Update();
                document_ = nullptr;
            }
            if (context_) {
                Rml::RemoveContext("python_listener_lifetime");
                context_ = nullptr;
            }
        }

        void attachPythonOwnedListeners() {
            ASSERT_TRUE(lfs::python::ensure_initialized());
            const lfs::python::GilAcquire gil;
            nb::module_::import_("lichtfeld");
            lfs::python::register_rml_document(kDocName, document_);
            auto rml = nb::module_::import_("lichtfeld.ui").attr("rml");
            nb::object doc = rml.attr("get_document")(kDocName);
            ASSERT_FALSE(doc.is_none());

            nb::object global = nb::module_::import_("__main__").attr("__dict__");
            nb::dict locals;
            locals["doc"] = doc;
            nb::exec(R"(
captured = []
hold = []

def on_click(event):
    captured.append(getattr(event, "type", lambda: "")())

target = doc.get_element_by_id("target")
assert target is not None
hold.append(target)
target.add_event_listener("click", on_click)
doc.add_event_listener("keydown", lambda event, box=captured: box.append("key"))

model = doc.create_data_model("lifetime_model")
model.bind_func("flag", lambda: True)
model.bind_event("act", lambda *args: captured.append("act"))
model.get_handle()
)",
                     global, locals);
        }

        void unloadWithoutGIL() {
            ASSERT_FALSE(PyGILState_Check());
            context_->RemoveDataModel("lifetime_model");
            lfs::python::unregister_rml_document(kDocName);
            context_->UnloadDocument(document_);
            context_->Update();
            document_ = nullptr;
        }

        static constexpr const char* kDocName = "listener_lifetime";
        StubRenderInterface render_interface_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
    };

    TEST_F(RmlPythonListenerLifetimeTest, UnloadDetachesPythonListenersWithoutGIL) {
        attachPythonOwnedListeners();
        unloadWithoutGIL();
        SUCCEED();
    }

    TEST_F(RmlPythonListenerLifetimeTest, RepeatedUnloadWithoutGILDoesNotDoubleFree) {
        attachPythonOwnedListeners();
        unloadWithoutGIL();

        document_ = context_->LoadDocumentFromMemory(kDocumentRml);
        ASSERT_TRUE(document_);
        document_->Show();
        context_->Update();
        attachPythonOwnedListeners();
        unloadWithoutGIL();
        SUCCEED();
    }

} // namespace
