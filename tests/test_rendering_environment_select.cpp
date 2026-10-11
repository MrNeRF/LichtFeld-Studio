/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python/gil.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <RmlUi/Core/RenderInterface.h>
#include <gtest/gtest.h>
#include <nanobind/eval.h>
#include <nanobind/nanobind.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {
    class EnvironmentRenderInterface final : public Rml::RenderInterface {
    public:
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
        void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
        Rml::TextureHandle LoadTexture(Rml::Vector2i&, const Rml::String&) override { return 0; }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 0; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    class RenderingEnvironmentSelectTest : public ::testing::Test {
    protected:
        void python(const char* code) {
            const lfs::python::GilAcquire gil;
            nb::exec(nb::str(code), nb::module_::import_("__main__").attr("__dict__"));
        }

        static void SetUpTestSuite() {
            ASSERT_TRUE(lfs::python::ensure_initialized());
            ASSERT_TRUE(Rml::Initialise());
        }

        static void TearDownTestSuite() { Rml::Shutdown(); }

        void SetUp() override {
            context = Rml::CreateContext("environment_select", {640, 480}, &renderer);
            ASSERT_NE(context, nullptr);
            document = context->LoadDocumentFromMemory("<rml><body/></rml>");
            ASSERT_NE(document, nullptr);
            {
                const lfs::python::GilAcquire gil;
                auto sys = nb::module_::import_("sys");
                sys.attr("path").attr("insert")(0, (std::filesystem::path(PROJECT_ROOT_PATH) / "src/python").string().c_str());
                sys.attr("path").attr("insert")(1, (std::filesystem::path(PROJECT_ROOT_PATH) / "build/src/python").string().c_str());
                lfs::python::register_rml_document("environment_select", document);
            }
            python(R"PY(
import lichtfeld as _vr_lf
from types import SimpleNamespace as _VrSettings
from lfs_plugins.rendering_panel import RenderingPanel as _VrPanel
_vr_original_settings = _vr_lf.get_render_settings
_vr_settings = _VrSettings(environment_mode="EQUIRECTANGULAR", environment_map_path="")
_vr_lf.get_render_settings = lambda: _vr_settings
_vr_doc = _vr_lf.ui.rml.get_document("environment_select")
_vr_panel = _VrPanel()
_vr_model = _vr_doc.create_data_model("rendering")
_vr_model.bind("environment_map_preset", _vr_panel._get_environment_map_preset, _vr_panel._set_environment_map_preset)
_vr_model.bind_func("environment_map_has_custom_option", _vr_panel._environment_map_has_custom_option)
_vr_model.bind_func("environment_map_last_custom_display_name", _vr_panel._environment_map_last_custom_display_name)
_vr_panel._handle = _vr_model.get_handle()
_vr_panel._doc = _vr_doc
)PY");
            std::ifstream file(std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/rmlui/resources/rendering.rml");
            std::ostringstream buffer;
            buffer << file.rdbuf();
            const auto rml = buffer.str();
            const auto start = rml.find("<select id=\"environment-map-preset\"");
            ASSERT_NE(start, std::string::npos);
            const auto end = rml.find("</select>", start);
            ASSERT_NE(end, std::string::npos);
            document->SetInnerRML("<div data-model=\"rendering\">" + rml.substr(start, end + 9 - start) + "</div>");
            document->Show();
            context->Update();
            select = dynamic_cast<Rml::ElementFormControlSelect*>(document->GetElementById("environment-map-preset"));
            ASSERT_NE(select, nullptr);
        }

        void TearDown() override {
            if (document) {
                const lfs::python::GilAcquire gil;
                nb::exec("_vr_lf.get_render_settings = _vr_original_settings\n_vr_panel._doc = None\n_vr_panel._handle = None",
                         nb::module_::import_("__main__").attr("__dict__"));
                context->RemoveDataModel("rendering");
                lfs::python::unregister_rml_document("environment_select");
                context->UnloadDocument(document);
            }
            if (context)
                Rml::RemoveContext("environment_select");
        }

        std::string label() {
            for (int i = 0; i < select->GetNumChildren(true); ++i) {
                auto* child = select->GetChild(i);
                if (child->GetTagName() == "selectvalue")
                    return child->GetInnerRML();
            }
            return {};
        }

        void sync() {
            python("_vr_panel._sync_environment_state()");
            context->Update();
        }

        EnvironmentRenderInterface renderer;
        Rml::Context* context = nullptr;
        Rml::ElementDocument* document = nullptr;
        Rml::ElementFormControlSelect* select = nullptr;
    };

    TEST_F(RenderingEnvironmentSelectTest, ClosedLabelTracksSuccessiveCustomMaps) {
        for (const auto* name : {"first.hdr", "second.hdr", "third.exr"}) {
            const std::string code = std::string("_vr_settings.environment_map_path = '/tmp/") + name + "'";
            python(code.c_str());
            sync();
            EXPECT_EQ(select->GetValue(), "__custom__");
            EXPECT_EQ(label(), name);
            EXPECT_EQ(select->GetNumOptions(), 3);
            python("assert _vr_settings.environment_map_path.endswith(_vr_panel._environment_map_last_custom_display_name())");
        }
    }

    TEST_F(RenderingEnvironmentSelectTest, PresetsAndRememberedCustomMapStillWork) {
        EXPECT_EQ(select->GetValue(), "0");
        EXPECT_EQ(label(), "Kloofendal Pure Sky");
        python("_vr_settings.environment_map_path = ''");
        sync();
        python("assert _vr_settings.environment_map_path == ''");
        python("_vr_settings.environment_map_path = '/tmp/custom.hdr'");
        sync();
        EXPECT_EQ(label(), "custom.hdr");
        select->SetSelection(1);
        context->Update();
        EXPECT_EQ(label(), "Alps Field");
        python("assert _vr_panel._get_environment_map_preset() == '1'");
        select->SetSelection(2);
        context->Update();
        EXPECT_EQ(label(), "custom.hdr");
        python("assert _vr_settings.environment_map_path == '/tmp/custom.hdr'");
        sync();
        EXPECT_EQ(label(), "custom.hdr");
    }
    TEST_F(RenderingEnvironmentSelectTest, UpdateCost) {
        const lfs::python::GilAcquire gil;
        auto global = nb::module_::import_("__main__").attr("__dict__");
        nb::object settings = global["_vr_settings"];
        nb::object sync_state = global["_vr_panel"].attr("_sync_environment_state");
        settings.attr("environment_map_path") = "/tmp/first.hdr";
        sync_state();
        context->Update();
        for (const bool changing : {false, true}) {
            std::vector<double> samples;
            for (int sample = 0; sample < 7; ++sample) {
                const auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < 1000; ++i) {
                    if (changing)
                        settings.attr("environment_map_path") = (i % 2) ? "/tmp/first.hdr" : "/tmp/second.hdr";
                    sync_state();
                    context->Update();
                }
                const auto elapsed = std::chrono::steady_clock::now() - start;
                samples.push_back(std::chrono::duration<double, std::micro>(elapsed).count() / 1000.0);
            }
            std::sort(samples.begin(), samples.end());
            RecordProperty(changing ? "changed_update_us" : "idle_update_us", std::to_string(samples[3]));
        }
    }
} // namespace
