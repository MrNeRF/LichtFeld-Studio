/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "gui/collision_export_panel.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/services.hpp"
#include "gui/collision_job_status.hpp"
#include "gui/gui_manager.hpp"
#include "gui/ui_context.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "gui/utils/splat_export_request.hpp"
#include "io/collision_export.hpp"
#include "python/python_runtime.hpp"
#include <RmlUi/Core.h>
#include <format>

namespace lfs::vis::gui {
    CollisionExportPanel::CollisionExportPanel(RmlUIManager* manager)
        : host_(manager, "native.collision_export", "rmlui/collision_export.rml") {}
    CollisionExportPanel::~CollisionExportPanel() {
        worker_.request_stop();
        if (worker_.joinable())
            worker_.join();
        if (document_)
            document_->RemoveEventListener("click", this);
    }
    void CollisionExportPanel::reloadRmlResources() {
        if (document_)
            document_->RemoveEventListener("click", this);
        document_ = nullptr;
        host_.reloadDocument();
    }
    void CollisionExportPanel::text(const char* id, const std::string& value) {
        auto* el = document_->GetElementById(id);
        const auto encoded = Rml::StringUtilities::EncodeRml(value);
        if (el->GetInnerRML() != encoded) {
            el->SetInnerRML(encoded);
            host_.markContentDirty();
        }
    }
    void CollisionExportPanel::showError(const std::string& message) {
        LOG_ERROR("Collision export: {}", message);
        const auto info = collisionFailureInfo(message);
        text("export-status", info.title);
        text("export-error", info.advice + "\n" + destination_ + "\n" + message);
        document_->GetElementById("export-error")->SetProperty("display", "block");
    }
    bool CollisionExportPanel::prepare(const PanelDrawContext& ctx) {
        scene_ = ctx.scene;
        if (!scene_ || !host_.ensureDocumentLoaded())
            return false;
        if (!document_) {
            document_ = host_.getDocument();
            document_->AddEventListener("click", this);
            text("title-text", LOC("collision.export_section"));
        }
        if (done_.exchange(false, std::memory_order_acq_rel)) {
            if (worker_.joinable())
                worker_.join();
            busy_ = false;
            std::lock_guard lock(mutex_);
            if (!error_.empty())
                showError(error_);
            else {
                progress_value_ = 1;
                text("export-status", LOCF("collision.export_complete", destination_));
            }
            text("export-stage", progress_);
            text("export-time", LOCF("collision.progress_finished", std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count()));
            document_->GetElementById("export-progress")->SetAttribute("aria-valuenow", progress_value_ * 100);
            document_->GetElementById("export-progress-fill")->SetProperty("width", std::format("{:.0f}%", progress_value_ * 100));
        }
        if (!busy_) {
            const auto requested = takeCollisionExportRequest();
            if (!requested.is_nil()) {
                target_ = requested;
                destination_.clear();
                text("export-status", "");
                document_->GetElementById("export-error")->SetProperty("display", "none");
                document_->GetElementById("export-job")->SetProperty("display", "none");
            }
        }
        const auto* node = scene_->getNodeByUuid(target_);
        for (const auto* id : {"export-voxels", "export-glb", "export-cancel"}) {
            const bool disabled = std::string_view(id) == "export-cancel" ? !busy_ || worker_.get_stop_token().stop_requested() : bool(busy_) || !node || (std::string_view(id) == "export-voxels" ? !node->collision : !node->collision && !node->mesh);
            auto* el = document_->GetElementById(id);
            if (disabled != el->HasAttribute("disabled")) {
                if (disabled)
                    el->SetAttribute("disabled", "disabled");
                else
                    el->RemoveAttribute("disabled");
            }
        }
        const auto now = std::chrono::steady_clock::now();
        if (busy_ && now >= next_update_) {
            std::lock_guard lock(mutex_);
            document_->GetElementById("export-job")->SetProperty("display", "block");
            text("export-stage", collisionTaskLabel(progress_));
            text("export-time", LOCF("collision.progress_finished", std::chrono::duration<double>(now - started_).count()));
            document_->GetElementById("export-progress-fill")->SetProperty("width", std::format("{:.0f}%", progress_value_ * 100));
            document_->GetElementById("export-progress")->SetAttribute("aria-valuenow", progress_value_ * 100);
            next_update_ = now + std::chrono::milliseconds(100);
        }
        return true;
    }
    void CollisionExportPanel::chooseExport(bool voxels) {
        if (busy_)
            throw std::runtime_error("Wait for the current collision export to finish");
        const auto* node = scene_->getNodeByUuid(target_);
        if (!node || (voxels ? !node->collision : !node->collision && !node->mesh))
            throw std::runtime_error("Select a collision node to export");
        auto path = voxels ? SaveJsonFileDialog("scene.voxel.json") : SaveGlbFileDialog("scene.collision.glb");
        if (path.empty())
            return;
        if (voxels && !path.filename().string().ends_with(".voxel.json")) {
            path.replace_extension();
            path += ".voxel.json";
        }
        auto bin = path;
        bin.replace_extension(".bin");
        if (std::filesystem::exists(path) || (voxels && std::filesystem::exists(bin))) {
            core::ModalRequest request;
            request.title = LOC("collision.export_section");
            request.style = core::ModalStyle::Warning;
            request.body_rml = "<p>" + Rml::StringUtilities::EncodeRml(LOCF("collision.export_overwrite", core::path_to_utf8(path))) + "</p>";
            request.buttons = {{LOC("common.cancel"), "secondary"}, {LOC("export.overwrite"), "warning"}};
            const auto target = target_;
            const auto payload = node->collision;
            const auto geometry = node->mesh;
            request.on_result = [this, voxels, path, target, payload, geometry](const core::ModalResult& result) {
                if (result.button_label == LOC("export.overwrite"))
                    try {
                        const auto* current = scene_->getNodeByUuid(target);
                        if (target_ != target || !current || current->collision != payload || (!payload && current->mesh != geometry))
                            throw std::runtime_error("Collision changed while confirming export. Select the intended node and export again.");
                        startExport(voxels, path);
                    } catch (const std::exception& e) { showError(e.what()); }
            };
            services().guiOrNull()->enqueueModal(std::move(request));
        } else
            startExport(voxels, std::move(path));
    }
    void CollisionExportPanel::startExport(bool voxels, std::filesystem::path path) {
        if (busy_)
            throw std::runtime_error("Wait for the current collision export to finish");
        const auto* node = scene_->getNodeByUuid(target_);
        if (!node || (voxels ? !node->collision : !node->collision && !node->mesh))
            throw std::runtime_error("Collision node no longer exists");
        const auto volume = node->collision;
        const auto mesh = !node->collision || node->collision_display.surface_ready ? node->mesh : nullptr;
        const auto transform = scene_->getWorldTransform(node->id);
        const auto stride = node->collision_display.surface_stride;
        const auto shape = node->collision_display.surface_shape;
        {
            std::lock_guard lock(mutex_);
            error_.clear();
            destination_ = core::path_to_utf8(path);
            progress_ = voxels ? "Packing SuperSplat voxel pair" : "Preparing collision surface";
            progress_value_ = 0;
        }
        busy_ = true;
        done_ = false;
        started_ = std::chrono::steady_clock::now();
        next_update_ = {};
        text("export-status", LOC("collision.exporting"));
        document_->GetElementById("export-error")->SetProperty("display", "none");
        document_->GetElementById("export-job")->SetProperty("display", "block");
        worker_ = std::jthread([this, voxels, path, volume, mesh, transform, stride, shape](std::stop_token stop) {
            try {
                auto report = [&](std::string message) {
                    std::lock_guard lock(mutex_);
                    progress_value_ = message.starts_with("Writing") ? 2.0f / 3 : 1.0f / 3;
                    progress_ = std::move(message);
                };
                if (voxels)
                    io::exportCollisionVoxels(*volume, transform, path, stop, report);
                else if (mesh) {
                    const auto vertices = mesh->vertices.to(core::Device::CPU).contiguous();
                    const auto indices = mesh->indices.to(core::Device::CPU).contiguous();
                    io::exportCollisionGlb({vertices.ptr<float>(), vertices.numel()}, {indices.ptr<int32_t>(), indices.numel()}, transform, path, stop, report);
                } else {
                    report("Building export surface at " + std::to_string(stride) + "x cell size");
                    const auto surface = stride == 1 ? volume->mesh(shape, stop, report) : volume->coarsened(stride, stop).mesh(shape, stop, report);
                    io::exportCollisionGlb(surface.vertices, surface.indices, transform, path, stop, report);
                }
            } catch (const std::exception& e) {
                std::lock_guard lock(mutex_);
                error_ = e.what();
            }
            done_.store(true, std::memory_order_release);
            python::request_redraw();
        });
    }
    void CollisionExportPanel::ProcessEvent(Rml::Event& event) {
        auto* el = event.GetTargetElement();
        while (el && el->GetId().empty())
            el = el->GetParentNode();
        if (!el)
            return;
        try {
            if (el->GetId() == "export-voxels" || el->GetId() == "export-glb")
                chooseExport(el->GetId() == "export-voxels");
            else if (el->GetId() == "export-cancel")
                worker_.request_stop();
        } catch (const std::exception& e) { showError(e.what()); }
    }
    void CollisionExportPanel::draw(const PanelDrawContext& ctx) {
        if (prepare(ctx))
            host_.draw(ctx);
    }
    void CollisionExportPanel::preload(const PanelDrawContext& ctx) { prepare(ctx); }
    PanelDirectRenderResult CollisionExportPanel::renderDirect(const PanelDirectRenderRequest& r, const PanelDrawContext& ctx) {
        host_.setFloating(r.space == PanelSpace::Floating);
        host_.setInput(r.input);
        host_.setInputClipY(r.clip_y_min, r.clip_y_max);
        host_.setForcedHeight(r.forced_height);
        if (prepare(ctx)) {
            if (r.mode == PanelDirectRenderMode::Preload)
                host_.prepareDirect(r.width, r.height);
            else if (r.mode != PanelDirectRenderMode::Measure)
                host_.drawDirect(r.x, r.y, r.width, r.height);
        }
        host_.setInput(nullptr);
        host_.setInputClipY(-1, -1);
        host_.setForcedHeight(0);
        return {.handled = true, .height = host_.getContentHeight()};
    }
} // namespace lfs::vis::gui
