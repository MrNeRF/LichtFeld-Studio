/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// VR for builds without the Vulkan viewport renderer, which draws the eyes.
#include "core/event_bridge/localization_manager.hpp"
#include "openxr_service.hpp"

namespace lfs::vis::vr {

    struct OpenXrService::Impl {
        std::string status_detail = "VR needs the Vulkan graphics backend with the Vulkan rasterizer.";
        EyeFrame mirror;
    };

    OpenXrService::OpenXrService() : impl_(std::make_unique<Impl>()) {}
    OpenXrService::~OpenXrService() = default;

    bool OpenXrService::prepare() { return false; }
    const VulkanDeviceHooks* OpenXrService::deviceHooks() const { return nullptr; }
    void OpenXrService::setUnavailable(std::string reason) { impl_->status_detail = std::move(reason); }
    bool OpenXrService::available() const { return false; }
    bool OpenXrService::active() const { return false; }
    bool OpenXrService::running() const { return false; }
    const char* OpenXrService::statusKey() const { return "vr.status_unavailable"; }
    const std::string& OpenXrService::statusDetail() const { return impl_->status_detail; }

    lfs::Status OpenXrService::start(GraphicsContext&, const glm::mat3&, const glm::vec3&) {
        return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Unavailable,
            .domain = lfs::ErrorDomain::Rendering,
            .user_message = LOCF(statusKey(), impl_->status_detail),
            .detection = LFS_SOURCE_SITE_CURRENT(),
        }));
    }

    void OpenXrService::stop() {}
    void OpenXrService::recenter() {}
    void OpenXrService::snapTurn(int) {}
    void OpenXrService::setMoveSpeed(float) {}
    lfs::Result<bool> OpenXrService::frame(glm::vec3, const RenderEyes&) { return false; }
    const EyeFrame& OpenXrService::mirror() const { return impl_->mirror; }

} // namespace lfs::vis::vr
