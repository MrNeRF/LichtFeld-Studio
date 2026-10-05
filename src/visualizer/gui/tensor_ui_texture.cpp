/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/ui_texture.hpp"

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "rendering/image_tensor.hpp"
#include "rendering/tensor_frame_uploads.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <format>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace lfs::vis::gui {
    namespace {
        std::mutex registry_mutex;
        std::unordered_map<std::uintptr_t, std::weak_ptr<const lfs::core::Tensor>> registry;
        std::atomic<std::uint64_t> generation{0};

        // Texture updates are ordered on the tensor timeline instead of waiting
        // for the whole queue, as Tensor::to(Device::GPU) does.
        lfs::core::Tensor uploadToGpu(const lfs::core::Tensor& host) {
            static std::mutex mutex;
            static lfs::vis::TensorFrameUploads uploads;
            const auto contiguous = host.contiguous();
            std::lock_guard lock(mutex);
            return uploads.upload(std::as_bytes(std::span(static_cast<const std::byte*>(contiguous.data_ptr()),
                                                          contiguous.bytes())),
                                  contiguous.shape(), contiguous.dtype());
        }

        lfs::Error textureError(lfs::ErrorCode code, std::string detail) {
            return lfs::make_error({.code = code,
                                    .domain = lfs::ErrorDomain::Rendering,
                                    .detail = std::move(detail),
                                    .detection = LFS_SOURCE_SITE_CURRENT()});
        }

        bool validRegion(const UiTexture::Region& region) {
            return region.pixels && region.texture_width > 0 && region.texture_height > 0 &&
                   region.x >= 0 && region.y >= 0 && region.width > 0 && region.height > 0 &&
                   region.x + region.width <= region.texture_width &&
                   region.y + region.height <= region.texture_height &&
                   (region.channels == 1 || region.channels == 3 || region.channels == 4);
        }
    } // namespace

    void connectUiTextureGraphics(GraphicsContext*) {}

    struct UiTexture::Impl {
        std::shared_ptr<lfs::core::Tensor> image;
        std::vector<std::uint8_t> host_rgba;
        int width = 0;
        int height = 0;

        [[nodiscard]] std::uintptr_t id() const noexcept {
            return reinterpret_cast<std::uintptr_t>(this);
        }

        void publish() {
            std::lock_guard lock(registry_mutex);
            registry[id()] = image;
            generation.fetch_add(1, std::memory_order_relaxed);
        }

        void unpublish() {
            std::lock_guard lock(registry_mutex);
            registry.erase(id());
        }

        bool uploadRegions(std::span<const Region> regions) {
            if (regions.empty())
                return false;
            const int new_width = regions.front().texture_width;
            const int new_height = regions.front().texture_height;
            if (new_width <= 0 || new_height <= 0)
                return false;
            if (width != new_width || height != new_height || host_rgba.empty()) {
                width = new_width;
                height = new_height;
                host_rgba.assign(static_cast<std::size_t>(width) * height * 4, 0);
            }
            for (const auto& region : regions) {
                if (!validRegion(region) || region.texture_width != width ||
                    region.texture_height != height)
                    return false;
                for (int y = 0; y < region.height; ++y) {
                    const auto* source = region.pixels +
                        static_cast<std::size_t>(y) * region.width * region.channels;
                    auto* target = host_rgba.data() +
                        (static_cast<std::size_t>(region.y + y) * width + region.x) * 4;
                    expandToRgba8(source, target, static_cast<std::size_t>(region.width),
                                  region.channels);
                }
            }
            auto host = lfs::core::Tensor::from_blob(
                host_rgba.data(), {static_cast<std::size_t>(height),
                                   static_cast<std::size_t>(width), 4},
                lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            image = std::make_shared<lfs::core::Tensor>(uploadToGpu(host));
            publish();
            return true;
        }

        void setImage(lfs::core::Tensor tensor) {
            if (tensor.device() != lfs::core::Device::GPU)
                tensor = uploadToGpu(tensor);
            width = static_cast<int>(tensor.size(1));
            height = static_cast<int>(tensor.size(0));
            host_rgba.clear();
            image = std::make_shared<lfs::core::Tensor>(std::move(tensor));
            publish();
        }
    };

    UiTexture::~UiTexture() {
        reset();
        delete impl_;
    }

    UiTexture::UiTexture(UiTexture&& other) noexcept
        : impl_(std::exchange(other.impl_, nullptr)) {}

    UiTexture& UiTexture::operator=(UiTexture&& other) noexcept {
        if (this != &other) {
            reset();
            delete impl_;
            impl_ = std::exchange(other.impl_, nullptr);
        }
        return *this;
    }

    bool UiTexture::upload(const std::uint8_t* pixels, int width, int height, int channels) {
        return uploadRegion(pixels, width, height, 0, 0, width, height, channels);
    }

    bool UiTexture::uploadRegion(const std::uint8_t* pixels, int texture_width,
                                 int texture_height, int x, int y, int width,
                                 int height, int channels) {
        const Region region{pixels, texture_width, texture_height, x, y, width, height, channels};
        return uploadRegions(std::span(&region, 1));
    }

    bool UiTexture::uploadRegions(std::span<const Region> regions) {
        if (!impl_)
            impl_ = new Impl;
        try {
            return impl_->uploadRegions(regions);
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): the bool API reports texture upload failure to RmlUi.
            return false;
        }
    }

    bool UiTexture::upload(const lfs::core::Tensor& source, int expected_width,
                           int expected_height, bool flip_y) {
        try {
            auto prepared = lfs::rendering::prepareImageRgba8(source, flip_y);
            if (!prepared.is_valid() || int(prepared.size(1)) != expected_width ||
                int(prepared.size(0)) != expected_height)
                return false;
            if (!impl_)
                impl_ = new Impl;
            impl_->setImage(std::move(prepared));
            return true;
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): the bool API reports tensor conversion failure to RmlUi.
            return false;
        }
    }

    lfs::Result<void> UiTexture::uploadLinearRgba(const lfs::core::Tensor& source) {
        if (!source.is_valid() || source.ndim() != 3 || source.size(2) != 4 ||
            source.dtype() != lfs::core::DataType::Float32)
            return lfs::Result<void>::failure(textureError(
                lfs::ErrorCode::InvalidArgument,
                "Linear UI texture requires a Float32 [H,W,4] tensor"));
        try {
            if (!impl_)
                impl_ = new Impl;
            impl_->setImage(source.contiguous());
            return {};
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): normalize tensor failures into the public structured result.
            return lfs::Result<void>::failure(textureError(lfs::ErrorCode::Unavailable,
                                                            error.what()));
        }
    }

    std::shared_ptr<const lfs::core::Tensor> UiTexture::image() const {
        return impl_ ? impl_->image : nullptr;
    }

    std::uintptr_t UiTexture::textureId() const { return impl_ ? impl_->id() : 0; }
    bool UiTexture::valid() const { return impl_ && impl_->image && impl_->image->is_valid(); }

    std::string UiTexture::rmlSrcUrl(int width, int height) const {
        return valid() ? std::format("lfs-tensor://{}?w={}&h={}", textureId(), width, height)
                       : std::string{};
    }

    void UiTexture::reset() {
        if (!impl_)
            return;
        impl_->unpublish();
        impl_->image.reset();
        impl_->host_rgba.clear();
    }

    std::uint64_t uiTextureGeneration() { return generation.load(std::memory_order_relaxed); }

    std::shared_ptr<const lfs::core::Tensor> uiTextureImage(std::uintptr_t id) {
        std::lock_guard lock(registry_mutex);
        const auto found = registry.find(id);
        return found == registry.end() ? nullptr : found->second.lock();
    }
} // namespace lfs::vis::gui
