// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/color_transfer.h"
#include "core/tensor.hpp"
#include "rendering/rasterizer/vulkan/src/display_color.h"
#include <algorithm>
#include <new>
#include <string>

namespace lfs::vis {
    struct FloatColorReadbackSettings {
        glm::vec3 background{};
        bool transparent = false;
        bool transmittance = false;
        uint32_t tone = 0;
        float exposure = 1;
    };
    // Raster color is display-referred, premultiplied float, before RGBA8
    // presentation and before PPISP/environment export post-processing. Native
    // Vulkan stores transmittance and leaves the background for composition;
    // the tensor rasterizer has already composed its background and alpha.
    // Reuse the actual presentation tone operators and the shared sRGB curve;
    // return relative linear BT709, straight alpha, without a display clamp.
    inline Result<core::Tensor> linearFloatColorReadback(
        const core::Tensor& source, const FloatColorReadbackSettings& settings) {
        if (!source.is_valid() || source.ndim() != 3 || source.size(2) != 4 ||
            (source.dtype() != core::DataType::Float16 && source.dtype() != core::DataType::Float32))
            return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::Rendering, .detail = "Float render readback requires HWC Float16/Float32 RGBA raster samples", .detection = LFS_SOURCE_SITE_CURRENT()});
        try {
            auto image = source.to(core::DataType::Float32).cpu().contiguous().clone();
            auto* pixels = image.ptr<float>();
            for (size_t pixel = 0; pixel < image.numel() / 4; ++pixel) {
                auto* rgba = pixels + pixel * 4;
                const float opacity = std::clamp(settings.transmittance ? 1.f - rgba[3] : rgba[3], 0.f, 1.f);
                glm::vec3 rgb{rgba[0], rgba[1], rgba[2]};
                if (settings.transmittance && !settings.transparent)
                    rgb += (1.f - opacity) * settings.background;
                if (settings.transparent)
                    rgb = opacity > 0.f ? rgb / opacity : glm::vec3(0);
                rgb = rendering::lfsDisplayTone(rgb, settings.tone, settings.exposure);
                for (size_t channel = 0; channel < 3; ++channel)
                    rgba[channel] = core::color::srgbToLinear(rgb[channel]);
                rgba[3] = settings.transparent ? opacity : 1.f;
            }
            return image;
        } catch (const Exception& error) {
            return error.error();
        } catch (const std::bad_alloc&) {
            return make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::Rendering, .detail = "Float color readback allocation failed", .detection = LFS_SOURCE_SITE_CURRENT()});
        } catch (const std::exception& error) {
            return make_error({.code = ErrorCode::Unavailable, .domain = ErrorDomain::Rendering, .detail = error.what(), .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    }
} // namespace lfs::vis
