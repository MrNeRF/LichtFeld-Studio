// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/decoded_video_frame.hpp"
#include <algorithm>
#include <format>
#include <limits>

namespace lfs::media {
    Result<void> validateDecodedVideoFrame(const DecodedVideoFrame& f) {
        const auto fail = [](std::string message) -> Result<void> {
            return Result<void>::failure(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = std::move(message), .detection = LFS_SOURCE_SITE_CURRENT()}));
        };
        if (f.width <= 0 || f.height <= 0 || f.plane_count < 1 || f.plane_count > 4 || f.component_count < 1 || f.component_count > 4 || f.chroma_w < 0 || f.chroma_w > 4 || f.chroma_h < 0 || f.chroma_h > 4)
            return fail(std::format("Invalid decoded frame layout (format={}, size={}x{}, planes={}, components={}, chroma={}x{})", f.format_name, f.width, f.height, f.plane_count, f.component_count, f.chroma_w, f.chroma_h));
        if (f.hardware)
            return f.hardware_handle ? Result<void>{} : fail(std::format("Missing decoded hardware handle (format={}, size={}x{})", f.format_name, f.width, f.height));
        // Unsupported packed-bit/palette layouts are rejected by the renderers.
        if (f.bitstream || f.palette || f.bayer)
            return {};
        std::array<size_t, 4> row_bytes{};
        for (int c = 0; c < f.component_count; ++c) {
            const auto& v = f.components[c];
            if (v.plane < 0 || v.plane >= f.plane_count || v.step <= 0 || v.step > 32 || v.offset < 0 || v.offset >= v.step || v.shift < 0 || v.depth < 1 || v.depth > 32 || v.shift > 32 || v.shift + v.depth > 8 * v.step)
                return fail(std::format("Invalid decoded component (component={}, plane={}, step={}, offset={}, shift={}, depth={})", c, v.plane, v.step, v.offset, v.shift, v.depth));
            const auto& p = f.planes[v.plane];
            const size_t bytes = (static_cast<size_t>(std::max(1, p.width)) - 1) * v.step + v.offset + (v.shift + v.depth + 7) / 8;
            row_bytes[v.plane] = std::max(row_bytes[v.plane], bytes);
        }
        for (int p = 0; p < f.plane_count; ++p) {
            const auto& plane = f.planes[p];
            bool chroma = false;
            for (int c = 1; c < 3 && c < f.component_count; ++c)
                chroma |= !f.rgb && f.components[c].plane == p && f.components[0].plane != p;
            const auto expected_width = chroma ? (static_cast<int64_t>(f.width) + (1 << f.chroma_w) - 1) >> f.chroma_w : f.width;
            const auto expected_height = chroma ? (static_cast<int64_t>(f.height) + (1 << f.chroma_h) - 1) >> f.chroma_h : f.height;
            if (!plane.data || plane.width != expected_width || plane.height != expected_height || plane.pitch == std::numeric_limits<std::ptrdiff_t>::min())
                return fail(std::format("Invalid decoded plane storage (plane={}, size={}x{}, pitch={}, row_bytes={}, data_present={})", p, plane.width, plane.height, plane.pitch, row_bytes[p], plane.data != nullptr));
            const size_t pitch = plane.pitch < 0 ? static_cast<size_t>(-plane.pitch) : static_cast<size_t>(plane.pitch);
            if (!row_bytes[p] || pitch < row_bytes[p] || pitch > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / static_cast<size_t>(plane.height))
                return fail(std::format("Invalid decoded plane stride (plane={}, pitch={}, row_bytes={}, height={})", p, plane.pitch, row_bytes[p], plane.height));
        }
        for (const auto* metadata : {&f.frame_hdr, &f.stream_hdr})
            if (metadata->hdr10_plus && (metadata->hdr10_plus->num_anchors < 0 || metadata->hdr10_plus->num_anchors > 15))
                return fail(std::format("Invalid HDR10+ anchor count (anchors={})", metadata->hdr10_plus->num_anchors));
        if (f.dovi && f.dovi->disable_residual)
            for (int c = 0; c < 3; ++c) {
                const auto& curve = f.dovi->curves[c];
                if (curve.num_pivots < 2 || curve.num_pivots > 9)
                    return fail(std::format("Invalid Dolby Vision pivot count (component={}, pivots={})", c, curve.num_pivots));
                for (int i = 0; i + 1 < curve.num_pivots; ++i)
                    if (curve.method[i] < 0 || curve.method[i] > 1 || (curve.method[i] == 1 && (curve.mmr_order[i] < 1 || curve.mmr_order[i] > 3)))
                        return fail(std::format("Invalid Dolby Vision mapping (component={}, piece={}, method={}, mmr_order={})", c, i, curve.method[i], curve.mmr_order[i]));
            }
        if (f.film_grain && f.film_grain->av1) {
            const auto& g = *f.film_grain->av1;
            if (g.num_points_y < 0 || g.num_points_y > 14 || g.num_points_uv[0] < 0 || g.num_points_uv[0] > 10 || g.num_points_uv[1] < 0 || g.num_points_uv[1] > 10 || g.ar_coeff_lag < 0 || g.ar_coeff_lag > 3)
                return fail(std::format("Invalid AV1 grain counts (y={}, uv=[{}, {}], ar_lag={})", g.num_points_y, g.num_points_uv[0], g.num_points_uv[1], g.ar_coeff_lag));
        }
        if (f.film_grain && f.film_grain->h274)
            for (int c = 0; c < 3; ++c) {
                const auto& g = *f.film_grain->h274;
                if (g.component_model_present[c] && (!g.lower[c] || !g.upper[c] || !g.model[c] || g.num_intensity_intervals[c] > 256 || g.num_model_values[c] > 6))
                    return fail(std::format("Invalid H274 grain storage (component={}, intervals={}, model_values={})", c, g.num_intensity_intervals[c], g.num_model_values[c]));
            }
        return {};
    }
} // namespace lfs::media
