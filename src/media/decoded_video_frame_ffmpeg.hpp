// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/error.hpp"
#include "media/decoded_video_frame.hpp"
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/dovi_meta.h>
#include <libavutil/film_grain_params.h>
#include <libavutil/hdr_dynamic_metadata.h>
#include <libavutil/hwcontext.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/pixdesc.h>
}
#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
namespace lfs::media::detail {
    template <class T>
    inline const T* frameMetadata(const AVFrame* f, AVFrameSideDataType type) {
        const auto* side = av_frame_get_side_data(f, type);
        return side && side->size >= sizeof(T) ? reinterpret_cast<const T*>(side->data) : nullptr;
    }
    template <class T>
    inline const T* streamMetadata(const AVStream* s, AVPacketSideDataType type) {
        if (!s)
            return nullptr;
        const auto* side = av_packet_side_data_get(s->codecpar->coded_side_data, s->codecpar->nb_coded_side_data, type);
        return side && side->size >= sizeof(T) ? reinterpret_cast<const T*>(side->data) : nullptr;
    }
    inline HdrFrameMetadata describeHdrMetadata(const AVMasteringDisplayMetadata* master,
                                                const AVContentLightMetadata* light, const AVDynamicHDRPlus* plus) {
        HdrFrameMetadata out;
        if (master) {
            auto& m = out.mastering.emplace();
            m.luminance = master->has_luminance;
            m.primaries = master->has_primaries;
            m.min_luma = static_cast<float>(av_q2d(master->min_luminance));
            m.max_luma = static_cast<float>(av_q2d(master->max_luminance));
            for (int i = 0; i < 3; ++i)
                m.display[i] = {static_cast<float>(av_q2d(master->display_primaries[i][0])), static_cast<float>(av_q2d(master->display_primaries[i][1]))};
            m.white = {static_cast<float>(av_q2d(master->white_point[0])), static_cast<float>(av_q2d(master->white_point[1]))};
        }
        if (light)
            out.content_light = std::array{float(light->MaxCLL), float(light->MaxFALL)};
        if (plus && plus->application_version < 2 && plus->num_windows > 0) {
            auto& h = out.hdr10_plus.emplace();
            const auto& p = plus->params[0];
            float hist = 0;
            for (int i = 0; i < std::min<int>(p.num_distribution_maxrgb_percentiles, 15); ++i)
                hist = std::max(hist, static_cast<float>(av_q2d(p.distribution_maxrgb[i].percentile)));
            for (int i = 0; i < 3; ++i) {
                h.maxscl[i] = av_q2d(p.maxscl[i]);
            }
            h.average_maxrgb = av_q2d(p.average_maxrgb);
            h.histogram_max = hist;
            h.tone_mapping = p.tone_mapping_flag == 1;
            h.target_luma = float(av_q2d(plus->targeted_system_display_maximum_luminance));
            h.knee_x = float(av_q2d(p.knee_point_x));
            h.knee_y = float(av_q2d(p.knee_point_y));
            h.num_anchors = std::min<int>(p.num_bezier_curve_anchors, 15);
            for (int i = 0; i < h.num_anchors; ++i)
                h.anchors[i] = float(av_q2d(p.bezier_curve_anchors[i]));
        }
        return out;
    }
    inline Result<DecodedVideoFrame> describeDecodedVideoFrame(const AVFrame* frame, const AVStream* stream = nullptr) {
        const auto fail = [&](std::string text) -> Result<DecodedVideoFrame> { return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = std::move(text), .detection = LFS_SOURCE_SITE_CURRENT()}); };
        if (!frame || frame->width <= 0 || frame->height <= 0)
            return fail(std::format("Decoded frame requires positive dimensions (frame_present={}, width={}, height={})", frame != nullptr, frame ? frame->width : 0, frame ? frame->height : 0));
        const auto* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame->format));
        if (!desc)
            return fail(std::format("Decoded frame has unknown pixel format (format={})", frame->format));
        DecodedVideoFrame out;
        out.width = frame->width;
        out.height = frame->height;
        out.hardware = (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) != 0;
        AVPixelFormat layout = static_cast<AVPixelFormat>(frame->format);
        if (out.hardware) {
            if (!frame->hw_frames_ctx || !frame->hw_frames_ctx->data)
                return fail(std::format("Hardware decoded frame has no frames context (format={}, size={}x{})", frame->format, frame->width, frame->height));
            const auto* hw = reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data);
            layout = hw->sw_format;
            // VideoToolbox stores its borrowed CVPixelBufferRef in slot 3.
            // Hardware planes are not CPU-addressable; adapters must download
            // until a device/synchronization-aware import contract is available.
            out.hardware_handle = frame->data[frame->format == AV_PIX_FMT_VIDEOTOOLBOX ? 3 : 0];
            desc = av_pix_fmt_desc_get(layout);
            if (!desc)
                return fail(std::format("Hardware decoded frame has unknown software format (format={})", int(layout)));
        }
        out.format_name = desc->name;
        out.plane_count = av_pix_fmt_count_planes(layout);
        out.component_count = desc->nb_components;
        if (out.plane_count < 0 || out.plane_count > 4 || out.component_count > 4)
            return fail(std::format("Decoded frame layout exceeds plane/component capacity (planes={}, components={}, format={})", out.plane_count, out.component_count, out.format_name));
        out.chroma_w = desc->log2_chroma_w;
        out.chroma_h = desc->log2_chroma_h;
        out.big_endian = desc->flags & AV_PIX_FMT_FLAG_BE;
        out.rgb = desc->flags & AV_PIX_FMT_FLAG_RGB;
        out.xyz = out.format_name.starts_with("xyz");
        out.palette = desc->flags & AV_PIX_FMT_FLAG_PAL;
        out.bitstream = desc->flags & AV_PIX_FMT_FLAG_BITSTREAM;
        out.floating = desc->flags & AV_PIX_FMT_FLAG_FLOAT;
        out.bayer = desc->flags & AV_PIX_FMT_FLAG_BAYER;
        out.alpha = desc->flags & AV_PIX_FMT_FLAG_ALPHA;
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(60, 11, 100)
        out.alpha_mode = frame->alpha_mode == AVALPHA_MODE_PREMULTIPLIED ? AlphaMode::Premultiplied : frame->alpha_mode == AVALPHA_MODE_STRAIGHT ? AlphaMode::Independent
                                                                                                                                                 : AlphaMode::Unknown;
#endif
        for (int c = 0; c < out.component_count; ++c) {
            const auto& v = desc->comp[c];
            out.components[c] = {v.plane, v.step, v.offset, v.shift, v.depth};
        }
        for (int p = 0; p < out.plane_count; ++p) {
            bool chroma = false;
            for (int c = 1; c < 3 && c < out.component_count; ++c)
                chroma |= !out.rgb && out.components[c].plane == p && out.components[0].plane != p;
            out.planes[p] = {out.hardware ? nullptr : frame->data[p], out.hardware ? 0 : frame->linesize[p], chroma ? AV_CEIL_RSHIFT(out.width, out.chroma_w) : out.width, chroma ? AV_CEIL_RSHIFT(out.height, out.chroma_h) : out.height};
        }
        out.color_primaries = static_cast<ColorPrimaries>(frame->color_primaries);
        out.color_trc = static_cast<ColorTransfer>(frame->color_trc);
        out.colorspace = static_cast<ColorMatrix>(frame->colorspace);
        out.color_range = static_cast<ColorRange>(frame->color_range);
        out.chroma_location = static_cast<ChromaLocation>(frame->chroma_location);
        if (frame->crop_left > static_cast<size_t>(frame->width) || frame->crop_right > static_cast<size_t>(frame->width) - frame->crop_left || frame->crop_top > static_cast<size_t>(frame->height) || frame->crop_bottom > static_cast<size_t>(frame->height) - frame->crop_top)
            return fail(std::format("Decoded frame crop exceeds extent (size={}x{}, crop=[{}, {}, {}, {}])", frame->width, frame->height, frame->crop_left, frame->crop_top, frame->crop_right, frame->crop_bottom));
        out.crop = {float(frame->crop_left), float(frame->crop_top), float(static_cast<size_t>(frame->width) - frame->crop_right), float(static_cast<size_t>(frame->height) - frame->crop_bottom)};
        out.frame_hdr = describeHdrMetadata(frameMetadata<AVMasteringDisplayMetadata>(frame, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA), frameMetadata<AVContentLightMetadata>(frame, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL), frameMetadata<AVDynamicHDRPlus>(frame, AV_FRAME_DATA_DYNAMIC_HDR_PLUS));
        out.stream_hdr = describeHdrMetadata(streamMetadata<AVMasteringDisplayMetadata>(stream, AV_PKT_DATA_MASTERING_DISPLAY_METADATA), streamMetadata<AVContentLightMetadata>(stream, AV_PKT_DATA_CONTENT_LIGHT_LEVEL), streamMetadata<AVDynamicHDRPlus>(stream, AV_PKT_DATA_DYNAMIC_HDR10_PLUS));
        if (const auto* sd = av_frame_get_side_data(frame, AV_FRAME_DATA_ICC_PROFILE))
            out.icc_profile = {sd->data, sd->size};
        if (const auto* sd = av_frame_get_side_data(frame, AV_FRAME_DATA_DOVI_RPU_BUFFER))
            out.dovi_rpu = {sd->data, sd->size};
        if (const auto* fg = frameMetadata<AVFilmGrainParams>(frame, AV_FRAME_DATA_FILM_GRAIN_PARAMS)) {
            auto& grain = out.film_grain.emplace();
            grain.seed = fg->seed;
            if (fg->type == AV_FILM_GRAIN_PARAMS_AV1) {
                auto& g = grain.av1.emplace();
                const auto& a = fg->codec.aom;
                g.num_points_y = a.num_y_points;
                g.chroma_scaling_from_luma = a.chroma_scaling_from_luma;
                g.scaling_shift = a.scaling_shift;
                g.ar_coeff_lag = a.ar_coeff_lag;
                g.ar_coeff_shift = a.ar_coeff_shift;
                g.grain_scale_shift = a.grain_scale_shift;
                g.overlap = a.overlap_flag;
                std::memcpy(g.points_y, a.y_points, sizeof(g.points_y));
                std::memcpy(g.points_uv, a.uv_points, sizeof(g.points_uv));
                std::memcpy(g.num_points_uv, a.num_uv_points, sizeof(g.num_points_uv));
                std::memcpy(g.ar_coeffs_y, a.ar_coeffs_y, sizeof(g.ar_coeffs_y));
                std::memcpy(g.ar_coeffs_uv, a.ar_coeffs_uv, sizeof(g.ar_coeffs_uv));
                for (int c = 0; c < 2; ++c) {
                    g.uv_mult[c] = static_cast<int8_t>(a.uv_mult[c]);
                    g.uv_mult_luma[c] = static_cast<int8_t>(a.uv_mult_luma[c]);
                    g.uv_offset[c] = static_cast<int16_t>(a.uv_offset[c]);
                }
            } else if (fg->type == AV_FILM_GRAIN_PARAMS_H274) {
                auto& g = grain.h274.emplace();
                const auto& a = fg->codec.h274;
                g.model_id = a.model_id;
                g.blending_mode_id = a.blending_mode_id;
                g.log2_scale_factor = a.log2_scale_factor;
                for (int c = 0; c < 3; ++c) {
                    g.component_model_present[c] = a.component_model_present[c];
                    g.num_intensity_intervals[c] = a.num_intensity_intervals[c];
                    g.num_model_values[c] = a.num_model_values[c];
                    g.lower[c] = a.intensity_interval_lower_bound[c];
                    g.upper[c] = a.intensity_interval_upper_bound[c];
                    g.model[c] = a.comp_model_value[c];
                }
            }
        }
        if (const auto* metadata = frameMetadata<AVDOVIMetadata>(frame, AV_FRAME_DATA_DOVI_METADATA)) {
            const auto* side = av_frame_get_side_data(frame, AV_FRAME_DATA_DOVI_METADATA);
            const auto fits = [&](size_t offset, size_t bytes) { return offset <= side->size && bytes <= side->size - offset; };
            if (!fits(metadata->header_offset, sizeof(AVDOVIRpuDataHeader)) || !fits(metadata->mapping_offset, sizeof(AVDOVIDataMapping)) || !fits(metadata->color_offset, sizeof(AVDOVIColorMetadata)) || metadata->num_ext_blocks < 0 || metadata->num_ext_blocks > AV_DOVI_MAX_EXT_BLOCKS ||
                (metadata->num_ext_blocks && (metadata->ext_block_size < sizeof(AVDOVIDmData) || metadata->ext_block_size > side->size / static_cast<size_t>(metadata->num_ext_blocks) || !fits(metadata->ext_block_offset, metadata->ext_block_size * metadata->num_ext_blocks))))
                return fail(std::format("Dolby Vision metadata offsets exceed storage (bytes={}, header={}, mapping={}, color={}, ext_count={}, ext_size={})", side->size, metadata->header_offset, metadata->mapping_offset, metadata->color_offset, metadata->num_ext_blocks, metadata->ext_block_size));
            const auto* header = av_dovi_get_header(metadata);
            const auto* mapping = av_dovi_get_mapping(metadata);
            const auto* color = av_dovi_get_color(metadata);
            if (header->bl_bit_depth < 1 || header->bl_bit_depth > 16 || header->coef_log2_denom > 30)
                return fail(std::format("Invalid Dolby Vision coefficient precision (bit_depth={}, denominator={})", header->bl_bit_depth, header->coef_log2_denom));
            auto& d = out.dovi.emplace();
            d.disable_residual = header->disable_residual_flag;
            for (int i = 0; i < 3; ++i)
                d.offset[i] = float(av_q2d(color->ycc_to_rgb_offset[i]));
            for (int i = 0; i < 9; ++i) {
                d.nonlinear[i] = float(av_q2d(color->ycc_to_rgb_matrix[i]));
                d.linear[i] = float(av_q2d(color->rgb_to_lms_matrix[i]));
            }
            d.source_min_pq = color->source_min_pq / 4095.0f;
            d.source_max_pq = color->source_max_pq / 4095.0f;
            if (const auto* l1 = av_dovi_find_level(metadata, 1))
                d.level1 = std::array{l1->l1.max_pq / 4095.0f, l1->l1.avg_pq / 4095.0f};
            if (d.disable_residual)
                for (int c = 0; c < 3; ++c) {
                    const auto& curve = mapping->curves[c];
                    auto& dst = d.curves[c];
                    if (curve.num_pivots < 2 || curve.num_pivots > 9)
                        return fail(std::format("Invalid Dolby Vision pivot count (component={}, pivots={})", c, curve.num_pivots));
                    dst.num_pivots = curve.num_pivots;
                    const float ps = 1.f / ((1 << header->bl_bit_depth) - 1), cs = 1.f / (1 << header->coef_log2_denom);
                    for (int i = 0; i < dst.num_pivots; ++i)
                        dst.pivots[i] = ps * curve.pivots[i];
                    for (int i = 0; i + 1 < dst.num_pivots; ++i) {
                        dst.method[i] = curve.mapping_idc[i];
                        if (curve.mapping_idc[i] == AV_DOVI_MAPPING_POLYNOMIAL) {
                            for (int k = 0; k < 3; ++k)
                                dst.polynomial[i][k] = k <= curve.poly_order[i] ? cs * curve.poly_coef[i][k] : 0.f;
                        } else if (curve.mapping_idc[i] == AV_DOVI_MAPPING_MMR) {
                            if (curve.mmr_order[i] > 3)
                                return fail(std::format("Invalid Dolby Vision MMR order (component={}, piece={}, order={})", c, i, curve.mmr_order[i]));
                            dst.mmr_order[i] = curve.mmr_order[i];
                            dst.mmr_constant[i] = cs * curve.mmr_constant[i];
                            for (int j = 0; j < curve.mmr_order[i]; ++j)
                                for (int k = 0; k < 7; ++k)
                                    dst.mmr[i][j][k] = cs * curve.mmr_coef[i][j][k];
                        }
                    }
                }
        }
        auto valid = validateDecodedVideoFrame(out);
        if (!valid)
            return std::move(valid).error();
        return out;
    }
} // namespace lfs::media::detail
