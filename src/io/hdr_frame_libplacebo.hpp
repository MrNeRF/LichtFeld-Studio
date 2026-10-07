// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/decoded_video_frame.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <libplacebo/renderer.h>
#include <libplacebo/utils/dolbyvision.h>
#include <libplacebo/utils/upload.h>
namespace lfs::io::detail {
    inline pl_color_system colorSystem(media::ColorMatrix code) {
        switch (code) {
        case media::ColorMatrix::Rgb: return PL_COLOR_SYSTEM_RGB;
        case media::ColorMatrix::Bt709: return PL_COLOR_SYSTEM_BT_709;
        case media::ColorMatrix::Bt470Bg:
        case media::ColorMatrix::Smpte170M: return PL_COLOR_SYSTEM_BT_601;
        case media::ColorMatrix::Smpte240M: return PL_COLOR_SYSTEM_SMPTE_240M;
        case media::ColorMatrix::Bt2020Ncl: return PL_COLOR_SYSTEM_BT_2020_NC;
        case media::ColorMatrix::Bt2020Cl: return PL_COLOR_SYSTEM_BT_2020_C;
        case media::ColorMatrix::Ictcp: return PL_COLOR_SYSTEM_BT_2100_PQ;
        case media::ColorMatrix::Ycgco: return PL_COLOR_SYSTEM_YCGCO;
        case media::ColorMatrix::YcgcoRe: return PL_COLOR_SYSTEM_YCGCO_RE;
        case media::ColorMatrix::YcgcoRo: return PL_COLOR_SYSTEM_YCGCO_RO;
        default: return PL_COLOR_SYSTEM_UNKNOWN;
        }
    }
    inline pl_color_primaries colorPrimaries(media::ColorPrimaries code) {
        switch (code) {
        case media::ColorPrimaries::Bt709: return PL_COLOR_PRIM_BT_709;
        case media::ColorPrimaries::Bt470M: return PL_COLOR_PRIM_BT_470M;
        case media::ColorPrimaries::Bt470Bg: return PL_COLOR_PRIM_BT_601_625;
        case media::ColorPrimaries::Smpte170M:
        case media::ColorPrimaries::Smpte240M: return PL_COLOR_PRIM_BT_601_525;
        case media::ColorPrimaries::Film: return PL_COLOR_PRIM_FILM_C;
        case media::ColorPrimaries::Bt2020: return PL_COLOR_PRIM_BT_2020;
        case media::ColorPrimaries::Smpte428: return PL_COLOR_PRIM_CIE_1931;
        case media::ColorPrimaries::Smpte431: return PL_COLOR_PRIM_DCI_P3;
        case media::ColorPrimaries::Smpte432: return PL_COLOR_PRIM_DISPLAY_P3;
        case media::ColorPrimaries::JedecP22: return PL_COLOR_PRIM_EBU_3213;
        default: return PL_COLOR_PRIM_UNKNOWN;
        }
    }
    inline pl_color_transfer colorTransfer(media::ColorTransfer code) {
        switch (code) {
        case media::ColorTransfer::Bt709:
        case media::ColorTransfer::Smpte170M:
        case media::ColorTransfer::Smpte240M:
        case media::ColorTransfer::Iec61966_2_4:
        case media::ColorTransfer::Bt1361:
        case media::ColorTransfer::Bt2020_10:
        case media::ColorTransfer::Bt2020_12: return PL_COLOR_TRC_BT_1886;
        case media::ColorTransfer::Gamma22: return PL_COLOR_TRC_GAMMA22;
        case media::ColorTransfer::Gamma28: return PL_COLOR_TRC_GAMMA28;
        case media::ColorTransfer::Linear: return PL_COLOR_TRC_LINEAR;
        case media::ColorTransfer::Srgb: return PL_COLOR_TRC_SRGB;
        case media::ColorTransfer::Pq: return PL_COLOR_TRC_PQ;
        case media::ColorTransfer::Smpte428: return PL_COLOR_TRC_ST428;
        case media::ColorTransfer::Hlg: return PL_COLOR_TRC_HLG;
        default: return PL_COLOR_TRC_UNKNOWN;
        }
    }
    inline void applyHdr(pl_hdr_metadata& out, const media::HdrFrameMetadata& in) {
        if (const auto& m = in.mastering) {
            if (m->luminance) {
                out.min_luma = m->min_luma;
                out.max_luma = m->max_luma;
                if (out.max_luma < 5 || out.min_luma >= out.max_luma)
                    out.max_luma = out.min_luma = 0;
            }
            if (m->primaries)
                out.prim = {{m->display[0].x, m->display[0].y}, {m->display[1].x, m->display[1].y}, {m->display[2].x, m->display[2].y}, {m->white.x, m->white.y}};
        }
        if (in.content_light) {
            out.max_cll = (*in.content_light)[0];
            out.max_fall = (*in.content_light)[1];
        }
        if (const auto& h = in.hdr10_plus) {
            for (int c = 0; c < 3; ++c)
                out.scene_max[c] = h->maxscl[c] ? static_cast<float>(10000 * h->maxscl[c]) : h->histogram_max * 10000;
            out.scene_avg = static_cast<float>(10000 * h->average_maxrgb);
            if (h->tone_mapping) {
                out.ootf.target_luma = h->target_luma;
                out.ootf.knee_x = h->knee_x;
                out.ootf.knee_y = h->knee_y;
                out.ootf.num_anchors = h->num_anchors;
                for (int i = 0; i < h->num_anchors; ++i)
                    out.ootf.anchors[i] = h->anchors[i];
            }
        }
    }
    inline bool mapDecodedFrame(pl_gpu gpu, pl_frame& out, pl_dovi_metadata& dovi, pl_tex* textures,
                                const media::DecodedVideoFrame& frame, std::string& error) {
        const auto valid = media::validateDecodedVideoFrame(frame);
        if (!valid) {
            error = valid.error().detail();
            return false;
        }
        if (frame.hardware || frame.palette || frame.bitstream || frame.bayer || frame.plane_count < 1 || frame.plane_count > 4 || frame.component_count < 1 || frame.component_count > 4) {
            error = std::format("libplacebo requires an uploadable decoded frame (format={}, hardware={}, planes={}, components={})", frame.format_name, frame.hardware, frame.plane_count, frame.component_count);
            return false;
        }
        out = {};
        out.num_planes = frame.plane_count;
        out.crop = {frame.crop[0], frame.crop[1], frame.crop[2], frame.crop[3]};
        out.repr.sys = colorSystem(frame.colorspace);
        out.repr.levels = frame.color_range == media::ColorRange::Full ? PL_COLOR_LEVELS_FULL : frame.color_range == media::ColorRange::Limited ? PL_COLOR_LEVELS_LIMITED
                                                                                                                                                : PL_COLOR_LEVELS_UNKNOWN;
        out.repr.alpha = !frame.alpha ? PL_ALPHA_NONE : frame.alpha_mode == media::AlphaMode::Premultiplied ? PL_ALPHA_PREMULTIPLIED
                                                    : frame.alpha_mode == media::AlphaMode::Independent     ? PL_ALPHA_INDEPENDENT
                                                                                                            : PL_ALPHA_UNKNOWN;
        out.repr.bits.color_depth = frame.components[0].depth;
        out.color.primaries = colorPrimaries(frame.color_primaries);
        out.color.transfer = colorTransfer(frame.color_trc);
        applyHdr(out.color.hdr, frame.frame_hdr);
        if (frame.colorspace == media::ColorMatrix::Ictcp && frame.color_trc == media::ColorTransfer::Hlg)
            out.repr.sys = PL_COLOR_SYSTEM_BT_2100_HLG;
        else if (frame.xyz)
            out.repr.sys = PL_COLOR_SYSTEM_XYZ;
        else if (frame.rgb) {
            out.repr.sys = PL_COLOR_SYSTEM_RGB;
            out.repr.levels = PL_COLOR_LEVELS_FULL;
        } else if (!pl_color_system_is_ycbcr_like(out.repr.sys))
            out.repr.sys = pl_color_system_guess_ycbcr(frame.width, frame.height);
        if (!frame.icc_profile.empty()) {
            out.profile.data = frame.icc_profile.data();
            out.profile.len = frame.icc_profile.size();
            pl_icc_profile_compute_signature(&out.profile);
        }
        if (const auto& g = frame.film_grain) {
            out.film_grain.seed = g->seed;
            if (const auto& a = g->av1) {
                out.film_grain.type = PL_FILM_GRAIN_AV1;
                auto& dst = out.film_grain.params.av1;
                dst.num_points_y = a->num_points_y;
                dst.chroma_scaling_from_luma = a->chroma_scaling_from_luma;
                dst.scaling_shift = a->scaling_shift;
                dst.ar_coeff_lag = a->ar_coeff_lag;
                dst.ar_coeff_shift = a->ar_coeff_shift;
                dst.grain_scale_shift = a->grain_scale_shift;
                dst.overlap = a->overlap;
                std::memcpy(dst.points_y, a->points_y, sizeof(dst.points_y));
                std::memcpy(dst.points_uv, a->points_uv, sizeof(dst.points_uv));
                std::memcpy(dst.num_points_uv, a->num_points_uv, sizeof(dst.num_points_uv));
                std::memcpy(dst.ar_coeffs_y, a->ar_coeffs_y, sizeof(dst.ar_coeffs_y));
                std::memcpy(dst.ar_coeffs_uv, a->ar_coeffs_uv, sizeof(dst.ar_coeffs_uv));
                for (int c = 0; c < 2; ++c) {
                    dst.uv_mult[c] = a->uv_mult[c];
                    dst.uv_mult_luma[c] = a->uv_mult_luma[c];
                    dst.uv_offset[c] = a->uv_offset[c];
                }
            } else if (const auto& a = g->h274) {
                out.film_grain.type = PL_FILM_GRAIN_H274;
                auto& dst = out.film_grain.params.h274;
                dst.model_id = a->model_id;
                dst.blending_mode_id = a->blending_mode_id;
                dst.log2_scale_factor = a->log2_scale_factor;
                for (int c = 0; c < 3; ++c) {
                    dst.component_model_present[c] = a->component_model_present[c];
                    dst.num_intensity_intervals[c] = a->num_intensity_intervals[c];
                    dst.num_model_values[c] = a->num_model_values[c];
                    dst.intensity_interval_lower_bound[c] = a->lower[c];
                    dst.intensity_interval_upper_bound[c] = a->upper[c];
                    dst.comp_model_value[c] = a->model[c];
                }
            }
        }
        if (frame.dovi && frame.dovi->disable_residual) {
            const auto& d = *frame.dovi;
            for (int c = 0; c < 3; ++c) {
                dovi.nonlinear_offset[c] = d.offset[c];
                const auto& in = d.curves[c];
                auto& dst = dovi.comp[c];
                dst.num_pivots = in.num_pivots;
                for (int i = 0; i < in.num_pivots; ++i)
                    dst.pivots[i] = in.pivots[i];
                for (int i = 0; i + 1 < in.num_pivots; ++i) {
                    dst.method[i] = in.method[i];
                    dst.mmr_order[i] = in.mmr_order[i];
                    dst.mmr_constant[i] = in.mmr_constant[i];
                    for (int k = 0; k < 3; ++k) {
                        dst.poly_coeffs[i][k] = in.polynomial[i][k];
                        for (int j = 0; j < 7; ++j)
                            dst.mmr_coeffs[i][k][j] = in.mmr[i][k][j];
                    }
                }
            }
            for (int i = 0; i < 9; ++i) {
                dovi.nonlinear.m[i / 3][i % 3] = d.nonlinear[i];
                dovi.linear.m[i / 3][i % 3] = d.linear[i];
            }
            out.repr.dovi = &dovi;
            out.repr.sys = PL_COLOR_SYSTEM_DOLBYVISION;
            out.color.primaries = PL_COLOR_PRIM_BT_2020;
            out.color.transfer = PL_COLOR_TRC_PQ;
            out.color.hdr.min_luma = pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NITS, d.source_min_pq);
            out.color.hdr.max_luma = pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NITS, d.source_max_pq);
            if (d.level1) {
                out.color.hdr.max_pq_y = (*d.level1)[0];
                out.color.hdr.avg_pq_y = (*d.level1)[1];
            }
        }
#ifdef PL_HAVE_LIBDOVI
        if (!frame.dovi_rpu.empty())
            pl_hdr_metadata_from_dovi_rpu(&out.color.hdr, frame.dovi_rpu.data(), frame.dovi_rpu.size());
#endif
        applyHdr(out.color.hdr, frame.stream_hdr);
        std::array<pl_plane_data, 4> planes{}, aligned{};
        pl_bit_encoding common{};
        bool all_aligned = true;
        for (int p = 0; p < frame.plane_count; ++p) {
            auto& data = planes[p];
            data.swapped = frame.big_endian;
            data.type = frame.floating ? PL_FMT_FLOAT : PL_FMT_UNORM;
            int sizes[4]{}, shifts[4]{};
            for (int c = 0; c < frame.component_count; ++c) {
                const auto& comp = frame.components[c];
                if (comp.plane != p)
                    continue;
                if ((data.swapped && comp.shift) || (data.pixel_stride && data.pixel_stride != size_t(comp.step))) {
                    error = std::format("Unsupported decoded plane component layout (plane={}, component={}, step={}, pixel_stride={}, shift={})", p, c, comp.step, data.pixel_stride, comp.shift);
                    return false;
                }
                sizes[c] = comp.depth;
                shifts[c] = comp.shift + 8 * comp.offset;
                data.pixel_stride = comp.step;
            }
            pl_plane_data_from_comps(&data, sizes, shifts);
            aligned[p] = data;
            pl_bit_encoding bits{};
            if (!pl_plane_data_align(&aligned[p], &bits))
                all_aligned = false;
            if (p == 0)
                common = bits;
            else if (!pl_bit_encoding_equal(&common, &bits))
                all_aligned = false;
        }
        if (all_aligned) {
            planes = aligned;
            out.repr.bits = common;
        } else
            out.repr.bits = {};
        for (int p = 0; p < frame.plane_count; ++p) {
            const auto& input = frame.planes[p];
            auto& data = planes[p];
            if (!input.data || !input.pitch || input.width <= 0 || input.height <= 0) {
                error = std::format("Invalid decoded plane storage (plane={}, pitch={}, size={}x{}, data_present={})", p, input.pitch, input.width, input.height, input.data != nullptr);
                return false;
            }
            data.width = input.width;
            data.height = input.height;
            data.row_stride = std::abs(input.pitch);
            data.pixels = input.pitch < 0 ? input.data + input.pitch * (input.height - 1) : input.data;
            if (!pl_upload_plane(gpu, &out.planes[p], &textures[p], &data)) {
                error = std::format("libplacebo could not upload decoded plane (plane={}, format={}, pitch={}, size={}x{})", p, frame.format_name, input.pitch, input.width, input.height);
                return false;
            }
            out.planes[p].flipped = input.pitch < 0;
        }
        if (frame.chroma_w || frame.chroma_h) {
            pl_chroma_location location = PL_CHROMA_UNKNOWN;
            switch (frame.chroma_location) {
            case media::ChromaLocation::Left: location = PL_CHROMA_LEFT; break;
            case media::ChromaLocation::Center: location = PL_CHROMA_CENTER; break;
            case media::ChromaLocation::TopLeft: location = PL_CHROMA_TOP_LEFT; break;
            case media::ChromaLocation::Top: location = PL_CHROMA_TOP_CENTER; break;
            case media::ChromaLocation::BottomLeft: location = PL_CHROMA_BOTTOM_LEFT; break;
            case media::ChromaLocation::Bottom: location = PL_CHROMA_BOTTOM_CENTER; break;
            default: break;
            }
            pl_frame_set_chroma_location(&out, location);
        }
        return true;
    }
} // namespace lfs::io::detail
