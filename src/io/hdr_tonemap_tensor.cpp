/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "hdr_tonemap_tensor.hpp"
#include "core/gpu_kernel_module.hpp"
#include "core/tensor.hpp"
#include "core/tensor_readback.hpp"
#include "core/tensor_upload.hpp"
#include "hdr_tonemap_program.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <mutex>
#include <numbers>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace lfs::io {
    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::GpuKernelModule;
        using lfs::core::Tensor;
        using lfs::core::TensorReadback;
        using lfs::core::TensorUpload;

        // The constants and math below mirror libplacebo v7.360 (colorspace.c,
        // tone_mapping.c, shaders/colorspace.c) for what HdrLibplaceboRenderer
        // renders: pl_render_default_params into an 8-bit sRGB target.
        constexpr float PQ_M1 = 2610.0f / 4096 / 4;
        constexpr float PQ_M2 = 2523.0f / 4096 * 128;
        constexpr float PQ_C1 = 3424.0f / 4096;
        constexpr float PQ_C2 = 2413.0f / 4096 * 32;
        constexpr float PQ_C3 = 2392.0f / 4096 * 32;
        constexpr float SDR_WHITE = 203.0f;
        constexpr float SDR_BLACK = SDR_WHITE / 1000.0f;
        constexpr float HDR_BLACK = 1e-6f;
        constexpr float HLG_REF = 1000.0f / SDR_WHITE;
        constexpr float PQ_PEAK = float(10000.0 / SDR_WHITE) * SDR_WHITE;
        constexpr unsigned PQ_MAX = (1u << 14) - 1;
        constexpr size_t GAMUT_FLOATS = 48 * 32 * 256 * 3;
        constexpr size_t DOVI_COMPONENT = 226;

        using Matrix = std::array<std::array<float, 3>, 3>;
        using Float4 = std::array<float, 4>;
        using Uint4 = std::array<uint32_t, 4>;

        struct Chromaticity {
            float x = 0.0f;
            float y = 0.0f;
        };
        struct Primaries {
            Chromaticity red, green, blue, white;
        };
        constexpr Chromaticity D65{0.3127f, 0.3290f};
        constexpr Primaries BT709{{0.640f, 0.330f}, {0.300f, 0.600f}, {0.150f, 0.060f}, D65};
        constexpr Primaries BT2020{{0.708f, 0.292f}, {0.170f, 0.797f}, {0.131f, 0.046f}, D65};

        struct Constants {
            std::array<Uint4, 3> component{};
            Uint4 layout{};
            Uint4 flags{};
            Float4 chroma{};
            Float4 sample{};
            std::array<Float4, 3> decode{};
            std::array<Float4, 3> dovi_lms{};
            Float4 hlg{};
            Float4 luma{};
            std::array<Float4, 3> rgb_to_lms{};
            std::array<Float4, 3> lms_to_rgb{};
            std::array<Float4, 3> tone{};
            Float4 target{};
            Float4 gamut{};
            std::array<Float4, 3> gamut_in{};
            std::array<Float4, 3> gamut_in_inverse{};
            std::array<Float4, 3> gamut_out{};
            std::array<Float4, 3> gamut_out_inverse{};
        };
        static_assert(sizeof(Constants) == 38 * 16);

        struct Parameters {
            uint64_t planes = 0;
            uint64_t source = 0;
            uint64_t image = 0;
            uint64_t constants = 0;
            uint64_t dovi = 0;
            uint64_t lut = 0;
            uint64_t dither = 0;
            uint64_t output = 0;
            uint64_t tiles = 0;
            uint64_t pq_table = 0;
            uint64_t scaler_lut = 0;
            uint64_t chroma_image = 0;
            uint32_t source_width = 0;
            uint32_t source_height = 0;
            uint32_t target_width = 0;
            uint32_t target_height = 0;
            uint32_t mode = 0;
            uint32_t taps = 0;
            float blur = 1.0f;
            uint32_t gamut_map = 0;
        };
        constexpr uint32_t FUSED_PEAK = 4;

        bool equal(const Chromaticity a, const Chromaticity b) {
            return std::fabs(a.x - b.x) < 1e-6 && std::fabs(a.y - b.y) < 1e-6;
        }

        bool equal(const Primaries& a, const Primaries& b) {
            return equal(a.red, b.red) && equal(a.green, b.green) && equal(a.blue, b.blue) &&
                   equal(a.white, b.white);
        }

        float side(const Chromaticity p, const Chromaticity a, const Chromaticity b) {
            return (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
        }

        // pl_primaries_valid: a nonzero triangle containing its white point.
        bool valid(const Primaries& p) {
            const float area = (p.blue.x - p.green.x) * (p.red.y - p.green.y) -
                               (p.red.x - p.green.x) * (p.blue.y - p.green.y);
            const float d1 = side(p.white, p.red, p.green);
            const float d2 = side(p.white, p.green, p.blue);
            const float d3 = side(p.white, p.blue, p.red);
            const bool negative = d1 < -1e-6f || d2 < -1e-6f || d3 < -1e-6f;
            const bool positive = d1 > 1e-6f || d2 > 1e-6f || d3 > 1e-6f;
            return std::fabs(area) > 1e-6 && !(negative && positive);
        }

        Matrix multiply(const Matrix& a, const Matrix& b) {
            Matrix result{};
            for (size_t i = 0; i < 3; ++i)
                for (size_t j = 0; j < 3; ++j)
                    result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            return result;
        }

        Matrix invert(const Matrix& m) {
            const double m00 = m[0][0], m01 = m[0][1], m02 = m[0][2];
            const double m10 = m[1][0], m11 = m[1][1], m12 = m[1][2];
            const double m20 = m[2][0], m21 = m[2][1], m22 = m[2][2];
            const double a00 = m11 * m22 - m21 * m12, a01 = -(m01 * m22 - m21 * m02), a02 = m01 * m12 - m11 * m02;
            const double a10 = -(m10 * m22 - m20 * m12), a11 = m00 * m22 - m20 * m02, a12 = -(m00 * m12 - m10 * m02);
            const double a20 = m10 * m21 - m20 * m11, a21 = -(m00 * m21 - m20 * m01), a22 = m00 * m11 - m10 * m01;
            const double det = 1.0 / (m00 * a00 + m10 * a01 + m20 * a02);
            return {{{float(det * a00), float(det * a01), float(det * a02)},
                     {float(det * a10), float(det * a11), float(det * a12)},
                     {float(det * a20), float(det * a21), float(det * a22)}}};
        }

        float cieX(const Chromaticity c) { return c.x / c.y; }
        float cieZ(const Chromaticity c) { return (1 - c.x - c.y) / c.y; }

        Matrix rgbToXyz(const Primaries& p) {
            const std::array<float, 4> x{cieX(p.red), cieX(p.green), cieX(p.blue), cieX(p.white)};
            const std::array<float, 4> z{cieZ(p.red), cieZ(p.green), cieZ(p.blue), cieZ(p.white)};
            const Matrix inverse = invert({{{x[0], x[1], x[2]}, {1, 1, 1}, {z[0], z[1], z[2]}}});
            Matrix result{};
            for (size_t i = 0; i < 3; ++i) {
                const float s = inverse[i][0] * x[3] + inverse[i][1] * 1 + inverse[i][2] * z[3];
                result[0][i] = s * x[i];
                result[1][i] = s * 1;
                result[2][i] = s * z[i];
            }
            return result;
        }

        // pl_ipt_rgb2lms: HPE LMS with 4% crosstalk, CAT16-adapted to D65.
        Matrix rgbToLms(const Primaries& p) {
            constexpr Matrix HPE{{{0.40024f, 0.70760f, -0.08081f}, {-0.22630f, 1.16532f, 0.04570f}, {0.0f, 0.0f, 0.91822f}}};
            constexpr float c = 0.04f;
            Matrix m = multiply({{{1 - 2 * c, c, c}, {c, 1 - 2 * c, c}, {c, c, 1 - 2 * c}}}, HPE);
            if (!equal(p.white, D65)) {
                constexpr Matrix CAT16{{{0.401288f, 0.650173f, -0.051461f},
                                        {-0.250268f, 1.204414f, 0.045854f},
                                        {-0.002079f, 0.048952f, 0.953127f}}};
                Matrix adapt{};
                for (size_t i = 0; i < 3; ++i) {
                    const float source = CAT16[i][0] * cieX(p.white) + CAT16[i][1] * 1 + CAT16[i][2] * cieZ(p.white);
                    const float target = CAT16[i][0] * cieX(D65) + CAT16[i][1] * 1 + CAT16[i][2] * cieZ(D65);
                    adapt[i][i] = target / source;
                }
                m = multiply(multiply(m, invert(CAT16)), multiply(adapt, CAT16));
            }
            return multiply(m, rgbToXyz(p));
        }

        std::array<Float4, 3> rows(const Matrix& m, const std::array<float, 3> offset = {}) {
            return {{{m[0][0], m[0][1], m[0][2], offset[0]},
                     {m[1][0], m[1][1], m[1][2], offset[1]},
                     {m[2][0], m[2][1], m[2][2], offset[2]}}};
        }

        float mix(const float a, const float b, const float x) { return x * b + (1 - x) * a; }
        float clampf(const float x, const float lo, const float hi) { return std::fmin(std::fmax(x, lo), hi); }

        float smoothstep(const float edge0, const float edge1, float x) {
            x = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
            return x * x * (3.0f - 2.0f * x);
        }

        float pqOetf(float x) {
            x = std::pow(std::fmax(x, 0.0f), PQ_M1);
            x = (PQ_C1 + PQ_C2 * x) / (1.0f + PQ_C3 * x);
            return std::pow(x, PQ_M2);
        }

        float pqEotf(float x) {
            x = std::pow(std::fmax(x, 0.0f), 1.0f / PQ_M2);
            x = std::fmax(x - PQ_C1, 0.0f) / (PQ_C2 - PQ_C3 * x);
            return std::pow(x, 1.0f / PQ_M1);
        }

        // gamut_mapping.c's PQ EOTF: 1024 steps, linearly interpolated.
        const std::vector<float>& pqTable() {
            static const std::vector<float> table = [] {
                std::vector<float> values(1025, 1.0f);
                for (size_t i = 0; i < 1024; ++i) {
                    double x = std::pow(i / 1023.0, 1.0 / double(PQ_M2));
                    x = std::max(x - double(PQ_C1), 0.0) / (double(PQ_C2) - double(PQ_C3) * x);
                    values[i] = float(std::pow(x, 1.0 / double(PQ_M1)));
                }
                return values;
            }();
            return table;
        }

        float pqEotfTable(const float x) {
            const float index = clampf(x, 0.0f, 1.0f) * 1023;
            const size_t i = static_cast<size_t>(std::floor(index));
            return mix(pqTable()[i], pqTable()[i + 1], index - std::floor(index));
        }

        // pl_hdr_rescale between nits and PQ.
        float nitsToPq(const float nits) {
            return nits ? pqOetf(std::fmax(nits, 0.0f) / SDR_WHITE * (SDR_WHITE / 10000.0f)) : 0.0f;
        }
        float pqToNits(const float pq) {
            return pq ? pqEotf(pq) * 10000.0f / SDR_WHITE * SDR_WHITE : 0.0f;
        }

        enum class Transfer { PQ,
                              HLG };

        struct Hdr {
            Primaries prim;
            float min_luma = 0.0f;
            float max_luma = 0.0f;
            std::array<float, 3> scene_max{};
            float scene_avg = 0.0f;
            float max_pq_y = 0.0f;
            float avg_pq_y = 0.0f;
        };

        struct Space {
            Primaries primaries = BT709;
            Transfer transfer = Transfer::PQ;
            Hdr hdr;
        };

        void mapHdr(Hdr& hdr, const media::HdrFrameMetadata& metadata) {
            if (const auto& master = metadata.mastering) {
                if (master->luminance) {
                    hdr.max_luma = master->max_luma;
                    hdr.min_luma = master->min_luma;
                    if (hdr.max_luma < 5.f || hdr.min_luma >= hdr.max_luma)
                        hdr.max_luma = hdr.min_luma = 0.f;
                }
                if (master->primaries)
                    hdr.prim = {{master->display[0].x, master->display[0].y},
                                {master->display[1].x, master->display[1].y},
                                {master->display[2].x, master->display[2].y},
                                {master->white.x, master->white.y}};
            }
            if (const auto& plus = metadata.hdr10_plus) {
                for (int c = 0; c < 3; ++c) {
                    hdr.scene_max[c] = 10000 * static_cast<float>(plus->maxscl[c]);
                    if (!hdr.scene_max[c])
                        hdr.scene_max[c] = plus->histogram_max * 10000;
                }
                hdr.scene_avg = 10000 * static_cast<float>(plus->average_maxrgb);
            }
        }

        // pl_color_space_infer_map of an HDR source towards the sRGB target.
        void infer(Space& space) {
            Hdr& hdr = space.hdr;
            const bool unknown_contrast = !hdr.min_luma;
            const float hdr_max = pqToNits(1.0f);
            float max_luma = hdr.max_luma ? clampf(hdr.max_luma, HDR_BLACK, hdr_max) : 0.0f;
            float min_luma = hdr.min_luma ? clampf(hdr.min_luma, HDR_BLACK, hdr_max) : 0.0f;
            if ((max_luma && min_luma >= max_luma) || min_luma >= hdr_max)
                min_luma = max_luma = 0.0f;
            if (space.transfer == Transfer::PQ)
                min_luma = HDR_BLACK;
            if (!max_luma)
                max_luma = space.transfer == Transfer::HLG ? 1000.0f : PQ_PEAK;
            hdr.max_luma = max_luma;
            if (!hdr.min_luma)
                hdr.min_luma = min_luma ? min_luma : HDR_BLACK;
            if (!valid(hdr.prim))
                hdr.prim = space.primaries;
            // HLG is black scaled: an untagged black point follows the target.
            if (unknown_contrast && space.transfer == Transfer::HLG)
                hdr.min_luma = SDR_BLACK;
        }

        // pl_color_space_nominal_luma_ex with PL_HDR_METADATA_ANY in PQ.
        void sourceRange(const Space& space, float& min_luma, float& max_luma, float& avg_luma) {
            const Hdr& hdr = space.hdr;
            min_luma = nitsToPq(hdr.min_luma);
            max_luma = nitsToPq(hdr.max_luma);
            avg_luma = 0.0f;
            const float scene = std::max({hdr.scene_max[0], hdr.scene_max[1], hdr.scene_max[2]});
            if (hdr.scene_avg && scene) {
                const Matrix xyz = rgbToXyz(hdr.prim);
                const float luma = xyz[1][0] * hdr.scene_max[0] + xyz[1][1] * hdr.scene_max[1] +
                                   xyz[1][2] * hdr.scene_max[2];
                max_luma = nitsToPq(luma);
                avg_luma = nitsToPq(luma / scene * hdr.scene_avg);
            }
            if (hdr.max_pq_y && hdr.avg_pq_y) {
                max_luma = hdr.max_pq_y;
                avg_luma = hdr.avg_pq_y;
            }
            const float hdr_min = nitsToPq(HDR_BLACK);
            max_luma = max_luma ? clampf(max_luma, hdr_min, 1.0f) : 0.0f;
            min_luma = min_luma ? clampf(min_luma, hdr_min, 1.0f) : 0.0f;
            if ((max_luma && min_luma >= max_luma) || min_luma >= 1.0f)
                min_luma = max_luma = 0.0f;
            if (!max_luma)
                max_luma = nitsToPq(space.transfer == Transfer::HLG ? 1000.0f : PQ_PEAK);
            if (space.transfer == Transfer::PQ || !min_luma)
                min_luma = hdr_min;
            if (avg_luma)
                avg_luma = clampf(avg_luma, min_luma, max_luma);
        }

        // pl_tone_map_spline with the default constants, PQ in and out.
        std::array<Float4, 3> spline(const Space& space) {
            float in_min, in_max, in_avg;
            sourceRange(space, in_min, in_max, in_avg);
            float out_min = nitsToPq(SDR_BLACK);
            float out_max = nitsToPq(SDR_WHITE);
            in_max = std::fmax(in_max, std::fmin(out_max, nitsToPq(100.0f)));
            if (std::fabs(in_max - out_max) < 1e-6)
                out_max = in_max;
            if (std::fabs(in_min - out_min) < 1e-6)
                out_min = in_min;
            out_max = std::fmin(out_max, in_max);

            // st2094_pick_knee
            const float src_knee_min = mix(in_min, in_max, 0.1f), src_knee_max = mix(in_min, in_max, 0.8f);
            const float dst_knee_min = mix(out_min, out_max, 0.1f), dst_knee_max = mix(out_min, out_max, 0.8f);
            const float src_knee = clampf(in_avg ? in_avg : mix(in_min, in_max, 0.4f), src_knee_min, src_knee_max);
            const float target = (src_knee - in_min) / (in_max - in_min);
            const float adapted = mix(out_min, out_max, target);
            const float tuning = 1.0f - smoothstep(0.8f, 0.4f, target) * smoothstep(0.1f, 0.4f, target);
            const float adaptation = mix(0.4f, 1.0f, tuning);
            const float dst_knee = clampf(mix(src_knee, adapted, adaptation), dst_knee_min, dst_knee_max);

            float slope = (dst_knee - out_min) / (src_knee - in_min);
            const float ratio = clampf(1.5f * (in_max / out_max - 1.0f), 0.2f, 1.2f);
            slope = std::pow(slope, (1.0f - 0.5f) * ratio);
            const float i0 = in_min - src_knee, i1 = in_max - src_knee;
            const float o0 = out_min - dst_knee, o1 = out_max - dst_knee;
            const float t = 2 * i1 * i1;
            return {{{in_min, in_max, out_min, out_max},
                     {src_knee, dst_knee, (o0 - slope * i0) / (i0 * i0), slope},
                     {(slope * i1 - o1) / (i1 * t), -3 * (slope * i1 - o1) / t, slope, 0.0f}}};
        }

        bool primaries(const media::ColorPrimaries value, Primaries& out) {
            switch (value) {
            case media::ColorPrimaries::Unspecified:
            case media::ColorPrimaries::Bt709: out = BT709; return true;
            case media::ColorPrimaries::Bt2020: out = BT2020; return true;
            case media::ColorPrimaries::Smpte431: out = {{0.680f, 0.320f}, {0.265f, 0.690f}, {0.150f, 0.060f}, {0.3140f, 0.3510f}}; return true;
            case media::ColorPrimaries::Smpte432: out = {{0.680f, 0.320f}, {0.265f, 0.690f}, {0.150f, 0.060f}, D65}; return true;
            default: return false;
            }
        }

        bool lumaCoefficients(const media::DecodedVideoFrame* frame, std::array<float, 3>& out) {
            switch (frame->colorspace) {
            case media::ColorMatrix::Bt709: out = {0.2126f, 0.7152f, 0.0722f}; return true;
            case media::ColorMatrix::Bt470Bg:
            case media::ColorMatrix::Smpte170M: out = {0.2990f, 0.5870f, 0.1140f}; return true;
            case media::ColorMatrix::Smpte240M: out = {0.2122f, 0.7013f, 0.0865f}; return true;
            case media::ColorMatrix::Bt2020Ncl: out = {0.2627f, 0.6780f, 0.0593f}; return true;
            case media::ColorMatrix::Unspecified:
                // pl_color_system_guess_ycbcr
                out = frame->width >= 1280 || frame->height > 576 ? std::array{0.2126f, 0.7152f, 0.0722f}
                                                                  : std::array{0.2990f, 0.5870f, 0.1140f};
                return true;
            default: return false;
            }
        }

        // pl_map_dovi_metadata, packed per component: pivot count, 9 pivots,
        // then 8 pieces of method, 3 polynomial, MMR order, constant, 3x7 MMR.
        std::vector<float> packDovi(const media::DolbyVisionMetadata& metadata) {
            std::vector<float> packed(3 * DOVI_COMPONENT, 0.f);
            for (size_t c = 0; c < 3; ++c) {
                const auto& curve = metadata.curves[c];
                float* out = packed.data() + c * DOVI_COMPONENT;
                out[0] = curve.num_pivots;
                for (int i = 0; i < curve.num_pivots; ++i)
                    out[1 + i] = curve.pivots[i];
                for (int i = 0; i + 1 < curve.num_pivots; ++i) {
                    float* piece = out + 10 + i * 27;
                    piece[0] = curve.method[i];
                    if (curve.method[i] == 0) {
                        for (int k = 0; k < 3; ++k)
                            piece[1 + k] = curve.polynomial[i][k];
                    } else {
                        piece[4] = curve.mmr_order[i];
                        piece[5] = curve.mmr_constant[i];
                        for (int j = 0; j < curve.mmr_order[i]; ++j)
                            for (int k = 0; k < 7; ++k)
                                piece[6 + j * 7 + k] = curve.mmr[i][j][k];
                    }
                }
            }
            return packed;
        }

        struct Source {
            Space space;
            Constants constants;
            std::array<std::span<const uint8_t>, 4> planes;
            size_t plane_bytes = 0;
            std::vector<float> dovi;
        };

        // pl_map_avframe_ex (map_dovi) and pl_frame_copy_stream_props, then the
        // plane sampling and pl_shader_decode_color of pass_read_image.
        bool describe(const media::DecodedVideoFrame* frame, Source& source, std::string& error) {
            const auto valid = media::validateDecodedVideoFrame(*frame);
            if (!valid) {
                error = valid.error().detail();
                return false;
            }
            const auto* desc = frame;
            if (frame->component_count != 3 || frame->hardware || frame->big_endian || frame->rgb || frame->palette || frame->bitstream || frame->floating || frame->bayer || frame->components[0].depth + frame->components[0].shift > 16) {
                error = std::format("HDR tensor tonemapper requires a software YCbCr frame (format={}, components={}, hardware={})", frame->format_name, frame->component_count, frame->hardware);
                return false;
            }
            Constants& c = source.constants;
            const int depth = desc->components[0].depth;
            const int sample_depth = depth + desc->components[0].shift > 8 ? 16 : 8;
            const int chroma_width = ((frame->width + (1 << desc->chroma_w) - 1) >> desc->chroma_w);
            const int chroma_height = ((frame->height + (1 << desc->chroma_h) - 1) >> desc->chroma_h);
            std::array<uint32_t, 4> plane_offset{};
            for (int plane = 0; plane < frame->plane_count; ++plane) {
                const size_t rows = frame->planes[plane].height;
                if (frame->planes[plane].pitch <= 0 || !frame->planes[plane].data) {
                    error = std::format("HDR tensor tonemapper requires positive plane strides (plane={}, pitch={}, data_present={})", plane, frame->planes[plane].pitch, frame->planes[plane].data != nullptr);
                    return false;
                }
                plane_offset[plane] = static_cast<uint32_t>(source.plane_bytes);
                const size_t bytes = static_cast<size_t>(frame->planes[plane].pitch) * rows;
                if (bytes > std::numeric_limits<uint32_t>::max() - source.plane_bytes) {
                    error = std::format("HDR tensor plane storage exceeds address budget (plane={}, bytes={}, total={})", plane, bytes, source.plane_bytes);
                    return false;
                }
                source.planes[plane] = {frame->planes[plane].data, bytes};
                source.plane_bytes += bytes;
            }
            for (size_t i = 0; i < 3; ++i) {
                const media::FrameComponent& comp = desc->components[i];
                c.component[i] = {plane_offset[comp.plane] + uint32_t(comp.offset),
                                  uint32_t(frame->planes[comp.plane].pitch), uint32_t(comp.step), uint32_t(comp.shift)};
            }
            c.layout = {sample_depth == 16, (1u << depth) - 1, uint32_t(chroma_width), uint32_t(chroma_height)};

            Space& space = source.space;
            mapHdr(space.hdr, frame->frame_hdr);
            const auto* dovi = frame->dovi && frame->dovi->disable_residual ? &*frame->dovi : nullptr;
            Matrix decode;
            std::array<double, 3> multiplier{1.0, 1.0, 1.0}, black{};
            const double expand = (1LL << sample_depth) / ((1LL << sample_depth) - 1.0);
            float sample_scale;
            if (dovi) {
                const auto* color = dovi;
                source.dovi = packDovi(*dovi);
                space.primaries = BT2020;
                space.transfer = Transfer::PQ;
                space.hdr.min_luma = pqToNits(color->source_min_pq);
                space.hdr.max_luma = pqToNits(color->source_max_pq);
                if (color->level1) {
                    space.hdr.max_pq_y = (*color->level1)[0];
                    space.hdr.avg_pq_y = (*color->level1)[1];
                }
                Matrix linear;
                for (size_t i = 0; i < 9; ++i) {
                    decode[i / 3][i % 3] = color->nonlinear[i];
                    linear[i / 3][i % 3] = color->linear[i];
                }
                for (size_t i = 0; i < 3; ++i)
                    black[i] = color->offset[i] * expand;
                constexpr Matrix DOVI_LMS_TO_RGB{{{3.06441879f, -2.16597676f, 0.10155818f},
                                                  {-0.65612108f, 1.78554118f, -0.12943749f},
                                                  {0.01736321f, -0.04725154f, 1.03004253f}}};
                c.dovi_lms = rows(multiply(DOVI_LMS_TO_RGB, linear));
                sample_scale = float(((1LL << sample_depth) - 1.0) / ((1LL << depth) - 1.0));
            } else {
                if (frame->color_trc == media::ColorTransfer::Pq) {
                    space.transfer = Transfer::PQ;
                } else if (frame->color_trc == media::ColorTransfer::Hlg) {
                    space.transfer = Transfer::HLG;
                } else {
                    error = "HDR tensor tonemapper requires PQ or HLG transfer";
                    return false;
                }
                std::array<float, 3> k;
                if (!primaries(frame->color_primaries, space.primaries) || !lumaCoefficients(frame, k)) {
                    error = "HDR tensor tonemapper does not support the frame's primaries or matrix";
                    return false;
                }
                // pl_color_repr_decode
                decode = {{{1, 0, 2 * (1 - k[0])},
                           {1, -2 * (1 - k[2]) * k[2] / k[1], -2 * (1 - k[0]) * k[0] / k[1]},
                           {1, 2 * (1 - k[2]), 0}}};
                const bool full = frame->color_range == media::ColorRange::Full;
                const double ymin = full ? 0.0 : 16 / 256.0 * expand, ymax = full ? 1.0 : 235 / 256.0 * expand;
                const double cmid = 128 / 256.0 * expand, cmax = full ? 1.0 : 240 / 256.0 * expand;
                multiplier = {1.0 / (ymax - ymin), 0.5 / (cmax - cmid), 0.5 / (cmax - cmid)};
                black = {ymin, cmid, cmid};
                sample_scale = full ? float(((1LL << sample_depth) - 1.0) / ((1LL << depth) - 1.0))
                                    : float(1LL << sample_depth) / float(1LL << depth);
            }
            std::array<float, 3> offset{};
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    decode[i][j] *= multiplier[j];
                    offset[i] -= decode[i][j] * black[j];
                }
            }
            c.decode = rows(decode, offset);
            c.sample = {1.0f / float((1 << sample_depth) - 1), sample_scale, 0.0f, 0.0f};
            mapHdr(space.hdr, frame->stream_hdr);

            // pl_chroma_location_offset, LEFT when unknown
            const media::ChromaLocation location = frame->chroma_location;
            const float shift_x = location == media::ChromaLocation::Center || location == media::ChromaLocation::Top ||
                                          location == media::ChromaLocation::Bottom
                                      ? 0.0f
                                      : -0.5f;
            const float shift_y = location == media::ChromaLocation::TopLeft || location == media::ChromaLocation::Top         ? -0.5f
                                  : location == media::ChromaLocation::BottomLeft || location == media::ChromaLocation::Bottom ? 0.5f
                                                                                                                               : 0.0f;
            const float rx = 1.0f / (1 << desc->chroma_w), ry = 1.0f / (1 << desc->chroma_h);
            c.chroma = {rx, (0.5f - shift_x) * rx - 0.5f, ry, (0.5f - shift_y) * ry - 0.5f};
            c.flags = {desc->chroma_w > 0, desc->chroma_h > 0, dovi != nullptr,
                       space.transfer == Transfer::HLG};
            infer(space);
            return true;
        }

        // The color map constants that do not depend on the measured peak.
        void colorConstants(const Space& space, Constants& c) {
            const float csp_min = space.hdr.min_luma / SDR_WHITE, csp_max = space.hdr.max_luma / SDR_WHITE;
            const float y = 1.2f * std::pow(1.111f, std::log2(csp_max / HLG_REF));
            const float b = std::sqrt(3 * std::pow(csp_min / csp_max, 1 / y));
            c.hlg = {1 - b, b, csp_max, y - 1};
            const Matrix xyz = rgbToXyz(space.primaries);
            c.luma = {xyz[1][0], xyz[1][1], xyz[1][2], 0.0f};
            const Matrix target_lms = rgbToLms(BT709);
            c.rgb_to_lms = rows(rgbToLms(space.primaries));
            c.lms_to_rgb = rows(invert(target_lms));
            const float min_luma = nitsToPq(SDR_BLACK), max_luma = nitsToPq(SDR_WHITE);
            const float black = SDR_BLACK / SDR_WHITE;
            c.target = {1.0f / (max_luma - min_luma), -min_luma / (max_luma - min_luma), 1 / (1 - black),
                        -black / (1 - black)};
            c.gamut = {min_luma, max_luma, pqEotfTable(min_luma) - 1e-6f, pqEotfTable(max_luma) + 1e-6f};
            const Matrix gamut_in = rgbToLms(space.hdr.prim);
            c.gamut_in = rows(gamut_in);
            c.gamut_in_inverse = rows(invert(gamut_in));
            c.gamut_out = rows(target_lms);
            c.gamut_out_inverse = rows(invert(target_lms));
        }

        // pl_generate_blue_noise(64) with a fixed seed in place of rand().
        std::vector<float> blueNoise() {
            constexpr unsigned BITS = 6, SIZE = 1u << BITS, SIZE2 = SIZE * SIZE, RADIUS = SIZE / 2 - 1;
            constexpr unsigned GAUSS = RADIUS * 2 + 1;
            const auto xy = [](const unsigned x, const unsigned y) { return x | (y << BITS); };
            std::vector<uint64_t> gauss(SIZE2, 0), total(SIZE2, 0);
            const double sigma = -std::log(1.5 / double(UINT64_MAX) * GAUSS * GAUSS) / RADIUS;
            for (unsigned gy = 0; gy <= RADIUS; ++gy) {
                for (unsigned gx = 0; gx <= gy; ++gx) {
                    const int cx = int(gx) - int(RADIUS), cy = int(gy) - int(RADIUS);
                    const uint64_t v = uint64_t(std::exp(-std::sqrt(double(cx * cx + cy * cy)) * sigma) /
                                                (GAUSS * GAUSS) * double(UINT64_MAX));
                    for (const unsigned at : {xy(gx, gy), xy(gy, gx), xy(gx, GAUSS - 1 - gy), xy(gy, GAUSS - 1 - gx),
                                              xy(GAUSS - 1 - gx, gy), xy(GAUSS - 1 - gy, gx),
                                              xy(GAUSS - 1 - gx, GAUSS - 1 - gy), xy(GAUSS - 1 - gy, GAUSS - 1 - gx)})
                        gauss[at] = v;
                }
            }
            std::vector<float> noise(SIZE2);
            std::vector<bool> taken(SIZE2, false);
            std::vector<unsigned> candidates;
            std::minstd_rand random;
            for (unsigned rank = 0; rank < SIZE2; ++rank) {
                uint64_t minimum = UINT64_MAX;
                candidates.clear();
                for (unsigned c = 0; c < SIZE2; ++c) {
                    if (taken[c] || total[c] > minimum)
                        continue;
                    if (total[c] != minimum) {
                        minimum = total[c];
                        candidates.clear();
                    }
                    candidates.push_back(c);
                }
                const unsigned chosen = candidates.size() == 1       ? candidates[0]
                                        : candidates.size() == SIZE2 ? SIZE2 / 2
                                                                     : candidates[random() % candidates.size()];
                taken[chosen] = true;
                const unsigned start = xy(RADIUS, RADIUS) + SIZE2 - chosen;
                for (unsigned c = 0; c < SIZE2; ++c)
                    total[c] += gauss[(start + c) & (SIZE2 - 1)];
                noise[chosen] = float(rank) / SIZE2;
            }
            return noise;
        }

        std::vector<float> lanczosTable() {
            std::vector<float> table(256 * 6);
            for (size_t row = 0; row < 256; ++row) {
                const float offset = float(row) / 255.0f;
                float sum = 0.0f;
                for (size_t tap = 0; tap < 6; ++tap) {
                    float x = std::fabs(float(tap) - 2.0f - offset);
                    float value = 0.0f;
                    if (x <= 3.0f) {
                        if (x < 1e-8f) {
                            value = 1.0f;
                        } else {
                            const float a = x * float(std::numbers::pi);
                            const float b = a / 3.0f;
                            value = std::sin(a) / a * (std::sin(b) / b);
                        }
                    }
                    table[row * 6 + tap] = value;
                    sum += value;
                }
                for (size_t tap = 0; tap < 6; ++tap)
                    table[row * 6 + tap] /= sum;
            }
            return table;
        }

    } // namespace

    class HdrTensorRenderer::Impl {
    public:
        bool isAvailable(std::string& error) {
            std::lock_guard lock(mutex_);
            return initialize(error);
        }

        bool tonemap(const media::DecodedVideoFrame* frame, const HdrFormat format,
                     const int width, const int height, const int rotation_degrees,
                     std::vector<unsigned char>& output, std::string& error,
                     HdrTonemapTiming* timing, const bool rgba, const bool peak_detection) {
            std::lock_guard lock(mutex_);
            if (timing)
                *timing = {};
            if (!frame || frame->width <= 0 || frame->height <= 0 || width <= 0 || height <= 0 ||
                static_cast<size_t>(width) > std::numeric_limits<uint32_t>::max() / 4 / static_cast<size_t>(height)) {
                error = "Invalid HDR frame or output dimensions";
                return false;
            }
            const int rotation = ((rotation_degrees / 90) % 4 + 4) % 4;
            const auto initialization_started = std::chrono::steady_clock::now();
            if (!initialize(error))
                return false;
            if (timing)
                timing->initialization_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - initialization_started).count();

            const auto render_started = std::chrono::steady_clock::now();
            Source source;
            if (!describe(frame, source, error))
                return false;
            if (format == HdrFormat::DOLBY_VISION_NATIVE && source.dovi.empty()) {
                error = "Dolby Vision Profile 5 metadata is missing or requires an enhancement layer";
                return false;
            }
            Space& space = source.space;
            Constants& constants = source.constants;
            colorConstants(space, constants);

            const uint32_t source_width = frame->width, source_height = frame->height;
            const uint32_t image_width = rotation % 2 ? height : width, image_height = rotation % 2 ? width : height;
            // pass_scale_main: downscaling in either axis picks the downscaler.
            const bool down = image_width < source_width || image_height < source_height;
            const bool up = !down && (image_width > source_width || image_height > source_height);
            // hdr_update_peak
            const float source_peak = space.transfer == Transfer::HLG ? space.hdr.max_luma : PQ_PEAK;
            const bool detect = peak_detection && source_peak > SDR_WHITE + 1e-6f && !space.hdr.avg_pq_y;
            consumePeak(detect);
            if (!detect) {
                peak_ = {};
            } else if (peak_.avg_pq) {
                space.hdr.max_pq_y = peak_.max_pq;
                space.hdr.avg_pq_y = peak_.avg_pq;
            }
            constants.tone = spline(space);

            Tensor& planes = uploadPlanes(source);
            Tensor& constant_tensor = upload(constant_tensor_, constant_upload_,
                                             std::as_bytes(std::span(&constants, 1)));
            Tensor dovi;
            if (!source.dovi.empty())
                dovi = Tensor::from_vector(source.dovi, {source.dovi.size()}, Device::GPU);
            Tensor image = Tensor::empty({source_height, source_width, 2}, Device::GPU, DataType::UInt32);
            Tensor chroma_image;
            if (constants.flags[0] && constants.flags[1]) {
                chroma_image = Tensor::empty({source_height, constants.layout[2]}, Device::GPU, DataType::UInt32);
                Parameters chroma;
                chroma.source_height = source_height;
                if (!dispatch("chroma_vertical", chroma, constants.layout[2], source_height, error, &planes,
                              nullptr, nullptr, &constant_tensor, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, &chroma_image))
                    return false;
            }
            Parameters decode;
            decode.source_width = source_width;
            decode.source_height = source_height;
            if (!dispatch("decode", decode, source_width, source_height, error, &planes, nullptr, &image,
                          &constant_tensor, source.dovi.empty() ? nullptr : &dovi, nullptr, nullptr, nullptr,
                          nullptr, nullptr, chroma_image.is_valid() ? &chroma_image : nullptr))
                return false;

            if (detect && up && !measure(image, source_width, source_height, constant_tensor, error))
                return false;
            uint32_t current_width = source_width, current_height = source_height;
            for (const bool vertical : {true, false}) {
                const uint32_t from = vertical ? current_height : current_width;
                const uint32_t to = vertical ? image_height : image_width;
                if (from == to)
                    continue;
                const float ratio = float(to) / float(from);
                const float blur = std::max(float(1.0 / ratio), 1.0f);
                Parameters scale;
                scale.source_width = current_width;
                scale.source_height = current_height;
                scale.target_width = vertical ? current_width : image_width;
                scale.target_height = vertical ? image_height : current_height;
                scale.mode = (vertical ? 1u : 0u) | (from == 2 * to ? 2u : 0u);
                scale.taps = up ? 0u : 2u * static_cast<uint32_t>(std::ceil(blur));
                scale.blur = blur;
                Tensor scaled = Tensor::empty({scale.target_height, scale.target_width, 2}, Device::GPU,
                                              DataType::UInt32);
                if (!dispatch("resample", scale, scale.target_width, scale.target_height, error, nullptr, &image,
                              &scaled))
                    return false;
                image = std::move(scaled);
                current_width = scale.target_width;
                current_height = scale.target_height;
            }
            const bool fused_peak = detect && !up && rotation == 0;
            if (detect && !up && !fused_peak && !measure(image, image_width, image_height, constant_tensor, error))
                return false;
            const bool gamut_map = !equal(space.hdr.prim, BT709);
            if (gamut_map && (!gamut_input_ || !equal(*gamut_input_, space.hdr.prim))) {
                gamut_lut_ = Tensor::empty({GAMUT_FLOATS}, Device::GPU, DataType::Float32);
                Parameters lut;
                if (!dispatch("gamut", lut, 32 * 256, 1, error, nullptr, nullptr, nullptr, &constant_tensor,
                              nullptr, &gamut_lut_, nullptr, nullptr, nullptr, &pq_table_))
                    return false;
                gamut_input_ = space.hdr.prim;
            }

            const uint32_t channels = rgba ? 4u : 3u;
            Tensor target = Tensor::empty({static_cast<size_t>(height), static_cast<size_t>(width), channels},
                                          Device::GPU, DataType::UInt8);
            Tensor peak_tiles;
            uint32_t peak_groups = 0;
            if (fused_peak) {
                peak_groups = ((image_width + 15) / 16) * ((image_height + 15) / 16);
                peak_tiles = Tensor::empty({static_cast<size_t>(peak_groups) * 3}, Device::GPU, DataType::UInt32);
            }
            Parameters render;
            render.source_width = image_width;
            render.source_height = image_height;
            render.target_width = width;
            render.target_height = height;
            render.mode = rotation | (fused_peak ? FUSED_PEAK : 0u);
            render.taps = channels;
            render.gamut_map = gamut_map;
            if (!dispatch("render", render, width, height, error, nullptr, &image, nullptr, &constant_tensor, nullptr,
                          gamut_map ? &gamut_lut_ : nullptr, &dither_, &target,
                          fused_peak ? &peak_tiles : nullptr))
                return false;
            if (fused_peak) {
                peak_groups_ = peak_groups;
                peak_readback_.enqueue(peak_tiles);
            }
            if (timing)
                timing->render_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - render_started).count();

            const auto readback_started = std::chrono::steady_clock::now();
            const Tensor host = target.cpu();
            output.resize(host.bytes());
            std::memcpy(output.data(), host.data_ptr(), host.bytes());
            if (timing)
                timing->readback_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - readback_started).count();
            return true;
        }

        void reset() {
            std::lock_guard lock(mutex_);
            consumePeak(false);
            peak_ = {};
        }

    private:
        struct Peak {
            float avg_pq = 0.0f;
            float max_pq = 0.0f;
        };

        bool initialize(std::string& error) {
            if (program_)
                return true;
            auto loaded = GpuKernelModule::load(hdr_tonemap_program_entries());
            if (!loaded) {
                error = std::string(loaded.error().detail());
                return false;
            }
            static const std::vector<float> noise = blueNoise();
            static const std::vector<float> scaler = lanczosTable();
            dither_ = Tensor::from_vector(noise, {noise.size()}, Device::GPU);
            pq_table_ = Tensor::from_vector(pqTable(), {pqTable().size()}, Device::GPU);
            scaler_lut_ = Tensor::from_vector(scaler, {scaler.size()}, Device::GPU);
            program_ = std::move(*loaded);
            return true;
        }

        Tensor& upload(Tensor& destination, TensorUpload& slot, const std::span<const std::byte> bytes) {
            if (slot.pending())
                slot.wait();
            if (!destination.is_valid() || destination.bytes() != bytes.size())
                destination = Tensor::empty({bytes.size()}, Device::GPU, DataType::UInt8);
            slot.enqueue_in_batch(destination, bytes);
            return destination;
        }

        Tensor& uploadPlanes(const Source& source) {
            if (!plane_tensor_.is_valid() || plane_tensor_.bytes() != source.plane_bytes)
                plane_tensor_ = Tensor::empty({source.plane_bytes}, Device::GPU, DataType::UInt8);
            size_t offset = 0;
            for (size_t plane = 0; plane < source.planes.size(); ++plane) {
                const auto bytes = std::as_bytes(source.planes[plane]);
                if (bytes.empty())
                    continue;
                TensorUpload& slot = plane_uploads_[plane];
                if (slot.pending())
                    slot.wait();
                slot.enqueue_in_batch(plane_tensor_.slice(0, offset, offset + bytes.size()), bytes);
                offset += bytes.size();
            }
            return plane_tensor_;
        }

        bool dispatch(const std::string_view function, const Parameters& parameters, const uint32_t threads_x,
                      const uint32_t threads_y, std::string& error, const Tensor* planes, const Tensor* source,
                      Tensor* image = nullptr, const Tensor* constants = nullptr, const Tensor* dovi = nullptr,
                      Tensor* lut = nullptr, const Tensor* dither = nullptr, Tensor* output = nullptr,
                      Tensor* tiles = nullptr, const Tensor* pq_table = nullptr, Tensor* chroma = nullptr) {
            using Access = GpuKernelModule::Access;
            const std::array bindings{
                GpuKernelModule::Binding{0, planes}, GpuKernelModule::Binding{8, source},
                GpuKernelModule::Binding{16, image, Access::ReadWrite}, GpuKernelModule::Binding{24, constants},
                GpuKernelModule::Binding{32, dovi}, GpuKernelModule::Binding{40, lut, Access::ReadWrite},
                GpuKernelModule::Binding{48, dither}, GpuKernelModule::Binding{56, output, Access::ReadWrite},
                GpuKernelModule::Binding{64, tiles, Access::ReadWrite}, GpuKernelModule::Binding{72, pq_table},
                GpuKernelModule::Binding{80, &scaler_lut_}, GpuKernelModule::Binding{88, chroma, Access::ReadWrite}};
            const bool linear = function == "measure" || function == "gamut";
            const bool render = function == "render";
            auto result = program_->dispatch(
                {.function = function,
                 .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                 .groups = {GpuKernelModule::groups_for(threads_x, linear ? 64 : render ? 16
                                                                                        : 8),
                            GpuKernelModule::groups_for(threads_y, linear ? 1 : render ? 16
                                                                                       : 8),
                            1},
                 .group = {linear ? 64u : render ? 16u
                                                 : 8u,
                           linear ? 1u : render ? 16u
                                                : 8u,
                           1}});
            if (!result) {
                error = std::string(result.error().detail());
                return false;
            }
            return true;
        }

        // pl_shader_detect_peak, then update_peak_buf's smoothing and scene
        // change hysteresis with pl_peak_detect_default_params.
        bool measure(const Tensor& image, const uint32_t width, const uint32_t height, const Tensor& constants,
                     std::string& error) {
            const uint32_t groups = ((width + 15) / 16) * ((height + 15) / 16);
            Tensor tiles = Tensor::empty({static_cast<size_t>(groups) * 3}, Device::GPU, DataType::UInt32);
            Parameters parameters;
            parameters.source_width = width;
            parameters.source_height = height;
            if (!dispatch("measure", parameters, groups, 1, error, nullptr, &image, nullptr, &constants, nullptr,
                          nullptr, nullptr, nullptr, &tiles))
                return false;
            peak_groups_ = groups;
            peak_readback_.enqueue(tiles);
            return true;
        }

        void consumePeak(const bool use_result) {
            if (!peak_readback_.pending())
                return;
            peak_values_.resize(static_cast<size_t>(peak_groups_) * 3);
            peak_readback_.wait(std::as_writable_bytes(std::span(peak_values_)));
            if (!use_result)
                return;
            const uint32_t* values = peak_values_.data();
            uint64_t sum = 0, active = 0;
            uint32_t peak = 0;
            for (uint32_t i = 0; i < peak_groups_; ++i) {
                const uint32_t pixels = 256 - values[i * 3 + 2];
                if (!pixels)
                    continue;
                ++active;
                sum += values[i * 3] / pixels;
                peak = std::max(peak, values[i * 3 + 1]);
            }
            float avg_pq = HDR_BLACK, max_pq = HDR_BLACK;
            if (active) {
                avg_pq = float(sum) / float(active * PQ_MAX);
                max_pq = float(peak) / PQ_MAX;
            }
            if (!peak_.avg_pq) {
                peak_ = {avg_pq, max_pq};
            } else {
                if (std::fabs(avg_pq - peak_.avg_pq) < 1.0f / PQ_MAX)
                    avg_pq = peak_.avg_pq;
                if (std::fabs(max_pq - peak_.max_pq) < 1.0f / PQ_MAX)
                    max_pq = peak_.max_pq;
            }
            const float coefficient = 1.0f - std::exp(-1.0f / 20.0f);
            peak_.avg_pq += coefficient * (avg_pq - peak_.avg_pq);
            peak_.max_pq += coefficient * (max_pq - peak_.max_pq);
            const float delta = float(active) / float(peak_groups_) * std::fabs(avg_pq - peak_.avg_pq);
            const float mix_coefficient = smoothstep(1.0f * 1e-2f, 3.0f * 1e-2f, delta);
            peak_.avg_pq = mix(peak_.avg_pq, avg_pq, mix_coefficient);
            peak_.max_pq = mix(peak_.max_pq, max_pq, mix_coefficient);
        }

        std::mutex mutex_;
        std::unique_ptr<GpuKernelModule> program_;
        Tensor dither_;
        Tensor pq_table_;
        Tensor scaler_lut_;
        Tensor plane_tensor_;
        Tensor constant_tensor_;
        Tensor gamut_lut_;
        std::array<TensorUpload, 4> plane_uploads_;
        TensorUpload constant_upload_;
        TensorReadback peak_readback_;
        std::vector<uint32_t> peak_values_;
        uint32_t peak_groups_ = 0;
        std::optional<Primaries> gamut_input_;
        Peak peak_;
    };

    HdrTensorRenderer::HdrTensorRenderer() : impl_(std::make_unique<Impl>()) {}
    HdrTensorRenderer::~HdrTensorRenderer() = default;

    bool HdrTensorRenderer::isAvailable(std::string& error) { return impl_->isAvailable(error); }

    bool HdrTensorRenderer::tonemapToSdr(const media::DecodedVideoFrame* frame,
                                         const HdrFormat format, const int width, const int height,
                                         std::vector<unsigned char>& output, std::string& error,
                                         HdrTonemapTiming* timing) {
        return impl_->tonemap(frame, format, width, height, 0, output, error, timing, false, false);
    }

    bool HdrTensorRenderer::tonemapToSdrRgba(const media::DecodedVideoFrame* frame,
                                             const HdrFormat format, const int width, const int height,
                                             const int rotation, std::vector<unsigned char>& output,
                                             std::string& error) {
        return impl_->tonemap(frame, format, width, height, rotation, output, error, nullptr, true, true);
    }

    void HdrTensorRenderer::reset() { impl_->reset(); }

} // namespace lfs::io
