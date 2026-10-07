// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/error.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
namespace lfs::media {
    // ISO/IEC 23091-2 colour identifiers, independent of decoder ABI.
    enum class ColorPrimaries { Bt709 = 1,
                                Unspecified = 2,
                                Bt470M = 4,
                                Bt470Bg = 5,
                                Smpte170M = 6,
                                Smpte240M = 7,
                                Film = 8,
                                Bt2020 = 9,
                                Smpte428 = 10,
                                Smpte431 = 11,
                                Smpte432 = 12,
                                JedecP22 = 22 };
    enum class ColorTransfer { Bt709 = 1,
                               Unspecified = 2,
                               Gamma22 = 4,
                               Gamma28 = 5,
                               Smpte170M = 6,
                               Smpte240M = 7,
                               Linear = 8,
                               Log = 9,
                               LogSqrt = 10,
                               Iec61966_2_4 = 11,
                               Bt1361 = 12,
                               Srgb = 13,
                               Bt2020_10 = 14,
                               Bt2020_12 = 15,
                               Pq = 16,
                               Smpte428 = 17,
                               Hlg = 18 };
    enum class ColorMatrix { Rgb = 0,
                             Bt709 = 1,
                             Unspecified = 2,
                             Fcc = 4,
                             Bt470Bg = 5,
                             Smpte170M = 6,
                             Smpte240M = 7,
                             Ycgco = 8,
                             Bt2020Ncl = 9,
                             Bt2020Cl = 10,
                             Smpte2085 = 11,
                             ChromaNcl = 12,
                             ChromaCl = 13,
                             Ictcp = 14,
                             YcgcoRe = 16,
                             YcgcoRo = 17 };
    enum class ColorRange { Unspecified,
                            Limited,
                            Full };
    enum class ChromaLocation { Unspecified,
                                Left,
                                Center,
                                TopLeft,
                                Top,
                                BottomLeft,
                                Bottom };
    enum class AlphaMode { Unknown,
                           Independent,
                           Premultiplied,
                           None };
    struct FrameComponent {
        int plane = 0, step = 0, offset = 0, shift = 0, depth = 0;
    };
    struct DecodedPlane {
        const std::uint8_t* data = nullptr;
        std::ptrdiff_t pitch = 0;
        int width = 0, height = 0;
    };
    struct Chromaticity {
        float x = 0, y = 0;
    };
    struct MasteringDisplay {
        bool luminance = false, primaries = false;
        float min_luma = 0, max_luma = 0;
        std::array<Chromaticity, 3> display{};
        Chromaticity white;
    };
    struct DynamicHdr10Plus {
        // Normalized HDR10+ values retain double precision. Each renderer
        // performs its established float conversion and nits scaling.
        std::array<double, 3> maxscl{};
        double average_maxrgb = 0;
        float histogram_max = 0;
        bool tone_mapping = false;
        float target_luma = 0, knee_x = 0, knee_y = 0;
        int num_anchors = 0;
        std::array<float, 15> anchors{};
    };
    struct HdrFrameMetadata {
        std::optional<MasteringDisplay> mastering;
        std::optional<std::array<float, 2>> content_light;
        std::optional<DynamicHdr10Plus> hdr10_plus;
    };
    struct DolbyVisionCurve {
        int num_pivots = 0;
        std::array<float, 9> pivots{};
        std::array<int, 8> method{}, mmr_order{};
        std::array<std::array<float, 3>, 8> polynomial{};
        std::array<float, 8> mmr_constant{};
        std::array<std::array<std::array<float, 7>, 3>, 8> mmr{};
    };
    struct DolbyVisionMetadata {
        bool disable_residual = false;
        std::array<float, 3> offset{};
        std::array<float, 9> nonlinear{}, linear{};
        std::array<DolbyVisionCurve, 3> curves{};
        float source_min_pq = 0, source_max_pq = 0;
        std::optional<std::array<float, 2>> level1;
    };
    struct Av1FilmGrain {
        int num_points_y = 0;
        std::uint8_t points_y[14][2]{};
        bool chroma_scaling_from_luma = false;
        int num_points_uv[2]{};
        std::uint8_t points_uv[2][10][2]{};
        int scaling_shift = 0, ar_coeff_lag = 0, ar_coeff_shift = 0, grain_scale_shift = 0;
        std::int8_t ar_coeffs_y[24]{}, ar_coeffs_uv[2][25]{}, uv_mult[2]{}, uv_mult_luma[2]{};
        std::int16_t uv_offset[2]{};
        bool overlap = false;
    };
    struct H274FilmGrain {
        int model_id = 0, blending_mode_id = 0, log2_scale_factor = 0;
        bool component_model_present[3]{};
        std::uint16_t num_intensity_intervals[3]{};
        std::uint8_t num_model_values[3]{};
        const std::uint8_t* lower[3]{};
        const std::uint8_t* upper[3]{};
        const std::int16_t (*model[3])[6]{};
    };
    struct FilmGrain {
        std::uint64_t seed = 0;
        std::optional<Av1FilmGrain> av1;
        std::optional<H274FilmGrain> h274;
    };
    // A synchronous borrowed view: planes, profile and grain tables remain
    // decoder-owned until the renderer returns. Numeric metadata is copied;
    // adapters never inspect decoder objects or retain plane pointers.
    struct DecodedVideoFrame {
        int width = 0, height = 0, plane_count = 0, component_count = 0, chroma_w = 0, chroma_h = 0;
        std::string_view format_name;
        std::array<DecodedPlane, 4> planes{};
        std::array<FrameComponent, 4> components{};
        bool hardware = false, big_endian = false, rgb = false, xyz = false, palette = false, bitstream = false, floating = false, bayer = false, alpha = false;
        AlphaMode alpha_mode = AlphaMode::Independent;
        // Borrowed native identity only, not an importable GPU frame contract.
        // Hardware views carry no CPU plane pointers. Current HDR adapters
        // require a downloaded software frame; device/sync import is separate.
        void* hardware_handle = nullptr;
        ColorPrimaries color_primaries = ColorPrimaries::Unspecified;
        ColorTransfer color_trc = ColorTransfer::Unspecified;
        ColorMatrix colorspace = ColorMatrix::Unspecified;
        ColorRange color_range = ColorRange::Unspecified;
        ChromaLocation chroma_location = ChromaLocation::Unspecified;
        std::array<float, 4> crop{};
        HdrFrameMetadata frame_hdr, stream_hdr;
        std::optional<DolbyVisionMetadata> dovi;
        std::span<const std::uint8_t> icc_profile, dovi_rpu;
        std::optional<FilmGrain> film_grain;
    };
    LFS_MEDIA_API Result<void> validateDecodedVideoFrame(const DecodedVideoFrame& frame);
} // namespace lfs::media
