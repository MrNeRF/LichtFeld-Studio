// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/file_frame_sink.hpp"
#include <array>
#include <format>
#include <string_view>
#include <type_traits>

namespace lfs::media {
    using namespace std::string_view_literals;
    template <class T>
    struct NamedOption {
        T value;
        std::string_view name;
    };
    template <class T>
    inline constexpr auto outputOptionValues = std::array<NamedOption<T>, 0>{};
    template <>
    inline constexpr auto outputOptionValues<FrameFileFormat> = std::array{
        NamedOption{FrameFileFormat::PNG, "png"sv}, NamedOption{FrameFileFormat::JPEG, "jpeg"sv}, NamedOption{FrameFileFormat::EXR, "exr"sv}};
    template <>
    inline constexpr auto outputOptionValues<ExrPrecision> = std::array{
        NamedOption{ExrPrecision::Half, "half"sv}, NamedOption{ExrPrecision::Float, "float"sv}};
    template <>
    inline constexpr auto outputOptionValues<ExrCompression> = std::array{
        NamedOption{ExrCompression::ZIP, "zip"sv}, NamedOption{ExrCompression::None, "none"sv}};
    template <>
    inline constexpr auto outputOptionValues<ColorTransfer> = std::array{
        NamedOption{ColorTransfer::Unspecified, "auto"sv}, NamedOption{ColorTransfer::Linear, "linear"sv},
        NamedOption{ColorTransfer::Srgb, "srgb"sv}, NamedOption{ColorTransfer::Bt709, "bt709"sv}};
    template <>
    inline constexpr auto outputOptionValues<ColorPrimaries> = std::array{
        NamedOption{ColorPrimaries::Unspecified, "auto"sv}, NamedOption{ColorPrimaries::Bt709, "bt709"sv}, NamedOption{ColorPrimaries::Bt2020, "bt2020"sv}};
    template <class T>
    constexpr std::string_view outputOptionName(T value) {
        for (const auto& option : outputOptionValues<T>)
            if (option.value == value)
                return option.name;
        return "unknown";
    }
    template <class T>
    Result<T> parseOutputOption(std::string_view text) {
        // Preserve the existing CLI alias; all frontends use canonical names.
        if constexpr (std::is_same_v<T, FrameFileFormat>)
            if (text == "jpg")
                text = "jpeg";
        for (const auto& option : outputOptionValues<T>)
            if (option.name == text)
                return option.value;
        return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = std::format("Unsupported media option (value='{}')", text), .detection = LFS_SOURCE_SITE_CURRENT()});
    }
    // Metadata/layout preflight, also used by the Studio dialog before Start.
    [[nodiscard]] LFS_MEDIA_API SinkResult validateFloatSdrSource(const StreamDescription&, FrameColor = {});
    // Same validation for extraction, file sinks and independently produced EXR.
    [[nodiscard]] LFS_MEDIA_API SinkResult validateExrOutputOptions(const ExrOutputOptions&);
} // namespace lfs::media
