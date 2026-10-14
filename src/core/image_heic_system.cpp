/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "image_heic_system.hpp"
#include <cstring>
#include <limits>

#ifdef _WIN32
#include <wincodec.h>
#include <windows.h>
#include <wrl/client.h>
#elif defined(__APPLE__)
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <memory>
#include <type_traits>
#endif

namespace lfs::core::image_codecs {
#ifdef _WIN32
    namespace {
        using Microsoft::WRL::ComPtr;
        struct Apartment {
            HRESULT status = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
            ~Apartment() {
                if (SUCCEEDED(status))
                    CoUninitialize();
            }
        };
        bool wic_status(HRESULT status, const char* operation, std::string& error) {
            if (SUCCEEDED(status))
                return true;
            error = "HEIF: HEVC system provider " + std::string(operation) + " failed (HRESULT " +
                    std::to_string(static_cast<unsigned long>(status)) + "); required Microsoft HEIF/HEVC support may be unavailable";
            return false;
        }
    } // namespace
    bool read_heic_system(const std::filesystem::path* path, std::span<const std::uint8_t> bytes,
                          const Probe& expected, int source_bits, int, bool, Image* image, std::string& error) {
        // WIC's qualified HEIF provider exposes only 8-bit RGB. Do not silently
        // quantize raw/training input or invent missing auxiliary alpha.
        if (source_bits != 8 || expected.channels != 3) {
            error = "HEIF: HEVC system provider unsupported for this precision or alpha; Windows qualification requires 8-bit RGB";
            return false;
        }
        Apartment apartment;
        if (FAILED(apartment.status) && apartment.status != RPC_E_CHANGED_MODE)
            return wic_status(apartment.status, "COM initialization", error);
        ComPtr<IWICImagingFactory> factory;
        if (!wic_status(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                         IID_PPV_ARGS(&factory)),
                        "factory", error))
            return false;
        ComPtr<IWICStream> stream;
        ComPtr<IWICBitmapDecoder> decoder;
        HRESULT status;
        if (path) {
            status = factory->CreateDecoderFromFilename(path->c_str(), &GUID_VendorMicrosoft, GENERIC_READ,
                                                        WICDecodeMetadataCacheOnDemand, &decoder);
        } else {
            if (bytes.size() > std::numeric_limits<DWORD>::max()) {
                error = "HEIF: HEVC memory input exceeds Windows stream limit";
                return false;
            }
            if (!wic_status(factory->CreateStream(&stream), "memory stream", error) ||
                !wic_status(stream->InitializeFromMemory(const_cast<BYTE*>(bytes.data()), static_cast<DWORD>(bytes.size())), "memory input", error))
                return false;
            status = factory->CreateDecoderFromStream(stream.Get(), &GUID_VendorMicrosoft, WICDecodeMetadataCacheOnDemand, &decoder);
        }
        if (!wic_status(status, "open", error))
            return false;
        GUID container{};
        if (!wic_status(decoder->GetContainerFormat(&container), "container", error))
            return false;
        if (container != GUID_ContainerFormatHeif) {
            error = "HEIF: inconsistent system decoder container";
            return false;
        }
        ComPtr<IWICBitmapFrameDecode> frame;
        if (!wic_status(decoder->GetFrame(0, &frame), "primary frame", error))
            return false;
        UINT width = 0, height = 0;
        GUID format{};
        if (!wic_status(frame->GetSize(&width, &height), "dimensions", error) ||
            !wic_status(frame->GetPixelFormat(&format), "pixel format", error))
            return false;
        if (width != static_cast<UINT>(expected.width) || height != static_cast<UINT>(expected.height) ||
            (format != GUID_WICPixelFormat32bppBGR && format != GUID_WICPixelFormat24bppBGR && format != GUID_WICPixelFormat24bppRGB)) {
            error = "HEIF: unsupported or inconsistent Windows native layout";
            return false;
        }
        const size_t step = format == GUID_WICPixelFormat32bppBGR ? 4 : 3;
        if (!image)
            return true;
        const size_t row = size_t(width) * step, size = row * height;
        if (size > std::numeric_limits<UINT>::max()) {
            error = "HEIF: Windows native pixel buffer exceeds API limit";
            return false;
        }
        std::vector<uint8_t> native(size);
        if (!wic_status(frame->CopyPixels(nullptr, static_cast<UINT>(row), static_cast<UINT>(size), native.data()), "pixel copy", error))
            return false;
        Image result{expected.width, expected.height, 3, SampleType::UInt8, std::vector<uint8_t>(size_t(width) * height * 3)};
        const bool rgb = format == GUID_WICPixelFormat24bppRGB;
        for (size_t pixel = 0; pixel < size_t(width) * height; ++pixel)
            for (size_t channel = 0; channel < 3; ++channel)
                result.data[pixel * 3 + channel] = native[pixel * step + (rgb ? channel : 2 - channel)];
        *image = std::move(result);
        return true;
    }
#elif defined(__APPLE__)
    namespace {
        template <typename T>
        struct Release {
            void operator()(T* value) const {
                if (value)
                    CFRelease(value);
            }
        };
        template <typename T>
        using Owned = std::unique_ptr<std::remove_pointer_t<T>, Release<std::remove_pointer_t<T>>>;
    } // namespace
    bool read_heic_system(const std::filesystem::path* path, std::span<const std::uint8_t> bytes,
                          const Probe& expected, int source_bits, int orientation, bool icc_over_hdr, Image* image, std::string& error) {
        // ICC policy is validated in the shared codec first. ImageIO still
        // refuses pixel decoding for the qualified contradictory HDR input.
        // Neither tone mapping nor a successful metadata probe fixes that.
        if (icc_over_hdr) {
            error = "HEIF: HEVC Apple system provider unsupported for conflicting HDR NCLX; ICC policy was validated but ImageIO cannot decode the pixels";
            return false;
        }
        if ((source_bits != 8 && source_bits != 10) || expected.channels != 3) {
            error = "HEIF: HEVC system provider unsupported for this precision or alpha; Apple qualification requires 8/10-bit RGB";
            return false;
        }
        Owned<CFDataRef> input;
        Owned<CGImageSourceRef> source;
        if (path) {
            const auto name = std::filesystem::absolute(*path).u8string();
            Owned<CFURLRef> url(CFURLCreateFromFileSystemRepresentation(nullptr,
                                                                        reinterpret_cast<const UInt8*>(name.data()), name.size(), false));
            if (url)
                source.reset(CGImageSourceCreateWithURL(url.get(), nullptr));
        } else if (bytes.size() <= static_cast<size_t>(std::numeric_limits<CFIndex>::max())) {
            input.reset(CFDataCreateWithBytesNoCopy(nullptr, bytes.data(), static_cast<CFIndex>(bytes.size()), kCFAllocatorNull));
            if (input)
                source.reset(CGImageSourceCreateWithData(input.get(), nullptr));
        }
        if (!source) {
            error = "HEIF: HEVC Apple system provider cannot open source";
            return false;
        }
        const auto primary = CGImageSourceGetPrimaryImageIndex(source.get());
        const void* keys[]{kCGImageSourceShouldAllowFloat, kCGImageSourceShouldCache};
        const void* values[]{kCFBooleanTrue, kCFBooleanFalse};
        Owned<CFDictionaryRef> options(CFDictionaryCreate(nullptr, keys, values, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
        Owned<CGImageRef> native(CGImageSourceCreateImageAtIndex(source.get(), primary, options.get()));
        if (!native) {
            error = "HEIF: HEVC Apple system provider cannot decode primary image";
            return false;
        }
        const size_t width = CGImageGetWidth(native.get()), height = CGImageGetHeight(native.get());
        const auto bpc = CGImageGetBitsPerComponent(native.get()), bpp = CGImageGetBitsPerPixel(native.get());
        const auto bitmap = CGImageGetBitmapInfo(native.get());
        const auto alpha = CGImageGetAlphaInfo(native.get());
        const auto space = CGImageGetColorSpace(native.get());
        const bool packed10 = source_bits == 10 && bpc == 10 && bpp == 32 &&
                              (bitmap & kCGBitmapByteOrderMask) == kCGBitmapByteOrder32Little &&
                              (bitmap & kCGImagePixelFormatMask) == kCGImagePixelFormatRGB101010 && alpha == kCGImageAlphaNoneSkipFirst;
        const bool rgb8 = source_bits == 8 && bpc == 8 && bpp == 32 &&
                          (bitmap & kCGBitmapByteOrderMask) == kCGBitmapByteOrderDefault && alpha == kCGImageAlphaNoneSkipLast;
        if ((!rgb8 && !packed10) || !space || CGColorSpaceGetModel(space) != kCGColorSpaceModelRGB ||
            !width || !height || width * height != size_t(expected.width) * expected.height) {
            error = "HEIF: unsupported or inconsistent Apple native layout";
            return false;
        }
        const bool transpose = orientation >= 5 && orientation <= 8;
        if (orientation < 1 || orientation > 8 ||
            (transpose ? height : width) != size_t(expected.width) || (transpose ? width : height) != size_t(expected.height)) {
            error = "HEIF: inconsistent Apple transformed dimensions";
            return false;
        }
        const auto row = CGImageGetBytesPerRow(native.get());
        if (row < width * 4 || row > std::numeric_limits<size_t>::max() / height) {
            error = "HEIF: invalid Apple native stride";
            return false;
        }
        if (!image)
            return true;
        const auto provider = CGImageGetDataProvider(native.get());
        Owned<CFDataRef> pixels(provider ? CGDataProviderCopyData(provider) : nullptr);
        if (!pixels || static_cast<size_t>(CFDataGetLength(pixels.get())) < row * height) {
            error = "HEIF: HEVC Apple system provider cannot copy native pixels";
            return false;
        }
        const auto* data = CFDataGetBytePtr(pixels.get());
        Image result{expected.width, expected.height, 3, expected.sample_type,
                     std::vector<uint8_t>(width * height * 3 * (packed10 ? 2 : 1))};
        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x) {
                size_t dx = x, dy = y;
                switch (orientation) {
                case 2: dx = width - 1 - x; break;
                case 3:
                    dx = width - 1 - x;
                    dy = height - 1 - y;
                    break;
                case 4: dy = height - 1 - y; break;
                case 5:
                    dx = y;
                    dy = x;
                    break;
                case 6:
                    dx = height - 1 - y;
                    dy = x;
                    break;
                case 7:
                    dx = height - 1 - y;
                    dy = width - 1 - x;
                    break;
                case 8:
                    dx = y;
                    dy = width - 1 - x;
                    break;
                }
                const auto* source_pixel = data + y * row + x * 4;
                if (packed10) {
                    uint32_t pixel;
                    std::memcpy(&pixel, source_pixel, 4);
                    for (size_t channel = 0; channel < 3; ++channel) {
                        const uint32_t code = (pixel >> (20 - channel * 10)) & 1023;
                        const auto normalized = static_cast<uint16_t>((code * 65535 + 511) / 1023);
                        std::memcpy(result.data.data() + ((dy * expected.width + dx) * 3 + channel) * 2, &normalized, 2);
                    }
                } else
                    std::memcpy(result.data.data() + (dy * expected.width + dx) * 3, source_pixel, 3);
            }
        *image = std::move(result);
        return true;
    }
#else
    bool read_heic_system(const std::filesystem::path*, std::span<const std::uint8_t>,
                          const Probe&, int, int, bool, Image*, std::string& error) {
        error = "HEIF: HEVC-coded HEIC is unsupported in this build; no qualified system provider is configured";
        return false;
    }
#endif
} // namespace lfs::core::image_codecs
