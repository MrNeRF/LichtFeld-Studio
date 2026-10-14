/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <wincodec.h>
#include <windows.h>
#include <wrl/client.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace {
    void check(HRESULT hr, const char* operation) {
        if (FAILED(hr)) {
            std::ostringstream message;
            message << operation << ": HRESULT 0x" << std::hex << static_cast<unsigned long>(hr);
            throw std::runtime_error(message.str());
        }
    }
    void guid(const char* name, REFGUID value) {
        wchar_t text[40]{};
        StringFromGUID2(value, text, 40);
        std::cout << name << '=';
        for (const auto c : text) {
            if (!c)
                break;
            std::cout << static_cast<char>(c);
        }
        std::cout << '\n';
    }
    struct Apartment {
        HRESULT status = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        Apartment() { check(status, "CoInitializeEx"); }
        ~Apartment() {
            if (SUCCEEDED(status))
                CoUninitialize();
        }
    };
    void save(const std::filesystem::path& path, const std::vector<BYTE>& data) {
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!output)
            throw std::runtime_error("Cannot write qualification output");
    }
    UINT stride(UINT width, UINT height, UINT bits) {
        const auto row = (static_cast<uint64_t>(width) * bits + 7) / 8;
        if (!width || !height || !bits || bits > 128 || row * height > 512ull * 1024 * 1024)
            throw std::runtime_error("Probe pixel allocation exceeds 512 MiB limit or has invalid dimensions");
        return static_cast<UINT>(row);
    }
} // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: heic_provider_probe input.heic output-directory\n";
            return 2;
        }
        const std::filesystem::path destination(argv[2]);
        std::filesystem::create_directories(destination);
        Apartment apartment;
        ComPtr<IWICImagingFactory> factory;
        check(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                               IID_PPV_ARGS(&factory)),
              "WIC factory");
        ComPtr<IWICBitmapDecoder> decoder;
        check(factory->CreateDecoderFromFilename(argv[1], nullptr, GENERIC_READ,
                                                 WICDecodeMetadataCacheOnDemand, &decoder),
              "Open system decoder");
        ComPtr<IWICBitmapDecoderInfo> info;
        check(decoder->GetDecoderInfo(&info), "Decoder info");
        GUID identifier{};
        check(info->GetCLSID(&identifier), "Decoder CLSID");
        guid("decoder_clsid", identifier);
        check(info->GetVendorGUID(&identifier), "Decoder vendor");
        guid("decoder_vendor", identifier);
        check(decoder->GetContainerFormat(&identifier), "Container format");
        guid("container", identifier);
        UINT count = 0;
        check(decoder->GetFrameCount(&count), "Frame count");
        std::cout << "frames=" << count << "\nselected_frame=0\n";
        // Do not claim WIC index zero is the HEIF primary until qualified.
        ComPtr<IWICBitmapFrameDecode> frame;
        check(decoder->GetFrame(0, &frame), "Frame zero");
        UINT width = 0, height = 0;
        check(frame->GetSize(&width, &height), "Dimensions");
        std::cout << "width=" << width << "\nheight=" << height << '\n';
        GUID native{};
        check(frame->GetPixelFormat(&native), "Native pixel format");
        guid("native_format", native);
        ComPtr<IWICComponentInfo> component;
        check(factory->CreateComponentInfo(native, &component), "Pixel format info");
        ComPtr<IWICPixelFormatInfo> pixel_info;
        check(component.As(&pixel_info), "Pixel format interface");
        UINT bits = 0, channels = 0;
        check(pixel_info->GetBitsPerPixel(&bits), "Native bits per pixel");
        check(pixel_info->GetChannelCount(&channels), "Native channel count");
        std::cout << "native_bits_per_pixel=" << bits << "\nnative_channels=" << channels << '\n';
        const UINT row = stride(width, height, bits);
        std::vector<BYTE> pixels(static_cast<size_t>(row) * height);
        check(frame->CopyPixels(nullptr, row, static_cast<UINT>(pixels.size()), pixels.data()), "Native CopyPixels");
        save(destination / "native.bin", pixels);
        std::cout << "native_copy=ok\nnative_stride=" << row << '\n';
        UINT profiles = 0;
        check(frame->GetColorContexts(0, nullptr, &profiles), "Color context count");
        std::cout << "color_contexts=" << profiles << '\n';
        if (profiles > 16)
            throw std::runtime_error("Too many color contexts for qualification probe");
        std::vector<ComPtr<IWICColorContext>> contexts(profiles);
        std::vector<IWICColorContext*> raw(profiles);
        for (UINT i = 0; i < profiles; ++i) {
            check(factory->CreateColorContext(&contexts[i]), "Create color context");
            raw[i] = contexts[i].Get();
        }
        if (profiles)
            check(frame->GetColorContexts(profiles, raw.data(), &profiles), "Read color contexts");
        for (UINT i = 0; i < profiles; ++i) {
            WICColorContextType type{};
            check(contexts[i]->GetType(&type), "Color context type");
            std::cout << "profile_" << i << "_type=" << type << '\n';
            if (type != WICColorContextProfile)
                continue;
            UINT size = 0;
            check(contexts[i]->GetProfileBytes(0, nullptr, &size), "ICC size");
            if (!size || size > 4 * 1024 * 1024)
                throw std::runtime_error("Invalid ICC size for probe");
            std::vector<BYTE> icc(size);
            check(contexts[i]->GetProfileBytes(size, icc.data(), &size), "ICC bytes");
            save(destination / ("profile-" + std::to_string(i) + ".icc"), icc);
        }
        ComPtr<IWICBitmapSourceTransform> transform;
        const HRESULT transform_status = frame.As(&transform);
        std::cout << "source_transform_status=0x" << std::hex << static_cast<unsigned long>(transform_status) << std::dec << '\n';
        if (SUCCEEDED(transform_status)) {
            const std::array formats{GUID_WICPixelFormat48bppRGB, GUID_WICPixelFormat64bppRGBA,
                                     GUID_WICPixelFormat128bppRGBAFloat};
            for (const auto& requested : formats) {
                auto closest = requested;
                const HRESULT hr = transform->GetClosestPixelFormat(&closest);
                guid("requested_format", requested);
                std::cout << "closest_status=0x" << std::hex << static_cast<unsigned long>(hr) << std::dec << '\n';
                if (SUCCEEDED(hr))
                    guid("closest_format", closest);
            }
        }
        std::cout << "qualification_only=1\ncolor_conversion_applied=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
