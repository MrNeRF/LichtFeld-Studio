/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

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
    void save(const std::filesystem::path& path, CFDataRef data) {
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(CFDataGetBytePtr(data)), CFDataGetLength(data));
        if (!output)
            throw std::runtime_error("Cannot write qualification output");
    }
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: heic_provider_probe input.heic output-directory\n";
            return 2;
        }
        const std::filesystem::path destination(argv[2]);
        std::filesystem::create_directories(destination);
        const std::string input = std::filesystem::absolute(argv[1]).string();
        Owned<CFURLRef> url(CFURLCreateFromFileSystemRepresentation(nullptr,
                                                                    reinterpret_cast<const UInt8*>(input.data()), input.size(), false));
        Owned<CGImageSourceRef> source(CGImageSourceCreateWithURL(url.get(), nullptr));
        if (!source)
            throw std::runtime_error("Cannot open ImageIO source");
        const auto primary = CGImageSourceGetPrimaryImageIndex(source.get());
        std::cout << "frames=" << CGImageSourceGetCount(source.get()) << "\nselected_primary=" << primary << '\n';
        const void* keys[]{kCGImageSourceShouldAllowFloat, kCGImageSourceShouldCache};
        const void* values[]{kCFBooleanTrue, kCFBooleanFalse};
        Owned<CFDictionaryRef> options(CFDictionaryCreate(nullptr, keys, values, 2,
                                                          &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
        Owned<CGImageRef> image(CGImageSourceCreateImageAtIndex(source.get(), primary, options.get()));
        if (!image)
            throw std::runtime_error("ImageIO could not decode primary image");
        std::cout << "width=" << CGImageGetWidth(image.get()) << "\nheight=" << CGImageGetHeight(image.get())
                  << "\nbits_per_component=" << CGImageGetBitsPerComponent(image.get())
                  << "\nbits_per_pixel=" << CGImageGetBitsPerPixel(image.get())
                  << "\nstride=" << CGImageGetBytesPerRow(image.get())
                  << "\nbitmap_info=" << CGImageGetBitmapInfo(image.get())
                  << "\nalpha_info=" << CGImageGetAlphaInfo(image.get()) << '\n';
        const auto height = CGImageGetHeight(image.get()), row = CGImageGetBytesPerRow(image.get());
        if (!height || !row || row > (512ull * 1024 * 1024) / height)
            throw std::runtime_error("Probe pixel buffer exceeds 512 MiB limit");
        Owned<CFDataRef> pixels(CGDataProviderCopyData(CGImageGetDataProvider(image.get())));
        if (!pixels)
            throw std::runtime_error("Cannot copy ImageIO pixel data");
        save(destination / "native.bin", pixels.get());
        const auto space = CGImageGetColorSpace(image.get());
        if (space) {
            Owned<CFDataRef> icc(CGColorSpaceCopyICCData(space));
            if (icc)
                save(destination / "profile.icc", icc.get());
            std::cout << "color_space_model=" << CGColorSpaceGetModel(space) << '\n';
        }
        Owned<CFDictionaryRef> properties(CGImageSourceCopyPropertiesAtIndex(source.get(), primary, nullptr));
        if (properties) {
            Owned<CFDataRef> plist(CFPropertyListCreateData(nullptr, properties.get(), kCFPropertyListXMLFormat_v1_0, 0, nullptr));
            if (plist)
                save(destination / "properties.plist", plist.get());
        }
        std::cout << "qualification_only=1\ncolor_conversion_applied=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
