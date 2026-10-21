// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/crash_handler.hpp"
#include "core/image_codecs.hpp"
#include "core/tensor_backend.hpp"
#include "io/image_sequence.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
    template <class T>
    void good(const T& value) {
        if (!value)
            throw std::runtime_error(std::string(value.error().detail()));
    }
    void require(bool value, const char* message) {
        if (!value)
            throw std::runtime_error(message);
    }
} // namespace

int main(int argc, char** argv) {
    using namespace lfs;
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / ("lfs-tensor-sequence-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        fs::path path;
        ~Cleanup() {
            std::error_code error;
            fs::remove_all(path, error);
        }
    } cleanup{root};
    try {
        const std::string_view name = argc > 1 ? argv[1] : "cpu";
        const bool cpu = name == "cpu";
        const auto backend = name == "cuda" ? core::GpuBackend::CUDA : name == "vulkan" ? core::GpuBackend::Vulkan
                                                                                        : core::GpuBackend::Metal;
        std::optional<core::GpuBackendScope> scope;
        if (!cpu) {
            good(core::set_default_gpu_backend(backend));
            if (!core::gpu_backend_available(backend))
                return 77;
            scope.emplace(backend);
        }
        struct Shutdown {
            ~Shutdown() { core::teardown_gpu_before_exit(); }
        } shutdown;
        io::TensorImageSequence output;
        io::TensorImageSequenceOptions options;
        options.output.files.output_directory = root / "png";
        options.output.files.filename_pattern = "frame_%04d";
        options.output.files.first_source_frame_number = 0;
        options.output.frame_count = 3;
        options.output.layout = {3, 5, 9, media::FramePixelFormat::RGB8, {media::ColorTransfer::Srgb, media::ColorPrimaries::Bt709}};
        options.input_color = options.output.layout.color;
        good(output.open(options));
        for (int index = 0; index < 3; ++index) {
            std::vector<float> samples(5 * 6 * 3, (31 + index * 40) / 255.f);
            auto image = core::Tensor::from_blob(samples.data(), {5, 6, 3}, core::Device::CPU, core::DataType::Float32);
            if (!cpu && index != 1)
                image = image.gpu();
            const auto view = image.slice(1, 0, 3);
            require(view.device() == ((!cpu && index != 1) ? core::Device::GPU : core::Device::CPU), "unexpected Tensor input device");
            if (view.device() == core::Device::GPU)
                require(core::gpu_backend_of(view) == backend, "unexpected Tensor GPU backend");
            good(output.write(view));
        }
        good(output.close());
        for (int index = 0; index < 3; ++index) {
            core::image_codecs::Image image;
            std::string error;
            require(core::image_codecs::decode(root / "png" / ("frame_000" + std::to_string(index) + ".png"), image, error), "Tensor output did not decode");
            require(image.width == 3 && image.height == 5 && image.channels == 3, "Tensor output shape changed");
            require(std::all_of(image.data.begin(), image.data.end(), [index](uint8_t sample) { return sample == 31 + index * 40; }), "Tensor output order or encoded samples changed");
        }
        options.output.files.output_directory = root / "budget";
        options.readback_budget = 1;
        require(!output.open(options) && !fs::exists(root / "budget"), "readback budget was not checked before output setup");
        options.readback_budget = 1024;
        options.output.files.output_directory = root / "float";
        options.output.files.format = media::FrameFileFormat::EXR;
        options.output.files.exr.precision = media::ExrPrecision::Float;
        options.output.frame_count = 1;
        options.output.layout = {3, 5, 3 * 4 * sizeof(float), media::FramePixelFormat::RGBAFloat32, {media::ColorTransfer::Linear, media::ColorPrimaries::Bt709, media::AlphaMode::Independent}};
        options.input_color = options.output.layout.color;
        std::vector<float> samples(5 * 3 * 4);
        for (size_t i = 0; i < samples.size(); i += 4) {
            samples[i] = -.125f;
            samples[i + 1] = .1234567f;
            samples[i + 2] = 1.5f;
            samples[i + 3] = 1.f;
        }
        auto raw = core::Tensor::from_blob(samples.data(), {5, 3, 4}, core::Device::CPU, core::DataType::Float32);
        auto image = cpu ? raw : raw.gpu();
        good(output.open(options));
        good(output.write(image));
        good(output.close());
        core::image_codecs::Image decoded;
        std::string error;
        require(core::image_codecs::decode(root / "float/frame_0000.exr", decoded, error, false), "raw EXR did not decode");
        require(decoded.sample_type == core::image_codecs::SampleType::Float32 && decoded.channels == 4, "raw precision/alpha changed");
        require(decoded.data.size() == samples.size() * sizeof(float) && std::memcmp(decoded.data.data(), samples.data(), decoded.data.size()) == 0, "linear EXR samples clipped or quantized");

        std::vector<uint16_t> half_samples(samples.size());
        std::vector<float> half_expected(samples.size());
        const uint16_t half_codes[] = {0xb000, 0x3400, 0x3e00, 0x3c00};
        const float half_values[] = {-.125f, .25f, 1.5f, 1.f};
        for (size_t i = 0; i < half_samples.size(); ++i) {
            half_samples[i] = half_codes[i % 4];
            half_expected[i] = half_values[i % 4];
        }
        auto half = core::Tensor::from_blob(half_samples.data(), {5, 3, 4}, core::Device::CPU, core::DataType::Float16);
        if (!cpu)
            half = half.gpu();
        options.output.files.output_directory = root / "half";
        good(output.open(options));
        good(output.write(half));
        good(output.close());
        require(core::image_codecs::decode(root / "half/frame_0000.exr", decoded, error, false), "Float16 producer output did not decode");
        require(decoded.data.size() == half_expected.size() * sizeof(float) && std::memcmp(decoded.data.data(), half_expected.data(), decoded.data.size()) == 0,
                "Float16 producer changed exact representable samples");

        options.input_color.transfer = media::ColorTransfer::Srgb;
        options.output.files.output_directory = root / "srgb";
        std::vector<float> encoded(samples.size(), .5f);
        auto srgb = core::Tensor::from_blob(encoded.data(), {5, 3, 4}, core::Device::CPU, core::DataType::Float32);
        if (!cpu)
            srgb = srgb.gpu();
        good(output.open(options));
        good(output.write(srgb));
        good(output.close());
        require(core::image_codecs::decode(root / "srgb/frame_0000.exr", decoded, error, false), "sRGB producer output did not decode");
        const double expected_rgb = .5 * std::pow((.5 + .055) / 1.055, 2.4);
        for (size_t i = 0; i < encoded.size(); ++i) {
            float value;
            std::memcpy(&value, decoded.data.data() + i * sizeof(float), sizeof(float));
            require(std::abs(value - (i % 4 == 3 ? .5 : expected_rgb)) < 3e-8,
                    "sRGB conversion or straight-alpha treatment differs from independent reference");
        }
        options.input_color.transfer = media::ColorTransfer::Linear;

        options.output.files.output_directory = root / "nan";
        good(output.open(options));
        samples[0] = std::numeric_limits<float>::quiet_NaN();
        require(!output.write(raw), "nonfinite Tensor sample accepted");
        require(!output.close(), "failed Tensor sequence completed");
        require(!fs::exists(root / "nan/frame_0000.exr"), "failed Tensor frame was published");
        std::cout << "PASS " << name << " image sequence Tensor readback, strided views, order, raw float, budget and failure\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
