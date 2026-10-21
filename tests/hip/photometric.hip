/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "photometric.hpp"
#include "photometric_oracle.hpp"
#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace {
    void check(hipError_t status) {
        if (status != hipSuccess)
            throw std::runtime_error(hipGetErrorString(status));
    }
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    struct Buffer {
        void* data = nullptr;
        explicit Buffer(size_t bytes) { check(hipMalloc(&data, bytes)); }
        ~Buffer() { (void)hipFree(data); }
        template <class T = float>
        T* as() { return static_cast<T*>(data); }
    };
} // namespace

int main() try {
    int runtime = 0;
    check(hipRuntimeGetVersion(&runtime));
    std::printf("HIP photometric runtime %d\n", runtime);
    unsigned cases = 0;
    double worst_relative = 0, worst_loss = 0;
    const unsigned dimensions[][2]{{1, 1}, {9, 7}, {11, 11}, {17, 13}, {35, 29}, {65, 37}, {9, 17}, {17, 9}, {32, 16}, {16, 32}};
    for (const auto& shape : dimensions) {
        const auto [w, h] = std::array<unsigned, 2>{shape[0], shape[1]};
        const size_t pixels = w * h, count = pixels * 3;
        std::vector<float> image(count), target(count), output(count + 16), map(pixels);
        for (size_t i = 0; i < count; ++i) {
            const bool mixed = w >= 17;
            image[i] = 0.15f + (mixed ? 0.65f : 0.3f) * float((i * 29 + 7) % 97) / 96;
            target[i] = (mixed ? 0.1f : 0.55f) + (mixed ? 0.8f : 0.3f) * float((i * 13 + 3) % 101) / 100;
            if (mixed && i % 11 == 0)
                target[i] = image[i]; // L1 subgradient at an exact tie.
        }
        Buffer device_image(count * 4), device_target(count * 4), device_gradient((count + 16) * 4),
            partials(count * 3 * sizeof(__half)), device_map(pixels * 4);
        check(hipMemcpy(device_image.data, image.data(), count * 4, hipMemcpyHostToDevice));
        check(hipMemcpy(device_target.data, target.data(), count * 4, hipMemcpyHostToDevice));
        for (bool valid : {false, true})
            for (float weight : {0.f, 0.2f, 1.f}) {
                std::fill(output.begin(), output.end(), 12345.f);
                check(hipMemcpy(device_gradient.data, output.data(), output.size() * 4, hipMemcpyHostToDevice));
                lfs_hip_photo::forward(h, w, device_image.as(), device_target.as(), partials.as<__half>(), device_map.as(), nullptr);
                lfs_hip_photo::backward(h, w, weight, valid, device_image.as(), device_target.as(), partials.as<__half>(), device_gradient.as(), nullptr);
                check(hipGetLastError());
                check(hipDeviceSynchronize());
                check(hipMemcpy(output.data(), device_gradient.data, output.size() * 4, hipMemcpyDeviceToHost));
                check(hipMemcpy(map.data(), device_map.data, map.size() * 4, hipMemcpyDeviceToHost));
                const auto expected = photo_oracle::evaluate(w, h, image, target, weight, valid);
                if (w == 17 && h == 13 && weight > 0) {
                    for (size_t sample : {size_t{0}, count / 2, count - 1}) {
                        const float original = image[sample];
                        image[sample] = original + 0.0002f;
                        const float plus_input = image[sample];
                        const double plus = photo_oracle::evaluate(w, h, image, target, weight, valid, false).loss;
                        image[sample] = original - 0.0002f;
                        const float minus_input = image[sample];
                        const double minus = photo_oracle::evaluate(w, h, image, target, weight, valid, false).loss;
                        image[sample] = original;
                        const double numerical = (plus - minus) / (double(plus_input) - minus_input);
                        require(std::abs(numerical - expected.gradient[sample]) < 1e-7 + 1e-4 * std::abs(numerical),
                                "CPU photometric analytic gradient differs from finite differences");
                    }
                }
                double error = 0, norm = 0, loss = 0;
                const unsigned x0 = valid && w > 10 ? 5 : 0, y0 = valid && h > 10 ? 5 : 0;
                const double normalized = 1.0 / ((w - 2 * x0) * (h - 2 * y0) * 3);
                for (size_t i = 0; i < count; ++i) {
                    require(std::isfinite(output[i]), "Nonfinite photometric gradient");
                    const double delta = output[i] - expected.gradient[i];
                    error += delta * delta;
                    norm += expected.gradient[i] * expected.gradient[i];
                    const size_t y = (i / w) % h, x = i % w;
                    if (x >= x0 && x < w - x0 && y >= y0 && y < h - y0)
                        loss += ((1 - weight) * std::abs(double(image[i]) - target[i]) + weight * (1 - map[i % pixels])) * normalized;
                }
                for (size_t i = count; i < output.size(); ++i)
                    require(output[i] == 12345.f, "Photometric partial tile overwrote guard");
                const double relative = std::sqrt(error / std::max(norm, 1e-30));
                worst_relative = std::max(worst_relative, relative);
                worst_loss = std::max(worst_loss, std::abs(loss - expected.loss));
                if (relative > 0.01 || std::abs(loss - expected.loss) > 5e-5)
                    std::fprintf(stderr, "%ux%u weight=%g valid=%d: relative gradient %.9g loss delta %.9g\n",
                                 w, h, weight, valid, relative, loss - expected.loss);
                require(relative < 0.01 && std::abs(loss - expected.loss) < 5e-5, "Photometric CPU value/gradient oracle mismatch");
                ++cases;
            }
    }
    std::printf("PASS %u HIP L1/SSIM CPU value/gradient cases, worst relative gradient %.9g, loss delta %.9g\n", cases, worst_relative, worst_loss);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
