/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Isolated wave32 training experiment using the checkout's generated gsplat
// forward/backward. Conservative tile lists are fixed and include every splat;
// this deliberately tests differentiation, not projection/sorting performance.
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include "Rasterization.h"
#include "hip_check.hpp"
#ifdef LFS_HIP_TEST_BINNING
#include "gsplat_binning.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {
    constexpr unsigned width = 35, height = 29, pixels = width * height;
#ifndef LFS_HIP_TEST_GAUSSIANS
#define LFS_HIP_TEST_GAUSSIANS 6
#endif
    constexpr unsigned n = LFS_HIP_TEST_GAUSSIANS, channels = pixels * 3;

    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    struct Buffer {
        float* data = nullptr;
        size_t count;
        explicit Buffer(size_t count) : count(count) {
            hip_check(hipMalloc(&data, count * sizeof(float)), "allocate");
        }
        ~Buffer() { (void)hipFree(data); }
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        void upload(const std::vector<float>& value) {
            require(value.size() == count, "Upload shape mismatch");
            hip_check(hipMemcpy(data, value.data(), count * sizeof(float), hipMemcpyHostToDevice), "upload");
        }
        std::vector<float> read() const {
            std::vector<float> result(count);
            hip_check(hipMemcpy(result.data(), data, count * sizeof(float), hipMemcpyDeviceToHost), "read");
            return result;
        }
        void zero(hipStream_t stream) { hip_check(hipMemsetAsync(data, 0, count * sizeof(float), stream), "zero"); }
    };

    struct Parameter {
        Buffer value, gradient, first, second;
        float rate;
        Parameter(size_t count, float rate) : value(count), gradient(count), first(count), second(count), rate(rate) {}
    };

    __global__ void mse_gradient(const float* image, const float* target, float* gradient) {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < channels)
            gradient[i] = 2.0f * (image[i] - target[i]) / channels;
    }

    __global__ void adam(float* parameter, const float* gradient, float* first, float* second,
                         size_t count, float rate, float correction1, float correction2) {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= count)
            return;
        const float m = first[i] = 0.9f * first[i] + 0.1f * gradient[i];
        const float v = second[i] = 0.999f * second[i] + 0.001f * gradient[i] * gradient[i];
        parameter[i] -= rate * (m / correction1) / (sqrtf(v / correction2) + 1.0e-8f);
    }

    double mse(const std::vector<float>& image, const std::vector<float>& target) {
        double result = 0;
        for (size_t i = 0; i < image.size(); ++i) {
            require(std::isfinite(image[i]), "Nonfinite image");
            const double difference = double(image[i]) - target[i];
            result += difference * difference;
        }
        return result / image.size();
    }

    void save(const char* path, const std::vector<float>& image) {
        std::ofstream file(path, std::ios::binary);
        file << "P6\n"
             << width << ' ' << height << "\n255\n";
        for (unsigned i = 0; i < pixels; ++i)
            for (unsigned c = 0; c < 3; ++c) {
                const auto value = static_cast<unsigned char>(std::clamp(image[c * pixels + i], 0.0f, 1.0f) * 255 + 0.5f);
                file.write(reinterpret_cast<const char*>(&value), 1);
            }
        require(bool(file), "Could not save diagnostic render");
    }

    // Independent double-precision CPU renderer for the identity camera. Uses
    // explicit row-major rotation and line distance rather than GLM/device code.
    std::vector<float> reference(const std::vector<float>& means, const std::vector<float>& quats,
                                 const std::vector<float>& scales, const std::vector<float>& colors,
                                 const std::vector<float>& opacities) {
        std::vector<float> image(channels);
        for (unsigned y = 0; y < height; ++y)
            for (unsigned x = 0; x < width; ++x) {
                const double ray[3] = {(x + 0.5 - width / 2.0) / 32, (y + 0.5 - height / 2.0) / 32, 1};
                double transmission = 1;
                for (unsigned g = 0; g < n; ++g) {
                    double q[4], norm = 0;
                    for (unsigned k = 0; k < 4; ++k)
                        norm += double(quats[g * 4 + k]) * quats[g * 4 + k];
                    for (unsigned k = 0; k < 4; ++k)
                        q[k] = quats[g * 4 + k] / std::sqrt(norm);
                    const auto [w, a, b, c] = std::array<double, 4>{q[0], q[1], q[2], q[3]};
                    const double rotation[3][3] = {
                        {1 - 2 * (b * b + c * c), 2 * (a * b - w * c), 2 * (a * c + w * b)},
                        {2 * (a * b + w * c), 1 - 2 * (a * a + c * c), 2 * (b * c - w * a)},
                        {2 * (a * c - w * b), 2 * (b * c + w * a), 1 - 2 * (a * a + b * b)}};
                    double origin[3]{}, direction[3]{}, direction_norm = 0;
                    for (unsigned row = 0; row < 3; ++row) {
                        for (unsigned col = 0; col < 3; ++col) {
                            origin[row] -= rotation[col][row] * means[g * 3 + col];
                            direction[row] += rotation[col][row] * ray[col];
                        }
                        const double scale = std::exp(scales[g * 3 + row]);
                        origin[row] /= scale;
                        direction[row] /= scale;
                        direction_norm += direction[row] * direction[row];
                    }
                    for (auto& value : direction)
                        value /= std::sqrt(direction_norm);
                    double distance = 0;
                    for (unsigned k = 0; k < 3; ++k) {
                        const double cross = direction[(k + 1) % 3] * origin[(k + 2) % 3] - direction[(k + 2) % 3] * origin[(k + 1) % 3];
                        distance += cross * cross;
                    }
                    const double alpha = std::min(0.999, std::exp(-distance / 2) / (1 + std::exp(-opacities[g])));
                    if (alpha < 1.0 / 255)
                        continue;
                    const double next = transmission * (1 - alpha);
                    if (next <= 1e-4)
                        break;
                    for (unsigned k = 0; k < 3; ++k)
                        image[k * pixels + y * width + x] += float(colors[g * 3 + k] * alpha * transmission);
                    transmission = next;
                }
            }
        return image;
    }
} // namespace

int main() try {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int runtime_version = 0;
    hip_check(hipRuntimeGetVersion(&runtime_version), "runtime version");
    std::printf("HIP runtime version: %d\n", runtime_version);
#ifdef _WIN32
    char module_path[32768]{};
    const auto module = GetModuleHandleA("amdhip64_7.dll");
    require(module && GetModuleFileNameA(module, module_path, sizeof(module_path)), "Cannot identify HIP runtime");
    std::printf("HIP runtime module: %s\n", module_path);
#endif
    hip_check(hipSetDevice(0), "select device");
    hipDeviceProp_t device{};
    hip_check(hipGetDeviceProperties(&device, 0), "device properties");
    require(device.warpSize == 32, "This rasterizer experiment is validated only for wave32");
    std::printf("Experimental gsplat HIP training: %s (%s), %ux%u, %u Gaussians\n", device.name, device.gcnArchName, width, height, n);
    hipStream_t stream = nullptr;
    hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create stream");
    Parameter means(n * 3, 0.002f), quats(n * 4, 0.002f), scales(n * 3, 0.003f), colors(n * 3, 0.015f), opacity(n, 0.01f);
    Parameter* parameters[] = {&means, &quats, &scales, &colors, &opacity};
    std::vector<float> hm(n * 3), hq(n * 4), hs(n * 3), hc(n * 3), ho(n);
    for (unsigned g = 0; g < n; ++g) {
        hm[g * 3] = (int(g % 3) - 1) * 0.65f;
        hm[g * 3 + 1] = (int((g / 3) % 2) * 2 - 1) * 0.45f;
        hm[g * 3 + 2] = 3.0f + (n > 6 ? 0.005f : 0.25f) * g;
        hq[g * 4] = 1;
        hq[g * 4 + 1] = 0.08f * (g % 6);
        hq[g * 4 + 2] = 0.03f * (g % 6);
        for (unsigned c = 0; c < 3; ++c) {
            hs[g * 3 + c] = std::log(0.22f + 0.06f * c + 0.01f * (g % 6));
            hc[g * 3 + c] = 0.2f + float((g * 3 + c * 5) % 7) * 0.1f;
        }
        ho[g] = (n > 6 ? -3.5f : 0.3f) + 0.1f * (g % 6);
    }
    means.value.upload(hm);
    quats.value.upload(hq);
    scales.value.upload(hs);
    colors.value.upload(hc);
    opacity.value.upload(ho);
    Buffer view(16), intrinsics(9), image(channels), alpha(pixels), target(channels), image_gradient(channels), alpha_gradient(pixels), last(pixels);
    view.upload({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
    intrinsics.upload({32, 0, width / 2.0f, 0, 32, height / 2.0f, 0, 0, 1});
    constexpr unsigned tiles = ((width + 15) / 16) * ((height + 15) / 16);
    Buffer offsets(tiles + 1), ids(tiles * n);
    std::vector<int32_t> host_offsets(tiles + 1), host_ids(tiles * n);
    for (unsigned t = 0; t < tiles; ++t) {
        host_offsets[t] = t * n;
        for (unsigned g = 0; g < n; ++g)
            host_ids[t * n + g] = g;
    }
    host_offsets[tiles] = tiles * n;
    hip_check(hipMemcpy(offsets.data, host_offsets.data(), (tiles + 1) * sizeof(int32_t), hipMemcpyHostToDevice), "tile offsets");
    hip_check(hipMemcpy(ids.data, host_ids.data(), tiles * n * sizeof(int32_t), hipMemcpyHostToDevice), "tile ids");
#ifdef LFS_HIP_TEST_BINNING
    HipGsplatBinning binning(n, width, height, stream);
    std::puts("Dynamic HIP projection, tile counts, prefix scan, depth sort and offsets enabled");
#endif
    const int32_t* active_offsets = reinterpret_cast<int32_t*>(offsets.data);
    const int32_t* active_ids = reinterpret_cast<int32_t*>(ids.data);
    unsigned active_capacity = tiles * n;
    const auto forward = [&] {
#ifdef LFS_HIP_TEST_BINNING
        binning.run(means.value.data, quats.value.data, scales.value.data, view.data, intrinsics.data, stream);
        active_offsets = binning.offsets();
        active_ids = binning.ids();
        active_capacity = binning.capacity();
#endif
        gsplat_lfs::launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel<3>(
            means.value.data, quats.value.data, scales.value.data, colors.value.data, opacity.value.data,
            nullptr, nullptr, nullptr, 1, n, active_capacity, width, height, 16,
            view.data, nullptr, intrinsics.data, PINHOLE, {}, ShutterType::GLOBAL, nullptr, nullptr, nullptr,
            active_offsets, active_ids,
            image.data, alpha.data, reinterpret_cast<int32_t*>(last.data), stream);
    };
    const auto backward = [&] {
        alpha_gradient.zero(stream);
        for (auto* parameter : parameters)
            parameter->gradient.zero(stream);
        hipLaunchKernelGGL(mse_gradient, dim3((channels + 255) / 256), dim3(256), 0, stream, image.data, target.data, image_gradient.data);
        hip_check(hipGetLastError(), "loss gradient");
        gsplat_lfs::launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel<3>(
            means.value.data, quats.value.data, scales.value.data, colors.value.data, opacity.value.data,
            nullptr, nullptr, nullptr, 1, n, active_capacity, width, height, 16,
            view.data, nullptr, intrinsics.data, PINHOLE, {}, ShutterType::GLOBAL, nullptr, nullptr, nullptr,
            active_offsets, active_ids, alpha.data,
            reinterpret_cast<int32_t*>(last.data), image_gradient.data, alpha_gradient.data,
            means.gradient.data, quats.gradient.data, scales.gradient.data, colors.gradient.data, opacity.gradient.data,
            nullptr, nullptr, nullptr, nullptr, stream);
    };
    forward();
    hip_check(hipStreamSynchronize(stream), "target render");
#ifdef LFS_HIP_TEST_BINNING
    binning.verify(stream);
#endif
    const auto rendered_target = image.read();
    const auto target_image = reference(hm, hq, hs, hc, ho);
    float max_error = 0;
    for (size_t i = 0; i < channels; ++i)
        max_error = std::max(max_error, std::abs(target_image[i] - rendered_target[i]));
    std::printf("Forward vs independent CPU: max absolute error %.9g\n", max_error);
    require(max_error < 3e-5f, "Independent CPU render mismatch");
    target.upload(target_image);
    save("hip-target.ppm", target_image);
    for (unsigned g = 0; g < n; ++g) {
        hm[g * 3] += 0.07f;
        hq[g * 4 + 2] += 0.04f;
        hs[g * 3 + 1] += 0.08f;
        hc[g * 3] *= 0.6f;
        hc[g * 3 + 1] *= 0.7f;
        ho[g] -= 0.25f;
    }
    means.value.upload(hm);
    quats.value.upload(hq);
    scales.value.upload(hs);
    colors.value.upload(hc);
    opacity.value.upload(ho);
    forward();
    backward();
    hip_check(hipStreamSynchronize(stream), "initial gradients");
    const auto initial_image = image.read();
    const double initial_loss = mse(initial_image, target_image);
    save("hip-initial.ppm", initial_image);
    // Finite differences of the rendered objective, across all parameter families.
    for (unsigned family = 0; family < 5; ++family) {
        auto& parameter = *parameters[family];
        const auto original = parameter.value.read();
        const auto gradient = parameter.gradient.read();
        double max_relative = 0;
        for (size_t index : {size_t(0), original.size() / 2, original.size() - 1}) {
            constexpr float epsilon = 0.0001f;
            auto perturbed = original;
            perturbed[index] += epsilon;
            parameter.value.upload(perturbed);
            forward();
            hip_check(hipStreamSynchronize(stream), "positive perturbation");
            const double positive = mse(image.read(), target_image);
            perturbed[index] = original[index] - epsilon;
            parameter.value.upload(perturbed);
            forward();
            hip_check(hipStreamSynchronize(stream), "negative perturbation");
            const double negative = mse(image.read(), target_image);
            const double numeric = (positive - negative) / (2 * epsilon);
            const double difference = std::abs(numeric - gradient[index]);
            max_relative = std::max(max_relative, difference / std::max(1e-6, std::abs(numeric)));
            std::printf("Gradient family %u index %zu: analytic %.8g numeric %.8g\n", family, index, gradient[index], numeric);
            require(std::isfinite(gradient[index]) && difference < 1e-6 + 0.02 * std::abs(numeric), "Finite-difference gradient mismatch");
        }
        parameter.value.upload(original);
        std::printf("Gradient family %u: max relative %.6g\n", family, max_relative);
    }
    for (auto* parameter : parameters) {
        parameter->first.zero(stream);
        parameter->second.zero(stream);
    }
    constexpr unsigned steps = 400;
    const auto start = std::chrono::steady_clock::now();
    for (unsigned step = 1; step <= steps; ++step) {
        forward();
        backward();
        for (auto* parameter : parameters) {
            const size_t count = parameter->value.count;
            hipLaunchKernelGGL(adam, dim3((count + 255) / 256), dim3(256), 0, stream,
                               parameter->value.data, parameter->gradient.data, parameter->first.data,
                               parameter->second.data, count, parameter->rate,
                               1.0f - std::pow(0.9f, float(step)), 1.0f - std::pow(0.999f, float(step)));
            hip_check(hipGetLastError(), "Adam launch");
        }
        if (step % 100 == 0) {
            forward();
            hip_check(hipStreamSynchronize(stream), "training checkpoint");
            std::printf("Step %u: MSE %.9g\n", step, mse(image.read(), target_image));
        }
    }
    hip_check(hipStreamSynchronize(stream), "training finish");
    const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const auto final_image = image.read();
    const double final_loss = mse(final_image, target_image);
    save("hip-final.ppm", final_image);
    require(final_loss < initial_loss * 0.02, "Training did not sufficiently reduce image loss");
    for (auto* parameter : parameters) {
        for (float value : parameter->value.read())
            require(std::isfinite(value), "Nonfinite trained parameter");
        for (float value : parameter->gradient.read())
            require(std::isfinite(value), "Nonfinite trained gradient");
    }
    std::printf("PASS native HIP raster training: MSE %.9g -> %.9g, %u steps in %.3f ms (%.3f ms/step)\n",
                initial_loss, final_loss, steps, elapsed, elapsed / steps);
#ifdef LFS_HIP_TEST_BINNING
    binning.verify(stream);
    auto culled_means = means.value.read();
    for (unsigned g = 0; g < n; ++g)
        culled_means[g * 3 + 2] = 2000 + g;
    means.value.upload(culled_means);
    forward();
    backward();
    binning.verify(stream);
    for (float value : image.read())
        require(value == 0, "Empty frame did not clear RGB");
    for (auto* parameter : parameters)
        for (float value : parameter->gradient.read())
            require(value == 0, "Empty frame produced a parameter gradient");
    std::puts("PASS: all-culled frame, zero intersections, RGB and gradients");
    std::puts("Toy scene with dynamic GPU binning, MSE and FP32 Adam: NOT a dataset-training benchmark.");
#else
    std::puts("Toy scene, fixed conservative tile lists, MSE and FP32 Adam: NOT a dataset-training benchmark.");
#endif
#ifndef LFS_HIP_TEST_BINNING
    // Opaque prefix with an unused tail: every tile has a different global
    // offset, and edge tiles contain inactive lanes. Clipping must preserve
    // both the prefix image and its gradients, including the exclusive stop.
    for (unsigned g = 0; g < n; ++g) {
        hm[g * 3] = hm[g * 3 + 1] = 0;
        hm[g * 3 + 2] = 3.f + 0.001f * g;
        for (unsigned c = 0; c < 4; ++c)
            hq[g * 4 + c] = c == 0 ? 1.f : 0.f;
        for (unsigned c = 0; c < 3; ++c) {
            hs[g * 3 + c] = std::log(10.f);
            hc[g * 3 + c] = 0.2f + 0.1f * c;
        }
        ho[g] = 8.f;
    }
    means.value.upload(hm);
    quats.value.upload(hq);
    scales.value.upload(hs);
    colors.value.upload(hc);
    opacity.value.upload(ho);
    target.upload(std::vector<float>(channels, 0));
    forward();
    backward();
    hip_check(hipStreamSynchronize(stream), "opaque prefix gradients");
    const auto opaque_image = image.read();
    std::array<std::vector<float>, 5> opaque_gradients;
    for (unsigned f = 0; f < 5; ++f)
        opaque_gradients[f] = parameters[f]->gradient.read();
    std::vector<int32_t> accepted(pixels);
    hip_check(hipMemcpy(accepted.data(), last.data, pixels * sizeof(int32_t), hipMemcpyDeviceToHost), "last accepted IDs");
    unsigned prefix = 0;
    for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x) {
            const unsigned tile = (y / 16) * ((width + 15) / 16) + x / 16;
            const int local = accepted[y * width + x] - int(tile * n);
            require(local >= 0 && local < int(n), "Opaque fixture has invalid accepted index");
            prefix = std::max(prefix, unsigned(local) + 1);
        }
    require(prefix < n, "Opaque fixture did not terminate before its tail");
    const size_t attributes[]{3, 4, 3, 3, 1};
    for (unsigned f = 0; f < 5; ++f)
        for (size_t i = size_t(prefix) * attributes[f]; i < opaque_gradients[f].size(); ++i)
            require(opaque_gradients[f][i] == 0, "Unaccepted opaque tail has a gradient");
    for (unsigned g = prefix; g < n; ++g)
        ho[g] = -100.f;
    opacity.value.upload(ho);
    forward();
    backward();
    hip_check(hipStreamSynchronize(stream), "transparent tail gradients");
    require(image.read() == opaque_image, "Unaccepted tail changed the image");
    for (unsigned f = 0; f < 5; ++f) {
        const auto actual = parameters[f]->gradient.read();
        for (size_t i = 0; i < actual.size(); ++i)
            require(std::isfinite(actual[i]) && std::abs(actual[i] - opaque_gradients[f][i]) <
                                                    1e-6f + 1e-5f * std::abs(opaque_gradients[f][i]),
                    "Unaccepted tail changed prefix gradients");
    }
    std::fill(ho.begin(), ho.end(), -100.f);
    opacity.value.upload(ho);
    target.upload(std::vector<float>(channels, 1.f));
    forward();
    backward();
    hip_check(hipStreamSynchronize(stream), "transparent frame completion");
    for (float value : image.read())
        require(value == 0, "Fully transparent tiles changed RGB");
    for (auto* parameter : parameters)
        for (float value : parameter->gradient.read())
            require(value == 0, "Fully transparent tiles produced gradients");
    std::printf("PASS opaque prefix %u/%u, unused tails and transparent tiles, including partial edge tiles\n", prefix, n);
#endif
    hip_check(hipStreamDestroy(stream), "destroy stream");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
