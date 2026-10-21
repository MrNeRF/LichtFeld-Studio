/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "backward.h"
#include "buffer_utils.h"
#include "forward.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace raster = fast_lfs::rasterization;
static void check(hipError_t result) {
    if (result != hipSuccess)
        throw std::runtime_error(hipGetErrorString(result));
}

struct Memory {
    std::vector<void*> allocations;
    size_t bytes = 0;
    ~Memory() {
        for (void* allocation : allocations)
            (void)hipFree(allocation);
    }
    char* allocate(size_t count) {
        count = std::max(size_t(256), count);
        if (count > 64 * 1024 * 1024 - bytes)
            throw std::runtime_error("FastGS probe memory budget exhausted");
        void* pointer = nullptr;
        check(hipMalloc(&pointer, std::max(size_t(256), count)));
        allocations.push_back(pointer);
        bytes += count;
        check(hipMemset(pointer, 0, std::max(size_t(256), count)));
        return static_cast<char*>(pointer);
    }
    template <class T>
    T* upload(std::initializer_list<T> data) {
        auto* result = reinterpret_cast<T*>(allocate(data.size() * sizeof(T)));
        check(hipMemcpy(result, data.begin(), data.size() * sizeof(T), hipMemcpyHostToDevice));
        return result;
    }
};


static std::vector<float> run(unsigned width, unsigned height, bool mip, int cull, unsigned extra_channels, bool densify, bool edge_scores, float gradient_scale = 1.0f, unsigned mixed_axes = 0, bool wide_splat = false) {
    const bool depth_channel = (extra_channels & 1u) != 0;
    const bool normal_channel = (extra_channels & 2u) != 0;
    Memory model, frame;
    const unsigned pixels = width * height;
    auto* means = model.upload<float3>({make_float3(0, 0, 3)});
    auto* scales = model.upload<float3>({wide_splat ? make_float3(4.0f) : make_float3(-1.6f, -1.4f, -1.9f)});
    auto* rotations = model.upload<float4>({make_float4(1, 0, 0, 0)});
    auto* opacity = model.upload<float>({0});
    auto* sh0 = model.upload<float3>({make_float3(0, 0, 0)});
    auto* matrix = model.upload<float4>({make_float4(1, 0, 0, 0), make_float4(0, 1, 0, 0), make_float4(0, 0, 1, 0), make_float4(0, 0, 0, 1)});
    auto* camera = model.upload<float3>({make_float3(0)});
    auto* image = reinterpret_cast<float*>(model.allocate(3 * pixels * sizeof(float)));
    auto* alpha = reinterpret_cast<float*>(model.allocate(pixels * sizeof(float)));
    auto* depth = depth_channel ? reinterpret_cast<float*>(model.allocate(pixels * sizeof(float))) : nullptr;
    auto* normal = normal_channel ? reinterpret_cast<float*>(model.allocate(3 * pixels * sizeof(float))) : nullptr;
    auto* gradient = reinterpret_cast<float*>(model.allocate(3 * pixels * sizeof(float)));
    std::vector<float> ones(3 * pixels, gradient_scale);
    if (mixed_axes) {
        // Opposite signs across original 8x8 domains, plus one saturating
        // domain with tiny neighbors to exercise mixed safe/unsafe merging.
        for (unsigned channel = 0; channel < 3; ++channel)
            for (unsigned y = 0; y < height; ++y)
                for (unsigned x = 0; x < width; ++x) {
                    const size_t index = channel * pixels + y * width + x;
                    if (mixed_axes >= 4) {
                        const unsigned quad = ((x % 16) / 8) + 2 * ((y % 16) / 8);
                        ones[index] *= quad == mixed_axes - 4 ? (channel & 1u ? -0.75f : 1.0f) : 1e-12f;
                    } else {
                        ones[index] *= (((mixed_axes == 1 ? x / 8 : mixed_axes == 2 ? y / 8 : x / 8 + y / 8) + channel) & 1u) ? -0.75f : 1.0f;
                    }
                }
    }
    check(hipMemcpy(gradient, ones.data(), ones.size() * sizeof(float), hipMemcpyHostToDevice));
    char* tiles = nullptr;
    auto allocator = [&](size_t bytes) { return frame.allocate(bytes); };
    raster::set_warp_cull_mode_for_testing(cull);
    auto forward = raster::forward(allocator, [](size_t) {}, allocator, [&](const void* source, size_t size) {
            char* destination = frame.allocate(size);
            check(hipMemcpy(destination, source, size, hipMemcpyDeviceToDevice));
            return destination; }, [&](size_t size) { tiles = frame.allocate(size); return tiles; }, means, scales, rotations, opacity, sh0, nullptr, nullptr, 0, 0, matrix, camera, image, alpha, depth, normal, nullptr, nullptr, 1, 1, 1, width, height, 30, 30, width / 2.0f, height / 2.0f, 0.01f, 100, mip, nullptr);
    check(hipDeviceSynchronize());
    if (forward.n_visible != 1 || forward.n_instances <= 0)
        throw std::runtime_error("FastGS forward lost visible Gaussian");
    std::vector<float> host_alpha(pixels), host_image(3 * pixels);
    check(hipMemcpy(host_alpha.data(), alpha, pixels * sizeof(float), hipMemcpyDeviceToHost));
    check(hipMemcpy(host_image.data(), image, 3 * pixels * sizeof(float), hipMemcpyDeviceToHost));
    double sum_alpha = 0;
    for (unsigned pixel = 0; pixel < pixels; ++pixel) {
        const float a = host_alpha[pixel];
        if (!std::isfinite(a) || a < 0 || a > 0.501f)
            throw std::runtime_error("FastGS invalid alpha");
        sum_alpha += a;
        for (unsigned channel = 0; channel < 3; ++channel)
            if (std::abs(host_image[channel * pixels + pixel] - 0.5f * a) > 1e-6f)
                throw std::runtime_error("FastGS color/alpha analytical mismatch");
    }
    if (sum_alpha < 1)
        throw std::runtime_error("FastGS rendered empty image");
    double expected_opacity_gradient = 0.75 * sum_alpha;
    std::vector<float> channel_values(3 * pixels);
    if (depth_channel) {
        check(hipMemcpy(channel_values.data(), depth, pixels * sizeof(float), hipMemcpyDeviceToHost));
        for (unsigned i = 0; i < pixels; ++i)
            expected_opacity_gradient += 0.5 * channel_values[i];
    }
    if (normal_channel) {
        check(hipMemcpy(channel_values.data(), normal, 3 * pixels * sizeof(float), hipMemcpyDeviceToHost));
        for (float value : channel_values)
            expected_opacity_gradient += 0.5 * value;
    }
    auto* helper = reinterpret_cast<float*>(model.allocate(16 * sizeof(float)));
#ifdef LFS_HIP_FAST_PACKED_GRADIENTS
    constexpr unsigned opacity_field = 5, color_field = 6, mean_field = 0, conic_field = 2;
#else
    constexpr unsigned opacity_field = 0, color_field = 1, mean_field = 4, conic_field = 6;
#endif
    auto* parameter_gradient = reinterpret_cast<float*>(model.allocate(14 * sizeof(float)));
    auto* statistics = reinterpret_cast<float*>(model.allocate(3 * sizeof(float)));
    raster::FusedAdamSettings settings{};
    unsigned gradient_offset = 0;
    for (auto [parameter, attributes] : {std::pair{&settings.means, 3}, {&settings.rotation, 4}, {&settings.scaling, 3}, {&settings.sh0, 3}, {&settings.opacity, 1}}) {
        parameter->hip_gradient = parameter_gradient + gradient_offset;
        parameter->n_primitives = 1;
        parameter->n_attributes = attributes;
        parameter->n_elements = attributes;
        gradient_offset += attributes;
    }
    raster::backward(densify ? gradient : nullptr, gradient, {}, depth_channel ? gradient : nullptr, normal_channel ? gradient : nullptr, image, alpha,
                     means, scales, rotations, opacity, nullptr, matrix, camera, forward.primitive_normals,
                     forward.per_primitive_buffers, tiles, forward.sorted_primitive_indices,
                     forward.primitive_work_indices, helper + opacity_field, reinterpret_cast<float3*>(helper + color_field),
                     reinterpret_cast<float2*>(helper + mean_field), reinterpret_cast<float3*>(helper + conic_field),
                     helper + 9, normal_channel ? reinterpret_cast<float3*>(helper + 10) : nullptr, nullptr, densify ? statistics : nullptr, 1, forward.n_instances, forward.n_visible,
                     1, 1, width, height, 30, 30, width / 2.0f, height / 2.0f,
                     mip, densify ? DensificationType::MCMC : DensificationType::None, settings, nullptr, 0, 0, edge_scores ? gradient : nullptr, edge_scores ? statistics + 2 : nullptr, nullptr);
    check(hipDeviceSynchronize());
    std::vector<float> host_helper(16);
    check(hipMemcpy(host_helper.data(), helper, 16 * sizeof(float), hipMemcpyDeviceToHost));
    if (gradient_scale != 1.0f || mixed_axes) {
        std::vector<float> parameter_values(14);
        check(hipMemcpy(parameter_values.data(), parameter_gradient, parameter_values.size() * sizeof(float), hipMemcpyDeviceToHost));
        host_helper.insert(host_helper.end(), parameter_values.begin(), parameter_values.end());
        for (float value : host_helper)
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite scaled clipping result");
        return host_helper;
    }
    for (unsigned channel = 0; channel < 3; ++channel)
        if (!std::isfinite(host_helper[color_field + channel]) ||
            std::abs(host_helper[color_field + channel] - sum_alpha) > 1e-3 * sum_alpha)
            throw std::runtime_error("FastGS color backward analytical mismatch");
    std::vector<float> host_parameter_gradient(14);
    check(hipMemcpy(host_parameter_gradient.data(), parameter_gradient, 14 * sizeof(float), hipMemcpyDeviceToHost));
    for (float value : host_parameter_gradient)
        if (!std::isfinite(value))
            throw std::runtime_error("FastGS parameter gradient is nonfinite");
    for (unsigned channel = 0; channel < 3; ++channel)
        if (std::abs(host_parameter_gradient[10 + channel] - 0.28209479177387814 * sum_alpha) > 0.001 * sum_alpha)
            throw std::runtime_error("FastGS SH0 gradient adapter mismatch");
    if (std::abs(host_parameter_gradient[13] - expected_opacity_gradient) > 0.001 * sum_alpha)
        throw std::runtime_error("FastGS opacity gradient adapter mismatch");
    float host_statistics[3];
    check(hipMemcpy(host_statistics, statistics, sizeof(host_statistics), hipMemcpyDeviceToHost));
    if (std::abs(host_statistics[2] - (edge_scores ? sum_alpha : 0)) > 0.001 * sum_alpha ||
        (densify && (std::abs(host_statistics[0] - sum_alpha) > 0.001 * sum_alpha ||
                     std::abs(host_statistics[1] - sum_alpha) > 0.001 * sum_alpha)))
        throw std::runtime_error("FastGS densification/edge statistics mismatch");
    for (unsigned field = 9; field < 13; ++field) {
        const double expected = (field == 9 ? depth_channel : normal_channel) ? sum_alpha : 0;
        if (std::abs(host_helper[field] - expected) > 0.001 * sum_alpha)
            throw std::runtime_error("FastGS depth/normal gradient mismatch");
    }
    std::printf("FastGS HIP %ux%u mip=%d cull=%d channels=%u densify=%d edge=%d: alpha %.9g passed\n",
                width, height, mip, cull, extra_channels, densify, edge_scores, sum_alpha);
    return host_helper;
}

int main() {
    try {
        hipDeviceProp_t properties{};
        check(hipGetDeviceProperties(&properties, 0));
        if (properties.warpSize != 32)
            throw std::runtime_error("FastGS HIP requires wave32");
        for (bool mip : {false, true})
            for (int cull : {0, 1})
                for (unsigned channels : {0u, 1u, 2u, 3u})
                    for (bool edge : {false, true})
                        for (bool densify : {false, true}) {
                            // The original CUDA test hook deliberately instantiates
                            // only None/without-normal for disabled culling. Exercise
                            // optional channels/statistics through production mode.
                            if (cull == 1 && (channels || densify))
                                continue;
                            run(35, 29, mip, cull, channels, densify, edge);
                            run(64, 48, mip, cull, channels, densify, edge);
                        }
        for (unsigned mixed : {0u, 1u, 2u, 3u})
            for (unsigned width : {17u, 35u})
                for (float scale : {1e-4f, -1e-4f, 100.0f, -100.0f, 1e8f, -1e8f}) {
                    const auto reference = run(width, 29, false, 1, 0, false, false, scale, mixed);
                    const auto optimized = run(width, 29, false, 0, 0, false, false, scale, mixed);
                    if (!mixed && std::abs(scale) >= 1e8f &&
                        std::none_of(reference.begin(), reference.begin() + 16,
                                     [](float value) { return std::abs(value) >= 9999.0f; }))
                        throw std::runtime_error("Clipping probe did not reach its saturation regime");
                    for (size_t field = 0; field < reference.size(); ++field)
                        if (std::abs(reference[field] - optimized[field]) > 0.05f + 5e-5f * std::abs(reference[field]))
                            throw std::runtime_error("Per-quad clipping changed in optimized RGB backward");
                }
        for (unsigned hot_quad : {4u, 5u, 6u, 7u})
            for (unsigned width : {17u, 35u})
                for (float scale : {1e8f, -1e8f}) {
                    const auto reference = run(width, 29, false, 1, 0, false, false, scale, hot_quad);
                    const auto optimized = run(width, 29, false, 0, 0, false, false, scale, hot_quad);
                    if (std::none_of(reference.begin(), reference.begin() + 16,
                                     [](float value) { return std::abs(value) >= 9999.0f; }))
                        throw std::runtime_error("Mixed clipping probe did not reach saturation");
                    for (size_t field = 0; field < reference.size(); ++field)
                        if (std::abs(reference[field] - optimized[field]) > 0.05f + 5e-5f * std::abs(reference[field]))
                            throw std::runtime_error("Mixed safe/unsafe quad clipping changed");
                }
        // Isotropic sigma >500px spans at least 48 tile rows and forces the
        // cooperative emitter, with just one live source in its first wave.
        run(1025, 769, false, 0, 0, false, false, 1.0f, 0, true);
        run(1025, 769, false, 1, 0, false, false, 1.0f, 0, true);
        std::puts("FastGS clipping: positive/negative small, mixed and saturating gradients match the uncull reference");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
