/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "effective_rank.cuh"
#include "joint_adam_codec.cuh"
#include "mean_step_scale.cuh"
#include "screen_share.cuh"
#include "sh_value_codec.cuh"
#include "warp_reduce.cuh"
#include <cmath>
#include <bit>
#include <limits>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(hipError_t result) {
    if (result != hipSuccess)
        throw std::runtime_error(hipGetErrorString(result));
}

__global__ void reduce(float* output) {
    using namespace lfs::core::warp_ops;
    const float value = float(threadIdx.x + 1);
    const float sum = block_reduce_sum(value);
    const float low = block_reduce_min(value);
    const float high = block_reduce_max(value);
    const float product = block_reduce_prod(1.0f);
    const float2 pair = block_reduce_min2(make_float2(value, -value));
    const float4 quad = block_reduce_min4(make_float4(value, -value, value * 2, -value * 2));
    if (threadIdx.x == 0) {
        output[0] = sum;
        output[1] = low;
        output[2] = high;
        output[3] = product;
        output[4] = pair.x;
        output[5] = pair.y;
        output[6] = quad.x;
        output[7] = quad.y;
        output[8] = quad.z;
        output[9] = quad.w;
    }
}

__global__ void rank_gradient(float* output) {
    const float values[3] = {0.0f, -2.0f, -3.0f};
    const auto penalty = lfs::training::effective_rank_penalty(values[0], values[1], values[2]);
    output[0] = penalty.value;
    for (unsigned axis = 0; axis < 3; ++axis) {
        float plus[3] = {values[0], values[1], values[2]};
        float minus[3] = {values[0], values[1], values[2]};
        constexpr float epsilon = 0.005f;
        plus[axis] += epsilon;
        minus[axis] -= epsilon;
        output[1 + axis] = penalty.gradient[axis];
        output[4 + axis] = (lfs::training::effective_rank_penalty(plus[0], plus[1], plus[2]).value -
                            lfs::training::effective_rank_penalty(minus[0], minus[1], minus[2]).value) /
                           (2 * epsilon);
    }
    output[7] = lfs::training::per_splat_mean_step_ratio(0, 0, 0, 0);
    output[8] = lfs::training::per_splat_mean_step_ratio(0, 0, 0, 2);
    output[9] = lfs::training::per_splat_mean_step_ratio(10, 10, 10, 1);
}

__global__ void all_lane_sum(float* output, unsigned pattern) {
    const unsigned lane = threadIdx.x & 31u;
    float value = float(lane + 1);
    if (pattern == 1) value = lane == 0 ? 1.25f : 0.0f;
    if (pattern == 2) value = lane == 31 ? -2.5f : 0.0f;
    if (pattern == 3) value = lane & 1u ? -0.25f : 0.125f;
    output[threadIdx.x] = lfs::core::warp_ops::warp_reduce_sum(value);
}

__global__ void bounded_division(float* output, unsigned base) {
    const unsigned i = base + threadIdx.x;
    const float denominator = 1e-4f + float(i & 1023u) * (0.9999f / 1023.0f);
    const float numerator = float((i * 97u) & 65535u) / 65535.0f;
    output[threadIdx.x] = lfs_fast_transmittance_divide(numerator, denominator);
}

// Every lane remains active; the selected source is uniform per wave, not
// necessarily uniform across the block. Floating payloads are compared as bits.
__global__ void uniform_payloads(unsigned* output, unsigned source) {
    const unsigned patterns[16] = {0u, 0x80000000u, 1u, 0x80000001u,
        0x7f800000u, 0xff800000u, 0x7fc12345u, 0xffc54321u,
        0x3f800000u, 0xbf800000u, 0x00800000u, 0x80800000u,
        0x7f7fffffu, 0xff7fffffu, 0x3dcccccdu, 0xbdcccccdu};
    const unsigned lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    const unsigned chosen = (source + wave) & 31u;
    const unsigned raw = patterns[(lane + 2 * source + wave) & 15u] ^ ((lane >> 4) * 0x1234u);
    const unsigned integer = threadIdx.x * 2654435761u ^ 0xdeadbeefu;
    output[2 * threadIdx.x] = __float_as_uint(lfs_fast_uniform_shuffle(__uint_as_float(raw), chosen));
    output[2 * threadIdx.x + 1] = lfs_fast_uniform_shuffle(integer, chosen);
}

__global__ void field_reduction(const float* input, float* output) {
    const unsigned lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    float values[16];
#pragma unroll
    for (unsigned field = 0; field < 16; ++field)
        values[field] = input[(wave * 16 + field) * 32 + lane];
    output[threadIdx.x] = lfs_fast_reduce_scatter16(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8], values[9], values[10], values[11], values[12], values[13], values[14], values[15], lane);
}

static void check_field_reduction(float* output) {
    float* input = nullptr;
    check(hipMalloc(&input, 512 * 16 * sizeof(float)));
    try {
        for (unsigned threads : {32u, 64u, 128u, 512u})
            for (unsigned pattern = 0; pattern < 8; ++pattern) {
                std::vector<float> host(threads * 16), actual(threads);
                for (unsigned wave = 0; wave < threads / 32; ++wave)
                    for (unsigned field = 0; field < 16; ++field)
                        for (unsigned lane = 0; lane < 32; ++lane) {
                            float value = float(100 * wave + 7 * field + lane + 1);
                            if (pattern == 1) value = lane == (field + wave) % 32 ? float(field + 1) * -0.25f : 0.0f;
                            if (pattern == 2) value *= lane & 1u ? -0.75f : 0.5f;
                            if (pattern == 3) value = float(int((lane * 97 + field * 83 + wave * 13) % 1000) - 500) * 0.001f;
                            if (pattern == 4 && lane == 17) {
                                if (field % 4 == 0) value = std::numeric_limits<float>::quiet_NaN();
                                if (field % 4 == 1) value = std::numeric_limits<float>::infinity();
                                if (field % 4 == 2) value = -std::numeric_limits<float>::infinity();
                            }
                            if (pattern == 5 && (lane == 0 || lane == 31))
                                value = lane ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
                            if (pattern == 6) value = std::bit_cast<float>((field + 1) * (lane + 1));
                            if (pattern == 7) value *= 1e30f;
                            host[(wave * 16 + field) * 32 + lane] = value;
                        }
                check(hipMemcpy(input, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice));
                field_reduction<<<1, threads>>>(input, output);
                check(hipGetLastError());
                check(hipMemcpy(actual.data(), output, actual.size() * sizeof(float), hipMemcpyDeviceToHost));
                for (unsigned thread = 0; thread < threads; ++thread) {
                    const unsigned wave = thread / 32, field = thread % 16;
                    double expected = 0.0, magnitude = 0.0;
                    for (unsigned lane = 0; lane < 32; ++lane) {
                        const float value = host[(wave * 16 + field) * 32 + lane];
                        expected += value;
                        magnitude += std::abs(double(value));
                    }
                    bool valid;
                    if (std::isnan(expected)) valid = std::isnan(actual[thread]);
                    else if (std::isinf(expected)) valid = double(actual[thread]) == expected;
                    else if (pattern <= 2 || pattern == 6) valid = actual[thread] == float(expected);
                    else valid = std::isfinite(actual[thread]) && std::abs(double(actual[thread]) - expected) <= 2e-6 * magnitude + 1e-40;
                    if (!valid) throw std::runtime_error("Field-distributed reduction differs from independent CPU sums");
                }
            }
        check(hipFree(input));
    } catch (...) {
        (void)hipFree(input);
        throw;
    }
    std::puts("HIP field reduction: dense/sparse/signed/NaN/Inf/subnormal/large values match CPU sums");
}

int main() {
    float* device = nullptr;
    try {
        hipDeviceProp_t properties{};
        check(hipGetDeviceProperties(&properties, 0));
        if (properties.warpSize != 32)
            throw std::runtime_error("FastGS HIP math requires wave32");
        check(hipMalloc(&device, 1024 * sizeof(float)));
        check_field_reduction(device);
        for (unsigned threads : {32u, 64u, 128u, 256u, 512u, 1024u}) {
            reduce<<<1, threads>>>(device);
            check(hipGetLastError());
            std::vector<float> actual(10);
            check(hipMemcpy(actual.data(), device, 10 * sizeof(float), hipMemcpyDeviceToHost));
            const float n = float(threads);
            const float expected[] = {n * (n + 1) / 2, 1, n, 1, 1, -n, 1, -n, 2, -2 * n};
            for (unsigned i = 0; i < 10; ++i)
                if (actual[i] != expected[i])
                    throw std::runtime_error("FastGS HIP block reduction mismatch");
            std::printf("FastGS HIP mathematical dependencies: %u threads passed\n", threads);
        }
        for (unsigned pattern = 0; pattern < 4; ++pattern) {
            for (unsigned threads : {32u, 128u, 1024u}) {
                all_lane_sum<<<1, threads>>>(device, pattern);
                check(hipGetLastError());
                std::vector<float> actual(threads);
                check(hipMemcpy(actual.data(), device, threads * sizeof(float), hipMemcpyDeviceToHost));
                const float expected[] = {528.0f, 1.25f, -2.5f, -2.0f};
                for (const float value : actual)
                    if (value != expected[pattern])
                        throw std::runtime_error("HIP wave sum must be correct in every lane, including sparse and signed inputs");
            }
        }
        for (unsigned base = 0; base < 65536; base += 1024) {
            bounded_division<<<1, 1024>>>(device, base);
            check(hipGetLastError());
            std::vector<float> actual(1024);
            check(hipMemcpy(actual.data(), device, actual.size() * sizeof(float), hipMemcpyDeviceToHost));
            for (unsigned j = 0; j < actual.size(); ++j) {
                const unsigned i = base + j;
                const float denominator = 1e-4f + float(i & 1023u) * (0.9999f / 1023.0f);
                const float numerator = float((i * 97u) & 65535u) / 65535.0f;
                const double expected = double(numerator) / double(denominator);
                if (!std::isfinite(actual[j]) || std::abs(double(actual[j]) - expected) > std::abs(expected) * 3e-7)
                    throw std::runtime_error("Bounded transmittance division exceeds FP32 error limit");
            }
        }
        const unsigned patterns[16] = {0u, 0x80000000u, 1u, 0x80000001u,
            0x7f800000u, 0xff800000u, 0x7fc12345u, 0xffc54321u,
            0x3f800000u, 0xbf800000u, 0x00800000u, 0x80800000u,
            0x7f7fffffu, 0xff7fffffu, 0x3dcccccdu, 0xbdcccccdu};
        for (unsigned threads : {32u, 64u, 128u, 512u})
            for (unsigned source = 0; source < 32; ++source) {
                uniform_payloads<<<1, threads>>>(reinterpret_cast<unsigned*>(device), source);
                check(hipGetLastError());
                std::vector<unsigned> actual(2 * threads);
                check(hipMemcpy(actual.data(), device, actual.size() * sizeof(unsigned), hipMemcpyDeviceToHost));
                for (unsigned thread = 0; thread < threads; ++thread) {
                    const unsigned wave = thread >> 5, chosen = (source + wave) & 31u;
                    const unsigned expected = patterns[(chosen + 2 * source + wave) & 15u] ^ ((chosen >> 4) * 0x1234u);
                    const unsigned integer = (wave * 32 + chosen) * 2654435761u ^ 0xdeadbeefu;
                    if (actual[2 * thread] != expected || actual[2 * thread + 1] != integer)
                        throw std::runtime_error("Uniform wave payload must preserve all float/integer bits");
                }
            }
        std::puts("Uniform wave payload: every source lane, multiple waves, finite/nonfinite/signed-zero bits exact");
        rank_gradient<<<1, 1>>>(device);
        check(hipGetLastError());
        std::vector<float> rank(10);
        check(hipMemcpy(rank.data(), device, 10 * sizeof(float), hipMemcpyDeviceToHost));
        if (!std::isfinite(rank[0]) || rank[0] <= 0)
            throw std::runtime_error("Effective-rank penalty is invalid");
        for (unsigned axis = 0; axis < 3; ++axis)
            if (!std::isfinite(rank[1 + axis]) || !std::isfinite(rank[4 + axis]) ||
                std::abs(rank[1 + axis] - rank[4 + axis]) > 0.01f)
                throw std::runtime_error("Effective-rank gradient differs from finite differences");
        if (rank[7] != 1 || rank[8] != 1 || rank[9] != 300)
            throw std::runtime_error("Per-splat mean step bounds mismatch");
        std::puts("FastGS HIP effective-rank finite differences and mean step bounds passed");
        check(hipFree(device));
        return 0;
    } catch (const std::exception& error) {
        if (device)
            (void)hipFree(device);
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
