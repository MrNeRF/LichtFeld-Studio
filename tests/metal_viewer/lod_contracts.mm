/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lod_selector.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace lfs::rendering::metal;
static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
static void run(id<MTLDevice> device) {
    LodSelector selector(device);
    auto queue = [device newCommandQueue];
    const auto upload = [&](const void* bytes, size_t count) { return BufferSlice{[device newBufferWithBytes:bytes length:count options:MTLResourceStorageModeShared]}; };
    const std::array<uint32_t, 21> links = {1, 2, 0xffffffffu, 3, 2, 0, 5, 2, 0,
                                            0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 2};
    for (int scenario = 0; scenario < 10; ++scenario) {
        const std::array<float, 7> sizes = scenario == 3 ? std::array<float, 7>{.05f, 1, 1, .02f, .02f, .02f, .02f} : scenario == 4 ? std::array<float, 7>{.525f, .1f, .1f, .02f, .02f, .02f, .02f}
                                                                                                                                    : std::array<float, 7>{1, .1f, .1f, .02f, .02f, .02f, .02f};
        std::array<uint32_t, 14> bounds{};
        std::array<simd_float4, 8> frames{};
        for (size_t page = 0; page < 2; ++page) {
            const size_t begin = page * 4, end = std::min(begin + 4, size_t(7));
            float lo = INFINITY, hi = -INFINITY;
            for (size_t n = begin; n < end; ++n) {
                lo = std::min(lo, std::log(sizes[n]));
                hi = std::max(hi, std::log(sizes[n]));
            }
            frames[page * 4 + 1] = {0, 0, -5, lo};
            frames[page * 4 + 2] = {0, 0, 0, hi - lo};
            for (size_t n = begin; n < end; ++n) {
                const uint32_t quant = hi > lo ? uint32_t(std::lround(std::clamp((std::log(sizes[n]) - lo) / (hi - lo), 0.f, 1.f) * 65535.f)) : 0;
                bounds[n * 2 + 1] = quant << 16;
            }
        }
        std::array<uint32_t, 2> chunks = {0, 1}, pages = {0, 1}, age{};
        if (scenario == 6) {
            chunks[1] = pages[1] = 0xffffffffu;
        }
        if (scenario == 7)
            age[1] = 10;
        if (scenario == 9)
            frames[1].z = frames[5].z = 5;
        LodTreeBuffers tree{upload(bounds.data(), sizeof(bounds)), upload(links.data(), sizeof(links)),
                            upload(chunks.data(), sizeof(chunks)), upload(age.data(), sizeof(age)), upload(frames.data(), sizeof(frames)), upload(pages.data(), sizeof(pages))};
        LodParameters p;
        p.node_count = 7;
        p.physical_node_count = scenario == 8 ? 5 : 7;
        p.logical_chunk_count = 2;
        p.chunk_splats = 4;
        p.output_capacity = scenario == 5 ? 1 : 7;
        p.pixel_scale_limit = scenario == 0 ? .21f : scenario == 2 || scenario == 5 || scenario == 6 || scenario == 7 || scenario == 8 ? .01f
                                                 : scenario == 3                                                                       ? .02f
                                                                                                                                       : .1f;
        p.viewport_foveation = 0;
        p.behind_camera_penalty = 1;
        p.cone_foveation = 1;
        if (scenario == 7) {
            p.fade_frames = 10;
            p.current_frame = 15;
        }
        if (scenario == 9)
            p.behind_camera_penalty = .2f;
        p.view_row0 = {1, 0, 0, 0};
        p.view_row1 = {0, 1, 0, 0};
        p.view_row2 = {0, 0, 1, 0};
        LodCutFrame cut(device, p.output_capacity, 7, 2);
        auto command = [queue commandBuffer];
        selector.encode(command, tree, p, cut);
        require(cut.busy(), "GPU LOD reservation was not retained until completion");
        bool rejected = false;
        try {
            selector.encode([queue commandBuffer], tree, p, cut);
        } catch (const std::logic_error&) { rejected = true; }
        require(rejected, "In-flight GPU cut was overwritten");
        rejected = false;
        try {
            (void)cut.status();
        } catch (const std::logic_error&) { rejected = true; }
        require(rejected, "Pending cut exposed a stale host count");
        const auto selected = cut.selection();
        auto read = [device newBufferWithLength:size_t(p.output_capacity) * 8 options:MTLResourceStorageModeShared];
        auto copy = [command blitCommandEncoder];
        [copy copyFromBuffer:selected.indices.buffer sourceOffset:0 toBuffer:read destinationOffset:0 size:size_t(p.output_capacity) * 4];
        [copy copyFromBuffer:selected.weights.buffer sourceOffset:0 toBuffer:read destinationOffset:size_t(p.output_capacity) * 4 size:size_t(p.output_capacity) * 4];
        [copy endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            throw std::runtime_error(command.error.localizedDescription.UTF8String ?: "Metal LOD command failed");
        const auto status = cut.status();
        require(status.selected <= p.output_capacity && !status.overflow, "GPU LOD budget published a truncated cut");
        const auto ids = static_cast<const uint32_t*>(read.contents);
        const auto weights = reinterpret_cast<const float*>(ids + p.output_capacity);
        std::vector<uint32_t> actual(ids, ids + status.selected);
        std::sort(actual.begin(), actual.end());
        const std::vector<uint32_t> expected = scenario == 7 ? std::vector<uint32_t>{1, 2, 3, 4, 5, 6} : scenario == 8                                                  ? std::vector<uint32_t>{2, 3, 4}
                                                                                                     : scenario == 0 || scenario == 3 || scenario == 5 || scenario == 9 ? std::vector<uint32_t>{0}
                                                                                                     : scenario == 2                                                    ? std::vector<uint32_t>{3, 4, 5, 6}
                                                                                                     : scenario == 4                                                    ? std::vector<uint32_t>{0, 1, 2}
                                                                                                                                                                        : std::vector<uint32_t>{1, 2};
        if (actual != expected) {
            std::fprintf(stderr, "Scenario %d IDs:", scenario);
            for (auto id : actual)
                std::fprintf(stderr, " %u", id);
            std::fprintf(stderr, "\n");
        }
        require(actual == expected, "LOD cut violated threshold, root reachability or missing-child residency");
        for (uint32_t n = 0; n < status.selected; ++n) {
            require(std::isfinite(weights[n]) && weights[n] >= 0 && weights[n] <= 1, "GPU LOD emitted invalid transition weights");
            if (scenario == 4) {
                const double t = (1.05 - 1) / .18, child = t;
                require(std::abs(weights[n] - (ids[n] == 0 ? 1 - child : child)) < .001, "LOD transition complement differs from its analytic curve");
            }
        }
        if (scenario == 7)
            for (uint32_t n = 0; n < status.selected; ++n)
                require(std::abs(weights[n] - (ids[n] == 3 ? 1.f : .5f)) < .001, "Page fade broke parent complement or child weight");
        if (scenario == 2) {
            // The projection consumes the GPU-produced count in the same command
            // stream; slack slots must be cleared before reading private cut IDs.
            SplatPreprocessor projection(device);
            const std::array<float, 21> xyz = {0, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0, 5};
            SplatBuffers input;
            input.count = 7;
            input.means = upload(xyz.data(), sizeof(xyz));
            std::array<float, 7> alpha{};
            alpha.fill(5);
            input.opacity_logits = upload(alpha.data(), sizeof(alpha));
            std::array<float, 21> color{};
            input.sh0 = upload(color.data(), sizeof(color));
            auto output = [device newBufferWithLength:7 * sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
            const Projection view{matrix_identity_float4x4, matrix_identity_float4x4, {0, 0, 0, 0}, {40, 40, 16, 16}, {.01f, 100, 1, .3f}, {32, 32, 0, 0}};
            for (uint32_t active : {4u, 1u, 0u, 8u}) {
                auto project = [queue commandBuffer];
                if (active == 4 || active == 1) {
                    p.pixel_scale_limit = active == 4 ? .01f : .21f;
                    selector.encode(project, tree, p, cut);
                } else {
                    // Deliberately corrupt the count as a defensive consumer
                    // contract. A truncated cut must never render a prefix.
                    *static_cast<uint32_t*>(selected.counter.buffer.contents) = active;
                }
                projection.encode(project, input, view, 0, PrimitiveMode::Points, {output}, {}, {}, {}, selected);
                [project commit];
                [project waitUntilCompleted];
                require(project.status == MTLCommandBufferStatusCompleted, "GPU-count projection pipeline failed");
                const auto projected = static_cast<const ProjectedSplat*>(output.contents);
                for (uint32_t n = 0; n < 7; ++n)
                    require((projected[n].bounds.z > projected[n].bounds.x) == (active <= 7 && n < active), "GPU cut slack retained stale geometry or consumed unwritten indices");
            }
            require(!cut.busy(), "Completed GPU LOD reservation stayed busy");
            auto invalid = p;
            invalid.pixel_scale_limit = NAN;
            bool rejected = false;
            try {
                selector.encode([queue commandBuffer], tree, invalid, cut);
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected && !cut.busy(), "Invalid LOD projection mutated a reservation");
        }
        if (scenario == 5)
            require(status.threshold_multiplier > 1, "LOD overflow did not refine its threshold GPU-side");
        if (scenario == 6)
            require(static_cast<const uint32_t*>(cut.touches().buffer.contents)[1] > 0, "Missing child page was not requested");
    }
    std::puts("Metal GPU LOD contracts passed: cuts, disconnected descendants, transitions, budget retries and missing pages.");
}
int main() {
    @autoreleasepool {
        auto device = MTLCreateSystemDefaultDevice();
        if (!device)
            return LFS_METAL_TEST_REQUIRE_DEVICE ? 1 : 77;
        try {
            run(device);
            return 0;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "%s\n", error.what());
            return 1;
        }
    }
}
