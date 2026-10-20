/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "device_requirements.hpp"
#include "core/rad_pool_quant.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_metal_reader.hpp"
#include "io/formats/rad.hpp"
#include "io/ply_to_rad_lod.hpp"
#include "metal_rad_pager.hpp"
#include "metal_viewport_renderer.hpp"
#include "window/vulkan_context.hpp"
#include <Python.h>
#include <bit>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <sys/mman.h>
#include <unistd.h>
using namespace lfs;
static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
static void run() {
    const core::GpuBackendScope scope(core::GpuBackend::Metal);
    std::string pattern = (std::filesystem::temp_directory_path() / "lfs-metal-rad-XXXXXX").string();
    require(mkdtemp(pattern.data()), "Cannot create RAD fixture directory");
    const std::filesystem::path directory(pattern), ply = directory / "scene.ply", rad = directory / "scene.rad";
    {
        std::ofstream out(ply, std::ios::binary);
        out << "ply\nformat binary_little_endian 1.0\nelement vertex 4099\n";
        for (const char* name : {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"})
            out << "property float " << name << "\n";
        out << "end_header\n";
        for (size_t n = 0; n < 4099; ++n) {
            const float values[]{float(int(n % 65) - 32) * .018f, float(int(n / 65) - 31) * .018f, -3.f - .001f * float(n % 7),
                                 1.f, -.5f, -.5f, 2.f, -5.f, -5.f, -5.f, 1.f, 0.f, 0.f, 0.f};
            out.write(reinterpret_cast<const char*>(values), sizeof(values));
        }
        require(out.good(), "Cannot write RAD input fixture");
    }
    io::PlyToRadLodOptions options;
    options.temp_dir = directory / "scratch";
    options.target_bucket_splats = 2048;
    options.chunk_size = 2048;
    options.builder = io::LodBuilder::kOctree;
    options.octree_bhatt_top_nodes = 32;
    const auto converted = io::convert_ply_to_rad_lod(ply, rad, options);
    require(bool(converted), "RAD fixture conversion failed");
    const auto loaded = io::load_rad(rad, {.out_of_core = true, .preview_splats = 2048});
    if (!loaded)
        throw std::runtime_error(loaded.error());
    const auto& model = *loaded;
    std::fprintf(stderr, "RAD fixture=%s resident=%zu nodes=%zu view=%d source=%d chunk=%u\n", directory.c_str(), size_t(model.size()), model.lod_tree ? model.lod_tree->total_nodes() : 0,
                 model.lod_tree ? model.lod_tree->meta_view.valid() : false, model.lod_tree ? model.lod_tree->rad_source.valid() : false,
                 model.lod_tree ? model.lod_tree->rad_source.chunk_size : 0);
    require(model.lod_tree && model.lod_tree->meta_view.valid() && model.size() < model.lod_tree->total_nodes(), "RAD fixture is not disk-paged");
    vis::VulkanContext context;
    require(context.initHeadless(), context.lastError().c_str());
    core::MetalTensorReader reader;
    vis::MetalRadPager pager(reader.device());
    pager.configure(model, context.device(), nullptr, {.pool_splats = 4096, .fade_frames = 0});
    // The shared pool policy keeps out-of-core pools at >= 2048 pages: this
    // small fixture pages every chunk whatever the splat budget.
    require(pager.pool().page_splats == 2048 && pager.physicalNodes() == pager.cache().snapshot().logical_chunks * 2048, "Native page-pool sizing differs");
    require(pager.cache().snapshot().resident_chunks == 0, "RAD pages published before any GPU upload");
    const auto await = [&](uint32_t chunk, std::span<const uint32_t> touches) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
        do {
            pager.advance(touches);
            if (pager.cache().snapshot().chunk_to_page[chunk] != 0xffffffffu)
                return;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (std::chrono::steady_clock::now() < deadline);
        throw std::runtime_error("RAD page publication did not converge");
    };
    await(0, {});
    std::vector<uint32_t> interest(pager.cache().snapshot().logical_chunks);
    require(interest.size() >= 3, "RAD fixture needs at least three native pages");
    interest[0] = 0xffffffffu;
    interest[1] = std::bit_cast<uint32_t>(2.f);
    await(1, interest);
    interest[1] = 0;
    interest[2] = std::bit_cast<uint32_t>(100.f);
    await(2, interest);
    const auto& mapping = pager.cache().snapshot().chunk_to_page;
    const uint32_t replaced_page = mapping[2];
    require(mapping[0] != 0xffffffffu, "Pinned root was evicted");
    require(mapping[1] != 0xffffffffu && mapping[1] != replaced_page && mapping[0] != replaced_page,
            "Native pages of distinct chunks alias");
    // Consume metadata produced by the actual packed decoder, on the native
    // tensor reader, before resetting an active producer/cache generation.
    const auto& metadata = pager.pool().regions[8];
    auto snapshot = [reader.device() newBufferWithLength:metadata.bytes() options:MTLResourceStorageModeShared];
    const std::array<const core::Tensor*, 1> tensors{&metadata};
    auto command = reader.submit(tensors, [&](id<MTLCommandBuffer> command, std::span<const core::MetalTensorView> views) {
        auto blit = [command blitCommandEncoder];
        [blit copyFromBuffer:views[0].buffer sourceOffset:views[0].offset toBuffer:snapshot destinationOffset:0 size:views[0].bytes];
        [blit endEncoding];
    });
    interest[2] = 0;
    interest.back() = std::bit_cast<uint32_t>(1000.f);
    pager.advance(interest);
    const auto signature = pager.signature();
    // A budget refresh keeps an unchanged pool; a new model replaces the generation.
    pager.configure(model, context.device(), nullptr, {.pool_splats = 6144, .fade_frames = 0});
    require(pager.signature() == signature, "A budget refresh with an unchanged page count reset the RAD cache");
    const auto replacement = io::load_rad(rad, {.out_of_core = true, .preview_splats = 2048});
    require(bool(replacement), "RAD replacement load failed");
    pager.configure(*replacement, context.device(), nullptr, {.pool_splats = 6144, .fade_frames = 0});
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted, "RAD page consumer failed during generation replacement");
    const auto links = static_cast<const core::RadMetaLinksQ*>(snapshot.contents);
    const auto& expected = model.lod_tree->meta_view.links[2 * 2048];
    require(links[replaced_page * 2048].child_start == expected.child_start &&
                links[replaced_page * 2048].parent == expected.parent,
            "Packed page metadata differs from logical sidecar records");
    require(pager.signature() != signature && pager.cache().snapshot().resident_chunks == 0, "RAD reset retained the previous generation");
    // Exercise the real desktop native adapter, including initial CPU preview,
    // GPU traversal, texture import, and deferred paging diagnostics.
    vis::MetalViewportRenderer renderer;
    renderer.setLodSettings(6144, .15f, 0);
    rendering::ViewportRenderRequest request;
    request.frame_view.size = {96, 64};
    request.sh_degree = 0;
    request.lod_gpu_traversal.enabled = true;
    request.lod_gpu_traversal.node_count = model.lod_tree->total_nodes();
    request.lod_gpu_traversal.output_capacity = 1024;
    request.lod_gpu_traversal.pixel_scale_limit = .01f;
    request.lod_gpu_traversal.object_to_view = glm::mat4(1);
    require(vis::MetalViewportRenderer::supports(model, request), "Native RAD adapter rejected a streamable CPU preview");
    {
        // A host cut drawn from the paged preview reads exactly its own
        // entries: end it at a PROT_NONE page so any over-read faults.
        const size_t page = size_t(getpagesize()), cut_count = 16;
        require(cut_count < size_t(model.size()), "RAD preview must exceed the host cut");
        auto* block = static_cast<std::byte*>(mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
        require(block != MAP_FAILED && mprotect(block + page, page, PROT_NONE) == 0, "Cannot guard the host cut");
        auto* cut = reinterpret_cast<uint32_t*>(block + page) - cut_count;
        for (uint32_t n = 0; n < cut_count; ++n)
            cut[n] = n;
        auto host_request = request;
        host_request.lod_gpu_traversal.enabled = false;
        host_request.lod_indices = cut;
        host_request.lod_count = cut_count;
        require(vis::MetalViewportRenderer::supports(model, host_request), "Native RAD adapter rejected a host cut of the paged preview");
        const auto output = renderer.render(context, model, host_request, vis::RenderTargetId{5});
        if (!output)
            throw std::runtime_error(format_for_developer(output.error()));
        auto pixels = core::Tensor::empty({64, 96, 4}, core::Device::CPU);
        require(bool(renderer.readColor(vis::RenderTargetId{5}, pixels, 0, 0)), "Host-cut RAD readback failed");
        require(bool(renderer.release(vis::RenderTargetId{5})), "Host-cut RAD release failed");
        munmap(block, 2 * page);
    }
    bool native_cut = false;
    bool settled = false;
    for (int frame = 0; frame < 80; ++frame) {
        const auto output = renderer.render(context, model, request, vis::RenderTargetId{1});
        if (!output)
            throw std::runtime_error(format_for_developer(output.error()));
        auto pixels = core::Tensor::empty({64, 96, 4}, core::Device::CPU);
        const auto read = renderer.readColor(vis::RenderTargetId{1}, pixels, 0, 0);
        require(bool(read), "Native RAD color readback failed");
        float max_color = 0;
        for (size_t p = 0; p < 64 * 96; ++p)
            max_color = std::max(max_color, pixels.ptr<float>()[p * 4]);
        require(max_color > .1f, "RAD streaming lost all visible coverage");
        const auto status = renderer.gpuLodSelectionStatus(vis::RenderTargetId{1});
        if (status.active) {
            // Below the 2048-page floor the pool pages every chunk.
            require(status.pool_pages == status.chunk_count && status.chunk_count >= 3 && status.selected > 0 && status.overflow == 0,
                    "Native RAD cut/diagnostics are not backed by the actual pool");
            native_cut = true;
            settled |= !output->lod_streaming_active;
        }
    }
    require(native_cut, "Native RAD never switched from preview to GPU selection");
    require(settled, "A stable native RAD cut keeps requesting redraws after paging settles");
    require(bool(renderer.release(vis::RenderTargetId{1})), "Native RAD release failed");
    request.gut = true;
    require(vis::MetalViewportRenderer::supports(model, request), "Native Spark GUT RAD frame fell back");
    const auto captured = renderer.render(context, model, request, vis::RenderTargetId{4}, false, true);
    require(bool(captured), "Native RAD capture did not wait for its pinned root");
    const auto complete = renderer.outputComplete(vis::RenderTargetId{4});
    require(bool(complete) && *complete, "Native RAD capture published the provisional CPU preview");
    auto gut_pixels = core::Tensor::empty({64, 96, 4}, core::Device::CPU);
    require(bool(renderer.readColor(vis::RenderTargetId{4}, gut_pixels, 0, 0)), "Native Spark GUT RAD readback failed");
    float gut_red = 0;
    for (size_t pixel = 0; pixel < 64 * 96; ++pixel)
        gut_red = std::max(gut_red, gut_pixels.ptr<float>()[pixel * 4]);
    require(gut_red > .1f, "Native Spark GUT RAD has no visible coverage");
    require(bool(renderer.release(vis::RenderTargetId{4})), "Native RAD capture release failed");
    // Legacy file blocks contain many native pages. Sidecar metadata and
    // physical pools must still use 2048-node pages, including a partial tail.
    const auto legacy = directory / "legacy.rad";
    options.chunk_size = 65536;
    require(bool(io::convert_ply_to_rad_lod(ply, legacy, options)), "Legacy RAD fixture conversion failed");
    auto legacy_model = io::load_rad(legacy);
    require(bool(legacy_model), "Legacy RAD load failed");
    require(legacy_model->lod_tree->rad_source.chunk_size == 65536, "Legacy RAD fixture block size differs");
    pager.configure(*legacy_model, context.device(), nullptr, {.pool_splats = 4096, .fade_frames = 0});
    {
        // A fully loaded RAD fills its pool from the resident splats at
        // configure: every page is published, with the sidecar's metadata.
        const auto& filled = pager.cache().snapshot();
        const auto opened_meta = io::open_rad_meta_sidecar(legacy);
        require(bool(opened_meta), "Legacy RAD sidecar unavailable");
        const auto& legacy_meta = *opened_meta;
        require(size_t(legacy_model->size()) == legacy_meta.node_count && filled.resident_chunks == filled.logical_chunks && !pager.pending(),
                "A fully loaded RAD streamed its pool instead of filling it from resident splats");
        const std::array<const core::Tensor*, 3> regions{&pager.pool().regions[6], &pager.pool().regions[7], &pager.pool().regions[8]};
        std::array<id<MTLBuffer>, 3> copies{};
        for (size_t n = 0; n < regions.size(); ++n)
            copies[n] = [reader.device() newBufferWithLength:regions[n]->bytes() options:MTLResourceStorageModeShared];
        auto readback = reader.submit(regions, [&](id<MTLCommandBuffer> command, std::span<const core::MetalTensorView> views) {
            auto blit = [command blitCommandEncoder];
            for (size_t n = 0; n < views.size(); ++n)
                [blit copyFromBuffer:views[n].buffer sourceOffset:views[n].offset toBuffer:copies[n] destinationOffset:0 size:views[n].bytes];
            [blit endEncoding];
        });
        [readback waitUntilCompleted];
        const auto* frames = static_cast<const core::radq::PageFrame*>(copies[0].contents);
        const auto* bounds = static_cast<const core::RadMetaBoundsQ*>(copies[1].contents);
        const auto* links = static_cast<const core::RadMetaLinksQ*>(copies[2].contents);
        for (size_t chunk = 0; chunk < filled.logical_chunks; ++chunk) {
            const uint32_t page = filled.chunk_to_page[chunk];
            const auto& record = legacy_meta.chunks[chunk];
            require(frames[page].bbox_min[0] == record.bbox_min[0] && frames[page].bbox_extent[2] == record.bbox_extent[2] &&
                        frames[page].log_size_range == record.log_size_range,
                    "Resident RAD page frame differs from the sidecar chunk record");
            for (size_t node = chunk * 2048; node < std::min<size_t>((chunk + 1) * 2048, legacy_meta.node_count); ++node) {
                const size_t slot = size_t(page) * 2048 + node - chunk * 2048;
                require(bounds[slot].qx == legacy_meta.bounds[node].qx && bounds[slot].qsize == legacy_meta.bounds[node].qsize &&
                            links[slot].child_start == legacy_meta.links[node].child_start && links[slot].parent == legacy_meta.links[node].parent,
                        "Resident RAD node metadata differs from the sidecar");
            }
        }
    }
    await(0, {});
    interest.assign(pager.cache().snapshot().logical_chunks, 0);
    interest[0] = 0xffffffffu;
    interest.back() = std::bit_cast<uint32_t>(4.f);
    await(uint32_t(interest.size() - 1), interest);
    require(pager.pool().page_splats == 2048, "Legacy RAD file block leaked into the GPU page stride");
    std::printf("RAD paging, generation replacement, preview host cut, native desktop cut and resident fill passed; fixture=%s\n", directory.c_str());
}
int main() {
    @autoreleasepool {
        if (!core::gpu_backend_available(core::GpuBackend::Metal)) {
            return lfs::metal_test::unavailableMetal4();
        }
        const auto home = std::filesystem::temp_directory_path() / ("lfs-metal-rad-home-" + std::to_string(getpid()));
        setenv("LFS_HOME", home.c_str(), 1);
        unsetenv("LFS_SAFE_MODE");
        Py_Initialize();
        try {
            run();
            return 0;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "%s\n", error.what());
            return 1;
        }
    }
}
