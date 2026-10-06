/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/events.hpp"
#include "scene/point_cloud_updates.hpp"
#include <chrono>
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <thread>

using namespace lfs::core;
using namespace lfs::vis;
using namespace std::chrono_literals;
namespace {
    Tensor host(size_t n = 2, DataType dtype = DataType::Float32) {
        auto t = Tensor::empty_pageable_host({n, size_t{3}}, dtype);
        if (t.bytes())
            std::memset(t.data_ptr(), 0, t.bytes());
        return t;
    }
    PointCloudUpdateInput input(size_t n = 2) { return {host(n), host(n), glm::vec3(0.0f), {}}; }
    PointCloudUpdateTarget target() {
        return {nullptr, 1, generate_uuid_v4(), std::make_shared<std::atomic<uint64_t>>(0), 0};
    }
    bool until(const std::function<bool()>& check) {
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        while (!check() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(1ms);
        return check();
    }
    auto prepare = [](PointCloudUpdateInput& in, const std::function<void()>& release) {
        auto cloud = std::make_shared<PointCloud>(host(in.points.size(0)), host(in.colors.size(0)));
        if (in.points.bytes())
            std::memcpy(cloud->means.data_ptr(), in.points.data_ptr(), in.points.bytes());
        if (in.colors.bytes())
            std::memcpy(cloud->colors.data_ptr(), in.colors.data_ptr(), in.colors.bytes());
        in = {};
        release();
        return PointCloudUpdateManager::Prepared{cloud, glm::vec3(4.0f)};
    };
} // namespace

TEST(PointCloudUpdates, SubmissionAndCancellationDoNotWaitForActiveWork) {
    std::promise<void> started, finish;
    auto released = finish.get_future().share();
    PointCloudUpdateManager manager([&](auto& in, const auto& release) {
        started.set_value();
        released.wait();
        return prepare(in, release); }, [] {});
    auto src = input();
    std::atomic<bool> owner_destroyed{false};
    src.source_owners = std::shared_ptr<void>(new int(0), [&](void* p) { delete static_cast<int*>(p); owner_destroyed = true; });
    auto ticket = manager.submit(target(), std::move(src));
    const auto began = started.get_future().wait_for(3s);
    EXPECT_EQ(began, std::future_status::ready);
    EXPECT_EQ(ticket->state(), "uploading");
    EXPECT_TRUE(ticket->cancel());
    EXPECT_EQ(ticket->state(), "cancelled");
    EXPECT_FALSE(ticket->inputsReleased());
    EXPECT_FALSE(owner_destroyed);
    // Ticket destruction cannot join the still-blocked worker.
    ticket.reset();
    finish.set_value();
    EXPECT_TRUE(until([&] { return manager.retainedRequests() == 0 && owner_destroyed.load(); }));
}

TEST(PointCloudUpdates, LatestCoalescesQueuedAndCompletedPreviews) {
    PointCloudUpdateManager manager(prepare, [] {});
    auto destination = target();
    auto first = manager.submit(destination, input());
    ASSERT_TRUE(until([&] { return manager.hasReady(); }));
    auto final = manager.submit(destination, input(3));
    EXPECT_EQ(first->state(), "superseded");
    ASSERT_TRUE(until([&] { return final->inputsReleased() && manager.hasReady(); }));
    size_t publications = 0;
    manager.publishReady([&](const auto&, const auto& result, auto&) {
        ++publications;
        EXPECT_EQ(result.cloud->size(), 3);
        EXPECT_EQ(result.cloud->colors.size(0), 3);
    });
    EXPECT_EQ(publications, 1);
    EXPECT_EQ(final->state(), "published");
    EXPECT_FALSE(final->cancel());
    EXPECT_TRUE(until([&] { return manager.retainedRequests() == 0; }));
}

TEST(PointCloudUpdates, CancelledReadyResultNeverPublishes) {
    PointCloudUpdateManager manager(prepare, [] {});
    auto ticket = manager.submit(target(), input());
    ASSERT_TRUE(until([&] { return manager.hasReady(); }));
    ASSERT_TRUE(ticket->cancel());
    manager.publishReady([](const auto&, const auto&, auto&) { ADD_FAILURE() << "Cancelled result published"; });
    EXPECT_TRUE(until([&] { return manager.retainedRequests() == 0; }));
}

TEST(PointCloudUpdates, FailurePreservesVisibleDataAndReleasesInputs) {
    PointCloudUpdateManager manager([](auto&, const auto&) -> PointCloudUpdateManager::Prepared { throw std::bad_alloc(); }, [] {});
    auto ticket = manager.submit(target(), input());
    ASSERT_TRUE(until([&] { return ticket->state() == "failed" && ticket->inputsReleased(); }));
    EXPECT_FALSE(ticket->error().empty());
    manager.publishReady([](const auto&, const auto&, auto&) { ADD_FAILURE() << "Failed upload published"; });
}

TEST(PointCloudUpdates, PublicationRejectsStaleSceneAndTarget) {
    PointCloudUpdateManager manager(prepare, [] {});
    auto destination = target();
    auto ticket = manager.submit(destination, input());
    ASSERT_TRUE(until([&] { return manager.hasReady(); }));
    destination.revision->fetch_add(1);
    manager.publishReady([](const auto& destination, const auto&, auto&) {
        if (destination.revision->load() != destination.expected_revision)
            throw std::runtime_error("target replaced");
    });
    EXPECT_EQ(ticket->state(), "failed");
    EXPECT_EQ(ticket->error(), "target replaced");
}

TEST(PointCloudUpdates, MetadataValidationAndEmptyCloud) {
    PointCloudUpdateManager manager(prepare, [] {});
    auto wrong_dtype = input();
    wrong_dtype.points = host(2, DataType::UInt8);
    EXPECT_THROW(manager.submit(target(), wrong_dtype), std::invalid_argument);
    auto wrong_count = input();
    wrong_count.colors = host(3);
    EXPECT_THROW(manager.submit(target(), wrong_count), std::invalid_argument);
    auto wrong_shape = input();
    wrong_shape.points = Tensor::empty_pageable_host({size_t{2}, size_t{4}});
    EXPECT_THROW(manager.submit(target(), wrong_shape), std::invalid_argument);
    auto empty = manager.submit(target(), input(0));
    ASSERT_TRUE(until([&] { return manager.hasReady(); }));
    manager.publishReady([](const auto&, const auto& result, auto&) { EXPECT_EQ(result.cloud->size(), 0); });
    EXPECT_EQ(empty->state(), "published");
}

TEST(PointCloudUpdates, BoundedQueueRetainsCancelledInputsUntilWorkerRetirement) {
    std::promise<void> started, finish;
    auto gate = finish.get_future().share();
    PointCloudUpdateManager manager([&](auto& in, const auto& release) {
        started.set_value();
        gate.wait();
        return prepare(in, release); }, [] {});
    auto destination = target();
    auto first = manager.submit(destination, input());
    EXPECT_EQ(started.get_future().wait_for(3s), std::future_status::ready);
    for (size_t i = 1; i < PointCloudUpdateManager::kMaxRequests; ++i)
        manager.submit(destination, input());
    EXPECT_EQ(first->state(), "superseded");
    EXPECT_FALSE(first->inputsReleased());
    EXPECT_EQ(manager.retainedRequests(), PointCloudUpdateManager::kMaxRequests);
    EXPECT_LE(manager.retainedBytes(), PointCloudUpdateManager::kMaxBytes);
    EXPECT_THROW(manager.submit(destination, input()), std::runtime_error);
    manager.cancelAll();
    finish.set_value();
    EXPECT_TRUE(until([&] { return manager.retainedRequests() == 0; }));
}

TEST(PointCloudUpdates, ScenePublicationUpdatesTrainingReferenceCountAndCentroidTogether) {
    Scene scene;
    auto before = std::make_shared<PointCloud>(host(2), host(2));
    const auto id = scene.addPointCloud("cloud", before);
    auto* node = scene.getNodeById(id);
    scene.setInitialPointCloud(before);
    auto after = std::make_shared<PointCloud>(host(3), host(3));
    auto retired = scene.publishNodePointCloud(node->uuid, after, glm::vec3(4.0f));
    EXPECT_EQ(retired.cloud, before);
    EXPECT_EQ(node->point_cloud, after);
    EXPECT_EQ(scene.getInitialPointCloud(), after);
    EXPECT_EQ(node->gaussian_count.load(), 3);
    EXPECT_EQ(node->centroid, glm::vec3(4.0f));
    EXPECT_TRUE(scene.isPointCloudModified());
    const auto uuid = node->uuid;
    scene.removeNode("cloud");
    EXPECT_THROW(scene.publishNodePointCloud(uuid, after, {}), std::runtime_error);
}

TEST(PointCloudUpdates, SceneResolutionDoesNotStartAnUploadOrRetireSourcesOnViewer) {
    std::atomic<bool> prepared{false};
    PointCloudUpdateManager manager([&](auto& in, const auto& release) {
        prepared = true;
        return prepare(in, release); }, [] {}, true);
    auto destination = target();
    auto ticket = manager.submit(destination, input());
    EXPECT_EQ(ticket->state(), "queued");
    EXPECT_FALSE(prepared);
    manager.resolveQueued([](auto& destination, auto& in) {
        destination.render_generation = 7;
        EXPECT_EQ(in.points.size(0), 2);
    });
    ASSERT_TRUE(until([&] { return ticket->inputsReleased(); }));
    EXPECT_TRUE(prepared);
    manager.publishReady([](const auto& destination, const auto&, auto&) { EXPECT_EQ(destination.render_generation, 7); });
    EXPECT_EQ(ticket->state(), "published");
}

TEST(PointCloudUpdates, ClearInvalidatesEpochButRenamePreservesIt) {
    Scene scene;
    const auto epoch = scene.pointCloudUpdateEpoch();
    const auto version = epoch->load();
    auto before = std::make_shared<PointCloud>(host(2), host(2));
    const auto id = scene.addPointCloud("cloud", before);
    const auto uuid = scene.getNodeById(id)->uuid;
    EXPECT_TRUE(scene.renameNode("cloud", "renamed"));
    EXPECT_EQ(epoch->load(), version);
    auto after = std::make_shared<PointCloud>(host(3), host(3));
    auto retired = scene.publishNodePointCloud(uuid, after, glm::vec3(4.0f));
    EXPECT_EQ(retired.cloud, before);
    EXPECT_EQ(scene.getNodeByUuid(uuid)->point_cloud, after);
    EXPECT_EQ(epoch->load(), version);
    scene.clear();
    EXPECT_NE(epoch->load(), version);
}

#if LFS_GRAPHICS_VULKAN
#include "rendering/point_cloud_vulkan_renderer.hpp"
#include "window/vulkan_graphics_context.hpp"
#include <cstdlib>
#include <iostream>

TEST(PointCloudUpdatesGpu, IndependentSnapshotAndRendererLeasesAvoidHostRoundtrip) {
    if (!std::getenv("LFS_ASYNC_GPU_TESTS"))
        GTEST_SKIP() << "Set LFS_ASYNC_GPU_TESTS=1 on a GPU host";
    VulkanGraphicsContext graphics;
    ASSERT_TRUE(graphics.initializeHeadless());
    graphics.connectTensorBackend();
    auto native_prepare = preparePointCloudUpdate(graphics.splatTensorAllocator(), [&](PointCloud& cloud, TensorCompletion ready) { graphics.preparePointCloudStorage(cloud, std::move(ready)); }, graphics.pointCloudUploadDevice());
    PointCloudUpdateManager manager(native_prepare, [] {});
    std::shared_ptr<PointCloud> current;
    auto destination = target();
    const auto publish = [&](const auto&, const auto& prepared, auto& retired) {
        retired.cloud = std::exchange(current, prepared.cloud);
    };
    auto source = input(1024);
    source.centroid.reset();
    auto ticket = manager.submit(destination, source);
    ASSERT_TRUE(until([&] { return ticket->inputsReleased() && manager.hasReady(); }));
    source.points.ptr<float>()[0] = 99.0f;
    manager.publishReady(publish);
    ASSERT_EQ(ticket->state(), "published");
    ASSERT_TRUE(current->render_buffers);
    EXPECT_EQ(current->means.cpu().ptr<float>()[0], 0.0f);
    EXPECT_EQ(current->colors.dtype(), DataType::Float32);
    EXPECT_NE(current->render_buffers->ready.semaphore, nullptr);
    PointCloudVulkanRenderer renderer;
    PointCloudVulkanRenderer::RenderRequest request;
    request.positions = &current->means;
    request.colors = &current->colors;
    request.prepared_buffers = current->render_buffers;
    request.size = {64, 64};
    auto frame = renderer.render(graphics.vulkanContext(), request, RenderTargetId{1});
    ASSERT_TRUE(frame) << frame.error();
    EXPECT_EQ(PointCloudOutputOwnershipTestAccess::hostVertexUploadBytes(renderer), 0);
    std::weak_ptr<const PointCloudRenderBuffers> previous = current->render_buffers;
    request.prepared_buffers.reset();
    current.reset();
    ticket.reset();
    ASSERT_TRUE(until([&] { return manager.retainedRequests() == 0; }));
    EXPECT_FALSE(previous.expired()) << "Renderer must retain storage through its previous frame fence";
    auto next = manager.submit(destination, input(2048));
    ASSERT_TRUE(until([&] { return manager.hasReady(); }));
    manager.publishReady(publish);
    request.positions = &current->means;
    request.colors = &current->colors;
    request.prepared_buffers = current->render_buffers;
    frame = renderer.render(graphics.vulkanContext(), request, RenderTargetId{1});
    ASSERT_TRUE(frame) << frame.error();
    EXPECT_EQ(PointCloudOutputOwnershipTestAccess::hostVertexUploadBytes(renderer), 0);
    EXPECT_TRUE(previous.expired()) << "Old leases may retire after the previous frame fence";
    renderer.reset();
    manager.shutdown();
}

TEST(PointCloudUpdatesGpu, MillionPointUploadMeasurements) {
    if (!std::getenv("LFS_ASYNC_GPU_BENCH"))
        GTEST_SKIP() << "Set LFS_ASYNC_GPU_BENCH=1 for one-million and five-million point preparation measurements";
    VulkanGraphicsContext graphics;
    ASSERT_TRUE(graphics.initializeHeadless());
    graphics.connectTensorBackend();
    auto native_prepare = preparePointCloudUpdate(graphics.splatTensorAllocator(), [&](PointCloud& cloud, TensorCompletion ready) { graphics.preparePointCloudStorage(cloud, std::move(ready)); }, graphics.pointCloudUploadDevice());
    PointCloudUpdateManager manager(native_prepare, [] {});
    PointCloudVulkanRenderer renderer;
    auto destination = target();
    std::shared_ptr<PointCloud> current;
    for (const size_t count : {size_t{1'000'000}, size_t{5'000'000}}) {
        auto data = input(count);
        const auto begin = std::chrono::steady_clock::now();
        auto ticket = manager.submit(destination, std::move(data));
        const auto submitted = std::chrono::steady_clock::now();
        const auto deadline = begin + 60s;
        while (!manager.hasReady() && ticket->state() != "failed" && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(1ms);
        ASSERT_NE(ticket->state(), "failed") << ticket->error();
        ASSERT_TRUE(manager.hasReady());
        const auto ready = std::chrono::steady_clock::now();
        manager.publishReady([&](const auto&, const auto& result, auto& retired) {
            retired.cloud = std::exchange(current, result.cloud);
        });
        PointCloudVulkanRenderer::RenderRequest request;
        request.positions = &current->means;
        request.colors = &current->colors;
        request.prepared_buffers = current->render_buffers;
        request.size = {320, 240};
        const auto published = std::chrono::steady_clock::now();
        auto frame = renderer.render(graphics.vulkanContext(), request, RenderTargetId{1});
        ASSERT_TRUE(frame) << frame.error();
        const auto first_frame = std::chrono::steady_clock::now();
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::cout << "ASYNC_POINT_CLOUD count=" << count << " submit_ms=" << ms(begin, submitted)
                  << " preparation_ms=" << ms(submitted, ready) << " publication_ms=" << ms(ready, published)
                  << " first_render_submit_ms=" << ms(published, first_frame)
                  << " host_vertex_upload_bytes=" << PointCloudOutputOwnershipTestAccess::hostVertexUploadBytes(renderer) << '\n';
        EXPECT_EQ(PointCloudOutputOwnershipTestAccess::hostVertexUploadBytes(renderer), 0);
        ASSERT_TRUE(until([&] { return manager.retainedRequests() == 0; }));
    }
    renderer.reset();
    manager.shutdown();
}

TEST(PointCloudUpdatesGpu, MergedViewPreservesNodeOrderAndEmptyReplacement) {
    if (!std::getenv("LFS_ASYNC_GPU_TESTS"))
        GTEST_SKIP() << "Set LFS_ASYNC_GPU_TESTS=1 on a GPU host";
    VulkanGraphicsContext graphics;
    ASSERT_TRUE(graphics.initializeHeadless());
    graphics.connectTensorBackend();
    auto native_prepare = preparePointCloudUpdate(graphics.splatTensorAllocator(), [&](PointCloud& cloud, TensorCompletion ready) { graphics.preparePointCloudStorage(cloud, std::move(ready)); }, graphics.pointCloudUploadDevice());
    auto before = std::make_shared<PointCloud>(host(1), host(1));
    auto after = std::make_shared<PointCloud>(host(1), host(1));
    before->means.ptr<float>()[0] = 1.0f;
    after->means.ptr<float>()[0] = 3.0f;
    auto in = input(1);
    in.points.ptr<float>()[0] = 2.0f;
    in.companions = {{before, glm::mat4(1.0f)}, {after, glm::mat4(1.0f)}};
    in.target_index = 1;
    auto result = native_prepare(in, [] {});
    ASSERT_TRUE(result.merged);
    ASSERT_TRUE(result.merged->render_buffers);
    auto points = result.merged->means.cpu();
    ASSERT_EQ(points.size(0), 3);
    EXPECT_FLOAT_EQ(points.ptr<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(points.ptr<float>()[3], 2.0f);
    EXPECT_FLOAT_EQ(points.ptr<float>()[6], 3.0f);
    in = input(0);
    in.companions = {{before, glm::mat4(1.0f)}, {after, glm::mat4(1.0f)}};
    in.target_index = 1;
    result = native_prepare(in, [] {});
    EXPECT_EQ(result.cloud->size(), 0);
    EXPECT_EQ(result.centroid, glm::vec3(0.0f));
    ASSERT_TRUE(result.merged);
    ASSERT_TRUE(result.merged->render_buffers);
    points = result.merged->means.cpu();
    ASSERT_EQ(points.size(0), 2);
    EXPECT_FLOAT_EQ(points.ptr<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(points.ptr<float>()[3], 3.0f);
}
#endif
