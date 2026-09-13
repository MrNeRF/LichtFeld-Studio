/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "rendering/selection_ops.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::GpuBackendScope;
    using lfs::core::Tensor;
    using lfs::rendering::ScreenWindowCameraModel;

    constexpr std::size_t kLockedWords = 8;
    constexpr std::size_t kScratchWords = 257;
    constexpr std::size_t kN = 5003;

    Tensor uploadU8(const std::vector<uint8_t>& values, const GpuBackend backend) {
        Tensor host = Tensor::empty({values.size()}, Device::CPU, DataType::UInt8);
        if (!values.empty()) {
            std::memcpy(host.ptr<uint8_t>(), values.data(), values.size());
        }
        GpuBackendScope scope(backend);
        return host.to(Device::GPU);
    }

    Tensor uploadBool(const std::vector<uint8_t>& values, const GpuBackend backend) {
        std::vector<bool> bits(values.begin(), values.end());
        GpuBackendScope scope(backend);
        return Tensor::from_vector(bits, {bits.size()}, Device::CPU).to(Device::GPU);
    }

    Tensor uploadI32(const std::vector<int>& values, const GpuBackend backend) {
        GpuBackendScope scope(backend);
        return Tensor::from_vector(values, {values.size()}, Device::CPU).to(Device::GPU);
    }

    Tensor uploadF32(const std::vector<float>& values, const lfs::core::TensorShape& shape,
                     const GpuBackend backend) {
        GpuBackendScope scope(backend);
        return Tensor::from_vector(values, shape, Device::CPU).to(Device::GPU);
    }

    Tensor emptyU8(const std::size_t n, const GpuBackend backend) {
        GpuBackendScope scope(backend);
        return Tensor::empty({n}, Device::GPU, DataType::UInt8);
    }

    Tensor emptyScratch(const GpuBackend backend) {
        GpuBackendScope scope(backend);
        return Tensor::zeros({kScratchWords}, Device::GPU, DataType::Int32);
    }

    std::array<uint32_t, kLockedWords> lockedHost(const std::vector<uint8_t>& groups) {
        std::array<uint32_t, kLockedWords> words{};
        for (const auto group : groups) {
            words[group / 32] |= (1u << (group % 32));
        }
        return words;
    }

    uint32_t* lockedDevice(const std::array<uint32_t, kLockedWords>& host) {
        uint32_t* ptr = nullptr;
        if (cudaMalloc(&ptr, sizeof(host)) != cudaSuccess) {
            return nullptr;
        }
        if (cudaMemcpy(ptr, host.data(), sizeof(host), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(ptr);
            return nullptr;
        }
        return ptr;
    }

    std::vector<uint8_t> toU8(const Tensor& tensor) {
        return tensor.cpu().to_vector_uint8();
    }

    std::vector<bool> toBool(const Tensor& tensor) {
        return tensor.cpu().to_vector_bool();
    }

    std::vector<int> toI32(const Tensor& tensor) {
        return tensor.cpu().to_vector_int();
    }

    void expectU8Equal(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, const char* tag) {
        ASSERT_EQ(a.size(), b.size()) << tag;
        for (std::size_t i = 0; i < a.size(); ++i) {
            EXPECT_EQ(a[i], b[i]) << tag << " index=" << i;
        }
    }

    void expectBoolEqual(const std::vector<bool>& a, const std::vector<bool>& b, const char* tag) {
        ASSERT_EQ(a.size(), b.size()) << tag;
        for (std::size_t i = 0; i < a.size(); ++i) {
            EXPECT_EQ(a[i], b[i]) << tag << " index=" << i;
        }
    }

    void expectI32Equal(const std::vector<int>& a, const std::vector<int>& b, const char* tag) {
        ASSERT_EQ(a.size(), b.size()) << tag;
        for (std::size_t i = 0; i < a.size(); ++i) {
            EXPECT_EQ(a[i], b[i]) << tag << " index=" << i;
        }
    }

    struct RandomMasks {
        std::vector<uint8_t> selected;
        std::vector<uint8_t> existing;
        std::vector<int> nodes;
        std::vector<bool> valid_nodes;
        std::vector<uint8_t> locked_groups;
    };

    RandomMasks makeRandomMasks(const std::size_t n, const std::uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> bit(0, 1);
        std::uniform_int_distribution<int> group(0, 7);
        std::uniform_int_distribution<int> node(0, 4);
        RandomMasks out;
        out.selected.resize(n);
        out.existing.resize(n);
        out.nodes.resize(n);
        out.valid_nodes = {true, false, true, true, false};
        out.locked_groups = {2, 5};
        for (std::size_t i = 0; i < n; ++i) {
            out.selected[i] = static_cast<uint8_t>(bit(rng));
            out.existing[i] = static_cast<uint8_t>(group(rng));
            out.nodes[i] = node(rng);
            if ((i % 17) == 0) {
                out.nodes[i] = -1;
            }
            if ((i % 29) == 0) {
                out.nodes[i] = 99;
            }
        }
        return out;
    }
} // namespace

class SelectionMaskOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!lfs::core::gpu_backend_available(GpuBackend::CUDA) ||
            !lfs::core::gpu_backend_available(GpuBackend::Vulkan)) {
            GTEST_SKIP() << "CUDA and Vulkan backends are required";
        }
    }
};

TEST_F(SelectionMaskOpsTest, ApplyGroupMatchesKernelOnRandomInputs) {
    const auto data = makeRandomMasks(kN, 20260906);
    const auto host_locked = lockedHost(data.locked_groups);
    uint32_t* d_locked = lockedDevice(host_locked);
    ASSERT_NE(d_locked, nullptr);
    const std::vector<bool> empty_nodes;
    const struct Case {
        bool add;
        bool replace;
        bool nodes;
        const char* name;
    } cases[] = {
        {true, false, true, "add_nodes"},
        {false, false, true, "remove_nodes"},
        {true, true, true, "replace_nodes"},
        {true, false, false, "add_all"},
        {true, true, false, "replace_all"},
        {false, false, false, "remove_all"},
    };
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        const std::vector<int>* nodes = test_case.nodes ? &data.nodes : nullptr;
        const std::vector<bool>& valid = test_case.nodes ? data.valid_nodes : empty_nodes;
        Tensor cuda_sel = uploadBool(data.selected, GpuBackend::CUDA);
        Tensor cuda_exist = uploadU8(data.existing, GpuBackend::CUDA);
        Tensor cuda_out = emptyU8(kN, GpuBackend::CUDA);
        Tensor cuda_scratch = emptyScratch(GpuBackend::CUDA);
        Tensor cuda_nodes;
        const Tensor* cuda_nodes_ptr = nullptr;
        if (nodes) {
            cuda_nodes = uploadI32(*nodes, GpuBackend::CUDA);
            cuda_nodes_ptr = &cuda_nodes;
        }
        lfs::rendering::apply_selection_group_tensor_mask(
            cuda_sel, cuda_exist, cuda_out, 3, d_locked, test_case.add, cuda_nodes_ptr, valid,
            test_case.replace, &cuda_scratch);
        const auto cuda_words = toU8(cuda_out);
        const auto cuda_scratch_words = toI32(cuda_scratch);

        Tensor vk_sel = uploadBool(data.selected, GpuBackend::Vulkan);
        Tensor vk_exist = uploadU8(data.existing, GpuBackend::Vulkan);
        Tensor vk_out = emptyU8(kN, GpuBackend::Vulkan);
        Tensor vk_scratch = emptyScratch(GpuBackend::Vulkan);
        Tensor vk_nodes;
        const Tensor* vk_nodes_ptr = nullptr;
        if (nodes) {
            vk_nodes = uploadI32(*nodes, GpuBackend::Vulkan);
            vk_nodes_ptr = &vk_nodes;
        }
        lfs::rendering::apply_selection_group_tensor_mask(
            vk_sel, vk_exist, vk_out, 3, host_locked.data(), test_case.add, vk_nodes_ptr, valid,
            test_case.replace, &vk_scratch);
        expectU8Equal(cuda_words, toU8(vk_out), "vulkan");
        expectI32Equal(cuda_scratch_words, toI32(vk_scratch), "vulkan_scratch");

        Tensor prog_sel = uploadBool(data.selected, GpuBackend::CUDA);
        Tensor prog_exist = uploadU8(data.existing, GpuBackend::CUDA);
        Tensor prog_out = emptyU8(kN, GpuBackend::CUDA);
        Tensor prog_scratch = emptyScratch(GpuBackend::CUDA);
        Tensor prog_nodes;
        const Tensor* prog_nodes_ptr = nullptr;
        if (nodes) {
            prog_nodes = uploadI32(*nodes, GpuBackend::CUDA);
            prog_nodes_ptr = &prog_nodes;
        }
        {
            GpuBackendScope cuda_scope(GpuBackend::CUDA);
            lfs::rendering::apply_selection_group_tensor_mask_program(
                prog_sel, prog_exist, prog_out, 3, host_locked.data(), test_case.add, prog_nodes_ptr,
                valid, test_case.replace, &prog_scratch);
        }
        expectU8Equal(cuda_words, toU8(prog_out), "cuda_program");
        expectI32Equal(cuda_scratch_words, toI32(prog_scratch), "cuda_program_scratch");
    }
    ASSERT_EQ(cudaFree(d_locked), cudaSuccess);
}

TEST_F(SelectionMaskOpsTest, ApplyGroupEmptyAllLockedAndSingleNode) {
    const auto host_locked = lockedHost({1, 2, 3, 4, 5, 6, 7});
    uint32_t* d_locked = lockedDevice(host_locked);
    ASSERT_NE(d_locked, nullptr);

    Tensor empty_sel = uploadBool({}, GpuBackend::CUDA);
    Tensor empty_exist = uploadU8({}, GpuBackend::CUDA);
    Tensor empty_out = emptyU8(0, GpuBackend::CUDA);
    lfs::rendering::apply_selection_group_tensor_mask(
        empty_sel, empty_exist, empty_out, 1, d_locked, true, nullptr, {}, false, nullptr);
    Tensor empty_vk_sel = uploadBool({}, GpuBackend::Vulkan);
    Tensor empty_vk_exist = uploadU8({}, GpuBackend::Vulkan);
    Tensor empty_vk_out = emptyU8(0, GpuBackend::Vulkan);
    lfs::rendering::apply_selection_group_tensor_mask(
        empty_vk_sel, empty_vk_exist, empty_vk_out, 1, host_locked.data(), true, nullptr, {}, false,
        nullptr);

    const std::vector<uint8_t> selected{1};
    const std::vector<uint8_t> existing{2};
    const std::vector<int> nodes{0};
    const std::vector<bool> valid{true};
    Tensor cuda_out = emptyU8(1, GpuBackend::CUDA);
    Tensor cuda_nodes = uploadI32(nodes, GpuBackend::CUDA);
    lfs::rendering::apply_selection_group_tensor_mask(
        uploadBool(selected, GpuBackend::CUDA), uploadU8(existing, GpuBackend::CUDA), cuda_out, 1,
        d_locked, true, &cuda_nodes, valid, false, nullptr);
    Tensor vk_out = emptyU8(1, GpuBackend::Vulkan);
    Tensor vk_nodes = uploadI32(nodes, GpuBackend::Vulkan);
    lfs::rendering::apply_selection_group_tensor_mask(
        uploadBool(selected, GpuBackend::Vulkan), uploadU8(existing, GpuBackend::Vulkan), vk_out, 1,
        host_locked.data(), true, &vk_nodes, valid, false, nullptr);
    expectU8Equal(toU8(cuda_out), toU8(vk_out), "single");
    EXPECT_EQ(toU8(cuda_out), (std::vector<uint8_t>{2}));

    const std::vector<uint8_t> all_sel(32, 1);
    std::vector<uint8_t> all_exist(32);
    for (std::size_t i = 0; i < all_exist.size(); ++i) {
        all_exist[i] = static_cast<uint8_t>(i % 8);
    }
    Tensor cuda_all = emptyU8(all_sel.size(), GpuBackend::CUDA);
    lfs::rendering::apply_selection_group_tensor_mask(
        uploadBool(all_sel, GpuBackend::CUDA), uploadU8(all_exist, GpuBackend::CUDA), cuda_all, 1,
        d_locked, true, nullptr, {}, false, nullptr);
    Tensor vk_all = emptyU8(all_sel.size(), GpuBackend::Vulkan);
    lfs::rendering::apply_selection_group_tensor_mask(
        uploadBool(all_sel, GpuBackend::Vulkan), uploadU8(all_exist, GpuBackend::Vulkan), vk_all, 1,
        host_locked.data(), true, nullptr, {}, false, nullptr);
    expectU8Equal(toU8(cuda_all), toU8(vk_all), "all_locked");
    ASSERT_EQ(cudaFree(d_locked), cudaSuccess);
}

TEST_F(SelectionMaskOpsTest, IndexedApplyMatchesKernelOnRandomInputs) {
    const auto data = makeRandomMasks(kN, 42);
    std::vector<int> visible;
    std::vector<uint8_t> vis_sel;
    visible.reserve(kN / 2);
    vis_sel.reserve(kN / 2);
    for (std::size_t i = 0; i < kN; i += 2) {
        visible.push_back(static_cast<int>(i));
        vis_sel.push_back(data.selected[i]);
    }
    visible[3] = -5;
    visible[5] = static_cast<int>(kN + 10);
    std::vector<int> vis_nodes(visible.size());
    for (std::size_t i = 0; i < visible.size(); ++i) {
        vis_nodes[i] = data.nodes[static_cast<std::size_t>(std::max(visible[i], 0)) % kN];
    }
    const auto host_locked = lockedHost(data.locked_groups);
    uint32_t* d_locked = lockedDevice(host_locked);
    ASSERT_NE(d_locked, nullptr);
    const struct Case {
        bool add;
        bool replace;
        const char* name;
    } cases[] = {{true, false, "add"}, {false, false, "remove"}, {true, true, "replace"}};
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        Tensor cuda_out = emptyU8(kN, GpuBackend::CUDA);
        Tensor cuda_nodes = uploadI32(vis_nodes, GpuBackend::CUDA);
        lfs::rendering::apply_selection_group_indexed_tensor_mask(
            uploadBool(vis_sel, GpuBackend::CUDA), uploadI32(visible, GpuBackend::CUDA),
            uploadU8(data.existing, GpuBackend::CUDA), cuda_out, 4, d_locked, test_case.add,
            &cuda_nodes, data.valid_nodes, test_case.replace);
        const auto cuda_words = toU8(cuda_out);

        Tensor vk_out = emptyU8(kN, GpuBackend::Vulkan);
        Tensor vk_nodes = uploadI32(vis_nodes, GpuBackend::Vulkan);
        lfs::rendering::apply_selection_group_indexed_tensor_mask(
            uploadBool(vis_sel, GpuBackend::Vulkan), uploadI32(visible, GpuBackend::Vulkan),
            uploadU8(data.existing, GpuBackend::Vulkan), vk_out, 4, host_locked.data(),
            test_case.add, &vk_nodes, data.valid_nodes, test_case.replace);
        expectU8Equal(cuda_words, toU8(vk_out), "vulkan");

        Tensor prog_out = emptyU8(kN, GpuBackend::CUDA);
        Tensor prog_nodes = uploadI32(vis_nodes, GpuBackend::CUDA);
        {
            GpuBackendScope cuda_scope(GpuBackend::CUDA);
            lfs::rendering::apply_selection_group_indexed_tensor_mask_program(
                uploadBool(vis_sel, GpuBackend::CUDA), uploadI32(visible, GpuBackend::CUDA),
                uploadU8(data.existing, GpuBackend::CUDA), prog_out, 4, host_locked.data(),
                test_case.add, &prog_nodes, data.valid_nodes, test_case.replace);
        }
        expectU8Equal(cuda_words, toU8(prog_out), "cuda_program");
    }
    ASSERT_EQ(cudaFree(d_locked), cudaSuccess);
}

TEST_F(SelectionMaskOpsTest, MergeOrMatchesKernel) {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> bit(0, 1);
    std::vector<uint8_t> acc(kN);
    std::vector<uint8_t> delta(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        acc[i] = static_cast<uint8_t>(bit(rng));
        delta[i] = static_cast<uint8_t>(bit(rng));
    }
    Tensor cuda_acc = uploadBool(acc, GpuBackend::CUDA);
    lfs::rendering::merge_selection_mask_or(cuda_acc, uploadBool(delta, GpuBackend::CUDA));
    const auto cuda_words = toBool(cuda_acc);

    Tensor vk_acc = uploadBool(acc, GpuBackend::Vulkan);
    lfs::rendering::merge_selection_mask_or(vk_acc, uploadBool(delta, GpuBackend::Vulkan));
    expectBoolEqual(cuda_words, toBool(vk_acc), "vulkan");

    Tensor prog_acc = uploadBool(acc, GpuBackend::CUDA);
    {
        GpuBackendScope cuda_scope(GpuBackend::CUDA);
        lfs::rendering::merge_selection_mask_or_program(prog_acc, uploadBool(delta, GpuBackend::CUDA));
    }
    expectBoolEqual(cuda_words, toBool(prog_acc), "cuda_program");
}

TEST_F(SelectionMaskOpsTest, NodeFilterMatchesKernel) {
    const auto data = makeRandomMasks(kN, 99);
    Tensor cuda_sel = uploadBool(data.selected, GpuBackend::CUDA);
    lfs::rendering::filter_selection_by_node_mask(
        cuda_sel, uploadI32(data.nodes, GpuBackend::CUDA), data.valid_nodes);
    const auto cuda_words = toBool(cuda_sel);

    Tensor vk_sel = uploadBool(data.selected, GpuBackend::Vulkan);
    lfs::rendering::filter_selection_by_node_mask(
        vk_sel, uploadI32(data.nodes, GpuBackend::Vulkan), data.valid_nodes);
    expectBoolEqual(cuda_words, toBool(vk_sel), "vulkan");

    Tensor prog_sel = uploadBool(data.selected, GpuBackend::CUDA);
    {
        GpuBackendScope cuda_scope(GpuBackend::CUDA);
        lfs::rendering::filter_selection_by_node_mask_program(
            prog_sel, uploadI32(data.nodes, GpuBackend::CUDA), data.valid_nodes);
    }
    expectBoolEqual(cuda_words, toBool(prog_sel), "cuda_program");
}

TEST_F(SelectionMaskOpsTest, CropFilterMatchesKernel) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> pos(-2.0f, 2.0f);
    std::vector<float> means(kN * 3);
    std::vector<uint8_t> selected(kN, 1);
    std::vector<int> nodes(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        means[i * 3] = pos(rng);
        means[i * 3 + 1] = pos(rng);
        means[i * 3 + 2] = pos(rng);
        nodes[i] = static_cast<int>(i % 2);
        if ((i % 11) == 0) {
            selected[i] = 0;
        }
    }
    const std::vector<float> identity{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1};
    const std::vector<float> crop_min{-0.5f, -0.25f, -1.0f};
    const std::vector<float> crop_max{0.75f, 1.0f, 0.5f};
    const std::vector<float> radii{0.8f, 1.1f, 0.6f};
    const std::vector<float> models{
        1, 0, 0, 0.4f,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        1, 0, 0, -0.4f,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1};
    const struct Case {
        bool crop_inv;
        bool ellip_inv;
        bool models;
        const char* name;
    } cases[] = {
        {false, false, false, "box"},
        {true, false, false, "box_inv"},
        {false, false, true, "box_models"},
        {false, true, false, "ellip_inv"},
    };
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        auto run = [&](const GpuBackend backend, const bool program) {
            Tensor sel = uploadBool(selected, backend);
            Tensor m = uploadF32(means, {kN, std::size_t{3}}, backend);
            Tensor crop_t = uploadF32(identity, {4, 4}, backend);
            Tensor cmin = uploadF32(crop_min, {3}, backend);
            Tensor cmax = uploadF32(crop_max, {3}, backend);
            Tensor ell_t = uploadF32(identity, {4, 4}, backend);
            Tensor ell_r = uploadF32(radii, {3}, backend);
            Tensor model_t = uploadF32(models, {2, 4, 4}, backend);
            Tensor idx = uploadI32(nodes, backend);
            const Tensor* model_ptr = test_case.models ? &model_t : nullptr;
            const Tensor* idx_ptr = test_case.models ? &idx : nullptr;
            if (program) {
                GpuBackendScope scope(backend);
                lfs::rendering::filter_selection_by_crop_program(
                    sel, m, &crop_t, &cmin, &cmax, test_case.crop_inv, &ell_t, &ell_r,
                    test_case.ellip_inv, model_ptr, idx_ptr);
            } else {
                lfs::rendering::filter_selection_by_crop(
                    sel, m, &crop_t, &cmin, &cmax, test_case.crop_inv, &ell_t, &ell_r,
                    test_case.ellip_inv, model_ptr, idx_ptr);
            }
            return toBool(sel);
        };
        const auto cuda_words = run(GpuBackend::CUDA, false);
        expectBoolEqual(cuda_words, run(GpuBackend::Vulkan, false), "vulkan");
        expectBoolEqual(cuda_words, run(GpuBackend::CUDA, true), "cuda_program");
    }
}

TEST_F(SelectionMaskOpsTest, ScreenWindowMatchesKernel) {
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> pos(-4.0f, 4.0f);
    std::vector<float> means(kN * 3);
    std::vector<uint8_t> selected(kN, 1);
    std::vector<int> nodes(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        means[i * 3] = pos(rng);
        means[i * 3 + 1] = pos(rng);
        means[i * 3 + 2] = pos(rng);
        nodes[i] = static_cast<int>(i % 2);
        if ((i % 13) == 0) {
            selected[i] = 0;
        }
    }
    const std::array<float, 9> view{
        0.9393727f, 0.0f, -0.3428978f,
        0.0f, 1.0f, 0.0f,
        0.3428978f, 0.0f, 0.9393727f};
    const std::array<float, 3> translation{1.0f, -2.0f, 0.5f};
    const std::vector<float> models{
        1, 0, 0, 0.25f,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        1, 0, 0, -0.25f,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1};
    const struct Case {
        ScreenWindowCameraModel model;
        bool with_models;
        const char* name;
    } cases[] = {
        {ScreenWindowCameraModel::Pinhole, false, "pinhole"},
        {ScreenWindowCameraModel::Orthographic, false, "ortho"},
        {ScreenWindowCameraModel::Equirectangular, false, "equirect"},
        {ScreenWindowCameraModel::Pinhole, true, "pinhole_models"},
        {ScreenWindowCameraModel::Equirectangular, true, "equirect_models"},
    };
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        auto run = [&](const GpuBackend backend, const bool program) {
            Tensor sel = uploadBool(selected, backend);
            Tensor m = uploadF32(means, {kN, std::size_t{3}}, backend);
            Tensor model_t = uploadF32(models, {2, 4, 4}, backend);
            Tensor idx = uploadI32(nodes, backend);
            const Tensor* model_ptr = test_case.with_models ? &model_t : nullptr;
            const Tensor* idx_ptr = test_case.with_models ? &idx : nullptr;
            if (program) {
                GpuBackendScope scope(backend);
                lfs::rendering::filter_selection_by_screen_window_program(
                    sel, m, view, translation, test_case.model, 1280, 720, 910.0f, 900.0f,
                    640.0f, 360.0f, 42.0f, 0.25f, 20.0f, 0.35f, 0.35f, 0.1f, -0.05f,
                    model_ptr, idx_ptr);
            } else {
                lfs::rendering::filter_selection_by_screen_window(
                    sel, m, view, translation, test_case.model, 1280, 720, 910.0f, 900.0f,
                    640.0f, 360.0f, 42.0f, 0.25f, 20.0f, 0.35f, 0.35f, 0.1f, -0.05f,
                    model_ptr, idx_ptr);
            }
            return toBool(sel);
        };
        const auto cuda_words = run(GpuBackend::CUDA, false);
        expectBoolEqual(cuda_words, run(GpuBackend::Vulkan, false), "vulkan");
        expectBoolEqual(cuda_words, run(GpuBackend::CUDA, true), "cuda_program");
    }
}

TEST_F(SelectionMaskOpsTest, EmptySelectionIsNoOp) {
    Tensor cuda_sel = uploadBool({}, GpuBackend::CUDA);
    Tensor vk_sel = uploadBool({}, GpuBackend::Vulkan);
    const std::vector<bool> valid{true, false};
    lfs::rendering::filter_selection_by_node_mask(cuda_sel, uploadI32({}, GpuBackend::CUDA), valid);
    lfs::rendering::filter_selection_by_node_mask(vk_sel, uploadI32({}, GpuBackend::Vulkan), valid);
    EXPECT_EQ(cuda_sel.numel(), 0u);
    EXPECT_EQ(vk_sel.numel(), 0u);
}
