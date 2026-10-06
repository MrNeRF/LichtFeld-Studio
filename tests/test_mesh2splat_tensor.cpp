/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Mesh2Splat as a tensor program: analytic contracts on every compiled tensor
// backend and, where the Vulkan converter is built, parity with it.

#include "core/material.hpp"
#include "core/mesh_data.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_backend.hpp"
#include "rendering/mesh2splat.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

namespace {
    using namespace lfs::core;

    constexpr float kShC0 = 0.28209479177387814f;
    constexpr int kResolution = 16;
    constexpr size_t kFaceSplats = kResolution * kResolution;

    Tensor floats(const std::vector<float>& values, const TensorShape& shape) {
        return Tensor::from_vector(values, shape, Device::CPU);
    }

    Tensor indices(const std::vector<int>& values, const TensorShape& shape) {
        return Tensor::from_vector(values, shape, Device::CPU);
    }

    float gamma(const float linear) { return std::pow(linear, 1.0f / 2.2f); }

    float srgb_to_linear(const uint8_t value) {
        const float c = value / 255.0f;
        return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
    }

    // Unit quad in z = 0 whose UVs follow its xy coordinates.
    MeshData textured_quad(TextureImage texture) {
        MeshData mesh(floats({0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0}, {4, 3}),
                      indices({0, 1, 2, 0, 2, 3}, {2, 3}));
        mesh.normals = floats({0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1}, {4, 3});
        mesh.texcoords = floats({0, 0, 1, 0, 1, 1, 0, 1}, {4, 2});
        Material material;
        material.albedo_tex = 1;
        mesh.materials.push_back(material);
        mesh.texture_images.push_back(std::move(texture));
        return mesh;
    }

    constexpr std::array<uint8_t, 4> kTexel{64, 128, 192, 255};

    TextureImage uniform_texture() {
        TextureImage image{.width = 2, .height = 2, .channels = 4};
        for (int texel = 0; texel < 4; ++texel)
            image.pixels.insert(image.pixels.end(), kTexel.begin(), kTexel.end());
        return image;
    }

    // Distinct RGB texels, so bilinear filtering and sRGB decoding matter.
    TextureImage gradient_texture() {
        TextureImage image{.width = 4, .height = 4, .channels = 3};
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                image.pixels.push_back(static_cast<uint8_t>(20 + 60 * x));
                image.pixels.push_back(static_cast<uint8_t>(30 + 70 * y));
                image.pixels.push_back(static_cast<uint8_t>(((x + y) & 1) ? 230 : 10));
            }
        }
        return image;
    }

    constexpr float kCubeMinColor = 0.1f;
    constexpr std::array<float, 2> kCubeBlue{0.2f, 0.7f}; // at z = 0 and z = 1

    MeshData colored_cube() {
        MeshData mesh(floats({0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                              0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1},
                             {8, 3}),
                      indices({0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
                               0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
                               0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5},
                              {12, 3}));
        std::vector<float> colors;
        for (int vertex = 0; vertex < 8; ++vertex) {
            colors.push_back((vertex & 1) ? 1.0f : 0.15f);
            colors.push_back((vertex & 2) ? 0.8f : kCubeMinColor);
            colors.push_back(kCubeBlue[(vertex & 4) ? 1 : 0]);
            colors.push_back(1.0f);
        }
        mesh.colors = floats(colors, {8, 4});
        return mesh;
    }

    const std::array<glm::vec4, 2> kSubmeshColors{glm::vec4(0.8f, 0.1f, 0.2f, 1.0f),
                                                  glm::vec4(0.15f, 0.75f, 0.3f, 1.0f)};

    // Two parallel quads drawn as separate submeshes with their own material.
    MeshData multi_material_quads() {
        MeshData mesh(floats({0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                              0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1},
                             {8, 3}),
                      indices({0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7}, {4, 3}));
        for (const auto& color : kSubmeshColors) {
            Material material;
            material.base_color = color;
            mesh.materials.push_back(material);
        }
        mesh.submeshes = {{0, 6, 0}, {6, 6, 1}};
        return mesh;
    }

    struct HostSplat {
        size_t count = 0;
        std::vector<float> means;
        std::vector<float> scales;
        std::vector<float> rotations;
        std::vector<float> colors; // RGB before the SH0 encoding
        std::vector<float> opacity;
    };

    std::vector<float> host_values(const Tensor& tensor) {
        const Tensor host = tensor.to(Device::CPU).contiguous();
        return {host.ptr<float>(), host.ptr<float>() + host.numel()};
    }

    HostSplat read_splat(const SplatData& splat) {
        HostSplat result{.count = static_cast<size_t>(splat.size()),
                         .means = host_values(splat.means()),
                         .scales = host_values(splat.scaling_raw()),
                         .rotations = host_values(splat.rotation_raw()),
                         .colors = host_values(splat.sh0()),
                         .opacity = host_values(splat.opacity_raw())};
        for (float& value : result.colors)
            value = value * kShC0 + 0.5f;
        return result;
    }

    Mesh2SplatOptions options() {
        Mesh2SplatOptions result;
        result.resolution_target = kResolution;
        return result;
    }

    HostSplat convert_tensor(const MeshData& mesh, const GpuBackend backend) {
        auto converted = lfs::rendering::mesh_to_splat_tensor(mesh, options(), backend);
        EXPECT_TRUE(converted) << (converted ? "" : std::string(converted.error().detail()));
        return converted ? read_splat(**converted) : HostSplat{};
    }

    // Every splat of an axis-aligned unit quad at resolution R covers one pixel:
    // the UV Jacobian is the identity, so scale is sigma / R in-plane and the
    // 1e-7 normal extent clamps to 1e-8.
    void expect_quad_splats(const HostSplat& splat, const size_t count) {
        ASSERT_EQ(splat.count, count);
        const float in_plane = std::log(options().sigma / kResolution);
        const float opacity = -std::log(1.0f / 0.999f - 1.0f);
        for (size_t i = 0; i < splat.count; ++i) {
            EXPECT_NEAR(splat.scales[i * 3], in_plane, 1e-5f);
            EXPECT_NEAR(splat.scales[i * 3 + 1], in_plane, 1e-5f);
            EXPECT_NEAR(splat.scales[i * 3 + 2], std::log(1e-8f), 1e-5f);
            // Unit (w, x, y, z) rotation whose local z (the thin axis) is the quad normal.
            const float* q = &splat.rotations[i * 4];
            EXPECT_NEAR(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3], 1.0f, 1e-5f);
            EXPECT_NEAR(std::abs(1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2])), 1.0f, 1e-5f);
            EXPECT_FLOAT_EQ(splat.opacity[i], opacity);
            for (int c = 0; c < 3; ++c) {
                EXPECT_GE(splat.means[i * 3 + c], -1e-6f);
                EXPECT_LE(splat.means[i * 3 + c], 1.0f + 1e-6f);
            }
        }
    }

    class TensorMesh2Splat : public testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!gpu_backend_available(GetParam()))
                GTEST_SKIP() << gpu_backend_name(GetParam()) << " is unavailable";
            // Compute-only backends have no raster artifacts for the program.
            const auto probe = lfs::rendering::mesh_to_splat_tensor(multi_material_quads(), options(), GetParam());
            if (!probe && probe.error().code() == lfs::ErrorCode::Unsupported)
                GTEST_SKIP() << probe.error().detail();
        }
    };

    TEST_P(TensorMesh2Splat, TexturedQuadCoversEveryPixelOnceWithTheSrgbTexel) {
        const HostSplat splat = convert_tensor(textured_quad(uniform_texture()), GetParam());
        expect_quad_splats(splat, kFaceSplats);
        for (size_t i = 0; i < splat.count; ++i) {
            EXPECT_EQ(splat.means[i * 3 + 2], 0.0f);
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(splat.colors[i * 3 + c], gamma(srgb_to_linear(kTexel[c])), 1e-5f);
        }
        // Pixel centers (i + 0.5) / R land on the splat means.
        std::vector<int> hits(kFaceSplats);
        for (size_t i = 0; i < splat.count; ++i) {
            const float x = splat.means[i * 3] * kResolution - 0.5f;
            const float y = splat.means[i * 3 + 1] * kResolution - 0.5f;
            EXPECT_NEAR(x, std::round(x), 1e-4f);
            EXPECT_NEAR(y, std::round(y), 1e-4f);
            const int px = static_cast<int>(std::round(x)), py = static_cast<int>(std::round(y));
            if (px >= 0 && px < kResolution && py >= 0 && py < kResolution)
                ++hits[py * kResolution + px];
        }
        EXPECT_TRUE(std::ranges::all_of(hits, [](const int n) { return n == 1; }));
    }

    TEST_P(TensorMesh2Splat, VertexColoredCubeCoversEachFace) {
        const HostSplat splat = convert_tensor(colored_cube(), GetParam());
        ASSERT_EQ(splat.count, 6 * kFaceSplats);
        std::array<size_t, 6> faces{}; // splats on x=0, x=1, y=0, y=1, z=0, z=1
        for (size_t i = 0; i < splat.count; ++i) {
            for (int axis = 0; axis < 3; ++axis) {
                const float coordinate = splat.means[i * 3 + axis];
                faces[axis * 2] += std::abs(coordinate) < 1e-6f;
                faces[axis * 2 + 1] += std::abs(coordinate - 1.0f) < 1e-6f;
            }
            for (int c = 0; c < 3; ++c) {
                EXPECT_GE(splat.colors[i * 3 + c], gamma(kCubeMinColor) - 1e-5f);
                EXPECT_LE(splat.colors[i * 3 + c], 1.0f + 1e-5f);
            }
            // Blue follows z alone, so it is constant on the z faces.
            const float z = splat.means[i * 3 + 2];
            if (std::abs(z) < 1e-6f || std::abs(z - 1.0f) < 1e-6f)
                EXPECT_NEAR(splat.colors[i * 3 + 2], gamma(z < 0.5f ? kCubeBlue[0] : kCubeBlue[1]), 1e-5f);
        }
        for (const size_t face : faces)
            EXPECT_EQ(face, kFaceSplats);
    }

    TEST_P(TensorMesh2Splat, SubmeshesUseTheirOwnMaterials) {
        const HostSplat splat = convert_tensor(multi_material_quads(), GetParam());
        expect_quad_splats(splat, 2 * kFaceSplats);
        std::array<size_t, 2> per_material{};
        for (size_t i = 0; i < splat.count; ++i) {
            const auto material = static_cast<size_t>(std::lround(splat.means[i * 3 + 2]));
            ASSERT_LT(material, 2u);
            ++per_material[material];
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(splat.colors[i * 3 + c], gamma(kSubmeshColors[material][c]), 1e-5f);
        }
        EXPECT_EQ(per_material[0], kFaceSplats);
        EXPECT_EQ(per_material[1], kFaceSplats);
    }

    INSTANTIATE_TEST_SUITE_P(CompiledBackends, TensorMesh2Splat, testing::ValuesIn(kCompiledGpuBackends),
                             [](const auto& info) { return std::string(gpu_backend_name(info.param)); });

#ifdef LFS_MESH2SPLAT_VULKAN_REFERENCE
    // Splat order follows fragment scheduling; compare rows sorted by mean.
    using Row = std::array<float, 14>;

    std::vector<Row> canonical_rows(const HostSplat& splat) {
        std::vector<Row> rows(splat.count);
        for (size_t i = 0; i < splat.count; ++i) {
            std::copy_n(&splat.means[i * 3], 3, rows[i].begin());
            std::copy_n(&splat.scales[i * 3], 3, rows[i].begin() + 3);
            std::copy_n(&splat.rotations[i * 4], 4, rows[i].begin() + 6);
            std::copy_n(&splat.colors[i * 3], 3, rows[i].begin() + 10);
            rows[i][13] = splat.opacity[i];
        }
        std::ranges::sort(rows, [](const Row& a, const Row& b) {
            return std::lexicographical_compare(a.begin(), a.begin() + 3, b.begin(), b.begin() + 3);
        });
        return rows;
    }

    struct Tolerance {
        float mean, log_scale, rotation, color;
    };

    void expect_vulkan_parity(const char* name, const MeshData& mesh, const GpuBackend backend,
                              const Tolerance tolerance) {
        auto reference = lfs::rendering::mesh_to_splat(mesh, options());
        ASSERT_TRUE(reference) << reference.error();
        const HostSplat vulkan = read_splat(**reference);
        const HostSplat tensor = convert_tensor(mesh, backend);
        ASSERT_EQ(tensor.count, vulkan.count) << name;
        const auto expected = canonical_rows(vulkan);
        const auto actual = canonical_rows(tensor);
        // mean, log scale, rotation, color, opacity
        constexpr std::array<int, 14> group{0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4};
        std::array<float, 5> max_diff{};
        for (size_t row = 0; row < expected.size(); ++row) {
            for (size_t c = 0; c < group.size(); ++c)
                max_diff[group[c]] = std::max(max_diff[group[c]], std::abs(expected[row][c] - actual[row][c]));
        }
        std::cout << "Mesh2Splat parity " << name << " Vulkan vs " << gpu_backend_name(backend)
                  << " tensor: count " << actual.size() << ", max |diff| mean " << max_diff[0]
                  << ", log scale " << max_diff[1] << ", rotation " << max_diff[2] << ", color "
                  << max_diff[3] << ", opacity " << max_diff[4] << '\n';
        EXPECT_LE(max_diff[0], tolerance.mean) << name;
        EXPECT_LE(max_diff[1], tolerance.log_scale) << name;
        EXPECT_LE(max_diff[2], tolerance.rotation) << name;
        EXPECT_LE(max_diff[3], tolerance.color) << name;
        EXPECT_EQ(max_diff[4], 0.0f) << name;
    }

    TEST_P(TensorMesh2Splat, MatchesTheVulkanConverter) {
        // The converter needs geometry shaders, which MoltenVK does not expose.
        if (const auto probe = lfs::rendering::mesh_to_splat(multi_material_quads(), options());
            !probe && probe.error().starts_with("No Vulkan device supports"))
            GTEST_SKIP() << probe.error();
        const Tolerance geometry{1e-6f, 1e-5f, 1e-6f, 1e-5f};
        // Hardware sRGB decoding can approximate the analytical transfer function.
        // Allow one input color step here; the analytic texel contract above stays tight.
        expect_vulkan_parity("textured_quad", textured_quad(uniform_texture()), GetParam(),
                             {geometry.mean, geometry.log_scale, geometry.rotation, 1.0f / 255.0f});
        expect_vulkan_parity("colored_cube", colored_cube(), GetParam(), geometry);
        expect_vulkan_parity("multi_material", multi_material_quads(), GetParam(), geometry);
        // Hardware samplers may quantize bilinear weights (commonly to 8 bits);
        // the tensor program filters in float.
        expect_vulkan_parity("gradient_texture", textured_quad(gradient_texture()), GetParam(),
                             {1e-6f, 1e-5f, 1e-6f, 1e-2f});
    }
#endif
} // namespace
