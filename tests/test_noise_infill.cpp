/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/mesh_data.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor.hpp"
#include "io/formats/ply.hpp"
#include "training/noise_infill.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <string_view>
#include <vector>

using namespace lfs::core;
namespace fs = std::filesystem;

namespace {

    constexpr float kChannel = 0.5f / 0.28209479f;

    // The unit cube [0, 1]^3 with outward faces; `drop_bottom` leaves its z = 0 face open, as a scan leaves the underside
    // of an object on a table. `split_faces` gives every face its own four vertices, as a textured mesh's loader splits them
    // along its seams.
    MeshData make_cube(const bool drop_bottom, const bool split_faces = false) {
        const std::vector<float> corners{0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
        std::vector<int> indices{4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5,
                                 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7};
        if (!drop_bottom)
            indices.insert(indices.end(), {0, 2, 1, 0, 3, 2});
        std::vector<float> vertices = corners;
        if (split_faces) {
            vertices.clear();
            // Each face is two triangles, six indices.
            for (size_t face = 0; face < indices.size() / 6; ++face) {
                const int first = int(vertices.size() / 3);
                std::vector<int> face_corners;
                for (size_t k = 0; k < 6; ++k) {
                    int& index = indices[6 * face + k];
                    auto at = std::ranges::find(face_corners, index);
                    if (at == face_corners.end()) {
                        vertices.insert(vertices.end(), corners.begin() + 3 * index, corners.begin() + 3 * index + 3);
                        face_corners.push_back(index);
                        at = face_corners.end() - 1;
                    }
                    index = first + int(at - face_corners.begin());
                }
            }
        }
        return MeshData(Tensor::from_vector(vertices, {vertices.size() / 3, 3}, Device::CPU),
                        Tensor::from_vector(indices, {indices.size() / 3, 3}, Device::CPU));
    }

    auto load_real_splat() {
        return lfs::io::load_ply(fs::path(LFS_SIMPLIFY_TEST_PLY));
    }

} // namespace

// Catches noise placed on or outside the surface, where it would show as false colour, a fill that leaves the inside
// hollow, and a cell tied to the mesh's extent, which would give a solid shape millions of noise splats.
TEST(NoiseInfillTest, FillsTheInsideOfAClosedMeshClearOfItsSurface) {
    const auto infill = lfs::training::make_noise_infill(make_cube(false));
    ASSERT_TRUE(infill.has_value()) << infill.error().user_message();
    EXPECT_NEAR(infill->cell, std::cbrt(1.0f / 200000.0f), 1e-4f);
    const auto means = infill->splats.means().cpu().contiguous().to_vector();
    for (const float v : means) {
        ASSERT_GT(v, lfs::training::kNoiseInfillErosion * infill->cell);
        ASSERT_LT(v, 1.0f - lfs::training::kNoiseInfillErosion * infill->cell);
    }
    // Every cell whose centre is more than one surface layer plus the erosion from the faces is filled.
    const float clear = (lfs::training::kNoiseInfillErosion + 1.5f) * infill->cell;
    const auto inner = static_cast<size_t>((1.0f - 2 * clear) / infill->cell);
    EXPECT_GE(means.size() / 3, inner * inner * inner);
    EXPECT_LT(means.size() / 3, 200000u);
}

// Catches an open boundary loop letting the outside flood the inside, which leaves an open scan without noise, and texture
// seams read as holes, whose caps would wall off parts of the inside.
TEST(NoiseInfillTest, CapsOpenLoopsButNotTextureSeams) {
    for (const bool split_faces : {false, true}) {
        SCOPED_TRACE(split_faces);
        const auto closed = lfs::training::make_noise_infill(make_cube(false, split_faces));
        const auto open = lfs::training::make_noise_infill(make_cube(true, split_faces));
        ASSERT_TRUE(closed.has_value()) << closed.error().user_message();
        ASSERT_TRUE(open.has_value()) << open.error().user_message();
        EXPECT_EQ(closed->capped_loops, 0u);
        EXPECT_EQ(open->capped_loops, 1u);
        EXPECT_EQ(open->splats.size(), closed->splats.size());
    }
}

// Catches a flat shape or a splat file going on with an empty infill instead of stopping with a message that names the cause.
TEST(NoiseInfillTest, AMeshWithoutAnInsideStopsWithAMessage) {
    const std::vector<float> vertices{0, 0, 0, 1, 0, 0, 0, 1, 0};
    const std::vector<int> indices{0, 1, 2};
    const auto flat = lfs::training::make_noise_infill(MeshData(Tensor::from_vector(vertices, {3, 3}, Device::CPU),
                                                                Tensor::from_vector(indices, {1, 3}, Device::CPU)));
    ASSERT_FALSE(flat.has_value());
    EXPECT_NE(flat.error().user_message().find("encloses no cell"), std::string_view::npos) << flat.error().user_message();

    const auto not_a_mesh = lfs::training::load_noise_infill(fs::path(LFS_SIMPLIFY_TEST_PLY), {0.0f, 0.0f, 0.0f});
    ASSERT_FALSE(not_a_mesh.has_value());
    EXPECT_FALSE(not_a_mesh.error().user_message().empty());
}

// Catches a second shell overflowing the buffer of surface splats, whose holes let the outside flood the inside, and shells
// wound against each other, whose signed volume understates the inside and would ask for millions of noise splats.
TEST(NoiseInfillTest, ShellsWoundAgainstEachOtherFillAsOneInside) {
    const auto cube = make_cube(false);
    auto vertices = cube.vertices.cpu().contiguous().to_vector();
    auto indices = cube.indices.cpu().contiguous().to_vector_int();
    const size_t outer_vertices = vertices.size() / 3, outer_indices = indices.size();
    for (size_t v = 0; v < outer_vertices * 3; ++v)
        vertices.push_back(0.05f + 0.9f * vertices[v]);
    for (size_t i = 0; i < outer_indices; i += 3) {
        indices.push_back(indices[i] + int(outer_vertices));
        indices.push_back(indices[i + 2] + int(outer_vertices));
        indices.push_back(indices[i + 1] + int(outer_vertices));
    }
    const auto infill = lfs::training::make_noise_infill(MeshData(Tensor::from_vector(vertices, {vertices.size() / 3, 3}, Device::CPU),
                                                                  Tensor::from_vector(indices, {indices.size() / 3, 3}, Device::CPU)));
    ASSERT_TRUE(infill.has_value()) << infill.error().user_message();
    EXPECT_EQ(infill->capped_loops, 0u);
    EXPECT_GT(infill->splats.size(), 50000);
    EXPECT_LT(infill->splats.size(), 400000);
}

// Catches a patch wound the other way reading as a hole, whose caps would wall off parts of the inside.
TEST(NoiseInfillTest, AFlippedFaceIsNotAnOpenLoop) {
    auto mesh = make_cube(false);
    auto indices = mesh.indices.cpu().contiguous().to_vector_int();
    std::swap(indices[0], indices[1]);
    std::swap(indices[3], indices[4]);
    const auto flipped = lfs::training::make_noise_infill(
        MeshData(mesh.vertices, Tensor::from_vector(indices, {indices.size() / 3, 3}, Device::CPU)));
    ASSERT_TRUE(flipped.has_value()) << flipped.error().user_message();
    EXPECT_EQ(flipped->capped_loops, 0u);
    EXPECT_GT(flipped->splats.size(), 0);
}

// Catches a colour draw the tensor library refuses, or one that leaves the six high-contrast colours or the noise rows.
TEST(NoiseInfillTest, ColoursStayOnTheSixColoursOfTheNoiseRows) {
    auto loaded = load_real_splat();
    ASSERT_TRUE(loaded);
    auto& model = loaded->value;
    const auto rows = static_cast<size_t>(model.size());
    const size_t noise = rows / 10;
    const auto sh0 = model.sh0().cpu().contiguous().to_vector();
    lfs::training::randomize_noise_colors(model, noise);
    const auto recoloured = model.sh0().cpu().contiguous().to_vector();
    std::array<size_t, 8> seen{};
    for (size_t row = 0; row < noise; ++row) {
        size_t bits = 0;
        for (size_t channel = 0; channel < 3; ++channel) {
            const float value = recoloured[3 * row + channel];
            ASSERT_NEAR(std::abs(value), kChannel, 1e-4f) << "row " << row;
            bits = bits * 2 + (value > 0.0f ? 1 : 0);
        }
        ++seen[bits];
    }
    EXPECT_EQ(seen[0], 0u) << "black is not one of the six colours";
    EXPECT_EQ(seen[7], 0u) << "white is not one of the six colours";
    for (size_t bits = 1; bits < 7; ++bits)
        EXPECT_GT(seen[bits], noise / 12) << "colour " << bits << " is under-drawn";
    for (size_t i = noise * 3; i < rows * 3; ++i)
        ASSERT_EQ(recoloured[i], sh0[i]) << "surface row " << i / 3 << " took a noise colour";
}

// Catches noise disturbing the surface before its start step, a start that does not show it or does not ask for the
// learning-rate restart, and colours that stop changing afterwards.
TEST(NoiseInfillTest, NoiseAppearsOnceAtItsStartStep) {
    auto loaded = load_real_splat();
    ASSERT_TRUE(loaded);
    auto& model = loaded->value;
    const size_t noise = static_cast<size_t>(model.size()) / 10;
    model.set_frozen_ranges({{0, noise}});
    model.opacity_raw().slice(0, 0, noise).fill_(lfs::training::kHiddenOpacity);
    param::OptimizationParameters params;
    params.iterations = 1000;
    ASSERT_EQ(lfs::training::noise_infill_start_iteration(params), 300);

    const auto before = model.sh0().cpu().contiguous().to_vector();
    EXPECT_FALSE(lfs::training::noise_infill_step(model, params, 299));
    EXPECT_EQ(model.sh0().cpu().contiguous().to_vector(), before);
    EXPECT_EQ(model.opacity_raw().cpu().contiguous().to_vector()[0], lfs::training::kHiddenOpacity);

    EXPECT_TRUE(lfs::training::noise_infill_step(model, params, 300));
    EXPECT_EQ(model.opacity_raw().cpu().contiguous().to_vector()[noise - 1], lfs::training::kShownNoiseOpacity);
    const auto first = model.sh0().cpu().contiguous().to_vector();
    EXPECT_NE(first, before);

    EXPECT_FALSE(lfs::training::noise_infill_step(model, params, 301));
    EXPECT_NE(model.sh0().cpu().contiguous().to_vector(), first);
}

// Catches noise that stays hidden for good once a shortened run has passed its start step, and noise that stops changing
// colour when a lengthened run moves the start step ahead of it.
TEST(NoiseInfillTest, NoiseFollowsAChangedIterationLimit) {
    auto loaded = load_real_splat();
    ASSERT_TRUE(loaded);
    auto& model = loaded->value;
    const size_t noise = static_cast<size_t>(model.size()) / 10;
    model.set_frozen_ranges({{0, noise}});
    model.opacity_raw().slice(0, 0, noise).fill_(lfs::training::kHiddenOpacity);
    param::OptimizationParameters params;
    params.iterations = 1000;

    EXPECT_TRUE(lfs::training::noise_infill_step(model, params, 500));
    EXPECT_EQ(model.opacity_raw().cpu().contiguous().to_vector()[noise - 1], lfs::training::kShownNoiseOpacity);

    params.iterations = 100000;
    const auto colours = model.sh0().cpu().contiguous().to_vector();
    EXPECT_FALSE(lfs::training::noise_infill_step(model, params, 501));
    EXPECT_NE(model.sh0().cpu().contiguous().to_vector(), colours);
}

// Catches a row cap mixing the noise with other frozen rows, which would then be recoloured and left out of every export.
TEST(NoiseInfillTest, ARowCapKeepsTheNoiseApartFromOtherFrozenRows) {
    auto loaded = load_real_splat();
    ASSERT_TRUE(loaded);
    auto& model = loaded->value;
    const auto rows = static_cast<size_t>(model.size());
    const size_t noise = rows / 10, added = rows / 10, added_start = rows - added;
    model.opacity_raw().slice(0, 0, noise).fill_(lfs::training::kShownNoiseOpacity);
    model.set_frozen_ranges({{0, noise}, {added_start, added}});
    const auto means = model.means().cpu().contiguous().to_vector();

    random_choose(model, static_cast<int>(noise + added + (rows - noise - added) / 2));

    EXPECT_EQ(lfs::training::noise_infill_rows(model), noise);
    ASSERT_EQ(model.frozen_ranges().size(), 2u);
    EXPECT_EQ(model.frozen_ranges()[1].count, added);
    const auto kept_means = model.means().cpu().contiguous().to_vector();
    for (size_t k = 0; k < 3; ++k) {
        EXPECT_EQ(kept_means[3 * model.frozen_ranges()[1].start + k], means[3 * added_start + k]);
        EXPECT_EQ(kept_means[k], means[k]);
    }
}

// Catches the noise taking rows from the surface, a cap appearing where there was none, and added frozen rows that lead a
// model being taken for noise.
TEST(NoiseInfillTest, NoiseStaysOutsideTheRowCap) {
    auto loaded = load_real_splat();
    ASSERT_TRUE(loaded);
    auto& model = loaded->value;
    model.set_frozen_ranges({{0, 100}});
    param::OptimizationParameters params;
    params.max_cap = 1000;
    EXPECT_EQ(lfs::training::with_noise_outside_cap(params, &model).max_cap, 1000);
    model.opacity_raw().slice(0, 0, 100).fill_(lfs::training::kShownNoiseOpacity);
    EXPECT_EQ(lfs::training::with_noise_outside_cap(params, &model).max_cap, 1100);
    EXPECT_EQ(lfs::training::with_noise_outside_cap(params, nullptr).max_cap, 1000);
    params.max_cap = 0;
    EXPECT_EQ(lfs::training::with_noise_outside_cap(params, &model).max_cap, 0);
}

// Catches noise reaching an export, and an override that is not undone.
TEST(NoiseInfillTest, NoiseRowsLeaveTheExportAndOverridesAreUndone) {
    auto loaded = load_real_splat();
    ASSERT_TRUE(loaded);
    auto& model = loaded->value;
    const auto rows = static_cast<size_t>(model.size());
    const size_t noise = rows / 10;
    model.set_frozen_ranges({{0, noise}});
    EXPECT_EQ(lfs::training::noise_infill_rows(model), 0u) << "the opacity of an added splat";
    EXPECT_FALSE(lfs::training::without_noise_infill(model));
    model.opacity_raw().slice(0, 0, noise).fill_(lfs::training::kShownNoiseOpacity);
    ASSERT_EQ(lfs::training::noise_infill_rows(model), noise);

    const auto opacity = model.opacity_raw().cpu().contiguous().to_vector();
    {
        const lfs::training::RowOverride hidden(model.opacity_raw(), 0, noise, lfs::training::kHiddenOpacity);
        const auto overridden = model.opacity_raw().cpu().contiguous().to_vector();
        EXPECT_EQ(overridden[noise - 1], lfs::training::kHiddenOpacity);
        EXPECT_EQ(overridden[noise], opacity[noise]);
    }
    EXPECT_EQ(model.opacity_raw().cpu().contiguous().to_vector(), opacity);

    const auto exported = lfs::training::without_noise_infill(model);
    ASSERT_TRUE(exported);
    ASSERT_EQ(static_cast<size_t>(exported->size()), rows - noise);
    const auto means = model.means().cpu().contiguous().to_vector();
    const auto exported_means = exported->means().cpu().contiguous().to_vector();
    for (size_t k = 0; k < 3; ++k)
        EXPECT_EQ(exported_means[k], means[3 * noise + k]);
}
