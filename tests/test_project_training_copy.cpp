/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/project_document.hpp"
#include "io/project_operations.hpp"
#include "licht_test_support.hpp"

#include <algorithm>
#include <filesystem>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <vector>

namespace {

    namespace fs = std::filesystem;
    using namespace lfs::io::project;
    using namespace lfs::test::licht;

    fs::path make_training_dataset(const fs::path& root) {
        const auto dataset = root / "dataset";
        fs::create_directories(dataset / "images");
        fs::create_directories(dataset / "sparse" / "0");
        for (const auto* name : {"images/a.jpg", "images/b.jpg", "sparse/0/cameras.bin",
                                 "sparse/0/images.bin", "sparse/0/points3D.bin"})
            write_file_bytes(dataset / name, byte_vector(name));
        return dataset;
    }

    ParameterManagerSnapshot training_parameters() {
        auto parameters = require_result(
            require_result(ProjectDocument::create(fixed_uuid(4000))).parameters().snapshot());
        parameters.dataset.images = "images";
        parameters.mrnf_current.iterations = 1234;
        return parameters;
    }

    std::vector<std::string> file_names(const fs::path& directory) {
        std::vector<std::string> names;
        for (const auto& entry : fs::directory_iterator(directory))
            names.push_back(entry.path().filename().string());
        std::ranges::sort(names);
        return names;
    }

    // The copy is untrained, has the parameters, and embeds the dataset; nothing else is left beside it.
    TEST(TrainingProject, EmbedsTheDataset) {
        TemporaryDirectory temporary;
        const auto dataset = make_training_dataset(temporary.path);
        const auto destination = temporary.path / "copy" / "training.licht";
        fs::create_directories(destination.parent_path());
        std::vector<float> fractions;

        const auto background = temporary.path / "background.png";
        const auto init = temporary.path / "init.ply";
        write_file_bytes(background, byte_vector("background"));
        write_file_bytes(init, byte_vector("init"));
        auto source = training_parameters();
        source.mrnf_current.bg_image_path = background;
        source.mrnf_current_references.background_image_reference = fixed_uuid(4002);
        const TrainingProjectInitialization initialization{
            .init_path = init,
            .data_origin = {1.0F, 2.0F, 3.0F},
            .scene_center = std::nullopt};

        const auto result = require_result(write_training_project(
            destination, std::move(source), dataset, initialization,
            [&](const float fraction, const std::string&) { fractions.push_back(fraction); }));

        EXPECT_EQ(result.images_embedded, 2u);
        EXPECT_EQ(result.sparse_embedded, 3u);
        EXPECT_EQ(result.card.project_uuid, require_result(inspect_project_card(destination)).project_uuid);
        EXPECT_EQ(file_names(destination.parent_path()), std::vector<std::string>{"training.licht"});
        ASSERT_FALSE(fractions.empty());
        EXPECT_EQ(fractions.back(), 1.0F);

        auto document = require_result(ProjectDocument::open(destination));
        EXPECT_FALSE(require_result(document.bound_checkpoint_uuid()).has_value());
        const auto parameters = require_result(document.parameters().snapshot());
        EXPECT_EQ(parameters.mrnf_current.iterations, 1234u);
        EXPECT_EQ(parameters.dataset.images, "images");
        const auto manifest = require_result(document.parameters().embedded_dataset());
        ASSERT_TRUE(manifest.has_value());
        EXPECT_TRUE(manifest->complete);
        EXPECT_EQ(manifest->entries.size(), 5u);

        // Files outside the dataset are referenced by this document's REFS.
        const auto background_reference = parameters.mrnf_current_references.background_image_reference;
        ASSERT_TRUE(background_reference.has_value());
        EXPECT_NE(*background_reference, fixed_uuid(4002));
        EXPECT_EQ(resolve_path_reference(document.references(), destination.parent_path(), *background_reference),
                  std::optional{background.lexically_normal()});
        const auto recorded = require_result(document.project().training_initialization());
        ASSERT_TRUE(recorded.has_value());
        EXPECT_EQ(recorded->data_origin, initialization.data_origin);
        ASSERT_TRUE(recorded->init_reference_uuid.has_value());
        EXPECT_EQ(resolve_path_reference(document.references(), destination.parent_path(), *recorded->init_reference_uuid),
                  std::optional{init.lexically_normal()});
    }

    // An existing file is never replaced; an embed canceled while scanning or writing removes the new file again.
    TEST(TrainingProject, RefusesAndCleansUp) {
        TemporaryDirectory temporary;
        const auto dataset = make_training_dataset(temporary.path);
        const auto existing = temporary.path / "existing.licht";
        {
            auto document = require_result(ProjectDocument::create(fixed_uuid(4001)));
            ProjectDocumentSaveOptions options;
            options.disk_reserve_bytes = 0;
            static_cast<void>(require_result(document.save(existing, options)));
        }
        const auto existing_bytes = read_file_bytes(existing);

        const auto refused = write_training_project(existing, training_parameters(), dataset);
        ASSERT_FALSE(refused);
        EXPECT_EQ(refused.error().code(), lfs::ErrorCode::AlreadyExists);
        EXPECT_EQ(read_file_bytes(existing), existing_bytes);

        const auto missing = write_training_project(
            temporary.path / "missing.licht", training_parameters(), temporary.path / "no-dataset");
        ASSERT_FALSE(missing);
        EXPECT_EQ(missing.error().code(), lfs::ErrorCode::NotFound);

        const auto canceled = write_training_project(
            temporary.path / "canceled.licht", training_parameters(), dataset, {}, {}, [] { return true; });
        ASSERT_FALSE(canceled);
        EXPECT_EQ(canceled.error().code(), lfs::ErrorCode::Cancelled);

        // Five scan checks and one before writing, then one per embedded
        // file: cancel after the first file has been written.
        int cancel_checks = 0;
        const auto canceled_while_writing = write_training_project(
            temporary.path / "canceled-writing.licht", training_parameters(), dataset, {}, {},
            [&cancel_checks] { return ++cancel_checks > 7; });
        ASSERT_FALSE(canceled_while_writing);
        EXPECT_EQ(canceled_while_writing.error().code(), lfs::ErrorCode::Cancelled);
        EXPECT_EQ(cancel_checks, 8);

        EXPECT_EQ(file_names(temporary.path), (std::vector<std::string>{"dataset", "existing.licht"}));
    }

    // A bare file name is written to the working directory, with its dataset.
    TEST(TrainingProject, WritesARelativeDestination) {
        TemporaryDirectory temporary;
        const auto dataset = make_training_dataset(temporary.path);
        struct WorkingDirectory {
            fs::path previous = fs::current_path();
            ~WorkingDirectory() { fs::current_path(previous); }
        } restore;
        fs::current_path(temporary.path);

        const auto result = require_result(write_training_project("relative.licht", training_parameters(), dataset));

        EXPECT_EQ(result.images_embedded, 2u);
        EXPECT_EQ(file_names(temporary.path), (std::vector<std::string>{"dataset", "relative.licht"}));
    }

} // namespace
