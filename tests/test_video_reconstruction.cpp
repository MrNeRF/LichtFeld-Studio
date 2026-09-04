/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/video/video_export_options.hpp"
#include "io/video/video_reconstruction.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

    using namespace lfs::io::video;

    class TestVideoReconstructionCatalog final : public VideoReconstructionCatalog {
    public:
        explicit TestVideoReconstructionCatalog(
            std::vector<VideoReconstructionBackendDescriptor> descriptors)
            : descriptors_(std::move(descriptors)) {}

        std::optional<VideoReconstructionBackendDescriptor> findBackend(
            const std::string_view backend_id) const override {
            ++lookup_count;
            for (const auto& descriptor : descriptors_) {
                if (descriptor.backend_id == backend_id)
                    return descriptor;
            }
            return std::nullopt;
        }

        mutable int lookup_count = 0;

    private:
        std::vector<VideoReconstructionBackendDescriptor> descriptors_;
    };

    VideoReconstructionBackendDescriptor availableBackend() {
        return {
            .backend_id = "com.example.reconstruction",
            .provider_id = "com.example",
            .provider_version = "1.2.3",
            .provider_digest = "sha256:0123456789abcdef",
            .available = true,
            .supports_offline_export = true,
            .supports_perspective = true,
            .supports_equirectangular = false,
            .required_resources =
                resourceMask(VideoReconstructionResource::Color) |
                resourceMask(VideoReconstructionResource::Depth),
            .presets = {
                {
                    .preset_id = "quality",
                    .input_scale = 0.67f,
                    .additional_resources =
                        resourceMask(VideoReconstructionResource::MotionVectors),
                },
                {.preset_id = "performance", .input_scale = 0.5f},
            },
        };
    }

    VideoReconstructionRequest pluginRequest(
        const VideoReconstructionFallback fallback = VideoReconstructionFallback::Abort) {
        return {
            .selection = {
                .backend_id = "com.example.reconstruction",
                .preset_id = "quality",
                .fallback = fallback,
            },
            .output_width = 1920,
            .output_height = 1080,
            .projection = VideoReconstructionProjection::Perspective,
        };
    }

    TEST(VideoReconstructionSelectionTest, MissingSerializedSelectionMigratesToNative) {
        const auto selection = deserializeVideoReconstructionSelection(std::nullopt);

        ASSERT_TRUE(selection) << selection.error().message;
        EXPECT_EQ(selection->backend_id, "native");
        EXPECT_EQ(selection->preset_id, "native");
        EXPECT_EQ(selection->fallback, VideoReconstructionFallback::Abort);
    }

    TEST(VideoReconstructionSelectionTest, VersionedSelectionRoundTripsStably) {
        const VideoReconstructionSelection source{
            .backend_id = "com.example.reconstruction",
            .preset_id = "quality",
            .fallback = VideoReconstructionFallback::Native,
        };

        const auto serialized = serializeVideoReconstructionSelection(source);
        ASSERT_TRUE(serialized) << serialized.error().message;
        EXPECT_EQ(
            *serialized,
            R"({"version":1,"backend_id":"com.example.reconstruction","preset_id":"quality","fallback":"native"})");

        const auto restored = deserializeVideoReconstructionSelection(*serialized);
        ASSERT_TRUE(restored) << restored.error().message;
        EXPECT_EQ(*restored, source);
    }

    TEST(VideoReconstructionSelectionTest, InvalidOrUnsupportedDataIsRejected) {
        EXPECT_FALSE(deserializeVideoReconstructionSelection(R"({})"));
        EXPECT_FALSE(deserializeVideoReconstructionSelection(
            R"({"version":2,"backend_id":"native","preset_id":"native","fallback":"abort"})"));
        EXPECT_FALSE(deserializeVideoReconstructionSelection(
            R"({"version":1,"backend_id":"UPPERCASE","preset_id":"quality","fallback":"abort"})"));
        EXPECT_FALSE(deserializeVideoReconstructionSelection(
            R"({"version":1,"backend_id":"native","preset_id":"quality","fallback":"abort"})"));
        EXPECT_FALSE(deserializeVideoReconstructionSelection(
            R"({"version":1,"backend_id":"native","preset_id":"native","fallback":"silent"})"));
    }

    TEST(VideoReconstructionSelectionTest, InvalidInMemoryFallbackIsRejected) {
        auto selection = VideoReconstructionSelection{};
        selection.fallback = static_cast<VideoReconstructionFallback>(
            std::numeric_limits<std::uint8_t>::max());

        const auto validation = validateVideoReconstructionSelection(selection);

        ASSERT_FALSE(validation);
        EXPECT_EQ(validation.error().issue, VideoReconstructionSelectionIssue::InvalidFallback);
        const auto serialized = serializeVideoReconstructionSelection(selection);
        ASSERT_FALSE(serialized);
        EXPECT_EQ(serialized.error(), validation.error());
    }

    TEST(VideoReconstructionSelectionTest, FutureVersionsAreDistinctFromCorruptVersions) {
        for (const std::string_view version : {"2", "18446744073709551615"}) {
            const auto selection = deserializeVideoReconstructionSelection(
                std::format(R"({{"version":{},"future_schema":true}})", version));
            ASSERT_FALSE(selection);
            EXPECT_EQ(selection.error().issue, VideoReconstructionSelectionIssue::UnsupportedVersion);
        }
        for (const std::string_view version : {"0", "-1", "1.0", "2.0", "true", "null", R"("2")"}) {
            SCOPED_TRACE(version);
            const auto selection = deserializeVideoReconstructionSelection(
                std::format(R"({{"version":{}}})", version));
            ASSERT_FALSE(selection);
            EXPECT_EQ(selection.error().issue, VideoReconstructionSelectionIssue::InvalidVersion);
        }
        const auto missing = deserializeVideoReconstructionSelection("{}");
        ASSERT_FALSE(missing);
        EXPECT_EQ(missing.error().issue, VideoReconstructionSelectionIssue::MissingField);
    }

    TEST(VideoReconstructionSelectionTest, IdentifierGrammarAndLengthAreBounded) {
        for (const std::string& id : {std::string("0example.backend-v1_test"), std::string(128, 'a')}) {
            const VideoReconstructionSelection selection{.backend_id = id, .preset_id = id};
            EXPECT_TRUE(serializeVideoReconstructionSelection(selection));
        }
        for (const std::string& id : {std::string{}, std::string(".example"), std::string("-example"),
                                      std::string("_example"), std::string("Upper"), std::string("has space"),
                                      std::string("has/slash"), std::string(129, 'a')}) {
            SCOPED_TRACE(id);
            EXPECT_FALSE(serializeVideoReconstructionSelection({.backend_id = id}));
            EXPECT_FALSE(serializeVideoReconstructionSelection({.backend_id = "example", .preset_id = id}));
        }
    }

    TEST(VideoReconstructionSelectionTest, JsonByteLimitAndUnknownMembersAreExplicit) {
        std::string serialized =
            R"({"version":1,"backend_id":"native","preset_id":"native","fallback":"abort","future":{"ignored":true}})";
        serialized.resize(8192, ' ');
        const auto selection = deserializeVideoReconstructionSelection(serialized);
        ASSERT_TRUE(selection);
        EXPECT_EQ(*selection, VideoReconstructionSelection{});
        serialized.push_back(' ');
        const auto oversized = deserializeVideoReconstructionSelection(serialized);
        ASSERT_FALSE(oversized);
        EXPECT_EQ(oversized.error().issue, VideoReconstructionSelectionIssue::SizeLimitExceeded);
        const auto broken = deserializeVideoReconstructionSelection(R"({"version":2,)");
        ASSERT_FALSE(broken);
        EXPECT_EQ(broken.error().issue, VideoReconstructionSelectionIssue::InvalidJson);
    }

    TEST(VideoReconstructionPlanTest, NativeDefaultIsExactAndDoesNotQueryPlugins) {
        TestVideoReconstructionCatalog catalog({availableBackend()});
        const auto plan = resolveVideoReconstructionPlan(
            {
                .output_width = 1920,
                .output_height = 1080,
            },
            catalog);

        ASSERT_TRUE(plan) << plan.error().message;
        EXPECT_EQ(catalog.lookup_count, 0);
        EXPECT_EQ(plan->inputWidth(), 1920);
        EXPECT_EQ(plan->inputHeight(), 1080);
        EXPECT_EQ(plan->outputWidth(), 1920);
        EXPECT_EQ(plan->outputHeight(), 1080);
        EXPECT_FLOAT_EQ(plan->inputScale(), 1.0f);
        EXPECT_FALSE(plan->requiresReconstruction());
        EXPECT_FALSE(plan->fellBack());
        EXPECT_EQ(plan->provenance().requested_backend_id, "native");
        EXPECT_EQ(plan->provenance().effective_backend_id, "native");
    }

    TEST(VideoReconstructionPlanTest, CatalogMetadataResolvesToAnImmutableExportPlan) {
        TestVideoReconstructionCatalog catalog({availableBackend()});
        const auto plan = resolveVideoReconstructionPlan(pluginRequest(), catalog);

        ASSERT_TRUE(plan) << plan.error().message;
        EXPECT_EQ(catalog.lookup_count, 1);
        EXPECT_EQ(plan->inputWidth(), 1286);
        EXPECT_EQ(plan->inputHeight(), 724);
        EXPECT_EQ(plan->outputWidth(), 1920);
        EXPECT_EQ(plan->outputHeight(), 1080);
        EXPECT_FLOAT_EQ(plan->inputScale(), 0.67f);
        EXPECT_TRUE(plan->requiresReconstruction());
        EXPECT_FALSE(plan->fellBack());
        EXPECT_EQ(
            plan->requiredResources(),
            resourceMask(VideoReconstructionResource::Color) |
                resourceMask(VideoReconstructionResource::Depth) |
                resourceMask(VideoReconstructionResource::MotionVectors));
        EXPECT_EQ(plan->provenance().requested_provider_id, "com.example");
        EXPECT_EQ(plan->provenance().effective_provider_version, "1.2.3");
        EXPECT_EQ(
            plan->provenance().effective_provider_digest,
            "sha256:0123456789abcdef");
    }

    TEST(VideoReconstructionPlanTest, MissingBackendHonorsAbortPolicy) {
        TestVideoReconstructionCatalog catalog({});
        const auto plan = resolveVideoReconstructionPlan(pluginRequest(), catalog);

        ASSERT_FALSE(plan);
        EXPECT_EQ(
            plan.error().issue,
            VideoReconstructionResolutionIssue::BackendNotFound);
    }

    TEST(VideoReconstructionPlanTest, InvalidOutputExtentFailsBeforeCatalogLookup) {
        TestVideoReconstructionCatalog catalog({availableBackend()});
        auto request = pluginRequest();
        request.output_width = 1919;

        const auto plan = resolveVideoReconstructionPlan(request, catalog);

        ASSERT_FALSE(plan);
        EXPECT_EQ(
            plan.error().issue,
            VideoReconstructionResolutionIssue::InvalidOutputExtent);
        EXPECT_EQ(catalog.lookup_count, 0);
    }

    TEST(VideoReconstructionPlanTest, MissingBackendFallsBackBeforeRenderingWhenRequested) {
        TestVideoReconstructionCatalog catalog({});
        const auto plan = resolveVideoReconstructionPlan(
            pluginRequest(VideoReconstructionFallback::Native), catalog);

        ASSERT_TRUE(plan) << plan.error().message;
        EXPECT_TRUE(plan->fellBack());
        EXPECT_FALSE(plan->requiresReconstruction());
        EXPECT_EQ(plan->inputWidth(), 1920);
        EXPECT_EQ(plan->inputHeight(), 1080);
        EXPECT_EQ(plan->provenance().requested_backend_id, "com.example.reconstruction");
        EXPECT_EQ(plan->provenance().effective_backend_id, "native");
        EXPECT_EQ(
            plan->provenance().resolution_issue,
            VideoReconstructionResolutionIssue::BackendNotFound);
    }

    TEST(VideoReconstructionPlanTest, InvalidProjectionCannotUseNativeOrFallbackOrQueryCatalog) {
        TestVideoReconstructionCatalog catalog({availableBackend()});
        for (const auto fallback : {VideoReconstructionFallback::Abort, VideoReconstructionFallback::Native}) {
            for (const bool native : {false, true}) {
                auto request = pluginRequest(fallback);
                if (native)
                    request.selection = {.fallback = fallback};
                request.projection = static_cast<VideoReconstructionProjection>(255);
                const auto plan = resolveVideoReconstructionPlan(request, catalog);
                ASSERT_FALSE(plan);
                EXPECT_EQ(plan.error().issue, VideoReconstructionResolutionIssue::InvalidProjection);
            }
        }
        EXPECT_EQ(catalog.lookup_count, 0);
        EXPECT_EQ(videoReconstructionResolutionIssueId(VideoReconstructionResolutionIssue::InvalidProjection),
                  "invalid_projection");
    }

    TEST(VideoReconstructionPlanTest, ExtentValidationMatchesEncoderIncludingPixelBudgetBoundary) {
        // Largest even width within the packed-RGB budget for a two-pixel height.
        constexpr int boundary = (std::numeric_limits<int>::max() / 3 / 2) & ~1;
        for (const auto& [width, height] : std::vector<std::pair<int, int>>{
                 {0, 1080},
                 {-2, 1080},
                 {1920, 0},
                 {1919, 1080},
                 {1920, 1079},
                 {2, 2},
                 {8192, 8192},
                 {boundary, 2},
                 {boundary + 2, 2},
                 {std::numeric_limits<int>::max() - 1, std::numeric_limits<int>::max() - 1}}) {
            SCOPED_TRACE(std::format("{}x{}", width, height));
            const VideoExportOptions options{.width = width, .height = height};
            const auto valid = validateVideoEncodingOptions(options);
            TestVideoReconstructionCatalog catalog({});
            auto request = pluginRequest(VideoReconstructionFallback::Native);
            request.output_width = width;
            request.output_height = height;
            const auto plan = resolveVideoReconstructionPlan(request, catalog);
            EXPECT_EQ(plan.has_value(), valid.has_value());
            if (!valid) {
                ASSERT_FALSE(plan);
                EXPECT_EQ(plan.error().issue, VideoReconstructionResolutionIssue::InvalidOutputExtent);
                EXPECT_EQ(plan.error().message, valid.error());
                EXPECT_EQ(catalog.lookup_count, 0);
            }
            request.selection = {};
            EXPECT_EQ(resolveVideoReconstructionPlan(request).has_value(), valid.has_value());
        }
    }

    TEST(VideoReconstructionPlanTest, UnavailableBackendPreservesReasonInFallbackProvenance) {
        auto descriptor = availableBackend();
        descriptor.available = false;
        descriptor.unavailable_reason_id = "runtime.device_unsupported";
        TestVideoReconstructionCatalog catalog({std::move(descriptor)});
        const auto plan = resolveVideoReconstructionPlan(
            pluginRequest(VideoReconstructionFallback::Native), catalog);

        ASSERT_TRUE(plan) << plan.error().message;
        EXPECT_TRUE(plan->fellBack());
        EXPECT_EQ(
            plan->provenance().resolution_issue,
            VideoReconstructionResolutionIssue::BackendUnavailable);
        EXPECT_EQ(
            plan->provenance().unavailable_reason_id,
            "runtime.device_unsupported");
        EXPECT_EQ(plan->provenance().requested_provider_version, "1.2.3");
    }

    TEST(VideoReconstructionPlanTest, UnsupportedProjectionAndPresetAreExplicit) {
        TestVideoReconstructionCatalog catalog({availableBackend()});
        auto request = pluginRequest();
        request.projection = VideoReconstructionProjection::Equirectangular;

        auto plan = resolveVideoReconstructionPlan(request, catalog);
        ASSERT_FALSE(plan);
        EXPECT_EQ(
            plan.error().issue,
            VideoReconstructionResolutionIssue::ProjectionUnsupported);

        request.projection = VideoReconstructionProjection::Perspective;
        request.selection.preset_id = "missing";
        plan = resolveVideoReconstructionPlan(request, catalog);
        ASSERT_FALSE(plan);
        EXPECT_EQ(
            plan.error().issue,
            VideoReconstructionResolutionIssue::PresetNotFound);
    }

    TEST(VideoReconstructionPlanTest, InvalidCatalogDescriptorCannotReachExecution) {
        auto descriptor = availableBackend();
        descriptor.presets.push_back(descriptor.presets.front());
        TestVideoReconstructionCatalog catalog({std::move(descriptor)});

        const auto plan = resolveVideoReconstructionPlan(pluginRequest(), catalog);

        ASSERT_FALSE(plan);
        EXPECT_EQ(
            plan.error().issue,
            VideoReconstructionResolutionIssue::InvalidDescriptor);
    }

} // namespace
