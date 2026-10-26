/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/sequencer_export_settings.hpp"
#include "io/video/video_export_schedule.hpp"
#include <gtest/gtest.h>

namespace lfs::vis::gui {
    using io::video::VideoPreset;

    TEST(SequencerExportSettingsTest, DraftDoesNotMutatePreferencesUntilApplied) {
        panels::SequencerUIState state;
        state.sequence_fps = 24.0f;
        state.playback_speed = 2.0f;
        state.equirectangular = true;
        state.follow_playback = true;
        SequencerExportSettings draft(state);
        draft.selectPreset(VideoPreset::TIKTOK_HD);
        draft.framerate = "60";
        draft.quality = "22";
        EXPECT_EQ(state.preset, VideoPreset::YOUTUBE_1080P);
        EXPECT_EQ(state.framerate, 30);
        EXPECT_EQ(state.quality, 18);
        const auto options = draft.validated();
        ASSERT_TRUE(options);
        EXPECT_TRUE(SequencerExportSettings::apply(*options, state));
        EXPECT_FALSE(SequencerExportSettings::apply(*options, state));
        EXPECT_EQ(state.preset, VideoPreset::TIKTOK_HD);
        EXPECT_EQ(state.outputWidth(), 1080);
        EXPECT_EQ(state.outputHeight(), 1920);
        EXPECT_EQ(state.framerate, 60);
        EXPECT_EQ(state.quality, 22);
        EXPECT_EQ(state.sequence_fps, 24.0f);
        EXPECT_EQ(state.playback_speed, 2.0f);
        EXPECT_TRUE(state.equirectangular);
        EXPECT_TRUE(state.follow_playback);
    }

    TEST(SequencerExportSettingsTest, EveryPresetPreservesExplicitFrameRateAndQuality) {
        panels::SequencerUIState state;
        state.framerate = 25;
        state.quality = 21;
        for (int p = 0; p < io::video::getPresetCount(); ++p) {
            SequencerExportSettings draft(state);
            const auto preset = static_cast<VideoPreset>(p);
            draft.selectPreset(preset);
            const auto options = draft.validated();
            ASSERT_TRUE(options) << p;
            EXPECT_EQ(options->framerate, 25);
            EXPECT_EQ(options->crf, 21);
            EXPECT_EQ(options->preset, preset);
            if (preset != VideoPreset::CUSTOM) {
                EXPECT_EQ(options->width, io::video::getPresetInfo(preset).width);
                EXPECT_EQ(options->height, io::video::getPresetInfo(preset).height);
            }
        }
    }

    TEST(SequencerExportSettingsTest, CustomDimensionsAndCustomSelectionSurviveReopening) {
        panels::SequencerUIState state;
        SequencerExportSettings draft(state);
        draft.width = "2048";
        draft.height = "1024";
        auto options = draft.validated();
        ASSERT_TRUE(options);
        EXPECT_EQ(options->preset, VideoPreset::CUSTOM);
        SequencerExportSettings::apply(*options, state);
        SequencerExportSettings reopened(state);
        EXPECT_EQ(reopened.preset, VideoPreset::CUSTOM);
        EXPECT_EQ(reopened.width, "2048");
        EXPECT_EQ(reopened.height, "1024");
        reopened.selectPreset(VideoPreset::CUSTOM);
        EXPECT_EQ(reopened.width, "2048");
        EXPECT_EQ(reopened.height, "1024");
        reopened.width = "1920";
        reopened.height = "1080";
        ASSERT_TRUE(reopened.validated());
        EXPECT_EQ(reopened.validated()->preset, VideoPreset::CUSTOM);
    }

    TEST(SequencerExportSettingsTest, RejectsInvalidNumbersAndOverflowWithoutChangingState) {
        const panels::SequencerUIState state;
        for (const std::string bad : {"", " ", "-1", "+30", "30fps", "1..5", "2147483648", "999999999999999999999999999"}) {
            SequencerExportSettings draft(state);
            draft.framerate = bad;
            EXPECT_FALSE(draft.validated()) << bad;
            draft.framerate = "30";
            draft.quality = bad;
            EXPECT_FALSE(draft.validated()) << bad;
        }
        for (const std::string bad : {"4294971136", "4294968376", "", "0", "-1920", "1920junk", "1921", "8194", "14"}) {
            SequencerExportSettings draft(state);
            draft.width = bad;
            EXPECT_FALSE(draft.validated()) << bad;
            draft.width = "1920";
            draft.height = bad;
            EXPECT_FALSE(draft.validated()) << bad;
        }
        EXPECT_EQ(state.outputWidth(), 1920);
        EXPECT_EQ(state.framerate, 30);
        EXPECT_EQ(state.quality, 18);
    }

    TEST(SequencerExportSettingsTest, EnforcesEncodingBoundsWithoutHiddenDimensionChanges) {
        const panels::SequencerUIState state;
        SequencerExportSettings draft(state);
        for (const char* fps : {"0", "1001"}) {
            draft.framerate = fps;
            EXPECT_FALSE(draft.validated());
        }
        draft.framerate = " 1000\t";
        for (const char* quality : {"0", "51"}) {
            draft.quality = quality;
            EXPECT_TRUE(draft.validated());
        }
        draft.quality = "52";
        EXPECT_FALSE(draft.validated());
        draft.quality = "18";
        draft.width = "8193";
        draft.height = "17";
        EXPECT_FALSE(draft.validated());
        draft.width = "8192";
        draft.height = "16";
        const auto options = draft.validated();
        ASSERT_TRUE(options);
        EXPECT_EQ(options->width, 8192);
        EXPECT_EQ(options->height, 16);
        EXPECT_EQ(options->preset, VideoPreset::CUSTOM);
    }

    TEST(SequencerExportSettingsTest, InvalidPresetCannotReachPersistedPreferences) {
        panels::SequencerUIState state;
        SequencerExportSettings draft(state);
        for (const int invalid : {-1, io::video::getPresetCount(), 1000}) {
            draft.preset = static_cast<VideoPreset>(invalid);
            EXPECT_FALSE(draft.validated());
        }
        EXPECT_EQ(state.preset, VideoPreset::YOUTUBE_1080P);
    }
    TEST(SequencerExportSettingsTest, LockedDimensionsUseStableAnchorAndNearestEvenDependentSide) {
        const panels::SequencerUIState state;
        SequencerExportSettings draft(state);
        draft.editDimension(true, "1280");
        EXPECT_EQ(draft.height, "720");
        EXPECT_EQ(draft.preset, VideoPreset::CUSTOM);
        draft.editDimension(false, "1080");
        EXPECT_EQ(draft.width, "1920");
        EXPECT_EQ(draft.aspectLabel(), "16:9");
        draft.editDimension(true, "1000");
        EXPECT_EQ(draft.height, "562");
        draft.editDimension(false, "1080");
        EXPECT_EQ(draft.width, "1920");
        // A malformed intermediate input is not clamped or used for arithmetic.
        draft.editDimension(true, "99999999999999999999999999");
        EXPECT_EQ(draft.height, "1080");
        EXPECT_FALSE(draft.validated());
    }

    TEST(SequencerExportSettingsTest, UnlockRelockAndSwapMaintainExplicitGlobalDimensions) {
        const panels::SequencerUIState state;
        SequencerExportSettings draft(state);
        draft.setAspectLocked(false);
        draft.editDimension(true, "1080");
        EXPECT_EQ(draft.height, "1080");
        EXPECT_EQ(draft.aspectLabel(), "1:1");
        draft.setAspectLocked(true);
        draft.editDimension(false, "2048");
        EXPECT_EQ(draft.width, "2048");
        draft.selectPreset(VideoPreset::YOUTUBE_1080P);
        draft.swapDimensions();
        EXPECT_EQ(draft.width, "1080");
        EXPECT_EQ(draft.height, "1920");
        EXPECT_EQ(draft.aspectLabel(), "9:16");
        draft.editDimension(false, "1280");
        EXPECT_EQ(draft.width, "720");
        draft.swapDimensions();
        EXPECT_EQ(draft.width, "1280");
        EXPECT_EQ(draft.height, "720");
        EXPECT_EQ(draft.framerate, "30");
        EXPECT_EQ(draft.quality, "18");
        EXPECT_EQ(state.outputWidth(), 1920);
    }

    TEST(SequencerExportSettingsTest, ExtremeLockedRatiosRejectUnencodableDependentSide) {
        panels::SequencerUIState state;
        state.preset = VideoPreset::CUSTOM;
        state.custom_width = 16;
        state.custom_height = 8192;
        SequencerExportSettings draft(state);
        draft.editDimension(true, "8192");
        EXPECT_EQ(draft.height, "4194304");
        EXPECT_FALSE(draft.validated());
        draft.editDimension(true, "16");
        ASSERT_TRUE(draft.validated());
        EXPECT_EQ(draft.height, "8192");
        draft.swapDimensions();
        draft.editDimension(true, "16");
        EXPECT_EQ(draft.height, "0");
        EXPECT_FALSE(draft.validated());
    }

    TEST(SequencerExportSettingsTest, EveryEvenWidthProducesNearestEvenHeightWithoutDrift) {
        const panels::SequencerUIState state;
        SequencerExportSettings draft(state);
        for (int width = 16; width <= 8192; width += 2) {
            draft.editDimension(true, std::to_string(width));
            const auto height = SequencerExportSettings::parseInteger(draft.height);
            ASSERT_TRUE(height);
            EXPECT_EQ(*height % 2, 0);
            EXPECT_LE(std::abs(*height - width * 9.0 / 16.0), 1.0);
        }
        draft.editDimension(true, "1920");
        EXPECT_EQ(draft.height, "1080");
    }
    TEST(SequencerExportSettingsTest, CommonFrameRatesAndManualValuesRemainIndependentOfResolution) {
        const panels::SequencerUIState state;
        SequencerExportSettings draft(state);
        for (const auto& fps : SequencerExportSettings::COMMON_FRAME_RATES) {
            ASSERT_TRUE(draft.selectFrameratePreset(fps.label));
            ASSERT_TRUE(draft.validated());
            EXPECT_EQ(draft.validated()->framerate, fps.rate.numerator);
            EXPECT_EQ(draft.validated()->framerate_denominator, fps.rate.denominator);
            EXPECT_EQ(draft.width, "1920");
            EXPECT_EQ(draft.height, "1080");
            EXPECT_EQ(draft.quality, "18");
        }
        for (const auto invalid : {"", "29.98", "31", "10000000000000000000", "-24"})
            EXPECT_FALSE(draft.selectFrameratePreset(invalid));
        draft.framerate = "31";
        ASSERT_TRUE(draft.validated());
        EXPECT_EQ(draft.validated()->framerate, 31);
    }

    TEST(SequencerExportSettingsTest, QualitySliderAndManualCrfAreExactInversesAcrossEveryValue) {
        const panels::SequencerUIState state;
        SequencerExportSettings draft(state);
        for (int crf = 0; crf <= 51; ++crf) {
            draft.quality = std::to_string(crf);
            ASSERT_TRUE(draft.qualitySliderValue());
            EXPECT_EQ(*draft.qualitySliderValue(), 51 - crf);
            ASSERT_TRUE(draft.selectQualitySlider(std::to_string(51 - crf) + ".000000"));
            ASSERT_TRUE(draft.validated());
            EXPECT_EQ(draft.validated()->crf, crf);
        }
        ASSERT_TRUE(draft.selectQualitySlider("33,000000"));
        EXPECT_EQ(draft.quality, "18");
        for (const auto invalid : {"33.000,000", "33,5", "", "51.1", "2.5", "33.", "NaN", "inf", "9999999999999999999999", "-1", "52"}) {
            const auto before = draft.quality;
            EXPECT_FALSE(draft.selectQualitySlider(invalid));
            EXPECT_EQ(draft.quality, before);
        }
        draft.quality = "52";
        EXPECT_FALSE(draft.qualitySliderValue());
        EXPECT_FALSE(draft.validated());
    }

    TEST(SequencerExportSettingsTest, FractionalRatesRemainExactThroughDraftStateAndSampling) {
        panels::SequencerUIState state;
        SequencerExportSettings draft(state);
        draft.framerate = "29,97";
        auto options = draft.validated();
        ASSERT_TRUE(options);
        EXPECT_EQ(options->framerate, 30000);
        EXPECT_EQ(options->framerate_denominator, 1001);
        EXPECT_EQ(draft.frameratePreset(), "29.97");
        EXPECT_EQ(io::video::parseVideoFrameRate("29.970"), io::video::parseVideoFrameRate("30000/1001"));
        SequencerExportSettings::apply(*options, state);
        EXPECT_EQ(SequencerExportSettings(state).framerate, "29.97");
        EXPECT_EQ(io::video::videoExportFrameCount(10.01f, 30000, 1001), 300);
        EXPECT_NEAR(io::video::videoExportSampleTime(300, 30000, 1001), 10.01, 0.00001);
        draft.framerate = "31.25";
        options = draft.validated();
        ASSERT_TRUE(options);
        EXPECT_EQ(options->framerate, 125);
        EXPECT_EQ(options->framerate_denominator, 4);
        EXPECT_TRUE(draft.frameratePreset().empty());
        for (const auto invalid : {"0/1", "30/0", "1/2", "1001/1", "30/1/2", "nan", "inf", "1e2", "1.1234567", "2147483648/1"})
            EXPECT_FALSE(io::video::parseVideoFrameRate(invalid)) << invalid;
    }

    TEST(SequencerExportSettingsTest, CameraRatioIsIndependentOfExportDraftAndBoundedAtEveryOrientation) {
        panels::SequencerUIState state;
        state.camera_width = 9;
        state.camera_height = 16;
        SequencerExportSettings draft(state);
        draft.selectPreset(VideoPreset::INSTAGRAM_SQUARE);
        auto options = draft.validated();
        ASSERT_TRUE(options);
        SequencerExportSettings::apply(*options, state);
        EXPECT_EQ(state.camera_width, 9);
        EXPECT_EQ(state.camera_height, 16);
        using namespace io::video;
        for (const auto ratio : {VideoFrameGeometry{16, 9}, {9, 16}, {1, 1}, {8192, 1}, {1, 8192}}) {
            for (const auto fit : {VideoFrameFit::FIT, VideoFrameFit::CROP, VideoFrameFit::STRETCH}) {
                const auto size = videoCameraRenderSize(1920, 1080, ratio.width, ratio.height, fit);
                EXPECT_GE(size.width, 1);
                EXPECT_LE(size.width, 8192);
                EXPECT_GE(size.height, 1);
                EXPECT_LE(size.height, 8192);
                EXPECT_LE(std::abs(size.width - size.height * static_cast<double>(ratio.width) / ratio.height),
                          1.0 + static_cast<double>(ratio.width) / ratio.height);
            }
        }
        const auto legacy = videoCameraRenderSize(1920, 1080, 0, 0, VideoFrameFit::FIT);
        EXPECT_EQ(legacy.width, 1920);
        EXPECT_EQ(legacy.height, 1080);
    }

    TEST(SequencerExportSettingsTest, MappingPreservesCenterAndFitCropStretchContracts) {
        using namespace io::video;
        const auto fit = videoFrameMapping(1600, 900, 1000, 1000, VideoFrameFit::FIT);
        EXPECT_FLOAT_EQ(fit.scale_x, fit.scale_y);
        EXPECT_LT(fit.offset_y, 0);
        EXPECT_NEAR(fit.offset_x, 0, 0.001f);
        const auto crop = videoFrameMapping(1600, 900, 1000, 1000, VideoFrameFit::CROP);
        EXPECT_FLOAT_EQ(crop.scale_x, crop.scale_y);
        EXPECT_GT(crop.offset_x, 0);
        EXPECT_NEAR(crop.offset_y, 0, 0.001f);
        for (const auto mode : {VideoFrameFit::FIT, VideoFrameFit::CROP, VideoFrameFit::STRETCH}) {
            const auto m = videoFrameMapping(1600, 900, 1000, 1000, mode);
            EXPECT_FLOAT_EQ(500 * m.scale_x + m.offset_x, 800);
            EXPECT_FLOAT_EQ(500 * m.scale_y + m.offset_y, 450);
        }
        const auto stretch = videoFrameMapping(1600, 900, 1000, 1000, VideoFrameFit::STRETCH);
        EXPECT_NE(stretch.scale_x, stretch.scale_y);
        EXPECT_NEAR(stretch.offset_x, 0, 0.001f);
        EXPECT_NEAR(stretch.offset_y, 0, 0.001f);
    }

    TEST(SequencerExportSettingsTest, RejectsInvalidFramingAndHiddenOverflow) {
        panels::SequencerUIState state;
        SequencerExportSettings draft(state);
        draft.framing = 99;
        EXPECT_FALSE(draft.validated());
        io::video::VideoExportOptions options;
        options.camera_width = 0;
        options.camera_height = 16;
        EXPECT_FALSE(io::video::validateVideoEncodingOptions(options));
        options.camera_width = 8193;
        EXPECT_FALSE(io::video::validateVideoEncodingOptions(options));
        options.camera_width = 16;
        options.framerate_denominator = 0;
        EXPECT_FALSE(io::video::validateVideoEncodingOptions(options));
    }

    TEST(SequencerExportSettingsTest, FittingIsOnlyNeededForDifferentCameraAndOutputRatios) {
        panels::SequencerUIState state;
        SequencerExportSettings draft(state);
        EXPECT_FALSE(draft.needsFraming());
        draft.editDimension(true, "1280");
        EXPECT_FALSE(draft.needsFraming());
        draft.framing = 1;
        EXPECT_EQ(draft.validated()->framing, io::video::VideoFrameFit::FIT);
        draft.swapDimensions();
        EXPECT_TRUE(draft.needsFraming());
        EXPECT_EQ(draft.validated()->framing, io::video::VideoFrameFit::CROP);
        draft.swapDimensions();
        EXPECT_FALSE(draft.needsFraming());
        state.camera_width = 32;
        state.camera_height = 18;
        EXPECT_FALSE(SequencerExportSettings(state).needsFraming());
        state.equirectangular = true;
        SequencerExportSettings pano(state);
        pano.swapDimensions();
        EXPECT_FALSE(pano.needsFraming());
        EXPECT_EQ(pano.validated()->framing, io::video::VideoFrameFit::FIT);
    }

} // namespace lfs::vis::gui
