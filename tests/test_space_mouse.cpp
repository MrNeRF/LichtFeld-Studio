/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/space_mouse.hpp"
#include "internal/viewport.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

namespace {
    using namespace lfs::vis::input;
    using namespace std::chrono_literals;
    using Camera = decltype(std::declval<Viewport>().camera);

    struct RecordedReport {
        int milliseconds;
        std::vector<std::uint8_t> bytes;
    };

    std::vector<RecordedReport> capture() {
        std::ifstream file(std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/space_mouse/compact_capture.txt");
        EXPECT_TRUE(file);
        std::vector<RecordedReport> result;
        std::string line;
        std::getline(file, line);
        while (std::getline(file, line)) {
            std::istringstream stream(line);
            int milliseconds = 0, size = 0;
            char colon = 0;
            if (!(stream >> milliseconds >> size >> colon) || colon != ':')
                continue;
            RecordedReport report{milliseconds, {}};
            unsigned value = 0;
            while (stream >> std::hex >> value)
                report.bytes.push_back(static_cast<std::uint8_t>(value));
            EXPECT_EQ(report.bytes.size(), static_cast<std::size_t>(size));
            result.push_back(std::move(report));
        }
        return result;
    }

    class RecordedSource final : public SpaceMouseSource {
    public:
        bool attached = true;
        bool failed = false;
        int scans = 0;
        int closes = 0;
        std::deque<std::vector<std::uint8_t>> reports;

        std::optional<SpaceMouseConnection> open() override {
            ++scans;
            return attached ? std::optional(SpaceMouseConnection{"SpaceMouse Compact", "recording"}) : std::nullopt;
        }
        SpaceMouseRead read(SpaceMouseReport& report) override {
            if (failed)
                return SpaceMouseRead::Disconnected;
            if (reports.empty())
                return SpaceMouseRead::Timeout;
            const auto decoded = decodeSpaceMouseHid(reports.front(), 350.0f);
            reports.pop_front();
            if (!decoded)
                return SpaceMouseRead::Timeout;
            report = *decoded;
            return SpaceMouseRead::Report;
        }
        void close() override { ++closes; }
    };

    Camera camera() {
        Camera result;
        result.R = glm::mat3(1.0f);
        result.t = {0, 0, 5};
        result.pivot = glm::vec3(0.0f);
        result.prePos = glm::vec2(0.0f);
        return result;
    }
} // namespace

TEST(SpaceMouseDecoder, RealCaptureCountsAndCompactButtons) {
    std::array<int, 4> counts{};
    std::vector<std::uint64_t> buttons;
    for (const auto& report : capture()) {
        ASSERT_FALSE(report.bytes.empty());
        ASSERT_LT(report.bytes[0], counts.size());
        ++counts[report.bytes[0]];
        const auto decoded = decodeSpaceMouseHid(report.bytes, 350.0f);
        ASSERT_TRUE(decoded);
        if (report.bytes[0] == 3)
            buttons.push_back(decoded->buttons);
    }
    EXPECT_EQ(counts, (std::array<int, 4>{0, 851, 852, 4}));
    EXPECT_EQ(buttons, (std::vector<std::uint64_t>{2, 0, 1, 0}));
}

TEST(SpaceMouseDecoder, RecordedGestureAxesAndSigns) {
    struct Gesture {
        const char* name;
        int start, end, dominant_start, dominant_end, report, axis, sign;
    };
    const Gesture gestures[] = {
        {"right", 7900, 9100, 8800, 9000, 1, 0, 1},
        {"forward", 11000, 12500, 11000, 12500, 1, 2, -1},
        {"up", 12500, 14500, 12500, 14500, 1, 1, 1},
        {"down", 14500, 16500, 14500, 16500, 1, 1, -1},
        {"nose down", 17500, 20900, 17500, 20900, 2, 0, -1},
        {"clockwise twist", 21000, 24000, 23400, 23600, 2, 1, -1},
        {"right roll", 24500, 27000, 24500, 27000, 2, 2, -1},
    };
    const auto reports = capture();
    for (const auto& gesture : gestures) {
        SCOPED_TRACE(gesture.name);
        glm::vec3 whole(0.0f), window(0.0f);
        int count = 0;
        for (const auto& report : reports) {
            if (report.bytes[0] != gesture.report || report.milliseconds < gesture.start || report.milliseconds >= gesture.end)
                continue;
            const auto decoded = decodeSpaceMouseHid(report.bytes, 350.0f);
            ASSERT_TRUE(decoded);
            const auto axes = gesture.report == 1 ? *decoded->translation : *decoded->rotation;
            whole += axes;
            if (report.milliseconds >= gesture.dominant_start && report.milliseconds < gesture.dominant_end) {
                window += axes;
                ++count;
            }
        }
        EXPECT_GT(count, 0);
        EXPECT_GT(whole[gesture.axis] * gesture.sign, 0.0f);
        EXPECT_GT(window[gesture.axis] * gesture.sign, 0.0f);
        for (int axis = 0; axis < 3; ++axis)
            if (axis != gesture.axis)
                EXPECT_GT(std::abs(window[gesture.axis]), std::abs(window[axis]));
    }
}

TEST(SpaceMouseDecoder, CombinedReportsAndDeviceScale) {
    const auto reports = capture();
    const auto translation = reports[15].bytes[0] == 1 ? reports[15].bytes : reports[16].bytes;
    const auto rotation = reports[15].bytes[0] == 2 ? reports[15].bytes : reports[16].bytes;
    ASSERT_EQ(translation[0], 1);
    ASSERT_EQ(rotation[0], 2);
    auto combined = translation;
    combined.insert(combined.end(), rotation.begin() + 1, rotation.end());
    combined.push_back(0);
    const auto decoded = decodeSpaceMouseHid(combined, 500.0f);
    ASSERT_TRUE(decoded);
    EXPECT_EQ(decoded->translation, decodeSpaceMouseHid(translation, 500.0f)->translation);
    EXPECT_EQ(decoded->rotation, decodeSpaceMouseHid(rotation, 500.0f)->rotation);
    EXPECT_NEAR(decoded->translation->z, decodeSpaceMouseHid(translation, 350.0f)->translation->z * 0.7f, 1e-6f);
}

TEST(SpaceMouseDecoder, LargerModelButtonArraysAndMalformedReports) {
    const std::array<std::uint8_t, 7> buttons{0x1c, 1, 0, 2, 0, 0, 0};
    ASSERT_TRUE(decodeSpaceMouseHid(buttons, 350));
    EXPECT_EQ(decodeSpaceMouseHid(buttons, 350)->buttons, 3);
    auto long_press = buttons;
    long_press[0] = 0x1d;
    EXPECT_TRUE(decodeSpaceMouseHid(long_press, 350)->long_press);
    for (const auto& bytes : {std::vector<std::uint8_t>{}, {1, 0}, {2, 0}, {3}, {0x1c, 0}, {0xff, 0}})
        EXPECT_FALSE(decodeSpaceMouseHid(bytes, 350));
    EXPECT_FALSE(decodeSpaceMouseHid(buttons, 0));
}

TEST(SpaceMouseDecoder, CompositeCollectionsAreNotNavigationDevices) {
    EXPECT_TRUE(isSpaceMouse(0x256f, 0xc635, 1, 8));
    EXPECT_TRUE(isSpaceMouse(0x256f, 0xc635, 0, 0));
    EXPECT_TRUE(isSpaceMouse(0x1234, 0x1234, 1, 8));
    EXPECT_FALSE(isSpaceMouse(0x256f, 0xc635, 1, 6));
    EXPECT_FALSE(isSpaceMouse(0x256f, 0xc650, 0, 0));
    EXPECT_EQ(spaceMouseProfile(0x256f, 0xc635)->full_scale, 350.0f);
}

TEST(SpaceMouseDecoder, LogicalRangeFromTheCapturedDescriptor) {
    std::ifstream file(std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/space_mouse/compact_descriptor.hex");
    ASSERT_TRUE(file);
    std::vector<std::uint8_t> descriptor;
    unsigned value = 0;
    while (file >> std::hex >> value)
        descriptor.push_back(static_cast<std::uint8_t>(value));
    ASSERT_EQ(descriptor.size(), 239);
    EXPECT_FLOAT_EQ(spaceMouseHidScale(descriptor, 500), 350);
    // Exercise the same descriptor layout advertising a different logical range.
    for (std::size_t i = 0; i + 2 < descriptor.size(); ++i) {
        if (descriptor[i] == 0x26 && descriptor[i + 1] == 0x5e && descriptor[i + 2] == 0x01) {
            descriptor[i + 1] = 0xf4;
        }
        if (descriptor[i] == 0x16 && descriptor[i + 1] == 0xa2 && descriptor[i + 2] == 0xfe) {
            descriptor[i + 1] = 0x0c;
        }
    }
    EXPECT_FLOAT_EQ(spaceMouseHidScale(descriptor, 350), 500);
    EXPECT_FLOAT_EQ(spaceMouseHidScale({}, 350), 350);
    const std::array<std::uint8_t, 2> truncated{0x26, 0x5e};
    EXPECT_FLOAT_EQ(spaceMouseHidScale(truncated, 350), 350);
}

TEST(SpaceMouseDecoder, SpnavNativePacketsAndCameraAxes) {
    // libspnav/src/proto.h: motion=0, press=1, release=2; eight native int32s.
    std::array<std::int32_t, 8> packet{0, 500, -250, -100, -500, -200, 300, 16};
    auto bytes = std::span(reinterpret_cast<const std::uint8_t*>(packet.data()), sizeof(packet));
    const auto motion = decodeSpaceMouseSpnav(bytes);
    ASSERT_TRUE(motion);
    EXPECT_EQ(*motion->translation, glm::vec3(1.0f, -0.5f, 0.2f));
    EXPECT_EQ(*motion->rotation, glm::vec3(-1.0f, -0.4f, -0.6f));
    packet = {1, 1, 1, 0, 0, 0, 0, 0};
    EXPECT_EQ(decodeSpaceMouseSpnav(bytes)->buttons, 2);
    packet[0] = 2;
    EXPECT_EQ(decodeSpaceMouseSpnav(bytes)->buttons, 0);
    EXPECT_EQ(decodeSpaceMouseSpnav(bytes)->button_mask, 2);
    packet[1] = 64;
    EXPECT_FALSE(decodeSpaceMouseSpnav(bytes));
    EXPECT_FALSE(decodeSpaceMouseSpnav(bytes.first(31)));
}

TEST(SpaceMouseResponse, DeadZoneMonotonicCurveAndLinearTime) {
    EXPECT_EQ(spaceMouseResponse(0.05f), 0);
    EXPECT_EQ(spaceMouseResponse(-0.05f), 0);
    float previous = 0.0f;
    for (int i = 0; i <= 100; ++i) {
        const float response = spaceMouseResponse(i / 100.0f);
        EXPECT_GE(response, previous);
        EXPECT_FLOAT_EQ(spaceMouseResponse(-i / 100.0f), -response);
        previous = response;
    }
    SpaceMouseMotion motion{{0.5f, -1.0f, 0.2f}, {1.0f, -0.5f, 0.3f}};
    const auto a = spaceMouseStep(motion, 2.0f, 1.0f, 0.01f, false, false);
    const auto b = spaceMouseStep(motion, 2.0f, 1.0f, 0.02f, false, false);
    EXPECT_EQ(a.translation * 2.0f, b.translation);
    EXPECT_EQ(a.rotation * 2.0f, b.rotation);
    EXPECT_EQ(spaceMouseStep(motion, 1, 1, 0.1f, false, true).rotation.z, 0);
}

// Catches a puck that orbits, pans or looks by its own rules instead of the mouse's: the same pixel drag through the
// mouse path and the 3D mouse path must leave identical cameras, in turntable and trackball orbit, pan and fly look.
TEST(SpaceMouseNavigation, PuckDrivesTheMouseOrbitPanAndLook) {
    const glm::vec2 drag{12.0f, -7.0f};
    for (const bool trackball : {false, true}) {
        auto mouse = camera(), puck = camera();
        mouse.startRotateAroundCenter({100, 100}, 0.0f);
        if (trackball)
            mouse.updateTrackballRotateAroundCenter(glm::vec2(100, 100) + drag, 0.0f);
        else
            mouse.updateRotateAroundCenter(glm::vec2(100, 100) + drag, 0.0f);
        puck.applySpaceMouseOrbit(drag, {}, 0.0f, trackball);
        EXPECT_LT(glm::length(puck.t - mouse.t), 1e-5f);
        EXPECT_LT(glm::length(puck.R[2] - mouse.R[2]) + glm::length(puck.R[1] - mouse.R[1]), 1e-5f);
        EXPECT_EQ(puck.pivot, mouse.pivot);
    }
    auto mouse = camera(), puck = camera();
    mouse.startPan({100, 100}, 0.0f);
    mouse.panDrag(glm::vec2(100, 100) + drag, 0.0f);
    puck.applySpaceMouseOrbit({}, drag, 0.0f, false);
    EXPECT_LT(glm::length(puck.t - mouse.t) + glm::length(puck.pivot - mouse.pivot), 1e-5f);
    mouse = camera();
    puck = camera();
    mouse.initScreenPos({100, 100});
    mouse.rotateFpv(glm::vec2(100, 100) + drag);
    puck.applySpaceMouseFly(drag, 0.0f, {}, true);
    EXPECT_EQ(puck.t, mouse.t);
    EXPECT_LT(glm::length(puck.R[2] - mouse.R[2]) + glm::length(puck.pivot - mouse.pivot), 1e-5f);
}

// Catches a fly movement in world instead of camera axes, or one that leaves the pivot behind: pushing right moves
// the camera and its pivot right together.
TEST(SpaceMouseNavigation, FlyMovesTheCameraWithItsPivot) {
    auto view = camera();
    view.applySpaceMouseFly({}, 0.0f, {1, 0, 0}, true);
    EXPECT_GT(view.t.x, 0);
    EXPECT_EQ(view.t - view.pivot, glm::vec3(0, 0, 5));
}

// Catches a navigation that ignores the scene's up axis or a roll that leaves a skewed rotation: the same steps on a
// Z-up scene give the Y-up result in its basis, and with the horizon free a roll tilts the right axis.
TEST(SpaceMouseNavigation, NavigationUpAxisAndRollPreference) {
    const auto basis = lfs::vis::navigationFrame(lfs::vis::NavigationUpAxis::DataZ);
    auto legacy = camera(), zup = camera();
    zup.setNavigationUpAxis(lfs::vis::NavigationUpAxis::DataZ);
    zup.R = basis * legacy.R;
    zup.t = basis * legacy.t;
    for (int i = 0; i < 100; ++i) {
        legacy.applySpaceMouseOrbit({0.5f, -1.0f}, {0.5f, -1.0f}, 0.0f, false);
        zup.applySpaceMouseOrbit({0.5f, -1.0f}, {0.5f, -1.0f}, 0.0f, false);
        legacy.applySpaceMouseFly({0.5f, -1.0f}, 0.0f, {0.001f, -0.002f, 0.001f}, true);
        zup.applySpaceMouseFly({0.5f, -1.0f}, 0.0f, {0.001f, -0.002f, 0.001f}, true);
    }
    EXPECT_LT(glm::length(zup.t - basis * legacy.t), 1e-4f);
    EXPECT_LT(glm::length(zup.R[1] - basis * legacy.R[1]), 1e-4f);
    for (const bool fly : {false, true}) {
        auto rolled = camera();
        if (fly)
            rolled.applySpaceMouseFly({}, 200.0f, {}, false);
        else
            rolled.applySpaceMouseOrbit({}, {}, 200.0f, true);
        EXPECT_GT(std::abs(rolled.R[0].y), 0.1f);
        EXPECT_NEAR(glm::determinant(rolled.R), 1, 1e-5f);
        EXPECT_NEAR(glm::length(rolled.t - rolled.pivot), 5, 1e-5f);
    }
}

TEST(SpaceMouseSession, HotPlugReconnectAndIndependentMotionExpiry) {
    RecordedSource source;
    source.attached = false;
    SpaceMouseSession session(source);
    const auto start = SpaceMouseSession::Clock::time_point(1s);
    EXPECT_FALSE(session.poll(start));
    source.attached = true;
    EXPECT_FALSE(session.poll(start + 1999ms));
    EXPECT_TRUE(session.poll(start + 2s));
    EXPECT_EQ(source.scans, 2);
    const auto recording = capture();
    for (const auto& report : recording)
        if (report.milliseconds >= 8400 && report.milliseconds < 8420)
            source.reports.push_back(report.bytes);
    ASSERT_FALSE(source.reports.empty());
    while (!source.reports.empty())
        session.poll(start + 2010ms);
    const auto moved = session.takeFrame();
    EXPECT_NE(moved.motion.translation, glm::vec3(0));
    EXPECT_NE(moved.motion.rotation, glm::vec3(0));
    source.reports.push_back({3, 2, 0});
    session.poll(start + 2200ms);
    session.poll(start + 2270ms);
    const auto expired = session.takeFrame();
    EXPECT_EQ(expired.motion, SpaceMouseMotion{});
    EXPECT_EQ(expired.pressed, 2);
    source.failed = true;
    EXPECT_TRUE(session.poll(start + 2300ms));
    EXPECT_FALSE(session.takeFrame().connection);
    EXPECT_EQ(source.closes, 1);
    source.failed = false;
    EXPECT_FALSE(session.poll(start + 4299ms));
    EXPECT_TRUE(session.poll(start + 4300ms));
    EXPECT_EQ(session.takeFrame().buttons, 0);
}

TEST(SpaceMouseSession, ShortPressBetweenFramesIsLatchedOnce) {
    RecordedSource source;
    SpaceMouseSession session(source);
    auto now = SpaceMouseSession::Clock::time_point(1s);
    session.poll(now);
    for (const auto& report : capture()) {
        if (report.bytes[0] == 3) {
            source.reports.push_back(report.bytes);
            session.poll(now += 1ms);
        }
    }
    const auto frame = session.takeFrame();
    EXPECT_EQ(frame.buttons, 0);
    EXPECT_EQ(frame.pressed, 3);
    EXPECT_EQ(session.takeFrame().pressed, 0);
}

TEST(SpaceMouseSession, TranslationReportsDoNotKeepOldRotationAlive) {
    RecordedSource source;
    SpaceMouseSession session(source);
    const auto start = SpaceMouseSession::Clock::time_point(1s);
    session.poll(start);
    const auto reports = capture();
    std::vector<std::uint8_t> translation;
    for (const auto& report : reports) {
        if (report.milliseconds >= 8400 && report.milliseconds < 8420) {
            source.reports.push_back(report.bytes);
            if (report.bytes[0] == 1)
                translation = report.bytes;
        }
    }
    ASSERT_FALSE(translation.empty());
    while (!source.reports.empty())
        session.poll(start + 10ms);
    source.reports.push_back(translation);
    session.poll(start + 200ms);
    session.poll(start + 270ms);
    const auto frame = session.takeFrame();
    EXPECT_NE(frame.motion.translation, glm::vec3(0));
    EXPECT_EQ(frame.motion.rotation, glm::vec3(0));
}

TEST(SpaceMouseNavigation, TranslationScalesWithFrameTimeAndNavigationSpeed) {
    const SpaceMouseMotion input{{0.6f, -0.2f, -0.5f}, {0, 0, 0}};
    auto coarse = camera(), fine = camera(), faster = camera();
    faster.setWasdSpeed(coarse.getWasdSpeed() * 2);
    const auto initial = coarse.t;
    for (int i = 0; i < 60; ++i) {
        const auto step = spaceMouseStep(input, 1, 1, 1.0f / 60, false, true);
        coarse.applySpaceMouseFly({}, 0.0f, step.translation, true);
        faster.applySpaceMouseFly({}, 0.0f, step.translation, true);
    }
    for (int i = 0; i < 120; ++i) {
        const auto step = spaceMouseStep(input, 1, 1, 1.0f / 120, false, true);
        fine.applySpaceMouseFly({}, 0.0f, step.translation, true);
    }
    EXPECT_LT(glm::length(coarse.t - fine.t), 1e-4f);
    EXPECT_LT(glm::length((coarse.t - initial) * 2.0f - (faster.t - initial)), 1e-4f);
}
