# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
# Standalone fixture checks: no compiler, CUDA, GUI or vcpkg required.
cmake_minimum_required(VERSION 3.24)
project(LichtFeldMediaFixtureTests LANGUAGES NONE)
enable_testing()
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
find_program(LFS_MEDIA_TEST_FFMPEG NAMES ffmpeg REQUIRED)
find_program(LFS_MEDIA_TEST_FFPROBE NAMES ffprobe REQUIRED)
add_test(NAME MediaFixturePreparation
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_fixture_preparation.py")
set_tests_properties(MediaFixturePreparation PROPERTIES
    LABELS "media;cpu;offline" TIMEOUT 180
    ENVIRONMENT "LFS_MEDIA_TEST_FFMPEG=${LFS_MEDIA_TEST_FFMPEG};LFS_MEDIA_TEST_FFPROBE=${LFS_MEDIA_TEST_FFPROBE}")
