/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/path_utils.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <nfd.h>

namespace {
    std::string selected_path;
    std::string suggested_name;
    nfdresult_t dialog_result = NFD_OKAY;
} // namespace

// Substitute only the OS picker: exercise production request construction and
// result handling without a display or user interaction.
extern "C" nfdresult_t __wrap_NFD_Init() { return NFD_OKAY; }
extern "C" void __wrap_NFD_Quit() {}
extern "C" nfdresult_t __wrap_NFD_SaveDialogU8_With_Impl(nfdversion_t, nfdu8char_t** out, const nfdsavedialogu8args_t* args) {
    suggested_name = args->defaultName ? args->defaultName : "";
    if (dialog_result == NFD_OKAY) {
        *out = static_cast<nfdu8char_t*>(std::malloc(selected_path.size() + 1));
        std::memcpy(*out, selected_path.c_str(), selected_path.size() + 1);
    }
    return dialog_result;
}

TEST(NativeSaveDialogTest, AllSaveDialogsPreserveSupportedSuffixes) {
    using namespace lfs::vis::gui;
    using Save = std::filesystem::path (*)(const std::string&, const std::filesystem::path&);
    const std::pair<Save, std::string> dialogs[] = {
        {SavePlyFileDialog, ".ply"},
        {SaveSogFileDialog, ".sog"},
        {SaveSpzFileDialog, ".spz"},
        {SaveRadFileDialog, ".rad"},
        {SaveProjectFileDialog, ".licht"},
        {SaveJsonFileDialog, ".json"},
        {SaveGlbFileDialog, ".glb"},
        {SaveLasFileDialog, ".las"},
        {SaveLazFileDialog, ".laz"},
        {SavePngFileDialog, ".png"},
        {SaveJpgFileDialog, ".jpg"},
        {SaveTextFileDialog, ".txt"},
        {SaveUsdFileDialog, ".usd"},
        {SaveUsdzFileDialog, ".usdz"},
        {SaveHtmlFileDialog, ".html"},
        {SaveMp4FileDialog, ".mp4"},
        {SavePythonFileDialog, ".py"}};
    for (const auto& [save, extension] : dialogs) {
        std::string upper = extension;
        for (char& c : upper)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        for (const auto& suffix : {extension, upper, std::string{}, std::string{".unsupported"}}) {
            SCOPED_TRACE(extension + " / " + suffix);
            selected_path = "/tmp/cloud-context" + suffix;
            const std::string expected_suffix = (suffix == extension || suffix == upper) ? suffix : suffix + extension;
            const auto result = save("cloud-context" + suffix, {});
            EXPECT_EQ(lfs::core::path_to_utf8(result), "/tmp/cloud-context" + expected_suffix);
            EXPECT_EQ(suggested_name, "cloud-context" + ((suffix == extension || suffix == upper) ? std::string{} : suffix));
        }
    }
    selected_path = "/tmp/cloud-context.SSOG";
    EXPECT_EQ(SaveSsogFileDialog("cloud-context.SSOG"), selected_path);
    EXPECT_EQ(suggested_name, "cloud-context");
}

TEST(NativeSaveDialogTest, CancelDoesNotCreateADestination) {
    dialog_result = NFD_CANCEL;
    EXPECT_TRUE(lfs::vis::gui::SavePlyFileDialog("cloud-context").empty());
    dialog_result = NFD_OKAY;
}
