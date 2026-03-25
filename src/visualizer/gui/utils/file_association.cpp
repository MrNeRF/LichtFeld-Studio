/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/utils/file_association.hpp"

#ifdef _WIN32
#include <array>
#include <core/executable_path.hpp>
#include <core/logger.hpp>
#include <shlobj.h>
#include <string>
#endif

namespace lfs::vis::gui {

#ifdef _WIN32

    namespace {

        struct ExtInfo {
            const wchar_t* ext;
            const wchar_t* prog_id;
            const wchar_t* friendly_name;
        };

        constexpr std::array<ExtInfo, 7> EXTENSIONS = {{
            {L".ply", L"LichtFeldStudio.ply", L"PLY Point Cloud"},
            {L".sog", L"LichtFeldStudio.sog", L"SOG Gaussian Splat"},
            {L".spz", L"LichtFeldStudio.spz", L"SPZ Gaussian Splat"},
            {L".usd", L"LichtFeldStudio.usd", L"USD Gaussian Splat"},
            {L".usda", L"LichtFeldStudio.usda", L"USDA Gaussian Splat"},
            {L".usdc", L"LichtFeldStudio.usdc", L"USDC Gaussian Splat"},
            {L".usdz", L"LichtFeldStudio.usdz", L"USDZ Gaussian Splat"},
        }};

        bool setRegString(HKEY parent, const std::wstring& subkey, const std::wstring& value_name,
                          const std::wstring& data) {
            HKEY key;
            LONG res = RegCreateKeyExW(parent, subkey.c_str(), 0, nullptr, 0, KEY_SET_VALUE, nullptr, &key, nullptr);
            if (res != ERROR_SUCCESS)
                return false;
            res = RegSetValueExW(key, value_name.empty() ? nullptr : value_name.c_str(), 0, REG_SZ,
                                 reinterpret_cast<const BYTE*>(data.c_str()),
                                 static_cast<DWORD>((data.size() + 1) * sizeof(wchar_t)));
            RegCloseKey(key);
            return res == ERROR_SUCCESS;
        }

        bool getRegString(HKEY parent, const std::wstring& subkey, const std::wstring& value_name,
                          std::wstring& out) {
            HKEY key;
            if (RegOpenKeyExW(parent, subkey.c_str(), 0, KEY_READ, &key) != ERROR_SUCCESS)
                return false;
            DWORD type = 0, size = 0;
            if (RegQueryValueExW(key, value_name.empty() ? nullptr : value_name.c_str(), nullptr, &type, nullptr,
                                 &size) != ERROR_SUCCESS ||
                type != REG_SZ) {
                RegCloseKey(key);
                return false;
            }
            out.resize(size / sizeof(wchar_t));
            RegQueryValueExW(key, value_name.empty() ? nullptr : value_name.c_str(), nullptr, nullptr,
                             reinterpret_cast<BYTE*>(out.data()), &size);
            RegCloseKey(key);
            while (!out.empty() && out.back() == L'\0')
                out.pop_back();
            return true;
        }

        bool deleteRegTree(HKEY parent, const std::wstring& subkey) {
            return RegDeleteTreeW(parent, subkey.c_str()) == ERROR_SUCCESS;
        }

    } // namespace

    bool registerFileAssociations() {
        const auto exe_path = lfs::core::getExecutablePath().wstring();
        const auto command = L"\"" + exe_path + L"\" \"%1\"";
        const auto icon = exe_path + L",0";
        const auto classes = std::wstring(L"Software\\Classes\\");

        bool ok = true;
        for (const auto& ext : EXTENSIONS) {
            const auto prog_key = classes + ext.prog_id;
            ok &= setRegString(HKEY_CURRENT_USER, prog_key, L"", ext.friendly_name);
            ok &= setRegString(HKEY_CURRENT_USER, prog_key + L"\\DefaultIcon", L"", icon);
            ok &= setRegString(HKEY_CURRENT_USER, prog_key + L"\\shell\\open\\command", L"", command);

            const auto ext_key = classes + ext.ext;
            ok &= setRegString(HKEY_CURRENT_USER, ext_key, L"", ext.prog_id);
            ok &= setRegString(HKEY_CURRENT_USER, ext_key + L"\\OpenWithProgids", ext.prog_id, L"");
        }

        SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);

        if (ok)
            LOG_INFO("File associations registered successfully");
        else
            LOG_WARN("Some file associations failed to register");

        return ok;
    }

    bool unregisterFileAssociations() {
        const auto classes = std::wstring(L"Software\\Classes\\");
        bool ok = true;

        for (const auto& ext : EXTENSIONS) {
            ok &= deleteRegTree(HKEY_CURRENT_USER, classes + ext.prog_id);

            std::wstring current_default;
            const auto ext_key = classes + ext.ext;
            if (getRegString(HKEY_CURRENT_USER, ext_key, L"", current_default) &&
                current_default == ext.prog_id) {
                HKEY def_key;
                if (RegOpenKeyExW(HKEY_CURRENT_USER, ext_key.c_str(), 0, KEY_SET_VALUE, &def_key) == ERROR_SUCCESS) {
                    RegDeleteValueW(def_key, nullptr);
                    RegCloseKey(def_key);
                }
            }

            HKEY owp_key;
            const auto owp_path = ext_key + L"\\OpenWithProgids";
            if (RegOpenKeyExW(HKEY_CURRENT_USER, owp_path.c_str(), 0, KEY_SET_VALUE, &owp_key) == ERROR_SUCCESS) {
                RegDeleteValueW(owp_key, ext.prog_id);
                RegCloseKey(owp_key);
            }
        }

        SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);
        LOG_INFO("File associations unregistered");
        return ok;
    }

    bool areFileAssociationsRegistered() {
        const auto classes = std::wstring(L"Software\\Classes\\");
        for (const auto& ext : EXTENSIONS) {
            std::wstring current;
            if (!getRegString(HKEY_CURRENT_USER, classes + ext.ext, L"", current) || current != ext.prog_id)
                return false;
        }
        return true;
    }

#else

    bool registerFileAssociations() { return false; }
    bool unregisterFileAssociations() { return false; }
    bool areFileAssociationsRegistered() { return false; }

#endif

} // namespace lfs::vis::gui
