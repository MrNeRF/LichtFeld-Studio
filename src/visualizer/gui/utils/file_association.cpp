/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/utils/file_association.hpp"

#ifdef _WIN32
#include <array>
#include <core/executable_path.hpp>
#include <core/logger.hpp>
#include <memory>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>
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

        constexpr wchar_t REGISTERED_APP_NAME[] = L"LichtFeld Studio";
        constexpr wchar_t CAPABILITIES_PATH[] = L"Software\\LichtFeldStudio\\Capabilities";
        constexpr wchar_t CAPABILITIES_FILE_ASSOCIATIONS_PATH[] =
            L"Software\\LichtFeldStudio\\Capabilities\\FileAssociations";
        constexpr wchar_t REGISTERED_APPLICATIONS_PATH[] = L"Software\\RegisteredApplications";
        constexpr wchar_t APPLICATION_DESCRIPTION[] =
            L"LichtFeld Studio supports PLY, SOG, SPZ, USD, USDA, USDC, and USDZ splat files.";

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
            LONG res = RegCreateKeyExW(parent, subkey.c_str(), 0, nullptr, 0, KEY_SET_VALUE,
                                       nullptr, &key, nullptr);
            if (res != ERROR_SUCCESS)
                return false;
            res = RegSetValueExW(key, value_name.empty() ? nullptr : value_name.c_str(), 0,
                                 REG_SZ, reinterpret_cast<const BYTE*>(data.c_str()),
                                 static_cast<DWORD>((data.size() + 1) * sizeof(wchar_t)));
            RegCloseKey(key);
            return res == ERROR_SUCCESS;
        }

        bool getRegString(HKEY parent, const std::wstring& subkey, const std::wstring& value_name,
                          std::wstring& out) {
            HKEY key;
            if (RegOpenKeyExW(parent, subkey.c_str(), 0, KEY_READ, &key) != ERROR_SUCCESS)
                return false;
            DWORD type = 0;
            DWORD size = 0;
            if (RegQueryValueExW(key, value_name.empty() ? nullptr : value_name.c_str(), nullptr,
                                 &type, nullptr, &size) != ERROR_SUCCESS ||
                type != REG_SZ) {
                RegCloseKey(key);
                return false;
            }
            out.resize(size / sizeof(wchar_t));
            RegQueryValueExW(key, value_name.empty() ? nullptr : value_name.c_str(), nullptr,
                             nullptr, reinterpret_cast<BYTE*>(out.data()), &size);
            RegCloseKey(key);
            while (!out.empty() && out.back() == L'\0')
                out.pop_back();
            return true;
        }

        bool deleteRegTree(HKEY parent, const std::wstring& subkey) {
            const LONG res = RegDeleteTreeW(parent, subkey.c_str());
            return res == ERROR_SUCCESS || res == ERROR_FILE_NOT_FOUND ||
                   res == ERROR_PATH_NOT_FOUND;
        }

        bool deleteRegValue(HKEY parent, const std::wstring& subkey,
                            const std::wstring& value_name) {
            HKEY key;
            const LONG open_res = RegOpenKeyExW(parent, subkey.c_str(), 0, KEY_SET_VALUE, &key);
            if (open_res == ERROR_FILE_NOT_FOUND || open_res == ERROR_PATH_NOT_FOUND)
                return true;
            if (open_res != ERROR_SUCCESS)
                return false;

            const LONG delete_res =
                RegDeleteValueW(key, value_name.empty() ? nullptr : value_name.c_str());
            RegCloseKey(key);
            return delete_res == ERROR_SUCCESS || delete_res == ERROR_FILE_NOT_FOUND;
        }

        struct ComRelease {
            void operator()(IUnknown* p) const {
                if (p)
                    p->Release();
            }
        };

        template <typename T>
        using ComPtr = std::unique_ptr<T, ComRelease>;

        class CoInitScope {
        public:
            explicit CoInitScope(const DWORD flags) : hr_(CoInitializeEx(nullptr, flags)) {}

            ~CoInitScope() {
                if (SUCCEEDED(hr_))
                    CoUninitialize();
            }

            [[nodiscard]] bool ready() const {
                return SUCCEEDED(hr_) || hr_ == RPC_E_CHANGED_MODE;
            }

        private:
            HRESULT hr_;
        };

        std::wstring uriEscapeComponent(const std::wstring& value) {
            std::wstring escaped;
            escaped.reserve(value.size() * 3);

            const auto hex = [](const unsigned int value) -> wchar_t {
                return static_cast<wchar_t>(value < 10 ? L'0' + value : L'A' + (value - 10));
            };

            for (const wchar_t ch : value) {
                const bool is_unreserved =
                    (ch >= L'a' && ch <= L'z') || (ch >= L'A' && ch <= L'Z') ||
                    (ch >= L'0' && ch <= L'9') || ch == L'-' || ch == L'_' ||
                    ch == L'.' || ch == L'~';
                if (is_unreserved) {
                    escaped.push_back(ch);
                    continue;
                }

                const unsigned int byte = static_cast<unsigned int>(ch) & 0xFFu;
                escaped.push_back(L'%');
                escaped.push_back(hex((byte >> 4) & 0xFu));
                escaped.push_back(hex(byte & 0xFu));
            }

            return escaped;
        }

        bool launchUri(const std::wstring& uri) {
            const auto result = reinterpret_cast<intptr_t>(
                ShellExecuteW(nullptr, L"open", uri.c_str(), nullptr, nullptr, SW_SHOWNORMAL));
            return result > 32;
        }

        bool queryEffectiveProgId(IApplicationAssociationRegistration* registration,
                                  const wchar_t* extension, std::wstring& out) {
            if (!registration)
                return false;

            LPWSTR current = nullptr;
            const HRESULT hr = registration->QueryCurrentDefault(extension, AT_FILEEXTENSION,
                                                                 AL_EFFECTIVE, &current);
            if (FAILED(hr) || !current)
                return false;

            out = current;
            CoTaskMemFree(current);
            return true;
        }

        bool queryUserChoiceProgId(const wchar_t* extension, std::wstring& out) {
            const auto user_choice_key =
                std::wstring(L"Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\FileExts\\") +
                extension + L"\\UserChoice";
            return getRegString(HKEY_CURRENT_USER, user_choice_key, L"ProgId", out);
        }

    } // namespace

    bool registerFileAssociations() {
        const auto exe_path = lfs::core::getExecutablePath();
        const auto exe_path_w = exe_path.wstring();
        const auto exe_name = exe_path.filename().wstring();
        const auto command = L"\"" + exe_path_w + L"\" \"%1\"";
        const auto icon = exe_path_w + L",0";
        const auto classes = std::wstring(L"Software\\Classes\\");
        const auto application_key = classes + L"Applications\\" + exe_name;

        bool ok = true;
        for (const auto& ext : EXTENSIONS) {
            const auto prog_key = classes + ext.prog_id;
            ok &= setRegString(HKEY_CURRENT_USER, prog_key, L"", ext.friendly_name);
            ok &= setRegString(HKEY_CURRENT_USER, prog_key + L"\\DefaultIcon", L"", icon);
            ok &= setRegString(HKEY_CURRENT_USER, prog_key + L"\\shell\\open\\command", L"",
                               command);

            const auto ext_key = classes + ext.ext;
            // Register as an available handler only. Windows should keep the effective
            // default app decision in the system-managed Default Apps flow.
            ok &= setRegString(HKEY_CURRENT_USER, ext_key + L"\\OpenWithProgids", ext.prog_id,
                               L"");
            ok &= setRegString(HKEY_CURRENT_USER, application_key + L"\\SupportedTypes", ext.ext,
                               L"");
            ok &= setRegString(HKEY_CURRENT_USER, CAPABILITIES_FILE_ASSOCIATIONS_PATH, ext.ext,
                               ext.prog_id);
        }

        ok &= setRegString(HKEY_CURRENT_USER, CAPABILITIES_PATH, L"ApplicationName",
                           REGISTERED_APP_NAME);
        ok &= setRegString(HKEY_CURRENT_USER, CAPABILITIES_PATH, L"ApplicationDescription",
                           APPLICATION_DESCRIPTION);
        ok &= setRegString(HKEY_CURRENT_USER, REGISTERED_APPLICATIONS_PATH,
                           REGISTERED_APP_NAME, CAPABILITIES_PATH);
        ok &= setRegString(HKEY_CURRENT_USER, application_key, L"FriendlyAppName",
                           REGISTERED_APP_NAME);
        ok &= setRegString(HKEY_CURRENT_USER, application_key + L"\\DefaultIcon", L"", icon);
        ok &= setRegString(HKEY_CURRENT_USER, application_key + L"\\shell\\open\\command", L"",
                           command);

        SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);

        if (ok)
            LOG_INFO("File associations registered successfully");
        else
            LOG_WARN("Some file associations failed to register");

        return ok;
    }

    bool unregisterFileAssociations() {
        const auto exe_path = lfs::core::getExecutablePath();
        const auto exe_name = exe_path.filename().wstring();
        const auto classes = std::wstring(L"Software\\Classes\\");
        bool ok = true;

        for (const auto& ext : EXTENSIONS) {
            ok &= deleteRegTree(HKEY_CURRENT_USER, classes + ext.prog_id);

            std::wstring current_default;
            const auto ext_key = classes + ext.ext;
            if (getRegString(HKEY_CURRENT_USER, ext_key, L"", current_default) &&
                current_default == ext.prog_id) {
                HKEY def_key;
                if (RegOpenKeyExW(HKEY_CURRENT_USER, ext_key.c_str(), 0, KEY_SET_VALUE,
                                  &def_key) == ERROR_SUCCESS) {
                    RegDeleteValueW(def_key, nullptr);
                    RegCloseKey(def_key);
                }
            }

            HKEY owp_key;
            const auto owp_path = ext_key + L"\\OpenWithProgids";
            if (RegOpenKeyExW(HKEY_CURRENT_USER, owp_path.c_str(), 0, KEY_SET_VALUE, &owp_key) ==
                ERROR_SUCCESS) {
                RegDeleteValueW(owp_key, ext.prog_id);
                RegCloseKey(owp_key);
            }
        }

        ok &= deleteRegTree(HKEY_CURRENT_USER, CAPABILITIES_FILE_ASSOCIATIONS_PATH);
        ok &= deleteRegTree(HKEY_CURRENT_USER, CAPABILITIES_PATH);
        ok &= deleteRegValue(HKEY_CURRENT_USER, REGISTERED_APPLICATIONS_PATH,
                             REGISTERED_APP_NAME);
        ok &= deleteRegTree(HKEY_CURRENT_USER, classes + L"Applications\\" + exe_name);

        SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);
        LOG_INFO("File associations unregistered");
        return ok;
    }

    bool areFileAssociationsRegistered() {
        CoInitScope coinit(COINIT_APARTMENTTHREADED);
        ComPtr<IApplicationAssociationRegistration> registration;
        if (coinit.ready()) {
            IApplicationAssociationRegistration* raw = nullptr;
            const HRESULT hr = CoCreateInstance(CLSID_ApplicationAssociationRegistration, nullptr,
                                                CLSCTX_INPROC_SERVER,
                                                IID_PPV_ARGS(&raw));
            if (SUCCEEDED(hr))
                registration.reset(raw);
        }

        const auto classes = std::wstring(L"Software\\Classes\\");
        for (const auto& ext : EXTENSIONS) {
            std::wstring current;
            if (!queryEffectiveProgId(registration.get(), ext.ext, current)) {
                if (!queryUserChoiceProgId(ext.ext, current) &&
                    !getRegString(HKEY_CURRENT_USER, classes + ext.ext, L"", current))
                    return false;
            }
            if (current != ext.prog_id)
                return false;
        }
        return true;
    }

    bool openFileAssociationSettings() {
        const auto deep_link =
            std::wstring(L"ms-settings:defaultapps?registeredAppUser=") +
            uriEscapeComponent(REGISTERED_APP_NAME);
        if (launchUri(deep_link))
            return true;
        if (launchUri(L"ms-settings:defaultapps"))
            return true;

        CoInitScope coinit(COINIT_APARTMENTTHREADED);
        if (!coinit.ready())
            return false;

        IApplicationAssociationRegistrationUI* raw = nullptr;
        const HRESULT hr =
            CoCreateInstance(CLSID_ApplicationAssociationRegistrationUI, nullptr,
                             CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&raw));
        if (FAILED(hr) || !raw)
            return false;

        ComPtr<IApplicationAssociationRegistrationUI> registration_ui(raw);
        return SUCCEEDED(registration_ui->LaunchAdvancedAssociationUI(REGISTERED_APP_NAME));
    }

#else

    bool registerFileAssociations() { return false; }
    bool unregisterFileAssociations() { return false; }
    bool areFileAssociationsRegistered() { return false; }
    bool openFileAssociationSettings() { return false; }

#endif

} // namespace lfs::vis::gui
