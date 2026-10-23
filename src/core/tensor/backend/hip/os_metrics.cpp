/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "os_metrics.hpp"
#include "core/native_fs.hpp"
#include "hip_backend_ops.hpp"
#include "physical_identity.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <core/tensor_backend.hpp>
#include <cstring>
#include <cwchar>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <dxgi1_4.h>
#include <pdh.h>
#include <pdhmsg.h>
#endif

namespace lfs::core {
    namespace {
        using Clock = std::chrono::steady_clock;
#ifdef _WIN32
        bool valid_luid(const LfsHipDeviceIdentity& id) {
            return std::any_of(std::begin(id.luid), std::end(id.luid), [](auto b) { return b != 0; });
        }
        struct BudgetState {
            std::mutex mutex;
            IDXGIAdapter3* adapter = nullptr;
            LUID luid{};
            UINT node = 0;
            bool initialized = false;
            Clock::time_point bound_at{};
            Clock::time_point sampled{};
            DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
            bool available = false;
            // Keep the COM reference at process teardown: the graphics runtime
            // can already be unloaded during static destruction.
            void bind(const LfsHipDeviceIdentity& id) {
                LUID selected{};
                std::memcpy(&selected, id.luid, sizeof(selected));
                const UINT selected_node = id.device_node_mask ? std::countr_zero(id.device_node_mask) : 0;
                if (initialized && selected.LowPart == luid.LowPart && selected.HighPart == luid.HighPart && node == selected_node &&
                    (adapter || Clock::now() - bound_at < std::chrono::seconds(5)))
                    return;
                if (adapter) adapter->Release();
                adapter = nullptr; luid = selected; node = selected_node;
                initialized = true; bound_at = Clock::now(); sampled = {}; available = false;
                if (!valid_luid(id)) return;
                IDXGIFactory4* factory = nullptr;
                if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory)))) return;
                factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter));
                factory->Release();
            }
        };
        struct UtilizationState {
            std::mutex mutex;
            PDH_HQUERY query = nullptr;
            PDH_HCOUNTER counter = nullptr;
            Clock::time_point retry_at{};
            Clock::time_point sampled{};
            std::array<uint8_t, 8> luid{};
            uint32_t node_mask = 0;
            float cached = -1.f;
            ~UtilizationState() { if (query) PdhCloseQuery(query); }
            float sample(const LfsHipDeviceIdentity& id) {
                std::lock_guard lock(mutex);
                if (!valid_luid(id)) return -1.f;
                const auto now = Clock::now();
                const bool same_device = std::equal(luid.begin(), luid.end(), id.luid) && node_mask == id.device_node_mask;
                if (same_device && now - sampled < std::chrono::milliseconds(500)) return cached;
                std::copy_n(id.luid, luid.size(), luid.begin()); node_mask = id.device_node_mask;
                sampled = now; cached = -1.f;
                if (!counter) {
                    if (now < retry_at) return cached;
                    retry_at = now + std::chrono::seconds(5);
                    if (query) { PdhCloseQuery(query); query = nullptr; }
                    if (PdhOpenQueryW(nullptr, 0, &query) != ERROR_SUCCESS) return cached;
                    if (PdhAddEnglishCounterW(query, L"\\GPU Engine(*)\\Utilization Percentage", 0, &counter) != ERROR_SUCCESS) return cached;
                    // A rate needs two samples. The worker calls again later;
                    // never sleep or block a rendering/training queue here.
                    PdhCollectQueryData(query);
                    return cached;
                }
                if (!counter || PdhCollectQueryData(query) != ERROR_SUCCESS) return cached;
                DWORD bytes = 0, count = 0;
                if (PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &bytes, &count, nullptr) != PDH_MORE_DATA || !bytes) return cached;
                std::vector<std::max_align_t> storage((bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t));
                auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(storage.data());
                if (PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &bytes, &count, items) != ERROR_SUCCESS) return cached;
                LUID selected{}; std::memcpy(&selected, id.luid, sizeof(selected));
                const unsigned node = id.device_node_mask ? std::countr_zero(id.device_node_mask) : 0;
                std::unordered_map<unsigned, double> engines;
                for (DWORD i = 0; i < count; ++i) {
                    const auto& item = items[i];
                    if (!item.szName || (item.FmtValue.CStatus != PDH_CSTATUS_VALID_DATA && item.FmtValue.CStatus != PDH_CSTATUS_NEW_DATA)) continue;
                    const auto* name = std::wcsstr(item.szName, L"_luid_");
                    unsigned high = 0, low = 0, physical = 0, engine = 0;
                    if (!name || std::swscanf(name, L"_luid_0x%x_0x%x_phys_%u_eng_%u", &high, &low, &physical, &engine) != 4 ||
                        high != static_cast<unsigned>(selected.HighPart) || low != selected.LowPart || physical != node) continue;
                    const double value = item.FmtValue.doubleValue;
                    if (std::isfinite(value) && value >= 0) engines[engine] += value;
                }
                // Sum clients of each engine, then select the busiest engine.
                // Adding independent engines would misreport more than 100%.
                for (const auto& [engine, value] : engines) cached = std::max(cached, static_cast<float>(std::clamp(value, 0., 100.)));
                return cached;
            }
        };
#endif
    }
    namespace internal {
        void hip_process_budget(const LfsHipDeviceIdentity& id, GpuDeviceInfo& output) {
#ifdef _WIN32
            static BudgetState state;
            std::lock_guard lock(state.mutex);
            state.bind(id);
            const auto now = Clock::now();
            if (state.adapter && now - state.sampled >= std::chrono::milliseconds(500)) {
                state.sampled = now;
                // LOCAL is the OS-managed local segment, including UMA. Do not
                // add NON_LOCAL or the BIOS reservation to this process budget.
                state.available = SUCCEEDED(state.adapter->QueryVideoMemoryInfo(state.node, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &state.memory)) && state.memory.Budget > 0;
            }
            if (state.available) {
                output.supports_process_memory_budget = true;
                output.process_memory_budget_bytes = static_cast<size_t>(state.memory.Budget);
                output.process_memory_used_bytes = static_cast<size_t>(state.memory.CurrentUsage);
            }
#else
            (void)id; (void)output;
#endif
        }
    }
    float hip_gpu_utilization() {
        try {
            const int ordinal = selected_hip_device();
#ifdef _WIN32
            static UtilizationState state;
            return state.sample(internal::hip_device_identity(LFS_HIP_RUNTIME_DEFAULT_MODULE, ordinal));
#elif defined(__linux__)
            const auto info = internal::hip_device_info(LFS_HIP_RUNTIME_DEFAULT_MODULE, ordinal);
            const auto pci = internal::pci_physical_id(info.pci_bus_id);
            if (pci.empty()) return -1.f;
            std::ifstream input(lfs::core::native_fs::native_path("/sys/bus/pci/devices/" + pci.substr(4) + "/gpu_busy_percent"));
            int percent = -1;
            if (input >> percent && percent >= 0 && percent <= 100) return static_cast<float>(percent);
#endif
        } catch (...) { /* LFS-CENSUS-OK(empty-catch): Optional telemetry returns unavailable instead of a different device sample. */  }
        return -1.f;
    }
}
