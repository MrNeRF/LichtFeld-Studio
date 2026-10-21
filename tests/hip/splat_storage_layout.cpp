/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../src/training/hip/splat_storage_layout.hpp"
#include <cstdio>
#include <limits>

int main() {
    using Storage = lfs::core::SplatExportableStorage;
    Storage::Control control;
    control.block = std::make_shared<lfs::core::ExportableBlock>();
    control.block->device_ptr = reinterpret_cast<void*>(uintptr_t(0x100000)); // Never dereferenced.
    control.block->reserved_bytes = 32 * 65536;
    control.capacity = 1024;
    const size_t order[] = {Storage::Means, Storage::Scaling, Storage::Rotation, Storage::Opacity, Storage::Sh0};
    for (size_t i = 0; i < 5; ++i) {
        control.region_offsets[order[i]] = (i + 1) * 65536;
        control.region_bytes[order[i]] = 65536;
    }
    const auto valid = control;
    unsigned rejected = 0;
    auto reject = [&](size_t count = 263) {
        try {
            (void)lfs::training::hip::splat_storage_layout(control, count);
            throw std::runtime_error("Invalid core storage accepted");
        } catch (const std::invalid_argument&) {
            ++rejected;
        }
        control = valid;
    };
    const auto layout = lfs::training::hip::splat_storage_layout(control, 263);
    const size_t expected[] = {1, 3, 2, 5, 4};
    const size_t attributes[] = {3, 4, 3, 3, 1};
    for (size_t i = 0; i < 5; ++i)
        if (layout.offsets_bytes[i] != expected[i] * 65536 || layout.float_counts[i] != 263 * attributes[i])
            return 1;
    control.poisoned = true;
    reject();
    control.block.reset();
    reject();
    reject(0);
    reject(1025);
    control.region_offsets[Storage::Rotation] = 4;
    reject();
    control.region_bytes[Storage::Scaling] = 4;
    reject();
    control.region_offsets[Storage::Opacity] = control.region_offsets[Storage::Sh0];
    reject();
    control.region_bytes[Storage::Opacity] = std::numeric_limits<size_t>::max();
    reject();
    control.region_bytes[Storage::ShN] = 256;
    reject();
    control.region_offsets[Storage::ShN] = 6 * 65536;
    control.region_bytes[Storage::ShN] = 65536;
    (void)lfs::training::hip::splat_storage_layout(control, 263, true);
    const auto valid_sh = control;
    unsigned sh_rejected = 0;
    auto reject_sh = [&] {
        try {
            (void)lfs::training::hip::splat_storage_layout(control, 263, true);
            throw std::runtime_error("Invalid FP32 SH storage accepted");
        } catch (const std::invalid_argument&) { ++sh_rejected; }
        control = valid_sh;
    };
    control.region_offsets[Storage::ShN] = control.region_offsets[Storage::Means] + 32768;
    reject_sh(); // Reserved base capacity, even beyond the live rows.
    control.region_offsets[Storage::ShN] += 4;
    reject_sh();
    control.region_bytes[Storage::ShN] = std::numeric_limits<size_t>::max();
    reject_sh();
    control.region_bytes[Storage::ShNBounds] = 256;
    reject_sh(); // A compressed layout is not an FP32 view.
    control = valid;
    // Copy-on-write block metadata so the valid reference remains intact.
    control.block = std::make_shared<lfs::core::ExportableBlock>(*control.block);
    control.block->chunks.push_back({0, 65536, {reinterpret_cast<void*>(uintptr_t(1)), 65536}});
    reject();
    std::printf("PASS actual core storage layout mapping and %u invalid contracts\n", rejected);
    return rejected == 10 && sh_rejected == 4 ? 0 : 1;
}
