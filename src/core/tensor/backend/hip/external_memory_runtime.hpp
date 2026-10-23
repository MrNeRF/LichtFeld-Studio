/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "external_memory_api.h"
#ifndef _WIN32
#include <unistd.h>
#endif
extern "C" int lfs_hip_buffer_import_external(LfsHipContext* context,
                                              const LfsHipExternalMemory* p,
                                              LfsHipBuffer** output,
                                              LfsHipError* error) {
    return boundary(error, [&] {
        require(output, "Missing HIP external buffer output");
        *output = nullptr;
        require(context && p && p->struct_bytes == sizeof(*p), "Invalid HIP external memory descriptor");
        require(p->allocation_bytes && p->allocation_bytes <= SIZE_MAX && p->mapped_bytes &&
                    p->mapped_bytes <= p->allocation_bytes && p->dedicated <= 1 && p->reserved == 0,
                "Invalid HIP external allocation range or flags");
        hipExternalMemoryHandleDesc desc{};
        desc.size = p->allocation_bytes;
        desc.flags = p->dedicated ? hipExternalMemoryDedicated : 0;
#ifdef _WIN32
        require(p->handle_type == LFS_HIP_EXTERNAL_WIN32 && p->handle &&
                    p->handle != UINT64_MAX && p->handle <= UINTPTR_MAX,
                "HIP external import requires a Win32 NT handle");
        desc.type = hipExternalMemoryHandleTypeOpaqueWin32;
        desc.handle.win32.handle = reinterpret_cast<void*>(static_cast<uintptr_t>(p->handle));
#else
        require(p->handle_type == LFS_HIP_EXTERNAL_FD && p->handle <= INT32_MAX,
                "HIP external import requires a valid file descriptor");
        desc.type = hipExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = static_cast<int>(p->handle);
#endif
        DeviceScope selected(context->device);
        auto buffer = std::make_unique<LfsHipBuffer>();
        buffer->bytes = static_cast<size_t>(p->mapped_bytes);
        context->buffers.insert(buffer.get());
        try {
#ifndef _WIN32
            desc.handle.fd = dup(desc.handle.fd);
            require(desc.handle.fd >= 0, "Cannot duplicate HIP external memory descriptor");
#endif
            const auto imported = hipImportExternalMemory(&buffer->external_memory, &desc);
#ifndef _WIN32
            if (imported != hipSuccess) (void)close(desc.handle.fd);
#endif
            check(imported);
            hipExternalMemoryBufferDesc mapping{};
            mapping.size = p->mapped_bytes;
            check(hipExternalMemoryGetMappedBuffer(&buffer->data, buffer->external_memory, &mapping));
            require(buffer->data != nullptr, "HIP external mapping returned a null address");
        } catch (...) {
            try {
                release_buffer_resources(buffer.get());
                context->buffers.erase(buffer.get());
            } catch (...) { /* LFS-CENSUS-OK(empty-catch): Retain the registered importer for context teardown and propagate the outer ABI failure. */
                // Preserve the registered handle for context teardown; the exporter
                // must remain alive when the import could not be safely destroyed.
                *output = buffer.release();
                std::fprintf(stderr, "[error] HIP failed external import cleanup; native context retains the import\n");
            }
            throw;
        }
        *output = buffer.release();
    });
}
