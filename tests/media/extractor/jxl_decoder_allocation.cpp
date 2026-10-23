// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <jxl/decode.h>
#include <new>

namespace {
    size_t remaining = 0;
    bool injected = false;
    struct Memory {
        size_t live = 0;
        static void* allocate(void* opaque, size_t bytes) {
            auto* pointer = std::malloc(bytes ? bytes : 1);
            if (pointer)
                ++static_cast<Memory*>(opaque)->live;
            return pointer;
        }
        static void release(void* opaque, void* pointer) {
            if (pointer) {
                --static_cast<Memory*>(opaque)->live;
                std::free(pointer);
            }
        }
    };
} // namespace

// ELF executable interposition reaches C++ allocations in the linked provider.
// The custom JXL manager separately verifies raw decoder ownership on throws.
void* operator new(size_t bytes) {
    using Allocate = void* (*)(size_t);
    static auto original = reinterpret_cast<Allocate>(dlsym(RTLD_NEXT, "_Znwm"));
    if (!original)
        std::abort();
    if (remaining && --remaining == 0) {
        injected = true;
        throw std::bad_alloc();
    }
    return original(bytes);
}

int main() {
    size_t failures = 0;
    for (size_t position = 1; position < 1024; ++position) {
        Memory memory;
        const JxlMemoryManager manager{&memory, Memory::allocate, Memory::release};
        injected = false;
        remaining = position;
        try {
            auto* decoder = JxlDecoderCreate(&manager);
            remaining = 0;
            if (!decoder)
                return 1;
            JxlDecoderDestroy(decoder);
        } catch (const std::bad_alloc&) {
            remaining = 0;
            if (!injected)
                return 2;
            ++failures;
        } catch (...) {
            remaining = 0;
            return 3;
        }
        if (memory.live) {
            std::cerr << "Decoder storage leaked at C++ allocation " << position << '\n';
            return 4;
        }
        if (!injected) {
            if (!failures)
                return 5;
            std::cout << failures << " injected constructor/setup failures; recovery succeeded\n";
            return 0;
        }
    }
    return 6;
}
