// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 80;

[SpecializationConstant] const uint kPhase = 0; // dense, indexed clear, indexed apply, initialize

struct Parameters {
    uint64_t output, selected, existing, locked, indices, categories, allowed;
    uint count, outputCount, allowedCount, label, mode, padding;
};

#include "byte_read.slangh"
bool eligible(Parameters p, uint row) {
    if (p.categories == 0) return true;
    const int category = ((Ptr<int, Access.Read>)p.categories)[row];
    if (category < 0 || uint(category) >= p.allowedCount) return false;
    return loadByte(p.allowed, uint(category)) != 0;
}
uint oldLabel(Parameters p, uint row) {
    if (p.existing == 0) return 0;
    return loadByte(p.existing, row);
}
uint updated(Parameters p, uint old, bool selected) {
    if (selected) {
        if (p.mode == 1) return old == p.label ? 0 : old;
        if (old != 0 && old != p.label && p.locked != 0) {
            if (loadByte(p.locked, old) != 0) return old;
        }
        return p.label;
    }
    return p.mode == 2 && old == p.label ? 0 : old;
}
void writeByte(uint64_t base, uint index, uint value) {
    const uint64_t address = base + index;
    Ptr<uint, Access.ReadWrite> word = (Ptr<uint, Access.ReadWrite>)(address & ~uint64_t(3));
    const uint shift = uint(address & uint64_t(3)) * 8;
    uint previous;
    InterlockedOr(word[0], 0u, previous);
    for (;;) {
        const uint desired = (previous & ~(255u << shift)) | (value << shift);
        if (desired == previous) return;
        uint actual;
        InterlockedCompareExchange(word[0], previous, desired, actual);
        if (actual == previous) return;
        previous = actual;
    }
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters p) {
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = groups.x * LFS_LOCAL_SIZE_X;
    if (kPhase == 0 || kPhase == 3) {
        const uint offset = uint(p.output & uint64_t(3));
        const uint words = (p.outputCount + offset + 3) / 4;
        Ptr<uint, Access.ReadWrite> output = (Ptr<uint, Access.ReadWrite>)(p.output & ~uint64_t(3));
        for (uint word = dispatchThreadId.x; word < words; word += stride) {
            uint value = output[word];
            [unroll]
            for (uint byte = 0; byte < 4; ++byte) {
                const uint slot = word * 4 + byte;
                if (slot < offset || slot - offset >= p.outputCount) continue;
                const uint row = slot - offset;
                const uint old = oldLabel(p, row);
                const uint next = kPhase == 3 ? old :
                    (eligible(p, row) ? updated(p, old, loadByte(p.selected, row) != 0) : old);
                value = (value & ~(255u << (byte * 8))) | (next << (byte * 8));
            }
            output[word] = value;
        }
    } else {
        Ptr<int, Access.Read> indices = (Ptr<int, Access.Read>)p.indices;
        for (uint row = dispatchThreadId.x; row < p.count; row += stride) {
            const int id = indices[row];
            if (id < 0 || uint(id) >= p.outputCount || !eligible(p, row)) continue;
            const bool selected = loadByte(p.selected, row) != 0;
            const uint old = oldLabel(p, uint(id));
            if (kPhase == 1) {
                if (!selected && old == p.label) writeByte(p.output, uint(id), 0);
            } else if (selected) {
                const uint next = updated(p, old, true);
                if (next != old) writeByte(p.output, uint(id), next);
                // A selected row restores its own label after replacement clears.
                else if (p.mode == 2 && old == p.label) writeByte(p.output, uint(id), next);
            }
        }
    }
}
