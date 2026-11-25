/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "calibration_program.hpp"
#include "core/error.hpp"
#include "core/gpu_kernel_module.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/calibration.hpp"
#include <array>
#include <cstddef>

namespace lfs::training {
    namespace {
        using namespace gpu_ops;
        using core::GpuKernelModule;
        using Tile = CalibrationTile;

        // Parameters in calibration.slang: 19 tensor pointers, then the scalars.
        struct Parameters {
            uint64_t pointers[19]{};
            Tile tile{};
            int slot = 0, background_image = 0, chunks = 0, margin = 0, first = 0, channels = 0;
            float values = 0, step = 0;
        };
        static_assert(sizeof(Parameters) == 224 && offsetof(Parameters, tile) == 152);
        enum Slot { Field,
                    Photo,
                    PhotoBytes,
                    Render,
                    Output,
                    Mask,
                    MaskBytes,
                    OutMask,
                    OutMaskBytes,
                    Structure,
                    OutStructure,
                    Edge,
                    OutEdge,
                    Alpha,
                    Background,
                    StableBackground,
                    Stable,
                    Grad,
                    Partials,
                    SlotCount };
        struct Tensors {
            std::array<const Tensor*, SlotCount> slots{};
            uint32_t writes = 0; // bit per written slot

            void read(Slot s, const Tensor* t) { slots[s] = t && t->is_valid() ? t : nullptr; }
            void write(Slot s, const Tensor* t) {
                read(s, t);
                writes |= 1u << s;
            }
        };
        bool bytes(const Tensor& t) { return t.dtype() != core::DataType::Float32; }

        void run(const char* function, In field, Parameters p, const Tensors& tensors, uint32_t groups) {
            const auto backend = core::gpu_backend_of(field).value_or(core::default_gpu_backend());
            const core::GpuBackendScope scope(backend);
            auto program = GpuKernelModule::load(calibration_program_entries(), backend);
            if (!program)
                throw lfs::Exception(std::move(program).error());
            std::array<GpuKernelModule::Binding, SlotCount> bindings;
            for (uint32_t s = 0; s < SlotCount; ++s)
                bindings[s] = {s * 8, tensors.slots[s],
                               tensors.writes & (1u << s) ? GpuKernelModule::Access::ReadWrite : GpuKernelModule::Access::Read};
            auto result = (*program)->dispatch({.function = function,
                                                .arguments = {std::as_bytes(std::span(&p, 1)), bindings},
                                                .groups = {groups, 1, 1},
                                                .group = {128, 1, 1}});
            if (!result)
                throw lfs::Exception(std::move(result).error());
        }

        Parameters parameters(const Tile& t, int slot) {
            Parameters p;
            p.tile = t;
            p.slot = slot;
            return p;
        }
        uint32_t pixel_groups(const Tile& t) { return uint32_t(t.w * t.h + 127) / 128; }

        void sample(const Tile& t, In field, int slot, In photo, In render, Out output, const CalibrationMaps& m) {
            Tensors s;
            s.read(Field, &field);
            s.read(bytes(photo) ? PhotoBytes : Photo, &photo);
            s.read(Render, &render);
            s.write(Output, &output);
            if (m.mask.is_valid()) {
                s.read(bytes(m.mask) ? MaskBytes : Mask, &m.mask);
                s.write(bytes(m.mask) ? OutMaskBytes : OutMask, &m.out_mask);
            }
            if (m.structure.is_valid()) {
                s.read(Structure, &m.structure);
                s.write(OutStructure, &m.out_structure);
            }
            if (m.edge.is_valid()) {
                s.read(Edge, &m.edge);
                s.write(OutEdge, &m.out_edge);
            }
            auto p = parameters(t, slot);
            if (m.alpha) {
                s.read(Alpha, m.alpha);
                s.read(Background, m.background);
                p.background_image = m.background->numel() != 3;
                if (m.stable) {
                    s.read(StableBackground, m.stable_background);
                    s.write(Stable, m.stable);
                }
            }
            run("resample", field, p, s, pixel_groups(t));
        }

        void accumulate(const Tile& t, In field, int slot, In image, In render, In grad, In weight,
                        float values, int margin, int chunks, bool first, Out partials) {
            Tensors s;
            s.read(Field, &field);
            s.read(Photo, &image);
            s.read(Render, &render);
            s.read(Grad, &grad);
            if (weight.is_valid())
                s.read(bytes(weight) ? MaskBytes : Mask, &weight);
            s.write(Partials, &partials);
            auto p = parameters(t, slot);
            p.chunks = chunks;
            p.first = first;
            p.margin = margin;
            p.values = values;
            run("accumulate", field, p, s, uint32_t((t.side - 1) * (t.side - 1) * chunks));
        }

        void update(const Tile& t, Out field, int slot, In partials, int chunks, float step) {
            Tensors s;
            s.write(Field, &field);
            s.read(Partials, &partials);
            auto p = parameters(t, slot);
            p.chunks = chunks;
            p.step = step;
            run("update", field, p, s, 1);
        }

        void clear_outside(const Tile& t, In field, int slot, Out grad) {
            Tensors s;
            s.read(Field, &field);
            s.write(Grad, &grad);
            run("clear_outside", field, parameters(t, slot), s, pixel_groups(t));
        }

        void follow(const Tile& t, In field, int slot, In prior, Out output, int channels) {
            Tensors s;
            s.read(Field, &field);
            s.read(Photo, &prior);
            s.write(Output, &output);
            auto p = parameters(t, slot);
            p.channels = channels;
            run("follow", field, p, s, pixel_groups(t));
        }
    } // namespace
    const gpu_ops::CalibrationOps& portable_calibration_ops() {
        static const gpu_ops::CalibrationOps ops{sample, accumulate, update, clear_outside, follow};
        return ops;
    }
} // namespace lfs::training
