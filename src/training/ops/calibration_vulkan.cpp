// SPDX-License-Identifier: GPL-3.0-or-later
#include "lfs/training/ops/calibration.hpp"
#include "vulkan/dispatch.hpp"
#include <vector>

namespace lfs::training {
    namespace {
        using namespace gpu_ops;
        using Tile = CalibrationTile;
        struct Params {
            uint64_t field, photo, render, output, mask, out_mask, structure, out_structure, edge, out_edge;
            uint64_t alpha, background, stable_background, stable, grad, partials;
            Tile tile;
            int photo_byte, mask_byte, background_image, chunks, margin, first, channels;
            float values, step;
        };

        void launch(const Params& p, uint32_t operation, uint32_t groups,
                    std::initializer_list<const Tensor*> inputs, std::initializer_list<const Tensor*> outputs) {
            std::vector<vulkan::StorageRef> reads, writes;
            for (const auto* t : inputs)
                if (t && t->is_valid())
                    reads.push_back(vulkan::ref(*t));
            for (const auto* t : outputs)
                if (t && t->is_valid())
                    writes.push_back(vulkan::ref(*t));
            vulkan::dispatch("calibration", p, reads, writes, groups, operation);
        }

        Params parameters(const Tile& t, In field, int slot) {
            Params p{};
            p.tile = t;
            p.field = vulkan::address(field) + size_t(slot) * t.side * t.side * 2 * sizeof(float);
            return p;
        }

        void sample(const Tile& t, In field, int slot, In photo, In render, Out output, const CalibrationMaps& m) {
            auto p = parameters(t, field, slot);
            p.photo = vulkan::address(photo);
            p.render = vulkan::address(render);
            p.output = vulkan::address(output);
            p.photo_byte = photo.dtype() == core::DataType::UInt8;
            p.mask = vulkan::address(m.mask);
            p.out_mask = p.mask ? vulkan::address(m.out_mask) : 0;
            p.mask_byte = p.mask && m.mask.dtype() != core::DataType::Float32;
            p.structure = vulkan::address(m.structure);
            p.out_structure = p.structure ? vulkan::address(m.out_structure) : 0;
            p.edge = vulkan::address(m.edge);
            p.out_edge = p.edge ? vulkan::address(m.out_edge) : 0;
            if (m.alpha) {
                p.alpha = vulkan::address(*m.alpha);
                p.background = vulkan::address(*m.background);
                p.background_image = m.background->numel() != 3;
                if (m.stable) {
                    p.stable_background = vulkan::address(*m.stable_background);
                    p.stable = vulkan::address(*m.stable);
                }
            }
            launch(p, 0, (t.w * t.h + 127) / 128,
                   {&field, &photo, &render, &m.mask, &m.structure, &m.edge, m.alpha, m.background, m.stable ? m.stable_background : nullptr},
                   {&output, p.mask ? &m.out_mask : nullptr, p.structure ? &m.out_structure : nullptr, p.edge ? &m.out_edge : nullptr, m.stable});
        }

        void accumulate(const Tile& t, In field, int slot, In image, In render, In grad, In weight,
                        float values, int margin, int chunks, bool first, Out partials) {
            auto p = parameters(t, field, slot);
            p.photo = vulkan::address(image);
            p.render = vulkan::address(render);
            p.grad = vulkan::address(grad);
            p.mask = vulkan::address(weight);
            p.mask_byte = p.mask && weight.dtype() != core::DataType::Float32;
            p.partials = vulkan::address(partials);
            p.chunks = chunks;
            p.first = first;
            p.margin = margin;
            p.values = values;
            launch(p, 1, (t.side - 1) * (t.side - 1) * chunks, {&field, &image, &render, &grad, &weight, first ? nullptr : &partials}, {&partials});
        }

        void update(const Tile& t, Out field, int slot, In partials, int chunks, float step) {
            auto p = parameters(t, field, slot);
            p.partials = vulkan::address(partials);
            p.chunks = chunks;
            p.step = step;
            launch(p, 2, 1, {&field, &partials}, {&field});
        }

        void clear(const Tile& t, In field, int slot, Out grad) {
            auto p = parameters(t, field, slot);
            p.grad = vulkan::address(grad);
            launch(p, 3, (t.w * t.h + 127) / 128, {&field, &grad}, {&grad});
        }

        void follow(const Tile& t, In field, int slot, In prior, Out output, int channels) {
            auto p = parameters(t, field, slot);
            p.photo = vulkan::address(prior);
            p.output = vulkan::address(output);
            p.channels = channels;
            launch(p, 4, (t.w * t.h + 127) / 128, {&field, &prior}, {&output});
        }
    } // namespace
    const gpu_ops::CalibrationOps& vulkan_calibration_ops() {
        static const gpu_ops::CalibrationOps ops{sample, accumulate, update, clear, follow};
        return ops;
    }
} // namespace lfs::training
