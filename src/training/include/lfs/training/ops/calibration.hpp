// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "lfs/training/ops/types.hpp"

namespace lfs::gpu_ops {
    struct CalibrationTile {
        int width, height, x, y, w, h, side;
        float sx, sy;
        int wrap;
    };
    struct CalibrationMaps {
        In mask, structure, edge;
        Out out_mask, out_structure, out_edge;
        const Tensor *alpha, *background, *stable_background;
        Tensor* stable;
    };
    struct CalibrationOps {
        void (*sample)(const CalibrationTile&, In field, int slot, In photo, In render, Out output, const CalibrationMaps&);
        void (*accumulate)(const CalibrationTile&, In field, int slot, In image, In render, In grad, In weight,
                           float values, int margin, int chunks, bool first, Out partials);
        void (*update)(const CalibrationTile&, Out field, int slot, In partials, int chunks, float step);
        void (*clear_outside)(const CalibrationTile&, In field, int slot, Out gradient);
        void (*follow)(const CalibrationTile&, In field, int slot, In prior, Out output, int channels);
    };
} // namespace lfs::gpu_ops
namespace lfs::training {
    const gpu_ops::CalibrationOps& cuda_calibration_ops();
    const gpu_ops::CalibrationOps& vulkan_calibration_ops();
} // namespace lfs::training
