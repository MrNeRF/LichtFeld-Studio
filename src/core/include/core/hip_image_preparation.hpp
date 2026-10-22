/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <core/export.hpp>
#include <memory>
namespace lfs::core {
    class Tensor;
    struct UndistortParams;
    // A bounded, private image producer. GPU buffers never escape its native
    // queue: the caller receives complete CPU bytes in the requested CHW or HWC layout for caching/upload.
    class HipImagePreparer {
    public:
        virtual ~HipImagePreparer() = default;
        virtual Tensor prepare(const Tensor& bytes, const UndistortParams&, bool interleaved = false) = 0;
    };
    LFS_CORE_API std::unique_ptr<HipImagePreparer> create_hip_image_preparer();
    // Automatic policy is conservative for unvalidated architectures. Explicit
    // opt-in can still exercise the generic producer on another supported GPU.
    LFS_CORE_API bool hip_image_preparation_default_enabled();
}
