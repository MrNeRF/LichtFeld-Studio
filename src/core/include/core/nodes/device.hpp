/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/nodes/types.hpp"

namespace lfs::nodes {
    // Evaluation owns this cache; it is never shared between worker/viewer threads.
    // Mesh uploads are keyed by immutable source identity and its edit generation.
    class LFS_CORE_API GeometryDeviceCache {
    public:
        Geometry convert(Geometry geometry, core::Device device);
        void clear();

    private:
        struct MeshUpload {
            std::weak_ptr<const core::MeshData> source;
            uint32_t generation = 0;
            core::Device device = core::Device::CPU;
            std::optional<core::GpuBackend> backend;
            MeshComponent value;
        };
        std::unordered_map<uint64_t, MeshUpload> meshes_;
    };

    LFS_CORE_API core::Device evaluation_device();
} // namespace lfs::nodes
