/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/mesh_data.hpp"
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

namespace lfs::io::mesh {

    using TriMesh = OpenMesh::TriMesh_ArrayKernelT<>;

    lfs::core::MeshData from_openmesh(const TriMesh& mesh);
    TriMesh to_openmesh(const lfs::core::MeshData& data);

} // namespace lfs::io::mesh
