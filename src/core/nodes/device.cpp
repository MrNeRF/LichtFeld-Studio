/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/nodes/device.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_execution.hpp"

#include <algorithm>

namespace lfs::nodes {
    core::Device evaluation_device() {
        return core::gpu_backend_available(core::TensorExecutionTarget::current().backend()) ? core::Device::GPU : core::Device::CPU;
    }

    void GeometryDeviceCache::clear() { meshes_.clear(); }

    Geometry GeometryDeviceCache::convert(Geometry geometry, const core::Device device) {
        using namespace core;
        const auto backend = device == Device::GPU ? std::optional{TensorExecutionTarget::current().backend()} : std::nullopt;
        const auto transfer = [&](const Tensor& value) {
            if (!value.is_valid())
                return value;
            // Already resident: keep the storage, only finishing deferred work so
            // cached and published results are complete when their fence passes.
            if (value.device() == device && gpu_backend_of(value) == backend)
                return value.is_deferred() ? value.contiguous() : value;
            return backend ? value.to(*backend) : value.to(Device::CPU);
        };
        const auto attributes = [&](AttributeMap& values) {
            for (auto& [_, tensor] : values)
                tensor = transfer(tensor);
        };
        if (geometry.splats) {
            auto& s = *geometry.splats;
            s.means = transfer(s.means);
            s.sh0 = transfer(s.sh0);
            s.shN = transfer(s.shN);
            s.scaling = transfer(s.scaling);
            s.rotation = transfer(s.rotation);
            s.opacity = transfer(s.opacity);
            attributes(s.attributes);
        }
        if (geometry.points) {
            geometry.points->positions = transfer(geometry.points->positions);
            geometry.points->colors = transfer(geometry.points->colors);
            if (geometry.points->colors.is_valid() && geometry.points->colors.dtype() == DataType::UInt8)
                geometry.points->colors = geometry.points->colors.to(DataType::Float32) / 255.0f;
            attributes(geometry.points->attributes);
        }
        if (!geometry.mesh || !geometry.mesh->mesh)
            return geometry;
        const auto source = geometry.mesh->mesh;
        const auto resident = [&](const Tensor& tensor) {
            return !tensor.is_valid() || (tensor.device() == device && gpu_backend_of(tensor) == backend);
        };
        const bool same_device = resident(source->vertices) && resident(source->indices) && resident(source->normals) &&
                                 resident(source->tangents) && resident(source->texcoords) && resident(source->colors) &&
                                 std::ranges::all_of(geometry.mesh->textures, resident);
        if (same_device && geometry.mesh->textures.size() == source->texture_images.size())
            return geometry;
        std::erase_if(meshes_, [](const auto& item) { return item.second.source.expired(); });
        auto& entry = meshes_[source->id()];
        if (entry.source.expired() || entry.generation != source->generation() || entry.device != device || entry.backend != backend) {
            auto mesh = std::make_shared<MeshData>();
            mesh->vertices = transfer(source->vertices);
            mesh->indices = transfer(source->indices);
            mesh->normals = transfer(source->normals);
            mesh->tangents = transfer(source->tangents);
            mesh->texcoords = transfer(source->texcoords);
            mesh->colors = transfer(source->colors);
            mesh->materials = source->materials;
            mesh->submeshes = source->submeshes;
            mesh->texture_images = source->texture_images;
            MeshComponent value{mesh};
            value.textures.resize(source->texture_images.size());
            if (source->has_texcoords() && !source->has_colors()) {
                for (const auto& material : source->materials) {
                    const auto id = material.albedo_tex;
                    if (!id || id > value.textures.size() || value.textures[id - 1].is_valid())
                        continue;
                    const auto& image = source->texture_images[id - 1];
                    if (image.width <= 0 || image.height <= 0 || image.channels < 3 ||
                        image.pixels.size() != static_cast<size_t>(image.width) * image.height * image.channels)
                        continue;
                    const auto pixels = Tensor::from_blob(const_cast<uint8_t*>(image.pixels.data()),
                                                          {static_cast<size_t>(image.width) * image.height, static_cast<size_t>(image.channels)},
                                                          Device::CPU, DataType::UInt8);
                    value.textures[id - 1] = transfer(pixels).slice(1, 0, 3).to(DataType::Float32) / 255.0f;
                }
            }
            entry = {source, source->generation(), device, backend, std::move(value)};
        }
        entry.source = source;
        geometry.mesh = entry.value;
        return geometry;
    }
} // namespace lfs::nodes
