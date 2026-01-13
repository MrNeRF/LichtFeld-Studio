/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "point_cloud_renderer.hpp"
#include "core/logger.hpp"
#include "gl_state_guard.hpp"
#include "shader_paths.hpp"
#include <vector>

namespace lfs::rendering {

    Result<void> PointCloudRenderer::initialize() {
        LOG_DEBUG("PointCloudRenderer::initialize() called on instance {}", static_cast<void*>(this));

        if (initialized_) {
            LOG_WARN("PointCloudRenderer already initialized!");
            return {};
        }

        LOG_TIMER_TRACE("PointCloudRenderer::initialize");

        // Create shader with geometry shader for equirectangular seam culling
        auto result = load_shader_with_geometry("point_cloud", "point_cloud.vert",
                                                "point_cloud.geom", "point_cloud.frag", false);
        if (!result) {
            LOG_ERROR("Failed to load point cloud shader: {}", result.error().what());
            return std::unexpected(result.error().what());
        }
        shader_ = std::move(*result);

        if (auto geom_result = createCubeGeometry(); !geom_result) {
            return geom_result;
        }

        initialized_ = true;
        LOG_INFO("PointCloudRenderer initialized successfully");
        return {};
    }

    Result<void> PointCloudRenderer::createCubeGeometry() {
        LOG_TIMER_TRACE("PointCloudRenderer::createCubeGeometry");

        // Create all resources first
        auto vao_result = create_vao();
        if (!vao_result) {
            LOG_ERROR("Failed to create VAO: {}", vao_result.error());
            return std::unexpected(vao_result.error());
        }

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            LOG_ERROR("Failed to create VBO: {}", vbo_result.error());
            return std::unexpected(vbo_result.error());
        }
        cube_vbo_ = std::move(*vbo_result);

        auto ebo_result = create_vbo(); // EBO is also a buffer
        if (!ebo_result) {
            LOG_ERROR("Failed to create EBO: {}", ebo_result.error());
            return std::unexpected(ebo_result.error());
        }
        cube_ebo_ = std::move(*ebo_result);

        auto instance_result = create_vbo();
        if (!instance_result) {
            LOG_ERROR("Failed to create instance VBO: {}", instance_result.error());
            return std::unexpected(instance_result.error());
        }
        instance_vbo_ = std::move(*instance_result);

        // Build VAO using VAOBuilder
        VAOBuilder builder(std::move(*vao_result));

        // Setup cube geometry
        std::span<const float> vertices_span(cube_vertices_,
                                             sizeof(cube_vertices_) / sizeof(float));
        builder.attachVBO(cube_vbo_, vertices_span, GL_STATIC_DRAW)
            .setAttribute({.index = 0,
                           .size = 3,
                           .type = GL_FLOAT,
                           .normalized = GL_FALSE,
                           .stride = 3 * sizeof(float),
                           .offset = nullptr,
                           .divisor = 0});

        // Instance layout: [pos(3f), color(3f), transform_index(1f)] = 28 bytes
        constexpr GLsizei INSTANCE_STRIDE = 7 * sizeof(float);
        builder.attachVBO(instance_vbo_)
            .setAttribute({.index = 1, .size = 3, .type = GL_FLOAT, .normalized = GL_FALSE, .stride = INSTANCE_STRIDE, .offset = nullptr, .divisor = 1})
            .setAttribute({.index = 2, .size = 3, .type = GL_FLOAT, .normalized = GL_FALSE, .stride = INSTANCE_STRIDE, .offset = reinterpret_cast<const void*>(3 * sizeof(float)), .divisor = 1})
            .setAttribute({.index = 3, .size = 1, .type = GL_FLOAT, .normalized = GL_FALSE, .stride = INSTANCE_STRIDE, .offset = reinterpret_cast<const void*>(6 * sizeof(float)), .divisor = 1});

        // Attach EBO - stays bound to VAO
        std::span<const unsigned int> indices_span(cube_indices_,
                                                   sizeof(cube_indices_) / sizeof(unsigned int));
        builder.attachEBO(cube_ebo_, indices_span, GL_STATIC_DRAW);

        // Build and store the VAO
        cube_vao_ = builder.build();

        LOG_DEBUG("Cube geometry created successfully");
        return {};
    }

    Tensor PointCloudRenderer::extractRGBFromSH(const Tensor& shs) {
        const float SH_C0 = 0.28209479177387814f;

        // Extract features_dc: shs[:, 0, :]
        // We need to slice along dimension 1 (the second dimension)
        Tensor features_dc = shs.slice(1, 0, 1).squeeze(1);

        // Calculate colors: features_dc * SH_C0 + 0.5
        Tensor colors = features_dc * SH_C0 + 0.5f;

        return colors.clamp(0.0f, 1.0f);
    }

    Result<void> PointCloudRenderer::render(const lfs::core::SplatData& splat_data,
                                            const glm::mat4& view,
                                            const glm::mat4& projection,
                                            const float voxel_size,
                                            const glm::vec3& background_color,
                                            const std::vector<glm::mat4>& model_transforms,
                                            const std::shared_ptr<lfs::core::Tensor>& transform_indices,
                                            const bool equirectangular) {
        if (splat_data.size() == 0)
            return {};

        const Tensor positions = splat_data.get_means();
        const Tensor shs = splat_data.get_shs();
        const Tensor colors = extractRGBFromSH(shs);

        return renderInternal(positions, colors, view, projection, voxel_size, background_color,
                              model_transforms, transform_indices, equirectangular);
    }

    Result<void> PointCloudRenderer::render(const lfs::core::PointCloud& point_cloud,
                                            const glm::mat4& view,
                                            const glm::mat4& projection,
                                            const float voxel_size,
                                            const glm::vec3& background_color,
                                            const std::vector<glm::mat4>& model_transforms,
                                            const std::shared_ptr<lfs::core::Tensor>& transform_indices,
                                            const bool equirectangular) {
        if (point_cloud.size() == 0)
            return {};

        const Tensor positions = point_cloud.means;
        Tensor colors = point_cloud.colors;
        if (colors.dtype() == lfs::core::DataType::UInt8)
            colors = colors.to(lfs::core::DataType::Float32) / 255.0f;

        return renderInternal(positions, colors, view, projection, voxel_size, background_color,
                              model_transforms, transform_indices, equirectangular);
    }

    Result<void> PointCloudRenderer::renderInternal(const Tensor& positions,
                                                    const Tensor& colors,
                                                    const glm::mat4& view,
                                                    const glm::mat4& projection,
                                                    const float voxel_size,
                                                    const glm::vec3& background_color,
                                                    const std::vector<glm::mat4>& model_transforms,
                                                    const std::shared_ptr<lfs::core::Tensor>& transform_indices,
                                                    const bool equirectangular) {
        if (!initialized_) {
            return std::unexpected("Renderer not initialized");
        }

        LOG_TIMER_TRACE("PointCloudRenderer::renderInternal");
        GLStateGuard state_guard;

        constexpr size_t MAX_POINT_COUNT = 50'000'000;
        const size_t num_points = positions.size(0);
        current_point_count_ = num_points;

        if (num_points > MAX_POINT_COUNT) {
            return std::unexpected("Point count exceeds limit");
        }

        // Build interleaved GPU buffer: [pos(3f), color(3f), transform_index(1f)]
        if (interleaved_cache_.size(0) != static_cast<int64_t>(num_points)) {
            interleaved_cache_ = Tensor::empty({num_points, 7}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        }
        interleaved_cache_.slice(1, 0, 3).copy_(positions);
        interleaved_cache_.slice(1, 3, 6).copy_(colors);
        if (transform_indices && transform_indices->numel() > 0) {
            interleaved_cache_.slice(1, 6, 7).copy_(transform_indices->unsqueeze(1));
        } else {
            interleaved_cache_.slice(1, 6, 7).fill_(0.0f);
        }
        const size_t buffer_size = interleaved_cache_.bytes();

#ifdef CUDA_GL_INTEROP_ENABLED
        if (use_interop_) {
            BufferBinder<GL_ARRAY_BUFFER> bind(instance_vbo_);
            if (interop_buffer_size_ != buffer_size) {
                interop_buffer_.reset();
                glBufferData(GL_ARRAY_BUFFER, buffer_size, nullptr, GL_DYNAMIC_DRAW);
                interop_buffer_.emplace();
                if (auto result = interop_buffer_->init(instance_vbo_.get(), buffer_size); result) {
                    interop_buffer_size_ = buffer_size;
                } else {
                    use_interop_ = false;
                    interop_buffer_.reset();
                    interop_buffer_size_ = 0;
                }
            }
            if (use_interop_ && interop_buffer_) {
                if (auto map_result = interop_buffer_->mapBuffer(); map_result) {
                    cudaMemcpy(*map_result, interleaved_cache_.data_ptr(), buffer_size, cudaMemcpyDeviceToDevice);
                    if (auto result = interop_buffer_->unmapBuffer(); !result) {
                        LOG_DEBUG("Failed to unmap buffer: {}", result.error());
                    }
                } else {
                    use_interop_ = false;
                    interop_buffer_.reset();
                    interop_buffer_size_ = 0;
                }
            }
        }
        if (!use_interop_)
#endif
        {
            const Tensor cpu_data = interleaved_cache_.cpu();
            BufferBinder<GL_ARRAY_BUFFER> bind(instance_vbo_);
            glBufferData(GL_ARRAY_BUFFER, buffer_size, cpu_data.data_ptr(), GL_DYNAMIC_DRAW);
        }

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);
        glClearColor(background_color.r, background_color.g, background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ShaderScope s(shader_);
        if (auto result = s->set("u_view", view); !result)
            return result;
        if (auto result = s->set("u_projection", projection); !result)
            return result;
        if (auto result = s->set("u_voxel_size", voxel_size); !result)
            return result;
        if (auto result = s->set("u_equirectangular", equirectangular); !result)
            return result;

        constexpr int MAX_TRANSFORMS = 64;
        const int num_transforms = static_cast<int>(std::min(model_transforms.size(), size_t(MAX_TRANSFORMS)));
        if (auto result = s->set("u_num_transforms", num_transforms); !result)
            return result;
        for (int i = 0; i < num_transforms; ++i) {
            if (auto result = s->set(std::format("u_model_transforms[{}]", i), model_transforms[i]); !result) {
                LOG_ERROR("Failed to set u_model_transforms[{}] uniform: {}", i, result.error());
            }
        }

        if (!cube_vao_ || cube_vao_.get() == 0)
            return std::unexpected("Invalid cube VAO");
        if (current_point_count_ == 0)
            return {};

        VAOBinder vao_bind(cube_vao_);
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0, static_cast<GLsizei>(current_point_count_));

        if (const GLenum err = glGetError(); err != GL_NO_ERROR) {
            return std::unexpected(std::format("OpenGL error: 0x{:x}", err));
        }
        return {};
    }

} // namespace lfs::rendering