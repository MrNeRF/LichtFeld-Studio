/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/image_texture_loader.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "io/pipelined_image_loader.hpp"

#include <glad/glad.h>

#include <exception>

namespace lfs::vis::image_texture {

    namespace {

        struct FormattedImage {
            lfs::core::Tensor tensor;
            int channels = 0;
            int width = 0;
            int height = 0;
        };

        std::optional<FormattedImage> format_image_tensor(const lfs::core::Tensor& tensor,
                                                          const std::string_view log_context) {
            if (!tensor.is_valid() || tensor.ndim() != 3 || tensor.numel() == 0) {
                return std::nullopt;
            }

            FormattedImage formatted;
            const auto& shape = tensor.shape();
            const int first_dim = static_cast<int>(shape[0]);
            const int last_dim = static_cast<int>(shape[2]);

            if (first_dim == 1 || first_dim == 3 || first_dim == 4) {
                formatted.channels = first_dim;
                formatted.height = static_cast<int>(shape[1]);
                formatted.width = static_cast<int>(shape[2]);
                formatted.tensor = tensor.permute({1, 2, 0}).contiguous();
            } else if (last_dim == 1 || last_dim == 3 || last_dim == 4) {
                formatted.channels = last_dim;
                formatted.height = static_cast<int>(shape[0]);
                formatted.width = static_cast<int>(shape[1]);
                formatted.tensor = tensor.contiguous();
            } else {
                LOG_WARN("Unsupported {} tensor shape: [{}, {}, {}]",
                         log_context,
                         static_cast<int>(shape[0]),
                         static_cast<int>(shape[1]),
                         static_cast<int>(shape[2]));
                return std::nullopt;
            }

            return formatted;
        }

        std::optional<LoadedTexture> upload_texture_cpu_fallback(FormattedImage formatted,
                                                                 const std::string_view log_context) {
            try {
                if (formatted.tensor.device() == lfs::core::Device::CUDA) {
                    formatted.tensor = formatted.tensor.cpu();
                }
                formatted.tensor = formatted.tensor.contiguous();

                if (formatted.tensor.dtype() != lfs::core::DataType::UInt8) {
                    formatted.tensor = (formatted.tensor.clamp(0.0f, 1.0f) * 255.0f)
                                           .to(lfs::core::DataType::UInt8);
                }

                unsigned int texture = 0;
                glGenTextures(1, &texture);
                glBindTexture(GL_TEXTURE_2D, texture);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

                const GLenum format = (formatted.channels == 1)   ? GL_RED
                                      : (formatted.channels == 4) ? GL_RGBA
                                                                  : GL_RGB;
                const GLenum internal = (formatted.channels == 1)   ? GL_R8
                                        : (formatted.channels == 4) ? GL_RGBA8
                                                                    : GL_RGB8;

                glTexImage2D(GL_TEXTURE_2D, 0, internal, formatted.width, formatted.height, 0,
                             format, GL_UNSIGNED_BYTE, formatted.tensor.ptr<unsigned char>());

                if (const GLenum gl_err = glGetError(); gl_err != GL_NO_ERROR) {
                    glBindTexture(GL_TEXTURE_2D, 0);
                    glDeleteTextures(1, &texture);
                    LOG_WARN("Failed CPU texture upload for {}: {}", log_context, static_cast<int>(gl_err));
                    return std::nullopt;
                }
                glBindTexture(GL_TEXTURE_2D, 0);

                LoadedTexture loaded;
                loaded.texture_id = texture;
                loaded.width = formatted.width;
                loaded.height = formatted.height;
                loaded.texcoord_scale = glm::vec2(1.0f);
                return loaded;
            } catch (const std::exception& e) {
                LOG_WARN("{} tensor CPU upload failed: {}", log_context, e.what());
                return std::nullopt;
            }
        }

    } // namespace

    std::optional<LoadedTexture> load_texture_from_tensor(const lfs::core::Tensor& tensor,
                                                          const std::string_view log_context) {
        auto formatted = format_image_tensor(tensor, log_context);
        if (!formatted) {
            return std::nullopt;
        }

        if (formatted->tensor.device() == lfs::core::Device::CUDA &&
            formatted->tensor.dtype() == lfs::core::DataType::Float32) {
            auto interop_texture = std::make_unique<lfs::rendering::CudaGLInteropTexture>();
            if (auto result = interop_texture->init(formatted->width, formatted->height); result) {
                if (auto upload_result = interop_texture->updateFromTensor(formatted->tensor); upload_result) {
                    LoadedTexture loaded;
                    loaded.texture_id = interop_texture->getTextureID();
                    loaded.width = formatted->width;
                    loaded.height = formatted->height;
                    loaded.texcoord_scale = glm::vec2(interop_texture->getTexcoordScaleX(),
                                                      interop_texture->getTexcoordScaleY());
                    loaded.interop_texture = std::move(interop_texture);
                    return loaded;
                } else {
                    LOG_WARN("Failed interop upload for {}: {}", log_context, upload_result.error());
                }
            } else {
                LOG_WARN("Failed interop init for {}: {}", log_context, result.error());
            }
        }

        return upload_texture_cpu_fallback(std::move(*formatted), log_context);
    }

    std::optional<LoadedTexture> load_texture_from_loader(
        lfs::io::PipelinedImageLoader& loader,
        const std::filesystem::path& path,
        const lfs::io::LoadParams& params,
        const std::string_view log_context) {
        if (!std::filesystem::exists(path)) {
            return std::nullopt;
        }

        try {
            auto tensor = loader.load_image_immediate(path, params);
            if (tensor.numel() == 0) {
                return std::nullopt;
            }
            return load_texture_from_tensor(tensor, log_context);
        } catch (const std::exception& e) {
            LOG_WARN("{} loader path failed for {}: {}",
                     log_context,
                     lfs::core::path_to_utf8(path),
                     e.what());
            return std::nullopt;
        }
    }

} // namespace lfs::vis::image_texture
