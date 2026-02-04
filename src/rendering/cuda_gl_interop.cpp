/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "config.h"

// clang-format off
// CRITICAL: GLAD must be included before GLFW to avoid OpenGL header conflicts
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "cuda_gl_interop.hpp"
#include <format>

#ifdef CUDA_GL_INTEROP_ENABLED
// Only include CUDA GL interop when enabled
#include <cuda_gl_interop.h>
#endif

namespace lfs::rendering {

    namespace {
        constexpr int GPU_ALIGNMENT = 16; // 16-pixel alignment for GPU texture efficiency
    }

    // Implementation for CudaGraphicsResourceDeleter
    void CudaGraphicsResourceDeleter::operator()(void* resource) const {
#ifdef CUDA_GL_INTEROP_ENABLED
        if (resource) {
            cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource_t>(resource));
            LOG_TRACE("Unregistered CUDA graphics resource");
        }
#endif
    }

    // Implementation for disabled interop version
    CudaGLInteropTextureImpl<false>::~CudaGLInteropTextureImpl() {
        cleanup();
    }

    Result<void> CudaGLInteropTextureImpl<false>::init(int width, int height) {
        LOG_TIMER_TRACE("CudaGLInteropTextureImpl<false>::init");
        LOG_DEBUG("Initializing non-interop texture: {}x{}", width, height);

        width_ = width;
        height_ = height;

        // Create regular OpenGL texture
        glGenTextures(1, &texture_id_);
        glBindTexture(GL_TEXTURE_2D, texture_id_);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        glBindTexture(GL_TEXTURE_2D, 0);

        GLenum gl_err = glGetError();
        if (gl_err != GL_NO_ERROR) {
            cleanup();
            LOG_ERROR("OpenGL error during texture creation: {}", gl_err);
            return std::unexpected(std::format("OpenGL error during texture creation: {}", gl_err));
        }

        LOG_DEBUG("Non-interop texture created successfully");
        return {};
    }

    Result<void> CudaGLInteropTextureImpl<false>::resize(int new_width, int new_height) {
        const int alloc_width = ((new_width + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT;
        const int alloc_height = ((new_height + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT;

        // Reuse if allocation already matches exactly
        if (texture_id_ != 0 && alloc_width == allocated_width_ && alloc_height == allocated_height_) {
            if (width_ != new_width || height_ != new_height) {
                width_ = new_width;
                height_ = new_height;
            }
            return {};
        }

        LOG_TRACE("Resize non-interop texture: {}x{} -> {}x{}",
                  allocated_width_, allocated_height_, alloc_width, alloc_height);

        lfs::core::CudaMemoryPool::instance().trim_cached_memory();

        auto result = init(alloc_width, alloc_height);
        if (result) {
            allocated_width_ = alloc_width;
            allocated_height_ = alloc_height;
            width_ = new_width;
            height_ = new_height;
        }
        return result;
    }

    Result<void> CudaGLInteropTextureImpl<false>::updateFromTensor(const Tensor& image) {
        // CPU fallback - this should not be called for non-interop version
        LOG_ERROR("CUDA-GL interop not available - use regular framebuffer upload");
        return std::unexpected("CUDA-GL interop not available - use regular framebuffer upload");
    }

    void CudaGLInteropTextureImpl<false>::cleanup() {
        if (texture_id_ != 0) {
            LOG_TRACE("Cleaning up non-interop texture");
            glDeleteTextures(1, &texture_id_);
            texture_id_ = 0;
        }
    }

#ifdef CUDA_GL_INTEROP_ENABLED
    // Full implementation for when interop is enabled
    CudaGLInteropTextureImpl<true>::CudaGLInteropTextureImpl()
        : texture_id_(0),
          cuda_resource_(nullptr),
          width_(0),
          height_(0),
          is_registered_(false) {
        LOG_DEBUG("Creating CUDA-GL interop texture");
    }

    CudaGLInteropTextureImpl<true>::~CudaGLInteropTextureImpl() {
        cleanup();
    }

    Result<void> CudaGLInteropTextureImpl<true>::init(int width, int height) {
        LOG_TIMER("CudaGLInteropTextureImpl<true>::init");

        // Clean up any existing resources
        cleanup();

        width_ = width;
        height_ = height;
        allocated_width_ = ((width + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT;
        allocated_height_ = ((height + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT;
        external_texture_ = false;

        glGenTextures(1, &texture_id_);
        glBindTexture(GL_TEXTURE_2D, texture_id_);

        // Set texture parameters BEFORE allocating storage
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Allocate texture storage (RGBA for better alignment)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, allocated_width_, allocated_height_, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        // CRITICAL: Unbind texture before registering with CUDA
        glBindTexture(GL_TEXTURE_2D, 0);

        // Check OpenGL errors
        GLenum gl_err = glGetError();
        if (gl_err != GL_NO_ERROR) {
            cleanup();
            LOG_ERROR("OpenGL error during texture creation: {}", gl_err);
            return std::unexpected(std::format("OpenGL error during texture creation: {}", gl_err));
        }

        // Clear any previous CUDA errors
        cudaGetLastError();

        // Register texture with CUDA
        cudaGraphicsResource_t raw_resource;
        cudaError_t err = cudaGraphicsGLRegisterImage(
            &raw_resource, texture_id_, GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsWriteDiscard);

        if (err != cudaSuccess) {
            cleanup();
            LOG_ERROR("Failed to register OpenGL texture with CUDA: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to register OpenGL texture with CUDA: {}",
                                               cudaGetErrorString(err)));
        }

        cuda_resource_.reset(raw_resource);
        is_registered_ = true;
        return {};
    }

    Result<void> CudaGLInteropTextureImpl<true>::initForDepth(const int width, const int height) {
        LOG_TIMER_TRACE("CudaGLInteropTextureImpl::initForDepth");
        cleanup();

        width_ = width;
        height_ = height;
        is_depth_format_ = true;
        external_texture_ = false;

        glGenTextures(1, &texture_id_);
        glBindTexture(GL_TEXTURE_2D, texture_id_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        if (const GLenum gl_err = glGetError(); gl_err != GL_NO_ERROR) {
            cleanup();
            return std::unexpected(std::format("GL error creating depth texture: {}", gl_err));
        }

        cudaGetLastError();
        cudaGraphicsResource_t raw_resource;
        const cudaError_t err = cudaGraphicsGLRegisterImage(
            &raw_resource, texture_id_, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

        if (err != cudaSuccess) {
            cleanup();
            return std::unexpected(std::format("CUDA register failed: {}", cudaGetErrorString(err)));
        }

        cuda_resource_.reset(raw_resource);
        is_registered_ = true;
        LOG_DEBUG("Depth interop initialized: {}x{}", width, height);
        return {};
    }

    Result<void> CudaGLInteropTextureImpl<true>::updateDepthFromTensor(const Tensor& depth) {
        LOG_TIMER_TRACE("CudaGLInteropTextureImpl::updateDepthFromTensor");

        if (!is_registered_) {
            return std::unexpected("Depth texture not initialized");
        }
        if (depth.device() != lfs::core::Device::CUDA) {
            return std::unexpected("Depth must be on CUDA");
        }

        // Handle [1, H, W] or [H, W] formats
        Tensor depth_2d = (depth.ndim() == 3 && depth.size(0) == 1) ? depth.squeeze(0) : depth;
        if (depth_2d.ndim() != 2) {
            return std::unexpected("Depth must be [H, W] or [1, H, W]");
        }

        const int h = static_cast<int>(depth_2d.size(0));
        const int w = static_cast<int>(depth_2d.size(1));

        if (w != width_ || h != height_) {
            if (auto result = initForDepth(w, h); !result) {
                return result;
            }
        }

        auto raw_resource = static_cast<cudaGraphicsResource_t>(cuda_resource_.get());
        cudaError_t err = cudaGraphicsMapResources(1, &raw_resource, 0);
        if (err != cudaSuccess) {
            return std::unexpected(std::format("Map failed: {}", cudaGetErrorString(err)));
        }

        const struct UnmapGuard {
            cudaGraphicsResource_t* res;
            ~UnmapGuard() {
                if (res)
                    cudaGraphicsUnmapResources(1, res, 0);
            }
        } guard{&raw_resource};

        cudaArray_t cuda_array;
        err = cudaGraphicsSubResourceGetMappedArray(&cuda_array, raw_resource, 0, 0);
        if (err != cudaSuccess) {
            return std::unexpected(std::format("Get array failed: {}", cudaGetErrorString(err)));
        }

        Tensor depth_contig = depth_2d.contiguous();
        if (depth_contig.dtype() != lfs::core::DataType::Float32) {
            depth_contig = depth_contig.to(lfs::core::DataType::Float32);
        }

        err = cudaMemcpy2DToArray(cuda_array, 0, 0, depth_contig.ptr<float>(),
                                  w * sizeof(float), w * sizeof(float), h,
                                  cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            return std::unexpected(std::format("Copy failed: {}", cudaGetErrorString(err)));
        }

        return {};
    }

    Result<void> CudaGLInteropTextureImpl<true>::initForReading(GLuint texture_id, int width, int height) {
        LOG_TIMER_TRACE("CudaGLInteropTextureImpl<true>::initForReading");
        LOG_DEBUG("Init interop for reading texture {}: {}x{}", texture_id, width, height);

        cleanup();

        texture_id_ = texture_id;
        width_ = width;
        height_ = height;
        external_texture_ = true; // Externally owned texture

        // Clear any previous CUDA errors
        cudaGetLastError();

        // Register existing texture with CUDA for reading
        cudaGraphicsResource_t raw_resource;
        cudaError_t err = cudaGraphicsGLRegisterImage(
            &raw_resource, texture_id, GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsReadOnly);

        if (err != cudaSuccess) {
            LOG_ERROR("Failed to register OpenGL texture for reading: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to register OpenGL texture for reading: {}",
                                               cudaGetErrorString(err)));
        }

        cuda_resource_.reset(raw_resource);
        is_registered_ = true;

        LOG_DEBUG("CUDA-GL interop texture registered for reading");
        return {};
    }

    Result<void> CudaGLInteropTextureImpl<true>::readToTensor(Tensor& output, int target_width, int target_height) {
        LOG_TIMER_TRACE("CudaGLInteropTextureImpl<true>::readToTensor");

        if (!is_registered_) {
            LOG_ERROR("Texture not registered");
            return std::unexpected("Texture not registered");
        }

        // Use target dimensions if provided, otherwise use full texture dimensions
        const int out_width = (target_width > 0 && target_width <= width_) ? target_width : width_;
        const int out_height = (target_height > 0 && target_height <= height_) ? target_height : height_;

        // Map CUDA resource
        auto raw_resource = static_cast<cudaGraphicsResource_t>(cuda_resource_.get());
        cudaError_t err = cudaGraphicsMapResources(1, &raw_resource, 0);
        if (err != cudaSuccess) {
            LOG_ERROR("Failed to map CUDA resource: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to map CUDA resource: {}",
                                               cudaGetErrorString(err)));
        }

        // RAII unmap guard
        struct UnmapGuard {
            cudaGraphicsResource_t* resource;
            ~UnmapGuard() {
                if (resource) {
                    cudaGraphicsUnmapResources(1, resource, 0);
                    LOG_TRACE("Unmapped CUDA resource");
                }
            }
        } unmap_guard{&raw_resource};

        // Get CUDA array from mapped resource
        cudaArray_t cuda_array;
        err = cudaGraphicsSubResourceGetMappedArray(&cuda_array, raw_resource, 0, 0);
        if (err != cudaSuccess) {
            LOG_ERROR("Failed to get CUDA array: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to get CUDA array: {}",
                                               cudaGetErrorString(err)));
        }

        // Allocate output tensor [H, W, 3] in float32
        if (!output.is_valid() || output.size(0) != static_cast<size_t>(out_height) ||
            output.size(1) != static_cast<size_t>(out_width) || output.size(2) != 3) {
            output = Tensor::empty({static_cast<size_t>(out_height),
                                    static_cast<size_t>(out_width),
                                    3},
                                   lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        }

        // Allocate temp buffer for RGBA data (only the region we need)
        auto rgba_temp = Tensor::empty({static_cast<size_t>(out_height),
                                        static_cast<size_t>(out_width),
                                        4},
                                       lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        // Copy from CUDA array to temp buffer (RGBA float32)
        // Only copy the target region (top-left corner of the texture)
        err = cudaMemcpy2DFromArray(
            rgba_temp.ptr<float>(),
            out_width * 4 * sizeof(float), // dst pitch
            cuda_array,
            0, 0,                          // src offset (x, y)
            out_width * 4 * sizeof(float), // width in bytes to copy
            out_height,                    // height (rows) to copy
            cudaMemcpyDeviceToDevice);

        if (err != cudaSuccess) {
            LOG_ERROR("Failed to copy from CUDA array: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to copy from CUDA array: {}",
                                               cudaGetErrorString(err)));
        }

        // Extract RGB channels (drop alpha)
        output = rgba_temp.slice(2, 0, 3).contiguous();

        LOG_TRACE("Successfully read texture to tensor ({}x{})", out_width, out_height);
        return {};
    }

    Result<void> CudaGLInteropTextureImpl<true>::resize(int new_width, int new_height) {
        const int alloc_width = ((new_width + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT;
        const int alloc_height = ((new_height + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT;

        // Reuse if allocation already matches exactly
        if (is_registered_ && alloc_width == allocated_width_ && alloc_height == allocated_height_) {
            if (width_ != new_width || height_ != new_height) {
                width_ = new_width;
                height_ = new_height;
            }
            return {};
        }

        LOG_TRACE("Resize interop texture: {}x{} -> {}x{}",
                  allocated_width_, allocated_height_, alloc_width, alloc_height);

        lfs::core::CudaMemoryPool::instance().trim_cached_memory();

        auto result = init(alloc_width, alloc_height);
        if (result) {
            allocated_width_ = alloc_width;
            allocated_height_ = alloc_height;
            width_ = new_width;
            height_ = new_height;
        }
        return result;
    }

    Result<void> CudaGLInteropTextureImpl<true>::updateFromTensor(const Tensor& image) {
        LOG_TIMER_TRACE("CudaGLInteropTextureImpl<true>::updateFromTensor");

        if (!is_registered_) {
            LOG_ERROR("Texture not initialized");
            return std::unexpected("Texture not initialized");
        }

        // Ensure tensor is CUDA, float32, and [H, W, C] format
        if (image.device() != lfs::core::Device::CUDA) {
            LOG_ERROR("Image must be on CUDA");
            return std::unexpected("Image must be on CUDA");
        }
        if (image.ndim() != 3) {
            LOG_ERROR("Image must be [H, W, C], got {} dimensions", image.ndim());
            return std::unexpected("Image must be [H, W, C]");
        }
        if (image.size(2) != 3 && image.size(2) != 4) {
            LOG_ERROR("Image must have 3 or 4 channels, got {}", image.size(2));
            return std::unexpected("Image must have 3 or 4 channels");
        }

        const int h = image.size(0);
        const int w = image.size(1);
        const int c = image.size(2);

        LOG_TRACE("updateFromTensor: {}x{}x{}, texture {}x{}", h, w, c, width_, height_);

        // Resize if needed
        if (auto result = resize(w, h); !result) {
            return result;
        }

        // Map CUDA resource
        auto raw_resource = static_cast<cudaGraphicsResource_t>(cuda_resource_.get());
        cudaError_t err = cudaGraphicsMapResources(1, &raw_resource, 0);
        if (err != cudaSuccess) {
            LOG_ERROR("Failed to map CUDA resource: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to map CUDA resource: {}",
                                               cudaGetErrorString(err)));
        }

        // RAII unmap guard
        struct UnmapGuard {
            cudaGraphicsResource_t* resource;
            ~UnmapGuard() {
                if (resource) {
                    cudaGraphicsUnmapResources(1, resource, 0);
                    LOG_TRACE("Unmapped CUDA resource");
                }
            }
        } unmap_guard{&raw_resource};

        // Get CUDA array from mapped resource
        cudaArray_t cuda_array;
        err = cudaGraphicsSubResourceGetMappedArray(&cuda_array, raw_resource, 0, 0);
        if (err != cudaSuccess) {
            LOG_ERROR("Failed to get CUDA array: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to get CUDA array: {}",
                                               cudaGetErrorString(err)));
        }

        // Convert to RGBA uint8 using cached tensors to avoid per-frame allocations
        const bool need_alpha = (c == 3);
        const size_t sh = static_cast<size_t>(h);
        const size_t sw = static_cast<size_t>(w);

        // Ensure cached RGBA buffer has correct size
        if (!cached_rgba_.is_valid() ||
            cached_rgba_.size(0) != sh ||
            cached_rgba_.size(1) != sw ||
            cached_rgba_.device() != image.device()) {
            cached_rgba_ = Tensor::empty({sh, sw, 4}, image.device(), image.dtype());
        }

        // Ensure cached uint8 buffer has correct size
        if (!cached_uint8_.is_valid() ||
            cached_uint8_.size(0) != sh ||
            cached_uint8_.size(1) != sw) {
            cached_uint8_ = Tensor::empty({sh, sw, 4}, image.device(), lfs::core::DataType::UInt8);
        }

        // Build RGBA: copy RGB channels and set alpha
        if (need_alpha) {
            LOG_TIMER_TRACE("CudaGLInteropTextureImpl<true>::channels");
            // Copy RGB into first 3 channels of cached RGBA
            cached_rgba_.slice(2, 0, 3).copy_(image);
            // Set alpha channel to 1.0
            cached_rgba_.slice(2, 3, 4).fill_(1.0f);
        } else {
            cached_rgba_.copy_(image);
        }

        // Convert to uint8: clamp, scale, and convert in-place to cached buffer
        if (cached_rgba_.dtype() != lfs::core::DataType::UInt8) {
            LOG_TIMER_TRACE("Converted image to uint8");
            cached_uint8_.copy_((cached_rgba_.clamp(0.0f, 1.0f) * 255.0f).to(lfs::core::DataType::UInt8));
        }

        const Tensor& rgba_image =
            (cached_rgba_.dtype() == lfs::core::DataType::UInt8) ? cached_rgba_ : cached_uint8_;

        // Copy to CUDA array
        err = cudaMemcpy2DToArray(
            cuda_array,
            0, 0, // offset
            rgba_image.ptr<uint8_t>(),
            w * 4, // pitch (RGBA = 4 bytes per pixel)
            w * 4, // width in bytes
            h,     // height
            cudaMemcpyDeviceToDevice);

        if (err != cudaSuccess) {
            LOG_ERROR("Failed to copy to CUDA array: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to copy to CUDA array: {}",
                                               cudaGetErrorString(err)));
        }

        // cudaGraphicsUnmapResources provides sync; explicit sync would block on VSync
        LOG_TRACE("Updated texture from CUDA tensor");
        return {};
    }

    void CudaGLInteropTextureImpl<true>::cleanup() {
        if (is_registered_) {
            cudaStreamSynchronize(0);
        }
        cuda_resource_.reset();
        is_registered_ = false;

        if (texture_id_ != 0 && !external_texture_) {
            glDeleteTextures(1, &texture_id_);
        }
        texture_id_ = 0;
        external_texture_ = false;
    }
#endif // CUDA_GL_INTEROP_ENABLED

    // ===== CudaGLInteropBuffer implementation =====

    // Non-interop version
    CudaGLInteropBufferImpl<false>::~CudaGLInteropBufferImpl() {
        cleanup();
    }

    Result<void> CudaGLInteropBufferImpl<false>::init(GLuint buffer_id, size_t size) {
        LOG_ERROR("CUDA-GL buffer interop not available");
        return std::unexpected("CUDA-GL buffer interop not available");
    }

    Result<void*> CudaGLInteropBufferImpl<false>::mapBuffer() {
        LOG_ERROR("CUDA-GL buffer interop not available");
        return std::unexpected("CUDA-GL buffer interop not available");
    }

    Result<void> CudaGLInteropBufferImpl<false>::unmapBuffer() {
        LOG_ERROR("CUDA-GL buffer interop not available");
        return std::unexpected("CUDA-GL buffer interop not available");
    }

    void CudaGLInteropBufferImpl<false>::cleanup() {
        buffer_id_ = 0;
        size_ = 0;
    }

#ifdef CUDA_GL_INTEROP_ENABLED
    // Interop-enabled version
    CudaGLInteropBufferImpl<true>::CudaGLInteropBufferImpl()
        : buffer_id_(0),
          cuda_resource_(nullptr),
          size_(0),
          is_registered_(false),
          mapped_ptr_(nullptr) {
        LOG_TRACE("Creating CUDA-GL interop buffer");
    }

    CudaGLInteropBufferImpl<true>::~CudaGLInteropBufferImpl() {
        cleanup();
    }

    Result<void> CudaGLInteropBufferImpl<true>::init(GLuint buffer_id, size_t size) {
        LOG_TIMER_TRACE("CudaGLInteropBufferImpl<true>::init");
        LOG_DEBUG("Registering OpenGL buffer {} ({} bytes) with CUDA", buffer_id, size);

        // Clean up any existing resources
        cleanup();

        buffer_id_ = buffer_id;
        size_ = size;

        // Clear any previous CUDA errors
        cudaGetLastError();

        // Register buffer with CUDA
        cudaGraphicsResource_t raw_resource;
        cudaError_t err = cudaGraphicsGLRegisterBuffer(
            &raw_resource, buffer_id, cudaGraphicsRegisterFlagsWriteDiscard);

        if (err != cudaSuccess) {
            cleanup();
            LOG_ERROR("Failed to register OpenGL buffer with CUDA: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to register OpenGL buffer with CUDA: {}",
                                               cudaGetErrorString(err)));
        }

        cuda_resource_.reset(raw_resource);
        is_registered_ = true;

        LOG_DEBUG("CUDA-GL buffer interop initialized successfully");
        return {};
    }

    Result<void*> CudaGLInteropBufferImpl<true>::mapBuffer() {
        LOG_TIMER_TRACE("CudaGLInteropBufferImpl<true>::mapBuffer");

        if (!is_registered_) {
            LOG_ERROR("Buffer not initialized");
            return std::unexpected("Buffer not initialized");
        }

        if (mapped_ptr_) {
            LOG_WARN("Buffer already mapped, returning existing pointer");
            return mapped_ptr_;
        }

        // Map CUDA resource
        auto raw_resource = static_cast<cudaGraphicsResource_t>(cuda_resource_.get());
        cudaError_t err = cudaGraphicsMapResources(1, &raw_resource, 0);
        if (err != cudaSuccess) {
            LOG_ERROR("Failed to map CUDA resource: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to map CUDA resource: {}",
                                               cudaGetErrorString(err)));
        }

        // Get device pointer
        size_t mapped_size;
        err = cudaGraphicsResourceGetMappedPointer(&mapped_ptr_, &mapped_size, raw_resource);
        if (err != cudaSuccess) {
            cudaGraphicsUnmapResources(1, &raw_resource, 0);
            LOG_ERROR("Failed to get mapped pointer: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to get mapped pointer: {}",
                                               cudaGetErrorString(err)));
        }

        if (mapped_size < size_) {
            LOG_WARN("Mapped size {} is less than expected size {}", mapped_size, size_);
        }

        LOG_TRACE("Mapped buffer to CUDA pointer: {}", mapped_ptr_);
        return mapped_ptr_;
    }

    Result<void> CudaGLInteropBufferImpl<true>::unmapBuffer() {
        LOG_TIMER_TRACE("CudaGLInteropBufferImpl<true>::unmapBuffer");

        if (!mapped_ptr_) {
            LOG_TRACE("Buffer not mapped, nothing to do");
            return {};
        }

        auto raw_resource = static_cast<cudaGraphicsResource_t>(cuda_resource_.get());
        cudaError_t err = cudaGraphicsUnmapResources(1, &raw_resource, 0);
        if (err != cudaSuccess) {
            LOG_ERROR("Failed to unmap CUDA resource: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to unmap CUDA resource: {}",
                                               cudaGetErrorString(err)));
        }

        mapped_ptr_ = nullptr;
        LOG_TRACE("Unmapped buffer successfully");
        return {};
    }

    void CudaGLInteropBufferImpl<true>::cleanup() {
        LOG_TRACE("Cleaning up CUDA-GL interop buffer");
        if (mapped_ptr_) {
            auto result = unmapBuffer(); // Best effort
            if (!result) {
                LOG_DEBUG("Failed to unmap buffer (non-critical): {}", result.error());
            }
        }
        cuda_resource_.reset();
        is_registered_ = false;
        buffer_id_ = 0;
        size_ = 0;
    }
#endif // CUDA_GL_INTEROP_ENABLED

    // InteropFrameBuffer implementation
    InteropFrameBuffer::InteropFrameBuffer(bool use_interop)
        : FrameBuffer(),
          use_interop_(use_interop) {
        LOG_DEBUG("Creating InteropFrameBuffer with interop: {}", use_interop);
    }

    Result<void> InteropFrameBuffer::uploadFromCUDA(const Tensor& cuda_image) {
        LOG_TIMER_TRACE("InteropFrameBuffer::uploadFromCUDA");

        // Lazy initialization on first use with actual image dimensions
        if (use_interop_ && !interop_texture_) {
            // Determine dimensions from the tensor
            int img_width, img_height;

            if (cuda_image.ndim() == 3) {
                if (cuda_image.size(2) == 3 || cuda_image.size(2) == 4) {
                    // [H, W, C] format
                    img_height = cuda_image.size(0);
                    img_width = cuda_image.size(1);
                } else {
                    // [C, H, W] format
                    img_height = cuda_image.size(1);
                    img_width = cuda_image.size(2);
                }
            } else {
                LOG_ERROR("Unexpected tensor dimensions: {}", cuda_image.ndim());
                use_interop_ = false;
            }

            if (use_interop_ && img_width > 0 && img_height > 0) {
                LOG_DEBUG("Lazy-initializing CUDA-GL interop with size {}x{}", img_width, img_height);
                interop_texture_.emplace();
                if (auto result = interop_texture_->init(img_width, img_height); !result) {
                    LOG_WARN("Failed to initialize CUDA-GL interop: {}", result.error());
                    LOG_INFO("Falling back to CPU copy mode");
                    interop_texture_.reset();
                    use_interop_ = false;
                }
            }
        }

        if (!use_interop_ || !interop_texture_) {
            // Fallback to CPU copy
            LOG_TRACE("Using CPU fallback for CUDA upload");
            auto cpu_image = cuda_image;
            if (cuda_image.device() == lfs::core::Device::CUDA) {
                cpu_image = cuda_image.cpu();
            }
            cpu_image = cpu_image.contiguous();

            // Handle both [H, W, C] and [C, H, W] formats
            Tensor formatted;
            size_t last_dim_size = cpu_image.size(cpu_image.ndim() - 1);
            if (last_dim_size == 3 || last_dim_size == 4) {
                // Already [H, W, C]
                formatted = cpu_image;
            } else {
                // Convert [C, H, W] to [H, W, C]
                formatted = cpu_image.permute({1, 2, 0}).contiguous();
            }

            // Convert to uint8 if needed
            if (formatted.dtype() != lfs::core::DataType::UInt8) {
                formatted = (formatted.clamp(0.0f, 1.0f) * 255.0f).to(lfs::core::DataType::UInt8);
            }

            uploadImage(formatted.ptr<unsigned char>(),
                        formatted.size(1), formatted.size(0));
            return {};
        }

        // Direct CUDA update
        LOG_TRACE("Using direct CUDA-GL interop update");
        auto result = interop_texture_->updateFromTensor(cuda_image);
        if (!result) {
            LOG_WARN("CUDA-GL interop update failed: {}", result.error());
            LOG_INFO("Falling back to CPU copy");
            use_interop_ = false;
            interop_texture_.reset();
            return uploadFromCUDA(cuda_image); // Retry with CPU fallback
        }
        return {};
    }

    Result<void> InteropFrameBuffer::uploadDepthFromCUDA(const Tensor& cuda_depth) {
        LOG_TIMER_TRACE("InteropFrameBuffer::uploadDepthFromCUDA");

#ifdef CUDA_GL_INTEROP_ENABLED
        // Lazy initialization
        if (use_depth_interop_ && !depth_interop_texture_) {
            int depth_width = 0, depth_height = 0;
            if (cuda_depth.ndim() == 3 && cuda_depth.size(0) == 1) {
                depth_height = static_cast<int>(cuda_depth.size(1));
                depth_width = static_cast<int>(cuda_depth.size(2));
            } else if (cuda_depth.ndim() == 2) {
                depth_height = static_cast<int>(cuda_depth.size(0));
                depth_width = static_cast<int>(cuda_depth.size(1));
            } else {
                use_depth_interop_ = false;
            }

            if (use_depth_interop_ && depth_width > 0 && depth_height > 0) {
                depth_interop_texture_.emplace();
                if (auto result = depth_interop_texture_->initForDepth(depth_width, depth_height); !result) {
                    LOG_WARN("Depth interop init failed: {}", result.error());
                    depth_interop_texture_.reset();
                    use_depth_interop_ = false;
                }
            }
        }

        if (use_depth_interop_ && depth_interop_texture_) {
            if (auto result = depth_interop_texture_->updateDepthFromTensor(cuda_depth); !result) {
                LOG_WARN("Depth interop update failed, falling back to CPU");
                use_depth_interop_ = false;
                depth_interop_texture_.reset();
                return uploadDepthFromCUDA(cuda_depth);
            }
            return {};
        }
#endif

        // CPU fallback
        auto depth_cpu = cuda_depth.cpu().contiguous();
        if (depth_cpu.ndim() == 3 && depth_cpu.size(0) == 1) {
            depth_cpu = depth_cpu.squeeze(0);
        }
        uploadDepth(depth_cpu.ptr<float>(), static_cast<int>(depth_cpu.size(1)),
                    static_cast<int>(depth_cpu.size(0)));
        return {};
    }

    void InteropFrameBuffer::resize(const int new_width, const int new_height) {
        FrameBuffer::resize(new_width, new_height);
        if (use_interop_ && interop_texture_) {
            if (auto result = interop_texture_->resize(new_width, new_height); !result) {
                LOG_WARN("Interop resize failed: {}", result.error());
                use_interop_ = false;
                interop_texture_.reset();
            }
        }
        // depth_interop_texture_ resizes lazily on next upload
    }

} // namespace lfs::rendering