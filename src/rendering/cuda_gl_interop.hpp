/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor_fwd.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <optional>

// Forward declare GLuint to avoid including OpenGL headers
typedef unsigned int GLuint;

// Forward declare CUDA functions
namespace lfs {
#ifdef CUDA_GL_INTEROP_ENABLED
    // Write interleaved position+color data directly to mapped VBO
    void launchWriteInterleavedPosColor(
        const float* positions,
        const float* colors,
        float* output,
        int num_points,
        cudaStream_t stream);
#endif
} // namespace lfs

// Include framebuffer after forward declarations
#include "framebuffer.hpp"

namespace lfs::rendering {

    // Import Tensor from lfs::core
    using lfs::core::Tensor;

    // Forward declaration for CUDA graphics resource
    struct CudaGraphicsResourceDeleter {
        void operator()(void* resource) const;
    };

    using CudaGraphicsResourcePtr = std::unique_ptr<void, CudaGraphicsResourceDeleter>;

    // Template declaration only - no implementation
    template <bool EnableInterop>
    class CudaGLInteropTextureImpl;

    // Full specialization for disabled interop
    template <>
    class CudaGLInteropTextureImpl<false> {
        GLuint texture_id_ = 0;
        int width_ = 0;
        int height_ = 0;

    public:
        CudaGLInteropTextureImpl() = default;
        ~CudaGLInteropTextureImpl();

        Result<void> init(int width, int height);
        Result<void> resize(int new_width, int new_height);
        Result<void> updateFromTensor(const Tensor& image);
        GLuint getTextureID() const { return texture_id_; }

    private:
        void cleanup();
    };

    // Full specialization for enabled interop
    template <>
    class CudaGLInteropTextureImpl<true> {
        GLuint texture_id_ = 0;
        CudaGraphicsResourcePtr cuda_resource_;
        int width_ = 0;
        int height_ = 0;
        bool is_registered_ = false;
        bool is_depth_format_ = false;  // True if R32F, false if RGBA8

    public:
        CudaGLInteropTextureImpl();
        ~CudaGLInteropTextureImpl();

        Result<void> init(int width, int height);
        Result<void> initForDepth(int width, int height);  // R32F format for depth
        Result<void> initForReading(GLuint texture_id, int width, int height);
        Result<void> resize(int new_width, int new_height);
        Result<void> updateFromTensor(const Tensor& image);
        Result<void> updateDepthFromTensor(const Tensor& depth);  // Single-channel float
        Result<void> readToTensor(Tensor& output);
        GLuint getTextureID() const { return texture_id_; }

    private:
        void cleanup();
    };

    // Type alias based on compile-time configuration
#ifdef CUDA_GL_INTEROP_ENABLED
    using CudaGLInteropTexture = CudaGLInteropTextureImpl<true>;
#else
    using CudaGLInteropTexture = CudaGLInteropTextureImpl<false>;
#endif

    // CUDA-GL Buffer Interop (for VBOs)
    template <bool EnableInterop>
    class CudaGLInteropBufferImpl;

    // Specialization for disabled interop
    template <>
    class CudaGLInteropBufferImpl<false> {
        GLuint buffer_id_ = 0;
        size_t size_ = 0;

    public:
        CudaGLInteropBufferImpl() = default;
        ~CudaGLInteropBufferImpl();

        Result<void> init(GLuint buffer_id, size_t size);
        Result<void*> mapBuffer();
        Result<void> unmapBuffer();
        GLuint getBufferID() const { return buffer_id_; }

    private:
        void cleanup();
    };

    // Specialization for enabled interop
    template <>
    class CudaGLInteropBufferImpl<true> {
        GLuint buffer_id_ = 0;
        CudaGraphicsResourcePtr cuda_resource_;
        size_t size_ = 0;
        bool is_registered_ = false;
        void* mapped_ptr_ = nullptr;

    public:
        CudaGLInteropBufferImpl();
        ~CudaGLInteropBufferImpl();

        Result<void> init(GLuint buffer_id, size_t size);
        Result<void*> mapBuffer();
        Result<void> unmapBuffer();
        GLuint getBufferID() const { return buffer_id_; }

    private:
        void cleanup();
    };

#ifdef CUDA_GL_INTEROP_ENABLED
    using CudaGLInteropBuffer = CudaGLInteropBufferImpl<true>;
#else
    using CudaGLInteropBuffer = CudaGLInteropBufferImpl<false>;
#endif

    // Modified FrameBuffer to support interop
    class InteropFrameBuffer : public FrameBuffer {
        std::optional<CudaGLInteropTexture> interop_texture_;
        std::optional<CudaGLInteropTexture> depth_interop_texture_;  // For depth upload
        bool use_interop_;
        bool use_depth_interop_ = true;

    public:
        explicit InteropFrameBuffer(bool use_interop = true);

        Result<void> uploadFromCUDA(const Tensor& cuda_image);
        Result<void> uploadDepthFromCUDA(const Tensor& cuda_depth);  // Direct CUDA→GL depth

        GLuint getInteropTexture() const {
            return use_interop_ && interop_texture_ ? interop_texture_->getTextureID() : getFrameTexture();
        }

        GLuint getDepthInteropTexture() const {
            return use_depth_interop_ && depth_interop_texture_ ? depth_interop_texture_->getTextureID() : getDepthTexture();
        }

        bool hasDepthInterop() const {
            return use_depth_interop_ && depth_interop_texture_.has_value();
        }

        void resize(int new_width, int new_height) override;
    };

} // namespace lfs::rendering