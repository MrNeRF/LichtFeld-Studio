/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::onnx_vulkan {

    enum class ElementType : std::uint8_t {
        Float32 = 1,
        Int32 = 6,
        Int64 = 7,
        Bool = 9,
    };

    [[nodiscard]] constexpr std::size_t element_size(const ElementType type) noexcept {
        switch (type) {
        case ElementType::Float32:
        case ElementType::Int32: return 4;
        case ElementType::Int64: return 8;
        case ElementType::Bool: return 1;
        }
        return 0;
    }

    struct ValueInfo {
        std::string name;
        ElementType type = ElementType::Float32;
        // A negative extent denotes a symbolic or otherwise dynamic dimension.
        std::vector<std::int64_t> shape;
    };

    struct TensorView {
        ElementType type = ElementType::Float32;
        std::span<const std::int64_t> shape;
        std::span<const std::byte> bytes;

        template <typename T>
        [[nodiscard]] std::span<const T> data_as() const noexcept {
            if (bytes.size() % sizeof(T) != 0)
                return {};
            return {reinterpret_cast<const T*>(bytes.data()), bytes.size() / sizeof(T)};
        }
    };

    class Tensor {
    public:
        Tensor() = default;
        Tensor(ElementType type, std::vector<std::int64_t> shape, std::vector<std::byte> bytes);

        template <typename T>
        [[nodiscard]] static Tensor from_vector(const ElementType type,
                                                std::vector<std::int64_t> shape,
                                                std::vector<T> values) {
            std::vector<std::byte> bytes(values.size() * sizeof(T));
            if (!bytes.empty())
                std::memcpy(bytes.data(), values.data(), bytes.size());
            return Tensor(type, std::move(shape), std::move(bytes));
        }

        [[nodiscard]] ElementType type() const noexcept { return type_; }
        [[nodiscard]] std::span<const std::int64_t> shape() const noexcept { return shape_; }
        [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return bytes_; }
        [[nodiscard]] TensorView view() const noexcept { return {type_, shape_, bytes_}; }

        template <typename T>
        [[nodiscard]] std::span<const T> data_as() const noexcept {
            return view().data_as<T>();
        }

    private:
        ElementType type_ = ElementType::Float32;
        std::vector<std::int64_t> shape_;
        std::vector<std::byte> bytes_;
    };

    struct NamedTensorView {
        std::string_view name;
        TensorView tensor;
    };

    struct NamedTensor {
        std::string name;
        Tensor tensor;
    };

    enum class ErrorCode : std::uint8_t {
        Io,
        MalformedModel,
        UnsupportedModel,
        InvalidInput,
        VulkanUnavailable,
        VulkanFailure,
        ExecutionFailure,
    };

    struct Error {
        ErrorCode code = ErrorCode::ExecutionFailure;
        std::string message;
        std::string node_name;
        std::string capability;
    };

    struct SessionOptions {
        std::optional<std::uint32_t> vulkan_device;
        std::filesystem::path pipeline_cache_path;
        std::uint64_t max_model_bytes = 2ull * 1024ull * 1024ull * 1024ull;
        std::uint64_t max_external_data_bytes = 2ull * 1024ull * 1024ull * 1024ull;
        bool enable_profiling = false;
        bool enable_cooperative_matrix = true;
    };

    class VulkanSession final {
    public:
        VulkanSession(VulkanSession&&) noexcept;
        VulkanSession& operator=(VulkanSession&&) noexcept;
        ~VulkanSession();

        VulkanSession(const VulkanSession&) = delete;
        VulkanSession& operator=(const VulkanSession&) = delete;

        [[nodiscard]] static std::expected<VulkanSession, Error>
        create(const std::filesystem::path& model_path, SessionOptions options = {});

        [[nodiscard]] std::span<const ValueInfo> inputs() const noexcept;
        [[nodiscard]] std::span<const ValueInfo> outputs() const noexcept;
        [[nodiscard]] std::string_view device_name() const noexcept;

        [[nodiscard]] std::expected<std::vector<NamedTensor>, Error>
        run(std::span<const NamedTensorView> named_inputs,
            std::span<const std::string_view> requested_outputs = {});

    private:
        struct Impl;
        explicit VulkanSession(std::unique_ptr<Impl> impl) noexcept;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::onnx_vulkan
