/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_serialization.hpp"

#include "core/contiguous_streambuf.hpp"
#include "core/cuda_error.hpp"
#include "core/path_utils.hpp"
#include "core/tensor_serialization_sink.hpp"

#include <chrono>
#include <fstream>
#include <ios>
#include <limits>
#include <string_view>
#include <vector>

namespace lfs::core {

    namespace {
        thread_local TensorSerializationSink*
            active_tensor_serialization_sink = nullptr;
        thread_local serialization_detail::TensorLoadTiming*
            active_tensor_load_timing = nullptr;
    } // namespace

    namespace serialization_detail {
        TensorLoadTimingScope::TensorLoadTimingScope(
            TensorLoadTiming& timing) noexcept
            : previous_(active_tensor_load_timing) {
            active_tensor_load_timing = &timing;
        }

        TensorLoadTimingScope::~TensorLoadTimingScope() {
            active_tensor_load_timing = previous_;
        }
    } // namespace serialization_detail

    std::uint64_t
    TensorSerializationDescriptor::payload_bytes() const {
        const auto elements = serialized_shape.elements();
        const auto element_bytes = dtype_size(dtype);
        if (element_bytes == 0 ||
            elements >
                std::numeric_limits<std::uint64_t>::max() /
                    element_bytes) {
            throw std::overflow_error(
                "Serialized tensor payload size overflows");
        }
        return static_cast<std::uint64_t>(elements) *
               element_bytes;
    }

    TensorSerializationSinkScope::TensorSerializationSinkScope(
        TensorSerializationSink& sink) noexcept
        : previous_(active_tensor_serialization_sink) {
        active_tensor_serialization_sink = &sink;
    }

    TensorSerializationSinkScope::~TensorSerializationSinkScope() {
        active_tensor_serialization_sink = previous_;
    }

    TensorSerializationSink*
    current_tensor_serialization_sink() noexcept {
        return active_tensor_serialization_sink;
    }

    ContiguousStreambuf::~ContiguousStreambuf() = default;

    namespace serialization_detail {
        void read_exact(std::istream& is,
                        void* const destination,
                        const std::size_t bytes,
                        const std::string_view field) {
            if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
                throw std::runtime_error("Serialized " + std::string(field) + " exceeds stream limits");
            }
            if (bytes == 0) {
                return;
            }
            is.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
            if (!is) {
                throw std::runtime_error("Truncated serialized " + std::string(field));
            }
        }

        void require_remaining_bytes(std::istream& is,
                                     const uint64_t required,
                                     const std::string_view field) {
            const auto current = is.tellg();
            if (current == std::streampos(-1)) {
                return;
            }

            is.seekg(0, std::ios::end);
            const auto end = is.tellg();
            is.seekg(current);
            if (!is || end == std::streampos(-1) || end < current ||
                static_cast<uint64_t>(end - current) < required) {
                throw std::runtime_error("Truncated serialized " + std::string(field));
            }
        }
    } // namespace serialization_detail

    void serialize_tensor_with_descriptor(
        std::ostream& os,
        const Tensor& tensor,
        const TensorSerializationDescriptor& descriptor,
        const Tensor* auxiliary_source) {
        if (!tensor.is_valid()) {
            throw std::runtime_error("Cannot serialize invalid tensor");
        }

        const TensorFileHeader header{
            TENSOR_FILE_MAGIC,
            TENSOR_FILE_VERSION,
            static_cast<uint8_t>(descriptor.dtype),
            static_cast<uint8_t>(descriptor.serialized_device),
            static_cast<uint16_t>(
                descriptor.serialized_shape.rank()),
            descriptor.serialized_shape.elements()};
        os.write(reinterpret_cast<const char*>(&header), sizeof(header));

        for (const size_t dim :
             descriptor.serialized_shape.dims()) {
            const uint64_t d = dim;
            os.write(reinterpret_cast<const char*>(&d), sizeof(d));
        }

        if (auto* sink = current_tensor_serialization_sink()) {
            sink->write_tensor_payload(
                os, tensor, auxiliary_source, descriptor);
            if (!os) {
                throw std::runtime_error(
                    "Failed to write tensor through snapshot sink");
            }
            return;
        }
        if (descriptor.encoding !=
                TensorPayloadEncoding::NativeContiguous ||
            auxiliary_source) {
            throw std::runtime_error(
                "Alternate tensor encoding requires a serialization sink");
        }
        const Tensor host = tensor.device() == Device::CUDA ? tensor.cpu() : tensor;
        const Tensor src = host.is_contiguous() ? host : host.contiguous();
        if (src.dtype() != descriptor.dtype ||
            src.shape() != descriptor.serialized_shape) {
            throw std::runtime_error(
                "Serialized tensor descriptor does not match source");
        }
        os.write(reinterpret_cast<const char*>(src.data_ptr()), src.bytes());

        if (!os) {
            throw std::runtime_error("Failed to write tensor");
        }
    }

    std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        serialize_tensor_with_descriptor(
            os, tensor,
            TensorSerializationDescriptor{
                .serialized_shape = tensor.shape(),
                .dtype = tensor.dtype(),
                .serialized_device = tensor.device(),
                .encoding =
                    TensorPayloadEncoding::NativeContiguous,
            });
        return os;
    }

    namespace {

        struct ParsedTensorPayload {
            DataType dtype = DataType::Float32;
            TensorShape shape;
            uint64_t payload_bytes = 0;
        };

        ParsedTensorPayload parse_serialized_tensor_header(std::istream& is) {
            TensorFileHeader header{};
            serialization_detail::read_exact(is, &header, sizeof(header), "tensor header");

            if (header.magic != TENSOR_FILE_MAGIC) {
                throw std::runtime_error("Invalid tensor file: wrong magic number");
            }
            if (header.version != TENSOR_FILE_VERSION) {
                throw std::runtime_error("Unsupported tensor file version");
            }
            if (header.rank > MAX_TENSOR_RANK) {
                throw std::runtime_error("Invalid tensor file: rank exceeds supported maximum");
            }
            if (header.dtype > static_cast<uint8_t>(DataType::Bool)) {
                throw std::runtime_error("Invalid tensor file: unsupported dtype");
            }
            if (header.device > static_cast<uint8_t>(Device::CUDA)) {
                throw std::runtime_error("Invalid tensor file: unsupported device");
            }

            std::vector<size_t> dims(header.rank);
            uint64_t checked_numel = 1;
            for (uint16_t i = 0; i < header.rank; ++i) {
                uint64_t d = 0;
                serialization_detail::read_exact(is, &d, sizeof(d), "tensor dimension");
                if (d > std::numeric_limits<size_t>::max()) {
                    throw std::runtime_error("Invalid tensor file: dimension exceeds platform size");
                }
                if (d != 0 && checked_numel > std::numeric_limits<uint64_t>::max() / d) {
                    throw std::runtime_error("Invalid tensor file: shape element count overflows");
                }
                checked_numel *= d;
                dims[i] = static_cast<size_t>(d);
            }

            const DataType dtype = static_cast<DataType>(header.dtype);
            if (checked_numel != header.numel) {
                throw std::runtime_error("Shape elements mismatch");
            }
            const auto item_size = dtype_size(dtype);
            if (item_size == 0 ||
                header.numel > std::numeric_limits<uint64_t>::max() / item_size) {
                throw std::runtime_error("Invalid tensor file: byte size overflows");
            }
            const uint64_t payload_bytes = header.numel * item_size;
            if (payload_bytes > MAX_SERIALIZED_TENSOR_BYTES) {
                throw std::runtime_error("Invalid tensor file: payload exceeds byte budget");
            }
            serialization_detail::require_remaining_bytes(is, payload_bytes, "tensor payload");
            return ParsedTensorPayload{
                .dtype = dtype,
                .shape = TensorShape(dims),
                .payload_bytes = payload_bytes,
            };
        }

        void seek_serialized_payload(std::istream& is, const uint64_t payload_bytes) {
            if (payload_bytes == 0) {
                return;
            }
            if (payload_bytes >
                static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
                throw std::runtime_error("Serialized tensor payload exceeds streamoff");
            }
            is.seekg(static_cast<std::streamoff>(payload_bytes), std::ios::cur);
            if (!is) {
                throw std::runtime_error("Failed to skip serialized tensor payload");
            }
        }

    } // namespace

    namespace serialization_detail {
        void skip_serialized_tensor(std::istream& is) {
            const auto parsed = parse_serialized_tensor_header(is);
            seek_serialized_payload(is, parsed.payload_bytes);
        }
    } // namespace serialization_detail

    namespace {
        constexpr std::uint64_t kPageableSerializedHostTensorBytes =
            256ull * 1024ull * 1024ull;

        void load_parsed_serialized_tensor(
            std::istream& is, Tensor& tensor, const ParsedTensorPayload& parsed,
            const bool use_pinned) {
            auto* const timing = active_tensor_load_timing;
            const auto run_timed =
                [timing](double serialization_detail::TensorLoadTiming::* member,
                         auto&& fn) {
                    if (timing == nullptr) {
                        fn();
                        return;
                    }
                    const auto started = std::chrono::steady_clock::now();
                    fn();
                    timing->*member +=
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - started)
                            .count();
                };
            Tensor loaded;
            run_timed(&serialization_detail::TensorLoadTiming::alloc_ms, [&] {
                loaded = Tensor::empty(parsed.shape, Device::CPU, parsed.dtype,
                                       use_pinned);
            });
            run_timed(&serialization_detail::TensorLoadTiming::read_ms, [&] {
                serialization_detail::read_exact(
                    is, loaded.data_ptr(), loaded.bytes(), "tensor payload");
            });
            tensor = std::move(loaded);
        }
    } // namespace

    namespace serialization_detail {
        void read_serialized_tensor(std::istream& is, Tensor& tensor,
                                    const bool use_pinned) {
            const auto parsed = parse_serialized_tensor_header(is);
            load_parsed_serialized_tensor(is, tensor, parsed, use_pinned);
        }

        void read_serialized_tensor_pageable_if_large(std::istream& is,
                                                      Tensor& tensor) {
            const auto parsed = parse_serialized_tensor_header(is);
            const bool use_pinned =
                parsed.payload_bytes < kPageableSerializedHostTensorBytes;
            load_parsed_serialized_tensor(is, tensor, parsed, use_pinned);
        }

        void read_serialized_tensor_device_from_span_or_host(
            std::istream& is, Tensor& tensor, const cudaStream_t stream) {
            const auto parsed = parse_serialized_tensor_header(is);
            const auto* const span_buf =
                dynamic_cast<const ContiguousStreambuf*>(is.rdbuf());
            if (span_buf != nullptr &&
                parsed.payload_bytes <=
                    std::numeric_limits<std::size_t>::max()) {
                const auto remaining = span_buf->remaining_bytes();
                if (remaining.size() <
                    static_cast<std::size_t>(parsed.payload_bytes)) {
                    throw std::runtime_error("Truncated serialized tensor payload");
                }
                auto* const timing = active_tensor_load_timing;
                const auto run_timed =
                    [timing](double TensorLoadTiming::* member, auto&& fn) {
                        if (timing == nullptr) {
                            fn();
                            return;
                        }
                        const auto started = std::chrono::steady_clock::now();
                        fn();
                        timing->*member +=
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - started)
                                .count();
                    };
                Tensor loaded;
                run_timed(&TensorLoadTiming::alloc_ms, [&] {
                    loaded = Tensor::empty(parsed.shape, Device::CUDA,
                                           parsed.dtype);
                    loaded.set_stream(stream);
                });
                if (parsed.payload_bytes > 0) {
                    run_timed(&TensorLoadTiming::read_ms, [&] {
                        LFS_CUDA_CHECK_MSG_STREAM_ARGS(
                            cudaMemcpyAsync(
                                loaded.data_ptr(), remaining.data(),
                                static_cast<std::size_t>(parsed.payload_bytes),
                                cudaMemcpyHostToDevice, stream),
                            stream,
                            reinterpret_cast<uintptr_t>(loaded.data_ptr()),
                            reinterpret_cast<uintptr_t>(remaining.data()),
                            static_cast<std::size_t>(parsed.payload_bytes),
                            "while uploading serialized tensor payload shape={} dtype={} to CUDA",
                            parsed.shape.str(), dtype_name(parsed.dtype));
                        loaded.record_stream(stream);
                    });
                }
                seek_serialized_payload(is, parsed.payload_bytes);
                tensor = std::move(loaded);
                return;
            }
            const bool use_pinned =
                parsed.payload_bytes < kPageableSerializedHostTensorBytes;
            load_parsed_serialized_tensor(is, tensor, parsed, use_pinned);
        }
    } // namespace serialization_detail

    std::istream& operator>>(std::istream& is, Tensor& tensor) {
        serialization_detail::read_serialized_tensor(is, tensor, true);
        return is;
    }

    void save_tensor(const Tensor& tensor, const std::string& filename) {
        std::ofstream file;
        if (!open_file_for_write(utf8_to_path(filename), std::ios::binary, file)) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        file << tensor;
    }

    Tensor load_tensor(const std::string& filename) {
        std::ifstream file;
        if (!open_file_for_read(utf8_to_path(filename), std::ios::binary, file)) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        Tensor tensor;
        file >> tensor;
        return tensor;
    }

} // namespace lfs::core
