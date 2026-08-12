/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs_onnx_vulkan/onnx_vulkan.hpp"
#include "model.hpp"
#include "operator_registry.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
using Bytes = std::vector<std::byte>;

namespace {
    struct Failure final : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    void require(const bool condition, const std::string_view message) {
        if (!condition)
            throw Failure(std::string(message));
    }

    namespace proto {
        void varint(Bytes& output, std::uint64_t value) {
            while (value >= 0x80) {
                output.push_back(static_cast<std::byte>((value & 0x7f) | 0x80));
                value >>= 7;
            }
            output.push_back(static_cast<std::byte>(value));
        }

        void key(Bytes& output, const std::uint32_t field, const std::uint32_t wire) {
            varint(output, (static_cast<std::uint64_t>(field) << 3) | wire);
        }

        void integer(Bytes& output, const std::uint32_t field, const std::int64_t value) {
            key(output, field, 0);
            varint(output, static_cast<std::uint64_t>(value));
        }

        void bytes(Bytes& output, const std::uint32_t field, const std::span<const std::byte> value) {
            key(output, field, 2);
            varint(output, value.size());
            output.insert(output.end(), value.begin(), value.end());
        }

        void string(Bytes& output, const std::uint32_t field, const std::string_view value) {
            bytes(output, field, std::as_bytes(std::span(value)));
        }

        void fixed32(Bytes& output, const std::uint32_t field, const std::uint32_t value) {
            key(output, field, 5);
            for (int shift = 0; shift < 32; shift += 8)
                output.push_back(static_cast<std::byte>((value >> shift) & 0xff));
        }

        [[nodiscard]] Bytes dimension(const std::int64_t extent) {
            Bytes result;
            integer(result, 1, extent);
            return result;
        }

        [[nodiscard]] Bytes value_info(const std::string_view name,
                                       const std::int64_t element_type,
                                       const std::span<const std::int64_t> shape) {
            Bytes shape_message;
            for (const auto extent : shape) {
                const auto dim = dimension(extent);
                bytes(shape_message, 1, dim);
            }
            Bytes tensor_type;
            integer(tensor_type, 1, element_type);
            bytes(tensor_type, 2, shape_message);
            Bytes type;
            bytes(type, 1, tensor_type);
            Bytes result;
            string(result, 1, name);
            bytes(result, 2, type);
            return result;
        }

        template <typename T>
        [[nodiscard]] Bytes raw_tensor(const std::string_view name,
                                       const std::int64_t element_type,
                                       const std::span<const std::int64_t> shape,
                                       const std::span<const T> values) {
            Bytes result;
            for (const auto extent : shape)
                integer(result, 1, extent);
            integer(result, 2, element_type);
            string(result, 8, name);
            bytes(result, 9, std::as_bytes(values));
            return result;
        }

        [[nodiscard]] Bytes typed_float_tensor(const std::string_view name,
                                               const std::span<const std::int64_t> shape,
                                               const std::span<const float> values) {
            Bytes result;
            for (const auto extent : shape)
                integer(result, 1, extent);
            integer(result, 2, 1);
            for (const float value : values)
                fixed32(result, 4, std::bit_cast<std::uint32_t>(value));
            string(result, 8, name);
            return result;
        }

        [[nodiscard]] Bytes external_entry(const std::string_view key_value,
                                           const std::string_view value) {
            Bytes result;
            string(result, 1, key_value);
            string(result, 2, value);
            return result;
        }

        [[nodiscard]] Bytes external_tensor(const std::string_view name,
                                            const std::span<const std::int64_t> shape,
                                            const std::string_view location,
                                            const std::string_view offset,
                                            const std::string_view length) {
            Bytes result;
            for (const auto extent : shape)
                integer(result, 1, extent);
            integer(result, 2, 1);
            string(result, 8, name);
            for (const auto& [key_value, value] : std::array{
                     std::pair{std::string_view{"location"}, location},
                     std::pair{std::string_view{"offset"}, offset},
                     std::pair{std::string_view{"length"}, length}}) {
                const auto entry = external_entry(key_value, value);
                bytes(result, 13, entry);
            }
            integer(result, 14, 1);
            return result;
        }

        [[nodiscard]] Bytes int_attribute(const std::string_view name, const std::int64_t value) {
            Bytes result;
            string(result, 1, name);
            integer(result, 3, value);
            integer(result, 20, 2);
            return result;
        }

        [[nodiscard]] Bytes ints_attribute(const std::string_view name,
                                           const std::span<const std::int64_t> values) {
            Bytes result;
            string(result, 1, name);
            for (const auto value : values)
                integer(result, 8, value);
            integer(result, 20, 7);
            return result;
        }

        [[nodiscard]] Bytes graph_attribute(const std::string_view name, const Bytes& graph_value) {
            Bytes result;
            string(result, 1, name);
            bytes(result, 6, graph_value);
            integer(result, 20, 5);
            return result;
        }

        [[nodiscard]] Bytes node(const std::string_view name,
                                 const std::string_view op,
                                 const std::span<const std::string_view> inputs,
                                 const std::span<const std::string_view> outputs,
                                 const std::span<const Bytes> attributes = {}) {
            Bytes result;
            for (const auto input : inputs)
                string(result, 1, input);
            for (const auto output : outputs)
                string(result, 2, output);
            string(result, 3, name);
            string(result, 4, op);
            for (const auto& attribute : attributes)
                bytes(result, 5, attribute);
            return result;
        }

        [[nodiscard]] Bytes graph(const std::span<const Bytes> nodes,
                                  const std::span<const Bytes> initializers,
                                  const std::span<const Bytes> inputs,
                                  const std::span<const Bytes> outputs) {
            Bytes result;
            for (const auto& node_value : nodes)
                bytes(result, 1, node_value);
            string(result, 2, "test");
            for (const auto& initializer : initializers)
                bytes(result, 5, initializer);
            for (const auto& input : inputs)
                bytes(result, 11, input);
            for (const auto& output : outputs)
                bytes(result, 12, output);
            return result;
        }

        [[nodiscard]] Bytes model(const Bytes& graph_value,
                                  const std::int64_t opset = 14,
                                  const std::string_view domain = {}) {
            Bytes opset_message;
            if (!domain.empty())
                string(opset_message, 1, domain);
            integer(opset_message, 2, opset);
            Bytes result;
            integer(result, 1, 7);
            bytes(result, 7, graph_value);
            bytes(result, 8, opset_message);
            return result;
        }
    }

    class TemporaryDirectory {
    public:
        TemporaryDirectory() {
            const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            path_ = fs::temp_directory_path() / ("lfs-onnx-tests-" + std::to_string(stamp));
            fs::create_directories(path_);
        }
        ~TemporaryDirectory() {
            std::error_code ignored;
            fs::remove_all(path_, ignored);
        }
        [[nodiscard]] const fs::path& path() const noexcept { return path_; }

    private:
        fs::path path_;
    };

    void write_file(const fs::path& path, const std::span<const std::byte> bytes) {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream)
            throw Failure("failed to write test fixture");
    }

    [[nodiscard]] Bytes identity_model() {
        const std::array<std::int64_t, 1> shape{2};
        const auto input = proto::value_info("x", 1, shape);
        const auto output = proto::value_info("y", 1, shape);
        const std::array<std::string_view, 1> node_inputs{"x"};
        const std::array<std::string_view, 1> node_outputs{"y"};
        const auto identity = proto::node("identity", "Identity", node_inputs, node_outputs);
        const std::array nodes{identity};
        const std::array inputs{input};
        const std::array outputs{output};
        return proto::model(proto::graph(nodes, {}, inputs, outputs));
    }

    [[nodiscard]] lfs::onnx_vulkan::detail::Model parse_and_validate(const fs::path& path) {
        auto model = lfs::onnx_vulkan::detail::parse_model(path, {});
        require(model.has_value(), model ? "" : model.error().message);
        auto valid = lfs::onnx_vulkan::detail::validate_model(*model);
        require(valid.has_value(), valid ? "" : valid.error().message);
        return std::move(*model);
    }

    void test_parser(const TemporaryDirectory& temporary) {
        auto valid_bytes = identity_model();
        proto::integer(valid_bytes, 99, 7);
        const auto valid_path = temporary.path() / "valid.onnx";
        write_file(valid_path, valid_bytes);
        const auto valid = parse_and_validate(valid_path);
        require(valid.ir_version == 7 && valid.opsets.at("") == 14, "IR/opset was parsed incorrectly");
        require(valid.graph.nodes.size() == 1 && valid.graph.nodes.front().op_type == "Identity",
                "valid graph contents were parsed incorrectly");

        const std::array<std::int64_t, 1> shape{2};
        const std::array<float, 2> values{1.25f, -2.5f};
        const auto raw = proto::raw_tensor("raw", 1, shape, std::span<const float>(values));
        const auto typed = proto::typed_float_tensor("typed", shape, values);
        const auto input = proto::value_info("x", 1, shape);
        const auto output = proto::value_info("y", 1, shape);
        const std::array<std::string_view, 2> add_inputs{"raw", "typed"};
        const std::array<std::string_view, 1> add_outputs{"y"};
        const auto add = proto::node("add", "Add", add_inputs, add_outputs);
        const std::array nodes{add};
        const std::array initializers{raw, typed};
        const std::array inputs{input};
        const std::array outputs{output};
        const auto tensors_path = temporary.path() / "tensors.onnx";
        write_file(tensors_path, proto::model(proto::graph(nodes, initializers, inputs, outputs)));
        const auto tensors = parse_and_validate(tensors_path);
        require(tensors.graph.initializers.size() == 2, "raw/typed initializers were not parsed");
        for (const auto& initializer : tensors.graph.initializers) {
            require(initializer.bytes.size() == values.size() * sizeof(float), "initializer byte count is wrong");
            require(std::memcmp(initializer.bytes.data(), values.data(), initializer.bytes.size()) == 0,
                    "initializer payload is wrong");
        }

        const std::array<float, 4> sidecar_values{9.0f, 3.0f, 4.0f, 5.0f};
        write_file(temporary.path() / "weights.bin", std::as_bytes(std::span(sidecar_values)));
        const auto external = proto::external_tensor("external", shape, "weights.bin", "4", "8");
        const std::array<std::string_view, 2> external_add_inputs{"x", "external"};
        const auto external_add = proto::node("external_add", "Add", external_add_inputs, add_outputs);
        const std::array external_nodes{external_add};
        const std::array external_initializers{external};
        const auto external_path = temporary.path() / "external.onnx";
        write_file(external_path,
                   proto::model(proto::graph(external_nodes, external_initializers, inputs, outputs)));
        const auto external_model = parse_and_validate(external_path);
        const auto external_data = reinterpret_cast<const float*>(
            external_model.graph.initializers.front().bytes.data());
        require(external_data[0] == 3.0f && external_data[1] == 4.0f,
                "external tensor offset/length was not honored");

        const auto traversal = proto::external_tensor("external", shape, "../weights.bin", "0", "8");
        const std::array traversal_initializers{traversal};
        const auto traversal_path = temporary.path() / "traversal.onnx";
        write_file(traversal_path,
                   proto::model(proto::graph(external_nodes, traversal_initializers, inputs, outputs)));
        const auto traversal_result = lfs::onnx_vulkan::detail::parse_model(traversal_path, {});
        require(!traversal_result && traversal_result.error().message.contains("escapes"),
                "external path traversal was not rejected");

        const std::array malformed{std::byte{0x80}};
        const auto malformed_path = temporary.path() / "malformed.onnx";
        write_file(malformed_path, malformed);
        require(!lfs::onnx_vulkan::detail::parse_model(malformed_path, {}),
                "unterminated protobuf varint was accepted");

        auto truncated = identity_model();
        truncated.pop_back();
        const auto truncated_path = temporary.path() / "truncated.onnx";
        write_file(truncated_path, truncated);
        require(!lfs::onnx_vulkan::detail::parse_model(truncated_path, {}),
                "truncated protobuf data was accepted");

        const auto opset_path = temporary.path() / "opset.onnx";
        const auto identity_graph = [&] {
            const std::array<std::string_view, 1> identity_inputs{"x"};
            const std::array<std::string_view, 1> identity_outputs{"y"};
            const auto node = proto::node("identity", "Identity", identity_inputs, identity_outputs);
            const std::array nodes_value{node};
            return proto::graph(nodes_value, {}, inputs, outputs);
        }();
        write_file(opset_path, proto::model(identity_graph, 13));
        const auto wrong_opset = lfs::onnx_vulkan::detail::parse_model(opset_path, {});
        require(!wrong_opset && wrong_opset.error().capability == "ai.onnx opset 13",
                "unsupported opset diagnostic is not exact");

        Bytes overflowing_tensor;
        proto::integer(overflowing_tensor, 1, std::numeric_limits<std::int64_t>::max());
        proto::integer(overflowing_tensor, 1, std::numeric_limits<std::int64_t>::max());
        proto::integer(overflowing_tensor, 2, 1);
        proto::string(overflowing_tensor, 8, "huge");
        const std::array overflow_initializers{overflowing_tensor};
        const auto overflow_path = temporary.path() / "overflow.onnx";
        write_file(overflow_path,
                   proto::model(proto::graph(external_nodes, overflow_initializers, inputs, outputs)));
        const auto overflow = lfs::onnx_vulkan::detail::parse_model(overflow_path, {});
        require(!overflow && overflow.error().message.contains("overflows"),
                "initializer size overflow was not rejected");

        auto sparse_graph = identity_graph;
        proto::bytes(sparse_graph, 15, std::span<const std::byte>{});
        const auto sparse_path = temporary.path() / "sparse.onnx";
        write_file(sparse_path, proto::model(sparse_graph));
        const auto sparse = lfs::onnx_vulkan::detail::parse_model(sparse_path, {});
        require(!sparse && sparse.error().capability == "GraphProto.sparse_initializer",
                "sparse initializer was not rejected explicitly");

        auto training_model = identity_model();
        proto::bytes(training_model, 20, std::span<const std::byte>{});
        const auto training_path = temporary.path() / "training.onnx";
        write_file(training_path, training_model);
        const auto training = lfs::onnx_vulkan::detail::parse_model(training_path, {});
        require(!training && training.error().capability == "ModelProto.training_info",
                "training graph was not rejected explicitly");

        const auto domain_path = temporary.path() / "domain.onnx";
        auto domain_model = identity_model();
        Bytes custom_opset;
        proto::string(custom_opset, 1, "example.custom");
        proto::integer(custom_opset, 2, 1);
        proto::bytes(domain_model, 8, custom_opset);
        write_file(domain_path, domain_model);
        const auto domain = lfs::onnx_vulkan::detail::parse_model(domain_path, {});
        require(!domain && domain.error().capability == "domain example.custom",
                "custom domain was not rejected explicitly");
    }

    void test_graph_validation(const TemporaryDirectory& temporary) {
        const std::array<std::int64_t, 1> shape{2};
        const auto input = proto::value_info("x", 1, shape);
        const auto output = proto::value_info("b", 1, shape);
        const std::array<std::string_view, 1> a_inputs{"b"};
        const std::array<std::string_view, 1> a_outputs{"a"};
        const std::array<std::string_view, 1> b_inputs{"a"};
        const std::array<std::string_view, 1> b_outputs{"b"};
        const auto a = proto::node("a", "Identity", a_inputs, a_outputs);
        const auto b = proto::node("b", "Identity", b_inputs, b_outputs);
        const std::array nodes{a, b};
        const std::array inputs{input};
        const std::array outputs{output};
        const auto cycle_path = temporary.path() / "cycle.onnx";
        write_file(cycle_path, proto::model(proto::graph(nodes, {}, inputs, outputs)));
        auto cycle = lfs::onnx_vulkan::detail::parse_model(cycle_path, {});
        require(cycle.has_value(), "cycle fixture did not parse");
        const auto cycle_valid = lfs::onnx_vulkan::detail::validate_model(*cycle);
        require(!cycle_valid && cycle_valid.error().message.contains("cycle"), "graph cycle was not rejected");

        const std::array<std::string_view, 1> relu_inputs{"x"};
        const std::array<std::string_view, 1> relu_outputs{"b"};
        const auto bogus = proto::int_attribute("bogus", 1);
        const std::array attributes{bogus};
        const auto relu = proto::node("bad_relu", "Relu", relu_inputs, relu_outputs, attributes);
        const std::array invalid_nodes{relu};
        const auto attribute_path = temporary.path() / "attribute.onnx";
        write_file(attribute_path,
                   proto::model(proto::graph(invalid_nodes, {}, inputs, outputs)));
        auto invalid_attribute = lfs::onnx_vulkan::detail::parse_model(attribute_path, {});
        require(invalid_attribute.has_value(), "invalid-attribute fixture did not parse");
        const auto attribute_valid = lfs::onnx_vulkan::detail::validate_model(*invalid_attribute);
        require(!attribute_valid && attribute_valid.error().node_name == "bad_relu" &&
                    attribute_valid.error().capability == "Relu attribute bogus",
                "invalid attribute diagnostic lacks the exact node/capability");

        const auto unsupported = proto::node("missing", "NotImplemented", relu_inputs, relu_outputs);
        const std::array unsupported_nodes{unsupported};
        const auto unsupported_path = temporary.path() / "unsupported.onnx";
        write_file(unsupported_path,
                   proto::model(proto::graph(unsupported_nodes, {}, inputs, outputs)));
        auto unsupported_model = lfs::onnx_vulkan::detail::parse_model(unsupported_path, {});
        require(unsupported_model.has_value(), "unsupported-op fixture did not parse");
        const auto unsupported_valid = lfs::onnx_vulkan::detail::validate_model(*unsupported_model);
        require(!unsupported_valid && unsupported_valid.error().node_name == "missing" &&
                    unsupported_valid.error().capability == "ai.onnx::NotImplemented@14",
                "unsupported operator diagnostic lacks the exact node/capability");

        const std::array<std::string_view, 1> first_inputs{"x"};
        const std::array<std::string_view, 1> first_outputs{"a"};
        const std::array<std::string_view, 1> second_inputs{"a"};
        const std::array<std::string_view, 1> second_outputs{"b"};
        const auto first = proto::node("duplicate", "Identity", first_inputs, first_outputs);
        const auto second = proto::node("duplicate", "Identity", second_inputs, second_outputs);
        const std::array duplicate_nodes{first, second};
        const auto duplicate_path = temporary.path() / "duplicate.onnx";
        write_file(duplicate_path,
                   proto::model(proto::graph(duplicate_nodes, {}, inputs, outputs)));
        auto duplicate = lfs::onnx_vulkan::detail::parse_model(duplicate_path, {});
        require(duplicate.has_value(), "duplicate-name fixture did not parse");
        const auto duplicate_valid = lfs::onnx_vulkan::detail::validate_model(*duplicate);
        require(!duplicate_valid && duplicate_valid.error().message.contains("duplicate node name"),
                "duplicate node name was not rejected");
    }

    void test_registry() {
        constexpr std::array<std::string_view, 43> expected{
            "Add", "Cast", "Clip", "Concat", "Constant", "ConstantOfShape", "Conv",
            "ConvTranspose", "Div", "Erf", "Equal", "Exp", "Expand", "Gather", "Gemm",
            "Identity", "If", "MatMul", "Mod", "Mul", "Neg", "Pad", "Pow", "Range",
            "Reciprocal", "ReduceL2", "ReduceMean", "ReduceSum", "Relu", "Reshape", "Resize",
            "Round", "Shape", "Sigmoid", "Slice", "Softmax", "Split", "Sqrt", "Squeeze",
            "Sub", "Transpose", "Unsqueeze", "Where"};
        for (const auto name : expected)
            require(lfs::onnx_vulkan::detail::find_operator_schema(name, 14) != nullptr,
                    "MoGe operator registry is incomplete");
        require(lfs::onnx_vulkan::detail::find_operator_schema("Add", 13) == nullptr,
                "operator registry ignored the requested schema version");
        require(lfs::onnx_vulkan::detail::find_operator_schema("Loop", 14) == nullptr,
                "unsupported control-flow operator was registered");
    }

    [[nodiscard]] Bytes arithmetic_model() {
        const std::array<std::int64_t, 2> x_shape{2, 1};
        const std::array<std::int64_t, 2> b_shape{1, 3};
        const std::array<std::int64_t, 2> y_shape{2, 3};
        const std::array<float, 3> bias{10.0f, 20.0f, 30.0f};
        const auto input = proto::value_info("x", 1, x_shape);
        const auto output = proto::value_info("y", 1, y_shape);
        const auto initializer = proto::raw_tensor("b", 1, b_shape, std::span<const float>(bias));
        const std::array<std::string_view, 2> node_inputs{"x", "b"};
        const std::array<std::string_view, 1> node_outputs{"y"};
        const auto add = proto::node("broadcast_add", "Add", node_inputs, node_outputs);
        const std::array nodes{add};
        const std::array initializers{initializer};
        const std::array inputs{input};
        const std::array outputs{output};
        return proto::model(proto::graph(nodes, initializers, inputs, outputs));
    }

    [[nodiscard]] Bytes pow_clip_model() {
        const std::array<std::int64_t, 2> shape{2, 3};
        const std::array<std::int64_t, 0> scalar_shape{};
        const std::array<float, 1> exponent{3.0f};
        const std::array<float, 1> minimum{-4.0f};
        const std::array<float, 1> maximum{10.0f};
        const auto input = proto::value_info("x", 1, shape);
        const auto output = proto::value_info("y", 1, shape);
        const std::array initializers{
            proto::raw_tensor("exponent", 1, scalar_shape, std::span<const float>(exponent)),
            proto::raw_tensor("minimum", 1, scalar_shape, std::span<const float>(minimum)),
            proto::raw_tensor("maximum", 1, scalar_shape, std::span<const float>(maximum)),
        };
        const std::array<std::string_view, 2> pow_inputs{"x", "exponent"};
        const std::array<std::string_view, 1> pow_outputs{"powered"};
        const std::array<std::string_view, 3> clip_inputs{"powered", "minimum", "maximum"};
        const std::array<std::string_view, 1> clip_outputs{"y"};
        const std::array nodes{
            proto::node("integer_power", "Pow", pow_inputs, pow_outputs),
            proto::node("bounded", "Clip", clip_inputs, clip_outputs),
        };
        const std::array inputs{input};
        const std::array outputs{output};
        return proto::model(proto::graph(nodes, initializers, inputs, outputs));
    }

    [[nodiscard]] Bytes matmul_model() {
        const std::array<std::int64_t, 2> x_shape{2, 3};
        const std::array<std::int64_t, 2> weight_shape{3, 2};
        const std::array<std::int64_t, 2> y_shape{2, 2};
        const std::array<float, 6> weights{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        const auto input = proto::value_info("x", 1, x_shape);
        const auto output = proto::value_info("y", 1, y_shape);
        const auto initializer = proto::raw_tensor("weights", 1, weight_shape,
                                                   std::span<const float>(weights));
        const std::array<std::string_view, 2> node_inputs{"x", "weights"};
        const std::array<std::string_view, 1> node_outputs{"y"};
        const auto matmul = proto::node("matrix_product", "MatMul", node_inputs, node_outputs);
        const std::array nodes{matmul};
        const std::array initializers{initializer};
        const std::array inputs{input};
        const std::array outputs{output};
        return proto::model(proto::graph(nodes, initializers, inputs, outputs));
    }

    [[nodiscard]] Bytes cooperative_matmul_model() {
        constexpr std::int64_t rows = 16;
        constexpr std::int64_t inner = 256;
        constexpr std::int64_t columns = 16;
        const std::array<std::int64_t, 2> x_shape{rows, inner};
        const std::array<std::int64_t, 2> weight_shape{inner, columns};
        const std::array<std::int64_t, 2> y_shape{rows, columns};
        const std::vector<float> weights(static_cast<std::size_t>(inner * columns), 1.0f);
        const auto input = proto::value_info("x", 1, x_shape);
        const auto output = proto::value_info("y", 1, y_shape);
        const auto initializer = proto::raw_tensor("weights", 1, weight_shape,
                                                   std::span<const float>(weights));
        const std::array<std::string_view, 2> node_inputs{"x", "weights"};
        const std::array<std::string_view, 1> node_outputs{"y"};
        const auto matmul = proto::node("cooperative_matrix_product", "MatMul", node_inputs, node_outputs);
        const std::array nodes{matmul};
        const std::array initializers{initializer};
        const std::array inputs{input};
        const std::array outputs{output};
        return proto::model(proto::graph(nodes, initializers, inputs, outputs));
    }

    [[nodiscard]] Bytes grouped_conv_model() {
        const std::array<std::int64_t, 4> x_shape{1, 2, 3, 3};
        const std::array<std::int64_t, 4> weight_shape{2, 1, 2, 2};
        const std::array<std::int64_t, 1> bias_shape{2};
        const std::array<std::int64_t, 4> y_shape{1, 2, 2, 2};
        const std::array<float, 8> weights{1.0f, 0.0f, 0.0f, -1.0f,
                                          0.0f, 1.0f, 1.0f, 0.0f};
        const std::array<float, 2> bias{0.5f, -1.0f};
        const auto input = proto::value_info("x", 1, x_shape);
        const auto output = proto::value_info("y", 1, y_shape);
        const std::array initializers{
            proto::raw_tensor("weights", 1, weight_shape, std::span<const float>(weights)),
            proto::raw_tensor("bias", 1, bias_shape, std::span<const float>(bias)),
        };
        const auto group = proto::int_attribute("group", 2);
        const std::array attributes{group};
        const std::array<std::string_view, 3> node_inputs{"x", "weights", "bias"};
        const std::array<std::string_view, 1> node_outputs{"y"};
        const auto conv = proto::node("grouped_convolution", "Conv", node_inputs, node_outputs, attributes);
        const std::array nodes{conv};
        const std::array inputs{input};
        const std::array outputs{output};
        return proto::model(proto::graph(nodes, initializers, inputs, outputs));
    }

    [[nodiscard]] Bytes softmax_model() {
        const std::array<std::int64_t, 2> shape{2, 3};
        const auto input = proto::value_info("x", 1, shape);
        const auto output = proto::value_info("y", 1, shape);
        const auto axis = proto::int_attribute("axis", -1);
        const std::array attributes{axis};
        const std::array<std::string_view, 1> node_inputs{"x"};
        const std::array<std::string_view, 1> node_outputs{"y"};
        const auto softmax = proto::node("stable_softmax", "Softmax", node_inputs, node_outputs, attributes);
        const std::array nodes{softmax};
        const std::array inputs{input};
        const std::array outputs{output};
        return proto::model(proto::graph(nodes, {}, inputs, outputs));
    }

    [[nodiscard]] Bytes reduction_model() {
        const std::array<std::int64_t, 2> x_shape{2, 3};
        const std::array<std::int64_t, 1> y_shape{3};
        const std::array<std::int64_t, 1> axis_shape{1};
        const std::array<std::int64_t, 1> axes{0};
        const auto input = proto::value_info("x", 1, x_shape);
        const auto output = proto::value_info("y", 1, y_shape);
        const auto initializer = proto::raw_tensor("axes", 7, axis_shape, std::span<const std::int64_t>(axes));
        const auto keepdims = proto::int_attribute("keepdims", 0);
        const std::array attributes{keepdims};
        const std::array<std::string_view, 2> node_inputs{"x", "axes"};
        const std::array<std::string_view, 1> node_outputs{"y"};
        const auto reduce = proto::node("reduce", "ReduceSum", node_inputs, node_outputs, attributes);
        const std::array nodes{reduce};
        const std::array initializers{initializer};
        const std::array inputs{input};
        const std::array outputs{output};
        return proto::model(proto::graph(nodes, initializers, inputs, outputs));
    }

    [[nodiscard]] Bytes view_model() {
        const std::array<std::int64_t, 2> x_shape{2, 3};
        const std::array<std::int64_t, 1> y_shape{6};
        const std::array<std::int64_t, 1> shape_shape{1};
        const std::array<std::int64_t, 1> target_shape{6};
        const std::array<std::int64_t, 2> permutation{1, 0};
        const auto input = proto::value_info("x", 1, x_shape);
        const auto output = proto::value_info("y", 1, y_shape);
        const auto initializer = proto::raw_tensor("target", 7, shape_shape,
                                                   std::span<const std::int64_t>(target_shape));
        const auto perm = proto::ints_attribute("perm", permutation);
        const std::array transpose_attributes{perm};
        const std::array<std::string_view, 1> transpose_inputs{"x"};
        const std::array<std::string_view, 1> transpose_outputs{"transposed"};
        const auto transpose = proto::node("transpose", "Transpose", transpose_inputs,
                                           transpose_outputs, transpose_attributes);
        const std::array<std::string_view, 2> reshape_inputs{"transposed", "target"};
        const std::array<std::string_view, 1> reshape_outputs{"y"};
        const auto reshape = proto::node("reshape", "Reshape", reshape_inputs, reshape_outputs);
        const std::array nodes{transpose, reshape};
        const std::array initializers{initializer};
        const std::array inputs{input};
        const std::array outputs{output};
        return proto::model(proto::graph(nodes, initializers, inputs, outputs));
    }

    [[nodiscard]] Bytes if_model() {
        const std::array<std::int64_t, 1> value_shape{2};
        const std::array<std::int64_t, 0> scalar_shape{};
        const auto condition = proto::value_info("condition", 9, scalar_shape);
        const auto input = proto::value_info("x", 1, value_shape);
        const auto output = proto::value_info("y", 1, value_shape);

        const std::array<std::string_view, 1> captured_input{"x"};
        const std::array<std::string_view, 1> then_output_name{"then_value"};
        const auto then_node = proto::node("then_identity", "Identity", captured_input, then_output_name);
        const auto then_output = proto::value_info("then_value", 1, value_shape);
        const std::array then_nodes{then_node};
        const std::array then_outputs{then_output};
        const auto then_graph = proto::graph(then_nodes, {}, {}, then_outputs);

        const std::array<std::string_view, 1> else_output_name{"else_value"};
        const auto else_node = proto::node("else_neg", "Neg", captured_input, else_output_name);
        const auto else_output = proto::value_info("else_value", 1, value_shape);
        const std::array else_nodes{else_node};
        const std::array else_outputs{else_output};
        const auto else_graph = proto::graph(else_nodes, {}, {}, else_outputs);

        const auto then_attribute = proto::graph_attribute("then_branch", then_graph);
        const auto else_attribute = proto::graph_attribute("else_branch", else_graph);
        const std::array attributes{then_attribute, else_attribute};
        const std::array<std::string_view, 1> if_inputs{"condition"};
        const std::array<std::string_view, 1> if_outputs{"y"};
        const auto if_node = proto::node("if", "If", if_inputs, if_outputs, attributes);
        const std::array nodes{if_node};
        const std::array inputs{condition, input};
        const std::array outputs{output};
        return proto::model(proto::graph(nodes, {}, inputs, outputs));
    }

    [[nodiscard]] bool close(const float lhs, const float rhs) {
        return std::abs(lhs - rhs) <= 1e-5f;
    }

    void run_model(const fs::path& path,
                   const std::span<const std::int64_t> shape,
                   const std::span<const float> input_values,
                   const std::span<const float> expected,
                   lfs::onnx_vulkan::SessionOptions options = {}) {
        auto session = lfs::onnx_vulkan::VulkanSession::create(path, std::move(options));
        if (!session) {
            if (session.error().code == lfs::onnx_vulkan::ErrorCode::VulkanUnavailable ||
                session.error().message.starts_with("vkCreateInstance") ||
                session.error().message.starts_with("vkEnumeratePhysicalDevices"))
                throw std::runtime_error("SKIP_VULKAN:" + session.error().message);
            throw Failure(session.error().message);
        }
        const std::array<lfs::onnx_vulkan::NamedTensorView, 1> inputs{{
            {"x", {lfs::onnx_vulkan::ElementType::Float32, shape,
                    std::as_bytes(input_values)}},
        }};
        auto first = session->run(inputs);
        require(first.has_value(), first ? "" : first.error().message);
        require(first->size() == 1, "unexpected output count");
        const auto actual = first->front().tensor.data_as<float>();
        require(actual.size() == expected.size(), "unexpected output element count");
        for (std::size_t index = 0; index < actual.size(); ++index)
            require(close(actual[index], expected[index]), "Vulkan operator result differs from scalar reference");
        auto second = session->run(inputs);
        require(second.has_value(), second ? "" : second.error().message);
        require(std::ranges::equal(first->front().tensor.bytes(), second->front().tensor.bytes()),
                "repeat run was not deterministic");
    }

    void test_vulkan_basics(const TemporaryDirectory& temporary) {
        const auto arithmetic_path = temporary.path() / "arithmetic.onnx";
        write_file(arithmetic_path, arithmetic_model());
        const std::array<std::int64_t, 2> arithmetic_shape{2, 1};
        const std::array<float, 2> arithmetic_input{1.0f, 2.0f};
        const std::array<float, 6> arithmetic_expected{11.0f, 21.0f, 31.0f, 12.0f, 22.0f, 32.0f};
        run_model(arithmetic_path, arithmetic_shape, arithmetic_input, arithmetic_expected);

        const auto reduction_path = temporary.path() / "reduction.onnx";
        write_file(reduction_path, reduction_model());
        const std::array<std::int64_t, 2> matrix_shape{2, 3};
        const std::array<float, 6> matrix{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        const std::array<float, 3> reduction_expected{5.0f, 7.0f, 9.0f};
        run_model(reduction_path, matrix_shape, matrix, reduction_expected);

        const auto view_path = temporary.path() / "view.onnx";
        write_file(view_path, view_model());
        const std::array<float, 6> view_expected{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
        run_model(view_path, matrix_shape, matrix, view_expected);

        const auto if_path = temporary.path() / "if.onnx";
        write_file(if_path, if_model());
        auto session = lfs::onnx_vulkan::VulkanSession::create(if_path);
        require(session.has_value(), session ? "" : session.error().message);
        const std::array<std::int64_t, 1> if_shape{2};
        const std::array<float, 2> if_values{2.0f, -3.0f};
        for (const bool selected_then : {false, true}) {
            const std::array<lfs::onnx_vulkan::NamedTensorView, 2> inputs{{
                {"condition", {lfs::onnx_vulkan::ElementType::Bool, {},
                                std::as_bytes(std::span(&selected_then, 1))}},
                {"x", {lfs::onnx_vulkan::ElementType::Float32, if_shape,
                        std::as_bytes(std::span(if_values))}},
            }};
            auto result = session->run(inputs);
            require(result.has_value(), result ? "" : result.error().message);
            const auto actual = result->front().tensor.data_as<float>();
            require(actual.size() == if_values.size(), "If result has the wrong size");
            for (std::size_t index = 0; index < actual.size(); ++index) {
                const float expected = selected_then ? if_values[index] : -if_values[index];
                require(close(actual[index], expected), "If selected the wrong branch");
            }
        }
    }

    void test_vulkan_elementwise(const TemporaryDirectory& temporary) {
        const auto path = temporary.path() / "pow_clip.onnx";
        write_file(path, pow_clip_model());
        const std::array<std::int64_t, 2> shape{2, 3};
        const std::array<float, 6> input{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
        const std::array<float, 6> expected{-4.0f, -1.0f, 0.0f, 1.0f, 8.0f, 10.0f};
        run_model(path, shape, input, expected);
    }

    void test_vulkan_matmul(const TemporaryDirectory& temporary) {
        const auto path = temporary.path() / "matmul.onnx";
        write_file(path, matmul_model());
        const std::array<std::int64_t, 2> shape{2, 3};
        const std::array<float, 6> input{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        const std::array<float, 4> expected{22.0f, 28.0f, 49.0f, 64.0f};
        run_model(path, shape, input, expected);

        const auto cooperative_path = temporary.path() / "cooperative_matmul.onnx";
        write_file(cooperative_path, cooperative_matmul_model());
        const std::array<std::int64_t, 2> cooperative_shape{16, 256};
        const std::vector<float> cooperative_input(16 * 256, 1.0f);
        const std::vector<float> cooperative_expected(16 * 16, 256.0f);
        run_model(cooperative_path, cooperative_shape, cooperative_input, cooperative_expected);
        lfs::onnx_vulkan::SessionOptions fallback_options;
        fallback_options.enable_cooperative_matrix = false;
        run_model(cooperative_path, cooperative_shape, cooperative_input, cooperative_expected,
                  std::move(fallback_options));
    }

    void test_vulkan_grouped_conv(const TemporaryDirectory& temporary) {
        const auto path = temporary.path() / "grouped_conv.onnx";
        write_file(path, grouped_conv_model());
        const std::array<std::int64_t, 4> shape{1, 2, 3, 3};
        const std::array<float, 18> input{
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
        };
        const std::array<float, 8> expected{-3.5f, -3.5f, -3.5f, -3.5f,
                                             13.0f, 11.0f, 7.0f, 5.0f};
        run_model(path, shape, input, expected);
    }

    void test_vulkan_softmax(const TemporaryDirectory& temporary) {
        const auto path = temporary.path() / "softmax.onnx";
        write_file(path, softmax_model());
        const std::array<std::int64_t, 2> shape{2, 3};
        const std::array<float, 6> input{1000.0f, 1001.0f, 1002.0f,
                                         -1000.0f, -1000.0f, -1000.0f};
        const float denominator = std::exp(-2.0f) + std::exp(-1.0f) + 1.0f;
        const std::array<float, 6> expected{
            std::exp(-2.0f) / denominator, std::exp(-1.0f) / denominator, 1.0f / denominator,
            1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f,
        };
        run_model(path, shape, input, expected);
    }
}

int main() {
    TemporaryDirectory temporary;
    int failures = 0;
    const auto run = [&](const std::string_view name, const std::function<void()>& test) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
        } catch (const std::runtime_error& error) {
            if (std::string_view(error.what()).starts_with("SKIP_VULKAN:")) {
                std::cout << "SKIP " << name << ' ' << std::string_view(error.what()).substr(12) << '\n';
                return;
            }
            ++failures;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    };
    run("parser", [&] { test_parser(temporary); });
    run("graph_validation", [&] { test_graph_validation(temporary); });
    run("operator_registry", test_registry);
    run("vulkan_basics", [&] { test_vulkan_basics(temporary); });
    run("vulkan_elementwise", [&] { test_vulkan_elementwise(temporary); });
    run("vulkan_matmul", [&] { test_vulkan_matmul(temporary); });
    run("vulkan_grouped_conv", [&] { test_vulkan_grouped_conv(temporary); });
    run("vulkan_softmax", [&] { test_vulkan_softmax(temporary); });
    if (failures != 0)
        std::cerr << failures << " ONNX Vulkan test group(s) failed\n";
    return failures == 0 ? 0 : 1;
}
