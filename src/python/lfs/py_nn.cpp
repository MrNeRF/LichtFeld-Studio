/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_nn.hpp"

#include "core/nn/models/sam2.hpp"
#include "core/tensor.hpp"
#include "preprocessing/preprocess.hpp"
#include "py_error.hpp"
#include "py_tensor.hpp"

#include <cuda_runtime.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::python {

    namespace {

        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;
        using lfs::core::TensorShape;
        using lfs::core::nn::models::Sam2;
        using lfs::core::nn::models::Sam2PointPrompt;

        bool ndarray_is_c_contiguous(const nb::ndarray<>& arr) {
            const size_t ndim = arr.ndim();
            if (ndim == 0 || !arr.stride_ptr()) {
                return true;
            }
            int64_t expected = 1;
            for (size_t i = ndim; i-- > 0;) {
                const int64_t extent = static_cast<int64_t>(arr.shape(i));
                if (extent == 0) {
                    return true;
                }
                if (extent != 1 && arr.stride(i) != expected) {
                    return false;
                }
                expected *= extent;
            }
            return true;
        }

        std::string ndarray_shape_text(const nb::ndarray<>& arr) {
            std::string text = "(";
            for (size_t i = 0; i < arr.ndim(); ++i) {
                if (i != 0) {
                    text += ", ";
                }
                text += std::to_string(arr.shape(i));
            }
            text += ")";
            return text;
        }

        void require_cuda() {
            int devices = 0;
            const cudaError_t status = cudaGetDeviceCount(&devices);
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::format("SAM2 requires CUDA ({})", cudaGetErrorString(status)));
            }
            if (devices <= 0) {
                throw std::runtime_error("SAM2 requires CUDA; no CUDA device is available");
            }
        }

        Tensor image_from_numpy(const nb::ndarray<>& arr) {
            if (arr.ndim() != 3 || arr.shape(2) != 3) {
                throw std::invalid_argument(
                    std::format("set_image expects a numpy RGB array of shape (H, W, 3); got ndim={} shape={}",
                                arr.ndim(), ndarray_shape_text(arr)));
            }
            const auto h = static_cast<size_t>(arr.shape(0));
            const auto w = static_cast<size_t>(arr.shape(1));
            if (h == 0 || w == 0) {
                throw std::invalid_argument("set_image image height and width must be > 0");
            }

            const bool is_u8 = arr.dtype() == nb::dtype<uint8_t>();
            const bool is_f32 = arr.dtype() == nb::dtype<float>();
            if (!is_u8 && !is_f32) {
                throw std::invalid_argument("set_image expects dtype uint8 or float32 (RGB in [0, 1])");
            }
            if (!ndarray_is_c_contiguous(arr)) {
                throw std::invalid_argument(
                    "set_image array must be C-contiguous; call numpy.ascontiguousarray() first");
            }

            const DataType dtype = is_u8 ? DataType::UInt8 : DataType::Float32;
            Tensor out = Tensor::empty(TensorShape({h, w, 3}), Device::CPU, dtype, true);
            const size_t bytes = h * w * 3 * (is_u8 ? size_t{1} : sizeof(float));
            std::memcpy(out.data_ptr(), arr.data(), bytes);
            return out;
        }

        float as_number(nb::handle value, std::string_view what) {
            try {
                if (nb::isinstance<nb::int_>(value) || nb::isinstance<nb::float_>(value)) {
                    return nb::cast<float>(value);
                }
                if (nb::hasattr(value, "item")) {
                    return nb::cast<float>(value.attr("item")());
                }
                return nb::cast<float>(value);
            } catch (const nb::cast_error&) {
                throw std::invalid_argument(std::format("{} must be a number", what));
            }
        }

        int as_int_label(nb::handle value, std::string_view what) {
            try {
                if (nb::isinstance<nb::int_>(value)) {
                    return nb::cast<int>(value);
                }
                if (nb::hasattr(value, "item")) {
                    return as_int_label(value.attr("item")(), what);
                }
                return static_cast<int>(nb::cast<long long>(value));
            } catch (const nb::cast_error&) {
                throw std::invalid_argument(std::format("{} must be an integer label", what));
            }
        }

        nb::sequence as_sequence(nb::handle value, const char* what) {
            if (value.is_none() || !nb::isinstance<nb::sequence>(value) || nb::isinstance<nb::str>(value)) {
                throw std::invalid_argument(what);
            }
            return nb::borrow<nb::sequence>(value);
        }

        std::vector<std::array<float, 2>> parse_points(nb::handle points) {
            if (points.is_none()) {
                return {};
            }
            const nb::sequence seq = as_sequence(
                points, "points must be a sequence of [x, y] pairs");
            std::vector<std::array<float, 2>> out;
            const auto n = nb::len(seq);
            out.reserve(static_cast<size_t>(n));
            for (Py_ssize_t i = 0; i < n; ++i) {
                nb::object item = seq[i];
                if (!nb::isinstance<nb::sequence>(item) || nb::isinstance<nb::str>(item) ||
                    nb::len(item) != 2) {
                    throw std::invalid_argument("each point must be a sequence of 2 numbers [x, y]");
                }
                const nb::sequence xy = nb::borrow<nb::sequence>(item);
                out.push_back({as_number(xy[0], "point x"), as_number(xy[1], "point y")});
            }
            return out;
        }

        std::vector<int> parse_labels(nb::handle labels) {
            if (labels.is_none()) {
                return {};
            }
            const nb::sequence seq = as_sequence(labels, "labels must be a sequence of integers");
            std::vector<int> out;
            const auto n = nb::len(seq);
            out.reserve(static_cast<size_t>(n));
            for (Py_ssize_t i = 0; i < n; ++i) {
                out.push_back(as_int_label(seq[i], "label"));
            }
            return out;
        }

        std::optional<std::array<float, 4>> parse_box(nb::handle box) {
            if (box.is_none()) {
                return std::nullopt;
            }
            const nb::sequence seq = as_sequence(
                box, "box must be a sequence of 4 numbers [x0, y0, x1, y1]");
            if (nb::len(seq) != 4) {
                throw std::invalid_argument("box must be a sequence of 4 numbers [x0, y0, x1, y1]");
            }
            return std::array<float, 4>{
                as_number(seq[0], "box x0"),
                as_number(seq[1], "box y0"),
                as_number(seq[2], "box x1"),
                as_number(seq[3], "box y1"),
            };
        }

        class PySam2 {
        public:
            PySam2(const PySam2&) = delete;
            PySam2& operator=(const PySam2&) = delete;
            PySam2(PySam2&&) noexcept = default;
            PySam2& operator=(PySam2&&) noexcept = default;

            explicit PySam2(std::optional<std::filesystem::path> weights = std::nullopt)
                : weights_(std::move(weights)) {}

            void set_image(nb::ndarray<> image) {
                Tensor hwc = image_from_numpy(image);
                {
                    nb::gil_scoped_release release;
                    ensure_loaded();
                    Tensor gpu = hwc.cuda();
                    if (gpu.dtype() == DataType::UInt8) {
                        gpu = gpu.to(DataType::Float32).mul(1.0f / 255.0f);
                    }
                    Tensor nchw = gpu.permute({2, 0, 1}).unsqueeze(0).contiguous();
                    unwrap(model_->set_image(nchw));
                }
            }

            nb::tuple predict(nb::object points, nb::object labels, nb::object box, bool multimask) {
                auto xy = parse_points(points);
                auto labs = parse_labels(labels);
                if (xy.size() != labs.size()) {
                    throw std::invalid_argument(std::format(
                        "points and labels must have the same length (got {} points and {} labels)",
                        xy.size(), labs.size()));
                }
                auto maybe_box = parse_box(box);
                if (xy.empty() && !maybe_box) {
                    throw std::invalid_argument("predict requires at least one point or a box");
                }

                std::vector<Sam2PointPrompt> prompts;
                prompts.reserve(xy.size());
                for (size_t i = 0; i < xy.size(); ++i) {
                    prompts.push_back({xy[i][0], xy[i][1], labs[i]});
                }

                Tensor masks;
                Tensor scores;
                {
                    nb::gil_scoped_release release;
                    ensure_loaded();
                    auto pred = unwrap(model_->predict(prompts, maybe_box, multimask));
                    masks = pred.masks.to(DataType::Float32).cpu().contiguous();
                    scores = pred.iou.to(DataType::Float32).cpu().contiguous();
                }
                return nb::make_tuple(PyTensor(std::move(masks)).numpy(),
                                      PyTensor(std::move(scores)).numpy());
            }

        private:
            void ensure_loaded() {
                if (model_) {
                    return;
                }
                require_cuda();
                std::filesystem::path path;
                if (weights_) {
                    path = *weights_;
                } else {
                    path = lfs::preprocessing::ensure_sam2_weights();
                }
                auto loaded = unwrap(Sam2::load(path, Device::CUDA));
                model_ = std::make_unique<Sam2>(std::move(loaded));
            }

            std::optional<std::filesystem::path> weights_;
            std::unique_ptr<Sam2> model_;
        };

    } // namespace

    void register_nn(nb::module_& m) {
        nb::class_<PySam2>(m, "Sam2", "SAM 2.1 image predictor")
            .def(nb::init<std::optional<std::filesystem::path>>(),
                 nb::arg("weights") = nb::none(),
                 "Create a SAM 2.1 predictor. weights=None resolves the default cached "
                 ".lfw via ensure_sam2_weights (downloads on first use).")
            .def("set_image", &PySam2::set_image, nb::arg("image"),
                 "Set the image for prompting. numpy HWC uint8 or float32 RGB in [0, 1], any size.")
            .def("predict", &PySam2::predict,
                 nb::arg("points") = nb::none(),
                 nb::arg("labels") = nb::none(),
                 nb::arg("box") = nb::none(),
                 nb::arg("multimask") = true,
                 "Predict masks from point and/or box prompts. Returns (masks [N,H,W] float32 logits, "
                 "scores [N] float32).");
    }

} // namespace lfs::python
