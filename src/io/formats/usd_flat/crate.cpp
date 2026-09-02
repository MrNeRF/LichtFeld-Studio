/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "crate.hpp"
#include "core/path_utils.hpp"
#include "half.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <numbers>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <crate-reader.hh>
#include <integerCoding.h>
#include <lz4-compression.hh>
#include <prim-types.hh>
#include <stream-reader.hh>

namespace lfs::io::usd_flat {

    namespace {

        constexpr const char* composition_error =
            "USD composition is unsupported: flat Gaussian interchange does not support references, payloads, subLayers, variants, inherits, specializes, or clips";

        bool is_composition_field(const std::string& name) {
            static constexpr const char* names[] = {
                "references", "payload", "subLayers", "variantSetNames", "variantSelection", "inherits", "specializes", "clips"};
            return std::find_if(std::begin(names), std::end(names), [&name](const char* candidate) {
                       return name == candidate;
                   }) != std::end(names);
        }

        class MappedFile {
        public:
            MappedFile() = default;
            MappedFile(const MappedFile&) = delete;
            MappedFile& operator=(const MappedFile&) = delete;
            MappedFile(MappedFile&& other) noexcept { *this = std::move(other); }
            MappedFile& operator=(MappedFile&& other) noexcept {
                if (this != &other) {
                    close();
                    data_ = other.data_;
                    size_ = other.size_;
                    fallback_ = std::move(other.fallback_);
#ifdef _WIN32
                    file_ = other.file_;
                    mapping_ = other.mapping_;
                    other.file_ = INVALID_HANDLE_VALUE;
                    other.mapping_ = nullptr;
#else
                    fd_ = other.fd_;
                    other.fd_ = -1;
#endif
                    other.data_ = nullptr;
                    other.size_ = 0;
                }
                return *this;
            }
            ~MappedFile() { close(); }

            [[nodiscard]] const std::uint8_t* data() const { return data_; }
            [[nodiscard]] std::size_t size() const { return size_; }

            static lfs::Result<MappedFile> open(const std::filesystem::path& path) {
                MappedFile result;
#ifdef _WIN32
                result.file_ = CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                           FILE_ATTRIBUTE_NORMAL, nullptr);
                if (result.file_ == INVALID_HANDLE_VALUE) {
                    return make_flat_error("Failed to open USD file: " + lfs::core::path_to_utf8(path));
                }
                LARGE_INTEGER file_size{};
                if (!GetFileSizeEx(result.file_, &file_size) || file_size.QuadPart < 0) {
                    return make_flat_error("Failed to determine USD file size");
                }
                result.size_ = static_cast<std::size_t>(file_size.QuadPart);
                if (result.size_ != 0) {
                    result.mapping_ = CreateFileMappingW(result.file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
                    if (result.mapping_) {
                        result.data_ = static_cast<const std::uint8_t*>(MapViewOfFile(result.mapping_, FILE_MAP_READ, 0, 0, 0));
                    }
                }
#else
                result.fd_ = ::open(path.c_str(), O_RDONLY);
                if (result.fd_ >= 0) {
                    struct stat file_status {};
                    if (::fstat(result.fd_, &file_status) == 0 && file_status.st_size >= 0) {
                        result.size_ = static_cast<std::size_t>(file_status.st_size);
                        if (result.size_ != 0) {
                            void* mapped = ::mmap(nullptr, result.size_, PROT_READ, MAP_PRIVATE, result.fd_, 0);
                            if (mapped != MAP_FAILED) {
                                result.data_ = static_cast<const std::uint8_t*>(mapped);
                            }
                        }
                    }
                }
#endif
                if (result.data_ || result.size_ == 0) {
                    return result;
                }

                std::ifstream input(path, std::ios::binary);
                if (!input) {
                    return make_flat_error("Failed to open USD file: " + lfs::core::path_to_utf8(path));
                }
                result.fallback_.resize(result.size_);
                input.read(reinterpret_cast<char*>(result.fallback_.data()), static_cast<std::streamsize>(result.size_));
                if (!input) {
                    return make_flat_error("Failed to read USD file: " + lfs::core::path_to_utf8(path));
                }
                result.data_ = result.fallback_.data();
                return result;
            }

        private:
            void close() noexcept {
#ifdef _WIN32
                if (data_ && mapping_) {
                    UnmapViewOfFile(data_);
                }
                if (mapping_) {
                    CloseHandle(mapping_);
                }
                if (file_ != INVALID_HANDLE_VALUE) {
                    CloseHandle(file_);
                }
                file_ = INVALID_HANDLE_VALUE;
                mapping_ = nullptr;
#else
                if (data_ && fallback_.empty() && size_ != 0) {
                    ::munmap(const_cast<std::uint8_t*>(data_), size_);
                }
                if (fd_ >= 0) {
                    ::close(fd_);
                }
                fd_ = -1;
#endif
                data_ = nullptr;
                size_ = 0;
                fallback_.clear();
            }

            const std::uint8_t* data_ = nullptr;
            std::size_t size_ = 0;
            std::vector<std::uint8_t> fallback_;
#ifdef _WIN32
            HANDLE file_ = INVALID_HANDLE_VALUE;
            HANDLE mapping_ = nullptr;
#else
            int fd_ = -1;
#endif
        };

        template <typename T>
        void append_vec3(const std::vector<T>& values, FlatAttribute& output) {
            output.components = 3;
            output.values.resize(values.size() * 3);
            for (std::size_t index = 0; index < values.size(); ++index) {
                output.values[index * 3 + 0] = static_cast<float>(values[index][0]);
                output.values[index * 3 + 1] = static_cast<float>(values[index][1]);
                output.values[index * 3 + 2] = static_cast<float>(values[index][2]);
            }
            output.authored = true;
        }

        void append_half_vec3(const std::vector<tinyusdz::value::half3>& values, FlatAttribute& output) {
            output.components = 3;
            output.values.resize(values.size() * 3);
            for (std::size_t index = 0; index < values.size(); ++index) {
                output.values[index * 3 + 0] = half_to_float(values[index][0].value);
                output.values[index * 3 + 1] = half_to_float(values[index][1].value);
                output.values[index * 3 + 2] = half_to_float(values[index][2].value);
            }
            output.authored = true;
        }

        void append_half_point3(const std::vector<tinyusdz::value::point3h>& values, FlatAttribute& output) {
            output.components = 3;
            output.values.resize(values.size() * 3);
            for (std::size_t index = 0; index < values.size(); ++index) {
                output.values[index * 3 + 0] = half_to_float(values[index][0].value);
                output.values[index * 3 + 1] = half_to_float(values[index][1].value);
                output.values[index * 3 + 2] = half_to_float(values[index][2].value);
            }
            output.authored = true;
        }

        void append_quat(const std::vector<tinyusdz::value::quatf>& values, FlatAttribute& output) {
            output.components = 4;
            output.values.resize(values.size() * 4);
            for (std::size_t index = 0; index < values.size(); ++index) {
                output.values[index * 4 + 0] = values[index].real;
                output.values[index * 4 + 1] = values[index].imag[0];
                output.values[index * 4 + 2] = values[index].imag[1];
                output.values[index * 4 + 3] = values[index].imag[2];
            }
            output.authored = true;
        }

        void append_quat_half(const std::vector<tinyusdz::value::quath>& values, FlatAttribute& output) {
            output.components = 4;
            output.values.resize(values.size() * 4);
            for (std::size_t index = 0; index < values.size(); ++index) {
                output.values[index * 4 + 0] = half_to_float(values[index].real.value);
                output.values[index * 4 + 1] = half_to_float(values[index].imag[0].value);
                output.values[index * 4 + 2] = half_to_float(values[index].imag[1].value);
                output.values[index * 4 + 3] = half_to_float(values[index].imag[2].value);
            }
            output.authored = true;
        }

        template <typename T>
        void append_scalar(const std::vector<T>& values, FlatAttribute& output) {
            output.components = 1;
            output.values.resize(values.size());
            for (std::size_t index = 0; index < values.size(); ++index) {
                output.values[index] = static_cast<float>(values[index]);
            }
            output.authored = true;
        }

        void append_scalar_half(const std::vector<tinyusdz::value::half>& values, FlatAttribute& output) {
            output.components = 1;
            output.values.resize(values.size());
            for (std::size_t index = 0; index < values.size(); ++index) {
                output.values[index] = half_to_float(values[index].value);
            }
            output.authored = true;
        }

        void read_value(const std::string& name, const std::string& type_name, const tinyusdz::crate::CrateValue& value, FlatPrim& prim) {
            FlatAttribute output;
            output.type_name = type_name;
            if (name == "positions" || name == "positionsh" || name == "scales" || name == "scalesh" ||
                name == "radiance:sphericalHarmonicsCoefficients" || name == "radiance:sphericalHarmonicsCoefficientsh" || name == "extent") {
                if (const auto* values = value.as<std::vector<tinyusdz::value::float3>>()) {
                    append_vec3(*values, output);
                } else if (const auto* values = value.as<std::vector<tinyusdz::value::point3f>>()) {
                    append_vec3(*values, output);
                } else if (const auto* values = value.as<std::vector<tinyusdz::value::double3>>()) {
                    append_vec3(*values, output);
                } else if (const auto* values = value.as<std::vector<tinyusdz::value::half3>>()) {
                    append_half_vec3(*values, output);
                } else if (const auto* values = value.as<std::vector<tinyusdz::value::point3h>>()) {
                    append_half_point3(*values, output);
                }
            } else if (name == "orientations" || name == "orientationsh") {
                if (const auto* values = value.as<std::vector<tinyusdz::value::quatf>>()) {
                    append_quat(*values, output);
                } else if (const auto* values = value.as<std::vector<tinyusdz::value::quath>>()) {
                    append_quat_half(*values, output);
                } else if (const auto* values = value.as<std::vector<tinyusdz::value::quatd>>()) {
                    output.components = 4;
                    output.values.resize(values->size() * 4);
                    for (std::size_t index = 0; index < values->size(); ++index) {
                        output.values[index * 4 + 0] = static_cast<float>((*values)[index].real);
                        output.values[index * 4 + 1] = static_cast<float>((*values)[index].imag[0]);
                        output.values[index * 4 + 2] = static_cast<float>((*values)[index].imag[1]);
                        output.values[index * 4 + 3] = static_cast<float>((*values)[index].imag[2]);
                    }
                    output.authored = true;
                }
            } else if (name == "opacities" || name == "opacitiesh") {
                if (const auto* values = value.as<std::vector<float>>()) {
                    append_scalar(*values, output);
                } else if (const auto* values = value.as<std::vector<tinyusdz::value::half>>()) {
                    append_scalar_half(*values, output);
                } else if (const auto* values = value.as<std::vector<double>>()) {
                    append_scalar(*values, output);
                }
            } else if (name == "radiance:sphericalHarmonicsDegree") {
                if (const auto* degree = value.as<int>()) {
                    output.values = {static_cast<float>(*degree)};
                    output.authored = true;
                }
            }
            if (output.authored) {
                prim.attributes[name] = std::move(output);
            }
        }

        std::string token_value(const tinyusdz::crate::CrateValue* value) {
            if (!value) {
                return {};
            }
            if (const auto token = value->get_value<tinyusdz::value::token>()) {
                return token->str();
            }
            return {};
        }

        void multiply(const std::array<double, 16>& left, const std::array<double, 16>& right, std::array<double, 16>& output) {
            std::array<double, 16> result{};
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    for (int index = 0; index < 4; ++index) {
                        result[static_cast<std::size_t>(row * 4 + column)] +=
                            left[static_cast<std::size_t>(row * 4 + index)] * right[static_cast<std::size_t>(index * 4 + column)];
                    }
                }
            }
            output = result;
        }

        bool invert_matrix(const std::array<double, 16>& input, std::array<double, 16>& output) {
            std::array<double, 32> augmented{};
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    augmented[static_cast<std::size_t>(row * 8 + column)] = input[static_cast<std::size_t>(row * 4 + column)];
                }
                augmented[static_cast<std::size_t>(row * 8 + row + 4)] = 1.0;
            }
            for (int column = 0; column < 4; ++column) {
                int pivot = column;
                for (int row = column + 1; row < 4; ++row) {
                    if (std::abs(augmented[static_cast<std::size_t>(row * 8 + column)]) >
                        std::abs(augmented[static_cast<std::size_t>(pivot * 8 + column)])) {
                        pivot = row;
                    }
                }
                const double pivot_value = augmented[static_cast<std::size_t>(pivot * 8 + column)];
                if (std::abs(pivot_value) < 1e-15) {
                    return false;
                }
                if (pivot != column) {
                    for (int index = 0; index < 8; ++index) {
                        std::swap(augmented[static_cast<std::size_t>(pivot * 8 + index)],
                                  augmented[static_cast<std::size_t>(column * 8 + index)]);
                    }
                }
                for (int index = 0; index < 8; ++index) {
                    augmented[static_cast<std::size_t>(column * 8 + index)] /= pivot_value;
                }
                for (int row = 0; row < 4; ++row) {
                    if (row == column) {
                        continue;
                    }
                    const double factor = augmented[static_cast<std::size_t>(row * 8 + column)];
                    for (int index = 0; index < 8; ++index) {
                        augmented[static_cast<std::size_t>(row * 8 + index)] -=
                            factor * augmented[static_cast<std::size_t>(column * 8 + index)];
                    }
                }
            }
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    output[static_cast<std::size_t>(row * 4 + column)] = augmented[static_cast<std::size_t>(row * 8 + column + 4)];
                }
            }
            return true;
        }

        lfs::Result<std::array<double, 16>> transform_matrix(const std::string& op, const tinyusdz::crate::CrateValue& value) {
            auto result = identity_matrix();
            const auto suffix_start = op.find(':', 8);
            const std::string base = suffix_start == std::string::npos ? op : op.substr(0, suffix_start);
            if (base == "xformOp:transform") {
                if (const auto matrix = value.get_value<tinyusdz::value::matrix4d>()) {
                    for (int row = 0; row < 4; ++row) {
                        for (int column = 0; column < 4; ++column) {
                            result[static_cast<std::size_t>(row * 4 + column)] = (*matrix).m[row][column];
                        }
                    }
                }
                return result;
            }
            if (base == "xformOp:translate" || base == "xformOp:scale" || base.rfind("xformOp:rotate", 0) == 0) {
                std::array<double, 3> vector{};
                if (const auto value_vector = value.get_value<tinyusdz::value::float3>()) {
                    for (int index = 0; index < 3; ++index)
                        vector[static_cast<std::size_t>(index)] = (*value_vector)[index];
                } else if (const auto value_vector = value.get_value<tinyusdz::value::double3>()) {
                    for (int index = 0; index < 3; ++index) {
                        vector[static_cast<std::size_t>(index)] = (*value_vector)[index];
                    }
                } else if (const auto value_vector = value.get_value<tinyusdz::value::half3>()) {
                    for (int index = 0; index < 3; ++index)
                        vector[static_cast<std::size_t>(index)] = half_to_float((*value_vector)[index].value);
                } else if (const auto scalar = value.get_value<float>()) {
                    vector[0] = *scalar;
                } else if (const auto scalar = value.get_value<double>()) {
                    vector[0] = *scalar;
                } else if (const auto scalar = value.get_value<tinyusdz::value::half>()) {
                    vector[0] = half_to_float(scalar->value);
                }
                if (base == "xformOp:rotateY") {
                    vector[1] = vector[0];
                } else if (base == "xformOp:rotateZ") {
                    vector[2] = vector[0];
                }
                if (base == "xformOp:translate") {
                    result[12] = vector[0];
                    result[13] = vector[1];
                    result[14] = vector[2];
                } else if (base == "xformOp:scale") {
                    result[0] = vector[0];
                    result[5] = vector[1];
                    result[10] = vector[2];
                } else if (base == "xformOp:rotateX" || base == "xformOp:rotateY" || base == "xformOp:rotateZ" ||
                           base == "xformOp:rotateXYZ" || base == "xformOp:rotateXZY" || base == "xformOp:rotateYXZ" ||
                           base == "xformOp:rotateYZX" || base == "xformOp:rotateZXY" || base == "xformOp:rotateZYX") {
                    constexpr double pi = std::numbers::pi;
                    const double rx = vector[0] * pi / 180.0;
                    const double ry = vector[1] * pi / 180.0;
                    const double rz = vector[2] * pi / 180.0;
                    auto axis = [](double angle, int index) {
                        auto matrix = identity_matrix();
                        const double c = std::cos(angle);
                        const double s = std::sin(angle);
                        const int a = (index + 1) % 3;
                        const int b = (index + 2) % 3;
                        matrix[static_cast<std::size_t>(a * 4 + a)] = c;
                        matrix[static_cast<std::size_t>(b * 4 + b)] = c;
                        matrix[static_cast<std::size_t>(a * 4 + b)] = s;
                        matrix[static_cast<std::size_t>(b * 4 + a)] = -s;
                        return matrix;
                    };
                    auto rxm = axis(rx, 0);
                    auto rym = axis(ry, 1);
                    auto rzm = axis(rz, 2);
                    if (base == "xformOp:rotateX")
                        return rxm;
                    if (base == "xformOp:rotateY")
                        return rym;
                    if (base == "xformOp:rotateZ")
                        return rzm;
                    const std::string order = base.substr(std::string("xformOp:rotate").size());
                    result = identity_matrix();
                    for (const char axis_name : order) {
                        if (axis_name != 'X' && axis_name != 'Y' && axis_name != 'Z') {
                            return make_flat_error("Unsupported USD xformOp: " + op);
                        }
                        if (axis_name == 'X')
                            multiply(result, rxm, result);
                        if (axis_name == 'Y')
                            multiply(result, rym, result);
                        if (axis_name == 'Z')
                            multiply(result, rzm, result);
                    }
                } else {
                    return make_flat_error("Unsupported USD xformOp: " + op);
                }
                return result;
            }
            if (base == "xformOp:orient") {
                std::array<double, 4> quaternion{};
                if (const auto quat = value.get_value<tinyusdz::value::quatf>()) {
                    quaternion = {quat->real, quat->imag[0], quat->imag[1], quat->imag[2]};
                } else if (const auto quat = value.get_value<tinyusdz::value::quatd>()) {
                    quaternion = {quat->real, quat->imag[0], quat->imag[1], quat->imag[2]};
                } else if (const auto quat = value.get_value<tinyusdz::value::quath>()) {
                    quaternion = {half_to_float(quat->real.value), half_to_float(quat->imag[0].value),
                                  half_to_float(quat->imag[1].value), half_to_float(quat->imag[2].value)};
                }
                const double w = quaternion[0], x = quaternion[1], y = quaternion[2], z = quaternion[3];
                result[0] = 1 - 2 * (y * y + z * z);
                result[1] = 2 * (x * y + z * w);
                result[2] = 2 * (x * z - y * w);
                result[4] = 2 * (x * y - z * w);
                result[5] = 1 - 2 * (x * x + z * z);
                result[6] = 2 * (y * z + x * w);
                result[8] = 2 * (x * z + y * w);
                result[9] = 2 * (y * z - x * w);
                result[10] = 1 - 2 * (x * x + y * y);
                return result;
            }
            return make_flat_error("Unsupported USD xformOp: " + op);
        }

        float read_float(const std::uint8_t* data) {
            float value = 0.0f;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }

        double read_double(const std::uint8_t* data) {
            double value = 0.0;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }

        bool decode_flat_array(const tinyusdz::crate::CrateReader& reader,
                               const tinyusdz::crate::ValueRep& rep,
                               const std::string& type_name,
                               FlatAttribute& output) {
            tinyusdz::crate::CrateReader::FlatArrayData array;
            if (!reader.GetUncompressedArrayData(rep, &array)) {
                return false;
            }
            const auto type = static_cast<tinyusdz::crate::CrateDataTypeId>(rep.GetType());
            std::size_t components = 1;
            std::size_t element_size = 0;
            bool quaternion = false;
            switch (type) {
            case tinyusdz::crate::CrateDataTypeId::CRATE_DATA_TYPE_HALF:
                element_size = 2;
                break;
            case tinyusdz::crate::CrateDataTypeId::CRATE_DATA_TYPE_FLOAT:
                element_size = 4;
                break;
            case tinyusdz::crate::CrateDataTypeId::CRATE_DATA_TYPE_DOUBLE:
                element_size = 8;
                break;
            case tinyusdz::crate::CrateDataTypeId::CRATE_DATA_TYPE_QUATH:
                element_size = 2;
                components = 4;
                quaternion = true;
                break;
            case tinyusdz::crate::CrateDataTypeId::CRATE_DATA_TYPE_QUATF:
                element_size = 4;
                components = 4;
                quaternion = true;
                break;
            case tinyusdz::crate::CrateDataTypeId::CRATE_DATA_TYPE_QUATD:
                element_size = 8;
                components = 4;
                quaternion = true;
                break;
            case tinyusdz::crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC3H:
                element_size = 2;
                components = 3;
                break;
            case tinyusdz::crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC3F:
                element_size = 4;
                components = 3;
                break;
            case tinyusdz::crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC3D:
                element_size = 8;
                components = 3;
                break;
            default:
                return false;
            }
            output.type_name = type_name;
            output.components = static_cast<int>(components);
            output.values.resize(static_cast<std::size_t>(array.count) * components);
            for (std::size_t index = 0; index < array.count; ++index) {
                const auto* element = array.data + index * components * element_size;
                for (std::size_t component = 0; component < components; ++component) {
                    const auto* component_data = element + component * element_size;
                    float value = 0.0f;
                    if (element_size == 2) {
                        std::uint16_t bits = 0;
                        std::memcpy(&bits, component_data, sizeof(bits));
                        value = half_to_float(bits);
                    } else if (element_size == 4) {
                        value = read_float(component_data);
                    } else {
                        value = static_cast<float>(read_double(component_data));
                    }
                    if (quaternion) {
                        const std::size_t destination = component == 3 ? 0 : component + 1;
                        output.values[index * 4 + destination] = value;
                    } else {
                        output.values[index * components + component] = value;
                    }
                }
            }
            output.authored = true;
            return true;
        }

        lfs::Result<FlatStage> read_crate_bytes(const std::uint8_t* data, const std::size_t size) {
            tinyusdz::StreamReader stream(data, size, false);
            tinyusdz::crate::CrateReader reader(&stream);
            if (!reader.ReadBootStrap() || !reader.ReadTOC() || !reader.ReadTokens() || !reader.ReadStrings() ||
                !reader.ReadFields() || !reader.ReadFieldSets() || !reader.ReadPaths() || !reader.ReadSpecs()) {
                return make_flat_error(reader.GetError().empty() ? "Failed to parse USD crate" : reader.GetError());
            }

            const auto& paths = reader.GetPaths();
            const auto& specs = reader.GetSpecs();
            FlatStage output;
            std::map<std::string, std::size_t> prim_indices;
            std::map<std::string, std::vector<tinyusdz::value::token>> xform_orders;
            std::map<std::string, std::map<std::string, tinyusdz::crate::CrateValue>> xform_values;
            const auto& field_indices = reader.GetFieldsetIndices();
            const auto& fields = reader.GetFields();
            const auto decode_fieldset = [&](const std::uint32_t fieldset, const auto& callback) -> lfs::Status {
                if (fieldset >= field_indices.size()) {
                    return lfs::Status::failure(make_flat_error("Malformed USD crate fieldset index"));
                }
                for (std::size_t index = fieldset; index < field_indices.size() && field_indices[index].value != std::numeric_limits<std::uint32_t>::max(); ++index) {
                    if (field_indices[index].value >= fields.size()) {
                        return lfs::Status::failure(make_flat_error("Malformed USD crate field index"));
                    }
                    const auto token = reader.GetToken(fields[field_indices[index].value].token_index);
                    if (!token) {
                        return lfs::Status::failure(make_flat_error("Malformed USD crate field token"));
                    }
                    const std::string name = token->str();
                    if (is_composition_field(name)) {
                        return lfs::Status::failure(make_flat_error(composition_error));
                    }
                    const auto& field = fields[field_indices[index].value];
                    if (callback(name, field.value_rep, static_cast<const tinyusdz::crate::CrateValue*>(nullptr))) {
                        continue;
                    }
                    tinyusdz::crate::CrateValue value;
                    if (!reader.UnpackValueRepForFlat(field.value_rep, &value)) {
                        return lfs::Status::failure(make_flat_error(reader.GetError().empty() ? "Failed to decode USD crate value" : reader.GetError()));
                    }
                    callback(name, field.value_rep, &value);
                }
                return {};
            };

            for (const auto& spec : specs) {
                if (spec.path_index.value >= paths.size()) {
                    return make_flat_error("Malformed USD crate path index");
                }
                const std::string path = paths[spec.path_index.value].full_path_name();
                if (static_cast<int>(spec.spec_type) == static_cast<int>(tinyusdz::SpecType::Prim)) {
                    std::string type_name;
                    const auto decoded = decode_fieldset(spec.fieldset_index.value, [&](const std::string& name,
                                                                                        const tinyusdz::crate::ValueRep&,
                                                                                        const tinyusdz::crate::CrateValue* value) {
                        if (name != "typeName") {
                            return true;
                        }
                        if (value) {
                            type_name = token_value(value);
                        }
                        return false;
                    });
                    if (!decoded) {
                        return decoded.error();
                    }
                    FlatPrim prim;
                    prim.path = path;
                    prim.type_name = std::move(type_name);
                    prim.local_transform = identity_matrix();
                    prim_indices[path] = output.prims.size();
                    output.prims.push_back(std::move(prim));
                }
            }

            for (const auto& spec : specs) {
                if (spec.path_index.value >= paths.size() || static_cast<int>(spec.spec_type) != static_cast<int>(tinyusdz::SpecType::Attribute)) {
                    continue;
                }
                const std::string property_path = paths[spec.path_index.value].full_path_name();
                const auto dot = property_path.find_last_of('.');
                if (dot == std::string::npos) {
                    continue;
                }
                const std::string prim_path = property_path.substr(0, dot);
                const auto prim_index = prim_indices.find(prim_path);
                if (prim_index == prim_indices.end()) {
                    continue;
                }
                const std::string name = property_path.substr(dot + 1);
                std::string type_name;
                tinyusdz::crate::CrateValue default_value;
                bool has_default = false;
                const auto type_decoded = decode_fieldset(spec.fieldset_index.value, [&](const std::string& field_name,
                                                                                         const tinyusdz::crate::ValueRep&,
                                                                                         const tinyusdz::crate::CrateValue* value) {
                    if (field_name != "typeName") {
                        return true;
                    }
                    if (value) {
                        type_name = token_value(value);
                    }
                    return false;
                });
                if (!type_decoded) {
                    return type_decoded.error();
                }
                auto& prim = output.prims[prim_index->second];
                const auto value_decoded = decode_fieldset(spec.fieldset_index.value, [&](const std::string& field_name,
                                                                                          const tinyusdz::crate::ValueRep& rep,
                                                                                          const tinyusdz::crate::CrateValue* value) {
                    if (field_name != "default") {
                        return true;
                    }
                    if (!value) {
                        if (name == "positions" || name == "positionsh" || name == "scales" || name == "scalesh" ||
                            name == "orientations" || name == "orientationsh" || name == "opacities" || name == "opacitiesh" ||
                            name == "radiance:sphericalHarmonicsCoefficients" || name == "radiance:sphericalHarmonicsCoefficientsh" ||
                            name == "extent") {
                            FlatAttribute direct;
                            if (decode_flat_array(reader, rep, type_name, direct)) {
                                prim.attributes[name] = std::move(direct);
                                has_default = true;
                                return true;
                            }
                        }
                        return false;
                    }
                    default_value = *value;
                    has_default = true;
                    return false;
                });
                if (!value_decoded) {
                    return value_decoded.error();
                }
                if (has_default) {
                    if (prim.attributes.find(name) != prim.attributes.end()) {
                        continue;
                    }
                    if (name == "xformOpOrder") {
                        if (const auto order = default_value.get_value<std::vector<tinyusdz::value::token>>()) {
                            xform_orders[prim_path] = *order;
                        }
                    } else if (name.rfind("xformOp:", 0) == 0) {
                        xform_values[prim_path][name] = std::move(default_value);
                    } else {
                        read_value(name, type_name, default_value, prim);
                    }
                }
            }

            for (const auto& spec : specs) {
                if (spec.path_index.value >= paths.size() || paths[spec.path_index.value].full_path_name() != "/") {
                    continue;
                }
                const auto decoded = decode_fieldset(spec.fieldset_index.value, [&](const std::string& name,
                                                                                    const tinyusdz::crate::ValueRep&,
                                                                                    const tinyusdz::crate::CrateValue* value) {
                    if (!value) {
                        return false;
                    }
                    if (name == "defaultPrim") {
                        const auto default_prim = token_value(value);
                        output.default_prim = default_prim.empty() ? std::string{} : "/" + default_prim;
                    } else if (name == "upAxis") {
                        output.up_axis = token_value(value);
                    } else if (name == "metersPerUnit") {
                        if (const auto meters = value->get_value<double>())
                            output.meters_per_unit = *meters;
                        if (const auto meters = value->get_value<float>())
                            output.meters_per_unit = *meters;
                    } else if (name == "customLayerData") {
                        if (const auto dictionary = value->get_value<tinyusdz::CustomDataType>()) {
                            for (const auto& item : *dictionary) {
                                if (const auto string_value = item.second.get_value<std::string>()) {
                                    output.custom_layer_data[item.first] = *string_value;
                                }
                            }
                        }
                    }
                    return false;
                });
                if (!decoded) {
                    return decoded.error();
                }
                break;
            }

            for (const auto& item : xform_orders) {
                const auto prim_index = prim_indices.find(item.first);
                if (prim_index == prim_indices.end())
                    continue;
                auto& prim = output.prims[prim_index->second];
                for (const auto& operation : item.second) {
                    std::string name = operation.str();
                    if (name == "!resetXformStack!") {
                        prim.reset_xform_stack = true;
                        continue;
                    }
                    bool inverted = false;
                    if (name.rfind("!invert!", 0) == 0) {
                        inverted = true;
                        name.erase(0, 8);
                    }
                    const auto value = xform_values[item.first].find(name);
                    if (value == xform_values[item.first].end()) {
                        return make_flat_error("USD xformOpOrder references missing operation " + name);
                    }
                    auto operation_matrix = transform_matrix(name, value->second);
                    if (!operation_matrix)
                        return operation_matrix.error();
                    if (inverted) {
                        std::array<double, 16> inverse{};
                        if (!invert_matrix(*operation_matrix, inverse)) {
                            return make_flat_error("USD xformOp is not invertible: " + name);
                        }
                        operation_matrix = inverse;
                    }
                    multiply(*operation_matrix, prim.local_transform, prim.local_transform);
                }
            }
            return output;
        }

    } // namespace

    lfs::Result<FlatStage> read_usdc(const std::filesystem::path& path) {
        auto file = MappedFile::open(path);
        if (!file) {
            return file.error();
        }
        return read_usdc_bytes(file->data(), file->size());
    }

    lfs::Result<FlatStage> read_usdc_bytes(const std::uint8_t* data, const std::size_t size) {
        if (!data && size != 0) {
            return make_flat_error("Invalid USD crate data");
        }
        return read_crate_bytes(data, size);
    }

    lfs::Status write_usdc(const FlatStage& stage, const std::filesystem::path& path) {
        struct Blob {
            std::vector<std::uint8_t> bytes;
            const FlatAttribute* attribute = nullptr;

            [[nodiscard]] std::size_t size() const {
                if (attribute) {
                    return sizeof(std::uint64_t) + attribute->values.size() * sizeof(float);
                }
                return bytes.size();
            }
        };
        struct FieldOut {
            std::string name;
            int type = 0;
            bool array = false;
            bool inline_value = false;
            std::uint64_t inline_payload = 0;
            std::size_t blob = std::numeric_limits<std::size_t>::max();
        };
        struct SpecOut {
            std::string path;
            int type = 0;
            std::vector<FieldOut> fields;
        };

        std::vector<std::string> tokens;
        std::map<std::string, std::uint32_t> token_indices;
        const auto token_index = [&](const std::string& value) {
            const auto found = token_indices.find(value);
            if (found != token_indices.end())
                return found->second;
            const auto index = static_cast<std::uint32_t>(tokens.size());
            tokens.push_back(value);
            token_indices.emplace(value, index);
            return index;
        };
        const auto add_blob = [](std::vector<Blob>& blobs, const void* data, const std::size_t size) {
            Blob blob;
            blob.bytes.resize(size);
            if (size != 0)
                std::memcpy(blob.bytes.data(), data, size);
            blobs.push_back(std::move(blob));
            return blobs.size() - 1;
        };
        const auto add_u64 = [](std::vector<std::uint8_t>& bytes, const std::uint64_t value) {
            const auto old_size = bytes.size();
            bytes.resize(old_size + sizeof(value));
            std::memcpy(bytes.data() + old_size, &value, sizeof(value));
        };
        const auto add_u32 = [](std::vector<std::uint8_t>& bytes, const std::uint32_t value) {
            const auto old_size = bytes.size();
            bytes.resize(old_size + sizeof(value));
            std::memcpy(bytes.data() + old_size, &value, sizeof(value));
        };
        const auto array_blob = [&](std::vector<Blob>& blobs, const FlatAttribute& attribute) {
            Blob blob;
            blob.attribute = &attribute;
            blobs.push_back(std::move(blob));
            return blobs.size() - 1;
        };
        const auto compressed_u32 = [](const std::vector<std::uint32_t>& values) -> lfs::Result<std::vector<std::uint8_t>> {
            std::vector<std::uint8_t> output(tinyusdz::Usd_IntegerCompression::GetCompressedBufferSize(values.size()));
            std::string error;
            const auto size = tinyusdz::Usd_IntegerCompression::CompressToBuffer(values.data(), values.size(), reinterpret_cast<char*>(output.data()), &error);
            if (!error.empty())
                return make_flat_error(error);
            output.resize(size);
            return output;
        };
        const auto compressed_i32 = [](const std::vector<std::int32_t>& values) -> lfs::Result<std::vector<std::uint8_t>> {
            std::vector<std::uint8_t> output(tinyusdz::Usd_IntegerCompression::GetCompressedBufferSize(values.size()));
            std::string error;
            const auto size = tinyusdz::Usd_IntegerCompression::CompressToBuffer(values.data(), values.size(), reinterpret_cast<char*>(output.data()), &error);
            if (!error.empty())
                return make_flat_error(error);
            output.resize(size);
            return output;
        };
        const auto compressed_bytes = [](const std::vector<std::uint8_t>& input) -> lfs::Result<std::vector<std::uint8_t>> {
            std::vector<std::uint8_t> output(tinyusdz::LZ4Compression::GetCompressedBufferSize(input.size()));
            std::string error;
            const auto size = tinyusdz::LZ4Compression::CompressToBuffer(reinterpret_cast<const char*>(input.data()), reinterpret_cast<char*>(output.data()), input.size(), &error);
            if (!error.empty())
                return make_flat_error(error);
            output.resize(size);
            return output;
        };
        std::vector<Blob> blobs;
        token_index("/");
        token_index("defaultPrim");
        token_index("upAxis");
        token_index("metersPerUnit");
        token_index("customLayerData");
        token_index("primChildren");
        token_index("specifier");
        token_index("typeName");
        token_index("properties");
        token_index("xformOpOrder");
        token_index("xformOp:transform");
        for (const auto& prim : stage.prims) {
            token_index(prim.path.substr(prim.path.find_last_of('/') + 1));
            token_index(prim.type_name);
            for (const auto& attribute : prim.attributes) {
                token_index(attribute.first);
                token_index(attribute.second.type_name);
            }
        }
        for (const auto& item : stage.custom_layer_data) {
            token_index(item.first);
            token_index(item.second);
        }
        std::map<std::string, std::uint32_t> string_indices;
        for (const auto& item : stage.custom_layer_data) {
            string_indices.emplace(item.first, 0);
            string_indices.emplace(item.second, 0);
        }
        std::uint32_t string_index = 0;
        for (auto& item : string_indices) {
            item.second = string_index++;
        }

        std::vector<SpecOut> specs;
        SpecOut root{"/", 7, {}};
        FieldOut root_children{"primChildren", 41, true, false, 0, 0};
        std::vector<std::uint32_t> root_child_tokens;
        for (const auto& prim : stage.prims) {
            if (prim.path.find('/', 1) == std::string::npos)
                root_child_tokens.push_back(token_index(prim.path.substr(prim.path.find_last_of('/') + 1)));
        }
        std::vector<std::uint8_t> root_children_bytes;
        add_u64(root_children_bytes, root_child_tokens.size());
        for (const auto value : root_child_tokens)
            add_u32(root_children_bytes, value);
        root_children.blob = add_blob(blobs, root_children_bytes.data(), root_children_bytes.size());
        root.fields.push_back(root_children);
        FieldOut default_prim{"defaultPrim", 11, false, true, token_index(stage.default_prim.substr(stage.default_prim.find_last_of('/') + 1)), 0};
        root.fields.push_back(default_prim);
        root.fields.push_back(FieldOut{"upAxis", 11, false, true, token_index(stage.up_axis), 0});
        const float meters_value = static_cast<float>(stage.meters_per_unit);
        std::uint32_t meters_bits = 0;
        std::memcpy(&meters_bits, &meters_value, sizeof(meters_bits));
        FieldOut meters{"metersPerUnit", 9, false, true, meters_bits, 0};
        root.fields.push_back(meters);
        if (!stage.custom_layer_data.empty()) {
            std::vector<std::uint8_t> dictionary;
            add_u64(dictionary, stage.custom_layer_data.size());
            for (const auto& item : stage.custom_layer_data) {
                add_u32(dictionary, string_indices[item.first]);
                add_u64(dictionary, 8);
                const auto rep = tinyusdz::crate::ValueRep(10, true, false, string_indices[item.second]);
                add_u64(dictionary, rep.GetData());
            }
            root.fields.push_back(FieldOut{"customLayerData", 31, false, false, 0, add_blob(blobs, dictionary.data(), dictionary.size())});
        }
        specs.push_back(std::move(root));

        for (const auto& prim : stage.prims) {
            SpecOut prim_spec{prim.path, 6, {}};
            prim_spec.fields.push_back(FieldOut{"specifier", 42, false, true, 0, 0});
            prim_spec.fields.push_back(FieldOut{"typeName", 11, false, true, token_index(prim.type_name), 0});
            std::vector<std::uint32_t> property_tokens;
            for (const auto& attribute : prim.attributes)
                property_tokens.push_back(token_index(attribute.first));
            if (!std::equal(prim.local_transform.begin(), prim.local_transform.end(), identity_matrix().begin())) {
                property_tokens.push_back(token_index("xformOp:transform"));
            }
            std::vector<std::uint8_t> properties;
            add_u64(properties, property_tokens.size());
            for (const auto value : property_tokens)
                add_u32(properties, value);
            prim_spec.fields.push_back(FieldOut{"properties", 41, true, false, 0, add_blob(blobs, properties.data(), properties.size())});
            if (property_tokens.size() > prim.attributes.size()) {
                std::vector<std::uint32_t> order{token_index("xformOp:transform")};
                std::vector<std::uint8_t> order_bytes;
                add_u64(order_bytes, order.size());
                for (const auto value : order)
                    add_u32(order_bytes, value);
                prim_spec.fields.push_back(FieldOut{"xformOpOrder", 41, true, false, 0, add_blob(blobs, order_bytes.data(), order_bytes.size())});
                std::vector<std::uint8_t> matrix;
                matrix.insert(matrix.end(), reinterpret_cast<const std::uint8_t*>(prim.local_transform.data()), reinterpret_cast<const std::uint8_t*>(prim.local_transform.data() + 16));
                prim_spec.fields.push_back(FieldOut{"xformOp:transform", 15, false, false, 0, add_blob(blobs, matrix.data(), matrix.size())});
            }
            specs.push_back(std::move(prim_spec));
            for (const auto& attribute : prim.attributes) {
                SpecOut property{prim.path + "." + attribute.first, 1, {}};
                property.fields.push_back(FieldOut{"typeName", 11, false, true, token_index(attribute.second.type_name), 0});
                if (attribute.first == "radiance:sphericalHarmonicsDegree") {
                    property.fields.push_back(FieldOut{"default", 3, false, true, static_cast<std::uint32_t>(attribute.second.values.front()), 0});
                } else {
                    property.fields.push_back(FieldOut{"default", attribute.second.components == 4 ? 17 : attribute.second.components == 1 ? 8
                                                                                                                                           : 24,
                                                       true, false, 0, array_blob(blobs, attribute.second)});
                }
                specs.push_back(std::move(property));
            }
        }

        std::vector<std::uint32_t> field_tokens;
        std::vector<FieldOut> flattened_fields;
        std::vector<std::uint32_t> fieldsets;
        std::vector<std::uint32_t> spec_fieldsets;
        for (const auto& spec : specs) {
            spec_fieldsets.push_back(static_cast<std::uint32_t>(fieldsets.size()));
            for (const auto& field : spec.fields) {
                field_tokens.push_back(token_index(field.name));
                fieldsets.push_back(static_cast<std::uint32_t>(flattened_fields.size()));
                flattened_fields.push_back(field);
            }
            fieldsets.push_back(std::numeric_limits<std::uint32_t>::max());
        }

        std::vector<std::string> paths;
        std::vector<std::int32_t> path_elements;
        std::vector<std::int32_t> jumps;
        paths.push_back("/");
        path_elements.push_back(stage.prims.empty() ? 0 : static_cast<std::int32_t>(token_index(stage.prims.front().path.substr(stage.prims.front().path.find_last_of('/') + 1))));
        jumps.push_back(stage.prims.empty() ? -2 : -1);
        for (const auto& spec : specs) {
            if (spec.path == "/")
                continue;
            paths.push_back(spec.path);
            const auto dot = spec.path.find_last_of('.');
            const auto slash = spec.path.find_last_of('/');
            const bool property = dot != std::string::npos && dot > slash;
            path_elements.push_back(property ? -static_cast<std::int32_t>(token_index(spec.path.substr(dot + 1))) : static_cast<std::int32_t>(token_index(spec.path.substr(slash + 1))));
            jumps.push_back(property ? (spec.path == specs.back().path ? -2 : 0) : -1);
        }

        const auto token_bytes = [&]() {
            std::vector<std::uint8_t> raw;
            for (const auto& token : tokens)
                raw.insert(raw.end(), token.begin(), token.end()), raw.push_back(0);
            return raw;
        }();
        auto compressed_tokens = compressed_bytes(token_bytes);
        if (!compressed_tokens)
            return lfs::Status::failure(compressed_tokens.error());
        std::vector<std::uint8_t> tokens_section;
        add_u64(tokens_section, tokens.size());
        add_u64(tokens_section, token_bytes.size());
        add_u64(tokens_section, compressed_tokens->size());
        tokens_section.insert(tokens_section.end(), compressed_tokens->begin(), compressed_tokens->end());
        std::vector<std::uint8_t> strings_section;
        add_u64(strings_section, string_indices.size());
        for (const auto& item : string_indices)
            add_u32(strings_section, token_index(item.first));
        auto compressed_field_tokens = compressed_u32(field_tokens);
        auto compressed_fieldsets = compressed_u32(fieldsets);
        auto compressed_path_elements = compressed_i32(path_elements);
        auto compressed_jumps = compressed_i32(jumps);
        if (!compressed_field_tokens || !compressed_fieldsets || !compressed_path_elements || !compressed_jumps)
            return lfs::Status::failure(make_flat_error("Failed to compress USD crate structure"));
        std::vector<std::uint32_t> path_indices(paths.size());
        for (std::size_t i = 0; i < paths.size(); ++i)
            path_indices[i] = static_cast<std::uint32_t>(i);
        auto compressed_path_indices = compressed_u32(path_indices);
        if (!compressed_path_indices)
            return lfs::Status::failure(compressed_path_indices.error());

        std::vector<std::uint8_t> fieldsets_section;
        add_u64(fieldsets_section, fieldsets.size());
        add_u64(fieldsets_section, compressed_fieldsets->size());
        fieldsets_section.insert(fieldsets_section.end(), compressed_fieldsets->begin(), compressed_fieldsets->end());
        std::vector<std::uint8_t> paths_section;
        add_u64(paths_section, paths.size());
        add_u64(paths_section, path_elements.size());
        add_u64(paths_section, compressed_path_indices->size());
        paths_section.insert(paths_section.end(), compressed_path_indices->begin(), compressed_path_indices->end());
        add_u64(paths_section, compressed_path_elements->size());
        paths_section.insert(paths_section.end(), compressed_path_elements->begin(), compressed_path_elements->end());
        add_u64(paths_section, compressed_jumps->size());
        paths_section.insert(paths_section.end(), compressed_jumps->begin(), compressed_jumps->end());

        std::vector<std::uint32_t> spec_paths(specs.size());
        std::vector<std::uint32_t> spec_types(specs.size());
        for (std::size_t index = 0; index < specs.size(); ++index) {
            spec_paths[index] = static_cast<std::uint32_t>(index);
            spec_types[index] = static_cast<std::uint32_t>(specs[index].type);
        }
        auto compressed_spec_paths = compressed_u32(spec_paths);
        auto compressed_spec_sets = compressed_u32(spec_fieldsets);
        auto compressed_spec_types = compressed_u32(spec_types);
        if (!compressed_spec_paths || !compressed_spec_sets || !compressed_spec_types)
            return lfs::Status::failure(make_flat_error("Failed to compress USD crate specs"));
        std::vector<std::uint8_t> specs_section;
        add_u64(specs_section, specs.size());
        add_u64(specs_section, compressed_spec_paths->size());
        specs_section.insert(specs_section.end(), compressed_spec_paths->begin(), compressed_spec_paths->end());
        add_u64(specs_section, compressed_spec_sets->size());
        specs_section.insert(specs_section.end(), compressed_spec_sets->begin(), compressed_spec_sets->end());
        add_u64(specs_section, compressed_spec_types->size());
        specs_section.insert(specs_section.end(), compressed_spec_types->begin(), compressed_spec_types->end());

        const auto padded = [](std::size_t value) { return (value + 7u) & ~std::size_t(7u); };
        const std::size_t header_size = 80;
        const std::size_t fields_size_without_reps = sizeof(std::uint64_t) + sizeof(std::uint64_t) + compressed_field_tokens->size() + sizeof(std::uint64_t);
        const std::size_t fields_reps_raw_size = flattened_fields.size() * sizeof(std::uint64_t);
        const std::size_t fields_reps_compressed_size = tinyusdz::LZ4Compression::GetCompressedBufferSize(fields_reps_raw_size);
        const std::size_t fields_section_size = fields_size_without_reps + fields_reps_compressed_size;
        const std::size_t section_data_end = padded(header_size + tokens_section.size()) + padded(strings_section.size()) + padded(fields_section_size) + padded(fieldsets_section.size()) + padded(paths_section.size()) + padded(specs_section.size());
        const std::size_t data_base = padded(section_data_end);
        std::vector<std::uint64_t> reps;
        reps.reserve(flattened_fields.size());
        std::size_t blob_offset = 0;
        std::vector<std::size_t> absolute_blob_offsets(blobs.size());
        for (std::size_t index = 0; index < blobs.size(); ++index) {
            absolute_blob_offsets[index] = data_base + blob_offset;
            blob_offset += padded(blobs[index].size());
        }
        for (const auto& field : flattened_fields) {
            if (field.inline_value) {
                reps.push_back(tinyusdz::crate::ValueRep(field.type, true, field.array, field.inline_payload).GetData());
            } else {
                reps.push_back(tinyusdz::crate::ValueRep(field.type, false, field.array, absolute_blob_offsets[field.blob]).GetData());
            }
        }
        std::vector<std::uint8_t> reps_raw(reps.size() * sizeof(std::uint64_t));
        if (!reps.empty())
            std::memcpy(reps_raw.data(), reps.data(), reps_raw.size());
        auto compressed_reps = compressed_bytes(reps_raw);
        if (!compressed_reps)
            return lfs::Status::failure(compressed_reps.error());
        std::vector<std::uint8_t> fields_section;
        add_u64(fields_section, flattened_fields.size());
        add_u64(fields_section, compressed_field_tokens->size());
        fields_section.insert(fields_section.end(), compressed_field_tokens->begin(), compressed_field_tokens->end());
        add_u64(fields_section, compressed_reps->size());
        fields_section.insert(fields_section.end(), compressed_reps->begin(), compressed_reps->end());

        struct Section {
            std::string name;
            std::vector<std::uint8_t>* bytes;
            std::size_t start = 0;
        };
        std::vector<Section> sections = {{"TOKENS", &tokens_section}, {"STRINGS", &strings_section}, {"FIELDS", &fields_section}, {"FIELDSETS", &fieldsets_section}, {"PATHS", &paths_section}, {"SPECS", &specs_section}};
        std::size_t offset = header_size;
        for (auto& section : sections) {
            offset = padded(offset);
            section.start = offset;
            offset += section.bytes->size();
        }
        offset = padded(offset);
        const std::size_t toc_offset = padded(data_base + blob_offset);
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output)
            return lfs::Status::failure(make_flat_error("Failed to open USD crate for writing: " + lfs::core::path_to_utf8(path)));

        const auto write_zeros = [&](std::size_t count) {
            static constexpr std::size_t chunk_size = 4096;
            const std::array<char, chunk_size> zeros{};
            while (count != 0) {
                const auto chunk = std::min(count, chunk_size);
                output.write(zeros.data(), static_cast<std::streamsize>(chunk));
                count -= chunk;
            }
        };
        const auto write_blob = [&](const Blob& blob) {
            if (!blob.attribute) {
                output.write(reinterpret_cast<const char*>(blob.bytes.data()), static_cast<std::streamsize>(blob.bytes.size()));
                return;
            }
            const auto& attribute = *blob.attribute;
            const std::uint64_t count = attribute.components == 0 ? 0 : attribute.values.size() / static_cast<std::size_t>(attribute.components);
            output.write(reinterpret_cast<const char*>(&count), sizeof(count));
            if (attribute.type_name.find("quat") == std::string::npos) {
                output.write(reinterpret_cast<const char*>(attribute.values.data()),
                             static_cast<std::streamsize>(attribute.values.size() * sizeof(float)));
                return;
            }
            constexpr std::size_t records_per_chunk = 1024;
            std::array<float, records_per_chunk * 4> chunk{};
            for (std::size_t first = 0; first < count; first += records_per_chunk) {
                const auto records = std::min(records_per_chunk, count - first);
                for (std::size_t index = 0; index < records; ++index) {
                    const float* values = attribute.values.data() + (first + index) * 4;
                    float* ordered = chunk.data() + index * 4;
                    ordered[0] = values[1];
                    ordered[1] = values[2];
                    ordered[2] = values[3];
                    ordered[3] = values[0];
                }
                output.write(reinterpret_cast<const char*>(chunk.data()), static_cast<std::streamsize>(records * 4 * sizeof(float)));
            }
        };

        std::array<std::uint8_t, 80> header{};
        std::memcpy(header.data(), "PXR-USDC", 8);
        header[8] = 0;
        header[9] = 8;
        header[10] = 0;
        std::memcpy(header.data() + 16, &toc_offset, sizeof(toc_offset));
        output.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
        offset = header_size;
        for (const auto& section : sections) {
            write_zeros(section.start - offset);
            output.write(reinterpret_cast<const char*>(section.bytes->data()), static_cast<std::streamsize>(section.bytes->size()));
            offset = section.start + section.bytes->size();
        }
        write_zeros(data_base - offset);
        offset = data_base;
        for (const auto& blob : blobs) {
            write_blob(blob);
            const auto blob_size = blob.size();
            write_zeros(padded(blob_size) - blob_size);
            offset += padded(blob_size);
        }
        write_zeros(toc_offset - offset);
        const std::uint64_t section_count = sections.size();
        output.write(reinterpret_cast<const char*>(&section_count), sizeof(section_count));
        for (const auto& section : sections) {
            std::array<char, 32> toc_entry{};
            std::memcpy(toc_entry.data(), section.name.c_str(), std::min<std::size_t>(section.name.size(), 15));
            const std::uint64_t start = section.start;
            const std::uint64_t section_size = section.bytes->size();
            std::memcpy(toc_entry.data() + 16, &start, sizeof(start));
            std::memcpy(toc_entry.data() + 24, &section_size, sizeof(section_size));
            output.write(toc_entry.data(), static_cast<std::streamsize>(toc_entry.size()));
        }
        if (!output)
            return lfs::Status::failure(make_flat_error("Failed to write USD crate: " + lfs::core::path_to_utf8(path)));
        return {};
    }

} // namespace lfs::io::usd_flat
