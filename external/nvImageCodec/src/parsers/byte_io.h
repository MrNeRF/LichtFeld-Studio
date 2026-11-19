/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <type_traits>
#include <nvimgcodec.h>
#include <cstring>
#include <cassert>
#include <stdexcept>

namespace nvimgcodec {

class UnexpectedEndOfStream : public std::runtime_error
{
  public:
    UnexpectedEndOfStream()
        : std::runtime_error("Unexpected end of stream")
    {
    }
};

namespace detail {

template <int nbytes, bool is_little_endian, typename T>
std::enable_if_t<std::is_integral<T>::value> ReadValueImpl(T &value, const uint8_t* data) {
  static_assert(sizeof(T) >= nbytes, "T can't hold the requested number of bytes");
  value = 0;
  constexpr unsigned pad = (sizeof(T) - nbytes) * 8;  // handle sign when nbytes < sizeof(T)
  for (int i = 0; i < nbytes; i++) {
    unsigned shift = is_little_endian ? (i*8) + pad: (sizeof(T)-1-i)*8;
    value |= T(data[i]) << shift;
  }
  value >>= pad;
}

template <int nbytes, bool is_little_endian, typename T>
std::enable_if_t<std::is_enum<T>::value> ReadValueImpl(T &value, const uint8_t* data) {
  using U = std::underlying_type_t<T>;
  static_assert(nbytes <= sizeof(U),
    "`nbytes` should not exceed the size of the underlying type of the enum");
  U tmp;
  ReadValueImpl<nbytes, is_little_endian>(tmp, data);
  value = static_cast<T>(tmp);
}

template <int nbytes, bool is_little_endian>
void ReadValueImpl(float &value, const uint8_t* data) {
  static_assert(nbytes == sizeof(float),
    "nbytes is expected to be the same as sizeof(float)");
  uint32_t tmp;
  ReadValueImpl<nbytes, is_little_endian>(tmp, data);
  memcpy(&value, &tmp, sizeof(float));
}

template <int nbytes, bool is_little_endian, typename T>
void ReadValueImpl(T &value, nvimgcodecIoStreamDesc_t* io_stream) {
  uint8_t data[nbytes];  // NOLINT [runtime/arrays]
  size_t read_nbytes = 0;
  auto status = io_stream->read(io_stream->instance, &read_nbytes, data, nbytes);
  if (status != NVIMGCODEC_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to read");
  }
  if (read_nbytes != nbytes) {
      throw UnexpectedEndOfStream();
  }
  return ReadValueImpl<nbytes, is_little_endian>(value, data);
}

}  // namespace detail


/**
 * @brief Reads value of size `nbytes` from a stream of bytes (little-endian)
 */
template <typename T, int nbytes = sizeof(T)>
T ReadValueLE(const uint8_t* data) {
  T ret;
  detail::ReadValueImpl<nbytes, true>(ret, data);
  return ret;
}

/**
 * @brief Reads value of size `nbytes` from a stream of bytes (big-endian)
 */
template <typename T, int nbytes = sizeof(T)>
T ReadValueBE(const uint8_t* data) {
  T ret;
  detail::ReadValueImpl<nbytes, false>(ret, data);
  return ret;
}

/**
 * @brief Reads value of size `nbytes` from an input stream (little-endian)
 */
template <typename T, int nbytes = sizeof(T)>
T ReadValueLE(nvimgcodecIoStreamDesc_t* stream) {
  T ret;
  detail::ReadValueImpl<nbytes, true>(ret, stream);
  return ret;
}

/**
 * @brief Reads value of size `nbytes` from an input stream (big-endian)
 */
template <typename T, int nbytes = sizeof(T)>
T ReadValueBE(nvimgcodecIoStreamDesc_t* stream) {
  T ret;
  detail::ReadValueImpl<nbytes, false>(ret, stream);
  return ret;
}

template <typename T>
T ReadValue(nvimgcodecIoStreamDesc_t* io_stream) {
    size_t read_nbytes = 0;
    T data;
    auto status = io_stream->read(io_stream->instance, &read_nbytes, &data, sizeof(T));
    if (status != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to read");
    if (read_nbytes != sizeof(T))
        throw UnexpectedEndOfStream();
    return data;
}

}  // namespace nvimgcodec
