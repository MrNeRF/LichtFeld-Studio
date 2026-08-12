/*
MIT License

Copyright (c) 2025 Niantic Labs
Copyright (c) 2025 Adobe Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef SPZ_SPLAT_EXTENSIONS_H_
#define SPZ_SPLAT_EXTENSIONS_H_

#include <cstdint>
#include <iostream>
#include <istream>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

// C-compatible extension node structure (used for C interop)
typedef struct SpzExtensionNode {
  uint32_t type;
  void* data;
  struct SpzExtensionNode* next;
} SpzExtensionNode;

namespace spz {
// Forward declaration to avoid circular dependency
// The full definition is in load-spz.h, which is included in splat-extensions.cc
struct PlyExtraElement;
enum class CoordinateSystem : uint32_t;
enum class SpzExtensionType : uint32_t {
  SPZ_ADOBE_safe_orbit_camera = 0xADBE0002u,
  SPZ_ADOBE_coordinate_system = 0xADBE0003u,
};

struct SpzExtensionBase {
  SpzExtensionType extensionType;
  explicit SpzExtensionBase(SpzExtensionType t) : extensionType(t) {
  }
  virtual ~SpzExtensionBase() = default;
  virtual uint32_t payloadBytes() const = 0;
  virtual void write(std::ostream& os) const = 0;
  virtual SpzExtensionBase* copyAsRawData() const = 0;

  // PLY extension I/O: check-and-read from PLY, write header lines, write data.
  virtual std::optional<std::shared_ptr<SpzExtensionBase>> tryReadFromPly(
      std::istream& in, const std::unordered_set<std::string>& elementNames) const = 0;
  virtual void writePlyHeader(std::ostream& out) const = 0;
  virtual void writePlyData(std::ostream& out) const = 0;
};

using SpzExtensionBasePtr = std::shared_ptr<SpzExtensionBase>;

// Built-in extension types are declared in their own headers, e.g. safe-orbit-camera-adobe.h
void readAllExtensions(std::istream& is, std::vector<SpzExtensionBasePtr>& out);

void writeAllExtensions(const std::vector<SpzExtensionBasePtr>& list, std::ostream& os);

void readExtensionsFromPly(std::istream& in, const std::vector<spz::PlyExtraElement>& extraElements, std::vector<SpzExtensionBasePtr>& extensions);

void writeExtensionsToPlyHeader(const std::vector<SpzExtensionBasePtr>& extensions, std::ostream& out);

void writeExtensionsToPlyData(const std::vector<SpzExtensionBasePtr>& extensions, std::ostream& out);

bool isKnownPlyExtensionElement(const std::string& elementName);

// Returns the coordinate system in which the Gaussian data is stored, as recorded by any
// behavior-modifying extension. Falls back to RUB if no such extension is present.
CoordinateSystem getPackedCoordinateSystem(const std::vector<SpzExtensionBasePtr>& extensions);

inline SpzExtensionNode* copyExtensions(const std::vector<SpzExtensionBasePtr> &extensions) {
  SpzExtensionNode* head = nullptr;
  SpzExtensionNode* tail = nullptr;

  for (const auto& ext : extensions) {
    if (!ext) continue;  // Skip null extensions

    SpzExtensionNode* node = new SpzExtensionNode{static_cast<uint32_t>(ext->extensionType), ext->copyAsRawData(), nullptr};

    if (!head) {
      head = node;
      tail = node;
    } else {
      tail->next = node;
      tail = node;
    }
  }

  return head;
}

template <typename T>
std::shared_ptr<T> findExtensionByType(const std::vector<SpzExtensionBasePtr>& list) {
  for (const auto& rec : list) {
    if (rec->extensionType == T::type())
      return std::dynamic_pointer_cast<T>(rec);
  }
  return nullptr;
}
}  // namespace spz

#endif  // SPZ_SPLAT_EXTENSIONS_H_