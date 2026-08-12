/*
MIT License

Copyright (c) 2026 Adobe Inc.

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

#include "coordinate-system-adobe.h"
#include "load-spz.h"

namespace spz {
namespace {
template <class T>
bool readExact(std::istream& is, T& out) {
  return static_cast<bool>(is.read(reinterpret_cast<char*>(&out), sizeof(T)));
}
}  // namespace

CoordinateSystem SpzExtensionCoordinateSystemAdobe::resolve() const {
  if (coordinateSystem == CoordinateSystem::UNSPECIFIED) {
    SpzLog("[SPZ WARNING] CoordinateSystemAdobe: coordinate system is UNSPECIFIED — falling back to RUB");
    return CoordinateSystem::RUB;
  }
  return coordinateSystem;
}

SpzExtensionCoordinateSystemAdobe::SpzExtensionCoordinateSystemAdobe()
    : SpzExtensionBase(SpzExtensionType::SPZ_ADOBE_coordinate_system) {}

uint32_t SpzExtensionCoordinateSystemAdobe::payloadBytes() const {
  return sizeof(uint32_t);  // CoordinateSystem value
}

void SpzExtensionCoordinateSystemAdobe::write(std::ostream& os) const {
  const uint32_t t = static_cast<uint32_t>(extensionType);
  const uint32_t len = payloadBytes();
  const uint32_t coordValue = static_cast<uint32_t>(coordinateSystem);
  SpzLogDebug("[SPZ] Writing extension: CoordinateSystemAdobe (coord=%u)", static_cast<unsigned>(coordValue));
  os.write(reinterpret_cast<const char*>(&t), sizeof(t));
  os.write(reinterpret_cast<const char*>(&len), sizeof(len));
  os.write(reinterpret_cast<const char*>(&coordValue), sizeof(coordValue));
}

std::optional<SpzExtensionBasePtr> SpzExtensionCoordinateSystemAdobe::read(std::istream& is) {
  uint32_t coordValue{};
  if (!readExact(is, coordValue)) {
    SpzLog("[SPZ ERROR] Failed to read CoordinateSystemAdobe payload");
    return std::nullopt;
  }
  constexpr uint32_t kMaxCoordinateSystem = 16;  // RBU, the highest defined value
  if (coordValue > kMaxCoordinateSystem) {
    SpzLog("[SPZ ERROR] CoordinateSystemAdobe extension has unknown coordinate system value: %u", static_cast<unsigned>(coordValue));
    return std::nullopt;
  }
  auto rec = std::make_shared<SpzExtensionCoordinateSystemAdobe>();
  rec->coordinateSystem = static_cast<CoordinateSystem>(coordValue);
  SpzLogDebug("[SPZ] Found extension: CoordinateSystemAdobe (coord=%u)", static_cast<unsigned>(coordValue));
  return std::optional{std::move(rec)};
}

SpzExtensionType SpzExtensionCoordinateSystemAdobe::type() {
  return SpzExtensionType::SPZ_ADOBE_coordinate_system;
}

SpzExtensionBase* SpzExtensionCoordinateSystemAdobe::copyAsRawData() const {
  return new SpzExtensionCoordinateSystemAdobe(*this);
}

// No PLY representation for this extension.
std::optional<SpzExtensionBasePtr> SpzExtensionCoordinateSystemAdobe::tryReadFromPly(
    std::istream&, const std::unordered_set<std::string>&) const {
  return std::nullopt;
}

void SpzExtensionCoordinateSystemAdobe::writePlyHeader(std::ostream&) const {}

void SpzExtensionCoordinateSystemAdobe::writePlyData(std::ostream&) const {}

}  // namespace spz
