/*
MIT License

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

#ifndef SPZ_SAFE_ORBIT_CAMERA_ADOBE_H_
#define SPZ_SAFE_ORBIT_CAMERA_ADOBE_H_

#include "splat-extensions.h"

namespace spz {

struct SpzExtensionSafeOrbitCameraAdobe : public SpzExtensionBase {
  float safeOrbitElevationMin = 0.0f;  // Minimum elevation for safe orbit (radians)
  float safeOrbitElevationMax = 0.0f;  // Maximum elevation for safe orbit (radians)
  float safeOrbitRadiusMin = 0.0f;     // Minimum radius for safe orbit

  static const std::unordered_set<std::string> kRequiredPlyElementNames;

  SpzExtensionSafeOrbitCameraAdobe();
  uint32_t payloadBytes() const override;
  void write(std::ostream& os) const override;
  SpzExtensionBase* copyAsRawData() const override;
  std::optional<std::shared_ptr<SpzExtensionBase>> tryReadFromPly(
      std::istream& in, const std::unordered_set<std::string>& elementNames) const override;
  void writePlyHeader(std::ostream& out) const override;
  void writePlyData(std::ostream& out) const override;
  static std::optional<SpzExtensionBasePtr> read(std::istream& is);
  static SpzExtensionType type();
};

}  // namespace spz

#endif  // SPZ_SAFE_ORBIT_CAMERA_ADOBE_H_
