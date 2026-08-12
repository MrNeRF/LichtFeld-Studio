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

#ifndef SPZ_COORDINATE_SYSTEM_ADOBE_H_
#define SPZ_COORDINATE_SYSTEM_ADOBE_H_

#include "splat-extensions.h"
#include "splat-types.h"

namespace spz {

// Records the coordinate system in which the Gaussian data is physically stored in the file.
//
// This is a descriptor, not an instruction. It labels what coordinate system the packed data is
// already in — it does not tell the reader to apply any conversion. packGaussians and
// unpackGaussians consume this extension internally and handle all conversions automatically via
// PackOptions::from and UnpackOptions::to. Consumers using those functions must not apply any
// additional conversion based on this extension; doing so will double-transform the data.
//
// Mechanics: when present with a non-UNSPECIFIED value, packGaussians writes data in
// coordinateSystem (instead of the default RUB), and unpackGaussians reads from coordinateSystem
// (instead of RUB) before converting to UnpackOptions::to.
//
// UNSPECIFIED is equivalent to the extension being absent: both fall back to RUB, but a present
// UNSPECIFIED value is likely a writer bug and will produce a warning.
struct SpzExtensionCoordinateSystemAdobe : public SpzExtensionBase {
  CoordinateSystem coordinateSystem = CoordinateSystem::UNSPECIFIED;

  SpzExtensionCoordinateSystemAdobe();
  uint32_t payloadBytes() const override;
  void write(std::ostream& os) const override;
  SpzExtensionBase* copyAsRawData() const override;
  std::optional<std::shared_ptr<SpzExtensionBase>> tryReadFromPly(
      std::istream& in, const std::unordered_set<std::string>& elementNames) const override;
  void writePlyHeader(std::ostream& out) const override;
  void writePlyData(std::ostream& out) const override;
  static std::optional<SpzExtensionBasePtr> read(std::istream& is);
  static SpzExtensionType type();

  // Returns the storage coordinate system recorded by this extension.
  // Returns RUB and logs a warning if coordinateSystem is UNSPECIFIED (likely a writer bug).
  CoordinateSystem resolve() const;
};

}  // namespace spz

#endif  // SPZ_COORDINATE_SYSTEM_ADOBE_H_
