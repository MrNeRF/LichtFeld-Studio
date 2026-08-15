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

#include "safe-orbit-camera-adobe.h"
#include "load-spz.h"

#include <sstream>
#include <unordered_set>

namespace spz {
    namespace {
        template <class T>
        bool readExact(std::istream& is, T& out) {
            return static_cast<bool>(is.read(reinterpret_cast<char*>(&out), sizeof(T)));
        }
    } // namespace

    const std::unordered_set<std::string> SpzExtensionSafeOrbitCameraAdobe::kRequiredPlyElementNames = {
        "safe_orbit_camera_elevation_min_max_radians",
        "safe_orbit_camera_radius_min",
    };

    SpzExtensionSafeOrbitCameraAdobe::SpzExtensionSafeOrbitCameraAdobe()
        : SpzExtensionBase(SpzExtensionType::SPZ_ADOBE_safe_orbit_camera) {}

    uint32_t SpzExtensionSafeOrbitCameraAdobe::payloadBytes() const {
        return sizeof(safeOrbitElevationMin) + sizeof(safeOrbitElevationMax) + sizeof(safeOrbitRadiusMin);
    }

    void SpzExtensionSafeOrbitCameraAdobe::write(std::ostream& os) const {
        const uint32_t t = static_cast<uint32_t>(extensionType);
        const uint32_t payloadBytes = this->payloadBytes();
        SpzLogDebug("[SPZ] Writing extension: SafeOrbitCameraAdobe");
        os.write(reinterpret_cast<const char*>(&t), sizeof(t));
        os.write(reinterpret_cast<const char*>(&payloadBytes), sizeof(payloadBytes));
        os.write(reinterpret_cast<const char*>(&safeOrbitElevationMin), sizeof(safeOrbitElevationMin));
        os.write(reinterpret_cast<const char*>(&safeOrbitElevationMax), sizeof(safeOrbitElevationMax));
        os.write(reinterpret_cast<const char*>(&safeOrbitRadiusMin), sizeof(safeOrbitRadiusMin));
    }

    std::optional<SpzExtensionBasePtr> SpzExtensionSafeOrbitCameraAdobe::read(std::istream& is) {
        SpzLogDebug("[SPZ] Found extension: SafeOrbitCameraAdobe");
        auto rec = std::make_shared<SpzExtensionSafeOrbitCameraAdobe>();
        if (!readExact(is, rec->safeOrbitElevationMin) || !readExact(is, rec->safeOrbitElevationMax) ||
            !readExact(is, rec->safeOrbitRadiusMin)) {
            SpzLog("[SPZ ERROR] Failed to read all fields for SafeOrbitCameraAdobe");
            return std::nullopt;
        }
        return std::optional{std::move(rec)};
    }

    SpzExtensionType SpzExtensionSafeOrbitCameraAdobe::type() {
        return SpzExtensionType::SPZ_ADOBE_safe_orbit_camera;
    }

    SpzExtensionBase* SpzExtensionSafeOrbitCameraAdobe::copyAsRawData() const {
        return new SpzExtensionSafeOrbitCameraAdobe(*this);
    }

    std::optional<SpzExtensionBasePtr> SpzExtensionSafeOrbitCameraAdobe::tryReadFromPly(
        std::istream& in, const std::unordered_set<std::string>& elementNames) const {
        for (const auto& name : kRequiredPlyElementNames) {
            if (elementNames.count(name) == 0)
                return std::nullopt;
        }
        float safeOrbitData[3];
        if (!readExact(in, safeOrbitData[0]) || !readExact(in, safeOrbitData[1]) ||
            !readExact(in, safeOrbitData[2]))
            return std::nullopt;
        auto rec = std::make_shared<SpzExtensionSafeOrbitCameraAdobe>();
        rec->safeOrbitElevationMin = safeOrbitData[0];
        rec->safeOrbitElevationMax = safeOrbitData[1];
        rec->safeOrbitRadiusMin = safeOrbitData[2];
        SpzLogDebug("[SPZ] Loaded safe orbit data: elevation [%f, %f], radius min %f",
                    rec->safeOrbitElevationMin, rec->safeOrbitElevationMax, rec->safeOrbitRadiusMin);
        return std::optional{std::move(rec)};
    }

    void SpzExtensionSafeOrbitCameraAdobe::writePlyHeader(std::ostream& out) const {
        out << "element safe_orbit_camera_elevation_min_max_radians 2\n";
        out << "property float safe_orbit_camera_elevation_min_max_radians\n";
        out << "element safe_orbit_camera_radius_min 1\n";
        out << "property float safe_orbit_camera_radius_min\n";
    }

    void SpzExtensionSafeOrbitCameraAdobe::writePlyData(std::ostream& out) const {
        float safeOrbitData[3] = {
            safeOrbitElevationMin,
            safeOrbitElevationMax,
            safeOrbitRadiusMin};
        out.write(reinterpret_cast<const char*>(safeOrbitData), sizeof(safeOrbitData));
    }

} // namespace spz
