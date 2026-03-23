#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ostream>

// 2D and 4D vectors, used for area and color calculations.
// Some helpers for color conversion too.
// This just saves using a library like glm (my personal preference)
// - and it keeps the dependencies of Zep to just the source folder contents

namespace Zep {

    const inline float ZPI = 3.14159862f;
    template <class T>
    struct NVec2 {
        NVec2(T xVal, T yVal)
            : x(xVal),
              y(yVal) {
        }

        explicit NVec2(T v)
            : x(v),
              y(v) {
        }

        NVec2()
            : x(0),
              y(0) {
        }

        T x;
        T y;

        bool operator==(const NVec2<T>& rhs) const {
            if (x == rhs.x && y == rhs.y)
                return true;
            return false;
        }

        bool operator!=(const NVec2<T>& rhs) const {
            return !(*this == rhs);
        }
    };
    template <class T>
    inline NVec2<T> operator+(const NVec2<T>& lhs, const NVec2<T>& rhs) {
        return NVec2<T>(lhs.x + rhs.x, lhs.y + rhs.y);
    }
    template <class T>
    inline NVec2<T> operator-(const NVec2<T>& lhs, const NVec2<T>& rhs) {
        return NVec2<T>(lhs.x - rhs.x, lhs.y - rhs.y);
    }
    template <class T>
    inline NVec2<T> operator/(const NVec2<T>& lhs, const NVec2<T>& rhs) {
        return NVec2<T>(lhs.x / rhs.x, lhs.y / rhs.y);
    }
    template <class T>
    inline NVec2<T>& operator+=(NVec2<T>& lhs, const NVec2<T>& rhs) {
        lhs.x += rhs.x;
        lhs.y += rhs.y;
        return lhs;
    }
    template <class T>
    inline NVec2<T>& operator-=(NVec2<T>& lhs, const NVec2<T>& rhs) {
        lhs.x -= rhs.x;
        lhs.y -= rhs.y;
        return lhs;
    }
    template <class T>
    inline NVec2<T> operator*(const NVec2<T>& lhs, float val) {
        return NVec2<T>(lhs.x * val, lhs.y * val);
    }
    template <class T>
    inline NVec2<T> operator*(const NVec2<T>& lhs, const NVec2<T>& rhs) {
        return NVec2<T>(lhs.x * rhs.x, lhs.y * rhs.y);
    }
    template <class T>
    inline NVec2<T> operator/(const NVec2<T>& lhs, float val) {
        return NVec2<T>(lhs.x / val, lhs.y / val);
    }
    template <class T>
    inline NVec2<T>& operator*=(NVec2<T>& lhs, float val) {
        lhs.x *= val;
        lhs.y *= val;
        return lhs;
    }
    template <class T>
    inline bool operator<(const NVec2<T>& lhs, const NVec2<T>& rhs) {
        if (lhs.x != rhs.x) {
            return lhs.x < rhs.x;
        }
        return lhs.y < rhs.y;
    }
    template <class T>
    inline NVec2<T> Clamp(const NVec2<T>& val, const NVec2<T>& min, const NVec2<T>& max) {
        return NVec2<T>(std::min(max.x, std::max(min.x, val.x)), std::min(max.y, std::max(min.y, val.y)));
    }
    template <class T>
    inline T ManhattanDistance(const NVec2<T>& l, const NVec2<T>& r) {
        return std::abs(l.x - r.x) + std::abs(r.y - l.y);
    }

    template <class T>
    std::ostream& operator<<(std::ostream& str, const NVec2<T>& v) {
        str << "(" << v.x << ", " << v.y << ")";
        return str;
    }

    using NVec2f = NVec2<float>;
    using NVec2i = NVec2<long>;
    using NVec2int = NVec2<int>;

    template <class T>
    struct NVec4 {
        NVec4(T xVal, T yVal, T zVal, T wVal)
            : x(xVal),
              y(yVal),
              z(zVal),
              w(wVal) {
        }

        explicit NVec4(T val)
            : NVec4(val, val, val, val) {
        }

        NVec4()
            : x(0),
              y(0),
              z(0),
              w(1) {
        }

        T x;
        T y;
        T z;
        T w;

        bool operator==(const NVec4<T>& rhs) const {
            if (x == rhs.x && y == rhs.y && z == rhs.z && w == rhs.w)
                return true;
            return false;
        }

        bool operator!=(const NVec4<T>& rhs) const {
            return !(*this = rhs);
        }
    };
    template <class T>
    inline NVec4<T> operator+(const NVec4<T>& lhs, const NVec4<T>& rhs) {
        return NVec4<T>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
    }
    template <class T>
    inline NVec4<T> operator-(const NVec4<T>& lhs, const NVec4<T>& rhs) {
        return NVec4<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
    }
    template <class T>
    inline NVec4<T>& operator+=(NVec4<T>& lhs, const NVec4<T>& rhs) {
        lhs.x += rhs.x;
        lhs.y += rhs.y;
        lhs.z += rhs.z;
        lhs.w += rhs.w;
        return lhs;
    }
    template <class T>
    inline NVec4<T>& operator*=(NVec4<T>& lhs, const NVec4<T>& rhs) {
        lhs.x *= rhs.x;
        lhs.y *= rhs.y;
        lhs.z *= rhs.z;
        lhs.w *= rhs.w;
        return lhs;
    }
    template <class T>
    inline NVec4<T>& operator-=(NVec4<T>& lhs, const NVec4<T>& rhs) {
        lhs.x -= rhs.x;
        lhs.y -= rhs.y;
        lhs.z -= rhs.z;
        lhs.w -= rhs.w;
        return lhs;
    }
    template <class T>
    inline NVec4<T> operator*(const NVec4<T>& lhs, float val) {
        return NVec4<T>(lhs.x * val, lhs.y * val, lhs.z * val, lhs.w * val);
    }
    template <class T>
    inline NVec4<T>& operator*=(NVec4<T>& lhs, float val) {
        lhs.x *= val;
        lhs.y *= val;
        lhs.z *= val;
        lhs.w *= val;
        return lhs;
    }
    template <class T>
    inline NVec4<T> Clamp(const NVec4<T>& val, const NVec4<T>& min, const NVec4<T>& max) {
        return NVec4<T>(std::min(max.x, std::max(min.x, val.x)),
                        std::min(max.y, std::max(min.y, val.y)),
                        std::min(max.z, std::max(min.z, val.z)),
                        std::min(max.w, std::max(min.w, val.w)));
    }

    inline uint32_t ToPacked(const NVec4<float>& val) {
        uint32_t col = 0;
        col |= uint32_t(val.x * 255.0f) << 24;
        col |= uint32_t(val.y * 255.0f) << 16;
        col |= uint32_t(val.z * 255.0f) << 8;
        col |= uint32_t(val.w * 255.0f);
        return col;
    }

    inline uint32_t ToPackedARGB(const NVec4<float>& val) {
        uint32_t col = 0;
        col |= uint32_t(val.w * 255.0f) << 24;
        col |= uint32_t(val.x * 255.0f) << 16;
        col |= uint32_t(val.y * 255.0f) << 8;
        col |= uint32_t(val.z * 255.0f);
        return col;
    }

    inline uint32_t ToPackedABGR(const NVec4<float>& val) {
        uint32_t col = 0;
        col |= uint32_t(val.w * 255.0f) << 24;
        col |= uint32_t(val.x * 255.0f);
        col |= uint32_t(val.y * 255.0f) << 8;
        col |= uint32_t(val.z * 255.0f) << 16;
        return col;
    }

    inline uint32_t ToPackedBGRA(const NVec4<float>& val) {
        uint32_t col = 0;
        col |= uint32_t(val.w * 255.0f) << 8;
        col |= uint32_t(val.x * 255.0f) << 16;
        col |= uint32_t(val.y * 255.0f) << 24;
        col |= uint32_t(val.z * 255.0f);
        return col;
    }

    inline float Luminosity(const NVec4<float>& intensity) {
        return (0.2126f * intensity.x + 0.7152f * intensity.y + 0.0722f * intensity.z);
    }

    inline NVec4<float> Mix(const NVec4<float>& c1, const NVec4<float>& c2, float factor) {
        NVec4<float> ret = c1 * (1.0f - factor);
        ret = ret + (c2 * factor);
        return ret;
    }

    inline NVec4<float> HSVToRGB(float h, float s, float v) {
        auto r = 0.0f, g = 0.0f, b = 0.0f;

        if (s == 0) {
            r = v;
            g = v;
            b = v;
        } else {
            int i;
            float f, p, q, t;

            if (h == 360)
                h = 0;
            else
                h = h / 60.0f;

            i = (int)trunc(h);
            f = h - i;

            p = v * (1.0f - s);
            q = v * (1.0f - (s * f));
            t = v * (1.0f - (s * (1.0f - f)));

            switch (i) {
            case 0:
                r = v;
                g = t;
                b = p;
                break;

            case 1:
                r = q;
                g = v;
                b = p;
                break;

            case 2:
                r = p;
                g = v;
                b = t;
                break;

            case 3:
                r = p;
                g = q;
                b = v;
                break;

            case 4:
                r = t;
                g = p;
                b = v;
                break;

            default:
                r = v;
                g = p;
                b = q;
                break;
            }
        }

        return NVec4<float>(r / 255.0f, g / 255.0f, b / 255.0f, 1.0f);
    }

    template <class T>
    inline std::ostream& operator<<(std::ostream& str, const NVec4<T>& region) {
        str << "(" << region.x << ", " << region.y << ", " << region.z << ", " << region.w << ")";
        return str;
    }

    using NVec4f = NVec4<float>;
    using NVec4i = NVec4<long>;

    template <class T>
    struct NRect {
        NRect(const NVec2<T>& topLeft, const NVec2<T>& bottomRight)
            : topLeftPx(topLeft),
              bottomRightPx(bottomRight) {
        }

        NRect(T left, T top, T width, T height)
            : topLeftPx(NVec2<T>(left, top)),
              bottomRightPx(NVec2<T>(left, top) + NVec2<T>(width, height)) {
        }

        NRect() {
        }

        NVec2f topLeftPx;
        NVec2f bottomRightPx;

        bool Contains(const NVec2<T>& pt) const {
            return topLeftPx.x <= pt.x && topLeftPx.y <= pt.y && bottomRightPx.x > pt.x && bottomRightPx.y > pt.y;
        }

        NVec2f BottomLeft() const {
            return NVec2f(topLeftPx.x, bottomRightPx.y);
        }
        NVec2f TopRight() const {
            return NVec2f(bottomRightPx.x, topLeftPx.y);
        }

        float Left() const {
            return topLeftPx.x;
        }
        float Right() const {
            return TopRight().x;
        }
        float Top() const {
            return TopRight().y;
        }
        float Bottom() const {
            return bottomRightPx.y;
        }
        float Height() const {
            return bottomRightPx.y - topLeftPx.y;
        }
        float Width() const {
            return bottomRightPx.x - topLeftPx.x;
        }
        NVec2f Center() const {
            return (bottomRightPx + topLeftPx) * .5f;
        }
        NVec2f Size() const {
            return bottomRightPx - topLeftPx;
        }
        bool Empty() const {
            return (Height() == 0.0f || Width() == 0.0f) ? true : false;
        }
        void Clear() {
            topLeftPx = NRect<T>();
            bottomRightPx = NRect<T>();
        }

        void SetSize(const NVec2f& size) {
            bottomRightPx = topLeftPx + size;
        }

        void SetHeight(const T height) {
            bottomRightPx.y = topLeftPx.y + height;
        }

        void Adjust(float x, float y, float z, float w) {
            topLeftPx.x += x;
            topLeftPx.y += y;
            bottomRightPx.x += z;
            bottomRightPx.y += w;
        }

        void Adjust(float x, float y) {
            topLeftPx.x += x;
            topLeftPx.y += y;
            bottomRightPx.x += x;
            bottomRightPx.y += y;
        }

        void Move(float x, float y) {
            auto width = Width();
            auto height = Height();
            topLeftPx.x = x;
            topLeftPx.y = y;
            bottomRightPx.x = x + width;
            bottomRightPx.y = y + height;
        }

        bool operator==(const NRect<T>& region) const {
            return (topLeftPx == region.topLeftPx) && (bottomRightPx == region.bottomRightPx);
        }
        bool operator!=(const NRect<T>& region) const {
            return !(*this == region);
        }
    };

    template <class T>
    inline NRect<T> operator*(const NRect<T>& lhs, float val) {
        return NRect<T>(lhs.topLeftPx * val, lhs.bottomRightPx * val);
    }
    template <class T>
    inline NRect<T> operator*(const NRect<T>& lhs, const NVec2f& val) {
        return NRect<T>(lhs.topLeftPx * val.x, lhs.bottomRightPx * val.y);
    }
    template <class T>
    inline NRect<T> operator-(const NRect<T>& lhs, const NRect<T>& rhs) {
        return NRect<T>(lhs.topLeftPx.x - rhs.topLeftPx.x, lhs.bottomRightPx.y - rhs.topLeftPx.y);
    }
    template <class T>
    inline std::ostream& operator<<(std::ostream& str, const NRect<T>& region) {
        str << region.topLeftPx << ", " << region.bottomRightPx << ", size: " << region.Width() << ", " << region.Height();
        return str;
    }

    enum FitCriteria {
        X,
        Y
    };

    template <class T>
    inline bool NRectFits(const NRect<T>& area, const NRect<T>& rect, FitCriteria criteria) {
        if (criteria == FitCriteria::X) {
            auto xDiff = rect.bottomRightPx.x - area.bottomRightPx.x;
            if (xDiff > 0) {
                return false;
            }
            xDiff = rect.topLeftPx.x - area.topLeftPx.x;
            if (xDiff < 0) {
                return false;
            }
        } else {
            auto yDiff = rect.bottomRightPx.y - area.bottomRightPx.y;
            if (yDiff > 0) {
                return false;
            }
            yDiff = rect.topLeftPx.y - area.topLeftPx.y;
            if (yDiff < 0) {
                return false;
            }
        }
        return true;
    }

    using NRectf = NRect<float>;

    inline float ZClamp(float x, float lowerlimit, float upperlimit) {
        x = std::max(lowerlimit, x);
        x = std::min(upperlimit, x);
        return x;
    }

    inline float ZSmoothStep(float edge0, float edge1, float x) {
        // Scale, bias and saturate x to 0..1 range
        x = ZClamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
        // Evaluate polynomial
        return x * x * (3 - 2 * x);
    }

} // namespace Zep
