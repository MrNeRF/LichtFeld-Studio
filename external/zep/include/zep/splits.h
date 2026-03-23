#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "zep/mcommon/math/math.h"

namespace Zep {

    namespace RegionFlags {
        enum Flags {
            Fixed = (1 << 0),
            Expanding = (1 << 1),
            AlignCenter = (1 << 2)
        };
    };

    enum class RegionLayoutType {
        VBox,
        HBox
    };

    struct Region {
        RegionLayoutType layoutType = RegionLayoutType::VBox;

        uint32_t flags = RegionFlags::Expanding;
        NRectf rect;
        NVec2f fixed_size = NVec2f(0.0f, 0.0f);
        NVec2f padding = NVec2f(0.0f, 0.0f);
        NVec4f margin = NVec4f(0.0f, 0.0f, 0.0f, 0.0f);
        std::string debugName;

        Region* pParent = nullptr;
        std::vector<std::shared_ptr<Region>> children;
    };

    inline std::ostream& operator<<(std::ostream& str, const Region& region) {
        static int indent = 0;
        auto do_indent = [&str](int sz) { for (int i = 0; i < sz; i++) str << " "; };

        do_indent(indent);
        str << std::hex << &region << "(" << region.debugName << ") -> ";

        str << "RC: " << region.rect << ", pParent: " << std::hex << region.pParent;
        if (!region.children.empty()) {
            str << std::endl;
            for (auto& child : region.children) {
                indent++;
                str << *child;
                indent--;
            }
        }
        str << std::endl;

        return str;
    }

    void LayoutRegion(Region& region);

} // namespace Zep
