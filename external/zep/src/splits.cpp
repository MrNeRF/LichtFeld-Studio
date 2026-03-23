#include "zep/splits.h"

namespace Zep {

    // Given a region layout, return the size that we want to use for the stack inside it if fixed.
    float SizeForLayout(RegionLayoutType layout, const NVec2f& v) {
        if (layout == RegionLayoutType::HBox) {
            return v.x;
        } else {
            return v.y;
        }
    }

    void LayoutRegion(Region& region) {
        float totalFixedSize = 0.0f;
        float expanders = 0.0f;
        float padding = 0.0f;
        for (auto& r : region.children) {
            if (r->flags & RegionFlags::Fixed) {
                // This region needs its size plus margin
                totalFixedSize += SizeForLayout(region.layoutType, r->fixed_size);
            } else {
                expanders += 1.0f;
            }
            padding += r->padding.x + r->padding.y;
        }

        NRectf currentRect = region.rect;
        currentRect.Adjust(region.margin.x, region.margin.y, -region.margin.z, -region.margin.w);

        auto remaining = ((region.layoutType == RegionLayoutType::HBox) ? currentRect.Width() : currentRect.Height()) - (totalFixedSize + padding);
        auto perExpanding = remaining / expanders;

        perExpanding = std::max(0.0f, perExpanding);

        for (auto& rcChild : region.children) {
            rcChild->rect = currentRect;

            if (region.layoutType == RegionLayoutType::VBox) {
                rcChild->rect.topLeftPx.y += rcChild->padding.x;
                if (rcChild->flags & RegionFlags::Fixed) {
                    rcChild->rect.bottomRightPx.y = rcChild->rect.Top() + SizeForLayout(region.layoutType, rcChild->fixed_size);
                    currentRect.topLeftPx.y = rcChild->rect.Bottom() + rcChild->padding.y;
                } else {
                    rcChild->rect.bottomRightPx.y = rcChild->rect.Top() + perExpanding;
                    currentRect.topLeftPx.y = rcChild->rect.Bottom() + rcChild->padding.y;
                }
            } else if (region.layoutType == RegionLayoutType::HBox) {
                rcChild->rect.topLeftPx.x += rcChild->padding.x;
                if (rcChild->flags & RegionFlags::Fixed) {
                    rcChild->rect.bottomRightPx.x = rcChild->rect.Left() + SizeForLayout(region.layoutType, rcChild->fixed_size);
                    currentRect.topLeftPx.x = rcChild->rect.Right() + rcChild->padding.y;
                } else {
                    rcChild->rect.bottomRightPx.x = rcChild->rect.Left() + perExpanding;
                    currentRect.topLeftPx.x = rcChild->rect.Right() + rcChild->padding.y;
                }
            }

            if (!(rcChild->flags & RegionFlags::Expanding) && rcChild->fixed_size.x != 0.0f && rcChild->fixed_size.y != 0.0f) {
                auto currentSize = rcChild->rect.Size();

                rcChild->rect.SetSize(Clamp(rcChild->fixed_size, NVec2f(0.0f, 0.0f), currentSize));

                if (region.layoutType == RegionLayoutType::HBox) {
                    currentRect.topLeftPx = rcChild->rect.TopRight();
                } else {
                    currentRect.topLeftPx = rcChild->rect.BottomLeft();
                }

                if (rcChild->flags & RegionFlags::AlignCenter) {
                    auto diff = currentSize - rcChild->rect.Size();
                    diff.x = std::max(0.0f, diff.x);
                    diff.y = std::max(0.0f, diff.y);
                    rcChild->rect.Adjust(diff.x / 2.0f, diff.y / 2.0f);
                }
            }
        }

        for (auto& r : region.children) {
            LayoutRegion(*r);
        }
    }

} // namespace Zep