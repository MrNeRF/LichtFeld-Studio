/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <glm/glm.hpp>

#include "geometry/euclidean_transform.hpp"

namespace lfs {
    namespace geometry {
        class BoundingBox {
        public:
            BoundingBox();
            virtual ~BoundingBox();

            // Set the bounding box from min/max points
            virtual void setBounds(const glm::vec3& min, const glm::vec3& max);

            // Set custom transform matrix for the bounding box
            void setworld2BBox(const geometry::EuclideanTransform& transform);
            void setworld2BBox(const glm::mat4& transform);
            // getter
            const geometry::EuclideanTransform& getworld2BBox() const { return world2BBox_; }
            const glm::mat4& getworld2BBoxMat4() const { return world2BBox_mat4_; }
            bool hasFullTransform() const { return has_full_transform_; }

            // Get current bounds
            glm::vec3 getMinBounds() const { return min_bounds_; }
            glm::vec3 getMaxBounds() const { return max_bounds_; }

        protected:
            // Bounding box properties
            glm::vec3 min_bounds_;
            glm::vec3 max_bounds_;
            // relative position of bounding box to the world
            EuclideanTransform world2BBox_;
            glm::mat4 world2BBox_mat4_{1.0f};
            bool has_full_transform_ = false;
        };
    } // namespace geometry
} // namespace lfs
