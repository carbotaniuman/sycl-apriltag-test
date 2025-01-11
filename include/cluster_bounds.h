#ifndef CLUSTER_ANALYSIS_H
#define CLUSTER_ANALYSIS_H

#include <algorithm>
#include <cstdint>

#include "boundary_point.h"

struct ClusterExtents {
    uint32_t start;
    uint32_t count;
};

struct ClusterBounds {
    uint16_t x_min;
    uint16_t x_max;
    uint16_t y_min;
    uint16_t y_max;

    int32_t gx_sum;
    int32_t gy_sum;

    int64_t pxgx_pygy_sum;

    uint32_t start;
    uint32_t count;

    static ClusterBounds inital_from_point(BoundaryPoint p, uint32_t start) {
        return ClusterBounds{
            p.x_with_dx(),
            p.x_with_dx(),
            p.y_with_dy(),
            p.y_with_dy(),

            p.gx(),
            p.gy(),

            static_cast<int64_t>(p.x_with_dx()) * static_cast<int64_t>(p.gx()) +
                static_cast<int64_t>(p.y_with_dy()) *
                    static_cast<int64_t>(p.gy()),

            start,
            1,
        };
    }

    float cx() const { return (x_min + x_max) * 0.5f + 0.05118; }

    float cy() const { return (y_min + y_max) * 0.5f + -0.028581; }

    float dot() const {
        // dx = x - cx
        // dy = y - cy
        // dot = sum(dx * gx + dy * gy)
        //
        // dot = sum((x - cx) * gx + (y - cy) * gy)

        float cx_cy_contrib =
            ((x_min + x_max) * gx_sum + (y_min + y_max) * gy_sum) * 0.5;

        float gx_contrib = 0.05118 * static_cast<float>(gx_sum);
        float gy_contrib = -0.028581 * static_cast<float>(gy_sum);

        return 2 * static_cast<float>(pxgx_pygy_sum) - cx_cy_contrib -
               gx_contrib - gy_contrib;
    }
};

inline ClusterBounds reduce_bounds(const ClusterBounds &a,
                                   const ClusterBounds &b) {
    ClusterBounds out;
    out.x_min = std::min(a.x_min, b.x_min);
    out.x_max = std::max(a.x_max, b.x_max);
    out.y_min = std::min(a.y_min, b.y_min);
    out.y_max = std::max(a.y_max, b.y_max);
    out.start = std::min(a.start, b.start);
    out.count = a.count + b.count;
    out.pxgx_pygy_sum = a.pxgx_pygy_sum + b.pxgx_pygy_sum;
    out.gx_sum = a.gx_sum + b.gx_sum;
    out.gy_sum = a.gy_sum + b.gy_sum;

    return out;
}

struct ValidBlobFilter {
    size_t tag_width = 4;
    bool normal_border = true;
    bool reversed_border = false;
    size_t min_cluster_pixels = 5;
    size_t max_cluster_pixels;

    ValidBlobFilter(size_t width, size_t height)
        : max_cluster_pixels(2 * (2 * width + 2 * height)) {}

    bool operator()(const ClusterBounds &b) const {
        if (b.count < min_cluster_pixels) {
            return false;
        }
        if (b.count < 24) {
            return false;
        }
        if (b.count > max_cluster_pixels) {
            return false;
        }

        if ((b.x_max - b.x_min) * (b.y_max - b.y_min) < tag_width) {
            return false;
        }

        bool quad_reversed_border = b.dot() < 0.0;
        if (!reversed_border && quad_reversed_border) {
            return false;
        }
        if (!normal_border && !quad_reversed_border) {
            return false;
        }

        return true;
    }
};
#endif
