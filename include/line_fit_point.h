#ifndef LINE_FIT_POINT_H
#define LINE_FIT_POINT_H

#include "boundary_point.h"

#include <cmath>

// This struct is similar to how AprilTag treats this
// but we have made a few changes to convert them to
// integers.
struct LineFitPoint {
    // 2x real value
    int64_t Mx, My;
    // 4x real value
    int64_t Mxx, Myy, Mxy;
    int64_t W;
    int64_t a, b;

    LineFitPoint &operator+=(const LineFitPoint &rhs) {
        Mx += rhs.Mx;
        My += rhs.My;
        Mxx += rhs.Mxx;
        Myy += rhs.Myy;
        Mxy += rhs.Mxy;
        W += rhs.W;
        return *this;
    }

    friend LineFitPoint operator+(LineFitPoint lhs, const LineFitPoint &rhs) {
        lhs += rhs;
        return lhs;
    }

    LineFitPoint &operator-=(const LineFitPoint &rhs) {
        Mx -= rhs.Mx;
        My -= rhs.My;
        Mxx -= rhs.Mxx;
        Myy -= rhs.Myy;
        Mxy -= rhs.Mxy;
        W -= rhs.W;
        return *this;
    }

    friend LineFitPoint operator-(LineFitPoint lhs, const LineFitPoint &rhs) {
        lhs -= rhs;
        return lhs;
    }
};

inline LineFitPoint compute_initial_linefit(ClusterPoint p, uint32_t width,
                                            uint32_t height,
                                            uint8_t *greyscaled) {
    // These numbers are double what they normally are.
    // We have the range to support this, and the 2x
    // will get divided out anyways when fitting lines.
    auto rx = static_cast<int64_t>(p.x_with_dx()) + 1;
    auto ry = static_cast<int64_t>(p.y_with_dy()) + 1;
    uint32_t x = rx / 2;
    uint32_t y = rx / 2;
    
    int64_t W = 1;
    if (x > 0 && x + 1 < width && y > 0 && y + 1 < height) {
        int grad_x =
            greyscaled[y * width + x + 1] - greyscaled[y * width + x - 1];
        int grad_y =
            greyscaled[(y + 1) * width + x] - greyscaled[(y - 1) * width + x];

        // This used to be done on doubles, but the maximum error
        // between them is 0.03162745, which is dwarved by the error
        // introduced by discarding the fractional portion.
        W += static_cast<int64_t>(
            std::hypotf(grad_x, grad_y)
        );
    }

    // We lose the fractional part of the hypot here, which should
    // be fine as it's a relatively small part of the value.
    return LineFitPoint{
        W * rx,
        W * ry,
        W * rx * rx,
        W * ry * ry,
        W * rx * ry,
        W
    };
}

#endif
