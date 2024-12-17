#ifndef LINE_FIT_POINT_H
#define LINE_FIT_POINT_H

#include "boundary_point.h"

#include <cmath>

struct LineFitPoint {
    double Mx, My;
    double Mxx, Myy, Mxy;
    double W;

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
};

inline LineFitPoint compute_initial_linefit(ClusterPoint p, uint32_t width,
                                            uint32_t height,
                                            uint8_t *greyscaled) {
    uint16_t x = (p.x_with_dx() + 1) / 2;
    uint16_t y = (p.y_with_dy() + 1) / 2;

    double W = 1;
    if (x > 0 && x + 1 < width && y > 0 && y < height) {
        int grad_x =
            greyscaled[y * width + x + 1] - greyscaled[y * width + x - 1];
        int grad_y =
            greyscaled[(y + 1) * width + x] - greyscaled[(y - 1) * width + x];
        W = std::hypotf(static_cast<double>(grad_x),
                        static_cast<double>(grad_y));
    }

    double fx = static_cast<double>(x);
    double fy = static_cast<double>(y);

    return LineFitPoint(W * fx, W * fy, W * fx * fx, W * fy * fy, W * fx * fy,
                        W);
}

#endif
