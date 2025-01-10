#ifndef LINE_FIT_POINT_H
#define LINE_FIT_POINT_H

#include "boundary_point.h"

#include <cmath>

struct LineFitPoint {
    double Mx, My;
    double Mxx, Myy, Mxy;
    double W;
    double a, b;

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

    double rx = p.x_with_dx() * 0.5 + 0.5;
    double ry = p.y_with_dy() * 0.5 + 0.5;
    uint16_t x = static_cast<uint16_t>(rx);
    uint16_t y = static_cast<uint16_t>(ry);

    double W = 1;
    if (x > 0 && x + 1 < width && y > 0 && y + 1 < height) {
        int grad_x =
            greyscaled[y * width + x + 1] - greyscaled[y * width + x - 1];
        int grad_y =
            greyscaled[(y + 1) * width + x] - greyscaled[(y - 1) * width + x];
        W = std::hypot(static_cast<double>(grad_x),
                       static_cast<double>(grad_y)) +
            1;
    }

    return LineFitPoint{W * rx,      W * ry,      W * rx * rx,
                        W * ry * ry, W * rx * ry, W};
}

#endif
