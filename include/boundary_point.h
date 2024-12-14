#ifndef POINTS_H
#define POINTS_H

#include <compare>
#include <cstdint>
#include <tuple>

enum class HalfPixel : uint16_t {
    RIGHT = 0,
    BOTTOM_LEFT = 1,
    BOTTOM = 2,
    BOTTOM_RIGHT = 3,
};

// Mask to get the coordinate half-pixel.
constexpr static uint16_t COORDINATE_HALF_PIXEL_MASK = (1U << 15) & (1U << 14);
// Mask to get the color direction (white-to-black (0) or black-to-white (1)).
constexpr static uint16_t COORDINATE_COLOR_DIRECTION_MASK = 1U << 15;
// Mask to get the coordinate value.
constexpr static uint16_t COORDINATE_VALUE_MASK = (1U << 14) - 1;

inline HalfPixel extract_half_pixel(uint16_t coord) {
    return static_cast<HalfPixel>((coord & COORDINATE_HALF_PIXEL_MASK) >> 14);
}

inline uint16_t pack_half_pixel(uint16_t coord, HalfPixel pixel) {
    return coord | (static_cast<uint16_t>(pixel) << 14);
}

// The struct has both 0 and UINT64_MAX as invalid states,
// depending on which one fits better.
struct BoundaryPoint {
    // x is packed as [HALF_PIXEL : 2, VALUE : 14]
    uint16_t packed_x;
    // x is packed as [COLOR_DIRECTION : 1, unused : 1, VALUE : 14]
    uint16_t packed_y;

    bool is_black_to_white() const {
        return packed_y & COORDINATE_COLOR_DIRECTION_MASK;
    }

    uint16_t x_value() const { return packed_x & COORDINATE_VALUE_MASK; }
    uint16_t y_value() const { return packed_y & COORDINATE_VALUE_MASK; }

    uint16_t x_with_dx() const { return 2 * x_value() + dx(); }
    uint16_t y_with_dy() const { return 2 * y_value() + dy(); }

    float calc_theta(float cx, float cy) const {
        float quadrants[2][2] = {{-1 * (1 << 16), 1 << 17}, {0, 1 << 16}};

        float dx = static_cast<float>(x_with_dx()) - cx;
        float dy = static_cast<float>(y_with_dy()) - cy;

        float quadrant = quadrants[dx > 0][dy > 0];

        if ((dx > 0) == (dy > 0)) {
            return quadrant + static_cast<float>(dy) / static_cast<float>(dx);
        } else {
            return quadrant - static_cast<float>(dx) / static_cast<float>(dy);
        }
    }

    int16_t dx() const {
        switch (extract_half_pixel(packed_x)) {
        case HalfPixel::RIGHT:
            return 1;
        case HalfPixel::BOTTOM_LEFT:
            return -1;
        case HalfPixel::BOTTOM:
            return 0;
        case HalfPixel::BOTTOM_RIGHT:
            return 1;
        default:
            return 0;
        }
    }

    int16_t dy() const {
        switch (extract_half_pixel(packed_x)) {
        case HalfPixel::RIGHT:
            return 0;
        case HalfPixel::BOTTOM_LEFT:
            return 1;
        case HalfPixel::BOTTOM:
            return 1;
        case HalfPixel::BOTTOM_RIGHT:
            return 1;
        default:
            return 0;
        }
    }

    int16_t gx() const { return is_black_to_white() ? dx() : -dx(); }

    int16_t gy() const { return is_black_to_white() ? dy() : -dy(); }
};

// The struct has both 0 and UINT64_MAX as invalid states,
// depending on which one fits better.
struct ClusterPoint {
    // x is packed as [HALF_PIXEL : 2, VALUE : 14]
    uint16_t packed_x;
    // x is packed as [COLOR_DIRECTION : 1, unused : 1, VALUE : 14]
    uint16_t packed_y;
    float slope;

    bool is_black_to_white() const {
        return packed_y & COORDINATE_COLOR_DIRECTION_MASK;
    }

    uint16_t x_value() const { return packed_x & COORDINATE_VALUE_MASK; }
    uint16_t y_value() const { return packed_y & COORDINATE_VALUE_MASK; }

    uint16_t x_with_dx() const { return 2 * x_value() + dx(); }
    uint16_t y_with_dy() const { return 2 * y_value() + dy(); }

    int16_t dx() const {
        switch (extract_half_pixel(packed_x)) {
        case HalfPixel::RIGHT:
            return 1;
        case HalfPixel::BOTTOM_LEFT:
            return -1;
        case HalfPixel::BOTTOM:
            return 0;
        case HalfPixel::BOTTOM_RIGHT:
            return 1;
        default:
            return 0;
        }
    }

    int16_t dy() const {
        switch (extract_half_pixel(packed_x)) {
        case HalfPixel::RIGHT:
            return 0;
        case HalfPixel::BOTTOM_LEFT:
            return 1;
        case HalfPixel::BOTTOM:
            return 1;
        case HalfPixel::BOTTOM_RIGHT:
            return 1;
        default:
            return 0;
        }
    }

    int16_t gx() const { return is_black_to_white() ? dx() : -dx(); }
    int16_t gy() const { return is_black_to_white() ? dy() : -dy(); }
};
#endif
