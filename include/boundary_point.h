#ifndef POINTS_H
#define POINTS_H

#include <compare>
#include <cstdint>
#include <tuple>

enum class HalfPixel : uint16_t {
    TOP_LEFT = 0,
    TOP = 1,
    LEFT = 2,
    BOTTOM_LEFT = 3,
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
    uint16_t x;
    // x is packed as [COLOR_DIRECTION : 1, unused : 1, VALUE : 14]
    uint16_t y;

    uint16_t first_blob;
    uint16_t second_blob;

    uint32_t blob_label() const { return (first_blob << 16) | second_blob; }
    bool is_black_to_white() const {
        return y & COORDINATE_COLOR_DIRECTION_MASK;
    }

    uint16_t x_value() const { return x & COORDINATE_VALUE_MASK; }
    uint16_t y_value() const { return y & COORDINATE_VALUE_MASK; }

    uint16_t x_with_dx() const { return 2 * x_value() + dx(); }
    uint16_t y_with_dy() const { return 2 * y_value() + dy(); }

    float calc_theta(float cx, float cy) const {
        float quadrants[2][2] = {{-1 * (1 << 16), 1 << 17}, {0, 1 << 16}};

        float dx = static_cast<float>(x_with_dx()) - cx;
        float dy = static_cast<float>(y_with_dy()) - cy;

        float quadrant = quadrants[dx > 0][dy > 0];
        if ((dx > 0) == (dy > 0)) {
            return quadrant + static_cast<float>(y) / static_cast<float>(x);
        } else {
            return quadrant - static_cast<float>(x) / static_cast<float>(y);
        }
    }

    int16_t dx() const {
        switch (extract_half_pixel(x)) {
        case HalfPixel::TOP_LEFT:
            return -1;
        case HalfPixel::TOP:
            return 0;
        case HalfPixel::LEFT:
            return -1;
        case HalfPixel::BOTTOM_LEFT:
            return 0;
        default:
            return 0;
        }
    }

    int16_t dy() const {
        switch (extract_half_pixel(x)) {
        case HalfPixel::TOP_LEFT:
            return -1;
        case HalfPixel::TOP:
            return -1;
        case HalfPixel::LEFT:
            return 0;
        case HalfPixel::BOTTOM_LEFT:
            return 1;
        default:
            return 0;
        }
    }

    int16_t gx() const { return is_black_to_white() ? dx() : -dx(); }

    int16_t gy() const { return is_black_to_white() ? dy() : -dy(); }

    bool operator==(const BoundaryPoint &) const = default;
    // auto operator<=>(const BoundaryPoint &other) const {
    //     auto left =
    //         std::make_tuple(first_blob, second_blob, x &
    //         COORDINATE_VALUE_MASK,
    //                         y & COORDINATE_VALUE_MASK, extract_half_pixel(x),
    //                         y & COORDINATE_COLOR_DIRECTION_MASK);
    //     auto right = std::make_tuple(other.first_blob, other.second_blob,
    //                                  other.x & COORDINATE_VALUE_MASK,
    //                                  other.y & COORDINATE_VALUE_MASK,
    //                                  extract_half_pixel(other.x),
    //                                  other.y &
    //                                  COORDINATE_COLOR_DIRECTION_MASK);
    //     return left <=> right;
    // }
};

// The struct has both 0 and UINT64_MAX as invalid states,
// depending on which one fits better.
struct ClusterPoint {
    // x is packed as [HALF_PIXEL : 2, VALUE : 14]
    uint16_t x;
    // x is packed as [COLOR_DIRECTION : 1, unused : 1, VALUE : 14]
    uint16_t y;
    float slope;

    bool is_black_to_white() const {
        return y & COORDINATE_COLOR_DIRECTION_MASK;
    }

    uint16_t x_value() const { return x & COORDINATE_VALUE_MASK; }
    uint16_t y_value() const { return y & COORDINATE_VALUE_MASK; }

    uint16_t x_with_dx() const { return 2 * x_value() + dx(); }
    uint16_t y_with_dy() const { return 2 * y_value() + dy(); }

    int16_t dx() const {
        switch (extract_half_pixel(x)) {
        case HalfPixel::TOP_LEFT:
            return -1;
        case HalfPixel::TOP:
            return 0;
        case HalfPixel::LEFT:
            return -1;
        case HalfPixel::BOTTOM_LEFT:
            return 0;
        default:
            return 0;
        }
    }

    int16_t dy() const {
        switch (extract_half_pixel(x)) {
        case HalfPixel::TOP_LEFT:
            return -1;
        case HalfPixel::TOP:
            return -1;
        case HalfPixel::LEFT:
            return 0;
        case HalfPixel::BOTTOM_LEFT:
            return 1;
        default:
            return 0;
        }
    }

    int16_t gx() const { return is_black_to_white() ? dx() : -dx(); }

    int16_t gy() const { return is_black_to_white() ? dy() : -dy(); }
};
#endif
