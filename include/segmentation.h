#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include "open_chaining.h"
#include <sycl/sycl.hpp>

// Mask to get the pixel color (whether 0 or 255).
constexpr static uint16_t LABEL_PIXEL_MASK = 1U << 15;
// Mask to get the label value.
constexpr static uint16_t LABEL_VALUE_MASK = ~LABEL_PIXEL_MASK;

sycl::event image_segmentation(sycl::queue &q, const uint8_t *thresholded,
                               uint32_t *label_scratch, uint16_t *labels, HashTable::Entry *sizes,
                               size_t sizes_elem, size_t width, size_t height,
                               const std::vector<sycl::event> &deps = {});

#endif
