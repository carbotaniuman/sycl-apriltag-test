#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <sycl/sycl.hpp>

sycl::event threshold_image(sycl::queue &q, const uint8_t *grayscale,
                            sycl::vec<uint8_t, 2> *extrema,
                            sycl::vec<uint8_t, 2> *filtered_extrema,
                            uint8_t *thresholded, size_t width, size_t height,
                            const std::vector<sycl::event> &deps = {});

#endif
