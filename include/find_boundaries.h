#ifndef BOUNDARIES_H
#define BOUNDARIES_H

#include "boundary_point.h"

#include <sycl/sycl.hpp>

sycl::event find_boundaries(sycl::queue &q, const uint32_t *labels,
                            const uint32_t *sizes, BoundaryPoint *points,
                            uint32_t *blob_labels, uint64_t *compacter_buffer,
                            size_t width, size_t height,
                            const std::vector<sycl::event> &deps = {});

#endif
