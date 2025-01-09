#ifndef BOUNDARIES_H
#define BOUNDARIES_H

#include "boundary_point.h"
#include "open_chaining.h"

#include <sycl/sycl.hpp>

sycl::event find_boundaries(sycl::queue &q, const uint16_t *labels,
                            const HashTable::Entry *sizes, size_t sizes_elems,
                            BoundaryPoint *points, size_t width, size_t height,
                            const std::vector<sycl::event> &deps = {});

#endif
