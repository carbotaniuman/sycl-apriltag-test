#ifndef FIT_QUAD_H
#define FIT_QUAD_H

#include <sycl/sycl.hpp>

#include "cluster_bounds.h"
#include "line_fit_point.h"

struct Peak {
    double error;
    uint32_t line_fit_point_index;
    uint16_t cluster_index;
};

void fit_lines_test2(sycl::queue &q, const LineFitPoint *points,
const uint16_t *cluster_indices, const ClusterExtents *cluster_extents, size_t points_size, Peak *found_peaks);

#endif
