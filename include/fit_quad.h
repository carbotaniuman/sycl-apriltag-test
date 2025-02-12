#ifndef FIT_QUAD_H
#define FIT_QUAD_H

#include <sycl/sycl.hpp>

#include "cluster_bounds.h"
#include "line_fit_point.h"

struct PeakExtents {
    uint32_t start;
    uint32_t count;
};

inline PeakExtents reduce_extents(const PeakExtents &a, const PeakExtents &b) {
    PeakExtents out;
    out.start = std::min(a.start, b.start);
    out.count = a.count + b.count;
    return out;
}

struct Corner {
    uint32_t line_fit_point_index;
    uint16_t cluster_index;
    float error;
};

struct FittedQuad {
    std::array<LineFitPoint, 4> moments;
    std::array<uint16_t, 4> num_in_moments;
    std::array<uint16_t, 4> indices;
};

struct QuadCorners {
  float corners[4][2];
  bool reversed_border;
  uint32_t blob_index;
};

std::tuple<LineFitPoint, size_t> get_moment(const LineFitPoint *points,
                                            size_t points_size, size_t i0,
                                            size_t i1);

void fit_line(LineFitPoint moment, size_t num_in_moment, float *line_params,
              float *err, float *mse);

void fit_lines(sycl::queue &q, const LineFitPoint *points,
               const uint16_t *cluster_indices,
               const ClusterExtents *cluster_extents, size_t points_size,
               Corner *found_corners,
               const std::vector<sycl::event> &deps = {});

void do_indexing(sycl::queue &q, const PeakExtents *extents,
                 size_t extents_count, const Corner *compacted_corners,
                 const LineFitPoint *points,
                 const ClusterExtents *cluster_extents, FittedQuad *fitted);

#endif
