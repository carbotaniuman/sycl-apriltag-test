#include "fit_quad.h"
#include <cmath>
#include <tuple>
#include <type_traits>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

// Computed via AprilTag stuff and then dumped here.
constexpr size_t POINTS_PER_END = 3;
constexpr std::array<float, 7> FILTER_DATA = {
    0x1.6c0504p-7, 0x1.152aaap-3, 0x1.368b3p-1, 0x1p+0,
    0x1.368b3p-1,  0x1.152aaap-3, 0x1.6c0504p-7};

std::tuple<LineFitPoint, size_t> get_moment(const LineFitPoint *points,
                                            size_t points_size, size_t i0,
                                            size_t i1) {
    LineFitPoint result;
    size_t total_elems;

    if (i0 < i1) {
        total_elems = i1 - i0 + 1;

        const LineFitPoint &right = points[i1];

        result.Mx = right.Mx;
        result.My = right.My;
        result.Mxx = right.Mxx;
        result.Mxy = right.Mxy;
        result.Myy = right.Myy;
        result.W = right.W;

        if (i0 > 0) {
            LineFitPoint left = points[i0 - 1];

            result.Mx -= left.Mx;
            result.My -= left.My;
            result.Mxx -= left.Mxx;
            result.Mxy -= left.Mxy;
            result.Myy -= left.Myy;
            result.W -= left.W;
        }
    } else {
        total_elems = points_size - i0 + i1 + 1;

        const LineFitPoint &left = points[i0 - 1];
        const LineFitPoint &end_range = points[points_size - 1];
        const LineFitPoint &right = points[i1];

        result.Mx = end_range.Mx - left.Mx + right.Mx;
        result.My = end_range.My - left.My + right.My;
        result.Mxx = end_range.Mxx - left.Mxx + right.Mxx;
        result.Mxy = end_range.Mxy - left.Mxy + right.Mxy;
        result.Myy = end_range.Myy - left.Myy + right.Myy;
        result.W = end_range.W - left.W + right.W;
    }
    return std::make_tuple(result, total_elems);
}

void fit_line(LineFitPoint moment, size_t num_in_moment, double *line_params,
              double *err, double *mse) {
    double Ex = moment.Mx / moment.W;
    double Ey = moment.My / moment.W;
    double Cxx = moment.Mxx / moment.W - Ex * Ex;
    double Cxy = moment.Mxy / moment.W - Ex * Ey;
    double Cyy = moment.Myy / moment.W - Ey * Ey;

    double dist = std::hypot((Cxx - Cyy), 2 * Cxy * Cxy);
    double eig_err = 0.5 * (Cxx + Cyy - dist);

    if (line_params) {
        line_params[0] = Ex;
        line_params[1] = Ey;

        double eig = 0.5 * (Cxx + Cyy + dist);
        double nx1 = Cxx - eig;
        double ny1 = Cxy;
        double M1 = nx1 * nx1 + ny1 * ny1;
        double nx2 = Cxy;
        double ny2 = Cyy - eig;
        double M2 = nx2 * nx2 + ny2 * ny2;

        double nx, ny, M;
        if (M1 > M2) {
            nx = nx1;
            ny = ny1;
            M = M1;
        } else {
            nx = nx2;
            ny = ny2;
            M = M2;
        }

        double length = std::sqrt(M);
        if (fabs(length) < 1e-12) {
            line_params[2] = 0;
            line_params[3] = 0;
        } else {
            line_params[2] = nx / length;
            line_params[3] = ny / length;
        }
    }

    if (err) {
        *err = num_in_moment * eig_err;
    }

    if (mse) {
        *mse = eig_err;
    }
}

// This kernel here fits lines to a group of points, and finds the points most
// likely to be corners. It does that by trying, for each point, to fit a line
// from some points before to some points after, and taking the points with the
// the highest error.
//
// To start with, we process an entire block of line-fit points at a time. Let
// us deal with the case where all of the points are in the same cluster.
// AprilTag performs a low-pass filter via convolution over the points, and then
// finds points which have a higher error than both of their neighbors to
// consider corners.
//
// While these two operations may make sense to occur in separate kernels, and
// indeed was done so in the past, the nontrivial nature of dealing with
// multiple clusters meant that a single kernel was deemed simpler.
//
// In order to avoid copious amount of global memory traffic, local memory is
// used to store the calculated errors from line fitting, as well as the error
// values after the low-pass filter for corner finding.
//
// Let us breifly illuminate the structure of the windows in this kernel. The
// X's represent threads that only load data into shared memory, the _
// represents a thread that calculates an error, and the (hexadecimal) numbers
// represent threads that do everything prior along with corner detection.
//
// X X X _ 1 2 3 4 _ X X X
//         X X X _ 5 6 7 8 _ X X X
//                 X X X _ 9 A B C _ X X X
//
// Now let us deal with the case where the points are not all from the same
// cluster. This is incredibly complicated, and relies on 2 sets of points to
// track the differences. Because of the size of the smallest cluster (24)
// compared to the see of the windows we care about during filtering and corner
// finding (3 and 1 respectively), we know that each pixel must border 0 or 1
// pixels in another cluster.
//
// For each pixel, we will calculate the initial unfiltered errors twice - one
// for the cluster we are part of, and once for the cluster `END_OFFSET` before
// or after us (only one of which exists). If we do not find a cluster
// `END_OFFSET` before or after us, we fill the result with NaN to aid
// debugging.
//
// For the filtering, we check if we are on a transition boundary between
// clusters (that is, the next pixel before or after us is of a different
// cluster). If so, in order to accomadate corner finding, we must calculate the
// filtered value as-if we were a part of that other cluster and store that in
// another bit of local storage.
void fit_lines(sycl::queue &q, const LineFitPoint *points,
               const uint16_t *cluster_indices,
               const ClusterExtents *cluster_extents, size_t points_size,
               Corner *found_corners, double *test) {
    constexpr size_t TARGETED_WG_SIZE = 32;
    constexpr size_t END_OFFSET = POINTS_PER_END + 1;

    std::cout << "points_size = " << points_size << std::endl;

    size_t count_needed = (points_size + 23) / 24 * TARGETED_WG_SIZE;
    q.submit([=](sycl::handler &h) {
         sycl::local_accessor<uint16_t> local_cluster_indices{
             sycl::range(TARGETED_WG_SIZE), h};
         sycl::local_accessor<ClusterExtents> local_cluster_extents{
             sycl::range(TARGETED_WG_SIZE), h};

         sycl::local_accessor<double> calculated_errors_same_cluster{
             sycl::range(TARGETED_WG_SIZE), h};
         sycl::local_accessor<double> calculated_errors_diff_cluster{
             sycl::range(TARGETED_WG_SIZE), h};
         sycl::local_accessor<double> filtered_errors_same_cluster{
             sycl::range(TARGETED_WG_SIZE), h};
         sycl::local_accessor<double> filtered_errors_diff_cluster{
             sycl::range(TARGETED_WG_SIZE), h};

         h.parallel_for(
             sycl::nd_range(sycl::range(count_needed),
                            sycl::range(TARGETED_WG_SIZE)),
             [=](sycl::nd_item<1> it) {
                 size_t local_linear_id = it.get_local_linear_id();
                 std::make_signed_t<size_t> x =
                     it.get_group(0) *
                         (it.get_local_range(0) - 2 * END_OFFSET) +
                     it.get_local_id(0) - POINTS_PER_END;

                 auto local_cluster_indices_ptr =
                     local_cluster_indices.get_pointer();
                 auto local_cluster_extents_ptr =
                     local_cluster_extents.get_pointer();

                 // Load the indices into local memory.
                 uint16_t cur_cluster_index = cluster_indices[sycl::clamp(
                     x, static_cast<decltype(x)>(0),
                     static_cast<decltype(x)>(points_size - 1))];
                 local_cluster_indices_ptr[local_linear_id] = cur_cluster_index;
                 ClusterExtents cur_extent = cluster_extents[cur_cluster_index];
                 local_cluster_extents_ptr[local_linear_id] = cur_extent;

                 // Barrier here to sync the indices.
                 it.barrier(sycl::access::fence_space::local_space);

                 // Calculate the error for our current index for our cluster.
                 auto calculated_errors_same_cluster_ptr =
                     calculated_errors_same_cluster.get_pointer();
                 {
                     size_t k_size =
                         sycl::min(static_cast<size_t>(20),
                                   static_cast<size_t>(cur_extent.count / 12));
                     size_t i0 =
                         (cur_extent.count + x - cur_extent.start - k_size) %
                         cur_extent.count;
                     size_t i1 =
                         (cur_extent.count + x - cur_extent.start + k_size) %
                         cur_extent.count;

                     // This feels optimizable
                     auto [moment, num_in_moment] = get_moment(
                         points + cur_extent.start, cur_extent.count, i0, i1);
                     fit_line(
                         moment, num_in_moment, nullptr,
                         &calculated_errors_same_cluster_ptr[local_linear_id],
                         nullptr);
                 }

                 // Because of the min cluster size of 24, we can (maybe) have a
                 // different cluster to the left of us or to the right of us
                 // (by 3), but not both.
                 auto calculated_errors_diff_cluster_ptr =
                     calculated_errors_diff_cluster.get_pointer();

                 // Maybe calculate an error for another cluster if that cluster
                 // would need our data (within `END_OFFSET` of another
                 // cluster).
                 {
                     size_t their_linear_id = local_linear_id;

                     // Check before us.
                     if (local_linear_id > END_OFFSET) {
                         // Only if the indices are not the same
                         if (local_cluster_indices_ptr[local_linear_id -
                                                       END_OFFSET] !=
                             cur_cluster_index) {
                             their_linear_id = local_linear_id - END_OFFSET;
                         }
                     }

                     // Check after us.
                     if (local_linear_id < it.get_local_range(0) - END_OFFSET) {
                         // Only if the indices are not the same
                         if (local_cluster_indices_ptr[local_linear_id +
                                                       END_OFFSET] !=
                             cur_cluster_index) {
                             their_linear_id = local_linear_id + END_OFFSET;
                         }
                     }

                     // Actually do the processing.
                     if (their_linear_id != local_linear_id) {
                         ClusterExtents their_extent =
                             local_cluster_extents_ptr[their_linear_id];

                         size_t k_size = sycl::min(
                             static_cast<size_t>(20),
                             static_cast<size_t>(their_extent.count / 12));
                         size_t i0 = (x + their_extent.count -
                                      their_extent.start - k_size) %
                                     their_extent.count;
                         size_t i1 = (x + their_extent.count -
                                      their_extent.start + k_size) %
                                     their_extent.count;

                         // This feels optimizable
                         auto [moment, num_in_moment] =
                             get_moment(points + their_extent.start,
                                        their_extent.count, i0, i1);
                         fit_line(moment, num_in_moment, nullptr,
                                  &calculated_errors_diff_cluster_ptr
                                      [local_linear_id],
                                  nullptr);
                     } else {
                         calculated_errors_diff_cluster_ptr[local_linear_id] =
                             std::numeric_limits<double>::quiet_NaN();
                     }
                 }
                 // Barrier here to sync both sets of cluster errors.
                 it.barrier(sycl::access::fence_space::local_space);

                 auto filtered_errors_same_cluster_ptr =
                     filtered_errors_same_cluster.get_pointer();

                 // If we are not just responsible for fetching data, calculate
                 // the filtered errors.
                 if (local_linear_id >= POINTS_PER_END &&
                     local_linear_id < it.get_local_range(0) - POINTS_PER_END) {
                     double acc = 0.0;
                     for (size_t i = 0; i < FILTER_DATA.size(); i++) {
                         size_t linear_id_to_check =
                             local_linear_id + i - POINTS_PER_END;
                         if (local_cluster_indices_ptr[linear_id_to_check] ==
                             cur_cluster_index) {
                             acc += calculated_errors_same_cluster_ptr
                                        [linear_id_to_check] *
                                    FILTER_DATA[i];
                         } else {
                             acc += calculated_errors_diff_cluster_ptr
                                        [linear_id_to_check] *
                                    FILTER_DATA[i];
                         }
                     }
                     filtered_errors_same_cluster_ptr[local_linear_id] = acc;
                 }

                 auto filtered_errors_diff_cluster_ptr =
                     filtered_errors_diff_cluster.get_pointer();

                 // Calculate the filtered errors as-if we were a part of a
                 // cluster before or after us.
                 {
                     size_t their_linear_id = local_linear_id;

                     // Check before us.
                     if (local_linear_id > 1) {
                         if (local_cluster_indices_ptr[local_linear_id - 1] !=
                             cur_cluster_index) {
                             their_linear_id = local_linear_id - 1;
                         }
                     }

                     // Check after us.
                     if (local_linear_id < it.get_local_range(0) - 1) {
                         // Only if the indices are not the same
                         if (local_cluster_indices_ptr[local_linear_id + 1] !=
                             cur_cluster_index) {
                             their_linear_id = local_linear_id + 1;
                         }
                     }

                     // Actually do the processing.
                     if (their_linear_id != local_linear_id) {
                         double acc = 0.0;
                         for (size_t i = 0; i < FILTER_DATA.size(); i++) {
                             size_t linear_id_to_check =
                                 their_linear_id + i - POINTS_PER_END;
                             if (local_cluster_indices_ptr
                                     [linear_id_to_check] !=
                                 cur_cluster_index) {
                                 acc += calculated_errors_diff_cluster_ptr
                                            [linear_id_to_check] *
                                        FILTER_DATA[i];
                             } else {
                                 acc += calculated_errors_same_cluster_ptr
                                            [linear_id_to_check] *
                                        FILTER_DATA[i];
                             }
                         }
                         filtered_errors_diff_cluster_ptr[their_linear_id] =
                             acc;
                     } else {
                         filtered_errors_diff_cluster_ptr[their_linear_id] =
                             std::numeric_limits<double>::quiet_NaN();
                     }
                 }

                 it.barrier(sycl::access::fence_space::local_space);

                 // We no longer need the barriers so early return all
                 // non-corner-finding threads.
                 if (local_linear_id < END_OFFSET ||
                     local_linear_id >= it.get_local_range(0) - END_OFFSET) {
                     return;
                 }

                 // Bounds check for output
                 if (x < 0 || x >= points_size) {
                     return;
                 }

                 double cur_error =
                     filtered_errors_same_cluster_ptr[local_linear_id];

                 // If our cluster is the same as the appropirate point, check
                 // in the same cluster array, else we check in the diff cluster
                 // array.
                 bool before_same_index =
                     cur_cluster_index ==
                     local_cluster_indices_ptr[local_linear_id - 1];
                 bool greater_before =
                     cur_error >
                     (before_same_index
                          ? filtered_errors_same_cluster_ptr
                          : filtered_errors_diff_cluster_ptr)[local_linear_id -
                                                              1];
                 bool after_same_index =
                     cur_cluster_index ==
                     local_cluster_indices_ptr[local_linear_id + 1];
                 bool greater_after =
                     cur_error >
                     (after_same_index
                          ? filtered_errors_same_cluster_ptr
                          : filtered_errors_diff_cluster_ptr)[local_linear_id +
                                                              1];

                 Corner maybe = {
                     cur_error,
                     static_cast<uint32_t>(x),
                     cur_cluster_index,
                 };

                 if (greater_before && greater_after) {
                     found_corners[x] = maybe;
                 }
             });
     }).wait();
}
