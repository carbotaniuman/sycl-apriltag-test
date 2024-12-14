#include "fit_quad.h"
#include "combinadics.h"
#include <cmath>
#include <tuple>
#include <type_traits>

constexpr float MAX_LINE_FIT_MSE = 10.0;

// Computed via AprilTag stuff and then dumped here.
constexpr size_t POINTS_PER_END = 3;
constexpr std::array<float, 7> FILTER_DATA = {
    0x1.6c0504p-7, 0x1.152aaap-3, 0x1.368b3p-1, 0x1p+0,
    0x1.368b3p-1,  0x1.152aaap-3, 0x1.6c0504p-7};

std::tuple<LineFitPoint, size_t> get_moment(const LineFitPoint *points,
                                            size_t points_size, size_t i0,
                                            size_t i1) {
    LineFitPoint result{};
    size_t total_elems = 0;

    if (i0 < i1) {
        total_elems = i1 - i0 + 1;

        result = points[i1];

        if (i0 > 0) {
            result -= points[i0 - 1];
        }
    } else if (i1 < i0) {
        total_elems = points_size - i0 + i1 + 1;

        result = points[points_size - 1] - points[i0 - 1] + points[i1];
    }
    return std::make_tuple(result, total_elems);
}

void fit_line(LineFitPoint moment, size_t num_in_moment, float *line_params,
              float *err, float *mse) {
    // These M* values are double what they should be due to the transform
    // we made converting them to integers, but we divide them out so it should
    // not be an issue.

    // Both sides are now scaled by (4 * W^2),
    // which means we can extract the real
    // value by dividing.
    int64_t Csxx = moment.Mxx * moment.W - moment.Mx * moment.Mx;
    int64_t Csxy = moment.Mxy * moment.W - moment.Mx * moment.My;
    int64_t Csyy = moment.Myy * moment.W - moment.My * moment.My;

    int64_t correction_factor = moment.W * moment.W * 4;
    
    // This hypot is now scaled by (4 * W^2) over
    // the real value.
    float sdist = std::hypotf(Csxx - Csyy, 2 * Csxy);
    // We want half of the value, so divide by the correction factor
    // and then divide by a further 2.
    float eig_err = (Csxx + Csyy - sdist) / (correction_factor * 2);


    if (line_params) {
        // double eig = 0.5 * (Cxx + Cyy + dist);
        // double nx1 = Cxx - eig;

        // nx1 = Cxx - (0.5 * Cxx) - (0.5 * Cyy) - (0.5 * dist);
        // nx1 = (0.5 * Cxx) - (0.5 * Cyy) - (0.5 * dist);
        // nx1 = 0.5 * (Cxx - Cyy - dist)

        // Scale everything by 2 to reduce operations performed.
        float nx1 = static_cast<float>(Csxx - Csyy) - sdist;
        float ny1 = 2 * Csxy;
        float nx2 = 2 * Csxy;
        float ny2 = static_cast<float>(Csyy - Csxx) - sdist;
        float M1 = nx1 * nx1 + ny1 * ny1;
        float M2 = nx2 * nx2 + ny2 * ny2;

        float nx, ny;
        if (M1 > M2) {
            nx = nx1;
            ny = ny1;
        } else {
            nx = nx2;
            ny = ny2;
        }

        line_params[0] = static_cast<float>(moment.Mx) / static_cast<float>(moment.W * 2);
        line_params[1] = static_cast<float>(moment.My) / static_cast<float>(moment.W * 2);

        float length = std::hypotf(nx, ny);
        if (std::fabs(length) < 1e-12) {
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
               Corner *found_corners, const std::vector<sycl::event> &deps) {
    constexpr size_t TARGETED_WG_SIZE = 32;
    constexpr size_t END_OFFSET = POINTS_PER_END + 1;

    // std::cout << "points_size = " << points_size << std::endl;

    size_t count_needed = (points_size + 23) / 24 * TARGETED_WG_SIZE;
    q.submit([=](sycl::handler &h) {
         h.depends_on(deps);
         sycl::local_accessor<uint16_t> local_cluster_indices{
             sycl::range(TARGETED_WG_SIZE), h};
         sycl::local_accessor<ClusterExtents> local_cluster_extents{
             sycl::range(TARGETED_WG_SIZE), h};

         sycl::local_accessor<float> calculated_errors_same_cluster{
             sycl::range(TARGETED_WG_SIZE), h};
         sycl::local_accessor<float> calculated_errors_diff_cluster{
             sycl::range(TARGETED_WG_SIZE), h};
         sycl::local_accessor<float> filtered_errors_same_cluster{
             sycl::range(TARGETED_WG_SIZE), h};
         sycl::local_accessor<float> filtered_errors_diff_cluster{
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

                 // Load the indices into local memory.
                 uint16_t cur_cluster_index = cluster_indices[sycl::clamp(
                     x, static_cast<decltype(x)>(0),
                     static_cast<decltype(x)>(points_size - 1))];
                 local_cluster_indices[local_linear_id] = cur_cluster_index;
                 ClusterExtents cur_extent = cluster_extents[cur_cluster_index];
                 local_cluster_extents[local_linear_id] = cur_extent;

                 // Barrier here to sync the indices.
                 it.barrier(sycl::access::fence_space::local_space);

                 // Calculate the error for our current index for our cluster.
                 {
                     size_t k_size =
                         sycl::min(static_cast<size_t>(20),
                                   static_cast<size_t>(cur_extent.count / 12));

                     if (k_size < 2) {
                         calculated_errors_same_cluster[local_linear_id] =
                             std::numeric_limits<float>::quiet_NaN();
                     } else {
                         size_t i0 = (cur_extent.count + x - cur_extent.start -
                                      k_size) %
                                     cur_extent.count;
                         size_t i1 = (cur_extent.count + x - cur_extent.start +
                                      k_size) %
                                     cur_extent.count;

                         // This feels optimizable
                         auto [moment, num_in_moment] =
                             get_moment(points + cur_extent.start,
                                        cur_extent.count, i0, i1);
                         fit_line(
                             moment, num_in_moment, nullptr,
                             &calculated_errors_same_cluster[local_linear_id],
                             nullptr);
                     }
                 }

                 // Because of the min cluster size of 24, we can (maybe) have a
                 // different cluster to the left of us or to the right of us
                 // (by 3), but not both.

                 // Maybe calculate an error for another cluster if that cluster
                 // would need our data (within `END_OFFSET` of another
                 // cluster).
                 {
                     size_t their_linear_id = local_linear_id;

                     // Check before us.
                     if (local_linear_id >= END_OFFSET) {
                         // Only if the indices are not the same
                         if (local_cluster_indices[local_linear_id -
                                                   END_OFFSET] !=
                             cur_cluster_index) {
                             their_linear_id = local_linear_id - END_OFFSET;
                         }
                     }

                     // Check after us.
                     if (local_linear_id < it.get_local_range(0) - END_OFFSET) {
                         // Only if the indices are not the same
                         if (local_cluster_indices[local_linear_id +
                                                   END_OFFSET] !=
                             cur_cluster_index) {
                             their_linear_id = local_linear_id + END_OFFSET;
                         }
                     }

                     // Actually do the processing.
                     if (their_linear_id != local_linear_id) {
                         ClusterExtents their_extent =
                             local_cluster_extents[their_linear_id];

                         size_t k_size = sycl::min(
                             static_cast<size_t>(20),
                             static_cast<size_t>(their_extent.count / 12));

                         if (k_size < 2) {
                             calculated_errors_diff_cluster[local_linear_id] =
                                 std::numeric_limits<float>::quiet_NaN();
                         } else {
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
                                      &calculated_errors_diff_cluster
                                          [local_linear_id],
                                      nullptr);
                         }
                     } else {
                         calculated_errors_diff_cluster[local_linear_id] =
                             std::numeric_limits<float>::quiet_NaN();
                     }
                 }
                 // Barrier here to sync both sets of cluster errors.
                 it.barrier(sycl::access::fence_space::local_space);

                 // If we are not just responsible for fetching data, calculate
                 // the filtered errors.
                 if (local_linear_id >= POINTS_PER_END &&
                     local_linear_id < it.get_local_range(0) - POINTS_PER_END) {
                     float acc = 0.0;
                     float diff_acc = 0.0;
                     for (size_t i = 0; i < FILTER_DATA.size(); i++) {
                         size_t linear_id_to_check =
                             local_linear_id + i - POINTS_PER_END;
                         if (local_cluster_indices[linear_id_to_check] ==
                             cur_cluster_index) {
                             acc += calculated_errors_same_cluster
                                        [linear_id_to_check] *
                                    FILTER_DATA[i];

                             diff_acc += calculated_errors_diff_cluster
                                             [linear_id_to_check] *
                                         FILTER_DATA[i];
                         } else {
                             diff_acc += calculated_errors_same_cluster
                                             [linear_id_to_check] *
                                         FILTER_DATA[i];

                             acc += calculated_errors_diff_cluster
                                        [linear_id_to_check] *
                                    FILTER_DATA[i];
                         }
                     }
                     filtered_errors_same_cluster[local_linear_id] = acc;
                     filtered_errors_diff_cluster[local_linear_id] = diff_acc;
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

                 float cur_error =
                     filtered_errors_same_cluster[local_linear_id];

                 // If our cluster is the same as the appropirate point, check
                 // in the same cluster array, else we check in the diff cluster
                 // array.
                 bool before_same_index =
                     cur_cluster_index ==
                     local_cluster_indices[local_linear_id - 1];
                 bool greater_before =
                     cur_error >
                     (before_same_index
                          ? filtered_errors_same_cluster
                          : filtered_errors_diff_cluster)[local_linear_id - 1];
                 bool after_same_index =
                     cur_cluster_index ==
                     local_cluster_indices[local_linear_id + 1];
                 bool greater_after =
                     cur_error >
                     (after_same_index
                          ? filtered_errors_same_cluster
                          : filtered_errors_diff_cluster)[local_linear_id + 1];

                 Corner maybe = {
                     static_cast<uint32_t>(x - cur_extent.start),
                     cur_cluster_index,
                     cur_error,
                 };

                 if (greater_before && greater_after) {
                     found_corners[x] = maybe;
                 }
             });
     }).wait();
}

template <class T> void sort_4(std::array<T, 4> &arr) {
    auto swap_index = [&arr](size_t a, size_t b) {
        if (!(arr[a] < arr[b])) {
            std::swap(arr[a], arr[b]);
        }
    };

    swap_index(0, 1);
    swap_index(2, 3);
    swap_index(0, 2);
    swap_index(1, 3);
    swap_index(1, 2);
}

using ChosenCombinadics = Combinadics<uint8_t, 15, 4, 16>;
void do_indexing(sycl::queue &q, const PeakExtents *extents,
                 size_t extents_count, const Corner *compacted_corners,
                 const LineFitPoint *points,
                 const ClusterExtents *cluster_extents, FittedQuad *fitted) {
    constexpr size_t CHOSEN_PER_EXTENT = 256;
    constexpr size_t TARGETED_WG_SIZE = 32;

    float cos_critical_rad = std::cos(10 * M_PI / 180);
    // laucnh 2d kernel {extents_count, 10 choose 4};
    q.parallel_for(
         sycl::nd_range(sycl::range(extents_count, CHOSEN_PER_EXTENT),
                        sycl::range(1, CHOSEN_PER_EXTENT)),
         [=](sycl::nd_item<2> it) {
             auto cluster_id = it.get_global_id(0);
             auto combination_number = it.get_global_id(1);
             PeakExtents peak_extent = extents[cluster_id];
             ClusterExtents cur_extent = cluster_extents[cluster_id];

             bool is_valid_combination =
                 combination_number <
                 ChosenCombinadics::n_choose_k(
                     sycl::min(peak_extent.count, static_cast<uint32_t>(10)),
                     4);

             float total_err = 0.0;
             bool is_valid_fit = is_valid_combination;

             std::array<LineFitPoint, 4> moments{};
             std::array<size_t, 4> num_in_moments{};
             std::array<size_t, 4> line_fit_indices{};

             if (is_valid_combination) {
                 auto indices = ChosenCombinadics::decode(combination_number);
                 // std::cout << indices[0] << " " << indices[1] << " " <<
                 // indices[2] << " " << indices[3] << std::endl;

                 line_fit_indices = {
                     compacted_corners[peak_extent.start + indices[0]]
                         .line_fit_point_index,
                     compacted_corners[peak_extent.start + indices[1]]
                         .line_fit_point_index,
                     compacted_corners[peak_extent.start + indices[2]]
                         .line_fit_point_index,
                     compacted_corners[peak_extent.start + indices[3]]
                         .line_fit_point_index};
                 sort_4(line_fit_indices);

                 std::array<float, 4> params01;
                 std::array<float, 4> params12;

                 LineFitPoint m0, m1, m2, m3;
                 uint16_t nim0, nim1, nim2, nim3;

                 {
                     auto [moment, num_in_moment] =
                         get_moment(points + cur_extent.start, cur_extent.count,
                                    line_fit_indices[0], line_fit_indices[1]);
                     float err, mse;
                     fit_line(moment, num_in_moment, params01.data(), &err,
                              &mse);
                     is_valid_fit &= mse <= MAX_LINE_FIT_MSE;
                     moments[0] = moment;
                     num_in_moments[0] = num_in_moment;
                     total_err += err;
                 }

                 {
                     auto [moment, num_in_moment] =
                         get_moment(points + cur_extent.start, cur_extent.count,
                                    line_fit_indices[1], line_fit_indices[2]);
                     float err, mse;
                     fit_line(moment, num_in_moment, params12.data(), &err,
                              &mse);
                     is_valid_fit &= mse <= MAX_LINE_FIT_MSE;
                     moments[1] = moment;
                     num_in_moments[1] = num_in_moment;
                     total_err += err;
                 }

                 {
                     auto [moment, num_in_moment] =
                         get_moment(points + cur_extent.start, cur_extent.count,
                                    line_fit_indices[2], line_fit_indices[3]);
                     float err, mse;
                     fit_line(moment, num_in_moment, nullptr, &err, &mse);
                     is_valid_fit &= mse <= MAX_LINE_FIT_MSE;
                     moments[2] = moment;
                     num_in_moments[2] = num_in_moment;
                     total_err += err;
                 }

                 {
                     auto [moment, num_in_moment] =
                         get_moment(points + cur_extent.start, cur_extent.count,
                                    line_fit_indices[3], line_fit_indices[0]);
                     float err, mse;
                     fit_line(moment, num_in_moment, nullptr, &err, &mse);
                     is_valid_fit &= mse <= MAX_LINE_FIT_MSE;
                     moments[3] = moment;
                     num_in_moments[3] = num_in_moment;
                     total_err += err;
                 }

                 float dot =
                     params01[2] * params12[2] + params01[3] * params12[3];

                 is_valid_fit &= std::fabs(dot) <= cos_critical_rad;
             }

             float current_err = is_valid_fit
                                      ? total_err
                                      : std::numeric_limits<float>::infinity();
             float lowest_err = sycl::reduce_over_group(
                 it.get_group(), current_err, sycl::minimum<>());

             if (lowest_err == std::numeric_limits<float>::infinity()) {
                 return;
             }

             // substitue for a ballot because SYCL doesn't have that
             auto lowest_combo = sycl::inclusive_scan_over_group(
                 it.get_group(),
                 ((current_err - lowest_err) < 0.1)
                     ? combination_number
                     : std::numeric_limits<size_t>::max(),
                 sycl::minimum<>());

             if (combination_number == lowest_combo) {
                 fitted[cluster_id] =
                     FittedQuad{
                        moments,
                        {static_cast<uint16_t>(num_in_moments[0]),
                                 static_cast<uint16_t>(num_in_moments[1]),
                                 static_cast<uint16_t>(num_in_moments[2]),
                                 static_cast<uint16_t>(num_in_moments[3])},      
                        {static_cast<uint16_t>(line_fit_indices[0]),
                                 static_cast<uint16_t>(line_fit_indices[1]),
                                 static_cast<uint16_t>(line_fit_indices[2]),
                                 static_cast<uint16_t>(line_fit_indices[3])}
                     };
             }
         })
        .wait();
}
