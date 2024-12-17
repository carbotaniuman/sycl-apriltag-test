#include "fit_quad.h"
#include <cmath>
#include <tuple>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>


// Computed via AprilTag stuff and then dumped here.
constexpr size_t POINTS_PER_END = 3;
constexpr std::array<float, 7> FILTER_DATA = {
    0x1.6c0504p-7,
    0x1.152aaap-3,
    0x1.368b3p-1,
    0x1p+0,
    0x1.368b3p-1,
    0x1.152aaap-3,
    0x1.6c0504p-7
};

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

void fit_lines_test2(sycl::queue &q, const LineFitPoint *points, const uint16_t *cluster_indices, const ClusterExtents *cluster_extents, size_t points_size, Peak *found_peaks) {
    constexpr size_t TARGETED_WG_SIZE = 32;
    size_t count_needed = (points_size + 24) / 24 * TARGETED_WG_SIZE;
    size_t k_size = std::min(static_cast<size_t>(20), points_size / 12);
    q.submit([=](sycl::handler &h) {
        sycl::local_accessor<double> calculated_errors{sycl::range(TARGETED_WG_SIZE), h};
        sycl::local_accessor<double> filtered_errors{sycl::range(TARGETED_WG_SIZE), h};

        // X X X _ 1 2 3 4 _ X X X
        //         X X X _ 5 6 7 8 _ X X X
        //                 X X X _ 9 A B C _ X X X
        h.parallel_for(
            sycl::nd_range(sycl::range(count_needed), sycl::range(TARGETED_WG_SIZE)),
            [=](sycl::nd_item<1> it) {
                constexpr size_t END_OFFSET = POINTS_PER_END + 1;
                size_t x = it.get_group(0) * (it.get_local_range(0) - 2 * END_OFFSET) + it.get_local_id(0) - POINTS_PER_END;
                size_t local_linear_id = it.get_local_linear_id();
                auto calculated_error_ptr = calculated_errors.get_pointer();
                auto filtered_error_ptr = filtered_errors.get_pointer();

                size_t i0 = (x + points_size - k_size) % points_size;
                size_t i1 = (x + k_size) % points_size;
                // This feels optimizable
                auto [moment, num_in_moment] = get_moment(points, points_size, i0, i1);
                fit_line(moment, num_in_moment, nullptr, &calculated_error_ptr[local_linear_id], nullptr);

                it.barrier(sycl::access::fence_space::local_space);

                // check for overlap.
                if (size_t local_x = it.get_local_id(0);
                    local_x < POINTS_PER_END || local_x >= it.get_local_range(1) - POINTS_PER_END) {
                    return;
                }

                double acc = 0.0;
                for (size_t i = 0; i < FILTER_DATA.size(); i++) {
                    acc += calculated_errors[local_linear_id - POINTS_PER_END + i] * FILTER_DATA[i];
                }
                filtered_error_ptr[local_linear_id] = acc;

                std::cout << "HUH" << std::endl;

                it.barrier(sycl::access::fence_space::local_space);

                std::cout << "HUH!" << std::endl;

                // now the thing that calculated the one extraneous point is no longer needed.
                if (size_t local_x = it.get_local_id(0);
                    local_x < END_OFFSET || local_x >= it.get_local_range(1) - END_OFFSET) {
                        std::cout << local_x << std::endl;
                    return;
                }
                
                std::cout << "TRY" << std::endl;
                // Bounds check for output
                if (x < 0 || x >= points_size) {
                    return;
                }

                double cur_error = filtered_error_ptr[local_linear_id];

                bool greater_before = cur_error > filtered_error_ptr[local_linear_id - 1];
                bool greater_after = cur_error > filtered_error_ptr[local_linear_id + 1];

                Peak maybe = {
                    cur_error,
                    static_cast<uint32_t>(x),
                    cluster_indices[x],
                };

                if (greater_before && greater_after) {
                    std::cout << "FOO" << std::endl;
                    found_peaks[x] = maybe;
                }
            });
    }).wait();
}
