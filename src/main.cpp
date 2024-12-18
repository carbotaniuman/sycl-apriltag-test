#include "combinadics.h"
#include "stb_image.h"
#include "stb_image_write.h"

#include <sycl/sycl.hpp>

#include "cluster_bounds.h"
#include "find_boundaries.h"
#include "fit_quad.h"
#include "line_fit_point.h"
#include "segmentation.h"
#include "threshold.h"

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include <algorithm>
#include <cassert>
#include <execution>
#include <iostream>

int main(int argc, char *argv[]) {
    sycl::queue q{sycl::cpu_selector_v, sycl::property::queue::in_order{}};
    // auto policy_e = oneapi::dpl::execution::make_device_policy(q);
    auto policy_e = oneapi::dpl::execution::par_unseq;

    std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

    int width, height, comp;
    stbi_uc *data =
        stbi_load("../decimate.png", &width, &height, &comp, STBI_grey);
    fprintf(stdout, "width: %d, height: %d, comp: %d\n", width, height, comp);

    auto grayscale_buffer = sycl::malloc_shared<uint8_t>(width * height, q);
    auto extrema_buffer =
        sycl::malloc_shared<sycl::vec<uint8_t, 2>>(width / 4 * height / 4, q);
    auto thresholded_buffer = sycl::malloc_shared<uint8_t>(width * height, q);

    auto copy_image = q.copy(data, grayscale_buffer, width * height);

    auto threshold =
        threshold_image(q, grayscale_buffer, extrema_buffer, thresholded_buffer,
                        width, height, {copy_image});

    {
        auto output = new uint8_t[width * height];

        q.copy(thresholded_buffer, output, width * height, threshold);
        q.wait();

        stbi_write_png("thresholded.png", width, height, 1, output, width * 1);
    }

    auto label_buffer = sycl::malloc_shared<uint32_t>(width * height, q);
    size_t sizes_elems = 1 << 16;
    auto sizes_buffer = sycl::malloc_shared<HashTable::Entry>(sizes_elems, q);

    auto zero_labels =
        q.memset(label_buffer, 0, width * height * sizeof(uint32_t));
    auto zero_sizes =
        q.memset(sizes_buffer, 0, sizes_elems * sizeof(HashTable::Entry));

    auto segment = image_segmentation(q, thresholded_buffer, label_buffer,
                                      sizes_buffer, sizes_elems, width, height,
                                      {threshold, zero_labels, zero_sizes});

    {
        auto labels_out = new uint32_t[width * height];
        auto sizes_out = new HashTable::Entry[sizes_elems];

        q.copy(label_buffer, labels_out, width * height, segment);
        q.copy(sizes_buffer, sizes_out, sizes_elems, segment);
        q.wait();

        uint32_t *colors = new uint32_t[width * height];
        uint8_t *images = new uint8_t[width * height * 3];

        srand(time(0));

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                uint32_t v = labels_out[y * width + x] & LABEL_VALUE_MASK;

                uint32_t color = colors[v];

                uint8_t r = color >> 16, g = color >> 8, b = color;

                if (color == 0 && v != 0) {
                    if (sizes_out[v].value < 25) {
                        r = 0;
                        g = 0;
                        b = 0;
                    } else {
                        int32_t bias = 60;
                        r = bias + (random() % (200 - bias));
                        g = bias + (random() % (200 - bias));
                        b = bias + (random() % (200 - bias));
                        colors[v] = (r << 16) | (g << 8) | b;
                    }
                }

                images[(y * width + x) * 3 + 0] = r;
                images[(y * width + x) * 3 + 1] = g;
                images[(y * width + x) * 3 + 2] = b;
            }
        }

        stbi_write_png("segmented.png", width, height, 3, images, width * 3);
    }

    auto points_buffer =
        sycl::malloc_shared<BoundaryPoint>(width * height * 4, q);
    auto zero_points =
        q.memset(points_buffer, 0, width * height * 4 * sizeof(BoundaryPoint));

    auto boundaries =
        find_boundaries(q, label_buffer, sizes_buffer, 1 << 16, points_buffer,
                        width, height, {segment, zero_points});

    {
        auto points_out = new BoundaryPoint[width * height * 4]();
        q.copy(points_buffer, points_out, width * height * 4, boundaries);
        q.wait();

        size_t present = 0;
        size_t zeroes = 0;
        for (size_t i = 0; i < width * height * 4; i++) {
            // std::cout << "x " << points_out[i].x << " y " << points_out[i].y
            // << std::endl;
            if (points_out[i] ==
                sycl::bit_cast<BoundaryPoint>(static_cast<uint64_t>(0))) {
                zeroes++;
                continue;
            }
            present++;
        }
        std::cout << "points has " << present << " out of "
                  << width * height * 4 << " with " << zeroes << " zeros"
                  << std::endl;
    }
    boundaries.wait();

    auto compacted_points =
        sycl::malloc_shared<BoundaryPoint>(width * height * 4, q);

    auto compacted_points_end = oneapi::dpl::copy_if(
        policy_e, points_buffer, points_buffer + width * height * 4,
        compacted_points, [](BoundaryPoint p) {
            return p != sycl::bit_cast<BoundaryPoint>(static_cast<uint64_t>(0));
        });
    oneapi::dpl::sort(policy_e, compacted_points, compacted_points_end,
                      [](const auto &left, const auto &right) {
                          return left.blob_label() < right.blob_label();
                      });
    auto compacted_points_count =
        std::distance(compacted_points, compacted_points_end);

    std::cout << "compacted points was " << compacted_points_count << std::endl;
    {
        auto points_out = new BoundaryPoint[compacted_points_count]();
        q.copy(compacted_points, points_out, compacted_points_count);
        q.wait();
        uint32_t first = 0;
        uint32_t first_exc = 0;

        uint8_t *cluster_image = new uint8_t[width * height * 3]();
        std::unordered_map<uint32_t, uint32_t> vs{};
        for (size_t i = 0; i < compacted_points_count; i++) {
            if (points_out[i].x == 0 && points_out[i].y == 0) {
                break;
            }

            uint32_t label = points_out[i].blob_label();
            if (first == 0) {
                first = label;
            } else if (first != label) {
                first_exc = i;
            }

            uint32_t color = 0;
            if (auto found = vs.find(label); found != vs.end()) {
                color = found->second;
            } else if (label != 0) {
                int32_t bias = 60;
                uint8_t r = bias + (random() % (200 - bias));
                uint8_t g = bias + (random() % (200 - bias));
                uint8_t b = bias + (random() % (200 - bias));
                vs[label] = (r << 16) | (g << 8) | b;
                color = vs[label];
            }
            uint8_t r = color >> 16, g = color >> 8, b = color;

            auto x = points_out[i].x & COORDINATE_VALUE_MASK;
            auto y = points_out[i].y & COORDINATE_VALUE_MASK;

            cluster_image[(y * width + x) * 3 + 0] = r;
            cluster_image[(y * width + x) * 3 + 1] = g;
            cluster_image[(y * width + x) * 3 + 2] = b;
        }
        stbi_write_png("clusters.png", width, height, 3, cluster_image,
                       width * 3);
    }

    auto transform_keys = dpl::make_transform_iterator(
        compacted_points, [](BoundaryPoint a) { return a.blob_label(); });

    auto transform_values = dpl::make_transform_iterator(
        dpl::make_zip_iterator(compacted_points,
                               dpl::counting_iterator<uint32_t>(0)),
        [](auto a) {
            return ClusterBounds::inital_from_point(std::get<0>(a),
                                                    std::get<1>(a));
        });

    auto values_buffer = sycl::malloc_shared<ClusterBounds>(1 << 16, q);

    auto [keys_end, values_end] = oneapi::dpl::reduce_by_segment(
        policy_e, transform_keys, transform_keys + compacted_points_count,
        transform_values, oneapi::dpl::discard_iterator(), values_buffer,
        std::equal_to<>(), reduce_bounds);

    std::cout << "Compacted this many bounds "
              << std::distance(values_buffer, values_end) << std::endl;

    // {
    //     size_t unfiltered_point_count = 0;
    //     for (auto a = values_buffer; a != values_end; a++) {
    //         std::cout << "i " << std::distance(values_buffer, a) <<
    //         std::endl; std::cout << "start " << a->start << std::endl;
    //         std::cout << "count " << a->count << std::endl;
    //         std::cout << "x_min " << a->x_min << std::endl;
    //         std::cout << "x_max " << a->x_max << std::endl;
    //         std::cout << "y_min " << a->y_min << std::endl;
    //         std::cout << "y_max " << a->y_max << std::endl;

    //         unfiltered_point_count += a->count;
    //     }
    //     std::cout << "before filtering count is " << unfiltered_point_count
    //     << std::endl;
    // }

    auto valid_blob_filter = ValidBlobFilter();

    auto filtered_values_buffer =
        sycl::malloc_shared<ClusterBounds>(1 << 16, q);
    auto filtered_values_end =
        oneapi::dpl::copy_if(policy_e, values_buffer, values_end,
                             filtered_values_buffer, valid_blob_filter);

    size_t filtered_values_distance =
        std::distance(filtered_values_buffer, filtered_values_end);

    std::cout << "filtered distance is " << filtered_values_distance
              << std::endl;

    // {
    //     size_t filtered_point_count = 0;
    //     for (auto a = filtered_values_buffer; a != filtered_values_end; a++)
    //     {
    //         std::cout << "i " << std::distance(filtered_values_buffer, a) <<
    //         std::endl; std::cout << "start " << a->start << std::endl;
    //         std::cout << "count " << a->count << std::endl;
    //         std::cout << "x_min " << a->x_min << std::endl;
    //         std::cout << "x_max " << a->x_max << std::endl;
    //         std::cout << "y_min " << a->y_min << std::endl;
    //         std::cout << "y_max " << a->y_max << std::endl;
    //         filtered_point_count += a->count;
    //     }
    //     std::cout << "after filtering count is " << filtered_point_count <<
    //     std::endl;
    // }

    auto filtered_cluster_indexes =
        sycl::malloc_shared<uint16_t>(width * height * 4, q);
    auto filtered_cluster_points =
        sycl::malloc_shared<ClusterPoint>(width * height * 4, q);

    auto transformed_to_cluster_points = oneapi::dpl::make_transform_iterator(
        oneapi::dpl::make_zip_iterator(
            compacted_points, oneapi::dpl::counting_iterator<uint32_t>(0)),
        [filtered_values_buffer, filtered_values_end](auto a) {
            uint32_t i = std::get<1>(a);

            auto found_iter = oneapi::dpl::lower_bound(
                filtered_values_buffer, filtered_values_end, i,
                [](ClusterBounds a, uint32_t b) {
                    if ((a.start + a.count) <= b) {
                        return true;
                    }
                    return false;
                });

            if (found_iter == filtered_values_end) {
                return std::make_tuple(ClusterPoint(),
                                       std::numeric_limits<uint16_t>::max());
            }

            const auto &cluster = *found_iter;

            if (i < cluster.start || i >= (cluster.start + cluster.count)) {
                return std::make_tuple(ClusterPoint(),
                                       std::numeric_limits<uint16_t>::max());
            }

            auto dist = std::distance(filtered_values_buffer, found_iter);

            BoundaryPoint p = std::get<0>(a);

            return std::make_tuple(
                ClusterPoint(p.x, p.y,
                             p.calc_theta(cluster.cx(), cluster.cy())),
                static_cast<uint16_t>(dist));
        });

    auto o_zipped_iterator = oneapi::dpl::make_zip_iterator(
        filtered_cluster_points, filtered_cluster_indexes);

    auto o_zipped_end = oneapi::dpl::copy_if(
        policy_e, transformed_to_cluster_points,
        transformed_to_cluster_points + compacted_points_count,
        o_zipped_iterator, [values_buffer](const auto &p) {
            if (std::get<1>(p) == std::numeric_limits<uint16_t>::max()) {
                return false;
            }
            return true;
        });

    auto rewritten_filtered_values_buffer =
        sycl::malloc_shared<ClusterExtents>(1 << 16, q);
    oneapi::dpl::transform_inclusive_scan(
        policy_e, filtered_values_buffer, filtered_values_end,
        rewritten_filtered_values_buffer,
        [](const ClusterExtents &left, const ClusterExtents &right) {
            return ClusterExtents(left.start + left.count, right.count);
        },
        [](const ClusterBounds &a) { return ClusterExtents(0, a.count); },
        ClusterExtents(0, 0));

    // {
    //     for (int i = 0; i < 5; i++) {
    //         std::cout << "!!!" << std::endl;
    //         std::cout << "old s " << filtered_values_buffer[i].start
    //                   << std::endl;
    //         std::cout << "old c " << filtered_values_buffer[i].count
    //                   << std::endl;
    //         std::cout << "new c " <<
    //         rewritten_filtered_values_buffer[i].start
    //                   << std::endl;
    //         std::cout << "new c " <<
    //         rewritten_filtered_values_buffer[i].count
    //                   << std::endl;
    //     }
    // }

    oneapi::dpl::sort(policy_e, o_zipped_iterator, o_zipped_end,
                      [](const auto &left, const auto &right) {
                          uint32_t l_index = std::get<1>(left);
                          uint32_t r_index = std::get<1>(right);

                          if (l_index != r_index) {
                              return l_index < r_index;
                          }

                          auto &l_point = std::get<0>(left);
                          auto &r_point = std::get<0>(right);

                          return l_point.slope < r_point.slope;
                      });

    auto filtered_points_count = std::distance(o_zipped_iterator, o_zipped_end);

    auto line_fit_points_buffer =
        sycl::malloc_shared<LineFitPoint>(width * height * 4, q);

    auto transformed_to_linefit_points = oneapi::dpl::make_transform_iterator(
        filtered_cluster_points,
        [width, height, grayscale_buffer](const ClusterPoint &a) {
            return compute_initial_linefit(a, width, height, grayscale_buffer);
        });

    auto asdasd = oneapi::dpl::inclusive_scan_by_segment(
        policy_e, filtered_cluster_indexes,
        filtered_cluster_indexes + filtered_points_count,
        transformed_to_linefit_points, line_fit_points_buffer);

    // {
    //     for (int i = 0; i < 10; i++) {
    //         std::cout << "i " << i << std::endl;
    //         std::cout << transformed_to_linefit_points[i].W << std::endl;
    //         std::cout << line_fit_points_buffer[i].W << std::endl;
    //     }
    // }

    // std::cout << "test tt" << std::distance(line_fit_points_buffer, asdasd)
    // << std::endl;

    auto found_corners_buffer =
        sycl::malloc_shared<Corner>(width * height * 4, q);
    q.memset(found_corners_buffer, 0, width * height * 4 * sizeof(Corner))
        .wait();

    int good_count = 0;
    double *question = new double[24];
    for (int i = 0; i < filtered_values_distance; i++) {
        auto cluster = rewritten_filtered_values_buffer[i];
        auto old_count = good_count;

        double *temp = new double[cluster.count]();

        for (int j = 0; j < cluster.count; j++) {
            size_t k_size = std::min(static_cast<size_t>(20),
                                     static_cast<size_t>(cluster.count / 12));
            size_t i0 = (j + cluster.count - k_size) % cluster.count;
            size_t i1 = (j + k_size) % cluster.count;

            // This feels optimizable
            auto [moment, num_in_moment] = get_moment(
                line_fit_points_buffer + cluster.start, cluster.count, i0, i1);
            fit_line(moment, num_in_moment, nullptr, &temp[j], nullptr);
        }

        constexpr std::array<float, 7> FILTER_DATA = {
            0x1.6c0504p-7, 0x1.152aaap-3, 0x1.368b3p-1, 0x1p+0,
            0x1.368b3p-1,  0x1.152aaap-3, 0x1.6c0504p-7};

        double *temp2 = new double[cluster.count]();
        for (int j = 0; j < cluster.count; j++) {
            double acc = 0.0;
            for (int k = 0; k < 7; k++) {
                size_t ind = (j + cluster.count + k - 3) % cluster.count;
                acc += temp[ind] * FILTER_DATA[k];
            }
            temp2[j] = acc;
        }

        for (int j = 0; j < cluster.count; j++) {
            size_t bef = (j + cluster.count - 1) % cluster.count;
            size_t aft = (j + 1) % cluster.count;

            if (temp2[j] > temp2[bef] && temp2[j] > temp2[aft]) {
                good_count++;
            }
        }
        std::cout << "index " << i << "has size" << cluster.count
                  << " with good count " << (good_count - old_count)
                  << std::endl;
    }

    double *test2 = new double;

    fit_lines(q, line_fit_points_buffer, filtered_cluster_indexes,
              rewritten_filtered_values_buffer, filtered_points_count,
              found_corners_buffer, test2);

    auto compacted_corners = sycl::malloc_shared<Corner>(width * height * 4, q);

    auto compacted_corners_end = oneapi::dpl::copy_if(
        policy_e, found_corners_buffer,
        found_corners_buffer + width * height * 4, compacted_corners,
        [](const Corner &p) { return p.error != 0; });

    {
        int ll = 0;
        int lcount = 0;

        for (auto a = compacted_corners; a != compacted_corners_end; a++) {
            if (a->cluster_index == ll) {
                lcount++;
            } else {
                std::cout << "index " << ll << " had count " << lcount
                          << std::endl;
                ll = a->cluster_index;
                lcount = 1;
            }
        }
        std::cout << "index " << ll << " had count " << lcount << std::endl;
    }

    oneapi::dpl::sort(policy_e, compacted_corners, compacted_corners_end,
                      [](const auto &left, const auto &right) {
                          if (left.cluster_index != right.cluster_index) {
                              return left.cluster_index < right.cluster_index;
                          }
                          return left.error < right.error;
                      });

    std::cout << "compacted corner distance is "
              << std::distance(compacted_corners, compacted_corners_end)
              << std::endl;

    std::cout << "good(?) distance is " << good_count << std::endl;

    Combinadics<uint8_t, 15, 4, 16> combinadics{};
}
