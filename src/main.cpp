#define NDEBUG
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
#include <chrono>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>

void process_fitted_quads(const std::vector<FittedQuad>& fit_quads_host_, size_t width, size_t height, uint8_t *greyscaled, bool debug);

void image_u8_draw_line(uint8_t *im, float x0, float y0, float x1, float y1,
                        int width, int height) {
    double dist = std::hypot((y1 - y0), (x1 - x0));
    if (dist == 0) {
        return;
    }
    double delta = 0.5 / dist;

    // terrible line drawing code
    for (float f = 0; f <= 1; f += delta) {
        int x = ((int)(x1 + (x0 - x1) * f));
        int y = ((int)(y1 + (y0 - y1) * f));

        if (x < 0 || y < 0 || x >= width || y >= height)
            continue;

        int idx = (y * width + x) * 3;
        im[idx + 0] = 255;
        im[idx + 1] = 255;
        im[idx + 2] = 255;
    }
}

template <class T>
void dumpPlainToCSV(const T *data, size_t size, const std::string &filename) {
    // Open the CSV file for writing
    std::ofstream csvFile(filename);

    if (!csvFile.is_open()) {
        std::cerr << "Failed to open file for writing.\n";
        return;
    }

    // Write the CSV header
    csvFile << "data" << "\n";

    for (size_t i = 0; i < size; ++i) {
        const auto &it = data[i];
        csvFile << it << "\n";
    }

    // Close the CSV file
    csvFile.close();

    std::cout << "CSV file '" << filename << "' has been written.\n";
}

template <class Iter>
void dumpLineFitPointsToCSV(Iter line_fit_points, size_t size,
                            const std::string &filename) {
    // Open the CSV file for writing
    std::ofstream csvFile(filename);

    if (!csvFile.is_open()) {
        std::cerr << "Failed to open file for writing.\n";
        return;
    }

    // Write the CSV header
    csvFile << "Mx,My,Mxx,Myy,Mxy,W" << "\n";

    for (size_t i = 0; i < size; ++i) {
        const auto &point = line_fit_points[i];
        csvFile << point.Mx << "," << point.My << "," << point.Mxx << ","
                << point.Myy << "," << point.Mxy << "," << point.W << "\n";
    }

    // Close the CSV file
    csvFile.close();

    std::cout << "CSV file '" << filename << "' has been written.\n";
}

void dumpClusterBoundsToCSV(const ClusterBounds *bounds, size_t size,
                            const std::string &filename) {
    // Open the CSV file for writing
    std::ofstream csvFile(filename);

    if (!csvFile.is_open()) {
        std::cerr << "Failed to open file for writing.\n";
        return;
    }

    // Write the CSV header
    csvFile
        << "x_min,x_max,y_min,y_max,gx_sum,gy_sum,pxgx_pygy_sum,start,count\n";

    // Loop over the array and write each point to the CSV
    for (size_t i = 0; i < size; ++i) {
        const auto &bound = bounds[i];
        csvFile << bound.x_min << "," << bound.x_max << "," << bound.y_min
                << "," << bound.y_max << "," << bound.gx_sum << ","
                << bound.gy_sum << "," << bound.pxgx_pygy_sum << ","
                << bound.start << "," << bound.count << "\n";
    }

    // Close the CSV file
    csvFile.close();

    std::cout << "CSV file '" << filename << "' has been written.\n";
}

void dumpCornerToCSV(const Corner *corner, size_t size,
                     const std::string &filename) {
    // Open the CSV file for writing
    std::ofstream csvFile(filename);

    if (!csvFile.is_open()) {
        std::cerr << "Failed to open file for writing.\n";
        return;
    }

    // Write the CSV header
    csvFile << "error,line_fit_point_index,cluster_index\n";

    // Loop over the array and write each point to the CSV
    for (size_t i = 0; i < size; ++i) {
        const auto &point = corner[i];
        csvFile << point.error << "," << point.line_fit_point_index << ","
                << point.cluster_index << "\n";
    }

    // Close the CSV file
    csvFile.close();

    std::cout << "CSV file '" << filename << "' has been written.\n";
}

template <class ExtentLike>
void dumpExtentLikeToCSV(const ExtentLike *extents, size_t size,
                         const std::string &filename) {
    // Open the CSV file for writing
    std::ofstream csvFile(filename);

    if (!csvFile.is_open()) {
        std::cerr << "Failed to open file for writing.\n";
        return;
    }

    // Write the CSV header
    csvFile << "start,count\n";

    // Loop over the array and write each point to the CSV
    for (size_t i = 0; i < size; ++i) {
        const auto &extent = extents[i];
        csvFile << extent.start << "," << extent.count << "\n";
    }

    // Close the CSV file
    csvFile.close();

    std::cout << "CSV file '" << filename << "' has been written.\n";
}

void dumpBoundaryPointsToCSV(const BoundaryPoint *boundaryPoints,
                             const uint64_t *blob_labels, size_t size,
                             const std::string &filename) {
    // Open the CSV file for writing
    std::ofstream csvFile(filename);

    if (!csvFile.is_open()) {
        std::cerr << "Failed to open file for writing.\n";
        return;
    }

    // Write the CSV header
    csvFile
        << "x_value,y_value,blob_label,is_black_to_white,dx,dy\n";

    // Loop over the array and write each point to the CSV
    for (size_t i = 0; i < size; ++i) {
        const auto &point = boundaryPoints[i];
        csvFile << point.x_value() << "," << point.y_value() << ","
                << blob_labels[i] << ","
                << point.is_black_to_white() << "," << point.dx() << ","
                << point.dy() << "\n";
    }

    // Close the CSV file
    csvFile.close();

    std::cout << "CSV file '" << filename << "' has been written.\n";
}

void dumpClusterPointsToCSV(const ClusterPoint *boundaryPoints, size_t size,
                            const std::string &filename) {
    // Open the CSV file for writing
    std::ofstream csvFile(filename);

    if (!csvFile.is_open()) {
        std::cerr << "Failed to open file for writing.\n";
        return;
    }

    // Write the CSV header
    csvFile << "x_value,y_value,is_black_to_white,dx,dy,slope\n";

    // Loop over the array and write each point to the CSV
    for (size_t i = 0; i < size; ++i) {
        const auto &point = boundaryPoints[i];
        csvFile << point.x_value() << "," << point.y_value() << ","
                << point.is_black_to_white() << "," << point.dx() << ","
                << point.dy() << "," << point.slope << "\n";
    }

    // Close the CSV file
    csvFile.close();

    std::cout << "CSV file '" << filename << "' has been written.\n";
}

int main(int argc, char *argv[]) {
    bool debug = true;
    bool prog = true;
    sycl::queue q;
    if (argc == 1) {
        q = sycl::queue{sycl::cpu_selector_v,
                        sycl::property::queue::in_order{}};
    } else {
        q = sycl::queue{sycl::default_selector_v,
                        sycl::property::queue::in_order{}};
    }

    auto policy_e = oneapi::dpl::execution::make_device_policy(q);
    // auto policy_e = oneapi::dpl::execution::par_unseq;

    std::cout << "Local memory size: "
              << q.get_device().get_info<sycl::info::device::local_mem_size>()
              << std::endl;

    std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    int width, height, comp;

    stbi_uc *data =
        stbi_load("../decimate2.png", &width, &height, &comp, STBI_grey);
    fprintf(stdout, "width: %d, height: %d, comp: %d\n", width, height, comp);

    auto grayscale_buffer = sycl::malloc_device<uint8_t>(width * height, q);
    auto extrema_buffer =
        sycl::malloc_device<sycl::vec<uint8_t, 2>>(width / 4 * height / 4, q);
    auto filtered_extrema_buffer =
        sycl::malloc_device<sycl::vec<uint8_t, 2>>(width / 4 * height / 4, q);
    auto thresholded_buffer = sycl::malloc_device<uint8_t>(width * height, q);
    auto scratch_label_buffer =
        sycl::malloc_device<uint32_t>(width * height, q);
    auto label_buffer = sycl::malloc_device<uint32_t>(width * height, q);
    auto label_sizes_buffer = sycl::malloc_device<uint32_t>(width * height, q);
    auto points_buffer =
        sycl::malloc_device<BoundaryPoint>(width * height * 4, q);
    auto blob_labels_buffer =
        sycl::malloc_device<uint64_t>(width * height * 4, q);
    auto compacted_points =
        sycl::malloc_device<BoundaryPoint>(width * height * 4, q);
    auto compacted_blob_labels =
        sycl::malloc_device<uint64_t>(width * height * 4, q);
    size_t sizes_elems = 1 << 16;
    auto values_buffer = sycl::malloc_device<ClusterBounds>(sizes_elems, q);
    auto filtered_values_buffer =
        sycl::malloc_device<ClusterBounds>(sizes_elems, q);
    auto filtered_cluster_indexes =
        sycl::malloc_device<uint16_t>(width * height * 4, q);
    auto filtered_cluster_points =
        sycl::malloc_device<ClusterPoint>(width * height * 4, q);
    auto prewritten_filtered_values_buffer =
        sycl::malloc_device<uint32_t>(sizes_elems, q);
    auto rewritten_filtered_values_buffer =
        sycl::malloc_device<ClusterExtents>(sizes_elems, q);
    auto pre_line_fit_points_buffer =
        sycl::malloc_device<LineFitPoint>(width * height * 4, q);
    auto line_fit_points_buffer =
        sycl::malloc_device<LineFitPoint>(width * height * 4, q);
    auto found_corners_buffer =
        sycl::malloc_device<Corner>(width * height * 4, q);
    auto compacted_corners = sycl::malloc_device<Corner>(width * height * 4, q);
    auto cluster_data_new_buffer =
        sycl::malloc_device<PeakExtents>(width * height * 4, q);
    auto output_quads = sycl::malloc_device<FittedQuad>(width * height, q);

    for (int i = 0; i < 1; i++) {
        auto zero_sizes =
            q.memset(label_sizes_buffer, 0, sizes_elems * sizeof(uint32_t));
        auto zero_points = q.memset(points_buffer, 0,
                                    width * height * 4 * sizeof(BoundaryPoint));
        auto fill_blob_labels = q.memset(blob_labels_buffer, 0xFF,
                                    width * height * 4 * sizeof(uint64_t));
        auto zero_corners = q.memset(found_corners_buffer, 0,
                                     width * height * 4 * sizeof(Corner));
        auto zero_quads =
            q.memset(output_quads, 0, width * height * sizeof(FittedQuad));
        q.wait();

        auto start = std::chrono::high_resolution_clock::now();
        auto copy_image = q.copy(data, grayscale_buffer, width * height);

        auto threshold =
            threshold_image(q, grayscale_buffer, extrema_buffer, filtered_extrema_buffer,
                            thresholded_buffer, width, height, {copy_image});
        threshold.wait();

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "1: " << duration.count() << std::endl;
        }

        if (debug) {
            auto output = new uint8_t[width * height];

            q.copy(thresholded_buffer, output, width * height, threshold);
            q.wait();

            stbi_write_png("thresholded.png", width, height, 1, output,
                           width * 1);
        }

        auto segment = image_segmentation(
            q, thresholded_buffer, scratch_label_buffer, label_buffer,
            label_sizes_buffer, sizes_elems, width, height,
            {threshold, zero_sizes});
        segment.wait();

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "2: " << duration.count() << std::endl;
        }

        if (debug) {
            auto scratch_labels_out = new uint32_t[width * height];
            auto labels_out = new uint32_t[width * height];
            auto sizes_out = new uint32_t[width * height];

            q.copy(scratch_label_buffer, scratch_labels_out, width * height, segment);
            q.copy(label_buffer, labels_out, width * height, segment);
            q.copy(label_sizes_buffer, sizes_out, width * height, segment);
            q.wait();
            
            dumpPlainToCSV(scratch_labels_out, width * height, "outneg1.csv");
            dumpPlainToCSV(labels_out, width * height, "outneg2.csv");
            dumpPlainToCSV(sizes_out, width * height, "outneg3.csv");

            uint32_t *colors = new uint32_t[width * height];
            uint8_t *images = new uint8_t[width * height * 3];

            srand(555);

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    uint32_t v = labels_out[y * width + x] & LABEL_VALUE_MASK;

                    uint32_t color = colors[v];

                    uint8_t r = color >> 16, g = color >> 8, b = color;

                    if (color == 0 && v != 0) {
                        if (sizes_out[v] < 25) {
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

            stbi_write_png("segmented.png", width, height, 3, images,
                           width * 3);
        }

        auto boundaries = find_boundaries(q, label_buffer, label_sizes_buffer, points_buffer, blob_labels_buffer, width, height,
                                          {segment, zero_points, fill_blob_labels});
        boundaries.wait();

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "3: " << duration.count() << std::endl;
        }

        if (debug) {
            auto points_out = new BoundaryPoint[width * height * 4]();
            auto blob_labels_out = new uint64_t[width * height * 4]();
            q.copy(points_buffer, points_out, width * height * 4, boundaries);
            q.copy(blob_labels_buffer, blob_labels_out, width * height * 4, boundaries);
            q.wait();

            dumpBoundaryPointsToCSV(points_out, blob_labels_out, width * height * 4,
                                    "out0.csv");

            size_t present = 0;
            size_t zeroes = 0;
            for (size_t i = 0; i < width * height * 4; i++) {
                if (sycl::bit_cast<uint32_t>(points_out[i]) == 0) {
                    zeroes++;
                    continue;
                }
                present++;
            }
            std::cout << "points has " << present << " out of "
                      << width * height * 4 << " with " << zeroes << " zeros"
                      << std::endl;

            uint8_t *cluster_image = new uint8_t[width * height * 3]();
            for (size_t i = 0; i < width * height * 4; i++) {
                if (sycl::bit_cast<uint32_t>(points_out[i]) == 0) {
                    continue;
                }

                auto x = points_out[i].x_value();
                auto y = points_out[i].y_value();
                auto label = blob_labels_out[i];

                // std::cout << points_out[i].packed_x << " " << points_out[i].packed_y << std::endl;

                cluster_image[(y * width + x) * 3 + 0] = 0;
                cluster_image[(y * width + x) * 3 + 1] = 0;
                cluster_image[(y * width + x) * 3 + 2] = 0;

                cluster_image[(y * width + x) * 3 + (label % 3)] = 255;
            }
            stbi_write_png("points.png", width, height, 3, cluster_image,
                           width * 3);
        }

        auto blob_label_points = dpl::make_zip_iterator(blob_labels_buffer, points_buffer);
        auto compacted_blob_label_points = dpl::make_zip_iterator(compacted_blob_labels, compacted_points);

        auto compacted_blob_label_end = oneapi::dpl::copy_if(
            policy_e, blob_label_points, blob_label_points + width * height * 4,
            compacted_blob_label_points, [](const auto& p) {
                return sycl::bit_cast<uint32_t>(std::get<1>(p)) != 0;
            });

        size_t compacted_points_count =
            std::distance(compacted_blob_label_points, compacted_blob_label_end);

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "4: " << duration.count() << std::endl;
        }

        if (debug) {
            auto points_out = new BoundaryPoint[width * height * 4]();
            auto blob_labels_out = new uint64_t[width * height * 4]();
            q.copy(compacted_points, points_out, width * height * 4);
            q.copy(compacted_blob_labels, blob_labels_out, width * height * 4);
            q.wait();
            dumpBoundaryPointsToCSV(points_out, blob_labels_out, compacted_points_count,
                                    "out1.csv");

            uint8_t *cluster_image = new uint8_t[width * height * 3]();
            for (size_t i = 0; i < compacted_points_count; i++) {
                if (sycl::bit_cast<uint32_t>(points_out[i]) == 0) {
                    break;
                }

                auto x = points_out[i].x_value();
                auto y = points_out[i].y_value();
                auto label = blob_labels_out[i];

                cluster_image[(y * width + x) * 3 + 0] = 0;
                cluster_image[(y * width + x) * 3 + 1] = 0;
                cluster_image[(y * width + x) * 3 + 2] = 0;

                cluster_image[(y * width + x) * 3 + (label % 3)] = 255;
            }
            stbi_write_png("points1.png", width, height, 3, cluster_image,
                           width * 3);
        }

        oneapi::dpl::sort_by_key(policy_e,
                          compacted_blob_labels,
                          compacted_blob_labels + compacted_points_count,
                          compacted_points);

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "5: " << duration.count() << std::endl;
        }

        if (debug) {
            auto points_out = new BoundaryPoint[compacted_points_count]();
            auto blob_labels_out = new uint64_t[compacted_points_count]();
            q.copy(compacted_points, points_out, compacted_points_count);
            q.copy(compacted_blob_labels, blob_labels_out, compacted_points_count);
            q.wait();

            dumpBoundaryPointsToCSV(points_out, blob_labels_out, compacted_points_count,
                                    "out2.csv");

            uint8_t *cluster_image = new uint8_t[width * height * 3]();
            std::unordered_map<uint32_t, uint32_t> vs{};
            for (size_t i = 0; i < compacted_points_count; i++) {
                if (sycl::bit_cast<uint32_t>(points_out[i]) == 0) {
                    std::cout << "???" << std::endl;
                    break;
                }

                uint32_t label = blob_labels_out[i];

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

                auto x = points_out[i].x_value();
                auto y = points_out[i].y_value();

                cluster_image[(y * width + x) * 3 + 0] = r;
                cluster_image[(y * width + x) * 3 + 1] = g;
                cluster_image[(y * width + x) * 3 + 2] = b;
            }
            stbi_write_png("clusters.png", width, height, 3, cluster_image,
                           width * 3);
        }

        auto transform_values = dpl::make_transform_iterator(
            dpl::make_zip_iterator(compacted_points,
                                   dpl::counting_iterator<uint32_t>(0)),
            [](auto a) {
                return sycl::bit_cast<sycl::vec<int64_t, 4>>(ClusterBounds::inital_from_point(std::get<0>(a),
                                                        std::get<1>(a)));
            });

        auto values_start = reinterpret_cast<sycl::vec<int64_t, 4>*>(values_buffer);
        auto [keys_end, values_end] = oneapi::dpl::reduce_by_segment(
            policy_e, compacted_blob_labels, compacted_blob_labels + compacted_points_count,
            transform_values, oneapi::dpl::discard_iterator(), values_start,
            std::equal_to<>(), [](const auto &left, const auto &right) {
                return sycl::bit_cast<sycl::vec<int64_t, 4>>(reduce_bounds(sycl::bit_cast<ClusterBounds>(left), sycl::bit_cast<ClusterBounds>(right)));
            });

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "6: " << duration.count() << std::endl;
        }

        if (debug) {
            auto values_out = new ClusterBounds[sizes_elems]();
            q.copy(values_buffer, values_out, std::distance(values_start, values_end));
            q.wait();
            
            dumpClusterBoundsToCSV(values_out,
                                   std::distance(values_start, values_end),
                                   "out2bounds.csv");
        }

        auto valid_blob_filter = ValidBlobFilter(width, height);

        auto filtered_values_end = oneapi::dpl::copy_if(
            policy_e, values_buffer,
            values_buffer + std::distance(values_start, values_end),
            filtered_values_buffer, valid_blob_filter);

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "6b: " << duration.count() << std::endl;
        }

        size_t filtered_values_distance =
            std::distance(filtered_values_buffer, filtered_values_end);

        auto transformed_to_cluster_points =
            oneapi::dpl::make_transform_iterator(
                oneapi::dpl::make_zip_iterator(
                    compacted_points,
                    oneapi::dpl::counting_iterator<uint32_t>(0)),
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
                        return std::make_tuple(
                            ClusterPoint(),
                            std::numeric_limits<uint16_t>::max());
                    }

                    const auto &cluster = *found_iter;

                    if (i < cluster.start ||
                        i >= (cluster.start + cluster.count)) {
                        return std::make_tuple(
                            ClusterPoint(),
                            std::numeric_limits<uint16_t>::max());
                    }

                    auto dist =
                        std::distance(filtered_values_buffer, found_iter);

                    BoundaryPoint p = std::get<0>(a);

                    return std::make_tuple(
                        ClusterPoint{p.packed_x, p.packed_y,
                                     p.calc_theta(cluster.cx(), cluster.cy())},
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

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "7: " << duration.count() << std::endl;
        }

        auto filtered_points_count =
            std::distance(o_zipped_iterator, o_zipped_end);

        if (debug) {
            auto filtered_cluster_out = new ClusterPoint[width * height * 4]();
            auto filtered_indexes_out = new uint16_t[width * height * 4]();
            q.copy(filtered_cluster_points, filtered_cluster_out, filtered_points_count);
            q.copy(filtered_cluster_indexes, filtered_indexes_out, filtered_points_count);
            q.wait();

            dumpClusterPointsToCSV(filtered_cluster_out,
                                   filtered_points_count, "out3.csv");
            dumpPlainToCSV(filtered_indexes_out, filtered_points_count,
                           "out4.csv");
        }

        auto transformed_extents_iter = dpl::make_transform_iterator(
            filtered_values_buffer, [](const auto &a) { return a.count; });

        oneapi::dpl::exclusive_scan(
            policy_e, transformed_extents_iter,
            transformed_extents_iter +
                std::distance(filtered_values_buffer, filtered_values_end),
            prewritten_filtered_values_buffer, 0);

        oneapi::dpl::transform(
            policy_e,
            transformed_extents_iter,
            transformed_extents_iter +
                std::distance(filtered_values_buffer, filtered_values_end),
            prewritten_filtered_values_buffer, rewritten_filtered_values_buffer,
            [](uint32_t count, uint32_t start) {
                return ClusterExtents{start, count};
            });

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "8: " << duration.count() << std::endl;
        }

        if (debug) {
            auto rewritten_filtered_out = new ClusterExtents[width * height * 4]();
            q.copy(rewritten_filtered_values_buffer, rewritten_filtered_out, std::distance(filtered_values_buffer, filtered_values_end));
            q.wait();

            dumpExtentLikeToCSV(
                rewritten_filtered_out,
                std::distance(filtered_values_buffer, filtered_values_end),
                "out4extents.csv");
        }

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

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "9: " << duration.count() << std::endl;
        }

        if (debug) {
            auto filtered_cluster_out = new ClusterPoint[width * height * 4]();
            q.copy(filtered_cluster_points, filtered_cluster_out, filtered_points_count);
            q.wait();

            dumpClusterPointsToCSV(filtered_cluster_out,
                                   filtered_points_count, "out5.csv");
        }

        oneapi::dpl::transform(
            policy_e, filtered_cluster_points,
            filtered_cluster_points + filtered_points_count,
            pre_line_fit_points_buffer,
            [width, height, grayscale_buffer](const ClusterPoint &a) {
                return compute_initial_linefit(a, width, height,
                                               grayscale_buffer);
            });

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "10: " << duration.count() << std::endl;
        }

        if (debug) {
            auto pre_line_fit_out = new LineFitPoint[width * height * 4]();
            q.copy(pre_line_fit_points_buffer, pre_line_fit_out, filtered_points_count);
            q.wait();

            dumpLineFitPointsToCSV(pre_line_fit_out,
                                   filtered_points_count, "out6a.csv");
        }

        auto asdasd_begin =
            reinterpret_cast<sycl::vec<double, 8> *>(line_fit_points_buffer);

        auto asdasd_end = oneapi::dpl::inclusive_scan_by_segment(
            policy_e, filtered_cluster_indexes,
            filtered_cluster_indexes + filtered_points_count,
            reinterpret_cast<sycl::vec<double, 8> *>(
                pre_line_fit_points_buffer),
            asdasd_begin);

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "11: " << duration.count() << std::endl;
        }

        auto line_fit_points_count = std::distance(asdasd_begin, asdasd_end);

        if (debug) {
            auto line_fit_out = new LineFitPoint[width * height * 4]();
            q.copy(line_fit_points_buffer, line_fit_out, line_fit_points_count);
            q.wait();

            dumpLineFitPointsToCSV(line_fit_out,
                                   line_fit_points_count, "out6b.csv");
        }

        fit_lines(q, line_fit_points_buffer, filtered_cluster_indexes,
                  rewritten_filtered_values_buffer, filtered_points_count,
                  found_corners_buffer, {zero_corners});

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "12: " << duration.count() << std::endl;
        }

        auto compacted_corners_end = oneapi::dpl::copy_if(
            policy_e, found_corners_buffer,
            found_corners_buffer + filtered_points_count, compacted_corners,
            [](const Corner &p) { return p.error != 0; });
        size_t compacted_corner_count =
            std::distance(compacted_corners, compacted_corners_end);

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "13: " << duration.count() << std::endl;
        }

        if (debug) {
            auto corner_out = new Corner[width * height * 4]();
            q.copy(compacted_corners, corner_out, compacted_corner_count);
            q.wait();

            dumpCornerToCSV(
                corner_out,
                compacted_corner_count,
                "out7.csv");
        }

        oneapi::dpl::sort(policy_e, compacted_corners, compacted_corners_end,
                          [](const auto &left, const auto &right) {
                              if (left.cluster_index != right.cluster_index) {
                                  return left.cluster_index <
                                         right.cluster_index;
                              }
                              return left.error > right.error;
                          });

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "14: " << duration.count() << std::endl;
        }

        if (debug) {
            auto corner_out = new Corner[width * height * 4]();
            q.copy(compacted_corners, corner_out, compacted_corner_count);
            q.wait();

            dumpCornerToCSV(corner_out, compacted_corner_count,
                            "out8.csv");
        }

        auto transform_corner_keys = dpl::make_transform_iterator(
            compacted_corners, [](Corner a) { return a.cluster_index; });

        auto transform_corner_values = dpl::make_transform_iterator(
            oneapi::dpl::counting_iterator<uint32_t>(0),
            [](uint32_t a) { return PeakExtents{a, 1}; });

        auto [corner_keys_end, corner_values_end] =
            oneapi::dpl::reduce_by_segment(
                policy_e, transform_corner_keys,
                transform_corner_keys + compacted_corner_count,
                transform_corner_values, oneapi::dpl::discard_iterator(),
                cluster_data_new_buffer, std::equal_to<>(),
                [](const auto &left, const auto &right) {
                    return reduce_extents(left, right);
                });

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "15: " << duration.count() << std::endl;
        }

        size_t cluster_data_new_count =
            std::distance(cluster_data_new_buffer, corner_values_end);

        if (debug) {
            auto cluster_point_out = new ClusterPoint[width * height * 4]();
            auto corner_out = new Corner[width * height * 4]();
            auto cluster_data_new_out = new PeakExtents[width * height]();
            auto rewritten_filtered_out = new ClusterExtents[width * height * 4]();
            q.copy(filtered_cluster_points, cluster_point_out, filtered_points_count);
            q.copy(compacted_corners, corner_out, compacted_corner_count);
            q.copy(cluster_data_new_buffer, cluster_data_new_out, cluster_data_new_count);
            q.copy(rewritten_filtered_values_buffer, rewritten_filtered_out, std::distance(filtered_values_buffer, filtered_values_end));
            q.wait();

            uint8_t *cluster_image = new uint8_t[width * height * 3]();
            for (int j = 0; j < filtered_points_count; j++) {
                auto x = cluster_point_out[j].x_value();
                auto y = cluster_point_out[j].y_value();

                cluster_image[(y * width + x) * 3 + 0] = 255;
                cluster_image[(y * width + x) * 3 + 1] = 255;
                cluster_image[(y * width + x) * 3 + 2] = 255;
            }

            for (int i = 0; i < cluster_data_new_count; i++) {
                const auto &peak_extent = cluster_data_new_out[i];
                const auto &extents = rewritten_filtered_out[i];

                for (int j = 0; j < peak_extent.count; j++) {
                    const auto &corner_test =
                        corner_out[peak_extent.start + j];

                    auto x =
                        cluster_point_out
                            [extents.start + corner_test.line_fit_point_index]
                                .x_value();
                    auto y =
                        cluster_point_out
                            [extents.start + corner_test.line_fit_point_index]
                                .y_value();

                    cluster_image[(y * width + x) * 3 + 0] = 255;
                    cluster_image[(y * width + x) * 3 + 1] = 0;
                    cluster_image[(y * width + x) * 3 + 2] = 0;

                    if (j < 10) {
                        cluster_image[(y * width + x) * 3 + ((i + 1) % 3)] =
                            255;
                    } else {
                        cluster_image[(y * width + x) * 3 + (i % 3)] = 255;
                    }
                }
            }
            stbi_write_png("peaks.png", width, height, 3, cluster_image,
                           width * 3);

            dumpExtentLikeToCSV(cluster_data_new_out, cluster_data_new_count,
                                "out7extents.csv");
        }

        zero_quads.wait();

        do_indexing(q, cluster_data_new_buffer, cluster_data_new_count,
                    compacted_corners, line_fit_points_buffer,
                    rewritten_filtered_values_buffer, output_quads);

        if (prog) {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);
            std::cout << "16: " << duration.count() << std::endl;
        }

        if (debug) {
            std::vector<FittedQuad> t;
            auto rewritten_filtered_out = new ClusterExtents[width * height * 4]();
            auto cluster_point_out = new ClusterPoint[width * height * 4]();
            auto quads_out = new FittedQuad[width * height]();
            q.copy(rewritten_filtered_values_buffer, rewritten_filtered_out, std::distance(filtered_values_buffer, filtered_values_end));
            q.copy(filtered_cluster_points, cluster_point_out, filtered_points_count);
            q.copy(output_quads, quads_out, cluster_data_new_count);
            q.wait();
            
            uint8_t *cluster_image = new uint8_t[width * height * 3]();
            for (int i = 0; i < cluster_data_new_count; i++) {
                const auto &quad = quads_out[i].indices;
                const auto &extents = rewritten_filtered_out[i];
                std::cout << i << " " << quad[0] << " " << quad[1] << " "
                          << quad[2] << " " << quad[3] << std::endl;

                if (quad[0] != 0 || quad[1] != 0 || quad[2] != 0 ||
                    quad[3] != 0) {
                    for (int i = 0; i < 4; i++) {
                        auto x =
                            cluster_point_out[extents.start + quad[i]]
                                .x_value();
                        auto y =
                            cluster_point_out[extents.start + quad[i]]
                                .y_value();

                        auto xx = cluster_point_out[extents.start +
                                                          quad[(i + 1) % 4]]
                                      .x_value();
                        auto yy = cluster_point_out[extents.start +
                                                          quad[(i + 1) % 4]]
                                      .y_value();

                        image_u8_draw_line(cluster_image, x, y, xx, yy, width,
                                           height);
                    }

                    t.push_back(quads_out[i]);
                }
            }
            stbi_write_png("quad.png", width, height, 3, cluster_image,
                           width * 3);

            process_fitted_quads(t, width, height, data, debug);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << duration.count() << std::endl;
    }
}

void decode_quads(const QuadCorners *corner_data, size_t corner_data_length,
                            size_t width, size_t height, uint8_t *greyscaled, bool debug);

void process_fitted_quads(const std::vector<FittedQuad>& fit_quads_host_, size_t width, size_t height, uint8_t *greyscaled, bool debug) {
    std::vector<QuadCorners> quad_corners_host_;
    int min_tag_width_ = 8;
    bool reversed_border_ = false;
        double cos_critical_rad = std::cos(10 * M_PI / 180);

        int i = 0;

  for (const FittedQuad &quad : fit_quads_host_) {
    if (!(quad.indices[0] != 0 || quad.indices[1] != 0 || quad.indices[2] != 0 ||
                    quad.indices[3] != 0)) {
      continue;
    }
    QuadCorners corners;
    // corners.blob_index = quad.blob_index;
    corners.reversed_border = reversed_border_;

    float lines[4][4];
    for (int i = 0; i < 4; i++) {
      float err;
      float mse;
      fit_line(quad.moments[i], quad.num_in_moments[i], lines[i], &err, &mse);
    }

    bool bad_determinant = false;
    for (int i = 0; i < 4; i++) {
      // solve for the intersection of lines (i) and (i+1)&3.
      // p0 + lambda0*u0 = p1 + lambda1*u1, where u0 and u1
      // are the line directions.
      //
      // lambda0*u0 - lambda1*u1 = (p1 - p0)
      //
      // rearrange (solve for lambdas)
      //
      // [u0_x   -u1_x ] [lambda0] = [ p1_x - p0_x ]
      // [u0_y   -u1_y ] [lambda1]   [ p1_y - p0_y ]
      //
      // remember that lines[i][0,1] = p, lines[i][2,3] = NORMAL vector.
      // We want the unit vector, so we need the perpendiculars. Thus, below
      // we have swapped the x and y components and flipped the y components.

      double A00 = lines[i][3], A01 = -lines[(i + 1) & 3][3];
      double A10 = -lines[i][2], A11 = lines[(i + 1) & 3][2];
      double B0 = -lines[i][0] + lines[(i + 1) & 3][0];
      double B1 = -lines[i][1] + lines[(i + 1) & 3][1];

      double det = A00 * A11 - A10 * A01;

      // inverse.
      double W00 = A11 / det, W01 = -A01 / det;
      if (fabs(det) < 0.001) {
        bad_determinant = true;
        break;
      }

      // solve
      double L0 = W00 * B0 + W01 * B1;

      // compute intersection
      corners.corners[i][0] = lines[i][0] + L0 * A00;
      corners.corners[i][1] = lines[i][1] + L0 * A10;
    }
    if (bad_determinant) {
      continue;
    }

    {
      float area = 0;

      // get area of triangle formed by points 0, 1, 2, 0
      float length[3], p;
      for (int i = 0; i < 3; i++) {
        int idxa = i;            // 0, 1, 2,
        int idxb = (i + 1) % 3;  // 1, 2, 0
        length[i] =
            hypotf((corners.corners[idxb][0] - corners.corners[idxa][0]),
                   (corners.corners[idxb][1] - corners.corners[idxa][1]));
      }
      p = (length[0] + length[1] + length[2]) / 2;

      area += sqrtf(p * (p - length[0]) * (p - length[1]) * (p - length[2]));

      // get area of triangle formed by points 2, 3, 0, 2
      for (int i = 0; i < 3; i++) {
        int idxs[] = {2, 3, 0, 2};
        int idxa = idxs[i];
        int idxb = idxs[i + 1];
        length[i] =
            hypotf((corners.corners[idxb][0] - corners.corners[idxa][0]),
                   (corners.corners[idxb][1] - corners.corners[idxa][1]));
      }
      p = (length[0] + length[1] + length[2]) / 2;

      area += sqrtf(p * (p - length[0]) * (p - length[1]) * (p - length[2]));

      if (area < 0.95 * min_tag_width_ * min_tag_width_) {
        continue;
      }
    }

    // reject quads whose cumulative angle change isn't equal to 2PI
    {
      bool reject_corner = false;
      for (int i = 0; i < 4; i++) {
        int i0 = i, i1 = (i + 1) & 3, i2 = (i + 2) & 3;

        float dx1 = corners.corners[i1][0] - corners.corners[i0][0];
        float dy1 = corners.corners[i1][1] - corners.corners[i0][1];
        float dx2 = corners.corners[i2][0] - corners.corners[i1][0];
        float dy2 = corners.corners[i2][1] - corners.corners[i1][1];
        float cos_dtheta =
            (dx1 * dx2 + dy1 * dy2) /
            sqrtf((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2));

        if (std::abs(cos_dtheta) > cos_critical_rad ||
            dx1 * dy2 < dy1 * dx2) {
          reject_corner = true;
          break;
        }
      }
      if (reject_corner) {
        continue;
      }
    }
    {
        quad_corners_host_.push_back(corners);
    }
  }

  decode_quads(quad_corners_host_.data(), quad_corners_host_.size(), width, height, greyscaled, debug);
}
