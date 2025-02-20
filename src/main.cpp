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

void dumpBoundaryPointsToCSV(const BoundaryPoint *boundaryPoints, size_t size,
                             const std::string &filename) {
    // Open the CSV file for writing
    std::ofstream csvFile(filename);

    if (!csvFile.is_open()) {
        std::cerr << "Failed to open file for writing.\n";
        return;
    }

    // Write the CSV header
    csvFile
        << "x_value,y_value,first_blob,second_blob,is_black_to_white,dx,dy\n";

    // Loop over the array and write each point to the CSV
    for (size_t i = 0; i < size; ++i) {
        const auto &point = boundaryPoints[i];
        csvFile << point.x_value() << "," << point.y_value() << ","
                << point.first_blob << "," << point.second_blob << ","
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
    bool debug = false;
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

    for (int i = 0; i < 10; i++) {
        int width, height, comp;
        stbi_uc *data =
            stbi_load("../decimate2.png", &width, &height, &comp, STBI_grey);
        fprintf(stdout, "width: %d, height: %d, comp: %d\n", width, height,
                comp);

        auto grayscale_buffer = sycl::malloc_shared<uint8_t>(width * height, q);
        auto extrema_buffer = sycl::malloc_shared<sycl::vec<uint8_t, 2>>(
            width / 4 * height / 4, q);
        auto thresholded_buffer =
            sycl::malloc_shared<uint8_t>(width * height, q);

        auto scratch_label_buffer = sycl::malloc_shared<uint32_t>(width * height, q);
        auto label_buffer = sycl::malloc_shared<uint16_t>(width * height, q);
        size_t sizes_elems = 1 << 16;
        auto sizes_buffer =
            sycl::malloc_shared<HashTable::Entry>(sizes_elems, q);

        auto points_buffer =
            sycl::malloc_shared<BoundaryPoint>(width * height * 4, q);

        auto compacted_points =
            sycl::malloc_shared<BoundaryPoint>(width * height * 4, q);

        size_t *compacted_points_count_ptr = sycl::malloc_shared<size_t>(1, q);

        auto trash_keys_buffer = sycl::malloc_shared<uint32_t>(1 << 16, q);
        auto values_buffer = sycl::malloc_shared<ClusterBounds>(1 << 16, q);
        auto filtered_values_buffer =
            sycl::malloc_shared<ClusterBounds>(1 << 16, q);
        auto filtered_cluster_indexes =
            sycl::malloc_shared<uint16_t>(width * height * 4, q);
        auto filtered_cluster_points =
            sycl::malloc_shared<ClusterPoint>(width * height * 4, q);
        auto rewritten_filtered_values_buffer =
            sycl::malloc_shared<ClusterExtents>(1 << 16, q);

        auto pre_line_fit_points_buffer =
            sycl::malloc_shared<LineFitPoint>(width * height * 4, q);

        auto line_fit_points_buffer =
            sycl::malloc_shared<LineFitPoint>(width * height * 4, q);
        auto found_corners_buffer =
            sycl::malloc_shared<Corner>(width * height * 4, q);
        auto compacted_corners =
            sycl::malloc_shared<Corner>(width * height * 4, q);
        auto cluster_data_new_buffer =
            sycl::malloc_shared<PeakExtents>(width * height, q);
        auto output_quads = sycl::malloc_shared<FittedQuad>(width * height, q);

        auto start = std::chrono::high_resolution_clock::now();

        auto copy_image = q.copy(data, grayscale_buffer, width * height);

        auto threshold =
            threshold_image(q, grayscale_buffer, extrema_buffer,
                            thresholded_buffer, width, height, {copy_image});
        threshold.wait();

        if (prog) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "1: " << duration.count() << std::endl;
        }

        if (debug) {
            auto output = new uint8_t[width * height];

            q.copy(thresholded_buffer, output, width * height, threshold);
            q.wait();

            stbi_write_png("thresholded.png", width, height, 1, output,
                           width * 1);
        }

        auto zero_labels =
            q.memset(label_buffer, 0, width * height * sizeof(uint16_t));
        auto zero_sizes =
            q.memset(sizes_buffer, 0, sizes_elems * sizeof(HashTable::Entry));

        auto segment = image_segmentation(
            q, thresholded_buffer, scratch_label_buffer, label_buffer, sizes_buffer, sizes_elems,
            width, height, {threshold, zero_labels, zero_sizes});
        segment.wait();

        if (prog) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "2: " << duration.count() << std::endl;
        }

        if (debug) {
            auto labels_out = new uint16_t[width * height];
            auto sizes_out = new HashTable::Entry[sizes_elems];

            q.copy(label_buffer, labels_out, width * height, segment);
            q.copy(sizes_buffer, sizes_out, sizes_elems, segment);
            q.wait();

            uint32_t *colors = new uint32_t[width * height];
            uint8_t *images = new uint8_t[width * height * 3];

            srand(555);

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

            stbi_write_png("segmented.png", width, height, 3, images,
                           width * 3);
        }

        auto zero_points = q.memset(points_buffer, 0,
                                    width * height * 4 * sizeof(BoundaryPoint));

        auto boundaries = find_boundaries(q, label_buffer, sizes_buffer,
                                          1 << 16, points_buffer, width, height,
                                          {segment, zero_points});
        boundaries.wait();

        if (prog) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "3: " << duration.count() << std::endl;
        }

        if (debug) {
            auto points_out = new BoundaryPoint[width * height * 4]();
            q.copy(points_buffer, points_out, width * height * 4, boundaries);
            q.wait();

            size_t present = 0;
            size_t zeroes = 0;
            for (size_t i = 0; i < width * height * 4; i++) {
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

            uint8_t *cluster_image = new uint8_t[width * height * 3]();
            for (size_t i = 0; i < width * height * 4; i++) {
                if (points_out[i] ==
                    sycl::bit_cast<BoundaryPoint>(static_cast<uint64_t>(0))) {
                    continue;
                }

                auto x = points_out[i].x_value();
                auto y = points_out[i].y_value();
                auto label = points_out[i].blob_label();

                cluster_image[(y * width + x) * 3 + 0] = 0;
                cluster_image[(y * width + x) * 3 + 1] = 0;
                cluster_image[(y * width + x) * 3 + 2] = 0;

                cluster_image[(y * width + x) * 3 + (label % 3)] = 255;
            }
            stbi_write_png("points.png", width, height, 3, cluster_image,
                           width * 3);
        }

        auto compacted_points_end = oneapi::dpl::copy_if(
            policy_e, points_buffer, points_buffer + width * height * 4,
            compacted_points, [](BoundaryPoint p) {
                return p !=
                       sycl::bit_cast<BoundaryPoint>(static_cast<uint64_t>(0));
            });

        size_t compacted_points_count =
            std::distance(compacted_points, compacted_points_end);
        if (prog) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "4: " << duration.count() << std::endl;
        }

        if (debug) {
            dumpBoundaryPointsToCSV(compacted_points, compacted_points_count,
                                    "out1.csv");

            uint8_t *cluster_image = new uint8_t[width * height * 3]();
            for (size_t i = 0; i < compacted_points_count; i++) {
                if (compacted_points[i] ==
                    sycl::bit_cast<BoundaryPoint>(static_cast<uint64_t>(0))) {
                    break;
                }

                auto x = compacted_points[i].x_value();
                auto y = compacted_points[i].y_value();
                auto label = compacted_points[i].blob_label();

                cluster_image[(y * width + x) * 3 + 0] = 0;
                cluster_image[(y * width + x) * 3 + 1] = 0;
                cluster_image[(y * width + x) * 3 + 2] = 0;

                cluster_image[(y * width + x) * 3 + (label % 3)] = 255;
            }
            stbi_write_png("points1.png", width, height, 3, cluster_image,
                           width * 3);
        }

        oneapi::dpl::sort(policy_e, compacted_points,
                          compacted_points + compacted_points_count,
                          [](const auto &left, const auto &right) {
                              return left.blob_label() < right.blob_label();
                          });

        if (prog) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "5: " << duration.count() << std::endl;
        }

        if (debug) {
            dumpBoundaryPointsToCSV(compacted_points, compacted_points_count,
                                    "out2.csv");

            auto points_out = new BoundaryPoint[compacted_points_count]();
            q.copy(compacted_points, points_out, compacted_points_count);
            q.wait();

            uint8_t *cluster_image = new uint8_t[width * height * 3]();
            std::unordered_map<uint32_t, uint32_t> vs{};
            for (size_t i = 0; i < compacted_points_count; i++) {
                if (points_out[i] ==
                    sycl::bit_cast<BoundaryPoint>(static_cast<uint64_t>(0))) {
                    std::cout << "???" << std::endl;
                    break;
                }

                uint32_t label = points_out[i].blob_label();

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

        auto transform_keys = dpl::make_transform_iterator(
            compacted_points, [](BoundaryPoint a) { return a.blob_label(); });

        auto transform_values = dpl::make_transform_iterator(
            dpl::make_zip_iterator(compacted_points,
                                   dpl::counting_iterator<uint32_t>(0)),
            [](auto a) {
                return ClusterBounds::inital_from_point(std::get<0>(a),
                                                        std::get<1>(a));
            });

        auto values_start = values_buffer;
        auto [keys_end, values_end] = oneapi::dpl::reduce_by_segment(
            policy_e, transform_keys, transform_keys + compacted_points_count,
            transform_values, trash_keys_buffer, values_start,
            std::equal_to<>(), [](const auto &left, const auto &right) {
                return reduce_bounds(left, right);
            });

        if (prog) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "6: " << duration.count() << std::endl;
        }

        if (debug) {
            dumpClusterBoundsToCSV(values_start,
                                   std::distance(values_start, values_end),
                                   "out2bounds.csv");
        }

        if (false) {
            std::filesystem::create_directories("clusters");
            for (size_t i = 0; i < std::distance(values_start, values_end);
                 i++) {
                uint8_t *cluster_image = new uint8_t[width * height * 3]();
                auto v = values_start[i];
                for (size_t j = v.start; j < v.start + v.count; j++) {
                    auto x = compacted_points[j].x_value();
                    auto y = compacted_points[j].y_value();
                    auto label = compacted_points[j].blob_label();

                    cluster_image[(y * width + x) * 3 + 0] = 0;
                    cluster_image[(y * width + x) * 3 + 1] = 0;
                    cluster_image[(y * width + x) * 3 + 2] = 0;

                    cluster_image[(y * width + x) * 3 + (label % 3)] = 255;
                }

                auto x = static_cast<int>(v.cx() / 2);
                auto y = static_cast<int>(v.cy() / 2);

                cluster_image[(y * width + x) * 3 + 0] = 255;
                cluster_image[(y * width + x) * 3 + 1] = 255;
                cluster_image[(y * width + x) * 3 + 2] = 255;

                auto x_min = static_cast<int>(v.x_min / 2);
                auto y_min = static_cast<int>(v.y_min / 2);
                auto x_max = static_cast<int>(v.x_max / 2);
                auto y_max = static_cast<int>(v.y_max / 2);

                for (int a = x_min; a <= x_max; a++) {
                    {
                        int yy = y_min;
                        cluster_image[(yy * width + a) * 3 + 0] = 255;
                        cluster_image[(yy * width + a) * 3 + 1] = 255;
                        cluster_image[(yy * width + a) * 3 + 2] = 255;
                    }

                    {
                        int yy = y_max;
                        cluster_image[(yy * width + a) * 3 + 0] = 255;
                        cluster_image[(yy * width + a) * 3 + 1] = 255;
                        cluster_image[(yy * width + a) * 3 + 2] = 255;
                    }
                }

                for (int a = y_min; a <= y_max; a++) {
                    {
                        int xx = x_min;
                        cluster_image[(a * width + xx) * 3 + 0] = 255;
                        cluster_image[(a * width + xx) * 3 + 1] = 255;
                        cluster_image[(a * width + xx) * 3 + 2] = 255;
                    }

                    {
                        int xx = x_max;
                        cluster_image[(a * width + xx) * 3 + 0] = 255;
                        cluster_image[(a * width + xx) * 3 + 1] = 255;
                        cluster_image[(a * width + xx) * 3 + 2] = 255;
                    }
                }
                char buffer[50];
                sprintf(buffer, "clusters/cluster%zu.png", i);
                stbi_write_png(buffer, width, height, 3, cluster_image,
                               width * 3);
                delete[] cluster_image;
            }
        }

        auto valid_blob_filter = ValidBlobFilter();

        auto filtered_values_end = oneapi::dpl::copy_if(
            policy_e, values_buffer,
            values_buffer + std::distance(values_start, values_end),
            filtered_values_buffer, valid_blob_filter);

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
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "7: " << duration.count() << std::endl;
        }

        auto filtered_points_count =
            std::distance(o_zipped_iterator, o_zipped_end);

        if (debug) {
            dumpClusterPointsToCSV(filtered_cluster_points,
                                   filtered_points_count, "out3.csv");
            dumpPlainToCSV(filtered_cluster_indexes, filtered_points_count,
                           "out4.csv");
        }

        auto transformed_extents_iter = dpl::make_transform_iterator(
            filtered_values_buffer,
            [](const auto &a) { return ClusterExtents{0, a.count}; });

        oneapi::dpl::inclusive_scan(
            policy_e, transformed_extents_iter,
            transformed_extents_iter +
                std::distance(filtered_values_buffer, filtered_values_end),
            rewritten_filtered_values_buffer,
            [](const ClusterExtents &left, const ClusterExtents &right) {
                return ClusterExtents{left.start + left.count, right.count};
            },
            ClusterExtents{0, 0});

        if (prog) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "8: " << duration.count() << std::endl;
        }

        if (debug) {
            dumpExtentLikeToCSV(
                rewritten_filtered_values_buffer,
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
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "9: " << duration.count() << std::endl;
        }

        if (debug) {
            dumpClusterPointsToCSV(filtered_cluster_points,
                                   filtered_points_count, "out5.csv");
        }

        if (false) {
            uint8_t *big_image = new uint8_t[width * height * 3]();

            char buffer[50];
            for (int f = 0;
                 f < std::distance(filtered_values_buffer, filtered_values_end);
                 f++) {
                sprintf(buffer, "slopes/%d/", f);
                std::filesystem::create_directories(buffer);
                auto xt = rewritten_filtered_values_buffer[f];
                uint8_t *cluster_image = new uint8_t[width * height * 3]();

                std::cout << "start at " << xt.start << " with count "
                          << xt.count << std::endl;
                for (int i = xt.start; i < xt.start + xt.count; i++) {
                    int ttt = i - xt.start;
                    auto p = filtered_cluster_points[i];

                    auto x = p.x_value();
                    auto y = p.y_value();

                    cluster_image[(y * width + x) * 3 + 0] = 255;
                    cluster_image[(y * width + x) * 3 + 1] = 255;
                    cluster_image[(y * width + x) * 3 + 2] = 255;

                    big_image[(y * width + x) * 3 + (f % 3)] = 255;

                    if (ttt % 10 == 0 || ttt == xt.count - 1) {
                        sprintf(buffer, "slopes/%d/%d.png", f, ttt);
                        stbi_write_png(buffer, width, height, 3, cluster_image,
                                       width * 3);
                    }
                }
                delete[] cluster_image;
            }
            stbi_write_png("clusters_filtered.png", width, height, 3, big_image,
                           width * 3);
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
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "10: " << duration.count() << std::endl;
        }

        if (debug) {
            dumpLineFitPointsToCSV(pre_line_fit_points_buffer,
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
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "11: " << duration.count() << std::endl;
        }

        auto line_fit_points_count = std::distance(asdasd_begin, asdasd_end);

        if (debug) {
            dumpLineFitPointsToCSV(line_fit_points_buffer,
                                   line_fit_points_count, "out6b.csv");
        }

        // std::cout << "filtered count " << filtered_points_count
        //           << " line fit points count " << line_fit_points_count
        //           << std::endl;

        q.memset(found_corners_buffer, 0, width * height * sizeof(Corner))
            .wait();

        fit_lines(q, line_fit_points_buffer, filtered_cluster_indexes,
                  rewritten_filtered_values_buffer, filtered_points_count,
                  found_corners_buffer);

        if (prog) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "12: " << duration.count() << std::endl;
        }

        auto compacted_corners_end = oneapi::dpl::copy_if(
            policy_e, found_corners_buffer,
            found_corners_buffer + width * height * 4, compacted_corners,
            [](const Corner &p) { return p.error != 0; });

        if (prog) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "13: " << duration.count() << std::endl;
        }

        if (debug) {
            dumpCornerToCSV(
                compacted_corners,
                std::distance(compacted_corners, compacted_corners_end),
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
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "14: " << duration.count() << std::endl;
        }

        size_t compacted_corner_count =
            std::distance(compacted_corners, compacted_corners_end);

        if (debug) {
            dumpCornerToCSV(compacted_corners, compacted_corner_count,
                            "out8.csv");
        }

        // std::cout << "compacted corner distance is " <<
        // compacted_corner_count
        //           << std::endl;

        auto transform_corner_keys = dpl::make_transform_iterator(
            compacted_corners, [](Corner a) { return a.cluster_index; });

        auto transform_corner_values = dpl::make_transform_iterator(
            oneapi::dpl::counting_iterator<uint32_t>(0),
            [](uint32_t a) { return PeakExtents{a, 1}; });

        auto [corner_keys_end, corner_values_end] =
            oneapi::dpl::reduce_by_segment(
                policy_e, transform_corner_keys,
                transform_corner_keys + compacted_corner_count,
                transform_corner_values, trash_keys_buffer,
                cluster_data_new_buffer, std::equal_to<>(),
                [](const auto &left, const auto &right) {
                    return reduce_extents(left, right);
                });

        if (prog) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "15: " << duration.count() << std::endl;
        }

        size_t cluster_data_new_count =
            std::distance(cluster_data_new_buffer, corner_values_end);

        if (debug) {
            uint8_t *cluster_image = new uint8_t[width * height * 3]();
            for (int j = 0; j < filtered_points_count; j++) {
                auto x = filtered_cluster_points[j].x_value();
                auto y = filtered_cluster_points[j].y_value();

                cluster_image[(y * width + x) * 3 + 0] = 255;
                cluster_image[(y * width + x) * 3 + 1] = 255;
                cluster_image[(y * width + x) * 3 + 2] = 255;
            }

            for (int i = 0; i < cluster_data_new_count; i++) {
                const auto &peak_extent = cluster_data_new_buffer[i];
                const auto &extents = rewritten_filtered_values_buffer[i];

                for (int j = 0; j < peak_extent.count; j++) {
                    const auto &corner_test =
                        compacted_corners[peak_extent.start + j];

                    auto x =
                        filtered_cluster_points
                            [extents.start + corner_test.line_fit_point_index]
                                .x_value();
                    auto y =
                        filtered_cluster_points
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

            dumpExtentLikeToCSV(cluster_data_new_buffer, cluster_data_new_count,
                                "out7extents.csv");
        }

        if (false) {
            uint8_t *big_image = new uint8_t[width * height * 3]();

            char buffer[50];
            for (int i = 0; i < cluster_data_new_count; i++) {
                sprintf(buffer, "peaks/%d/", i);
                std::filesystem::create_directories(buffer);

                const auto &peak_extent = cluster_data_new_buffer[i];
                const auto &xt = rewritten_filtered_values_buffer[i];

                uint8_t *cluster_image = new uint8_t[width * height * 3]();

                std::cout << "start at " << xt.start << " with count "
                          << xt.count << std::endl;
                for (int j = 0; j < peak_extent.count; j++) {
                    const auto &corner_test =
                        compacted_corners[peak_extent.start + j];

                    auto x = filtered_cluster_points
                                 [xt.start + corner_test.line_fit_point_index]
                                     .x_value();
                    auto y = filtered_cluster_points
                                 [xt.start + corner_test.line_fit_point_index]
                                     .y_value();

                    if (j < 10) {
                        cluster_image[(y * width + x) * 3 + 0] = 255;
                        cluster_image[(y * width + x) * 3 + 1] = 255;
                        cluster_image[(y * width + x) * 3 + 2] = 255;
                    } else {
                        cluster_image[(y * width + x) * 3 + (i % 3)] = 255;
                    }

                    if (true) {
                        sprintf(buffer, "peaks/%d/%d.png", i, j);
                        stbi_write_png(buffer, width, height, 3, cluster_image,
                                       width * 3);
                    }
                }
                delete[] cluster_image;
            }
        }

        auto zero_quads =
            q.memset(output_quads, 0, width * height * sizeof(FittedQuad));
        zero_quads.wait();

        do_indexing(q, cluster_data_new_buffer, cluster_data_new_count,
                    compacted_corners, line_fit_points_buffer,
                    rewritten_filtered_values_buffer, output_quads);

        if (prog) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            std::cout << "16: " << duration.count() << std::endl;
        }

        if (debug) {
            uint8_t *cluster_image = new uint8_t[width * height * 3]();
            for (int i = 0; i < cluster_data_new_count; i++) {
                const auto &quad = output_quads[i].indices;
                const auto &extents = rewritten_filtered_values_buffer[i];
                std::cout << i << " " << quad[0] << " " << quad[1] << " "
                          << quad[2] << " " << quad[3] << std::endl;

                if (quad[0] != 0 || quad[1] != 0 || quad[2] != 0 ||
                    quad[3] != 0) {
                    for (int i = 0; i < 4; i++) {
                        auto x =
                            filtered_cluster_points[extents.start + quad[i]]
                                .x_value();
                        auto y =
                            filtered_cluster_points[extents.start + quad[i]]
                                .y_value();

                        auto xx = filtered_cluster_points[extents.start +
                                                          quad[(i + 1) % 4]]
                                      .x_value();
                        auto yy = filtered_cluster_points[extents.start +
                                                          quad[(i + 1) % 4]]
                                      .y_value();

                        image_u8_draw_line(cluster_image, x, y, xx, yy, width,
                                           height);
                    }
                }
            }
            stbi_write_png("quad.png", width, height, 3, cluster_image,
                           width * 3);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << duration.count() << std::endl;
    }

    std::cout << "ttt1" << std::endl;
}
