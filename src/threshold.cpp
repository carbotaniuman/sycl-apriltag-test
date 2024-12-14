#include "threshold.h"

sycl::event threshold_image(sycl::queue &q, const uint8_t *grayscale,
                            sycl::vec<uint8_t, 2> *extrema,
                            uint8_t *thresholded, size_t width, size_t height,
                            const std::vector<sycl::event> &deps) {
    auto minmax_event = q.parallel_for(
        sycl::range(height / 4, width / 4), deps, [=](sycl::item<2> it) {
            size_t width = it.get_range(1);
            size_t height = it.get_range(0);

            size_t x = it.get_id(1);
            size_t y = it.get_id(0);

            size_t linear_id = it.get_linear_id();

            uint8_t cur_min = 255;
            uint8_t cur_max = 0;

#pragma unroll
            for (size_t i = 0; i < 4; i++) {
#pragma unroll
                for (size_t j = 0; j < 4; j++) {
                    uint8_t val =
                        grayscale[(y * 4 + i) * (width * 4) + (x * 4 + j)];
                    cur_min = sycl::min(cur_min, val);
                    cur_max = sycl::max(cur_max, val);
                }
            }

            extrema[linear_id] = {cur_min, cur_max};
        });

    auto thres_event = q.parallel_for(
        sycl::range(height, width), minmax_event, [=](sycl::item<2> it) {
            size_t width = it.get_range(1);
            size_t height = it.get_range(0);

            size_t x = it.get_id(1);
            size_t y = it.get_id(0);

            size_t linear_id = it.get_linear_id();

            uint8_t cur_min = 255;
            uint8_t cur_max = 0;

#pragma unroll
            for (int i = -1; i <= 1; i++) {
#pragma unroll
                for (int j = -1; j <= 1; j++) {
                    int extrema_x = x / 4 + i;
                    int extrema_y = y / 4 + j;
                    if (extrema_x < 0 || extrema_x >= width / 4 ||
                        extrema_y < 0 || extrema_y >= height / 4) {
                        continue;
                    }
                    sycl::vec<uint8_t, 2> val =
                        extrema[extrema_y * (width / 4) + extrema_x];
                    cur_min = sycl::min(cur_min, val.x());
                    cur_max = sycl::max(cur_max, val.y());
                }
            }

            cur_min = sycl::min(cur_min, cur_max);

            uint8_t val = grayscale[linear_id];
            uint8_t comp = cur_min + (cur_max - cur_min) / 2;

            if (cur_max - cur_min < 5) {
                thresholded[linear_id] = 127;
            } else {
                if (val > comp) {
                    thresholded[linear_id] = 255;
                } else {
                    thresholded[linear_id] = 0;
                }
            }
        });

    return thres_event;
}
