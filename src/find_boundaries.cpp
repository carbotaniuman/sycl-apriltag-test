#include "find_boundaries.h"
#include "segmentation.h"

// This implementation of finding boundaries is similar to the
// one AprilTag uses, with some modifications made to facilitate
// efficient GPU computation. Of particular note is the output
// format. Instead of using a hashmap, the labels are concatenated
// along with the data.

sycl::event find_boundaries(sycl::queue &q, const uint32_t *labels,
                            const HashTable::Entry *sizes, size_t sizes_elems,
                            BoundaryPoint *points, size_t width, size_t height,
                            const std::vector<sycl::event> &deps) {
    // The boundaries operation is done in blocks of 4 wide x 3 high, where only
    // the bottom-middle 2 x 2 blocks are actually updated - this reduces extra
    // memory reads by a factor of 2 compaired to the naive approach.
    //
    // This means we need 2 times the width and 1.5 times the height of work
    // items. Mapping this back to the proper indices is the responsibility of
    // user code.
    auto init_event = q.submit([=](sycl::handler &h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 2> shared_labels{sycl::range(3, 4), h};

        h.parallel_for(
            sycl::nd_range(sycl::range(height * 3 / 2, width * 2),
                           sycl::range(3, 4)),
            [=](sycl::nd_item<2> it) {
                // Derived x' = x - a*(x//b) - o
                // x = global index, x' = memory access index
                // a = block overlap size, b = block size
                size_t x = it.get_group(1) * (it.get_local_range(1) - 2) +
                           it.get_local_id(1) - 1;
                size_t y = it.get_group(0) * (it.get_local_range(0) - 1) +
                           it.get_local_id(0) - 1;
                size_t linear_id = y * width + x;

                size_t local_linear_id = it.get_local_linear_id();
                size_t local_width = it.get_local_range(1);

                auto shared_ptr = shared_labels.get_pointer();
                uint32_t local_label_img;

                // Bog standards bounds check.
                if (x < 0 || x >= width || y < 0 || y >= width) {
                    local_label_img = shared_ptr[local_linear_id] = 0;
                } else {
                    local_label_img = shared_ptr[local_linear_id] =
                        labels[linear_id];
                }

                it.barrier(sycl::access::fence_space::local_space);

                // This is similar to BKE in that a block of 16 pixels is
                // loaded, but because we are doing this per work-item now we
                // have to early exit some of them. We can probably do this
                // better by leveraging subgroups or something, but I'm not
                // sure.

                // x has overlap on both sides.
                if (size_t local_x = it.get_local_id(1);
                    local_x == 0 || local_x == it.get_local_range(1) - 1) {
                    return;
                }
                // y only overlaps on top.
                if (size_t local_y = it.get_local_id(0); local_y == 0) {
                    return;
                }

                uint16_t local_label =
                    static_cast<uint16_t>(local_label_img & LABEL_VALUE_MASK);

                bool local_label_too_small = sizes[local_label].value < 25;

                auto handle_pixel_test = [&](int pixel_dy, int pixel_dx,
                                             size_t point_offset,
                                             HalfPixel half_pixel) {
                    uint32_t test_label_img =
                        shared_ptr[local_linear_id + pixel_dy * local_width +
                                   pixel_dx];

                    uint32_t test_local = local_label_img & LABEL_PIXEL_MASK;
                    uint32_t test_other = test_label_img & LABEL_PIXEL_MASK;

                    if (test_other != test_local) {
                        uint16_t test_label = static_cast<uint16_t>(
                            test_label_img & LABEL_VALUE_MASK);

                        bool test_label_too_small =
                            sizes[test_label].value < 25;

                        BoundaryPoint maybe{
                            pack_half_pixel(x, half_pixel),
                            // Really I want a sycl::select but that's not
                            // supported.
                            (test_other > test_local)
                                ? static_cast<uint16_t>(
                                      (COORDINATE_COLOR_DIRECTION_MASK | y))
                                : static_cast<uint16_t>(y),
                            sycl::min(local_label, test_label),
                            sycl::max(local_label, test_label),
                        };

                        // Supposedly this let's us do coalesced memory stores
                        // which is super cool and performant.
                        size_t output_index =
                            width * height * point_offset + linear_id;

                        if (!local_label_too_small && !test_label_too_small) {
                            points[output_index] = maybe;
                        }
                    }
                };

                handle_pixel_test(-1, -1, 0, HalfPixel::TOP_LEFT);
                handle_pixel_test(-1, 0, 1, HalfPixel::TOP);
                handle_pixel_test(0, -1, 2, HalfPixel::LEFT);
                handle_pixel_test(1, -1, 3, HalfPixel::BOTTOM_LEFT);
            });
    });

    return init_event;
}