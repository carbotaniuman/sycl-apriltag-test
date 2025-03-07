#include "find_boundaries.h"
#include "label_compacter.h"
#include "segmentation.h"

// This implementation of finding boundaries is similar to the
// one AprilTag uses, with some modifications made to facilitate
// efficient GPU computation. Of particular note is the output
// format. Instead of using a hashmap, the labels are stored
// in a parallel array.

sycl::event find_boundaries(sycl::queue &q, const uint32_t *labels,
                            const uint32_t *sizes, BoundaryPoint *points,
                            uint32_t *blob_labels, uint64_t *compacter_buffer,
                            size_t width, size_t height,
                            const std::vector<sycl::event> &deps) {
    constexpr size_t block_height = 8;
    constexpr size_t block_height_work_done = block_height - 1;
    constexpr size_t block_width = 32;
    constexpr size_t block_width_work_done = block_width - 2;
    // The boundaries operation is done in blocks, where the rows on either side
    // and the columns at the bottom are ignored. This means we need to dispatch
    // additional dimensions in order to get enough work items to process our image.
    size_t dispatch_height = (height + block_height_work_done - 1) / block_height_work_done * block_height;
    size_t dispatch_width = (width + block_width_work_done - 1) / block_width_work_done * block_width;
    
    auto init_event = q.submit([=](sycl::handler &h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 2> shared_labels{sycl::range(block_height, block_width), h};

        h.parallel_for(
            sycl::nd_range(sycl::range(dispatch_height, dispatch_width),
                           sycl::range(block_height, block_width)),
            [=](sycl::nd_item<2> it) {
                LabelCompacter compacter{compacter_buffer};
                // Derived x' = x - a*(x//b) - o
                // x = global index, x' = memory access index
                // a = block overlap size, b = block size
                size_t x = it.get_group(1) * (it.get_local_range(1) - 2) +
                           it.get_local_id(1);
                size_t y = it.get_group(0) * (it.get_local_range(0) - 1) +
                           it.get_local_id(0);
                
                size_t linear_id = y * width + x;

                size_t local_linear_id = it.get_local_linear_id();
                size_t local_width = it.get_local_range(1);

                auto shared_label_ptr = shared_labels.get_pointer();
                uint32_t local_label_img;

                // Bog standards bounds check.
                if (x >= width || y >= height) {
                    local_label_img = shared_label_ptr[local_linear_id] = 0;
                } else {
                    local_label_img = shared_label_ptr[local_linear_id] =
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
                // y only overlaps on the bottom.
                if (size_t local_y = it.get_local_id(0);
                    local_y == it.get_local_range(0) - 1) {
                    return;
                }

                if (x >= width || y >= height) {
                    return;
                }

                uint32_t local_label = local_label_img & LABEL_VALUE_MASK;

                bool local_label_too_small = sizes[local_label] < 25;
                uint32_t test_local = local_label_img & LABEL_PIXEL_MASK;

                auto handle_pixel_test = [&](int pixel_dy, int pixel_dx,
                                             size_t point_offset,
                                             HalfPixel half_pixel) {
                    uint32_t test_label_img =
                        shared_label_ptr[local_linear_id + pixel_dy * local_width +
                                   pixel_dx];

                    uint32_t test_other = test_label_img & LABEL_PIXEL_MASK;

                    if (test_other != test_local) {
                        uint32_t test_label = test_label_img & LABEL_VALUE_MASK;

                        bool test_label_too_small =
                            sizes[test_label] < 25;

                        BoundaryPoint maybe{
                            pack_half_pixel(x, half_pixel),
                            // Really I want a sycl::select but that's not
                            // supported.
                            (test_other > test_local)
                                ? static_cast<uint16_t>(
                                      (COORDINATE_COLOR_DIRECTION_MASK | y))
                                : static_cast<uint16_t>(y),                        
                        };

                        uint64_t extended_blob_label = 
                            (
                                (static_cast<uint64_t>(sycl::min(local_label, test_label)) << 32) |
                                static_cast<uint64_t>(sycl::max(local_label, test_label))
                            );

                        // Supposedly this let's us do coalesced memory stores
                        // which is super cool and performant.
                        size_t output_index =
                            width * height * point_offset + linear_id;

                        if (!local_label_too_small && !test_label_too_small) {
                            points[output_index] = maybe;
                            blob_labels[output_index] = compacter.lookup<sycl::memory_scope::device>(extended_blob_label);
                        }
                    }
                };

                handle_pixel_test(0, 1, 0, HalfPixel::RIGHT);
                // `BOTTOM_RIGHT` on the left pixel and `BOTTOM_LEFT`
                // on our pixel will result in duplicates - let's check
                // for this case and ignore them. For some reason I can't
                // understand, this is really important and results in 2-wide
                // borders otherwise, which messes with the quad detection.
                {
                    uint32_t left_label_img = shared_label_ptr[local_linear_id - 1];
                    uint32_t test_left = left_label_img & LABEL_PIXEL_MASK;
                    uint32_t down_label_img =
                        shared_label_ptr[local_linear_id + local_width];
                    uint32_t test_down = down_label_img & LABEL_PIXEL_MASK;

                    bool would_duplicate =
                        test_left != test_down &&
                        sizes[left_label_img & LABEL_VALUE_MASK] >= 25 &&
                        sizes[down_label_img & LABEL_VALUE_MASK] >= 25;
                    if (!would_duplicate) {
                        handle_pixel_test(1, -1, 1, HalfPixel::BOTTOM_LEFT);
                    }
                }
                handle_pixel_test(1, 0, 2, HalfPixel::BOTTOM);
                handle_pixel_test(1, 1, 3, HalfPixel::BOTTOM_RIGHT);
            });
    });

    return init_event;
}
