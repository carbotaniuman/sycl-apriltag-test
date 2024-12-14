#include "segmentation.h"
#include "union_find.h"

// This file inmplements Connected Component Labelling in order to find image
// regions (blobs) that are deeemd important.
//
// The following file is an implementation of the BKE algorithm proposed in this
// paper: Allegretti, Stefano, et al. “Optimized Block-Based Algorithms to Label
// Connected Components on GPUs.” IEEE Transactions on Parallel and Distributed
// Systems, vol. 31, no. 2, 1 Feb. 2020, pp. 423–438,
// https://doi.org/10.1109/tpds.2019.2934683.
//
// or via a direct link:
// https://iris.unimore.it/bitstream/11380/1179616/1/2018_TPDS_Optimized_Block_Based_Algorithms_to_Label_Connected_Components_on_GPUs.pdf
//
// Futher information on KE (of which BKE is derived from) can be found here.
// Komura, Yukihiro. “GPU-Based Cluster-Labeling Algorithm without the Use of
// Conventional Iteration: Application to the Swendsen–Wang Multi-Cluster Spin
// Flip Algorithm.” Computer Physics Communications, vol. 194, Sept. 2015, pp.
// 54–58, https://doi.org/10.1016/j.cpc.2015.04.015. Accessed 8 Feb. 2020.
//
// Unfortunately, said paper is not freely available.

// Given our block is C, our neighboring blocks are as follows:
//
//  P | Q | R
// ---+---+---
//  S | C | X
// ---+---+---
//  X | X | X
//
// X's are don't cares.
//
// Because AprilTag requires 2 different union-finds, one for 255s
// and one for 0s, we have had to duplicate this enum twice. We treat
// 255s as white and 0 as black. This involves changing the label memory
// layout a bit. Because we operate on blocks of 2x2 with our image
// dimensions required to be even, we can fit arrange the data as follows.
//
//  0 | 1
// ---+---
//  2 | 3
//
// Where `uint32_t` 0 contains the labels for the 255s, `uint32_t` 1 contains
// the labels for the 0s, and `uint32_t` 2 contains the information byte,
// which is really a `uint16_t`. `uint32_t` 3 is unused.
struct BkeBitmap {
    constexpr static uint16_t TOP_LEFT_255 = 1 << 0;
    constexpr static uint16_t TOP_RIGHT_255 = 1 << 1;
    constexpr static uint16_t BOTTOM_LEFT_255 = 1 << 2;
    constexpr static uint16_t BOTTOM_RIGHT_255 = 1 << 3;
    constexpr static uint16_t BITMASK_POS_255 = 0b1111;
    // This is unused in the original paper, but has been placed
    // here to allow some nicer code structure.
    constexpr static uint16_t MUST_UNION_P_255 = 1 << 4;
    constexpr static uint16_t MUST_UNION_Q_255 = 1 << 5;
    constexpr static uint16_t MUST_UNION_R_255 = 1 << 6;
    constexpr static uint16_t MUST_UNION_S_255 = 1 << 7;

    // Our own unique bits because we have to duplicate them.
    constexpr static uint16_t TOP_LEFT_0 = 1 << 8;
    constexpr static uint16_t TOP_RIGHT_0 = 1 << 9;
    constexpr static uint16_t BOTTOM_LEFT_0 = 1 << 10;
    constexpr static uint16_t BOTTOM_RIGHT_0 = 1 << 11;
    constexpr static uint16_t BITMASK_POS_0 = 0b1111 << 8;

    constexpr static uint16_t MUST_UNION_P_0 = 1 << 12;
    constexpr static uint16_t MUST_UNION_Q_0 = 1 << 13;
    constexpr static uint16_t MUST_UNION_R_0 = 1 << 14;
    constexpr static uint16_t MUST_UNION_S_0 = 1 << 15;
};

sycl::event
internal_compress_labels(sycl::queue &q, uint32_t *labels, size_t width,
                         size_t height,
                         const std::vector<sycl::event> &deps = {}) {
    return q.parallel_for(sycl::range(height / 2, width / 2), deps,
                          [labels](sycl::item<2> it) {
                              size_t width = it.get_range(1) * 2;
                              size_t height = it.get_range(0) * 2;

                              size_t x = it.get_id(1) * 2;
                              size_t y = it.get_id(0) * 2;

                              size_t image_linear_id = y * width + x;

                              UnionFind<sycl::memory_scope_device> uf{labels};
                              uf.find_compress(image_linear_id);
                              uf.find_compress(image_linear_id + 1);
                          });
}

// kernel args should half height and width of images:
// top left of each 2x2 block
sycl::event image_segmentation(sycl::queue &q, const uint8_t *thresholded,
                               uint32_t *label_scratch, uint32_t *labels,
                               uint32_t *sizes, size_t sizes_elem,
                               size_t width, size_t height,
                               const std::vector<sycl::event> &deps) {
    // This is the `INITIALIZATION` step of the BKE algorithm.
    auto init_event = q.parallel_for(
        sycl::range(height / 2, width / 2), deps,
        [thresholded, label_scratch](sycl::item<2> it) {
            size_t width = it.get_range(1) * 2;
            size_t height = it.get_range(0) * 2;

            size_t x = it.get_id(1) * 2;
            size_t y = it.get_id(0) * 2;

            size_t image_linear_id = y * width + x;

            // The bits in this byte correspond to `BkeBitmap`
            // and save us from having to refetch image data
            // a second time in future phases.
            uint16_t information_byte = 0;

            // The bits in this `uint16_t` represent
            // whether we need to check the pixels surrounding
            // us, where 5 is the top-left pixel of our block
            // and 10 is the bottom-right.
            //
            //  0 | 1 | 2 | 3
            // ---+---+---+---
            //  4 | 5 | 6 | 7
            // ---+---+---+---
            //  8 | 9 | 10| 11
            // ---+---+---+---
            //  12| 13| 14| 15
            //
            // We really only care about bits 0, 1, 2, 4, and 8,
            // but the original paper tracked the interior so
            // we will too. Despite that, we will ignore pixel 10
            // as that saves some lookups and won't matter.
            //
            // Although we have 2 sets to run, we use this for both
            // the 255s and the 0s.
            uint32_t pixels_to_check = 0;

            // Technically speaking this is not correct, AprilTag does
            // 8-connectivity on the 255s and only 4-connectivity on 0s.
            // This shouldn't matter that much, so just do 8-connectivity
            // on both.
            if (thresholded[image_linear_id] == 255) {
                pixels_to_check |= 0x777;
                information_byte |= BkeBitmap::TOP_LEFT_255;
            } else if (thresholded[image_linear_id] == 0) {
                pixels_to_check |= 0x777;
                information_byte |= BkeBitmap::TOP_LEFT_0;
            }

            if (thresholded[image_linear_id + 1] == 255) {
                pixels_to_check |= 0x777 << 1;
                information_byte |= BkeBitmap::TOP_RIGHT_255;
            } else if (thresholded[image_linear_id + 1] == 0) {
                pixels_to_check |= 0x777 << 1;
                information_byte |= BkeBitmap::TOP_RIGHT_0;
            }

            if (thresholded[image_linear_id + width] == 255) {
                pixels_to_check |= 0x777 << 4;
                information_byte |= BkeBitmap::BOTTOM_LEFT_255;
            } else if (thresholded[image_linear_id + width] == 0) {
                pixels_to_check |= 0x777 << 4;
                information_byte |= BkeBitmap::BOTTOM_LEFT_0;
            }

            if (thresholded[image_linear_id + width + 1] == 255) {
                information_byte |= BkeBitmap::BOTTOM_RIGHT_255;
            } else if (thresholded[image_linear_id + width + 1] == 0) {
                information_byte |= BkeBitmap::BOTTOM_RIGHT_0;
            }

            // now we need the bounds checks (not present in paper)
            // these magic bits mask out the portions of the
            // 16-pixel grid that is out of bounds.
            if (x == 0) {
                pixels_to_check &= 0xEEEE;
            } else if (x + 2 >= width) {
                pixels_to_check &= 0x7777;
            }

            if (y == 0) {
                pixels_to_check &= 0xFFF0;
            } else if (y + 2 >= height) {
                pixels_to_check &= 0x0FFF;
            }

            // Block-based komura equivalence needs the id of the first matching
            // block. Calculate the offset compared to our current block.
            int32_t label_offset_255 = 0;
            int32_t label_offset_0 = 1;

            auto handle_pixel_test = [&](int pixel_dy, int pixel_dx,
                                         uint8_t bit_index,
                                         uint8_t extra_information) {
                size_t test_index =
                    image_linear_id + pixel_dy * width + pixel_dx;
                if (pixels_to_check & (1 << bit_index)) {
                    uint8_t thres_pixel = thresholded[test_index];

                    if (thres_pixel == 255) {
                        if (label_offset_255 == 0) {
                            switch (extra_information) {
                            case BkeBitmap::MUST_UNION_P_255:
                                label_offset_255 = -2 * width - 2;
                                break;
                            case BkeBitmap::MUST_UNION_Q_255:
                                label_offset_255 = -2 * width;
                                break;
                            case BkeBitmap::MUST_UNION_R_255:
                                label_offset_255 = -2 * width + 2;
                                break;
                            case BkeBitmap::MUST_UNION_S_255:
                                label_offset_255 = -2;
                                break;
                            default:
                                break;
                            }
                        } else {
                            information_byte |= extra_information;
                        }
                    } else if (thres_pixel == 0) {
                        if (label_offset_0 == 1) {
                            // The labels for the 0s is one block to the right.
                            switch (extra_information) {
                            case BkeBitmap::MUST_UNION_P_255:
                                label_offset_0 = -2 * width - 2 + 1;
                                break;
                            case BkeBitmap::MUST_UNION_Q_255:
                                label_offset_0 = -2 * width + 1;
                                break;
                            case BkeBitmap::MUST_UNION_R_255:
                                label_offset_0 = -2 * width + 2 + 1;
                                break;
                            case BkeBitmap::MUST_UNION_S_255:
                                label_offset_0 = -2 + 1;
                                break;
                            default:
                                break;
                            }
                        } else {
                            // The information for the 0s is shifted 8 bits
                            // left.
                            information_byte |= extra_information << 8;
                        }
                    }
                }
            };

            handle_pixel_test(-1, -1, 0, BkeBitmap::MUST_UNION_P_255);
            handle_pixel_test(-1, 0, 1, BkeBitmap::MUST_UNION_Q_255);
            handle_pixel_test(-1, 1, 2, BkeBitmap::MUST_UNION_Q_255);
            handle_pixel_test(-1, 2, 3, BkeBitmap::MUST_UNION_R_255);
            handle_pixel_test(0, -1, 4, BkeBitmap::MUST_UNION_S_255);
            handle_pixel_test(1, -1, 8, BkeBitmap::MUST_UNION_S_255);

            label_scratch[image_linear_id] = image_linear_id + label_offset_255;
            label_scratch[image_linear_id + 1] =
                image_linear_id + label_offset_0;
            label_scratch[image_linear_id + width] =
                static_cast<uint32_t>(information_byte);
        });

    // Addtional premerge `COMPRESSION` step.
    auto pre_compression_event =
        internal_compress_labels(q, label_scratch, width, height, {init_event});

    // This is the `MERGE` step of the BKE algorithm.
    auto merge_event = q.parallel_for(
        sycl::range(height / 2, width / 2), pre_compression_event,
        [label_scratch](sycl::item<2> it) {
            size_t width = it.get_range(1) * 2;
            size_t height = it.get_range(0) * 2;

            size_t x = it.get_id(1) * 2;
            size_t y = it.get_id(0) * 2;

            size_t image_linear_id = y * width + x;

            // Both the 255s and the 0s are in the same union-find array
            // with spacing to keep them from accidentally being merged.
            UnionFind<sycl::memory_scope_device> uf{label_scratch};

            uint16_t information_byte =
                static_cast<uint16_t>(label_scratch[image_linear_id + width]);

            if (information_byte & BkeBitmap::MUST_UNION_Q_255) {
                uf.merge(image_linear_id, image_linear_id - 2 * width);
            }
            if (information_byte & BkeBitmap::MUST_UNION_R_255) {
                uf.merge(image_linear_id, image_linear_id - 2 * width + 2);
            }
            if (information_byte & BkeBitmap::MUST_UNION_S_255) {
                uf.merge(image_linear_id, image_linear_id - 2);
            }

            if (information_byte & BkeBitmap::MUST_UNION_Q_0) {
                uf.merge(image_linear_id + 1, image_linear_id - 2 * width + 1);
            }
            if (information_byte & BkeBitmap::MUST_UNION_R_0) {
                uf.merge(image_linear_id + 1, image_linear_id - 2 * width + 3);
            }
            if (information_byte & BkeBitmap::MUST_UNION_S_0) {
                uf.merge(image_linear_id + 1, image_linear_id - 1);
            }
        });

    // This is the `COMPRESSION` step of the BKE algorithm.
    auto compression_event = internal_compress_labels(q, label_scratch, width,
                                                      height, {merge_event});

    // This is the `FINAL_LABELLING` step of the BKE algorithm.
    // This is extended from traditional BKE in order to also keep
    // track of the amount of pixels in each label. This also allows
    // us to compact the label ids a little bit.
    auto final_labelling_event = q.parallel_for(
        sycl::range(height / 2, width / 2), compression_event,
        [label_scratch, labels, sizes, sizes_elem](sycl::item<2> it) {
            using atomic_elem_ref = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope_device>;
            size_t width = it.get_range(1) * 2;
            size_t height = it.get_range(0) * 2;

            size_t x = it.get_id(1) * 2;
            size_t y = it.get_id(0) * 2;

            size_t image_linear_id = y * width + x;
            size_t kernel_linear_id = it.get_linear_id();

            uint32_t label_255 = label_scratch[image_linear_id];
            uint32_t label_0 = label_scratch[image_linear_id + 1];
            uint16_t information_byte =
                static_cast<uint16_t>(label_scratch[image_linear_id + width]);

            uint32_t count_255 =
                sycl::popcount(information_byte & BkeBitmap::BITMASK_POS_255);
            atomic_elem_ref size_255_ref{sizes[label_255]};
            if (count_255 > 0) {
                size_255_ref.fetch_add(count_255);
            }
            
            uint32_t count_0 =
                sycl::popcount(information_byte & BkeBitmap::BITMASK_POS_0);
            atomic_elem_ref size_0_ref{sizes[label_0]};
            if (count_0 > 0) {
                size_0_ref.fetch_add(count_0);
            }

            uint32_t top_left, top_right;

            if (information_byte & BkeBitmap::TOP_LEFT_255) {
                top_left = LABEL_PIXEL_MASK | label_255;
            } else if (information_byte & BkeBitmap::TOP_LEFT_0) {
                top_left = label_0;
            } else {
                top_left = 0;
            }

            if (information_byte & BkeBitmap::TOP_RIGHT_255) {
                top_right = LABEL_PIXEL_MASK | label_255;
            } else if (information_byte & BkeBitmap::TOP_RIGHT_0) {
                top_right = label_0;
            } else {
                top_right = 0;
            }

            reinterpret_cast<sycl::vec<uint32_t, 2> *>(labels)[image_linear_id / 2] =
                sycl::vec(top_left, top_right);

            uint32_t bottom_left, bottom_right;

            if (information_byte & BkeBitmap::BOTTOM_LEFT_255) {
                bottom_left = LABEL_PIXEL_MASK | label_255;
            } else if (information_byte & BkeBitmap::BOTTOM_LEFT_0) {
                bottom_left = label_0;
            } else {
                bottom_left = 0;
            }

            if (information_byte & BkeBitmap::BOTTOM_RIGHT_255) {
                bottom_right = LABEL_PIXEL_MASK | label_255;
            } else if (information_byte & BkeBitmap::BOTTOM_RIGHT_0) {
                bottom_right = label_0;
            } else {
                bottom_right = 0;
            }

            reinterpret_cast<sycl::vec<uint32_t, 2> *>(labels)[(image_linear_id + width) / 2] =
                sycl::vec(bottom_left, bottom_right);
        });

    return final_labelling_event;
}
