#include "fit_quad.h"
#include "apriltag.h"
#include "tag36h11.h"

#include "stb_image_write.h"

struct quad_decode_task_data
{
    int i0, i1;
    zarray_t *quads;
    apriltag_detector_t *td;

    image_u8_t *im;
    zarray_t *detections;

    image_u8_t *im_samples;
};

static void image_u8_draw_line(uint8_t *im, float x0, float y0, float x1, float y1,
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
        im[idx + 1] = 0;
        im[idx + 2] = 0;
    }
}

extern "C" {
    void quad_decode_task(void *_u);
}

zarray_t *quad_decode_task(QuadCorners& data, image_u8_t *im) {
    apriltag_detector_t *tag_detector = apriltag_detector_create();
    apriltag_family_t *tag_family = tag36h11_create();
    apriltag_detector_add_family(tag_detector, tag_family);
    tag_detector->quad_decimate = 1;
    tag_detector->nthreads = 6;
    tag_detector->debug = false;
    tag_detector->wp = workerpool_create(tag_detector->nthreads);
    tag_detector->qtp.min_white_black_diff = 5;

    zarray_t *detections = zarray_create(sizeof(apriltag_detection_t *));
    zarray_t *quads = zarray_create(sizeof(struct quad));
    struct quad quad_original{};
    memcpy(quad_original.p, data.corners, 8 * sizeof(float));
    zarray_add(quads, &quad_original);

    quad_decode_task_data task{};
    task.i0 = 0;
    task.i1 = 1;
    task.quads = quads;
    task.td = tag_detector;
    task.im = im;
    task.detections = detections;
    
    quad_decode_task(&task);

    zarray_get(quads, 0, &quad_original);

    return detections;
}

void decode_quads(const QuadCorners *corner_data, size_t corner_data_length,
                            size_t width, size_t height, uint8_t *greyscaled, bool debug) {
    image_u8_t im_orig{
        .width = static_cast<int32_t>(width),
        .height = static_cast<int32_t>(height),
        .stride = static_cast<int32_t>(width),
        .buf = greyscaled,
    };

    auto writez = new uint8_t[width * height * 3]();
    auto writer = new uint8_t[width * height * 3]();

    size_t found = 0;
    for (size_t asd = 0; asd < corner_data_length; asd++) {
        // std::cout << "processing " << i << std::endl;
        QuadCorners curr = corner_data[asd];
        auto ret = quad_decode_task(curr, &im_orig);
        size_t det_size = zarray_size(ret);
        found += det_size;

        if (debug) {
            for (size_t j = 0; j < det_size; j++) {
                apriltag_detection_t *detected;
                zarray_get(ret, j, &detected);
                std::cout << reinterpret_cast<uint64_t>(detected) << std::endl;
                std::cout << "decoded id " << detected->id << std::endl;

                for (int k = 0; k < 4; k++) {
                    image_u8_draw_line(writer, curr.corners[k][0], curr.corners[k][1],
                                            curr.corners[(k + 1) % 4][0], curr.corners[(k + 1) % 4][1], width,
                                                height);
                }

                for (int k = 0; k < 4; k++) {
                    image_u8_draw_line(writez, detected->p[k][0], detected->p[k][1],
                                            detected->p[(k + 1) % 4][0], detected->p[(k + 1) % 4][1], width,
                                            height);
                }
            }
        }
    }

    if (debug) {
        stbi_write_png("decoded.png", width, height, 3, writez,
                           width * 3);

        stbi_write_png("decoded1.png", width, height, 3, writer,
                           width * 3);
    }

    std::cout << "found " << found << " detections" << std::endl;
}

