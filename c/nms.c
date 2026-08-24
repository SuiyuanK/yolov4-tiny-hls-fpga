#include "nms.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ---- class names ---- */
static const char* const voc_class_name[20] = {
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
};

static const char* const coco_class_name[80] = {
    "person","bicycle","car","motorbike","aeroplane",
    "bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird",
    "cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon",
    "bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut",
    "cake","chair","sofa","pottedplant","bed",
    "diningtable","toilet","tvmonitor","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock",
    "vase","scissors","teddy bear","hair drier","toothbrush"
};

/* ---- helper types ---- */
typedef struct {
    float box[4];  /* x1, y1, x2, y2 */
    float score;
    float pred;    /* class id */
} Box;

/* ---- helper functions ---- */
static float nms_max(float x, float y) { return (x > y) ? x : y; }
static float nms_min(float x, float y) { return (x < y) ? x : y; }
static float nms_abs(float x) { return (x > 0.0f) ? x : -x; }

static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

static float calc_iou(const float* r1, const float* r2)
{
    float xi1 = nms_max(r1[0], r2[0]);
    float yi1 = nms_max(r1[1], r2[1]);
    float xi2 = nms_min(r1[2], r2[2]);
    float yi2 = nms_min(r1[3], r2[3]);

    float inter = 0.0f;
    if (xi2 > xi1 && yi2 > yi1) {
        inter = (xi2 - xi1) * (yi2 - yi1);
    }
    float a1 = nms_abs(r1[0] - r1[2]) * nms_abs(r1[1] - r1[3]);
    float a2 = nms_abs(r2[0] - r2[2]) * nms_abs(r2[1] - r2[3]);
    return inter / (a1 + a2 - inter);
}

/* qsort comparator: descending by score */
static int cmp_box_desc(const void* a, const void* b)
{
    float sa = ((const Box*)a)->score;
    float sb = ((const Box*)b)->score;
    if (sa > sb) return -1;
    if (sa < sb) return  1;
    return 0;
}

/* ---- decode_box (single scale) ---- */
static void decode_box(const float* in, float* out, int ch, int h, int w)
{
    float anchor_w[3], anchor_h[3];
    int m, r, c, k;

    if (h == 13) {
        anchor_w[0] = 2.5312f; anchor_w[1] = 4.2188f; anchor_w[2] = 10.7500f;
        anchor_h[0] = 2.5625f; anchor_h[1] = 5.2812f; anchor_h[2] = 9.9688f;
    } else {
        anchor_w[0] = 1.4375f; anchor_w[1] = 2.3125f; anchor_w[2] = 5.0625f;
        anchor_h[0] = 1.6875f; anchor_h[1] = 3.6250f; anchor_h[2] = 5.1250f;
    }

    for (m = 0; m < 3; m++) {
        for (r = 0; r < h; r++) {
            for (c = 0; c < w; c++) {
                int idx = m * h * w + r * w + c;
                int ch3 = ch / 3;
                int base_in = m * ch3 * h * w;

                out[idx * 7 + 0] = (sigmoid_f(in[base_in + 0 * h * w + r * w + c]) + (float)c) / (float)h;
                out[idx * 7 + 1] = (sigmoid_f(in[base_in + 1 * h * w + r * w + c]) + (float)r) / (float)h;
                out[idx * 7 + 2] = expf(in[base_in + 2 * h * w + r * w + c]) * anchor_w[m] / (float)h;
                out[idx * 7 + 3] = expf(in[base_in + 3 * h * w + r * w + c]) * anchor_h[m] / (float)h;
                out[idx * 7 + 4] = sigmoid_f(in[base_in + 4 * h * w + r * w + c]);

                float max_val = -100.0f;
                int   max_idx = -1;
                for (k = 0; k < ch3 - 5; k++) {
                    float v = in[base_in + (5 + k) * h * w + r * w + c];
                    if (v > max_val) {
                        max_val = v;
                        max_idx = k;
                    }
                }
                out[idx * 7 + 5] = sigmoid_f(max_val);
                out[idx * 7 + 6] = (float)max_idx;
            }
        }
    }
}

void decode_box_top(float* in0, float* in1, float* out, int ch)
{
    decode_box(in0, out, ch, 13, 13);
    decode_box(in1, out + (size_t)3u * 13u * 13u * 7u, ch, 26, 26);
}

/* ---- NMS ---- */
static int nms_core(Box* boxes, int n, float threshold, Box* results)
{
    int res_cnt = 0;
    int i;
    /* boxes is already sorted desc by score (done before call) */
    /* mark deleted */
    int* deleted = (int*)calloc((size_t)n, sizeof(int));
    if (deleted == NULL) {
        return 0;
    }

    for (i = 0; i < n; i++) {
        if (deleted[i]) continue;
        results[res_cnt++] = boxes[i];
        {
            int j;
            for (j = i + 1; j < n; j++) {
                if (!deleted[j] && calc_iou(boxes[i].box, boxes[j].box) > threshold) {
                    deleted[j] = 1;
                }
            }
        }
    }
    free(deleted);
    return res_cnt;
}

/* ---- full post-processing ---- */
static void non_max_suppression(float* prediction, int num_class,
    float conf_thres, float nms_thres, float image_shape[2])
{
    const int n = 3 * 13 * 13 + 3 * 26 * 26;  /* 2535 */
    int i, j;
    int reserve_cnt = 0;

    /* convert (cx, cy, w, h) to (x1, y1, x2, y2) */
    for (i = 0; i < n; i++) {
        float x = prediction[i * 7 + 0];
        float y = prediction[i * 7 + 1];
        float w = prediction[i * 7 + 2];
        float h = prediction[i * 7 + 3];
        prediction[i * 7 + 0] = x - w / 2.0f;
        prediction[i * 7 + 1] = y - h / 2.0f;
        prediction[i * 7 + 2] = x + w / 2.0f;
        prediction[i * 7 + 3] = y + h / 2.0f;
    }

    /* confidence filter */
    int* reserve = (int*)calloc((size_t)n, sizeof(int));
    if (reserve == NULL) {
        return;
    }
    for (i = 0; i < n; i++) {
        if (prediction[i * 7 + 4] * prediction[i * 7 + 5] >= conf_thres) {
            reserve[i] = 1;
            reserve_cnt++;
        }
    }
    if (reserve_cnt == 0) {
        printf("no object detected!\n");
        free(reserve);
        return;
    }

    /* collect reserved detections */
    float* detection = (float*)malloc((size_t)reserve_cnt * 7u * sizeof(float));
    if (detection == NULL) {
        free(reserve);
        return;
    }
    j = 0;
    for (i = 0; i < n; i++) {
        if (reserve[i]) {
            memcpy(&detection[j * 7], &prediction[i * 7], 7u * sizeof(float));
            j++;
        }
    }

    /* find unique class labels */
    int unique_labels[256];
    int unique_cnt = 0;
    for (i = 0; i < reserve_cnt; i++) {
        int cls = (int)detection[i * 7 + 6];
        int found = 0;
        for (j = 0; j < unique_cnt; j++) {
            if (unique_labels[j] == cls) { found = 1; break; }
        }
        if (!found && unique_cnt < 256) {
            unique_labels[unique_cnt++] = cls;
        }
    }

    /* per-class NMS */
    Box* class_boxes = (Box*)malloc((size_t)reserve_cnt * sizeof(Box));
    Box* results     = (Box*)malloc((size_t)reserve_cnt * sizeof(Box));
    int  total_res   = 0;
    if (class_boxes == NULL || results == NULL) {
        free(class_boxes);
        free(results);
        free(detection);
        free(reserve);
        return;
    }

    for (i = 0; i < unique_cnt; i++) {
        int cls = unique_labels[i];
        int cb_cnt = 0;
        for (j = 0; j < reserve_cnt; j++) {
            if ((int)detection[j * 7 + 6] == cls) {
                class_boxes[cb_cnt].box[0] = detection[j * 7 + 0];
                class_boxes[cb_cnt].box[1] = detection[j * 7 + 1];
                class_boxes[cb_cnt].box[2] = detection[j * 7 + 2];
                class_boxes[cb_cnt].box[3] = detection[j * 7 + 3];
                class_boxes[cb_cnt].score   = detection[j * 7 + 4] * detection[j * 7 + 5];
                class_boxes[cb_cnt].pred    = (float)cls;
                cb_cnt++;
            }
        }
        /* sort descending by score */
        qsort(class_boxes, (size_t)cb_cnt, sizeof(Box), cmp_box_desc);
        total_res += nms_core(class_boxes, cb_cnt, nms_thres, &results[total_res]);
    }

    /* rescale and print */
    for (i = 0; i < total_res; i++) {
        float x1 = results[i].box[0];
        float y1 = results[i].box[1];
        float x2 = results[i].box[2];
        float y2 = results[i].box[3];

        int top    = (int)(y1 * image_shape[0]);
        int left   = (int)(x1 * image_shape[1]);
        int bottom = (int)(y2 * image_shape[0]);
        int right  = (int)(x2 * image_shape[1]);

        if (top < 0)    top = 0;
        if (left < 0)   left = 0;
        if (bottom > (int)image_shape[0]) bottom = (int)image_shape[0];
        if (right  > (int)image_shape[1]) right  = (int)image_shape[1];

        if (num_class == 20) {
            printf("box:(%d,%d,%d,%d), score:%.4f, pred:%s\n",
                   top, left, bottom, right,
                   results[i].score,
                   voc_class_name[(int)results[i].pred]);
        } else {
            printf("box:(%d,%d,%d,%d), score:%.4f, pred:%s\n",
                   top, left, bottom, right,
                   results[i].score,
                   coco_class_name[(int)results[i].pred]);
        }
    }

    free(class_boxes);
    free(results);
    free(detection);
    free(reserve);
}

void detector(float* in0, float* in1, int ch,
              float conf_thres, float nms_thres, float image_shape[2])
{
    int total = ch * 13 * 13 + ch * 26 * 26;
    float* tmp = (float*)malloc((size_t)total * sizeof(float));
    if (tmp == NULL) {
        return;
    }
    decode_box_top(in0, in1, tmp, ch);
    non_max_suppression(tmp, ch / 3 - 5, conf_thres, nms_thres, image_shape);
    free(tmp);
}