#ifndef YOLO4_TINY_H
#define YOLO4_TINY_H

#include "basic_block.h"
#include "yolo4_tiny_params.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "timer_log.h"

#ifdef __cplusplus
extern "C" {
#endif

// void print_debug_info(const char* name, data_t* data, size_t total_size) {
//     printf("--- [Debug] %s (Total Size: %zu) ---\n", name, total_size);

//     // 打印前 256 个元素
//     printf("Front 256 (128 per line):\n");
//     for (size_t i = 0; i < 256 && i < total_size; i++) {
//         printf("%d ", (int)data[i]);
//         if ((i + 1) % 128 == 0) {
//             printf("\n"); // 每 128 个元素换行
//         }
//     }
//     // 如果最后一行没满 128 导致没换行，手动补一个
//     if (total_size < 256 && total_size % 128 != 0) printf("\n");

//     printf("\n");

//     // 打印后 256 个元素
//     printf("Back 256 (128 per line):\n");
//     size_t start = (total_size > 256) ? (total_size - 256) : 0;
    
//     for (size_t i = start; i < total_size; i++) {
//         printf("%d ", (int)data[i]);
//         // 这里的换行逻辑：当 i 是相对于 start 的第 128 的倍数时换行
//         // 或者计算 (i - start + 1) % 128 == 0
//         if ((i - start + 1) % 128 == 0) {
//             printf("\n");
//         }
//     }
//     // 处理末尾不足 128 的换行
//     if ((total_size - start) % 128 != 0) printf("\n");

//     printf("-----------------------------------\n\n");
// }


/* quantisation scale – must match your training export */
#ifndef SCALE
#define SCALE 512
#endif

typedef struct {
    CSPDarkNet* backbone;
    BasicConv*  conv_forP5_conv;
    /* yolo_headP4 */
    BasicConv*  yolo_headP4_basic_conv1;
    data_t*     yolo_headP4_w2;
    data_t*     yolo_headP4_b2;
    /* yolo_headP5 */
    BasicConv*  yolo_headP5_basic_conv1;
    data_t*     yolo_headP5_w2;
    data_t*     yolo_headP5_b2;
    /* upsample */
    BasicConv*  upsample_conv;
} Yolo4_Tiny;

static inline Yolo4_Tiny* Yolo4_Tiny_create(void)
{
    Yolo4_Tiny* self = (Yolo4_Tiny*)malloc(sizeof(Yolo4_Tiny));
    if (self == NULL) { return NULL; }

    self->backbone = CSPDarkNet_create();
    self->conv_forP5_conv = BasicConv_create(13, 13, 1, 1, 0, 512, 256,
        (data_t*)CONV_P5_W_ADDR, (data_t*)CONV_P5_B_ADDR);

    self->yolo_headP4_basic_conv1 = BasicConv_create(26, 26, 3, 1, 1, 384, 256,
        (data_t*)HEADP4_W1_ADDR, (data_t*)HEADP4_B1_ADDR);
    self->yolo_headP4_w2 = (data_t*)HEADP4_W2_ADDR;
    self->yolo_headP4_b2 = (data_t*)HEADP4_B2_ADDR;

    self->yolo_headP5_basic_conv1 = BasicConv_create(13, 13, 3, 1, 1, 256, 512,
        (data_t*)HEADP5_W1_ADDR, (data_t*)HEADP5_B1_ADDR);
    self->yolo_headP5_w2 = (data_t*)HEADP5_W2_ADDR;
    self->yolo_headP5_b2 = (data_t*)HEADP5_B2_ADDR;

    self->upsample_conv = BasicConv_create(13, 13, 1, 1, 0, 256, 128,
        (data_t*)UPSAMPLE_W_ADDR, (data_t*)UPSAMPLE_B_ADDR);

    if (self->backbone == NULL || self->conv_forP5_conv == NULL ||
        self->yolo_headP4_basic_conv1 == NULL || 
        self->yolo_headP5_basic_conv1 == NULL ||
        self->upsample_conv == NULL) {
        CSPDarkNet_destroy(self->backbone);
        BasicConv_destroy(self->conv_forP5_conv);
        BasicConv_destroy(self->yolo_headP4_basic_conv1);
        BasicConv_destroy(self->yolo_headP5_basic_conv1);
        BasicConv_destroy(self->upsample_conv);
        free(self);
        return NULL;
    }

    return self;
}

static inline void Yolo4_Tiny_destroy(Yolo4_Tiny* self)
{
    if (self == NULL) { return; }
    CSPDarkNet_destroy(self->backbone);
    BasicConv_destroy(self->conv_forP5_conv);
    BasicConv_destroy(self->yolo_headP4_basic_conv1);
    /* 不再 free w2/b2，它们在固定内存区域 */
    BasicConv_destroy(self->yolo_headP5_basic_conv1);
    BasicConv_destroy(self->upsample_conv);
    free(self);
}

static inline void Yolo4_Tiny_load_weight(Yolo4_Tiny* self)
{
    CSPDarkNet_load_weight(self->backbone);

    /* conv_forP5 */
    BasicConv_load_weight(self->conv_forP5_conv,
        YOLO4_TINY_CONV_FORP5_W_OFF, YOLO4_TINY_CONV_FORP5_B_OFF);

    /* yolo_headP4 */
    read_params(YOLO4_TINY_HEADP4_W1_OFF,
                self->yolo_headP4_basic_conv1->weight, 256u * 384u * 9u);
    read_params(YOLO4_TINY_HEADP4_B1_OFF,
                self->yolo_headP4_basic_conv1->bias, 256u);
    read_params(YOLO4_TINY_HEADP4_W2_OFF,
                self->yolo_headP4_w2, (unsigned)(OUT_CH * 256 * 1));
    read_params(YOLO4_TINY_HEADP4_B2_OFF,
                self->yolo_headP4_b2, (unsigned)OUT_CH);

    /* yolo_headP5 */
    read_params(YOLO4_TINY_HEADP5_W1_OFF,
                self->yolo_headP5_basic_conv1->weight, 512u * 256u * 9u);
    read_params(YOLO4_TINY_HEADP5_B1_OFF,
                self->yolo_headP5_basic_conv1->bias, 512u);
    read_params(YOLO4_TINY_HEADP5_W2_OFF,
                self->yolo_headP5_w2, (unsigned)(OUT_CH * 512 * 1));
    read_params(YOLO4_TINY_HEADP5_B2_OFF,
                self->yolo_headP5_b2, (unsigned)OUT_CH);

    /* upsample */
    BasicConv_load_weight(self->upsample_conv,
        YOLO4_TINY_UPSAMPLE_W_OFF, YOLO4_TINY_UPSAMPLE_B_OFF);
}

static inline void Yolo4_Tiny_forward(Yolo4_Tiny* self,
    data_t* in, data_t* out0, data_t* out1)
{
    data_t* feat1 = NULL;
    data_t* feat2 = NULL;
    data_t* P5 = NULL;
    data_t* out0_tmp = NULL;
    data_t* P5_Upsample = NULL;
    data_t* P4 = NULL;
    data_t* out1_tmp = NULL;

    uint32_t us;

    feat1 = (data_t*)malloc((size_t)256 * 26 * 26 * sizeof(data_t));
    feat2 = (data_t*)malloc((size_t)512 * 13 * 13 * sizeof(data_t));
    if (feat1 == NULL || feat2 == NULL) {
        free(feat1);
        free(feat2);
        return;
    }

    /* backbone (per-layer timings are recorded inside CSPDarkNet) */
    CSPDarkNet_forward(self->backbone, in, feat1, feat2);
    // print_debug_info("Backbone feat1", feat1, (size_t)256 * 26 * 26);
    // print_debug_info("Backbone feat2", feat2, (size_t)512 * 13 * 13);
    P5 = (data_t*)malloc((size_t)256 * 13 * 13 * sizeof(data_t));
    if (P5 == NULL) {
        free(feat1);
        free(feat2);
        return;
    }

    /* conv_forP5 */
    log_timer_start(TIMER_CONV_FOR_P5);
    BasicConv_forward(self->conv_forP5_conv, feat2, P5);
    log_timer_end(TIMER_CONV_FOR_P5);
    free(feat2);

    out0_tmp = (data_t*)malloc((size_t)512 * 13 * 13 * sizeof(data_t));
    if (out0_tmp == NULL) {
        free(feat1);
        free(P5);
        return;
    }
    // print_debug_info("conv_forP5 output", P5, (size_t)256 * 13 * 13);

    /* out0: yolo_headP5 */
    log_timer_start(TIMER_YOLO_HEADP5);
    BasicConv_forward(self->yolo_headP5_basic_conv1, P5, out0_tmp);
    conv_leakyrelu(512u, (uint32_t)OUT_CH, 0u, 1u, 1u, 13u, 13u,
                   out0_tmp, self->yolo_headP5_w2, self->yolo_headP5_b2,
                   out0, 0u, 128u, 13u);
    log_timer_end(TIMER_YOLO_HEADP5);
    // print_debug_info("yolo_headP5 output", out0, (size_t)512 * 13 * 13);
    free(out0_tmp);

    P5_Upsample = (data_t*)malloc((size_t)128 * 13 * 13 * sizeof(data_t));
    if (P5_Upsample == NULL) {
        free(feat1);
        free(P5);
        return;
    }

    /* upsample_conv */
    log_timer_start(TIMER_UPSAMPLE_CONV);
    BasicConv_forward(self->upsample_conv, P5, P5_Upsample);
    log_timer_end(TIMER_UPSAMPLE_CONV);
    // print_debug_info("P5_Upsample", P5_Upsample, (size_t)128 * 13 * 13);

    P4 = (data_t*)malloc((size_t)384 * 26 * 26 * sizeof(data_t));
    if (P4 == NULL) {
        free(feat1);
        free(P5);
        free(P5_Upsample);
        return;
    }
    memcpy(P4 + 128 * 26 * 26, feat1, (size_t)256 * 26 * 26 * sizeof(data_t));
    free(feat1);
    free(P5);

    /* upsample (hw accelerator) */
    log_timer_start(TIMER_UPSAMPLE);
    sampling(P5_Upsample, P4, 128u, 13u, 0u);
    log_timer_end(TIMER_UPSAMPLE);
    free(P5_Upsample);

    out1_tmp = (data_t*)malloc((size_t)256 * 26 * 26 * sizeof(data_t));
    if (out1_tmp == NULL) {
        free(P4);
        return;
    }

    /* out1: yolo_headP4 */
    log_timer_start(TIMER_YOLO_HEADP4);
    BasicConv_forward(self->yolo_headP4_basic_conv1, P4, out1_tmp);
    conv_leakyrelu(256u, (uint32_t)OUT_CH, 0u, 1u, 1u, 26u, 26u,
                   out1_tmp, self->yolo_headP4_w2, self->yolo_headP4_b2,
                   out1, 0u, 128u, 13u);
    log_timer_end(TIMER_YOLO_HEADP4);
    // print_debug_info("yolo_headP4 output", out1, (size_t)256 * 26 * 26);
    free(P4);
    free(out1_tmp);
}

#ifdef __cplusplus
}
#endif

#endif /* YOLO4_TINY_H */