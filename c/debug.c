#include "debug.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolo4_tiny.h"
#include "yolo4_tiny_params.h"
#include "nms.h"
#include "sd_driver.h"
#include "yolo_timer.h"
#include "timer_log.h"

#include "math.h"

#ifndef SCALE
#define SCALE 512
#endif

extern uint32_t RegRead(uint32_t addr);

/* SD sector for input image */
#define IMAGE_SECTOR  YOLO4_TINY_BYTE_TO_SECTOR(YOLO4_TINY_IMAGE_BYTE_OFFSET)

static uint16_t byte_reverse16(uint16_t v)
{
    return (uint16_t)((v >> 8) | (v << 8));
}

static uint32_t byte_reverse32(uint32_t v)
{
    return ((v & 0x000000FFu) << 24) |
           ((v & 0x0000FF00u) << 8) |
           ((v & 0x00FF0000u) >> 8) |
           ((v & 0xFF000000u) >> 24);
}

static void print_head_tail_hex(const char* tag, const data_t* buf, size_t len)
{
    size_t head_n = (len < 16u) ? len : 16u;
    size_t tail_start = (len > 16u) ? (len - 16u) : 0u;

    printf("%s len=%u\n", tag, (unsigned int)len);
    printf("  head(ord):");
    for (size_t i = 0; i < head_n; i++) {
        printf(" %04X", (unsigned int)(uint16_t)buf[i]);
    }
    printf("\n");

    printf("  head(rev):");
    for (size_t i = 0; i < head_n; i++) {
        printf(" %04X", (unsigned int)byte_reverse16((uint16_t)buf[i]));
    }
    printf("\n");

    printf("  tail(ord):");
    for (size_t i = tail_start; i < len; i++) {
        printf(" %04X", (unsigned int)(uint16_t)buf[i]);
    }
    printf("\n");

    printf("  tail(rev):");
    for (size_t i = tail_start; i < len; i++) {
        printf(" %04X", (unsigned int)byte_reverse16((uint16_t)buf[i]));
    }
    printf("\n");
}

static void print_head_tail_fp32(const char* tag, const float* buf, size_t len)
{
    size_t head_n = (len < 16u) ? len : 16u;
    size_t tail_start = (len > 16u) ? (len - 16u) : 0u;
    uint32_t bits;
    const uint8_t* p;

    printf("%s len=%u\n", tag, (unsigned int)len);
    printf("  head(bits):");
    for (size_t i = 0; i < head_n; i++) {
        memcpy(&bits, &buf[i], sizeof(bits));
        printf(" %08X", (unsigned int)bits);
    }
    printf("\n");

    printf("  head(bits_rev):");
    for (size_t i = 0; i < head_n; i++) {
        memcpy(&bits, &buf[i], sizeof(bits));
        printf(" %08X", (unsigned int)byte_reverse32(bits));
    }
    printf("\n");

    printf("  head(le):");
    for (size_t i = 0; i < head_n; i++) {
        p = (const uint8_t*)&buf[i];
        printf(" %02X%02X%02X%02X",
               (unsigned int)p[0],
               (unsigned int)p[1],
               (unsigned int)p[2],
               (unsigned int)p[3]);
    }
    printf("\n");

    printf("  tail(bits):");
    for (size_t i = tail_start; i < len; i++) {
        memcpy(&bits, &buf[i], sizeof(bits));
        printf(" %08X", (unsigned int)bits);
    }
    printf("\n");

    printf("  tail(bits_rev):");
    for (size_t i = tail_start; i < len; i++) {
        memcpy(&bits, &buf[i], sizeof(bits));
        printf(" %08X", (unsigned int)byte_reverse32(bits));
    }
    printf("\n");

    printf("  tail(le):");
    for (size_t i = tail_start; i < len; i++) {
        p = (const uint8_t*)&buf[i];
        printf(" %02X%02X%02X%02X",
               (unsigned int)p[0],
               (unsigned int)p[1],
               (unsigned int)p[2],
               (unsigned int)p[3]);
    }
    printf("\n");
}

void yolo_test(void)
{
    data_t* in = NULL;
    data_t* out0 = NULL;
    data_t* out1 = NULL;
    float* out0_fp32 = NULL;
    float* out1_fp32 = NULL;

    float image_shape[2];
    uint32_t us;
    int i;

    Yolo4_Tiny* yolo;

    image_shape[0] = 1330.0f;
    image_shape[1] = 1330.0f;

    in = (data_t*)malloc(3u * 416u * 416u * sizeof(data_t));
    out0 = (data_t*)malloc((size_t)OUT_CH * 13u * 13u * sizeof(data_t));
    out1 = (data_t*)malloc((size_t)OUT_CH * 26u * 26u * sizeof(data_t));
    if (in == NULL || out0 == NULL || out1 == NULL) {
        free(in);
        free(out0);
        free(out1);
        printf("alloc failed in yolo_test\n");
        return;
    }

    /* load input image from SD */
    if (read_params(IMAGE_SECTOR, in, 3u * 416u * 416u) != 0) {
        printf("read input image failed\n");
        free(in);
        free(out0);
        free(out1);
        return;
    }

    /* create & load model */
    conv_init();
    yolo = Yolo4_Tiny_create();
    if (yolo == NULL) {
        printf("Yolo4_Tiny_create failed\n");
        free(in);
        free(out0);
        free(out1);
        return;
    }
    Yolo4_Tiny_load_weight(yolo);

    /* inference */
    Yolo4_Tiny_forward(yolo, in, out0, out1);

    // print_head_tail_hex("out0(raw fixed)", out0, (size_t)OUT_CH * 13u * 13u);
    // print_head_tail_hex("out1(raw fixed)", out1, (size_t)OUT_CH * 26u * 26u);

    free(in);
    in = NULL;

    out0_fp32 = (float*)malloc((size_t)OUT_CH * 13u * 13u * sizeof(float));
    out1_fp32 = (float*)malloc((size_t)OUT_CH * 26u * 26u * sizeof(float));
    if (out0_fp32 == NULL || out1_fp32 == NULL) {
        printf("alloc fp32 buffer failed\n");
        Yolo4_Tiny_destroy(yolo);
        free(out0);
        free(out1);
        free(out0_fp32);
        free(out1_fp32);
        return;
    }


    /* fix16 -> FP32 */
    for (i = 0; i < OUT_CH * 13 * 13; i++) {
        out0_fp32[i] = (float)out0[i] / (float)SCALE;
    }
    for (i = 0; i < OUT_CH * 26 * 26; i++) {
        out1_fp32[i] = (float)out1[i] / (float)SCALE;
    }

    // print_head_tail_fp32("out0(fp32)", out0_fp32, (size_t)OUT_CH * 13u * 13u);
    // print_head_tail_fp32("out1(fp32)", out1_fp32, (size_t)OUT_CH * 26u * 26u);

    // // 在后处理之前加这段
    // {
    // // out0: 75*13*13 = 12675, 排布 [75, 13, 13]
    // // objectness 在通道 4, 29, 54 (每 25 个通道一个 anchor)
    // int obj_channels[3] = {4, 29, 54};
    
    // for (int a = 0; a < 3; a++) {
    //     int16_t max_val = -32768;
    //     int max_pos = 0;
    //     int base = obj_channels[a] * 13 * 13;
    //     for (int i = 0; i < 13*13; i++) {
    //         if (out0[base + i] > max_val) {
    //             max_val = out0[base + i];
    //             max_pos = i;
    //         }
    //     }
    //     float fp = (float)max_val / 512.0f;
    //     float sig = 1.0f / (1.0f + expf(-fp));
    //     printf("out0 anchor%d max obj: raw=%d fp=%.4f sigmoid=%.4f at pos=%d\n",
    //            a, max_val, fp, sig, max_pos);
    // }
    
    // int obj_channels1[3] = {4, 29, 54};
    // for (int a = 0; a < 3; a++) {
    //     int16_t max_val = -32768;
    //     int max_pos = 0;
    //     int base = obj_channels1[a] * 26 * 26;
    //     for (int i = 0; i < 26*26; i++) {
    //         if (out1[base + i] > max_val) {
    //             max_val = out1[base + i];
    //             max_pos = i;
    //         }
    //     }
    //     float fp = (float)max_val / 512.0f;
    //     float sig = 1.0f / (1.0f + expf(-fp));
    //     printf("out1 anchor%d max obj: raw=%d fp=%.4f sigmoid=%.4f at pos=%d\n",
    //            a, max_val, fp, sig, max_pos);
    // }
    // }

    /* post-processing: decode + NMS */
    log_timer_start(TIMER_POST_PROCESS);
    detector(out0_fp32, out1_fp32, OUT_CH, 0.5f, 0.3f, image_shape);
    log_timer_end(TIMER_POST_PROCESS);

    /* 汇总并打印每层时间（以微秒为单位），Backbone 为各子层之和，总延迟不包含后处理 */
    {
        int ids[] = {TIMER_BASIC_CONV1, TIMER_BASIC_CONV2, TIMER_RESBLOCK1, TIMER_RESBLOCK2, TIMER_RESBLOCK3, TIMER_BASIC_CONV3,
                     TIMER_CONV_FOR_P5, TIMER_YOLO_HEADP5, TIMER_UPSAMPLE_CONV, TIMER_UPSAMPLE, TIMER_YOLO_HEADP4, TIMER_POST_PROCESS};
        const char* names[] = {"Basic Conv1","Basic Conv2","Resblock1","Resblock2","Resblock3","Basic Conv3",
                               "conv_forP5","yolo_headP5+basic_conv","upsample_conv","upsample","yolo_headP4+basic_conv","post-processing"};
        const int n = sizeof(ids)/sizeof(ids[0]);
        uint32_t sum_us = 0; /* network inference only, excludes software post-processing */
        uint32_t backbone_us = 0;
        uint32_t post_process_us = 0;
        for (int i = 0; i < n; i++) {
            uint32_t dt = timer_get_delta_ticks(ids[i]);
            uint32_t t_us = yolo_timer_ticks_to_us(0, dt);
            printf("It took %u us, %s\n", (unsigned)t_us, names[i]);
            if (i >= 0 && i <= 5) backbone_us += t_us; /* first 6 entries are backbone */
            if (ids[i] == TIMER_POST_PROCESS) {
                post_process_us = t_us;
            } else {
                sum_us += t_us;
            }
        }
        printf("==============================\n");
        printf("[Hardware] Backbone total is %u us (%.6f s)\n", (unsigned)backbone_us, (float)backbone_us / 1000000.0);
        printf("[Hardware] Inference total is %u us (%.6f s)\n", (unsigned)sum_us, (float)sum_us / 1000000.0);
        printf("[Software] Post-processing is %u us (%.6f s)\n", (unsigned)post_process_us, (float)post_process_us / 1000000.0);
        printf("==============================\n");
    }



    /* cleanup */
    Yolo4_Tiny_destroy(yolo);
    free(out0);
    free(out1);
    free(out0_fp32);
    free(out1_fp32);
}