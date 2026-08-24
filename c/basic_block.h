#ifndef BASIC_BLOCK_H
#define BASIC_BLOCK_H

#include "basic_op.h"
#include "sd_driver.h"
#include "yolo_timer.h"
#include "yolo4_tiny_params.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "timer_log.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============ data type ============ */
typedef int16_t data_t;

/* ============ BasicConv ============ */
typedef struct {
    int h;
    int w;
    int s;
    int k;
    int p;
    int ch_in;
    int ch_out;
    data_t* weight;
    data_t* bias;
} BasicConv;

static inline BasicConv* BasicConv_create(int h, int w, int k, int s, int p, int ch_in, int ch_out, data_t* weight_addr, data_t* bias_addr)
{
    BasicConv* self = (BasicConv*)malloc(sizeof(BasicConv));
    if (self == NULL) { return NULL; }
    self->h      = h;
    self->w      = w;
    self->k      = k;
    self->s      = s;
    self->p      = p;
    self->ch_in  = ch_in;
    self->ch_out = ch_out;
    self->weight = weight_addr;  /* 直接使用传入的固定地址 */
    self->bias   = bias_addr;
    return self;
}

static inline void BasicConv_destroy(BasicConv* self)
{
    if (self == NULL) { return; }
    /* 不再 free weight 和 bias，因为它们在固定内存区域 */
    free(self);
}

/* folded_conv + leakyrelu (act=1) */
static inline void BasicConv_forward(const BasicConv* self, data_t* in, data_t* out)
{
    conv_leakyrelu((uint32_t)self->ch_in, (uint32_t)self->ch_out,
                   (uint32_t)self->p, (uint32_t)self->s, (uint32_t)self->k,
                   (uint32_t)self->h, (uint32_t)self->w,
                   in, self->weight, self->bias, out,
                   1u,   /* act = leakyrelu */
                   128u, /* c   (fixed for yolov4-tiny) */
                   13u);  /* size(fixed for yolov4-tiny) */
}

static inline void BasicConv_load_weight(BasicConv* self, unsigned int w_sector, unsigned int b_sector)
{
    read_params(w_sector, self->weight,
                (unsigned int)(self->ch_out * self->ch_in * self->k * self->k));
    read_params(b_sector, self->bias,
                (unsigned int)(self->ch_out));
}

/* ============ Resblock_body ============ */
typedef struct {
    int h;
    int w;
    int ch_in;
    int ch_out;
    BasicConv* conv1;
    BasicConv* conv2;
    BasicConv* conv3;
    BasicConv* conv4;
} Resblock_body;

static inline Resblock_body* Resblock_body_create(int h, int w, int ch_in, int ch_out,
    data_t* c1_w, data_t* c1_b, data_t* c2_w, data_t* c2_b,
    data_t* c3_w, data_t* c3_b, data_t* c4_w, data_t* c4_b)
{
    Resblock_body* self = (Resblock_body*)malloc(sizeof(Resblock_body));
    if (self == NULL) { return NULL; }
    self->h      = h;
    self->w      = w;
    self->ch_in  = ch_in;
    self->ch_out = ch_out;
    self->conv1  = BasicConv_create(h, w, 3, 1, 1, ch_in,    ch_out, c1_w, c1_b);
    self->conv2  = BasicConv_create(h, w, 3, 1, 1, ch_out/2, ch_out/2, c2_w, c2_b);
    self->conv3  = BasicConv_create(h, w, 3, 1, 1, ch_out/2, ch_out/2, c3_w, c3_b);
    self->conv4  = BasicConv_create(h, w, 1, 1, 0, ch_out,   ch_out, c4_w, c4_b);
    if (self->conv1 == NULL || self->conv2 == NULL ||
        self->conv3 == NULL || self->conv4 == NULL) {
        BasicConv_destroy(self->conv1);
        BasicConv_destroy(self->conv2);
        BasicConv_destroy(self->conv3);
        BasicConv_destroy(self->conv4);
        free(self);
        return NULL;
    }
    return self;
}

static inline void Resblock_body_destroy(Resblock_body* self)
{
    if (self == NULL) { return; }
    BasicConv_destroy(self->conv1);
    BasicConv_destroy(self->conv2);
    BasicConv_destroy(self->conv3);
    BasicConv_destroy(self->conv4);
    free(self);
}

static inline void Resblock_body_load_weight(Resblock_body* self,
    unsigned int w1_sec, unsigned int b1_sec,
    unsigned int w2_sec, unsigned int b2_sec,
    unsigned int w3_sec, unsigned int b3_sec,
    unsigned int w4_sec, unsigned int b4_sec)
{
    read_params(w1_sec, self->conv1->weight,
                (unsigned int)(self->ch_in  * self->ch_out * 9));
    read_params(b1_sec, self->conv1->bias,
                (unsigned int)(self->ch_out));

    read_params(w2_sec, self->conv2->weight,
                (unsigned int)(self->ch_out/2 * self->ch_out/2 * 9));
    read_params(b2_sec, self->conv2->bias,
                (unsigned int)(self->ch_out/2));

    read_params(w3_sec, self->conv3->weight,
                (unsigned int)(self->ch_out/2 * self->ch_out/2 * 9));
    read_params(b3_sec, self->conv3->bias,
                (unsigned int)(self->ch_out/2));

    read_params(w4_sec, self->conv4->weight,
                (unsigned int)(self->ch_out * self->ch_out * 1));
    read_params(b4_sec, self->conv4->bias,
                (unsigned int)(self->ch_out));
}

static inline void Resblock_body_forward(Resblock_body* self,
    data_t* in, data_t* feat, data_t* out)
{
    // data_t* route = (data_t*)malloc((size_t)self->ch_out * (size_t)self->h * (size_t)self->w * sizeof(data_t));
    // data_t* conv3_out = (data_t*)malloc((size_t)self->ch_out * (size_t)self->h * (size_t)self->w * sizeof(data_t));
    
    // if (route == NULL || conv3_out == NULL) {
    //     free(route);
    //     free(conv3_out);
    //     return;
    // }
    
    // static data_t route[1500000];
    // static data_t conv3_out[1500000];
    
    data_t* route = (data_t*)ROUTE_BUF_ADDR;
    data_t* conv3_out = (data_t*)CONV3_OUT_BUF_ADDR;

    /* conv1: in -> route */
    BasicConv_forward(self->conv1, in, route);

    /* conv2: route 的后半通道 -> conv3_out 的后半 */
    BasicConv_forward(self->conv2,
                      route + (self->ch_out / 2) * self->h * self->w,
                      conv3_out + (self->ch_out / 2) * self->h * self->w);

    /* conv3: conv3_out 后半 -> conv3_out 前半 */
    BasicConv_forward(self->conv3,
                      conv3_out + (self->ch_out / 2) * self->h * self->w,
                      conv3_out);

    /* conv4: conv3_out -> feat */
    BasicConv_forward(self->conv4, conv3_out, feat);

    /* maxpool(concat(route, feat)) */
    sampling(route, out, (uint32_t)(self->ch_out), (uint32_t)(self->h), 1u);
    sampling(feat,  out + self->ch_out * self->h * self->w / 4, (uint32_t)(self->ch_out), (uint32_t)(self->h), 1u);
    
    // free(route);
    // free(conv3_out);

}

/* ============ CSPDarkNet ============ */
typedef struct {
    BasicConv*     basic_conv1;
    BasicConv*     basic_conv2;
    BasicConv*     basic_conv3;
    Resblock_body* resblock1;
    Resblock_body* resblock2;
    Resblock_body* resblock3;
} CSPDarkNet;

static inline CSPDarkNet* CSPDarkNet_create(void)
{
    CSPDarkNet* self = (CSPDarkNet*)malloc(sizeof(CSPDarkNet));
    if (self == NULL) { return NULL; }
    
    self->basic_conv1 = BasicConv_create(416, 416, 3, 2, 1, 3, 32, 
        (data_t*)BC1_W_ADDR, (data_t*)BC1_B_ADDR);
    self->basic_conv2 = BasicConv_create(208, 208, 3, 2, 1, 32, 64,
        (data_t*)BC2_W_ADDR, (data_t*)BC2_B_ADDR);
    self->basic_conv3 = BasicConv_create(13, 13, 3, 1, 1, 512, 512,
        (data_t*)BC3_W_ADDR, (data_t*)BC3_B_ADDR);
    
    self->resblock1 = Resblock_body_create(104, 104, 64, 64,
        (data_t*)RB1_C1_W_ADDR, (data_t*)RB1_C1_B_ADDR,
        (data_t*)RB1_C2_W_ADDR, (data_t*)RB1_C2_B_ADDR,
        (data_t*)RB1_C3_W_ADDR, (data_t*)RB1_C3_B_ADDR,
        (data_t*)RB1_C4_W_ADDR, (data_t*)RB1_C4_B_ADDR);
    
    self->resblock2 = Resblock_body_create(52, 52, 128, 128,
        (data_t*)RB2_C1_W_ADDR, (data_t*)RB2_C1_B_ADDR,
        (data_t*)RB2_C2_W_ADDR, (data_t*)RB2_C2_B_ADDR,
        (data_t*)RB2_C3_W_ADDR, (data_t*)RB2_C3_B_ADDR,
        (data_t*)RB2_C4_W_ADDR, (data_t*)RB2_C4_B_ADDR);
    
    self->resblock3 = Resblock_body_create(26, 26, 256, 256,
        (data_t*)RB3_C1_W_ADDR, (data_t*)RB3_C1_B_ADDR,
        (data_t*)RB3_C2_W_ADDR, (data_t*)RB3_C2_B_ADDR,
        (data_t*)RB3_C3_W_ADDR, (data_t*)RB3_C3_B_ADDR,
        (data_t*)RB3_C4_W_ADDR, (data_t*)RB3_C4_B_ADDR);
    
    if (self->basic_conv1 == NULL || self->basic_conv2 == NULL ||
        self->basic_conv3 == NULL || self->resblock1 == NULL ||
        self->resblock2 == NULL || self->resblock3 == NULL) {
        BasicConv_destroy(self->basic_conv1);
        BasicConv_destroy(self->basic_conv2);
        BasicConv_destroy(self->basic_conv3);
        Resblock_body_destroy(self->resblock1);
        Resblock_body_destroy(self->resblock2);
        Resblock_body_destroy(self->resblock3);
        free(self);
        return NULL;
    }
    return self;
}

static inline void CSPDarkNet_destroy(CSPDarkNet* self)
{
    if (self == NULL) { return; }
    BasicConv_destroy(self->basic_conv1);
    BasicConv_destroy(self->basic_conv2);
    BasicConv_destroy(self->basic_conv3);
    Resblock_body_destroy(self->resblock1);
    Resblock_body_destroy(self->resblock2);
    Resblock_body_destroy(self->resblock3);
    free(self);
}

static inline void CSPDarkNet_load_weight(CSPDarkNet* self)
{
    BasicConv_load_weight(self->basic_conv1,
        YOLO4_TINY_BASIC_CONV1_W_OFF, YOLO4_TINY_BASIC_CONV1_B_OFF);
    BasicConv_load_weight(self->basic_conv2,
        YOLO4_TINY_BASIC_CONV2_W_OFF, YOLO4_TINY_BASIC_CONV2_B_OFF);
    BasicConv_load_weight(self->basic_conv3,
        YOLO4_TINY_BASIC_CONV3_W_OFF, YOLO4_TINY_BASIC_CONV3_B_OFF);

    Resblock_body_load_weight(self->resblock1,
        YOLO4_TINY_RB1_W1_OFF, YOLO4_TINY_RB1_B1_OFF,
        YOLO4_TINY_RB1_W2_OFF, YOLO4_TINY_RB1_B2_OFF,
        YOLO4_TINY_RB1_W3_OFF, YOLO4_TINY_RB1_B3_OFF,
        YOLO4_TINY_RB1_W4_OFF, YOLO4_TINY_RB1_B4_OFF);

    Resblock_body_load_weight(self->resblock2,
        YOLO4_TINY_RB2_W1_OFF, YOLO4_TINY_RB2_B1_OFF,
        YOLO4_TINY_RB2_W2_OFF, YOLO4_TINY_RB2_B2_OFF,
        YOLO4_TINY_RB2_W3_OFF, YOLO4_TINY_RB2_B3_OFF,
        YOLO4_TINY_RB2_W4_OFF, YOLO4_TINY_RB2_B4_OFF);

    Resblock_body_load_weight(self->resblock3,
        YOLO4_TINY_RB3_W1_OFF, YOLO4_TINY_RB3_B1_OFF,
        YOLO4_TINY_RB3_W2_OFF, YOLO4_TINY_RB3_B2_OFF,
        YOLO4_TINY_RB3_W3_OFF, YOLO4_TINY_RB3_B3_OFF,
        YOLO4_TINY_RB3_W4_OFF, YOLO4_TINY_RB3_B4_OFF);
}

static inline void CSPDarkNet_forward(CSPDarkNet* self,
    data_t* in, data_t* feat1, data_t* feat2)
{
    data_t* basic_conv1_out;
    data_t* basic_conv2_out;
    data_t* resblock1_out;
    data_t* resblock2_out;
    data_t* resblock3_out;
    data_t* tmp;

    uint32_t us;

    // printf("===Backbone CSPDarkNet Start===\n");

    basic_conv1_out = (data_t*)malloc((size_t)32 * 208 * 208 * sizeof(data_t));
    if (basic_conv1_out == NULL) {
        return;
    }

    log_timer_start(TIMER_BASIC_CONV1);
    BasicConv_forward(self->basic_conv1, in, basic_conv1_out);
    log_timer_end(TIMER_BASIC_CONV1);

    basic_conv2_out = (data_t*)malloc((size_t)64 * 104 * 104 * sizeof(data_t));
    if (basic_conv2_out == NULL) {
        free(basic_conv1_out);
        return;
    }

    log_timer_start(TIMER_BASIC_CONV2);
    BasicConv_forward(self->basic_conv2, basic_conv1_out, basic_conv2_out);
    log_timer_end(TIMER_BASIC_CONV2);

    // // 输出basic_conv2_out前4和后4个元素
    // printf("basic_conv2_out[0..3]: %d %d %d %d\n",
    //     basic_conv2_out[0], basic_conv2_out[1], basic_conv2_out[2], basic_conv2_out[3]);
    // size_t bc2_size = (size_t)64 * 104 * 104;
    // printf("basic_conv2_out[-4..-1]: %d %d %d %d\n",
    //     basic_conv2_out[bc2_size-4], basic_conv2_out[bc2_size-3], basic_conv2_out[bc2_size-2], basic_conv2_out[bc2_size-1]);
    
    free(basic_conv1_out);

    resblock1_out = (data_t*)malloc((size_t)128 * 52 * 52 * sizeof(data_t));
    tmp = (data_t*)malloc((size_t)64 * 104 * 104 * sizeof(data_t));
    if (resblock1_out == NULL || tmp == NULL) {
        free(basic_conv2_out);
        free(resblock1_out);
        free(tmp);
        return;
    }
    
    log_timer_start(TIMER_RESBLOCK1);
    Resblock_body_forward(self->resblock1, basic_conv2_out, tmp, resblock1_out);
    log_timer_end(TIMER_RESBLOCK1);
    free(basic_conv2_out);
    free(tmp);

    resblock2_out = (data_t*)malloc((size_t)256 * 26 * 26 * sizeof(data_t));
    tmp = (data_t*)malloc((size_t)128 * 52 * 52 * sizeof(data_t));
    if (resblock2_out == NULL || tmp == NULL) {
        free(resblock1_out);
        free(resblock2_out);
        free(tmp);
        return;
    }

    log_timer_start(TIMER_RESBLOCK2);
    Resblock_body_forward(self->resblock2, resblock1_out, tmp, resblock2_out);
    log_timer_end(TIMER_RESBLOCK2);
    free(resblock1_out);
    free(tmp);

    resblock3_out = (data_t*)malloc((size_t)512 * 13 * 13 * sizeof(data_t));
    tmp = (data_t*)malloc((size_t)256 * 26 * 26 * sizeof(data_t));
    if (resblock3_out == NULL || tmp == NULL) {
        free(resblock2_out);
        free(resblock3_out);
        free(tmp);
        return;
    }

    log_timer_start(TIMER_RESBLOCK3);
    Resblock_body_forward(self->resblock3, resblock2_out, feat1, resblock3_out);
    log_timer_end(TIMER_RESBLOCK3);
    free(resblock2_out);
    free(tmp);

    log_timer_start(TIMER_BASIC_CONV3);
    BasicConv_forward(self->basic_conv3, resblock3_out, feat2);
    log_timer_end(TIMER_BASIC_CONV3);
    free(resblock3_out);

    // printf("===Backbone CSPDarkNet End===\n");
}

#ifdef __cplusplus
}
#endif

#endif /* BASIC_BLOCK_H */