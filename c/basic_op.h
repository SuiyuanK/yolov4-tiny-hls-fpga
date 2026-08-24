#ifndef BASIC_OP_H
#define BASIC_OP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void conv_init(void);
void conv_leakyrelu_addr(uint32_t ch_in,
                         uint32_t ch_out,
                         uint32_t pad,//not used
                         uint32_t stride,
                         uint32_t kernel,
                         uint32_t fsize,
                         uint32_t fsize_w,//not used
                         uint64_t in1,
                         uint64_t w1,
                         uint64_t b,
                         uint64_t out1,
                         uint32_t act,
                         uint32_t upsample_c,
                         uint32_t upsample_size);
void conv_leakyrelu(uint32_t ch_in,
                    uint32_t ch_out,
                    uint32_t pad,//not used
                    uint32_t stride,
                    uint32_t kernel,
                    uint32_t fsize,
                    uint32_t fsize_w,//not used
                    int16_t* in1,
                    int16_t* w1,
                    int16_t* b,
                    int16_t* out1,
                    uint32_t act,
                    uint32_t upsample_c,
                    uint32_t upsample_size);
void sampling(int16_t* in,int16_t* out,uint32_t ch,uint32_t fsize,uint32_t mode);

#ifdef __cplusplus
}
#endif

#endif