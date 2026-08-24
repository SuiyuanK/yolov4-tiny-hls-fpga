#include "basic_op.h"

#include <stdio.h>

#include "common_func.h"

#define CONV_CTRL_BASEADDR  (0x1f300000u)

// #define CONV_REG_AP_CTRL       (0x00u)
// #define CONV_REG_IN1_L         (0x10u)
// #define CONV_REG_IN1_H         (0x14u)
// #define CONV_REG_W1_L          (0x1cu)
// #define CONV_REG_W1_H          (0x20u)
// #define CONV_REG_B_L           (0x28u)
// #define CONV_REG_B_H           (0x2cu)
// #define CONV_REG_OUT1_L        (0x34u)
// #define CONV_REG_OUT1_H        (0x38u)
// #define CONV_REG_CH_IN         (0x40u)
// #define CONV_REG_CH_OUT        (0x48u)
// #define CONV_REG_FSIZE         (0x50u)
// #define CONV_REG_STRIDE        (0x58u)
// #define CONV_REG_KERNEL        (0x60u)
// #define CONV_REG_ACT           (0x68u)
// #define CONV_REG_UPSAMPLE_C    (0x70u)
// #define CONV_REG_UPSAMPLE_SIZE (0x78u)

// #define CONV_AP_START       (0x01u)
// #define CONV_AP_IDLE        (0x04u)

#define CONV_REG_AP_CTRL       (0x00u)
#define CONV_REG_IN1_L         (0x10u)
#define CONV_REG_IN1_H         (0x14u)
#define CONV_REG_W1_L          (0x1cu)
#define CONV_REG_W1_H          (0x20u)
#define CONV_REG_B_L           (0x28u)
#define CONV_REG_B_H           (0x2cu)
#define CONV_REG_OUT1_L        (0x34u)
#define CONV_REG_OUT1_H        (0x38u)
#define CONV_REG_OUT2_L        (0x40u) 
#define CONV_REG_OUT2_H        (0x44u) 
#define CONV_REG_CH_IN         (0x4cu) 
#define CONV_REG_CH_OUT        (0x54u) 
#define CONV_REG_FSIZE         (0x5cu) 
#define CONV_REG_STRIDE        (0x64u) 
#define CONV_REG_KERNEL        (0x6cu) 
#define CONV_REG_ACT           (0x74u) 
#define CONV_REG_UPSAMPLE_C    (0x7cu) 
#define CONV_REG_UPSAMPLE_SIZE (0x84u) 

// Control bit masks
#define CONV_AP_START       (0x01u) // bit 0
#define CONV_AP_DONE        (0x02u) // bit 1
#define CONV_AP_IDLE        (0x04u) // bit 2
#define CONV_AP_READY       (0x08u) // bit 3

static void conv_write64(unsigned int offset, uint64_t value)
{
    RegWrite(CONV_CTRL_BASEADDR + offset, (unsigned int)(value & 0xffffffffu));
    RegWrite(CONV_CTRL_BASEADDR + offset + 4u, (unsigned int)((value >> 32) & 0xffffffffu));
}

static uint64_t conv_ptr_to_u64(const void* ptr)
{
    return (uint64_t)(uintptr_t)ptr;
}

void conv_init(void)
{
    (void)RegRead(CONV_CTRL_BASEADDR + CONV_REG_AP_CTRL);
}

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
                         uint32_t upsample_c, //channel 数
                         uint32_t upsample_size) //c, size用于设置upsample算子  yolov4-tiny 固定为 c=128, size=13
{
    unsigned int status;

    conv_init();

    conv_write64(CONV_REG_IN1_L, in1);
    conv_write64(CONV_REG_W1_L, w1);
    conv_write64(CONV_REG_B_L, b);
    conv_write64(CONV_REG_OUT1_L, out1);
    conv_write64(CONV_REG_OUT2_L, out1);

    RegWrite(CONV_CTRL_BASEADDR + CONV_REG_CH_IN, ch_in);
    RegWrite(CONV_CTRL_BASEADDR + CONV_REG_CH_OUT, ch_out);
    RegWrite(CONV_CTRL_BASEADDR + CONV_REG_FSIZE, fsize);
    RegWrite(CONV_CTRL_BASEADDR + CONV_REG_STRIDE, stride);
    RegWrite(CONV_CTRL_BASEADDR + CONV_REG_KERNEL, kernel);
    RegWrite(CONV_CTRL_BASEADDR + CONV_REG_ACT, act);
    RegWrite(CONV_CTRL_BASEADDR + CONV_REG_UPSAMPLE_C, upsample_c);
    RegWrite(CONV_CTRL_BASEADDR + CONV_REG_UPSAMPLE_SIZE, upsample_size);

    RegWrite(CONV_CTRL_BASEADDR + CONV_REG_AP_CTRL, CONV_AP_START);

    while (1) {
        status = RegRead(CONV_CTRL_BASEADDR + CONV_REG_AP_CTRL);
        if ((status & CONV_AP_IDLE) != 0u) {
            return;
        }
    }
}

void conv_leakyrelu(uint32_t ch_in,
                    uint32_t ch_out,
                    uint32_t pad,
                    uint32_t stride,
                    uint32_t kernel,
                    uint32_t fsize,
                    uint32_t fsize_w,
                    int16_t* in1,
                    int16_t* w1,
                    int16_t* b,
                    int16_t* out1,
                    uint32_t act,
                    uint32_t upsample_c,
                    uint32_t upsample_size)
{
    conv_leakyrelu_addr(ch_in, ch_out, pad, stride, kernel, fsize, fsize_w,
                        conv_ptr_to_u64(in1), conv_ptr_to_u64(w1), conv_ptr_to_u64(b), conv_ptr_to_u64(out1),
                        act, upsample_c, upsample_size);
}

void sampling(int16_t* in,int16_t* out,uint32_t ch,uint32_t fsize,uint32_t mode){
	if(mode==1)   //maxpool
	    conv_leakyrelu(ch,ch,0,0,2,fsize,fsize,in,in,in,out,0,128,13);
	else //upsample
		conv_leakyrelu(ch,ch,0,0,0,fsize,fsize,in,in,in,out,0,128,13);
}





