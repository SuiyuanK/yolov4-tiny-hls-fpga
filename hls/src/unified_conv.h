#ifndef UNIFIED_CONV_H
#define UNIFIED_CONV_H

#include "type.h"

// ========== 卷积模式 ==========
#define MODE_STD  0           // 3x3 标准卷积 (stride=1/2)
#define MODE_PW   1           // 1x1 point-wise

void unified_conv(const axi_t* in1,
                  const axi_t* w1,
                  const axi_t* bias,
                  axi_t* out1,
                  data_t* out2,
                  unsigned short ch_in,
                  unsigned short ch_out,
                  unsigned short fm_size,
                  unsigned short stride,
                  unsigned short act,
                  bool mode);

#endif