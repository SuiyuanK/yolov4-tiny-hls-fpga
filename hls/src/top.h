// #include"pwconv.h"
// #include"std_conv.h"
// #include"std_conv_k3_s1.h"
// #include"std_conv_k3_s2.h"
#include"maxpool.h"
#include"upsample.h"
#include"unified_conv.h"

// include std_conv, pwconv, maxpool, upsample 
void conv(const axi_t* in1,
		  const axi_t* w1,
		  const axi_t* b,
		  axi_t* out1,
		  data_t* out2,
		  int ch_in,int ch_out,
		  int fsize,int stride,int kernel,int act,int upsample_c,int upsample_size);
