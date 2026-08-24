#include"top.h"


void conv(const axi_t* in1,
		  const axi_t* w1,
		  const axi_t* b,
		  axi_t* out1,
		  data_t* out2,
		  int ch_in,int ch_out,
		  int fsize,int stride,int kernel,int act,int upsample_c,int upsample_size){
// AXI4-Lite
#pragma HLS INTERFACE s_axilite port=ch_in bundle=CTRL
#pragma HLS INTERFACE s_axilite port=ch_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=fsize bundle=CTRL
#pragma HLS INTERFACE s_axilite port=stride bundle=CTRL
#pragma HLS INTERFACE s_axilite port=kernel bundle=CTRL
#pragma HLS INTERFACE s_axilite port=act bundle=CTRL
#pragma HLS INTERFACE s_axilite port=upsample_c bundle=CTRL
#pragma HLS INTERFACE s_axilite port=upsample_size bundle=CTRL

#pragma HLS INTERFACE s_axilite port=in1 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=w1 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=b bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out1 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out2 bundle=CTRL


#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

/*
num_read_outstanding=<int> 和 num_write_outstanding=<int>
*/

#pragma HLS INTERFACE m_axi depth=60000*4 port=in1 	offset=slave bundle=R  max_read_burst_length=256 num_write_outstanding = 1
#pragma HLS INTERFACE m_axi depth=65536*4 port=w1 	offset=slave bundle=R  max_read_burst_length=256 num_write_outstanding = 1
#pragma HLS INTERFACE m_axi depth=128*4   port=b 	offset=slave bundle=R  max_read_burst_length=256 num_write_outstanding = 1
#pragma HLS INTERFACE m_axi 			  port=out1 offset=slave bundle=W1 max_write_burst_length=256 num_read_outstanding = 1
#pragma HLS INTERFACE m_axi 			  port=out2 offset=slave bundle=W2 max_write_burst_length=256 num_read_outstanding = 1

	if(kernel==3){
		// std_conv(in1,
		// 		 w1,
		// 		 b,
		// 		 out1,
		// 		 ch_in,ch_out,
		// 		 fsize,stride,act);
		// int pad = (kernel == 3) ? 1 : 0;
		
			unified_conv(in1, w1, b, out1, out2, ch_in, ch_out, fsize, stride, act, MODE_STD); 		
	}
	else if(kernel==1){
		// pwconv(in1,w1,b,out1,ch_in,ch_out,fsize,act);
			unified_conv(in1, w1, b, out1, out2, /*ch_in=*/ch_in, /*ch_out=*/ch_out,
					/*fm_size=*/fsize, /*stride=*/1, /*act=*/act, MODE_PW);
	}
	else if(kernel==2){     //kernel==2 maxpool; kernel==0 upsample
			maxpool(in1,out1,out2,ch_in,fsize,fsize);
	}
	else{
			upsample(in1,out1,out2,upsample_c,upsample_size);
	}
	
}
