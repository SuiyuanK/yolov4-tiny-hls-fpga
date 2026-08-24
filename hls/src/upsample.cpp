#include"upsample.h"

#define MAX_SIZE 52 // 根据实际情况调整

// int last_read_input = 0, last_read_weight = 0, last_read_bias = 0, last_write = 0; // for debug

// 128*13*13 -> 128*26*26
// (c,size,size) -> (c,2*size,2*size)
void upsample_load_input(const axi_t* in, data_t* line_out_0, int n, int i, int size) {
    int start_16 = n * size * size + i * size; 
    bool is_odd_start = start_16 & 1; // 记录起点是不是奇数 (0或1)
    // cout<<"n="<<n<<" i="<<i<<" start_16="<<start_16<<" is_odd_start="<<is_odd_start<<"\n"<<endl; // for debug

    int start_32 = start_16 / 2;
    int end_32   = (start_16 + (size - 1)) / 2;
    int read_len = end_32 - start_32 + 1; 

    for (int k = 0; k < read_len; ++k) {
    #pragma HLS PIPELINE II=1
        axi_t tmp = *(in + start_32 + k);
        
        data_t val_low, val_high;
        val_low(15, 0)  = tmp(15, 0);
        val_high(15, 0) = tmp(31, 16);
        
        int out_idx_low  = k * 2 - is_odd_start;
        int out_idx_high = out_idx_low + 1;
        
        if (out_idx_low >= 0 && out_idx_low < size) {
            line_out_0[2 * out_idx_low]     = val_low;
            line_out_0[2 * out_idx_low + 1] = val_low;
        }
        if (out_idx_high >= 0 && out_idx_high < size) {
            line_out_0[2 * out_idx_high]     = val_high;
            line_out_0[2 * out_idx_high + 1] = val_high;
        }
    }
}

void upsample_store_output(axi_t* out1, data_t* out2, data_t* line_out_0, int n, int i, int size) {
    int os = 2 * size;
    int wr_addr_0 = n * os * os + (2 * i) * os;
    int wr_addr_1 = wr_addr_0 + os;

    // 两行地址各自打包写
    int addrs[2];
    addrs[0] = wr_addr_0;
    addrs[1] = wr_addr_1;

    for (int line = 0; line < 2; line++) {
        int base = addrs[line];
        bool is_odd_start = base & 1;
        int pair_num = (os - is_odd_start) / 2;
        int base_32 = (base + is_odd_start) / 2;

        // 打包
        for (int p = 0; p < pair_num; p++) {
        #pragma HLS PIPELINE II=1
            int k_lo = is_odd_start + p * 2;
            axi_t packed;
            packed(15, 0)  = line_out_0[k_lo](15, 0);
            packed(31, 16) = line_out_0[k_lo + 1](15, 0);
            *(out1 + base_32 + p) = packed;
        }
        // 落单
        if (is_odd_start) {
            *(out2 + base) = line_out_0[0];
        }
        int covered = is_odd_start + pair_num * 2;
        if (covered < os) {
            *(out2 + base + (os - 1)) = line_out_0[os - 1];
        }
    }
}

void upsample_tile(const axi_t* in, axi_t* out1, data_t* out2, int n, int i, int size) {
    data_t line_out_0[MAX_SIZE*2];
    // #pragma HLS DATAFLOW
    upsample_load_input(in, line_out_0, n, i, size);
    upsample_store_output(out1, out2, line_out_0, n, i, size);
}

void upsample(const axi_t* in, axi_t* out1, data_t* out2, int c, int size) {
#pragma HLS INLINE off
    for (int n = 0; n < c; n++) {
        for (int i = 0; i < size; i++) {
            upsample_tile(in, out1, out2, n, i, size);
        }
    }
}



// void upsample(data_t* in,data_t* out, int c, int size){
// 	data_t tmp;
// 	int os = 2*size; // out_size
// 	for(unsigned short n=0;n<c;n++)
// 		for(unsigned short i=0;i<size;i++)
// 			for(unsigned short j=0;j<size;j++){
// #pragma HLS PIPELINE
// 				int wr_base_addr=n*os*os + 2*i*os + 2*j;
// 				int rd_base_addr=n*size*size + i*size;
// 				tmp=*(in+rd_base_addr+j);
// 				*(out+wr_base_addr)=tmp;
// 				*(out+wr_base_addr+1)=tmp;
// 				*(out+wr_base_addr+os)=tmp;
// 				*(out+wr_base_addr+os+1)=tmp;
// 			}
// 	// cout<<"last_read_input="<<last_read_input<<"\n"<<endl; // for debug
// 	// cout<<"last_write="<<last_write<<"\n"<<endl; // for debug
// }


// void upsample_load_input(data_t* in, hls::stream<data_t>& fifo_out, int n, int i, int size) {
//     for (int jj = 0; jj < size; jj++) {
//     #pragma HLS PIPELINE II=2
//         data_t val = *(in + (n * size * size + i * size) + jj);
//         fifo_out.write(val);
//         fifo_out.write(val); 
//     }
// }

// void upsample_store_output(data_t* out, hls::stream<data_t>& fifo_out, int n, int i, int size) {
//     int os = 2 * size;
//     int wr_addr_0 = n * os * os + (2 * i) * os;
//     int wr_addr_1 = wr_addr_0 + os;
    
//     data_t row_cache[MAX_SIZE * 2]; 

//     for (int kk = 0; kk < os; kk++) {
//     #pragma HLS PIPELINE II=1
//         data_t val = fifo_out.read(); 
//         *(out + wr_addr_0 + kk) = val;
//         row_cache[kk] = val; 
//     }
    
//     for (int kk = 0; kk < os; kk++) {
//     #pragma HLS PIPELINE II=1
//         *(out + wr_addr_1 + kk) = row_cache[kk]; 
//     }
// }

// void upsample_tile(data_t* in, data_t* out, int n, int i, int size) {
//     hls::stream<data_t> fifo_out("fifo_out");
//     #pragma HLS STREAM depth=MAX_SIZE*2 variable=fifo_out 
//     #pragma HLS DATAFLOW
//     upsample_load_input(in, fifo_out, n, i, size);
//     upsample_store_output(out, fifo_out, n, i, size);
// }

// void upsample(data_t* in, data_t* out, int c, int size) {
// #pragma HLS INLINE off
//     for (int n = 0; n < c; n++) {
//         for (int i = 0; i < size; i++) {
//             upsample_tile(in, out, n, i, size);
//         }
//     }
// }