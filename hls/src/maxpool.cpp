#include"maxpool.h"

#define Tn 32    // channel
#define Tr 13   // row
#define Tc 13   // col


//load in[n:n+Tn][r*2:r*2+Tr*2][c*2:c*2+Tc*2]

void maxpool_load_input(const axi_t* in,
                data_t fm_in_buff[Tn][2*Tr][2*Tc],
                unsigned short n, unsigned short r, unsigned short c,
                unsigned short fsize,
                int size) {
#pragma HLS ARRAY_PARTITION variable=fm_in_buff cyclic factor=2 dim=3

// #pragma HLS INLINE off
    int base_addr = n * size + 2 * r * fsize + 2 * c;

    for (unsigned short nn = 0; nn < Tn; nn++) {
        for (unsigned short ii = 0; ii < 2 * Tr; ii++) {
            int offset = base_addr + nn * size + ii * fsize;
            
            int start_16 = offset;
            bool is_odd_start = start_16 & 1; // 记录起点是不是奇数
            // cout<<"n="<<n<<" r="<<r<<" c="<<c<<" offset="<<offset<<" start_16="<<start_16<<" is_odd_start="<<is_odd_start<<endl;
            int start_32 = start_16 / 2;
            // int end_32   = (start_16 + (2 * Tc - 1)) / 2;
            // int read_len = end_32 - start_32 + 1;
            // cout<<"start_32="<<start_32<<" end_32="<<end_32<<" read_len="<<read_len<<"\n"<<endl;

            for (unsigned short k = 0; k < Tc + 1; ++k) {
                #pragma HLS PIPELINE II=1
                axi_t tmp = *(in + start_32 + k);
                data_t val_low, val_high;
                val_low(15, 0)  = tmp(15, 0);
                val_high(15, 0) = tmp(31, 16);
                
                int out_idx_low  = k * 2 - is_odd_start;
                int out_idx_high = out_idx_low + 1;
                
                if (out_idx_low >= 0 && out_idx_low < 2 * Tc) {
                    fm_in_buff[nn][ii][out_idx_low] = val_low;
                }
                if (out_idx_high >= 0 && out_idx_high < 2 * Tc) {
                    fm_in_buff[nn][ii][out_idx_high] = val_high;
                }
            }
        }
    }
}

void maxpool_compute(data_t fm_in_buff[Tn][Tr*2][Tc*2],
			 data_t fm_out_buff[Tn][Tr][Tc]){
// #pragma HLS INLINE off
	data_t tmp1,tmp2,tmp3,tmp4;
	data_t max1,max2,max;
	for(unsigned short n=0;n<Tn;n++)
		for(unsigned short i=0;i<Tr;i++)
			for(unsigned short j=0;j<Tc;j++){
#pragma HLS PIPELINE II=1
				tmp1=fm_in_buff[n][i*2][j*2];
				tmp2=fm_in_buff[n][i*2][j*2+1];
				tmp3=fm_in_buff[n][i*2+1][j*2];
				tmp4=fm_in_buff[n][i*2+1][j*2+1];
				max1=(tmp1>tmp2)?tmp1:tmp2;
				max2=(tmp3>tmp4)?tmp3:tmp4;
				max =(max1>max2)?max1:max2;
				fm_out_buff[n][i][j]=max;
			}
}



void maxpool_store_output(axi_t* out1, data_t* out2,
                  data_t fm_out_buff[Tn][Tr][Tc],
                  unsigned short n, unsigned short r, unsigned short c,
                  unsigned short fsize,
                  int size) {
// #pragma HLS ARRAY_PARTITION variable=fm_out_buff cyclic factor=2 dim=3

    int base_addr = n * size + r * fsize + c;
    
    for (unsigned short nn = 0; nn < Tn; nn++) {
        for (unsigned short ii = 0; ii < Tr; ii++) {
			for (unsigned short t = 0; t < Tc; t++) {
            #pragma HLS PIPELINE II=1
				*(out2 + base_addr + nn * size + ii * fsize + t) = fm_out_buff[nn][ii][t];
			}
        }
    }

    // // ===== 打包写 out1 =====
    // for (unsigned short nn = 0; nn < Tn; nn++)
    //     for (unsigned short ii = 0; ii < Tr; ii++) {
    //         int base = base_addr + nn * size + ii * fsize;
    //         bool is_odd_start = base & 1;
    //         unsigned short pair_num = (Tc - is_odd_start) / 2;
    //         int base_32 = (base + is_odd_start) / 2;

    //         for (unsigned short p = 0; p < pair_num; p++) {
    //         #pragma HLS PIPELINE II=1
    //             unsigned short t_lo = is_odd_start + p * 2;
    //             unsigned short t_hi = t_lo + 1;
    //             axi_t packed;
    //             packed(15, 0)  = fm_out_buff[nn][ii][t_lo](15, 0);
    //             packed(31, 16) = fm_out_buff[nn][ii][t_hi](15, 0);
    //             *(out1 + base_32 + p) = packed;
    //         }
    //     }

    // // ===== 落单写 out2 =====
    // for (unsigned short nn = 0; nn < Tn; nn++)
    //     for (unsigned short ii = 0; ii < Tr; ii++) {
    //     #pragma HLS PIPELINE II=2
    //         int base = base_addr + nn * size + ii * fsize;
    //         bool is_odd_start = base & 1;
    //         unsigned short pair_num = (Tc - is_odd_start) / 2;
    //         unsigned short covered  = is_odd_start + pair_num * 2;

    //         if (is_odd_start) {
    //             *(out2 + base) = fm_out_buff[nn][ii][0];
    //         }
    //         if (covered < Tc) {
    //             *(out2 + base + (Tc - 1)) = fm_out_buff[nn][ii][Tc - 1];
    //         }
    //     }
}

void maxpool_tile(const axi_t* in, axi_t* out1, data_t* out2,
                  unsigned short n, unsigned short r, unsigned short c,
                  int h, int size) {
    data_t fm_in_buff[Tn][Tr*2][Tc*2];
    #pragma HLS ARRAY_PARTITION variable=fm_in_buff cyclic factor=2 dim=3
    data_t fm_out_buff[Tn][Tr][Tc];
    // #pragma HLS ARRAY_PARTITION variable=fm_out_buff cyclic factor=2 dim=3

    // 提前完成所有计算和类型转换，消除隐式临时变量
    unsigned short h_cast = (unsigned short)h;
    unsigned short h_half = (unsigned short)(h / 2);
    int size_quarter = size / 4;

    #pragma HLS DATAFLOW
    // 传入干净的变量，符合 Dataflow 的规范形态
    maxpool_load_input(in, fm_in_buff, n, r, c, h_cast, size);
    maxpool_compute(fm_in_buff, fm_out_buff);
    maxpool_store_output(out1, out2, fm_out_buff, n, r, c, h_half, size_quarter);
}


void maxpool(const axi_t* in, axi_t* out1, data_t* out2, int ch, int h, int w){
#pragma HLS INLINE off
	int size=h*w;
	for(unsigned short n=0;n<ch;n+=Tn)
		for(unsigned short i=0;i<h/2;i+=Tr)
			for(unsigned short j=0;j<w/2;j+=Tc){
                maxpool_tile(in, out1, out2, n, i, j, h, size);
			}
}
