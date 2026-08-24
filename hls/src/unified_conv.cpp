#include "unified_conv.h"

// ========== Tile 参数（合并后统一） ==========
#define Tr   13
#define Tc   13
#define Tp   (Tr*Tc)          // 169
#define Tn   4
#define Tm   64
#define K    3                // 最大 kernel
#define P    1
#define MAX_LEN 512

#define TRin ((Tr-1)*2+K)     // 27
#define TCin ((Tr-1)*2+K)     // 27


// =========================================================
// 激活
// =========================================================
static data_t leaky_relu(data_t x){
// #pragma HLS INLINE off
    const data_t alpha=(data_t)0.1;
    // cout << "x=" << x << ", x*alpha=" << (x*alpha) << ", approx=" << (x(15,5) + x(15,4) + x(15,1) + x)(15,0) << endl;
	if(x>=(data_t)0){
		return x;
	}
	else{
        return x*alpha;
        // 0.1(data_t) = 51(uint) = 32+16+2+1 
		// return (x(15,5) + x(15,4) + x(15,1) + x)(15,0);
	}
}

// =========================================================
// LOAD INPUT（统一）
//   std : 加载 [Tn][TRin][TCin]，含 padding
//   pw  : 加载 Tp 个像素，平铺进 fm_in_buff[nn][rr_idx][cc_idx]
//         其中 (rr_idx, cc_idx) 与 (rr, cc) 同范围 [0,Tr)
// =========================================================
static void load_input_unified(data_t fm_in_buff[Tn][TRin][TCin],
                               const axi_t* in1,
                               unsigned short n,
                               unsigned short fm_row, unsigned short fm_col,
                               unsigned short fm_size,
                               unsigned short stride,
                               unsigned short ch_in,
                               unsigned short basePixAddr,
                               unsigned short mode)
{
    #pragma HLS ARRAY_PARTITION variable=fm_in_buff cyclic factor=2 dim=2
    #pragma HLS ARRAY_PARTITION variable=fm_in_buff cyclic factor=2 dim=3
    #pragma HLS DEPENDENCE variable=fm_in_buff type=intra false
    unsigned short nn, rr, cc;
    ap_uint<18> size = fm_size * fm_size;
    unsigned short actual_tn = ((ch_in - n) < Tn) ? (ch_in - n) : Tn;

    if (mode == MODE_PW) {
        // 1x1：连续读 Tp 个像素，平铺到 [rr][cc]，rr ∈ [0,Tr), cc ∈ [0,Tc)
        int base_addr = n * size + basePixAddr;
        for (nn = 0; nn < actual_tn; nn++) {
            int start_16 = base_addr + nn * size;
            bool is_odd_start = start_16 & 1;
            int start_32 = start_16 / 2;

            for (unsigned short k = 0; k < Tp / 2 + 1; ++k) {
            #pragma HLS PIPELINE II=1
                axi_t tmp_axi = *(in1 + start_32 + k);
                data_t v0, v1;
                v0(15, 0) = tmp_axi(15, 0);
                v1(15, 0) = tmp_axi(31, 16);
                
                int out_idx_low = k * 2 - is_odd_start;
                int out_idx_high = out_idx_low + 1;
                
                if (out_idx_low >= 0 && out_idx_low < Tp) {
                    fm_in_buff[nn][out_idx_low / Tc][out_idx_low % Tc] = v0;
                }
                if (out_idx_high >= 0 && out_idx_high < Tp) {
                    fm_in_buff[nn][out_idx_high / Tc][out_idx_high % Tc] = v1;
                }
            }
        }
        return;
    }


    // -------- std 模式 --------
    int base_addr1 = n * size + (fm_row - P) * fm_size + (fm_col - P);
    int base_addr2 = base_addr1 + fm_row * fm_size + fm_col;

    if (stride == 1) {
        for (nn = 0; nn < actual_tn; nn++)
            for (rr = 0; rr < Tr + K - 1; rr++) {
                int rr_offset = rr * fm_size;
                int start_16 = base_addr1 + nn * size + rr_offset;
                bool is_odd_start = start_16 & 1;
                int start_32 = start_16 / 2;
                
                for (unsigned short k = 0; k < (Tc + K - 1) / 2 + 1; ++k) {
                #pragma HLS PIPELINE II=1
                    axi_t tmp_axi = *(in1 + start_32 + k);
                    data_t v0, v1;
                    v0(15, 0) = tmp_axi(15, 0);
                    v1(15, 0) = tmp_axi(31, 16);
                    
                    int out_idx_low = k * 2 - is_odd_start;
                    int out_idx_high = out_idx_low + 1;

                    if (out_idx_low >= 0 && out_idx_low < Tc + K - 1) {
                        bool in_range0 = (fm_row + rr >= P && fm_row + rr < fm_size + P &&
                                          fm_col + out_idx_low >= P && fm_col + out_idx_low < fm_size + P);
                        fm_in_buff[nn][rr][out_idx_low] = in_range0 ? v0 : (data_t)0;
                    }
                    if (out_idx_high >= 0 && out_idx_high < Tc + K - 1) {
                        bool in_range1 = (fm_row + rr >= P && fm_row + rr < fm_size + P &&
                                          fm_col + out_idx_high >= P && fm_col + out_idx_high < fm_size + P);
                        fm_in_buff[nn][rr][out_idx_high] = in_range1 ? v1 : (data_t)0;
                    }
                }
            }
    } else { // stride==2
        for (nn = 0; nn < actual_tn; nn++)
            for (rr = 0; rr < TRin; rr++) {
                int rr_offset = rr * fm_size;
                int start_16 = base_addr2 + nn * size + rr_offset;
                bool is_odd_start = start_16 & 1;
                int start_32 = start_16 / 2;
                
                for (unsigned short k = 0; k < TCin / 2 + 1; ++k) {
                #pragma HLS PIPELINE II=1
                    axi_t tmp_axi = *(in1 + start_32 + k);
                    data_t v0, v1;
                    v0(15, 0) = tmp_axi(15, 0);
                    v1(15, 0) = tmp_axi(31, 16);
                    
                    int out_idx_low = k * 2 - is_odd_start;
                    int out_idx_high = out_idx_low + 1;
                    
                    if (out_idx_low >= 0 && out_idx_low < TCin) {
                        bool in_range0 = (fm_row*2 + rr >= P && fm_row*2 + rr < fm_size + P &&
                                          fm_col*2 + out_idx_low >= P && fm_col*2 + out_idx_low < fm_size + P);
                        fm_in_buff[nn][rr][out_idx_low] = in_range0 ? v0 : (data_t)0;
                    }
                    if (out_idx_high >= 0 && out_idx_high < TCin) {
                        bool in_range1 = (fm_row*2 + rr >= P && fm_row*2 + rr < fm_size + P &&
                                          fm_col*2 + out_idx_high >= P && fm_col*2 + out_idx_high < fm_size + P);
                        fm_in_buff[nn][rr][out_idx_high] = in_range1 ? v1 : (data_t)0;
                    }
                }
            }
    }
}




// -------- std 模式 (FIFO + dataflow 版本) --------

// 子函数 1: 从 AXI 读数据推入 FIFO
static void load_axi_stream(
    const axi_t *w1,
    hls::stream<axi_t> &fifo,
    int base_addr,
    int actual_tm,
    int ch_in_kk
) {
    for (int mm = 0; mm < actual_tm; mm++) {
        int start_16 = base_addr + mm * ch_in_kk;
        int start_32 = start_16 / 2;
        
        for (int b = 0; b < (Tn * K * K) / 2 + 1; b++) {
        #pragma HLS PIPELINE II=1
            fifo.write(*(w1 + start_32 + b));
        }
    }
}

// 子函数 2: 从 FIFO 取数据，拆包写入 wt_buff
static void distribute_to_buff(
    hls::stream<axi_t> &fifo,
    data_t wt_buff[Tm][Tn][K][K],
    int n,
    int ch_in,
    int actual_tm,
    int base_addr,
    int ch_in_kk
) {
    for (int mm = 0; mm < actual_tm; mm++) {
        int start_16 = base_addr + mm * ch_in_kk;
        bool is_odd_start = start_16 & 1;
        int start_32 = start_16 / 2;

        for (int k = 0; k < (Tn * K * K) / 2 + 1; ++k) {
        #pragma HLS PIPELINE II=2
            axi_t tmp_axi = fifo.read();
            data_t v0, v1;
            v0(15, 0) = tmp_axi(15, 0);
            v1(15, 0) = tmp_axi(31, 16);

            int out_idx_low = k * 2 - is_odd_start;
            int out_idx_high = out_idx_low + 1;

            if (out_idx_low >= 0 && out_idx_low < Tn * K * K) {
                unsigned short nn_0 = out_idx_low / (K * K);
                unsigned short i0   = (out_idx_low % (K * K)) / K;
                unsigned short j0   = out_idx_low % K;
                wt_buff[mm][nn_0][i0][j0] = ((n + nn_0) < ch_in) ? v0 : (data_t)0;
            }

            if (out_idx_high >= 0 && out_idx_high < Tn * K * K) {
                unsigned short nn_1 = out_idx_high / (K * K);
                unsigned short i1   = (out_idx_high % (K * K)) / K;
                unsigned short j1   = out_idx_high % K;
                wt_buff[mm][nn_1][i1][j1] = ((n + nn_1) < ch_in) ? v1 : (data_t)0;
            }
        }
    }
}

// 顶层 wrapper: 使用 dataflow
void load_weight_std(
    const axi_t *w1,
    data_t wt_buff[Tm][Tn][K][K],
    int m, int n,
    int ch_in,
    int actual_tm
) {

    int ch_in_kk  = ch_in * K * K;
    int base_addr = m * ch_in_kk + n * K * K;

    hls::stream<axi_t> fifo;
    #pragma HLS STREAM variable=fifo
    #pragma HLS DATAFLOW

    load_axi_stream     (w1, fifo, base_addr, actual_tm, ch_in_kk);
    distribute_to_buff  (fifo, wt_buff, n, ch_in, actual_tm, base_addr, ch_in_kk);
}

// =========================================================
// LOAD WEIGHT（统一）
//   std : weight 布局 [M][N][K][K]
//   pw  : weight 布局 [M][N]，存放到 wt_buff[mm][nn][0][0]，其余写 0
// =========================================================
static void load_weight_unified(data_t wt_buff[Tm][Tn][K][K],
                                const axi_t* w1,
                                unsigned short n, unsigned short m,
                                unsigned short ch_in, unsigned short ch_out,
                                unsigned short mode)
{
    // #pragma HLS ARRAY_PARTITION variable=wt_buff complete dim=2

    unsigned short mm, nn, k;
    unsigned short actual_tm = ((ch_out - m) < Tm) ? (ch_out - m) : Tm;

    if (mode == MODE_PW) {
        for (mm = 0; mm < actual_tm; mm++) {
            int start_16 = (m + mm) * ch_in + n;
            bool is_odd_start = start_16 & 1;
            int start_32 = start_16 / 2;

            for (unsigned short k = 0; k < Tn / 2 + 1; ++k) {
            #pragma HLS PIPELINE II=1
                axi_t tmp_axi = *(w1 + start_32 + k);
                data_t v0, v1;
                v0(15, 0) = tmp_axi(15, 0);
                v1(15, 0) = tmp_axi(31, 16);
                
                int out_idx_low = k * 2 - is_odd_start;
                int out_idx_high = out_idx_low + 1;
                
                if (out_idx_low >= 0 && out_idx_low < Tn) {
                    wt_buff[mm][out_idx_low][0][0] = v0;                
                }
                if (out_idx_high >= 0 && out_idx_high < Tn) {
                    wt_buff[mm][out_idx_high][0][0] = v1;
                }
            }
        }
        return;
    }

    // -------- std 模式 --------
    // int ch_in_kk = ch_in * K * K;
    // int base_addr = m * ch_in_kk + n * K * K;
    // for (mm = 0; mm < actual_tm; mm++) {
    //     for (k = 0; k < Tn * K * K; k += 2) {
    //     #pragma HLS PIPELINE II=1
    //         int addr = base_addr + mm * ch_in_kk + k;
    //         axi_t tmp_axi = *(w1 + addr / 2);
    //         data_t v0, v1;
    //         v0 = tmp_axi(15, 0);
    //         v1 = tmp_axi(31, 16);
            
    //         unsigned short nn_0 = k / (K * K);
    //         unsigned short i0   = (k % (K * K)) / K;
    //         unsigned short j0   = k % K;
    //         wt_buff[mm][nn_0][i0][j0] = ((n + nn_0) < ch_in) ? v0 : (data_t)0;
    //         cout << "::" << k/2 << endl;
    //         cout << "Loading weight (" << mm << "," << nn_0 << "," << i0 << "," << j0 << "): " << wt_buff[mm][nn_0][i0][j0] << endl;
    //         if (k + 1 < Tn * K * K) {
    //             unsigned short nn_1 = (k + 1) / (K * K);
    //             unsigned short i1   = ((k + 1) % (K * K)) / K;
    //             unsigned short j1   = (k + 1) % K;
    //             wt_buff[mm][nn_1][i1][j1] = ((n + nn_1) < ch_in) ? v1 : (data_t)0;
    //             cout << "Loading weight (" << mm << "," << nn_1 << "," << i1 << "," << j1 << "): " << wt_buff[mm][nn_1][i1][j1] << endl;
    //         }
    //     }
    // }

    load_weight_std(w1, wt_buff, m, n, ch_in, actual_tm);
}

// =========================================================
// LOAD BIAS（共用）
// =========================================================
static void load_bias_unified(data_t fm_out_buff[Tm][Tr][Tc],
                              data_t bias_buff[MAX_LEN/Tm][Tm],
                              unsigned short m)
{
    unsigned short rr, cc, mm;
    for (rr = 0; rr < Tr; rr++)
        for (cc = 0; cc < Tc; cc++) {
        #pragma HLS PIPELINE II=1
            for (mm = 0; mm < Tm; mm++) {
            #pragma HLS UNROLL
                fm_out_buff[mm][rr][cc] = bias_buff[m / Tm][mm];
            }
        }
}

// =========================================================
// COMPUTE（统一核心，关键资源共享点）
//   std : K_eff=3, S=stride
//   pw  : K_eff=1, S=1
// =========================================================
static void compute_unified(data_t fm_in_buff[Tn][TRin][TCin],
                            data_t fm_out_buff[Tm][Tr][Tc],
                            data_t wt_buff[Tm][Tn][K][K],
                            unsigned short stride,
                            unsigned short mode)
{
#pragma HLS ARRAY_PARTITION variable=fm_in_buff  complete dim=1
#pragma HLS ARRAY_PARTITION variable=fm_out_buff complete dim=1
#pragma HLS ARRAY_PARTITION variable=wt_buff     complete dim=1
#pragma HLS ARRAY_PARTITION variable=wt_buff     complete dim=2

    unsigned short K_eff = (mode == MODE_PW) ? (unsigned short)1 : (unsigned short)K;
    unsigned short S     = (mode == MODE_PW) ? (unsigned short)1 : stride;

    unsigned short kx, ky, rr, cc, nn, mm;
    for (kx = 0; kx < K_eff; kx++){
    #pragma HLS LOOP_TRIPCOUNT min=1 max=3
        for (ky = 0; ky < K_eff; ky++){
        #pragma HLS LOOP_TRIPCOUNT min=1 max=3
            for (rr = 0; rr < Tr; rr++){
                for (cc = 0; cc < Tc; cc++) {
                #pragma HLS PIPELINE II=1
                    for (mm = 0; mm < Tm; mm++) {
                    #pragma HLS UNROLL
                        for (nn = 0; nn < Tn; nn++) {
                        #pragma HLS UNROLL
                            fm_out_buff[mm][rr][cc] +=
                                fm_in_buff[nn][rr*S + kx][cc*S + ky] *
                                wt_buff[mm][nn][kx][ky];
                        }
                    }
                }
            }
        }
    }
    
}

// =========================================================
// STORE OUTPUT（统一）
//   std : 写 [m+mm][fm_row+rr][fm_col+cc]
//   pw  : 写 [m+mm][basePixAddr + i]，i ∈ [0,Tp)
// =========================================================
static void store_output_unified(data_t fm_out_buff[Tm][Tr][Tc],
                                 axi_t* out1,
                                 data_t* out2,
                                 unsigned short fm_row, unsigned short fm_col,
                                 unsigned short m,
                                 unsigned short act,
                                 unsigned short o_fm_size,
                                 unsigned short ch_out,
                                 unsigned short basePixAddr,
                                 unsigned short mode)
{
#pragma HLS ARRAY_PARTITION variable=fm_out_buff cyclic factor=2 dim=3
// #pragma HLS INLINE off
    unsigned short mm, rr, cc;
    unsigned short actual_tm = ((ch_out - m) < Tm) ? (ch_out - m) : Tm;
    ap_uint<18> size = o_fm_size * o_fm_size;
    if (mode == MODE_PW) {
        // ===== PW 打包写 =====
        for (mm = 0; mm < actual_tm; mm++) {
            int base = (m + mm) * size + basePixAddr;   // 该块 Tp 个像素的起始 16bit 地址
            bool is_odd_start = base & 1;
            unsigned short pair_num = (Tp - is_odd_start) / 2;
            int base_32 = (base + is_odd_start) / 2;

            for (unsigned short p = 0; p < pair_num; p++) {
            #pragma HLS PIPELINE II=1
                unsigned short i_lo = is_odd_start + p * 2;
                unsigned short i_hi = i_lo + 1;

                data_t v_lo = fm_out_buff[mm][i_lo / Tc][i_lo % Tc];
                data_t v_hi = fm_out_buff[mm][i_hi / Tc][i_hi % Tc];
                data_t o_lo = (act == 1) ? leaky_relu(v_lo) : v_lo;
                data_t o_hi = (act == 1) ? leaky_relu(v_hi) : v_hi;

                axi_t packed;
                packed(15, 0)  = o_lo(15, 0);
                packed(31, 16) = o_hi(15, 0);
                *(out1 + base_32 + p) = packed;
            }
        }

        // ===== PW 落单写 =====
        for (mm = 0; mm < actual_tm; mm++) {
        #pragma HLS PIPELINE II=2
            int base = (m + mm) * size + basePixAddr;
            bool is_odd_start = base & 1;
            unsigned short pair_num = (Tp - is_odd_start) / 2;
            unsigned short covered  = is_odd_start + pair_num * 2;

            // 奇起点首元素
            if (is_odd_start) {
                data_t v = fm_out_buff[mm][0][0];
                data_t o = (act == 1) ? leaky_relu(v) : v;
                *(out2 + base) = o;
            }
            // 末尾落单 (i = Tp-1)
            if (covered < Tp) {
                unsigned short i_last = Tp - 1;
                data_t v = fm_out_buff[mm][i_last / Tc][i_last % Tc];
                data_t o = (act == 1) ? leaky_relu(v) : v;
                *(out2 + base + i_last) = o;
            }
        }
        return;
    }

    // -------- std 模式 --------
    for (mm = 0; mm < actual_tm; mm++)
        for (rr = 0; rr < Tr; rr++)
            for (cc = 0; cc < Tc; cc++) {
            #pragma HLS PIPELINE II=1
                data_t v = fm_out_buff[mm][rr][cc];
                data_t o = (act == 1) ? leaky_relu(v) : v;
                int base = (m + mm) * size + (fm_row + rr) * o_fm_size + fm_col;
                *(out2 + base + cc) = o;
            }
    // // -------- std 模式 --------
    // // ===== 打包写 out1 =====
    // for (mm = 0; mm < actual_tm; mm++)
    //     for (rr = 0; rr < Tr; rr++) {
    //         int base = (m + mm) * size + (fm_row + rr) * o_fm_size + fm_col;
    //         bool is_odd_start = base & 1;
    //         unsigned short pair_num = (Tc - is_odd_start) / 2;
    //         int base_32 = (base + is_odd_start) / 2;

    //         for (unsigned short p = 0; p < pair_num; p++) {
    //         #pragma HLS PIPELINE II=1
    //             unsigned short cc_lo = is_odd_start + p * 2;
    //             unsigned short cc_hi = cc_lo + 1;
    //             data_t v_lo = fm_out_buff[mm][rr][cc_lo];
    //             data_t v_hi = fm_out_buff[mm][rr][cc_hi];
    //             data_t o_lo = (act == 1) ? leaky_relu(v_lo) : v_lo;
    //             data_t o_hi = (act == 1) ? leaky_relu(v_hi) : v_hi;

    //             axi_t packed;
    //             packed(15, 0)  = o_lo(15, 0);
    //             packed(31, 16) = o_hi(15, 0);
    //             *(out1 + base_32 + p) = packed;
    //         }
    //     }

    // // ===== 落单写 out2 =====
    // for (mm = 0; mm < actual_tm; mm++)
    //     for (rr = 0; rr < Tr; rr++) {
    //     #pragma HLS PIPELINE II=2
    //         int base = (m + mm) * size + (fm_row + rr) * o_fm_size + fm_col;
    //         bool is_odd_start = base & 1;
    //         unsigned short pair_num = (Tc - is_odd_start) / 2;
    //         unsigned short covered  = is_odd_start + pair_num * 2;

    //         if (is_odd_start) {
    //             data_t v = fm_out_buff[mm][rr][0];
    //             data_t o = (act == 1) ? leaky_relu(v) : v;
    //             *(out2 + base) = o;
    //         }
    //         if (covered < Tc) {
    //             data_t v = fm_out_buff[mm][rr][Tc - 1];
    //             data_t o = (act == 1) ? leaky_relu(v) : v;
    //             *(out2 + base + (Tc - 1)) = o;
    //         }
    //     }
}
// static void store_output_unified(data_t fm_out_buff[Tm][Tr][Tc],
//                                  data_t* out1,
//                                  unsigned short fm_row, unsigned short fm_col,
//                                  unsigned short m,
//                                  unsigned short act,
//                                  unsigned short o_fm_size,
//                                  unsigned short ch_out,
//                                  unsigned short basePixAddr,
//                                  unsigned short mode)
// {
// // #pragma HLS INLINE off
//     unsigned short mm, rr, cc;
//     unsigned short actual_tm = ((ch_out - m) < Tm) ? (ch_out - m) : Tm;
//     ap_uint<18> size = o_fm_size * o_fm_size;

//     if (mode == MODE_PW) {
//         // if(actual_tm != 0) cout << "store_output_unified (PW): m=" << m << ", basePixAddr=" << basePixAddr << ", o_fm_size=" << o_fm_size << endl;
//         for (mm = 0; mm < actual_tm; mm++) {
//             for (unsigned short i = 0; i < Tp; i++) {
//             #pragma HLS PIPELINE II=1
//                 unsigned short rr_idx = i / Tc;
//                 unsigned short cc_idx = i % Tc;
//                 data_t v = fm_out_buff[mm][rr_idx][cc_idx];
//                 data_t o = (act == 1) ? leaky_relu(v) : v;
//                 *(out1 + (m + mm) * size + basePixAddr + i) = o;
//             }
//         }
//         return;
//     }

//     // -------- std 模式 --------
//     for (mm = 0; mm < actual_tm; mm++)
//         for (rr = 0; rr < Tr; rr++)
//             for (cc = 0; cc < Tc; cc++) {
//             #pragma HLS PIPELINE II=1
//                 data_t v = fm_out_buff[mm][rr][cc];
//                 data_t o = (act == 1) ? leaky_relu(v) : v;
//                 int base = (m + mm) * size + (fm_row + rr) * o_fm_size + fm_col;
//                 *(out1 + base + cc) = o;
//             }
// }

// =========================================================
// load_in_wt + load_compute（保留 DATAFLOW 包装）
// =========================================================
static void load_in_wt(data_t fm_in_buff[Tn][TRin][TCin],
                       data_t wt_buff[Tm][Tn][K][K],
                       const axi_t* in1, const axi_t* w1,
                       unsigned short ti,
                       unsigned short fm_row, unsigned short fm_col,
                       unsigned short fm_size, unsigned short stride,
                       unsigned short ch_in,
                       unsigned short m, unsigned short ch_out,
                       unsigned short basePixAddr,
                       unsigned short mode)
{
    load_input_unified(fm_in_buff, in1, ti, fm_row, fm_col, fm_size,
                       stride, ch_in, basePixAddr, mode);
    load_weight_unified(wt_buff, w1, ti, m, ch_in, ch_out, mode);
}

static void load_compute(data_t fm_in_buff_load[Tn][TRin][TCin],
                         data_t wt_buff_load[Tm][Tn][K][K],
                         data_t fm_in_buff_comp[Tn][TRin][TCin],
                         data_t wt_buff_comp[Tm][Tn][K][K],
                         data_t fm_out_buff[Tm][Tr][Tc],
                         const axi_t* in1, const axi_t* w1,
                         unsigned short ti,
                         unsigned short fm_row, unsigned short fm_col,
                         unsigned short fm_size, unsigned short stride,
                         unsigned short ch_in,
                         unsigned short m, unsigned short ch_out,
                         unsigned short basePixAddr,
                         unsigned short mode)
{
    #pragma HLS DATAFLOW
    load_in_wt(fm_in_buff_load, wt_buff_load, in1, w1, ti, fm_row, fm_col,
               fm_size, stride, ch_in, m, ch_out, basePixAddr, mode);
    compute_unified(fm_in_buff_comp, fm_out_buff, wt_buff_comp, stride, mode);
}

// =========================================================
// compute_output（保留手动 pingpong：fm_in_buff1/2、wt_buff1/2）
// =========================================================
static void compute_output(data_t fm_out_buff[Tm][Tr][Tc],
                           data_t bias_buff[MAX_LEN/Tm][Tm],
                           const axi_t* in1, const axi_t* w1,
                           unsigned short m,
                           unsigned short fm_size,
                           unsigned short ch_in, unsigned short ch_out,
                           unsigned short fm_row, unsigned short fm_col,
                           unsigned short stride,
                           unsigned short basePixAddr,
                           unsigned short mode)
{
    data_t fm_in_buff1[Tn][TRin][TCin];
    data_t fm_in_buff2[Tn][TRin][TCin];
    data_t wt_buff1[Tm][Tn][K][K];
    data_t wt_buff2[Tm][Tn][K][K];
    #pragma HLS ARRAY_PARTITION variable=fm_in_buff1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=fm_in_buff2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wt_buff1    complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wt_buff1    complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wt_buff2    complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wt_buff2    complete dim=2

    #pragma HLS ARRAY_PARTITION variable=fm_in_buff1 cyclic factor=2 dim=2
    #pragma HLS ARRAY_PARTITION variable=fm_in_buff1 cyclic factor=2 dim=3
    #pragma HLS ARRAY_PARTITION variable=fm_in_buff2 cyclic factor=2 dim=2
    #pragma HLS ARRAY_PARTITION variable=fm_in_buff2 cyclic factor=2 dim=3

    unsigned short ti = 0;

    load_bias_unified(fm_out_buff, bias_buff, m);
    load_in_wt(fm_in_buff1, wt_buff1, in1, w1, ti,
               fm_row, fm_col, fm_size, stride, ch_in, m, ch_out,
               basePixAddr, mode);

    bool pingpong = true;
    for (ti = Tn; ti < ch_in; ti += Tn) {
        if (pingpong) {
            load_compute(fm_in_buff2, wt_buff2, fm_in_buff1, wt_buff1, fm_out_buff,
                         in1, w1, ti, fm_row, fm_col, fm_size, stride,
                         ch_in, m, ch_out, basePixAddr, mode);
            pingpong = false;
        } else {
            load_compute(fm_in_buff1, wt_buff1, fm_in_buff2, wt_buff2, fm_out_buff,
                         in1, w1, ti, fm_row, fm_col, fm_size, stride,
                         ch_in, m, ch_out, basePixAddr, mode);
            pingpong = true;
        }
    }
    // 末尾：让 m 传成 ch_out（actual_tm=0），不真正写权重，只完成最后一拍 compute
    if (pingpong) {
        load_compute(fm_in_buff2, wt_buff2, fm_in_buff1, wt_buff1, fm_out_buff,
                     in1, w1, ch_in, fm_row, fm_col, fm_size, stride,
                     ch_in, ch_out, ch_out, basePixAddr, mode);
    } else {
        load_compute(fm_in_buff1, wt_buff1, fm_in_buff2, wt_buff2, fm_out_buff,
                     in1, w1, ch_in, fm_row, fm_col, fm_size, stride,
                     ch_in, ch_out, ch_out, basePixAddr, mode);
    }
}

// =========================================================
// next_block（统一）
//   std : 沿 (c, r, m) 推进
//   pw  : 沿 (basePixAddr, baseChannel) 推进
// =========================================================
static void next_block_unified(unsigned short r, unsigned short c, unsigned short m,
                               unsigned short &next_r, unsigned short &next_c, unsigned short &next_m,
                               unsigned short basePixAddr, unsigned short baseChannel,
                               unsigned short &next_basePixAddr, unsigned short &next_baseChannel,
                               unsigned short o_fm_size,
                               unsigned short mode)
{
    if (mode == MODE_PW) {
        if (basePixAddr + Tp >= (o_fm_size * o_fm_size)) {
            next_basePixAddr = 0;
            next_baseChannel = baseChannel + Tm;
        } else {
            next_basePixAddr = basePixAddr + Tp;
            next_baseChannel = baseChannel;
        }
        // 保持 std 输出参数稳定
        next_r = 0; next_c = 0; next_m = 0;
    } else {
        if (c + Tc >= o_fm_size) {
            if (r + Tr >= o_fm_size) { next_m = m + Tm; next_r = 0;      next_c = 0; }
            else                     { next_m = m;      next_r = r + Tr; next_c = 0; }
        } else {
            next_m = m; next_r = r; next_c = c + Tc;
        }
        next_basePixAddr = 0; next_baseChannel = 0;
    }
}






// =========================================================
// compute_store（保留手动 pingpong 的 DATAFLOW 包装）
// =========================================================
static void compute_store(data_t fm_out_buff_comp[Tm][Tr][Tc],
                          data_t fm_out_buff_store[Tm][Tr][Tc],
                          data_t bias_buff[MAX_LEN/Tm][Tm],
                          const axi_t* in1, const axi_t* w1, axi_t* out1, data_t* out2,
                          // compute 的目标块
                          unsigned short next_m,
                          unsigned short next_r, unsigned short next_c,
                          unsigned short next_basePixAddr, unsigned short next_baseChannel,
                          // store 的源块
                          unsigned short m, unsigned short r, unsigned short c,
                          unsigned short basePixAddr, unsigned short baseChannel,
                          // 通用参数
                          unsigned short fm_size, unsigned short o_fm_size,
                          unsigned short ch_in, unsigned short ch_out,
                          unsigned short stride, unsigned short act,
                          unsigned short mode)
{

    // if (mode == MODE_PW) {
    //     {
    //         #pragma HLS DATAFLOW
    //         compute_output(fm_out_buff_comp, bias_buff, in1, w1,
    //                    next_baseChannel,
    //                    fm_size, ch_in, ch_out,
    //                    0, 0, 1,
    //                    next_basePixAddr, mode);
    //         store_output_unified(fm_out_buff_store, out1,
    //                             0, 0, baseChannel,
    //                             act, fm_size, ch_out, basePixAddr, mode);
    //     }
        
    // } else {
    //     {
    //         #pragma HLS DATAFLOW
    //         compute_output(fm_out_buff_comp, bias_buff, in1, w1,
    //                     next_m,
    //                     fm_size, ch_in, ch_out,
    //                     next_r, next_c, stride,
    //                     0, mode);
    //         store_output_unified(fm_out_buff_store, out1,
    //                             r, c, m,
    //                             act, o_fm_size, ch_out, 0, mode);
    //     }
    // }
    bool is_pw = (mode == MODE_PW);

    unsigned short co_m       = is_pw ? next_baseChannel : next_m;
    unsigned short co_r       = is_pw ? 0 : next_r;
    unsigned short co_c       = is_pw ? 0 : next_c;
    unsigned short co_stride  = is_pw ? 1 : stride;
    unsigned short co_pixAddr = is_pw ? next_basePixAddr : 0;

    unsigned short so_r       = is_pw ? 0 : r;
    unsigned short so_c       = is_pw ? 0 : c;
    unsigned short so_m       = is_pw ? baseChannel : m;
    unsigned short so_fm_size = is_pw ? fm_size : o_fm_size;
    unsigned short so_pixAddr = is_pw ? basePixAddr : 0;

    #pragma HLS DATAFLOW
    
    compute_output(fm_out_buff_comp, bias_buff, in1, w1,
                co_m, fm_size, ch_in, ch_out,
                co_r, co_c, co_stride, co_pixAddr, mode);

    store_output_unified(fm_out_buff_store, out1, out2,
                        so_r, so_c, so_m,
                        act, so_fm_size, ch_out, so_pixAddr, mode);
    
}

// static int counter = 0;

// =========================================================
// 顶层 unified_conv
// =========================================================
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
                  bool mode)
{
#pragma HLS INLINE off
    data_t bias_buff[MAX_LEN/Tm][Tm];
    data_t fm_out1[Tm][Tr][Tc];
    data_t fm_out2[Tm][Tr][Tc];
    #pragma HLS ARRAY_PARTITION variable=fm_out1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=fm_out2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=fm_out1 cyclic factor=2 dim=3
    #pragma HLS ARRAY_PARTITION variable=fm_out2 cyclic factor=2 dim=3

    // 拷 bias
    // int start_16 = 0;
    // bool is_odd_start = 0 & 1;

    for (int k = 0; k < (ch_out - 1) / 2 + 1; ++k) {
        #pragma HLS PIPELINE II=1
        axi_t tmp = *(bias + k);
        data_t b_val0, b_val1;
        b_val0(15,0) = tmp(15,0);
        b_val1(15,0) = tmp(31,16);

        int out_idx_low = k * 2 - 0; // is_odd_start=0
        int out_idx_high = out_idx_low + 1;

        if (out_idx_low >= 0 && out_idx_low < ch_out) {
            bias_buff[out_idx_low / Tm][out_idx_low % Tm] = b_val0;
        }
        if (out_idx_high >= 0 && out_idx_high < ch_out) {
            bias_buff[out_idx_high / Tm][out_idx_high % Tm] = b_val1;
        }
    }

    // 输出特征图尺寸
    unsigned short o_fm_size = (mode == MODE_PW) ? fm_size :
                               ((stride == 1) ? fm_size : (fm_size / 2));

    // 块坐标
    unsigned short r = 0, c = 0, m = 0;
    unsigned short next_r = 0, next_c = 0, next_m = 0;
    unsigned short basePixAddr = 0, baseChannel = 0;
    unsigned short next_basePixAddr = 0, next_baseChannel = 0;

    bool pingpong = true;

    // ---- 首块：compute 到 fm_out1；让 store 端 m/baseChannel = ch_out 不真正写 ----
    compute_store(fm_out1, fm_out2, bias_buff, in1, w1, out1, out2,
                  /*next_m*/ m, /*next_r*/ r, /*next_c*/ c,
                  /*next_basePixAddr*/ basePixAddr, /*next_baseChannel*/ baseChannel,
                  /*m*/ ch_out, /*r*/ r, /*c*/ c,
                  /*basePixAddr*/ basePixAddr, /*baseChannel*/ ch_out,
                  fm_size, o_fm_size, ch_in, ch_out, stride, act, mode);

    while (true) {
        // counter++;
        next_block_unified(r, c, m, next_r, next_c, next_m,
                       basePixAddr, baseChannel,
                       next_basePixAddr, next_baseChannel,
                       o_fm_size, mode);
        if (pingpong) {
            // store fm_out1，compute next 到 fm_out2
            compute_store(fm_out2, fm_out1, bias_buff, in1, w1, out1, out2,
                          next_m, next_r, next_c,
                          next_basePixAddr, next_baseChannel,
                          m, r, c,
                          basePixAddr, baseChannel,
                          fm_size, o_fm_size, ch_in, ch_out, stride, act, mode);
            pingpong = false;
        } else {
            // store fm_out2，compute next 到 fm_out1
            compute_store(fm_out1, fm_out2, bias_buff, in1, w1, out1, out2,
                          next_m, next_r, next_c,
                          next_basePixAddr, next_baseChannel,
                          m, r, c,
                          basePixAddr, baseChannel,
                          fm_size, o_fm_size, ch_in, ch_out, stride, act, mode);
            pingpong = true;
        }

        // cout << "Block " << counter << ": m=" << m << ", r=" << r << ", c=" << c
        //      << ", basePixAddr=" << basePixAddr << ", baseChannel=" << baseChannel << endl;
        // 推进坐标
        m = next_m;   r = next_r;   c = next_c;
        basePixAddr = next_basePixAddr;
        baseChannel = next_baseChannel;

        // 结束判定
        bool done;
        if (mode == MODE_PW) {
            done = (basePixAddr + Tp >= (fm_size * fm_size)) &&
                   (baseChannel + Tm >= ch_out);
        } else {
            // 与原 std 一致：用 next_m 判定
            // 这里先算出 next，再用 next_m 比较
            next_block_unified(r, c, m, next_r, next_c, next_m,
                               basePixAddr, baseChannel,
                               next_basePixAddr, next_baseChannel,
                               o_fm_size, mode);
            done = (next_m >= ch_out);
        }

        if (done) break;
    }
    // cout << "unified_conv done! Total blocks: " << counter << endl;
    // counter = 0;


    // 末块 store
        if (pingpong) {
        store_output_unified(fm_out1, out1, out2,
                             r, c, (mode == MODE_PW) ? baseChannel : m,
                             act,
                             o_fm_size,
                             ch_out,
                             basePixAddr,
                             mode);
    } else {
        store_output_unified(fm_out2, out1, out2,
                             r, c, (mode == MODE_PW) ? baseChannel : m,
                             act,
                             o_fm_size,
                             ch_out,
                             basePixAddr,
                             mode);
    }
}