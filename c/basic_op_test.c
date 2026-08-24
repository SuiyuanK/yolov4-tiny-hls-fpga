#include <stdio.h>
#include "basic_block.h"
#include "basic_op_test.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

// 打印数组前50和后50个元素
void print_debug_info(const char* name, data_t* data, size_t total_size) {
    printf("--- [Debug] %s (Size: %zu) ---\n", name, total_size);
    printf("Front 50: ");
    for (size_t i = 0; i < 50 && i < total_size; i++) {
        printf("%d ", data[i]);
    }
    printf("\nBack 50:  ");
    size_t start = (total_size > 50) ? (total_size - 50) : 0;
    for (size_t i = start; i < total_size; i++) {
        printf("%d ", data[i]);
    }
    printf("\n-----------------------------------\n\n");
}
// 打印数组前256和后256个元素，按128个元素一行排列
// void print_debug_info(const char* name, data_t* data, size_t total_size) {
//     printf("--- [Debug] %s (Total Size: %zu) ---\n", name, total_size);

//     // 打印前 256 个元素
//     printf("Front 256 (128 per line):\n");
//     for (size_t i = 0; i < 256 && i < total_size; i++) {
//         printf("%d ", (int)data[i]);
//         if ((i + 1) % 128 == 0) {
//             printf("\n"); // 每 128 个元素换行
//         }
//     }
//     // 如果最后一行没满 128 导致没换行，手动补一个
//     if (total_size < 256 && total_size % 128 != 0) printf("\n");

//     printf("\n");

//     // 打印后 256 个元素
//     printf("Back 256 (128 per line):\n");
//     size_t start = (total_size > 256) ? (total_size - 256) : 0;
    
//     for (size_t i = start; i < total_size; i++) {
//         printf("%d ", (int)data[i]);
//         // 这里的换行逻辑：当 i 是相对于 start 的第 128 的倍数时换行
//         // 或者计算 (i - start + 1) % 128 == 0
//         if ((i - start + 1) % 128 == 0) {
//             printf("\n");
//         }
//     }
//     // 处理末尾不足 128 的换行
//     if ((total_size - start) % 128 != 0) printf("\n");

//     printf("-----------------------------------\n\n");
// }

static unsigned int compare_and_print_diff(const char* tag,
                                           const int16_t* got,
                                           const int16_t* ref,
                                           unsigned int ch,
                                           unsigned int h,
                                           unsigned int w,
                                           unsigned int max_print)
{
    unsigned int i;
    unsigned int len;
    unsigned int mismatch_cnt;
    unsigned int printed;
    unsigned int hw;

    len = ch * h * w;
    hw = h * w;
    mismatch_cnt = 0u;
    printed = 0u;

    for (i = 0u; i < len; i++) {
        if (got[i] != ref[i]) {
            if (printed < max_print) {
                unsigned int ch_idx;
                unsigned int rem;
                unsigned int row;
                unsigned int col;
                int got_v;
                int ref_v;
                int diff;

                ch_idx = i / hw;
                rem = i % hw;
                row = rem / w;
                col = rem % w;
                got_v = (int)got[i];
                ref_v = (int)ref[i];
                diff = got_v - ref_v;

                printf("%s diff[%u]: idx=%u ch=%u row=%u col=%u got=%d ref=%d diff=%d\n",
                       tag, printed, i, ch_idx, row, col, got_v, ref_v, diff);
                printed++;
            }
            mismatch_cnt++;
        }
    }

    if (mismatch_cnt > max_print) {
        printf("%s diff output truncated: printed=%u, total_mismatch=%u\n",
               tag, max_print, mismatch_cnt);
    } else {
        printf("%s total_mismatch=%u\n", tag, mismatch_cnt);
    }

    return mismatch_cnt;
}

#define FRAC_BITS 9
#define FIXED_ONE (1 << FRAC_BITS)
#define LEAKY_CONSTANT 51 // float_to_fixed(0.1, 9)

// 定点乘法：Q9 * Q9 -> Q9
static inline int16_t fixed_mul(int16_t a, int16_t b) {
    int32_t tmp = (int32_t)a * (int32_t)b;
    tmp = (tmp + (1 << (FRAC_BITS - 1))) >> FRAC_BITS;
    if (tmp > 32767)  tmp = 32767;
    if (tmp < -32768) tmp = -32768;
    return (int16_t)tmp;
}

// 核心卷积逻辑
void conv_core(int ch_in, int ch_out, int pad, int stride, int k, int h, int w,
               int16_t* in, int16_t* weight, int16_t* bias, int16_t* out) {
    int h_o = (stride == 0) ? h : (h - k + 2 * pad) / stride + 1;
    int w_o = (stride == 0) ? w : (w - k + 2 * pad) / stride + 1;

    for (int m = 0; m < ch_out; m++) {
        for (int i = 0; i < h_o; i++) {
            for (int j = 0; j < w_o; j++) {
                // 初始化累加器为 bias
                int32_t acc = (int32_t)bias[m];
                
                for (int n = 0; n < ch_in; n++) {
                    for (int kx = 0; kx < k; kx++) {
                        for (int ky = 0; ky < k; ky++) {
                            int row = i * stride + kx - pad;
                            int col = j * stride + ky - pad;
                            
                            if (row >= 0 && row < h && col >= 0 && col < w) {
                                int addr_i = n * h * w + row * w + col;
                                int addr_w = m * ch_in * k * k + n * k * k + kx * k + ky;
                                
                                int32_t mul = (int32_t)in[addr_i] * (int32_t)weight[addr_w];
                                // 按照你验证过的逻辑：每步乘加后立即缩放
                                acc += (mul + (1 << (FRAC_BITS - 1))) >> FRAC_BITS;
                            }
                        }
                    }
                }
                // 饱和截断
                if (acc > 32767)  acc = 32767;
                else if (acc < -32768) acc = -32768;
                out[m * h_o * w_o + i * w_o + j] = (int16_t)acc;
            }
        }
    }
}

// MaxPool 实现
void maxpool_core(int16_t* in, int16_t* out, int ch, int h, int w) {
    int h_o = h / 2;
    int w_o = w / 2;
    for (int n = 0; n < ch; n++) {
        for (int i = 0; i < h_o; i++) {
            for (int j = 0; j < w_o; j++) {
                int16_t max_val = -32768;
                for (int kx = 0; kx < 2; kx++) {
                    for (int ky = 0; ky < 2; ky++) {
                        int r = i * 2 + kx;
                        int c = j * 2 + ky;
                        int16_t val = in[n * h * w + r * w + c];
                        if (val > max_val) max_val = val;
                    }
                }
                out[n * h_o * w_o + i * w_o + j] = max_val;
            }
        }
    }
}

// --- 以下是你要求的封装入口 ---

// 统一的卷积+激活入口
void conv_leakyrelu_c(uint32_t ch_in, uint32_t ch_out, uint32_t pad, uint32_t stride,
                    uint32_t kernel, uint32_t fsize, uint32_t fsize_w,
                    int16_t* in, int16_t* w, int16_t* b, int16_t* out,
                    uint32_t act) {
    
    // 1. 执行卷积
    conv_core((int)ch_in, (int)ch_out, (int)pad, (int)stride, (int)kernel, 
              (int)fsize, (int)fsize_w, in, w, b, out);

    // 2. 如果开启激活 (act=1)
    if (act == 1) {
        int h_o = (fsize - kernel + 2 * pad) / stride + 1;
        int w_o = (fsize_w - kernel + 2 * pad) / stride + 1;
        int total_elements = (int)ch_out * h_o * w_o;
        
        for (int i = 0; i < total_elements; i++) {
            if (out[i] < 0) {
                out[i] = fixed_mul(out[i], (int16_t)LEAKY_CONSTANT);
            }
        }
    }
}

// 抽样/上采样封装
void sampling_c(int16_t* in, int16_t* out, uint32_t ch, uint32_t fsize, uint32_t mode) {
    if (mode == 1) { // MaxPool
        maxpool_core(in, out, (int)ch, (int)fsize, (int)fsize);
    } else { // Upsample (最近邻插值)
        int h_o = fsize * 2;
        int w_o = fsize * 2;
        for (int c = 0; c < (int)ch; c++) {
            for (int i = 0; i < h_o; i++) {
                for (int j = 0; j < w_o; j++) {
                    out[c * h_o * w_o + i * w_o + j] = in[c * fsize * fsize + (i / 2) * fsize + (j / 2)];
                }
            }
        }
    }
}




void test_resblock1_with_ref_input(void) {
    // 1. 初始化 Resblock1 (104x104, 64ch_in, 64ch_out)
    Resblock_body* rb1 = Resblock_body_create(104, 104, 64, 64,
        (data_t*)RB1_C1_W_ADDR, (data_t*)RB1_C1_B_ADDR,
        (data_t*)RB1_C2_W_ADDR, (data_t*)RB1_C2_B_ADDR,
        (data_t*)RB1_C3_W_ADDR, (data_t*)RB1_C3_B_ADDR,
        (data_t*)RB1_C4_W_ADDR, (data_t*)RB1_C4_B_ADDR);
    
    Resblock_body_load_weight(rb1,
        YOLO4_TINY_RB1_W1_OFF, YOLO4_TINY_RB1_B1_OFF,
        YOLO4_TINY_RB1_W2_OFF, YOLO4_TINY_RB1_B2_OFF,
        YOLO4_TINY_RB1_W3_OFF, YOLO4_TINY_RB1_B3_OFF,
        YOLO4_TINY_RB1_W4_OFF, YOLO4_TINY_RB1_B4_OFF);


    // 2. 准备缓冲区
    size_t in_size = 64 * 104 * 104;
    size_t out_size = 128 * 52 * 52; // MaxPool下采样后
    
    data_t* input_buf = (data_t*)malloc(in_size * sizeof(data_t));
    data_t* feat_buf  = (data_t*)malloc(in_size * sizeof(data_t));
    data_t* output_buf = (data_t*)malloc(out_size * sizeof(data_t));

    // 3. 将 BasicConv2 的参考结果作为输入加载进来
    printf("Loading reference input from BC2_OFF (Sector: %u)...\n", 
           (unsigned int)YOLO4_TINY_CPP_SIM_BK_BC2_OFF);
    read_params(YOLO4_TINY_CPP_SIM_BK_BC2_OFF, input_buf, (unsigned int)in_size);
    print_debug_info("Input (Ref from BC2)", input_buf, in_size);

    // 4. 执行内部逐层前向传播并打印
    data_t* route = (data_t*)ROUTE_BUF_ADDR;
    data_t* conv3_out = (data_t*)CONV3_OUT_BUF_ADDR;
    // static data_t route[1500000];
    // static data_t conv3_out[1500000];

    // memset(route, 0, WORKING_BUF_SIZE * sizeof(data_t));
    // memset(conv3_out, 0, WORKING_BUF_SIZE * sizeof(data_t));
    // print_debug_info("(route)", route, WORKING_BUF_SIZE);
    // print_debug_info("(conv3_out)", conv3_out, WORKING_BUF_SIZE);
    

    // --- Step 1: conv1 ---
    BasicConv_forward(rb1->conv1, input_buf, route);
    print_debug_info("RB1_Conv1_Output (route)", route, (size_t)rb1->ch_out * rb1->h * rb1->w);

    // --- Step 2: conv2 (使用route后半部分) ---
    int half_ch = rb1->ch_out / 2;
    size_t offset = (size_t)half_ch * rb1->h * rb1->w;
    BasicConv_forward(rb1->conv2, route + offset, conv3_out + offset);
    print_debug_info("RB1_Conv2_Output (conv3_out_half)", conv3_out + offset, offset);

    // --- Step 3: conv3 ---
    BasicConv_forward(rb1->conv3, conv3_out + offset, conv3_out);
    print_debug_info("RB1_Conv3_Output (conv3_out_front)", conv3_out, offset);
    
    print_debug_info("RB1_Conv2_Output (conv3_out_half)", conv3_out + offset, offset);
    // --- Step 4: conv4 ---
    BasicConv_forward(rb1->conv4, conv3_out, feat_buf);
    
    print_debug_info("RB1_Conv4_Output (feat)", feat_buf, (size_t)rb1->ch_out * rb1->h * rb1->w);

    // --- Step 5: Final Sampling (Concat + MaxPool) ---
    sampling(route, output_buf, (uint32_t)(rb1->ch_out), (uint32_t)(rb1->h), 1u);
    sampling(feat_buf, output_buf + (rb1->ch_out * rb1->h * rb1->w / 4), (uint32_t)(rb1->ch_out), (uint32_t)(rb1->h), 1u);
    print_debug_info("RB1_Final_Output (out)", output_buf, out_size);

    // 释放资源
    free(input_buf); free(feat_buf); free(output_buf);
    Resblock_body_destroy(rb1);
}

void test_resblock2_with_ref_input(void) {
    // 初始化 Resblock2 (52x52, 128ch_in, 128ch_out)
    Resblock_body* rb2 = Resblock_body_create(52, 52, 128, 128,
        (data_t*)RB2_C1_W_ADDR, (data_t*)RB2_C1_B_ADDR,
        (data_t*)RB2_C2_W_ADDR, (data_t*)RB2_C2_B_ADDR,
        (data_t*)RB2_C3_W_ADDR, (data_t*)RB2_C3_B_ADDR,
        (data_t*)RB2_C4_W_ADDR, (data_t*)RB2_C4_B_ADDR);
    
    Resblock_body_load_weight(rb2,
        YOLO4_TINY_RB2_W1_OFF, YOLO4_TINY_RB2_B1_OFF,
        YOLO4_TINY_RB2_W2_OFF, YOLO4_TINY_RB2_B2_OFF,
        YOLO4_TINY_RB2_W3_OFF, YOLO4_TINY_RB2_B3_OFF,
        YOLO4_TINY_RB2_W4_OFF, YOLO4_TINY_RB2_B4_OFF);

    // 准备缓冲区
    size_t in_size = 128 * 52 * 52;
    size_t out_size = 256 * 26 * 26; // MaxPool下采样后
    data_t* input_buf = (data_t*)malloc(in_size * sizeof(data_t));
    data_t* feat_buf  = (data_t*)malloc(in_size * sizeof(data_t));
    data_t* output_buf = (data_t*)malloc(out_size * sizeof(data_t));

    // 从前一层 Resblock1 的参考结果加载输入
    printf("Loading reference input from RB1_OFF (Sector: %u)...\n",
           (unsigned int)YOLO4_TINY_CPP_SIM_BK_RB1_OFF);
    read_params(YOLO4_TINY_CPP_SIM_BK_RB1_OFF, input_buf, (unsigned int)in_size);
    print_debug_info("Input (Ref from RB1)", input_buf, in_size);

    data_t* route = (data_t*)ROUTE_BUF_ADDR;
    data_t* conv3_out = (data_t*)CONV3_OUT_BUF_ADDR;

    // conv1
    BasicConv_forward(rb2->conv1, input_buf, route);
    print_debug_info("RB2_Conv1_Output (route)", route, (size_t)rb2->ch_out * rb2->h * rb2->w);

    // conv2 (route 后半)
    int half_ch = rb2->ch_out / 2;
    size_t offset = (size_t)half_ch * rb2->h * rb2->w;
    BasicConv_forward(rb2->conv2, route + offset, conv3_out + offset);
    print_debug_info("RB2_Conv2_Output (conv3_out_half)", conv3_out + offset, offset);

    // conv3
    BasicConv_forward(rb2->conv3, conv3_out + offset, conv3_out);
    print_debug_info("RB2_Conv3_Output (conv3_out_front)", conv3_out, offset);
    print_debug_info("RB2_Conv2_Output (conv3_out_half)", conv3_out + offset, offset);

    // conv4
    BasicConv_forward(rb2->conv4, conv3_out, feat_buf);
    print_debug_info("RB2_Conv4_Output (feat)", feat_buf, (size_t)rb2->ch_out * rb2->h * rb2->w);

    // concat + maxpool
    sampling(route, output_buf, (uint32_t)(rb2->ch_out), (uint32_t)(rb2->h), 1u);
    sampling(feat_buf, output_buf + (rb2->ch_out * rb2->h * rb2->w / 4), (uint32_t)(rb2->ch_out), (uint32_t)(rb2->h), 1u);
    print_debug_info("RB2_Final_Output (out)", output_buf, out_size);

    free(input_buf); free(feat_buf); free(output_buf);
    Resblock_body_destroy(rb2);
}

void test_resblock3_with_ref_input(void) {
    // 初始化 Resblock3 (26x26, 256ch_in, 256ch_out)
    Resblock_body* rb3 = Resblock_body_create(26, 26, 256, 256,
        (data_t*)RB3_C1_W_ADDR, (data_t*)RB3_C1_B_ADDR,
        (data_t*)RB3_C2_W_ADDR, (data_t*)RB3_C2_B_ADDR,
        (data_t*)RB3_C3_W_ADDR, (data_t*)RB3_C3_B_ADDR,
        (data_t*)RB3_C4_W_ADDR, (data_t*)RB3_C4_B_ADDR);
    
    Resblock_body_load_weight(rb3,
        YOLO4_TINY_RB3_W1_OFF, YOLO4_TINY_RB3_B1_OFF,
        YOLO4_TINY_RB3_W2_OFF, YOLO4_TINY_RB3_B2_OFF,
        YOLO4_TINY_RB3_W3_OFF, YOLO4_TINY_RB3_B3_OFF,
        YOLO4_TINY_RB3_W4_OFF, YOLO4_TINY_RB3_B4_OFF);

    // 准备缓冲区
    size_t in_size = 256 * 26 * 26;
    size_t out_size = 512 * 13 * 13; // MaxPool下采样后
    data_t* input_buf = (data_t*)malloc(in_size * sizeof(data_t));
    data_t* feat_buf  = (data_t*)malloc(in_size * sizeof(data_t));
    data_t* output_buf = (data_t*)malloc(out_size * sizeof(data_t));

    // 从前一层 Resblock2 的参考结果加载输入
    printf("Loading reference input from RB2_OFF (Sector: %u)...\n",
           (unsigned int)YOLO4_TINY_CPP_SIM_BK_RB2_OFF);
    read_params(YOLO4_TINY_CPP_SIM_BK_RB2_OFF, input_buf, (unsigned int)in_size);
    print_debug_info("Input (Ref from RB2)", input_buf, in_size);

    data_t* route = (data_t*)ROUTE_BUF_ADDR;
    data_t* conv3_out = (data_t*)CONV3_OUT_BUF_ADDR;

    // conv1
    BasicConv_forward(rb3->conv1, input_buf, route);
    print_debug_info("RB3_Conv1_Output (route)", route, (size_t)rb3->ch_out * rb3->h * rb3->w);

    // conv2 (route 后半)
    int half_ch = rb3->ch_out / 2;
    size_t offset = (size_t)half_ch * rb3->h * rb3->w;
    BasicConv_forward(rb3->conv2, route + offset, conv3_out + offset);
    print_debug_info("RB3_Conv2_Output (conv3_out_half)", conv3_out + offset, offset);

    // conv3
    BasicConv_forward(rb3->conv3, conv3_out + offset, conv3_out);
    print_debug_info("RB3_Conv3_Output (conv3_out_front)", conv3_out, offset);
    print_debug_info("RB3_Conv2_Output (conv3_out_half)", conv3_out + offset, offset);

    // conv4
    BasicConv_forward(rb3->conv4, conv3_out, feat_buf);
    print_debug_info("RB3_Conv4_Output (feat)", feat_buf, (size_t)rb3->ch_out * rb3->h * rb3->w);

    // concat + maxpool
    sampling(route, output_buf, (uint32_t)(rb3->ch_out), (uint32_t)(rb3->h), 1u);
    sampling(feat_buf, output_buf + (rb3->ch_out * rb3->h * rb3->w / 4), (uint32_t)(rb3->ch_out), (uint32_t)(rb3->h), 1u);
    print_debug_info("RB3_Final_Output (out)", output_buf, out_size);

    free(input_buf); free(feat_buf); free(output_buf);
    Resblock_body_destroy(rb3);
}



void test_all_resblock_with_ref_input(void) {
    // 按顺序执行 Resblock1 -> Resblock2 -> Resblock3
    // 2 使用 1 的 output 作为输入，3 使用 2 的 output 作为输入
    Resblock_body* rb1 = Resblock_body_create(104, 104, 64, 64,
        (data_t*)RB1_C1_W_ADDR, (data_t*)RB1_C1_B_ADDR,
        (data_t*)RB1_C2_W_ADDR, (data_t*)RB1_C2_B_ADDR,
        (data_t*)RB1_C3_W_ADDR, (data_t*)RB1_C3_B_ADDR,
        (data_t*)RB1_C4_W_ADDR, (data_t*)RB1_C4_B_ADDR);
    Resblock_body* rb2 = Resblock_body_create(52, 52, 128, 128,
        (data_t*)RB2_C1_W_ADDR, (data_t*)RB2_C1_B_ADDR,
        (data_t*)RB2_C2_W_ADDR, (data_t*)RB2_C2_B_ADDR,
        (data_t*)RB2_C3_W_ADDR, (data_t*)RB2_C3_B_ADDR,
        (data_t*)RB2_C4_W_ADDR, (data_t*)RB2_C4_B_ADDR);
    Resblock_body* rb3 = Resblock_body_create(26, 26, 256, 256,
        (data_t*)RB3_C1_W_ADDR, (data_t*)RB3_C1_B_ADDR,
        (data_t*)RB3_C2_W_ADDR, (data_t*)RB3_C2_B_ADDR,
        (data_t*)RB3_C3_W_ADDR, (data_t*)RB3_C3_B_ADDR,
        (data_t*)RB3_C4_W_ADDR, (data_t*)RB3_C4_B_ADDR);

    if (rb1 == NULL || rb2 == NULL || rb3 == NULL) {
        Resblock_body_destroy(rb1);
        Resblock_body_destroy(rb2);
        Resblock_body_destroy(rb3);
        return;
    }

    Resblock_body_load_weight(rb1,
        YOLO4_TINY_RB1_W1_OFF, YOLO4_TINY_RB1_B1_OFF,
        YOLO4_TINY_RB1_W2_OFF, YOLO4_TINY_RB1_B2_OFF,
        YOLO4_TINY_RB1_W3_OFF, YOLO4_TINY_RB1_B3_OFF,
        YOLO4_TINY_RB1_W4_OFF, YOLO4_TINY_RB1_B4_OFF);
    Resblock_body_load_weight(rb2,
        YOLO4_TINY_RB2_W1_OFF, YOLO4_TINY_RB2_B1_OFF,
        YOLO4_TINY_RB2_W2_OFF, YOLO4_TINY_RB2_B2_OFF,
        YOLO4_TINY_RB2_W3_OFF, YOLO4_TINY_RB2_B3_OFF,
        YOLO4_TINY_RB2_W4_OFF, YOLO4_TINY_RB2_B4_OFF);
    Resblock_body_load_weight(rb3,
        YOLO4_TINY_RB3_W1_OFF, YOLO4_TINY_RB3_B1_OFF,
        YOLO4_TINY_RB3_W2_OFF, YOLO4_TINY_RB3_B2_OFF,
        YOLO4_TINY_RB3_W3_OFF, YOLO4_TINY_RB3_B3_OFF,
        YOLO4_TINY_RB3_W4_OFF, YOLO4_TINY_RB3_B4_OFF);

    size_t rb1_in_size = 64 * 104 * 104;
    size_t rb1_feat_size = 64 * 104 * 104;
    size_t rb1_out_size = 128 * 52 * 52;
    size_t rb2_feat_size = 128 * 52 * 52;
    size_t rb2_out_size = 256 * 26 * 26;
    size_t rb3_feat_size = 256 * 26 * 26;
    size_t rb3_out_size = 512 * 13 * 13;
    const unsigned int max_print = 4u;

    data_t* rb1_input = (data_t*)malloc(rb1_in_size * sizeof(data_t));
    data_t* rb1_feat = (data_t*)malloc(rb1_feat_size * sizeof(data_t));
    data_t* rb1_out = (data_t*)malloc(rb1_out_size * sizeof(data_t));
    data_t* rb2_feat = (data_t*)malloc(rb2_feat_size * sizeof(data_t));
    data_t* rb2_out = (data_t*)malloc(rb2_out_size * sizeof(data_t));
    data_t* rb3_feat = (data_t*)malloc(rb3_feat_size * sizeof(data_t));
    data_t* rb3_out = (data_t*)malloc(rb3_out_size * sizeof(data_t));
    data_t* rb1_ref = (data_t*)malloc(rb1_out_size * sizeof(data_t));
    data_t* rb2_ref = (data_t*)malloc(rb2_out_size * sizeof(data_t));
    data_t* rb3_ref = (data_t*)malloc(rb3_out_size * sizeof(data_t));

    if (rb1_input == NULL || rb1_feat == NULL || rb1_out == NULL ||
        rb2_feat == NULL || rb2_out == NULL || rb3_feat == NULL || rb3_out == NULL ||
        rb1_ref == NULL || rb2_ref == NULL || rb3_ref == NULL) {
        free(rb1_input); free(rb1_feat); free(rb1_out);
        free(rb2_feat); free(rb2_out); free(rb3_feat); free(rb3_out);
        free(rb1_ref); free(rb2_ref); free(rb3_ref);
        Resblock_body_destroy(rb1);
        Resblock_body_destroy(rb2);
        Resblock_body_destroy(rb3);
        return;
    }

    read_params(YOLO4_TINY_CPP_SIM_BK_BC2_OFF, rb1_input, (unsigned int)rb1_in_size);
    read_params(YOLO4_TINY_CPP_SIM_BK_RB1_OFF, rb1_ref, (unsigned int)rb1_out_size);
    read_params(YOLO4_TINY_CPP_SIM_BK_RB2_OFF, rb2_ref, (unsigned int)rb2_out_size);
    read_params(YOLO4_TINY_CPP_SIM_BK_RB3_OFF, rb3_ref, (unsigned int)rb3_out_size);

    Resblock_body_forward(rb1, rb1_input, rb1_feat, rb1_out);
    print_debug_info("RB1_feat", rb1_feat, rb1_feat_size);
    print_debug_info("RB1_output", rb1_out, rb1_out_size);
    compare_and_print_diff("RB1_output", rb1_out, rb1_ref, 128u, 52u, 52u, max_print);

    Resblock_body_forward(rb2, rb1_out, rb2_feat, rb2_out);
    print_debug_info("RB2_feat", rb2_feat, rb2_feat_size);
    print_debug_info("RB2_output", rb2_out, rb2_out_size);
    compare_and_print_diff("RB2_output", rb2_out, rb2_ref, 256u, 26u, 26u, max_print);

    Resblock_body_forward(rb3, rb2_out, rb3_feat, rb3_out);
    print_debug_info("RB3_feat", rb3_feat, rb3_feat_size);
    print_debug_info("RB3_output", rb3_out, rb3_out_size);
    compare_and_print_diff("RB3_output", rb3_out, rb3_ref, 512u, 13u, 13u, max_print);

    free(rb1_input); free(rb1_feat); free(rb1_out);
    free(rb2_feat); free(rb2_out); free(rb3_feat); free(rb3_out);
    free(rb1_ref); free(rb2_ref); free(rb3_ref);
    Resblock_body_destroy(rb1);
    Resblock_body_destroy(rb2);
    Resblock_body_destroy(rb3);
    
}











