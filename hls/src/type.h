#ifndef TYPE_H
#define TYPE_H

#include<ap_fixed.h>
#include<ap_int.h>
#include <hls_stream.h>


#include<iostream>// for debug
#include<stdio.h> //for debug
using namespace std; //for debug

// 1 位符号位
// 6 位整数位
// 9 位小数位
typedef ap_fixed<16,7,AP_RND,AP_SAT> data_t;
typedef ap_uint<32> axi_t;

// typedef float data_t; // for debug  float C语言原生数据类型，仿真比 ap_fixed 快很多

#endif