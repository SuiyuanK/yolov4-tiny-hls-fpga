#include <stdio.h> 
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include "debug.h"
#include "led.h"
#include "sd_driver.h"
#include "basic_op.h"
#include "basic_op_test.h"


//BSP板级支持包所需全局变量
unsigned long UART_BASE = 0x1f000000;					//UART16550的虚地址
unsigned long CONFREG_TIMER_BASE = 0x1f20e000;			//CONFREG计数器的虚地址
unsigned long CONFREG_CLOCKS_PER_SEC = 50000000L;		//CONFREG时钟频率
unsigned long CORE_CLOCKS_PER_SEC = 33000000L;			//处理器核时钟频率

unsigned long CONFREG_SWITCH_ADDR = 0x1f20f060;
unsigned long CONFREG_TIMER_ADDR = 0x1f20e000;

int main(int argc, char** argv)
{


	// int16_t data[2704];
	// int ret;
	// unsigned int i;
	// unsigned int data_len = (unsigned int)(sizeof(data) / sizeof(data[0]));

	// for (i = 0u; i < data_len; ++i) {
	// 	data[i] = 1u;
	// }

	// ret = read_params(0x8080u, data, data_len);
	// if (ret != 0) {
	// 	printf("read_params failed, ret=%d\n", ret);
	// 	return ret;
	// }

	// printf("data base addr: 0x%08X\n", (unsigned int)(uintptr_t)&data[0]);
	// printf("---- first 800 ----\n");
	// for (i = 0u; i < 800u; ++i) {
	// 	if (data[i] == 1u || (uint16_t)data[i] == 0xFFFFu) {
	// 		printf("data[%u] = 0x%04X", i, (unsigned int)data[i]);
	// 	} else {
	// 		printf("data[%u] = %.6f", i, (data[i]) / 512.0f);
	// 	}
	// 	if (((i + 1u) % 4u) == 0u) {
	// 		printf("\n");
	// 	} else {
	// 		printf("\t");
	// 	}
	// }
	// if ((800u % 4u) != 0u) {
	// 	printf("\n");
	// }

	// printf("---- last 800 ----\n");
	// for (i = data_len - 800u; i < data_len; ++i) {
	// 	if (data[i] == 1u || (uint16_t)data[i] == 0xFFFFu) {
	// 		printf("data[%u] = 0x%04X", i, (unsigned int)data[i]);
	// 	} else {
	// 		printf("data[%u] = %.6f", i, (data[i]) / 512.0f);
	// 	}
	// 	if ((((i - (data_len - 800u)) + 1u) % 4u) == 0u) {
	// 		printf("\n");
	// 	} else {
	// 		printf("\t");
	// 	}
	// }
	// if ((800u % 4u) != 0u) {
	// 	printf("\n");
	// }
	

	// 先使能 timer
	printf("\n\n\n");
	RegWrite(CONFREG_TIMER_ADDR + 0x8, 1);  // TIMER_EN_ADDR = TIMER_ADDR + 0x8
	setLedPin(0x5555u); // 再把LED全开了，看看是否成功点亮  
	test_resblock1_with_ref_input();
	test_resblock2_with_ref_input();
	test_resblock3_with_ref_input();
	printf("\n\n\n");
	yolo_test();
	printf("\n\n\n");
	// test_all_resblock_with_ref_input();
	printf("\n\n\n");
	
	setLedPin(0xFFFFu);
	return 0;
}
