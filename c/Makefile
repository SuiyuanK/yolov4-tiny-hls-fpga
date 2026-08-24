TARGET = yolov4_tiny

CFLAGS += -O3 -g

#根据SIMU宏选择串口波特率,0：FPGA上板；1：仿真
CFLAGS += -DSIMU=0

C_SRCS := $(wildcard ./*.c )
# CPP_SRCS := $(wildcard ./*.cpp)

OBJDIR = obj
COMMON_DIR = /mnt/hgfs/Desktop/yolov4_tiny/vivado_prj/My_SoC_int16/sdk/software/SoC_bsp
GCC_DIR=/home/sui/tools/toolchains/loongson-gnu-toolchain-8.3-x86_64-loongarch32r-linux-gnusf-v2.0
PICOLIBC_DIR=/home/sui/tools/toolchains/picolibc
include /mnt/hgfs/Desktop/yolov4_tiny/vivado_prj/My_SoC_int16/sdk/software/SoC_bsp/common.mk
