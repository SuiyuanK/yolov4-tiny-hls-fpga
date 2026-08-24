# YOLOv4-tiny on LoongArch32 SoC + FPGA Accelerator (Vivado HLS)

基于自研 LoongArch32 SoC 与 FPGA 加速器的 YOLOv4-tiny 整数推理系统。CPU 运行裸机 C 程序,负责从 SD 卡读取权重/输入并通过 MMIO 驱动 FPGA 加速器;卷积、池化、上采样等所有密集计算均由 Vivado HLS 实现并综合的硬件加速器完成。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│  LoongArch32 SoC (裸机 C,无 OS/SDK)                          │
│  ├─ SD 卡驱动 (sd_driver.c)         ← 权重/图像/黄金结果      │
│  ├─ 加速器驱动 (basic_op.c)         → MMIO @ 0x1f300000      │
│  └─ 模型图 (basic_block.h / yolo4_tiny.h)                    │
│      只做 malloc/memcpy 拼接和后处理                         │
└──────────────────────┬──────────────────────────────────────┘
                       │ AXI-Lite 寄存器协议 + AXI MM 流
┌──────────────────────▼──────────────────────────────────────┐
│  FPGA 加速器 (Vivado HLS)                                    │
│  ├─ unified_conv: k3s1/k3s2 卷积 + 1x1 pointwise + maxpool   │
│  └─ upsample / maxpool (通过同一 conv 内核参数化复用)         │
└─────────────────────────────────────────────────────────────┘
```

- **硬件**:统一的 `conv()` 算子(见 `hls/src/top.h`),通过 `act` / `kernel` / `stride` / `upsample_c` / `upsample_size` 参数切换卷积(k3s2、k3s1、k1s1)、maxpool(k=2)与 upsample(k=0)三种模式。
- **软件**:`Yolo4_Tiny_forward()` 只负责拼接(concat)、内存分配以及 yolo head 的 1x1 卷积后处理;backbone(3 个 ResBlock 核心)与 neck 的卷积全部在加速器上执行。

## 目录结构

| 目录 | 说明 |
| --- | --- |
| `c/` | SoC 裸机 C 程序:模型图、加速器 MMIO 驱动、SD 卡 DMA、构建固件 |
| `hls/` | Vivado HLS 源码(`src/`)与测试台(`sim/`) |
| `For SD card/` | Python 权重转换工具、量化流程与放置到 SD 卡的数据 |

## 数据格式(Q3.13 定点)

软件与硬件之间的定点契约:

| 位置 | 定义 |
| --- | --- |
| HLS `hls/src/type.h` | `data_t = ap_fixed<16,7,AP_RND,AP_SAT>`(1 符号位 + 6 整数位 + 9 小数位) |
| C `c/basic_block.h` | `typedef int16_t data_t` |
| 缩放因子 | `SCALE = 2^9 = 512`(`c/yolo4_tiny.h`、`For SD card/float_bin_to_fixed16.py`) |

加速器内存接口为 `axi_t = ap_uint<32>`,每个 32 位字打包 2 个 int16 样本。

## 量化流程

`For SD card/` 提供两条流程:

1. **Q3.13 定点(已验证 / 当前使用)**:`float_bin_to_fixed16.py` 将 float32 `.bin`(BN 已折叠)转换为 int16 Q3.13,输出到 `fixed16_output/`,即 FPGA 板端程序实际消费的数据。
2. **INT8 量化(⚠️ 未完成,实验性)**:`quantize_model.py` 实现 per-channel INT8 权重 + INT32 偏置 + 逐层 requant(multiplier/shift),依赖 `统计.txt` 中的逐层激活 min/max;`conv.py` 为 Python 参考实现(仍含 Windows 硬编码路径)。当前 C/HLS 代码仍为 INT16 单流,**尚未接入 INT8 流程**。

## 构建

### FPGA 软件(`c/`)

> ⚠️ 无法在当前仓库内独立构建:`c/Makefile` 依赖外部 BSP(`SoC_bsp/common.mk`,提供 `RegRead`/`RegWrite` 等)以及 Loongson LoongArch32 工具链 + picolibc,均位于原始开发机(`/mnt/hgfs/...`、`/home/sui/tools/...`)。

```bash
make             # 产物在 c/obj/ 下: .elf / .bin / axi_ram.mif / .coe / rom.vlog
```

- `-DSIMU=0`:FPGA 上板运行;`-DSIMU=1`:仿真(两者 UART 波特率不同)。

### 硬件(HLS)

- 无 TCL 构建脚本:在 Vivado HLS GUI 中创建工程,启用 `hls/src/` 为源码目录、`hls/sim/tb_*.cpp` 为测试台。
- `tb_upsample.cpp` 自包含(随机输入 + 参考实现);`tb_conv2d.cpp` / `tb_maxpool.cpp` 需要编辑文件路径与 `C/SIZE/k/s/p` 宏(仍为 Windows 硬编码路径),输入为一个值一行的 `.txt`。

## SD 卡布局

- 权重/图像的字节偏移通过 `YOLO4_TINY_BYTE_TO_SECTOR` 换算为扇区号(见 `c/yolo4_tiny_params.h` 顶部注释与定义),SD 卡镜像必须严格匹配。
- `OUT_CH = 75`(VOC,25 类 × 3);改为 COCO(255)或调整网络结构时,需同时重新生成 SD 卡偏移与 RAM 固定内存区(`0x46000000`,ALIGN_16 手写链)布局,超出 32MB 会触发 `#error`。

## 板端测试

`main.c` 依次运行:

- `test_resblock{1,2,3}_with_ref_input()`:以 SD 卡 `CPP_SIM_BK` 区域的黄金输入/输出做逐 int16 比对,打印差异。
- `yolo_test()`:读入 SD 卡上的测试图像(`img.bin`)做完整推理,输出两个 head 的预测结果。

LED 指示进度/完成(`main.c`)。所有测试只能在 SoC 板端运行。

## 注释语言

C/HLS 源码注释与提交历史均为中文,新代码请保持该风格。
