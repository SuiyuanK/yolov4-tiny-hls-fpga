# AGENTS.md

YOLOv4-tiny inference on a custom LoongArch32 SoC + FPGA accelerator (Vivado HLS). The CPU (bare-metal C) reads weights/input from SD card and drives an HLS conv/maxpool/upsample accelerator over MMIO; the accelerator handles backbone + neck, CPU does nothing else.

## Layout
- `c/` — bare-metal SoC software (no OS/SDK). `basic_block.h`/`yolo4_tiny.h` = model graph, `basic_op.c` = accelerator driver, `sd_driver.c` = SD DMA, `main.c` = test entry.
- `hls/` — Vivado HLS sources (`src/`) and testbenches (`sim/`). `top.cpp` = unified conv, `maxpool.cpp`, `upsample.cpp`; `type.h` = data types.
- `For SD card/` — Python tools + weight/activation data that get placed on the SD card.

## Build — `c/` is NOT self-contained
- `c/Makefile:12-14` includes an external BSP (`.../SoC_bsp/common.mk`, provides `common_func.h` → `RegRead`/`RegWrite`) and expects a Loongson LoongArch32 toolchain + picolibc under `/home/sui/tools/...`. The BSP, and toolchains are not in this repo — they live on the original dev machine (VMware `hgfs` path). No build here without them.
- `-DSIMU=0` (on-board FPGA) vs `=1` (simulation), `c/Makefile:6`.
- All `c/*.c` get compiled (wildcard, `c/Makefile:8`), output firmware images in `c/obj/` (git-ignored).
- Comments and history are in Chinese; keep that style.

## Fixed-point contract (software ↔ HLS)
- Q3.13: HLS `data_t = ap_fixed<16,7,AP_RND,AP_SAT>` (`hls/src/type.h:16`) ↔ C `typedef int16_t data_t` (`c/basic_block.h:18`). Scale factor `2^9 = 512` appears as `SCALE` in `c/yolo4_tiny.h:53` and `For SD card/float_bin_to_fixed16.py`. These are linked — never change one side alone.
- Accelerator I/O is `axi_t = ap_uint<32>` with two int16 samples packed per word (`hls/src/type.h:17`); pass `(axi_t*)` casts as the sim TBs do.
- `hls/src/ddd.txt` = per-operator dimension table; `For SD card/统计.txt` = float activation min/max per layer (INT8 flow input).

## Weight/quantization flows (`For SD card/`)
- **Active flow:** `float_bin_to_fixed16.py` converts float32 `.bin` → int16 Q3.13 into `fixed16_output/` (run from `For SD card/`; defaults to `folded_weight` + `yolo_test`). This is what the FPGA software consumes.
- **Experimental INT8 flow:** `quantize_model.py` + `conv.py` → `quantized/` (per-channel INT8 weights, INT32 bias, requant mult/shift from `统计.txt`); `conv.py` still has hardcoded `D:/` Windows paths. The C/HLS code is INT16-only today — don't mix flows.
- `folded_weight/` = float32 bins with BN already folded; per-layer files `w.bin`/`b.bin` (or `w1..w4.bin`/`b1..b4.bin` in ResBlocks).

## SD card layout & memory map (load-bearing)
- Weight/image byte offsets are absolute SD offsets, converted via `YOLO4_TINY_BYTE_TO_SECTOR` (`byte/512`) in `c/yolo4_tiny_params.h:33-86`; the comment block is a human-readable copy. A card image must match exactly.
- At runtime the SoC `read_params()`s every weight/bias from SD into a fixed RAM area at `0x46000000` (`c/yolo4_tiny_params.h:117`); the layout is hand-chained `ALIGN_16` arithmetic that `#error`s past 32MB.
- Changing `OUT_CH` (currently 75 = VOC 25×3; COCO would be 255) or any layer shape requires regenerating BOTH the SD sector offsets and the `0x46000000` layout.
- `YOLO4_TINY_CPP_SIM_BK_*_OFF` (`yolo4_tiny_params.h:101-115`) is a separate SD region holding golden intermediate results for the resblock self-tests.

## Accelerator driver (`c/basic_op.c`)
- Conv MMIO at `CONV_CTRL_BASEADDR 0x1f300000`; register offsets there must match the HLS block AXI-lite interface of `conv()` (`hls/src/top.h`) — if the HLS port list changes, update basic_op.c too (stale old hardcoded maps are commented above the active ones, `c/basic_op.c:9-48`).
- `sampling()` multiplexes maxpool (k=2) and upsample (k=0) through the same conv kernel (`c/basic_op.c:136-141`); `upsample_c=128`, `upsample_size=13` are fixed for yolov4-tiny.
- BSP globals in `main.c:14-20` (UART `0x1f000000`, CONFREG `0x1f20e000`, counter freq) must match the SoC address map.

## HLS simulation (`hls/sim/`)
- `tb_conv2d.cpp`/`tb_maxpool.cpp` hardcode Windows `D:/...` text-file paths and per-test `#define`s (C/SIZE/k/s/p) — edit both before use; inputs are `.txt` (one value per line).
- `tb_upsample.cpp` is self-contained (synthetic input, compares against `baseline_upsample`).
- No TCL build script in the repo: the Vivado HLS project is created in the GUI, pointing at `hls/src/` and `hls/sim/tb_*.cpp`.

## On-board tests
- `main.c` runs `test_resblock{1,2,3}_with_ref_input()` then `yolo_test()`; per-block tests load golden inputs/results from the `CPP_SIM_BK` SD sectors (see `c/basic_op_test.c`), compare exact int16 and print diffs. LEDs mark progress/done (`c/main.c:80,90`). Nothing in `c/` can be run outside the SoC/board.
