import numpy as np
import os

# ================================================================
# 可配置参数 (修改这里即可)
# ================================================================

# 层参数
C_IN    = 3
SIZE_IN = 416
C_OUT   = 32
K       = 3
P       = 1
S       = 2

# 文件路径
WEIGHTS_PATH    = "D:/SuiyuanBa/Desktop/yolov4_tiny/HLS/For SD card/quantized/folded_weight/BasicConv1/w_int8.bin"
BIAS_PATH       = "D:/SuiyuanBa/Desktop/yolov4_tiny/HLS/For SD card/quantized/folded_weight/BasicConv1/b_int32.bin"
INPUTS_PATH     = "D:/SuiyuanBa/Desktop/yolov4_tiny/HLS/For SD card/quantized/yolo_test/img_int8.bin"
MULT_PATH       = "D:/SuiyuanBa/Desktop/yolov4_tiny/HLS/For SD card/quantized/folded_weight/BasicConv1/w_requant_mult.bin"
SHIFT_PATH      = "D:/SuiyuanBa/Desktop/yolov4_tiny/HLS/For SD card/quantized/folded_weight/BasicConv1/w_requant_shift.bin"
OUTPUT_PATH     = "D:/SuiyuanBa/Desktop/yolov4_tiny/HLS/For SD card/quantized/folded_weight/BasicConv1/py_output_int8.bin"
OUTPUT_TXT_PATH = "D:/SuiyuanBa/Desktop/yolov4_tiny/HLS/For SD card/quantized/folded_weight/BasicConv1/py_output.txt"

# HLS 对比
COMPARE_HLS     = True   # True: 开启对比, False: 跳过
HLS_OUTPUT_PATH = "D:/SuiyuanBa/Desktop/yolov4_tiny/HLS/For SD card/quantized/folded_weight/BasicConv1/hls_output_int8.bin"
HLS_OUTPUT_TXT  = "D:/SuiyuanBa/Desktop/yolov4_tiny/HLS/For SD card/quantized/folded_weight/BasicConv1/hls_output.txt"

# ================================================================
# 计算推导参数
# ================================================================
SIZE_OUT     = (SIZE_IN + 2 * P - K) // S + 1
WEIGHT_COUNT = C_OUT * C_IN * K * K
INPUT_COUNT  = C_IN * SIZE_IN * SIZE_IN
OUT_SIZE     = C_OUT * SIZE_OUT * SIZE_OUT

# ================================================================
# 读 bin 文件
# ================================================================
def load_bin(path, dtype, count):
    data = np.fromfile(path, dtype=dtype)
    assert data.size == count, \
        f"(x_x) {path}: expected {count} elements, got {data.size}"
    print(f"  (^_^) Loaded: {path}  ({data.size} elems, {data.nbytes} bytes)")
    return data

# ================================================================
# INT8 卷积 + 再量化
# ================================================================
def conv2d_int8(input_int8, weight_int8, bias_int32, scale_m0, scale_shift):
    """
    input_int8 : [C_IN, SIZE_IN, SIZE_IN]   int8
    weight_int8: [C_OUT, C_IN, K, K]        int8
    bias_int32 : [C_OUT]                    int32
    scale_m0   : [C_OUT]                    int32
    scale_shift: [C_OUT]                    uint8
    return     : [C_OUT, SIZE_OUT, SIZE_OUT] int8
    """

    # 1. padding
    if P > 0:
        inp = np.pad(input_int8,
                     ((0, 0), (P, P), (P, P)),
                     mode='constant', constant_values=0)
    else:
        inp = input_int8

    # 2. INT32 卷积累加
    psum = np.zeros((C_OUT, SIZE_OUT, SIZE_OUT), dtype=np.int32)

    for m in range(C_OUT):
        for oh in range(SIZE_OUT):
            for ow in range(SIZE_OUT):
                ih = oh * S
                iw = ow * S
                patch = inp[:, ih:ih+K, iw:iw+K]
                val   = np.sum(patch.astype(np.int32) *
                               weight_int8[m].astype(np.int32))
                psum[m, oh, ow] = val + bias_int32[m]

        if (m + 1) % 8 == 0 or m == C_OUT - 1:
            print(f"    conv channel {m+1}/{C_OUT} done")

    print(f"    psum range: [{psum.min()}, {psum.max()}]")

    # 3. 再量化 (per-channel)
    out = np.zeros((C_OUT, SIZE_OUT, SIZE_OUT), dtype=np.int8)

    for m in range(C_OUT):
        scaled = psum[m].astype(np.int64) * np.int64(scale_m0[m])
        sh     = int(scale_shift[m])
        if sh > 0:
            rounded = (scaled + (1 << (sh - 1))) >> sh
        else:
            rounded = scaled
        out[m] = np.clip(rounded, -128, 127).astype(np.int8)

    return out

# ================================================================
# main
# ================================================================
def main():
    print("=" * 60)
    print("  INT8 Conv2D  Python Reference")
    print("=" * 60)
    print(f"  Input : {C_IN} x {SIZE_IN} x {SIZE_IN}")
    print(f"  Output: {C_OUT} x {SIZE_OUT} x {SIZE_OUT} = {OUT_SIZE}")
    print(f"  Kernel: {K}x{K}  stride={S}  pad={P}")
    print(f"  HLS compare: {'ON' if COMPARE_HLS else 'OFF'}")
    print()

    # ---------- 1. 读文件 ----------
    print("[1] Loading files ...")
    weight = load_bin(WEIGHTS_PATH, np.int8,  WEIGHT_COUNT).reshape(C_OUT, C_IN, K, K)
    bias   = load_bin(BIAS_PATH,    np.int32, C_OUT)
    inp    = load_bin(INPUTS_PATH,  np.int8,  INPUT_COUNT).reshape(C_IN, SIZE_IN, SIZE_IN)
    m0     = load_bin(MULT_PATH,    np.int32, C_OUT)
    shift  = load_bin(SHIFT_PATH,   np.uint8, C_OUT)

    print(f"\n  weight[0:8] = {weight.flatten()[:8]}")
    print(f"  bias[0:8]   = {bias[:8]}")
    print(f"  input[0:8]  = {inp.flatten()[:8]}")
    print(f"  m0[0:8]     = {m0[:8]}")
    print(f"  shift[0:8]  = {shift[:8]}")

    # ---------- 2. 卷积 ----------
    print(f"\n[2] Running conv ...")
    out = conv2d_int8(inp, weight, bias, m0, shift)

    print(f"\n  output range: [{out.min()}, {out.max()}]")
    print(f"  output[0:16] = {out.flatten()[:16]}")
    zero_ratio = np.sum(out == 0) / out.size * 100
    print(f"  zero ratio:  {zero_ratio:.2f}%")

    # ---------- 3. 保存 ----------
    print(f"\n[3] Saving ...")
    out.flatten().tofile(OUTPUT_PATH)
    print(f"  (^_^) Saved bin: {OUTPUT_PATH}")
    np.savetxt(OUTPUT_TXT_PATH, out.flatten(), fmt='%d')
    print(f"  (^_^) Saved txt: {OUTPUT_TXT_PATH}")

    # ---------- 4. 与 HLS 对比 ----------
    if not COMPARE_HLS:
        print(f"\n[4] HLS compare OFF, skip")
    elif not os.path.exists(HLS_OUTPUT_PATH):
        print(f"\n[4] (o_O) HLS file not found: {HLS_OUTPUT_PATH}")
    else:
        print(f"\n[4] Comparing with HLS output ...")
        print(f"  HLS bin: {HLS_OUTPUT_PATH}")

        ref = np.fromfile(HLS_OUTPUT_PATH, dtype=np.int8)
        if ref.size != out.size:
            print(f"  (o_O) size mismatch: py={out.size} hls={ref.size}")
        else:
            py_flat  = out.flatten().astype(np.int16)
            hls_flat = ref.astype(np.int16)
            diff     = np.abs(py_flat - hls_flat)

            max_diff = int(diff.max())
            mean_diff = diff.mean()
            exact    = int(np.sum(diff == 0))
            within1  = int(np.sum(diff <= 1))
            total    = out.size

            print(f"  max diff:     {max_diff}")
            print(f"  mean diff:    {mean_diff:.4f}")
            print(f"  exact match:  {exact}/{total} ({100*exact/total:.2f}%)")
            print(f"  within +/-1:  {within1}/{total} ({100*within1/total:.2f}%)")

            if max_diff > 1:
                # 打印差异最大的位置
                idx = int(np.argmax(diff))
                c   = idx // (SIZE_OUT * SIZE_OUT)
                rem = idx %  (SIZE_OUT * SIZE_OUT)
                h   = rem // SIZE_OUT
                w   = rem %  SIZE_OUT
                print(f"\n  worst @ [{c},{h},{w}]: py={py_flat[idx]} hls={hls_flat[idx]} diff={diff[idx]}")
                print(f"\n  (T_T) FAIL  max_diff={max_diff}")
            else:
                print(f"\n  \\(^o^)/ PASS!")

        # HLS txt 存在则提示
        if os.path.exists(HLS_OUTPUT_TXT):
            print(f"  (^_^) HLS txt also available: {HLS_OUTPUT_TXT}")

    print(f"\n  (*^_^*) Done!")


if __name__ == "__main__":
    main()