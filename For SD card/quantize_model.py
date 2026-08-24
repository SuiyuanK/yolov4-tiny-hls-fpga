import numpy as np
import json
import os
import re
from pathlib import Path

# ================================================================
# 量化配置
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_DIR = os.path.join(BASE_DIR, "folded_weight")
INPUT_DIR = os.path.join(BASE_DIR, "yolo_test")
QUANT_ROOT_DIR = os.path.join(BASE_DIR, "quantized")
QUANT_WEIGHT_DIR = os.path.join(QUANT_ROOT_DIR, "folded_weight")
QUANT_INPUT_DIR = os.path.join(QUANT_ROOT_DIR, "yolo_test")
ACTIVATION_STATS_FILE = os.path.join(BASE_DIR, "统计.txt")
OUTPUT_JSON = os.path.join(QUANT_ROOT_DIR, "quantization_params.json")

WEIGHT_BITS = 8
BIAS_BITS = 32

# ================================================================
# 网络层定义
# 新增: M (输出通道数), C (输入通道数), K (卷积核大小)
# 用于将一维权重 reshape 为 [M, C, K, K] 以实现 per-channel 量化
# ================================================================
LAYER_DEFINITIONS = [
    {
        "name": "Backbone_basic_conv1",
        "weight_dir": "BasicConv1",
        "input_act": "Yolo4_Tiny",
        "output_act": "Backbone_basic_conv1/output",
        "M": 32, "C": 3, "K": 3,
    },
    {
        "name": "Backbone_basic_conv2",
        "weight_dir": "BasicConv2",
        "input_act": "Backbone_basic_conv2",
        "output_act": "Backbone_basic_conv2/output",
        "M": 64, "C": 32, "K": 3,
    },
    {
        "name": "Backbone_resblock1_conv1",
        "weight_dir": "ResBlock1",
        "weight_file_prefix": "w1",
        "input_act": "Backbone_resblock1_conv1",
        "output_act": "Backbone_resblock1_conv1/output",
        "M": 64, "C": 64, "K": 3,
    },
    {
        "name": "Backbone_resblock1_conv2",
        "weight_dir": "ResBlock1",
        "weight_file_prefix": "w2",
        "input_act": "Backbone_resblock1_conv2",
        "output_act": "Backbone_resblock1_conv2/output",
        "M": 32, "C": 32, "K": 3,
    },
    {
        "name": "Backbone_resblock1_conv3",
        "weight_dir": "ResBlock1",
        "weight_file_prefix": "w3",
        "input_act": "Backbone_resblock1_conv3",
        "output_act": "Backbone_resblock1_conv3/output",
        "M": 32, "C": 32, "K": 3,
    },
    {
        "name": "Backbone_resblock1_conv4",
        "weight_dir": "ResBlock1",
        "weight_file_prefix": "w4",
        "input_act": "Backbone_resblock1_conv4",
        "output_act": "Backbone_resblock1_conv4/output",
        "M": 64, "C": 64, "K": 1,  # 1x1 conv (pw)
    },
    {
        "name": "Backbone_resblock2_conv1",
        "weight_dir": "ResBlock2",
        "weight_file_prefix": "w1",
        "input_act": "Backbone_resblock2_conv1",
        "output_act": "Backbone_resblock2_conv1/output",
        "M": 128, "C": 128, "K": 3,
    },
    {
        "name": "Backbone_resblock2_conv2",
        "weight_dir": "ResBlock2",
        "weight_file_prefix": "w2",
        "input_act": "Backbone_resblock2_conv2",
        "output_act": "Backbone_resblock2_conv2/output",
        "M": 64, "C": 64, "K": 3,
    },
    {
        "name": "Backbone_resblock2_conv3",
        "weight_dir": "ResBlock2",
        "weight_file_prefix": "w3",
        "input_act": "Backbone_resblock2_conv3",
        "output_act": "Backbone_resblock2_conv3/output",
        "M": 64, "C": 64, "K": 3,
    },
    {
        "name": "Backbone_resblock2_conv4",
        "weight_dir": "ResBlock2",
        "weight_file_prefix": "w4",
        "input_act": "Backbone_resblock2_conv4",
        "output_act": "Backbone_resblock2_conv4/output",
        "M": 128, "C": 128, "K": 1,
    },
    {
        "name": "Backbone_resblock3_conv1",
        "weight_dir": "ResBlock3",
        "weight_file_prefix": "w1",
        "input_act": "Backbone_resblock3_conv1",
        "output_act": "Backbone_resblock3_conv1/output",
        "M": 256, "C": 256, "K": 3,
    },
    {
        "name": "Backbone_resblock3_conv2",
        "weight_dir": "ResBlock3",
        "weight_file_prefix": "w2",
        "input_act": "Backbone_resblock3_conv2",
        "output_act": "Backbone_resblock3_conv2/output",
        "M": 128, "C": 128, "K": 3,
    },
    {
        "name": "Backbone_resblock3_conv3",
        "weight_dir": "ResBlock3",
        "weight_file_prefix": "w3",
        "input_act": "Backbone_resblock3_conv3",
        "output_act": "Backbone_resblock3_conv3/output",
        "M": 128, "C": 128, "K": 3,
    },
    {
        "name": "Backbone_resblock3_conv4",
        "weight_dir": "ResBlock3",
        "weight_file_prefix": "w4",
        "input_act": "Backbone_resblock3_conv4",
        "output_act": "Backbone_resblock3_conv4/output",
        "M": 256, "C": 256, "K": 1,
    },
    {
        "name": "Backbone_basic_conv3",
        "weight_dir": "BasicConv3",
        "input_act": "Backbone_basic_conv3",
        "output_act": "Backbone_basic_conv3/output",
        "M": 512, "C": 512, "K": 3,
    },
    {
        "name": "conv_forP5",
        "weight_dir": "conv_forP5",
        "input_act": "conv_forP5",
        "output_act": "conv_forP5/output",
        "M": 256, "C": 512, "K": 1,
    },
    {
        "name": "yolo_headP5_conv1",
        "weight_dir": "yolo_headP5",
        "weight_file_prefix": "w1",
        "input_act": "yolo_headP5_conv1",
        "output_act": "yolo_headP5_conv1/output",
        "M": 512, "C": 256, "K": 3,
    },
    {
        "name": "yolo_headP5_conv2",
        "weight_dir": "yolo_headP5",
        "weight_file_prefix": "w2",
        "input_act": "yolo_headP5_conv2",
        "output_act": "yolo_headP5_conv2/output",
        "M": 75, "C": 512, "K": 1,  # 3*(80+5)=255 for COCO, 根据实际调整
    },
    {
        "name": "upsample_conv",
        "weight_dir": "upsample",
        "input_act": "upsample_conv",
        "output_act": "upsample_conv/output",
        "M": 128, "C": 256, "K": 1,
    },
    {
        "name": "yolo_headP4_conv1",
        "weight_dir": "yolo_headP4",
        "weight_file_prefix": "w1",
        "input_act": "yolo_headP4_conv1",
        "output_act": "yolo_headP4_conv1/output",
        "M": 256, "C": 384, "K": 3,  # 128(upsample) + 256(backbone) = 384
    },
    {
        "name": "yolo_headP4_conv2",
        "weight_dir": "yolo_headP4",
        "weight_file_prefix": "w2",
        "input_act": "yolo_headP4_conv2",
        "output_act": "yolo_headP4_conv2/output",
        "M": 75, "C": 256, "K": 1,  # 同上
    },
]


# ================================================================
# 辅助函数
# ================================================================


def find_layer_bin_files(weight_path, weight_file_prefix):
    """按层前缀查找对应的权重/偏置文件。"""
    if weight_file_prefix in (None, "", "*"):
        weight_files = sorted(weight_path.glob("w.bin")) + sorted(weight_path.glob("weight.bin"))
        bias_files = sorted(weight_path.glob("b.bin")) + sorted(weight_path.glob("bias.bin"))
        return list(dict.fromkeys(weight_files)), list(dict.fromkeys(bias_files))

    weight_files = sorted(weight_path.glob(f"{weight_file_prefix}.bin"))
    if not weight_files:
        weight_files = sorted(weight_path.glob(f"{weight_file_prefix}*.bin"))

    bias_prefix = None
    if weight_file_prefix.startswith("w") and len(weight_file_prefix) > 1:
        bias_prefix = f"b{weight_file_prefix[1:]}"

    bias_files = []
    if bias_prefix:
        bias_files = sorted(weight_path.glob(f"{bias_prefix}.bin"))
        if not bias_files:
            bias_files = sorted(weight_path.glob(f"{bias_prefix}*.bin"))

    return list(dict.fromkeys(weight_files)), list(dict.fromkeys(bias_files))


def load_activation_stats(txt_path):
    """从 统计.txt 加载每层的激活值统计信息"""
    stats = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(
                r'\[(.*?)\] (input|output): min = ([\-\.\deE\+]+), max = ([\-\.\deE\+]+)',
                line
            )
            if match:
                layer_base = match.group(1)
                io_type = match.group(2)
                min_val = float(match.group(3))
                max_val = float(match.group(4))

                if io_type == 'input':
                    stats[layer_base] = {'min': min_val, 'max': max_val}
                else:
                    stats[f"{layer_base}/output"] = {'min': min_val, 'max': max_val}
    return stats


def compute_activation_scale(min_val, max_val):
    """
    对称量化 scale: threshold / 127
    """
    threshold = max(abs(min_val), abs(max_val))
    if threshold < 1e-10:
        return 1.0 / 127.0
    return threshold / 127.0


def quantize_weight_per_channel(weight_float_4d):
    """
    逐通道（per output channel）对称量化权重到 INT8
    weight_float_4d: shape [M, C, K, K] 或 [M, C]（1x1视为[M,C,1,1]）
    返回:
        weight_int8: 同 shape, int8
        w_scale: shape [M], 每个输出通道的 scale (float64)
    """
    M = weight_float_4d.shape[0]
    w_scale = np.zeros(M, dtype=np.float64)
    weight_int8 = np.zeros_like(weight_float_4d, dtype=np.int8)

    for m in range(M):
        channel_weight = weight_float_4d[m].flatten().astype(np.float64)
        abs_max = np.max(np.abs(channel_weight))

        if abs_max < 1e-10:
            w_scale[m] = 1.0 / 127.0
            weight_int8[m] = 0
        else:
            w_scale[m] = abs_max / 127.0
            q = np.round(channel_weight / w_scale[m])
            q = np.clip(q, -128, 127)
            weight_int8[m] = q.reshape(weight_float_4d[m].shape).astype(np.int8)

    return weight_int8, w_scale


def quantize_bias_per_channel(bias_float, input_scale, w_scale_per_channel):
    """
    偏置量化 (per-channel):
        b_int32[m] = round(b_float[m] / (S_in * S_w[m]))

    bias_float: shape [M]
    input_scale: float (S_in)
    w_scale_per_channel: shape [M] (S_w)
    返回: bias_int32 shape [M], dtype int32
    """
    M = len(bias_float)
    bias_int32 = np.zeros(M, dtype=np.int32)

    for m in range(M):
        bias_scale = input_scale * w_scale_per_channel[m]
        if bias_scale < 1e-20:
            bias_int32[m] = 0
        else:
            val = np.round(bias_float[m].astype(np.float64) / bias_scale)
            val = np.clip(val, -(2**31), 2**31 - 1)
            bias_int32[m] = int(val)

    return bias_int32


def compute_requant_params(input_scale, w_scale_per_channel, output_scale):
    """
    计算再量化参数（逐通道）:
        real_scale[m] = (S_in * S_w[m]) / S_out
    分解为:
        real_scale[m] = M0[m] * 2^(-shift[m])
    其中 M0[m] ∈ [2^30, 2^31)

    返回:
        multiplier: shape [M], dtype int32  — scale_m0_all
        rshift:     shape [M], dtype uint8  — scale_shift_all
    """
    M = len(w_scale_per_channel)
    multiplier = np.zeros(M, dtype=np.int32)
    rshift = np.zeros(M, dtype=np.uint8)

    for m in range(M):
        real_scale = (input_scale * w_scale_per_channel[m]) / output_scale

        if real_scale <= 0:
            multiplier[m] = 0
            rshift[m] = 0
            continue

        # 归一化: 找 shift 使 M0 = real_scale * 2^shift ∈ [2^30, 2^31)
        # 即 M0_normalized = real_scale * 2^shift, 0.5 <= M0_normalized < 1.0
        # M0_int = round(M0_normalized * 2^31) ∈ [2^30, 2^31)
        # 总移位量 total_shift = shift + 31

        M0_normalized = real_scale
        shift = 0

        if M0_normalized >= 1.0:
            while M0_normalized >= 1.0:
                M0_normalized /= 2.0
                shift -= 1
        elif M0_normalized < 0.5:
            while M0_normalized < 0.5:
                M0_normalized *= 2.0
                shift += 1

        # M0_normalized 现在在 [0.5, 1.0)
        M0_int = int(np.round(M0_normalized * (2**31)))
        if M0_int >= 2**31:
            M0_int = 2**31 - 1  # 防溢出

        total_shift = shift + 31

        # 安全检查
        if total_shift < 0:
            # real_scale 非常大（不常见），左移 M0 补偿
            M0_int = M0_int << (-total_shift)
            if M0_int >= 2**31:
                M0_int = 2**31 - 1
            total_shift = 0

        total_shift = min(total_shift, 63)

        multiplier[m] = M0_int
        rshift[m] = total_shift

    return multiplier, rshift


def quantize_input_symmetric(data, scale):
    """对输入数据进行对称量化"""
    q = np.round(data.astype(np.float64) / scale)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q


# ================================================================
# 主量化流程
# ================================================================

def quantize_all_layers(activation_stats):
    """遍历所有定义的层，进行 per-channel 量化"""

    os.makedirs(QUANT_WEIGHT_DIR, exist_ok=True)

    all_params = {"layers": {}}

    for layer_def in LAYER_DEFINITIONS:
        layer_name = layer_def["name"]
        weight_dir_name = layer_def["weight_dir"]
        input_act_name = layer_def["input_act"]
        output_act_name = layer_def["output_act"]
        M = layer_def["M"]
        C = layer_def["C"]
        K = layer_def["K"]
        weight_file_prefix = layer_def.get("weight_file_prefix", "*")

        print(f"\n{'='*60}")
        print(f"量化层: {layer_name}  [M={M}, C={C}, K={K}]")
        print(f"{'='*60}")

        # ---- 获取输入/输出激活 scale ----
        if input_act_name not in activation_stats:
            print(f"  ❌ 找不到输入激活 '{input_act_name}'，跳过")
            continue
        if output_act_name not in activation_stats:
            print(f"  ❌ 找不到输出激活 '{output_act_name}'，跳过")
            continue

        input_stats = activation_stats[input_act_name]
        output_stats = activation_stats[output_act_name]

        S_in = compute_activation_scale(input_stats['min'], input_stats['max'])
        S_out = compute_activation_scale(output_stats['min'], output_stats['max'])

        print(f"  S_in  = {S_in:.6e}  (from '{input_act_name}')")
        print(f"  S_out = {S_out:.6e}  (from '{output_act_name}')")

        # ---- 查找权重/偏置文件 ----
        weight_path = Path(WEIGHT_DIR) / weight_dir_name
        if not weight_path.exists():
            print(f"  ❌ 权重目录不存在 '{weight_path}'，跳过")
            continue

        out_layer_dir = Path(QUANT_WEIGHT_DIR) / weight_dir_name
        out_layer_dir.mkdir(parents=True, exist_ok=True)

        weight_files, bias_files = find_layer_bin_files(weight_path, weight_file_prefix)

        # 兜底: 按文件大小猜测
        if not weight_files and not bias_files:
            all_bins = sorted(weight_path.glob("*.bin"))
            if len(all_bins) >= 2:
                sizes = [(f, f.stat().st_size) for f in all_bins]
                sizes.sort(key=lambda x: x[1], reverse=True)
                weight_files = [item[0] for item in sizes[:-1]]
                bias_files = [sizes[-1][0]]
            elif len(all_bins) == 1:
                weight_files = all_bins
                bias_files = []

        print(f"  权重文件: {[f.name for f in weight_files]}")
        print(f"  偏置文件: {[f.name for f in bias_files]}")

        # ---- 量化权重 (per-channel) ----
        w_scale_per_channel = None  # shape [M]

        for wf in weight_files:
            weight_float = np.fromfile(wf, dtype=np.float32)
            expected_size = M * C * K * K
            print(f"  权重 '{wf.name}': size={weight_float.size}, expected={expected_size}")

            if weight_float.size != expected_size:
                print(f"    ⚠️  大小不匹配! 预期 M*C*K*K = {M}*{C}*{K}*{K} = {expected_size}")
                print(f"    → 尝试自动推断 C = {weight_float.size // (M * K * K)}")
                C_inferred = weight_float.size // (M * K * K)
                if C_inferred * M * K * K == weight_float.size:
                    C_actual = C_inferred
                    print(f"    → 使用推断的 C = {C_actual}")
                else:
                    print(f"    ❌ 无法推断形状，跳过此权重文件")
                    continue
            else:
                C_actual = C

            # Reshape 为 [M, C, K, K]
            weight_4d = weight_float.reshape(M, C_actual, K, K)
            print(f"    reshape → [{M}, {C_actual}, {K}, {K}]")
            print(f"    range: [{weight_float.min():.6f}, {weight_float.max():.6f}]")

            # Per-channel 量化
            weight_int8, w_scale_per_channel = quantize_weight_per_channel(weight_4d)

            print(f"    S_w per-channel: min={w_scale_per_channel.min():.6e}, "
                  f"max={w_scale_per_channel.max():.6e}, "
                  f"mean={w_scale_per_channel.mean():.6e}")

            # 保存 INT8 权重 (展平回一维，与原始存储格式一致)
            out_weight_file = out_layer_dir / f"{wf.stem}_int8.bin"
            weight_int8.flatten().tofile(out_weight_file)
            print(f"    → 保存: {out_weight_file.name}")

        if w_scale_per_channel is None:
            print(f"  ❌ 未成功量化任何权重文件，跳过此层")
            continue

        # ---- 量化偏置 (per-channel) ----
        bias_int32 = None

        for bf in bias_files:
            bias_float = np.fromfile(bf, dtype=np.float32)
            M_bias = bias_float.size
            print(f"  偏置 '{bf.name}': M={M_bias}")

            if M_bias != M:
                print(f"    ⚠️  偏置通道数 {M_bias} ≠ 权重输出通道数 {M}")
                if M_bias < M:
                    print(f"    → 偏置补零到 {M} 通道")
                    bias_float_padded = np.zeros(M, dtype=np.float32)
                    bias_float_padded[:M_bias] = bias_float
                    bias_float = bias_float_padded
                else:
                    print(f"    → 截断偏置到 {M} 通道")
                    bias_float = bias_float[:M]

            print(f"    range: [{bias_float.min():.6f}, {bias_float.max():.6f}]")

            # Per-channel 偏置量化: b_int32[m] = round(b_float[m] / (S_in * S_w[m]))
            bias_int32 = quantize_bias_per_channel(
                bias_float, S_in, w_scale_per_channel
            )

            out_bias_file = out_layer_dir / f"{bf.stem}_int32.bin"
            bias_int32.tofile(out_bias_file)
            print(f"    → 保存: {out_bias_file.name}")
            print(f"    → bias_int32 range: [{bias_int32.min()}, {bias_int32.max()}]")

        # ---- 计算再量化参数 (per-channel) ----
        multiplier_arr, rshift_arr = compute_requant_params(
            S_in, w_scale_per_channel, S_out
        )

        # 保存再量化参数
        # 文件名统一用层的权重前缀
        prefix = weight_file_prefix if weight_file_prefix not in (None, "", "*") else "w"
        mult_file = out_layer_dir / f"{prefix}_requant_mult.bin"
        shift_file = out_layer_dir / f"{prefix}_requant_shift.bin"
        multiplier_arr.tofile(mult_file)
        rshift_arr.tofile(shift_file)

        print(f"  再量化参数 (per-channel, M={M}):")
        print(f"    M0   range: [{multiplier_arr.min()}, {multiplier_arr.max()}]")
        print(f"    shift range: [{rshift_arr.min()}, {rshift_arr.max()}]")
        print(f"    → 保存: {mult_file.name}, {shift_file.name}")

        # ---- 验证: 抽样检查再量化精度 ----
        real_scales = (S_in * w_scale_per_channel) / S_out
        reconstructed = multiplier_arr.astype(np.float64) * (2.0 ** (-rshift_arr.astype(np.float64)))
        rel_error = np.abs(reconstructed - real_scales) / (np.abs(real_scales) + 1e-20)
        print(f"  再量化精度验证:")
        print(f"    real_scale range: [{real_scales.min():.6e}, {real_scales.max():.6e}]")
        print(f"    最大相对误差: {rel_error.max():.2e}")

        # ---- 保存到 JSON ----
        all_params["layers"][layer_name] = {
            "input_act": input_act_name,
            "output_act": output_act_name,
            "M": M,
            "C": C,
            "K": K,
            "S_in": float(S_in),
            "S_out": float(S_out),
            "S_w_min": float(w_scale_per_channel.min()),
            "S_w_max": float(w_scale_per_channel.max()),
            "S_w_mean": float(w_scale_per_channel.mean()),
            "real_scale_min": float(real_scales.min()),
            "real_scale_max": float(real_scales.max()),
            "M0_int_min": int(multiplier_arr.min()),
            "M0_int_max": int(multiplier_arr.max()),
            "shift_min": int(rshift_arr.min()),
            "shift_max": int(rshift_arr.max()),
            "requant_max_rel_error": float(rel_error.max()),
            "weight_files": [f.name for f in weight_files],
            "bias_files": [f.name for f in bias_files],
        }

    return all_params


def quantize_inputs(activation_stats):
    """量化输入数据"""
    if not os.path.exists(INPUT_DIR):
        print(f"\n⚠️  输入目录不存在: {INPUT_DIR}")
        return

    input_files = list(Path(INPUT_DIR).glob("*.bin"))
    if not input_files:
        print(f"\n⚠️  {INPUT_DIR} 下没有 .bin 文件")
        return

    os.makedirs(QUANT_INPUT_DIR, exist_ok=True)

    first_input_act = "Yolo4_Tiny"
    if first_input_act not in activation_stats:
        print(f"⚠️  找不到输入激活 '{first_input_act}'")
        return

    stats = activation_stats[first_input_act]
    S_input = compute_activation_scale(stats['min'], stats['max'])

    print(f"\n{'='*60}")
    print(f"量化输入数据")
    print(f"{'='*60}")
    print(f"  S_input = {S_input:.6e}")
    print(f"  范围: [{stats['min']:.3f}, {stats['max']:.3f}]")

    for input_file in input_files:
        data = np.fromfile(input_file, dtype=np.float32)
        print(f"\n  处理: {input_file.name} (size={data.size})")
        print(f"    数据范围: [{data.min():.4f}, {data.max():.4f}]")

        data_int8 = quantize_input_symmetric(data, S_input)

        out_file = Path(QUANT_INPUT_DIR) / f"{input_file.stem}_int8.bin"
        data_int8.tofile(out_file)
        print(f"    → {out_file.name}")

    scale_file = Path(QUANT_INPUT_DIR) / "input_scale.json"
    with open(scale_file, 'w') as f:
        json.dump({"S_input": float(S_input), "activation": first_input_act}, f, indent=2)
    print(f"\n  输入 scale 保存到: {scale_file}")


# ================================================================
# 主函数
# ================================================================

def main():
    print("=" * 60)
    print("  INT8 量化工具 (对称量化, per-channel 权重, 逐层再量化)")
    print("=" * 60)

    if not os.path.exists(WEIGHT_DIR):
        print(f"❌ 权重目录不存在: {WEIGHT_DIR}")
        return

    if not os.path.exists(ACTIVATION_STATS_FILE):
        print(f"❌ 激活统计文件不存在: {ACTIVATION_STATS_FILE}")
        return

    # 加载激活统计
    print(f"\n加载激活统计: {ACTIVATION_STATS_FILE}")
    activation_stats = load_activation_stats(ACTIVATION_STATS_FILE)
    print(f"  共 {len(activation_stats)} 条记录")

    # 打印可用的激活名列表 (调试用)
    print(f"  可用的激活名:")
    for name in sorted(activation_stats.keys()):
        s = activation_stats[name]
        print(f"    {name}: [{s['min']:.4f}, {s['max']:.4f}]")

    # 量化所有层
    all_params = quantize_all_layers(activation_stats)

    # 保存总参数
    os.makedirs(QUANT_ROOT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_params, f, indent=2, ensure_ascii=False)
    print(f"\n量化参数 JSON: {OUTPUT_JSON}")

    # 量化输入
    quantize_inputs(activation_stats)

    # 打印总结
    print(f"\n{'='*60}")
    print(f"  ✅ 量化完成!")
    print(f"{'='*60}")
    print(f"\n输出目录结构:")
    print(f"  {QUANT_ROOT_DIR}/")
    print(f"  ├── quantization_params.json     ← 所有层参数汇总")
    print(f"  ├── folded_weight/")
    print(f"  │   ├── BasicConv1/")
    print(f"  │   │   ├── w_int8.bin           ← INT8 权重 [M*C*K*K]")
    print(f"  │   │   ├── b_int32.bin          ← INT32 偏置 [M]")
    print(f"  │   │   ├── w_requant_mult.bin   ← M0 乘子 [M] (int32)")
    print(f"  │   │   └── w_requant_shift.bin  ← 移位数 [M] (uint8)")
    print(f"  │   ├── ResBlock1/")
    print(f"  │   │   ├── w1_int8.bin          ← 子层1权重")
    print(f"  │   │   ├── b1_int32.bin")
    print(f"  │   │   ├── w1_requant_mult.bin")
    print(f"  │   │   ├── w1_requant_shift.bin")
    print(f"  │   │   ├── w2_int8.bin          ← 子层2权重")
    print(f"  │   │   └── ...")
    print(f"  │   └── ...")
    print(f"  └── yolo_test/")
    print(f"      ├── *_int8.bin               ← INT8 输入")
    print(f"      └── input_scale.json")
    print(f"\nHLS 硬件再量化公式 (per-channel):")
    print(f"  对每个输出通道 m:")
    print(f"    scaled = (int64)psum_int32[m] * (int64)M0[m]")
    print(f"    rounded = (scaled + (1 << (shift[m]-1))) >> shift[m]")
    print(f"    output_int8[m] = saturate(rounded, -128, 127)")


if __name__ == "__main__":
    main()