import struct
import os
from pathlib import Path

def bin_to_txt(bin_file_path, txt_file_path=None):
    """
    将 .bin 文件按 int16 格式读取并输出为 .txt 文件
    
    Args:
        bin_file_path: .bin 文件的路径
        txt_file_path: 输出 .txt 文件的路径，如果为 None 则自动生成
    """
    bin_path = Path(bin_file_path)
    
    if not bin_path.exists():
        print(f"错误: 文件 {bin_file_path} 不存在")
        return False
    
    # 如果没有指定输出路径，自动生成
    if txt_file_path is None:
        txt_file_path = bin_path.with_suffix('.txt')
    
    try:
        # 读取 .bin 文件
        with open(bin_path, 'rb') as f:
            data = f.read()
        
        # 计算 int16 的个数
        num_values = len(data) // 2
        if len(data) % 2 != 0:
            print(f"警告: 文件字节数 {len(data)} 不是 2 的倍数，多出的字节会被忽略")
        
        # 按 int16 解析
        values = struct.unpack(f'<{num_values}h', data[:num_values * 2])
        
        # 输出为 .txt 文件
        with open(txt_file_path, 'w') as f:
            for i, val in enumerate(values):
                if i % 16 == 0 and i > 0:
                    f.write('\n')
                f.write(f"{val:6d} ")
            f.write('\n')
        
        print(f"✓ 转换成功: {bin_path} → {txt_file_path}")
        print(f"  共转换了 {num_values} 个 int16 值")
        return True
        
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        return False

def batch_convert(directory=None):
    """
    批量转换目录中的所有 .bin 文件
    
    Args:
        directory: 要转换的目录，如果为 None 则使用当前目录
    """
    if directory is None:
        directory = os.getcwd()
    
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"错误: {directory} 不是有效的目录")
        return
    
    # 查找所有 .bin 文件
    bin_files = list(dir_path.rglob('*.bin'))
    
    if not bin_files:
        print(f"在 {directory} 中没有找到 .bin 文件")
        return
    
    print(f"找到 {len(bin_files)} 个 .bin 文件")
    print("-" * 50)
    
    success_count = 0
    for bin_file in bin_files:
        if bin_to_txt(bin_file):
            success_count += 1
    
    print("-" * 50)
    print(f"转换完成: 成功 {success_count}/{len(bin_files)} 个文件")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # 如果提供了命令行参数
        arg = sys.argv[1]
        if len(sys.argv) > 2:
            # 两个参数：输入文件和输出文件
            bin_to_txt(arg, sys.argv[2])
        else:
            # 一个参数：可能是文件或目录
            path = Path(arg)
            if path.is_file():
                bin_to_txt(arg)
            elif path.is_dir():
                batch_convert(arg)
    else:
        # 没有参数，批量转换当前目录
        print("使用方式:")
        print("  python bin_to_txt.py                           # 转换当前目录的所有 .bin 文件")
        print("  python bin_to_txt.py <directory>              # 转换指定目录的所有 .bin 文件")
        print("  python bin_to_txt.py <input.bin>              # 转换单个 .bin 文件")
        print("  python bin_to_txt.py <input.bin> <output.txt> # 指定输出文件名")
        print()
        print("开始批量转换当前目录...")
        batch_convert()
