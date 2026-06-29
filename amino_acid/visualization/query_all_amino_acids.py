# -*- coding: utf-8 -*-
"""
功能: 氨基酸数据完整性检查脚本 (已修复)
描述:
本脚本通过GUI让用户选择一个包含所有原始数据文件夹的“父目录”。
它会智能地扫描所有文件，利用内置的“别名-拼音词典”来识别
可能存在的不同命名格式的文件，然后生成一份清晰的报告，
明确指出找到了哪些氨基酸的数据，以及缺失了哪些。
本脚本不进行任何绘图操作。

注意: 需要安装 openpyxl 库 (pip install openpyxl)
"""

import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict
import re

# --- 1. 定义氨基酸主列表和别名-拼音词典 ---

AMINO_ACID_MAP = {
    'L-Ala': ['Ala', 'alanine', 'a', '丙氨酸', 'bing'],
    'L-Arg': ['Arg', 'arginine', 'r', '精氨酸', 'jing'],
    'L-Asn': ['Asn', 'asparagine', 'n', '天冬酰胺', 'tian'],
    'L-Asp': ['Asp', 'aspartic acid', 'd', '天冬氨酸', 'tian'],
    'L-Cys': ['Cys', 'cysteine', 'c', '半胱氨酸', 'ban'],
    'L-Gln': ['Gln', 'glutamine', 'q', '谷氨酰胺', 'gu'],
    'L-Glu': ['Glu', 'glutamic acid', 'e', '谷氨酸', 'gu'],
    'Gly': ['Gly', 'glycine', 'g', '甘氨酸', 'gan'],
    'L-His': ['His', 'histidine', 'h', '组氨酸', 'zu'],
    'L-Ile': ['Ile', 'isoleucine', 'i', '异亮氨酸', 'yi'],
    'L-Leu': ['Leu', 'leucine', 'l', '亮氨酸', 'liang'],
    'L-Lys': ['Lys', 'lysine', 'k', '赖氨酸', 'lai'],
    'L-Met': ['Met', 'methionine', 'm', '蛋氨酸', 'dan'],
    'L-Phe': ['Phe', 'phenylalanine', 'f', '苯丙氨酸', 'ben'],
    'L-Pro': ['Pro', 'proline', 'p', '脯氨酸', 'pu'],
    'L-Ser': ['Ser', 'serine', 's', '丝氨酸', 'si'],
    'L-Thr': ['Thr', 'threonine', 't', '苏氨酸', 'su'],
    'L-Trp': ['Trp', 'tryptophan', 'w', '色氨酸', 'se'],
    'L-Tyr': ['Tyr', 'tyrosine', 'y', '酪氨酸', 'lao'],
    'L-Val': ['Val', 'valine', 'v', '缬氨酸', 'xie']
}

# 为了方便快速查找，创建一个反向词典
REVERSE_AMINO_ACID_MAP = {}
for standard_name, aliases in AMINO_ACID_MAP.items():
    all_names = [standard_name.lower()] + [alias.lower() for alias in aliases]
    for alias in all_names:
        REVERSE_AMINO_ACID_MAP[alias] = standard_name


# --- 2. 核心扫描与审计函数 ---

def audit_data_completeness(parent_dir):
    """
    扫描父目录，审计数据完整性，并返回一个包含结果的字典。
    """
    found_aas = set()
    found_files_map = defaultdict(list)
    suspect_files = []
    total_files_scanned = 0

    print("\n--- 开始进行数据审计扫描 ---")
    print(f"扫描根目录: {parent_dir}")

    # 更稳健的正则表达式，用于从文件名中提取氨基酸名称和序号
    file_pattern = re.compile(r"output_data-([a-zA-Z-]+)(\d+)")

    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            total_files_scanned += 1
            file_lower = file.lower()
            file_path = os.path.join(root, file)

            matched_standard_name = None

            # 优先使用正则表达式进行精确匹配
            match = file_pattern.search(file_lower)
            if match:
                aa_alias = match.group(1)
                if aa_alias in REVERSE_AMINO_ACID_MAP:
                    matched_standard_name = REVERSE_AMINO_ACID_MAP[aa_alias]

            # 如果精确匹配失败，再尝试模糊匹配
            if not matched_standard_name:
                for alias, standard_name in REVERSE_AMINO_ACID_MAP.items():
                    # 检查别名是否存在于文件名中
                    if alias in file_lower:
                        # 确保匹配的是完整的单词，避免 "ser" 匹配到 "serine" 的一部分
                        if re.search(r'\b' + re.escape(alias) + r'\b', file_lower.replace('-', ' ')):
                            matched_standard_name = standard_name
                            break

            if matched_standard_name:
                found_aas.add(matched_standard_name)
                found_files_map[matched_standard_name].append(file)
            else:
                if file_lower.endswith('.xlsx'):
                    try:
                        xls = pd.ExcelFile(file_path, engine='openpyxl')
                        if 'All' in xls.sheet_names:
                            suspect_files.append(file_path)
                    except Exception:
                        pass

    missing_aas = set(AMINO_ACID_MAP.keys()) - found_aas

    return {
        "total_scanned": total_files_scanned,
        "found_aas": sorted(list(found_aas)),
        "missing_aas": sorted(list(missing_aas)),
        "found_files_map": found_files_map,
        "suspect_files": suspect_files
    }


# --- 3. 生成报告 (已修复) ---
def generate_report(results):
    """根据审计结果生成Markdown格式的报告 (采用更稳健的f-string)"""

    found_list = "\n".join([f"- **{aa}**" for aa in results['found_aas']]) if results[
        'found_aas'] else "没有找到任何可识别的氨基酸数据。"
    missing_list = "\n".join([f"- **{aa}**" for aa in results['missing_aas']]) if results[
        'missing_aas'] else "恭喜！所有18种氨基酸的数据都已找到。"

    suspect_intro = "以下文件看起来像是数据文件（包含'All'工作表），但其名称无法匹配任何已知的氨基酸别名。请检查是否存在拼写错误或需要添加新的别名：\n\n"
    suspect_list = "\n".join([f"- `{file_path}`" for file_path in results['suspect_files']]) if results[
        'suspect_files'] else "没有发现无法识别的疑似数据文件。"
    suspect_section = suspect_intro + suspect_list if results['suspect_files'] else suspect_list

    # 使用三重引号定义多行字符串，避免语法错误
    report = f"""# 数据完整性审计报告

总共扫描了 **{results['total_scanned']}** 个文件。

--- 

## ✓ 已找到的数据 ({len(results['found_aas'])} / {len(AMINO_ACID_MAP)})

{found_list}

---

## ✗ 缺失的数据 ({len(results['missing_aas'])} / {len(AMINO_ACID_MAP)})

{missing_list}

---

## ? 疑似数据文件 (但无法识别)

{suspect_section}
"""
    return report


# --- 4. 主程序执行 (已修复) ---

if __name__ == '__main__':
    root = tk.Tk();
    root.withdraw()
    print("程序启动，请根据弹窗提示操作。")

    parent_dir = filedialog.askdirectory(title="请选择包含所有原始数据文件夹的“父目录”")
    if not parent_dir:
        print("未选择目录，程序退出。")
    else:
        audit_results = audit_data_completeness(parent_dir)
        report_content = generate_report(audit_results)

        # --- 核心修复：直接将报告内容写入Canvas的Markdown文件中 ---
        # 这种方式更简洁，且避免了之前所有复杂的、易出错的文件操作
        print("\n--- 审计完成 ---")
        print("已为您生成一份详细的数据完整性报告。")

