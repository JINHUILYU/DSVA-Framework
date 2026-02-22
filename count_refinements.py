#!/usr/bin/env python3
"""
统计 DSVA 和 Ablation 的 refinement_iterations 总数
"""

import json
import os
from pathlib import Path


def count_refinement_iterations(base_path):
    """
    统计指定路径下所有 JSON 文件的 refinement_iterations 总和
    
    Args:
        base_path: 基础路径（dsva 或 ablation）
    
    Returns:
        int: refinement_iterations 的总数
    """
    total_iterations = 0
    file_count = 0
    
    # 遍历所有子文件夹和文件
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'refinement_iterations' in data:
                            iterations = data['refinement_iterations']
                            total_iterations += iterations
                            file_count += 1
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error reading {file_path}: {e}")
    
    return total_iterations, file_count


def main():
    # 设置基础路径
    project_root = Path(__file__).parent
    dsva_path = project_root / 'data' / 'output' / 'dsva'
    ablation_path = project_root / 'data' / 'output' / 'ablation'
    
    # 检查路径是否存在
    if not dsva_path.exists():
        print(f"DSVA path not found: {dsva_path}")
        return
    if not ablation_path.exists():
        print(f"Ablation path not found: {ablation_path}")
        return
    
    # 统计 DSVA 的 refinement_iterations
    print("Counting DSVA refinement_iterations...")
    dsva_total, dsva_files = count_refinement_iterations(dsva_path)
    
    # 统计 Ablation 的 refinement_iterations
    print("Counting Ablation refinement_iterations...")
    ablation_total, ablation_files = count_refinement_iterations(ablation_path)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"DSVA refinement_iterations 总数: {dsva_total}")
    print(f"  - 统计文件数: {dsva_files}")
    if dsva_files > 0:
        print(f"  - 平均每个文件: {dsva_total / dsva_files:.2f}")
    
    print(f"\nAblation refinement_iterations 总数: {ablation_total}")
    print(f"  - 统计文件数: {ablation_files}")
    if ablation_files > 0:
        print(f"  - 平均每个文件: {ablation_total / ablation_files:.2f}")
    
    print(f"\n总计: {dsva_total + ablation_total}")
    print(f"总文件数: {dsva_files + ablation_files}")
    print("=" * 60)


if __name__ == '__main__':
    main()
