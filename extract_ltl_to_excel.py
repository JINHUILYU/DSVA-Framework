#!/usr/bin/env python3
"""
Extract final_ltl_formula from JSON files in a folder and save to Excel.
Usage: python extract_ltl_to_excel.py <folder_path>
Example: python extract_ltl_to_excel.py data/output/dsva/gpt-3.5-turbo
"""

import json
import os
import sys
import re
from pathlib import Path
import pandas as pd


def extract_ltl_formulas(folder_path):
    """
    从指定文件夹中的所有JSON文件提取final_ltl_formula
    
    Args:
        folder_path: 包含JSON文件的文件夹路径
    
    Returns:
        包含提取数据的DataFrame
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"文件夹不存在: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"路径不是文件夹: {folder_path}")
    
    # 收集所有JSON文件并按文件名中的数字排序
    def get_file_number(filename):
        """从文件名中提取数字，如 result_1_xxx.json -> 1"""
        match = re.search(r'result_(\d+)', filename.name)
        return int(match.group(1)) if match else float('inf')
    
    json_files = sorted(folder.glob("*.json"), key=get_file_number)
    
    if not json_files:
        raise ValueError(f"文件夹中没有找到JSON文件: {folder_path}")
    
    # 提取数据
    data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
                data.append({
                    'filename': json_file.name,
                    'input_sentence': content.get('input_sentence', ''),
                    'final_ltl_formula': content.get('final_ltl_formula', ''),
                    'success': content.get('success', False),
                    'similarity_score': content.get('stage_results', [{}])[-1].get('verification_result', {}).get('similarity_score', None) if content.get('stage_results') else None,
                    'refinement_iterations': content.get('refinement_iterations', 0),
                    'total_processing_time': content.get('total_processing_time', 0),
                })
        except Exception as e:
            print(f"警告: 无法处理文件 {json_file.name}: {str(e)}")
            continue
    
    if not data:
        raise ValueError(f"没有成功提取到任何数据")
    
    return pd.DataFrame(data)


def save_to_excel(df, output_path):
    """
    将DataFrame保存到Excel文件
    
    Args:
        df: 要保存的DataFrame
        output_path: 输出Excel文件路径
    """
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"成功保存到: {output_path}")
    print(f"共处理 {len(df)} 个文件")


def main():
    if len(sys.argv) != 2:
        print("用法: python extract_ltl_to_excel.py <folder_path>")
        print("示例: python extract_ltl_to_excel.py data/output/dsva/gpt-3.5-turbo")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    try:
        # 提取数据
        print(f"正在处理文件夹: {folder_path}")
        df = extract_ltl_formulas(folder_path)
        
        # 生成输出文件名
        folder_name = Path(folder_path).name
        parent_folder = Path(folder_path).parent.name
        output_filename = f"{parent_folder}_{folder_name}_ltl_formulas.xlsx"
        output_path = Path(folder_path) / output_filename
        
        # 保存到Excel
        save_to_excel(df, output_path)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
