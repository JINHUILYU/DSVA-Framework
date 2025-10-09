"""
BLEU分数计算模块
用于计算生成的 MTL 表达式的质量
"""

import re
import math
from typing import List, Dict, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class BLEUCalculator:
    """BLEU分数计算器"""
    
    def __init__(self, max_n: int = 4, smoothing: bool = True):
        """
        初始化BLEU计算器
        
        Args:
            max_n: 最大n-gram长度，默认为4
            smoothing: 是否使用平滑处理，避免0分
        """
        self.max_n = max_n
        self.smoothing = smoothing
    
    def tokenize(self, text: str) -> List[str]:
        """
        文本分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词后的token列表
        """
        # 处理MTL公式中的特殊符号
        text = text.lower()
        # 保留MTL算子和符号
        text = re.sub(r'([∧∨¬→()[\],_])', r' \1 ', text)
        # 分割其他标点符号
        text = re.sub(r'([.!?;:])', r' \1 ', text)
        # 处理多个空格
        text = re.sub(r'\s+', ' ', text)
        
        tokens = text.strip().split()
        return [token for token in tokens if token]
    
    def get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """
        获取n-gram
        
        Args:
            tokens: token列表
            n: n-gram长度
            
        Returns:
            n-gram列表
        """
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def calculate_precision(self, candidate_tokens: List[str], 
                          reference_tokens: List[str], n: int) -> float:
        """
        计算n-gram精确度
        
        Args:
            candidate_tokens: 候选句子的tokens
            reference_tokens: 参考句子的tokens
            n: n-gram长度
            
        Returns:
            n-gram精确度
        """
        candidate_ngrams = self.get_ngrams(candidate_tokens, n)
        reference_ngrams = self.get_ngrams(reference_tokens, n)
        
        if not candidate_ngrams:
            return 0.0
        
        # 计算候选句子中每个n-gram的出现次数
        candidate_counts = Counter(candidate_ngrams)
        # 计算参考句子中每个n-gram的出现次数
        reference_counts = Counter(reference_ngrams)
        
        # 计算匹配的n-gram数量
        matches = 0
        for ngram, count in candidate_counts.items():
            matches += min(count, reference_counts.get(ngram, 0))
        
        # 应用平滑处理
        if self.smoothing and matches == 0:
            matches = 1e-7
            
        precision = matches / len(candidate_ngrams)
        return precision
    
    def calculate_brevity_penalty(self, candidate_length: int, 
                                reference_length: int) -> float:
        """
        计算简洁性惩罚因子
        
        Args:
            candidate_length: 候选句子长度
            reference_length: 参考句子长度
            
        Returns:
            简洁性惩罚因子
        """
        if candidate_length > reference_length:
            return 1.0
        elif candidate_length == 0:
            return 0.0
        else:
            return math.exp(1 - reference_length / candidate_length)
    
    def calculate_bleu(self, candidate: str, reference: str) -> Dict[str, float]:
        """
        计算BLEU分数
        
        Args:
            candidate: 候选句子
            reference: 参考句子
            
        Returns:
            包含各种BLEU分数的字典
        """
        # 分词
        candidate_tokens = self.tokenize(candidate)
        reference_tokens = self.tokenize(reference)
        
        if not candidate_tokens or not reference_tokens:
            return {
                'bleu_score': 0.0,
                'bleu_1': 0.0,
                'bleu_2': 0.0,
                'bleu_3': 0.0,
                'bleu_4': 0.0,
                'brevity_penalty': 0.0,
                'candidate_length': len(candidate_tokens),
                'reference_length': len(reference_tokens)
            }
        
        # 计算各个n-gram的精确度
        precisions = []
        for n in range(1, self.max_n + 1):
            precision = self.calculate_precision(candidate_tokens, reference_tokens, n)
            precisions.append(precision)
        
        # 计算简洁性惩罚
        bp = self.calculate_brevity_penalty(len(candidate_tokens), len(reference_tokens))
        
        # 计算BLEU分数
        if all(p > 0 for p in precisions):
            # 几何平均
            log_precisions = [math.log(p) for p in precisions]
            bleu_score = bp * math.exp(sum(log_precisions) / len(log_precisions))
        else:
            bleu_score = 0.0
        
        return {
            'bleu_score': bleu_score,
            'bleu_1': precisions[0] if len(precisions) > 0 else 0.0,
            'bleu_2': precisions[1] if len(precisions) > 1 else 0.0,
            'bleu_3': precisions[2] if len(precisions) > 2 else 0.0,
            'bleu_4': precisions[3] if len(precisions) > 3 else 0.0,
            'brevity_penalty': bp,
            'candidate_length': len(candidate_tokens),
            'reference_length': len(reference_tokens),
            'candidate_tokens': candidate_tokens,
            'reference_tokens': reference_tokens
        }
    
    def calculate_corpus_bleu(self, candidates: List[str], 
                            references: List[str]) -> Dict[str, float]:
        """
        计算语料库级别的BLEU分数
        
        Args:
            candidates: 候选句子列表
            references: 参考句子列表
            
        Returns:
            语料库BLEU分数
        """
        if len(candidates) != len(references):
            raise ValueError("候选句子和参考句子数量不匹配")
        
        total_candidate_length = 0
        total_reference_length = 0
        total_matches = [0] * self.max_n
        total_candidate_ngrams = [0] * self.max_n
        
        for candidate, reference in zip(candidates, references):
            candidate_tokens = self.tokenize(candidate)
            reference_tokens = self.tokenize(reference)
            
            total_candidate_length += len(candidate_tokens)
            total_reference_length += len(reference_tokens)
            
            for n in range(1, self.max_n + 1):
                candidate_ngrams = self.get_ngrams(candidate_tokens, n)
                reference_ngrams = self.get_ngrams(reference_tokens, n)
                
                candidate_counts = Counter(candidate_ngrams)
                reference_counts = Counter(reference_ngrams)
                
                matches = 0
                for ngram, count in candidate_counts.items():
                    matches += min(count, reference_counts.get(ngram, 0))
                
                total_matches[n-1] += matches
                total_candidate_ngrams[n-1] += len(candidate_ngrams)
        
        # 计算精确度
        precisions = []
        for n in range(self.max_n):
            if total_candidate_ngrams[n] > 0:
                precision = total_matches[n] / total_candidate_ngrams[n]
            else:
                precision = 0.0
            precisions.append(precision)
        
        # 计算简洁性惩罚
        bp = self.calculate_brevity_penalty(total_candidate_length, total_reference_length)
        
        # 计算BLEU分数
        if all(p > 0 for p in precisions):
            log_precisions = [math.log(p) for p in precisions]
            bleu_score = bp * math.exp(sum(log_precisions) / len(log_precisions))
        else:
            bleu_score = 0.0
        
        return {
            'corpus_bleu': bleu_score,
            'corpus_bleu_1': precisions[0],
            'corpus_bleu_2': precisions[1],
            'corpus_bleu_3': precisions[2],
            'corpus_bleu_4': precisions[3],
            'brevity_penalty': bp,
            'total_candidate_length': total_candidate_length,
            'total_reference_length': total_reference_length
        }

def demo_bleu_calculation():
    """演示BLEU分数计算"""
    print("=== BLEU分数计算演示 ===\n")
    
    calculator = BLEUCalculator()
    
    # 测试用例
    test_cases = [
        {
            "name": "MTL公式对比",
            "candidate": "G(p → F_[0,10](q ∧ r))",
            "reference": "G(p → F_[0,10](q ∧ r))"
        },
        {
            "name": "相似MTL公式",
            "candidate": "G(p → F_[0,10](q ∧ r))",
            "reference": "G(p → F_[0,5](q ∧ r))"
        },
        {
            "name": "不同MTL公式",
            "candidate": "G(p → F_[0,10](q ∧ r))",
            "reference": "F(p ∧ G(q → r))"
        },
        {
            "name": "自然语言对比",
            "candidate": "The cat is on the mat",
            "reference": "The cat sits on the mat"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"{i}. {case['name']}")
        print(f"   候选: {case['candidate']}")
        print(f"   参考: {case['reference']}")
        
        result = calculator.calculate_bleu(case['candidate'], case['reference'])
        
        print(f"   BLEU分数: {result['bleu_score']:.4f}")
        print(f"   BLEU-1: {result['bleu_1']:.4f}")
        print(f"   BLEU-2: {result['bleu_2']:.4f}")
        print(f"   BLEU-3: {result['bleu_3']:.4f}")
        print(f"   BLEU-4: {result['bleu_4']:.4f}")
        print(f"   简洁性惩罚: {result['brevity_penalty']:.4f}")
        print(f"   长度: {result['candidate_length']} vs {result['reference_length']}")
        print()
    
    # 语料库级别BLEU演示
    print("=== 语料库BLEU演示 ===")
    candidates = [case['candidate'] for case in test_cases]
    references = [case['reference'] for case in test_cases]
    
    corpus_result = calculator.calculate_corpus_bleu(candidates, references)
    print(f"语料库BLEU: {corpus_result['corpus_bleu']:.4f}")
    print(f"语料库BLEU-1: {corpus_result['corpus_bleu_1']:.4f}")
    print(f"语料库BLEU-2: {corpus_result['corpus_bleu_2']:.4f}")
    print(f"语料库BLEU-3: {corpus_result['corpus_bleu_3']:.4f}")
    print(f"语料库BLEU-4: {corpus_result['corpus_bleu_4']:.4f}")

if __name__ == "__main__":
    demo_bleu_calculation()
