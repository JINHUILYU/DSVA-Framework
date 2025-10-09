"""
DSV框架消融实验模块
实现三种消融实验：
1. 移除"解构"阶段 - 端到端生成对比
2. 移除"验证"阶段 - 无验证输出对比  
3. 移除"修正循环" - 无迭代修正对比
"""

import json
import time
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from dsv_framework import DSVFramework, DSVProcessResult, TokenUsage
from end2end import End2EndNL2MTL

logger = logging.getLogger(__name__)

@dataclass
class AblationResult:
    """消融实验结果"""
    experiment_name: str
    input_sentence: str
    final_mtl_formula: Optional[str]
    success: bool
    processing_time: float
    token_usage: TokenUsage
    additional_info: Dict[str, Any]

@dataclass
class AblationComparison:
    """消融实验对比结果"""
    input_sentence: str
    full_dsv_result: AblationResult
    ablation_results: Dict[str, AblationResult]
    baseline_results: Dict[str, AblationResult]

class DSVAblationStudy:
    """DSV消融实验类"""
    
    def __init__(self, config_path: str = "config/dsv_config.json"):
        """初始化消融实验"""
        self.dsv_framework = DSVFramework(config_path)
        self.end2end_processor = End2EndNL2MTL()
        self.config = self.dsv_framework.config
        
        # 获取消融实验配置
        self.ablation_config = self.config.get("ablation_studies", {})
        self.experiments_enabled = self.ablation_config.get("experiments", {})
        
        logger.info("DSV消融实验初始化完成")
        logger.info(f"启用的实验: {list(self.experiments_enabled.keys())}")
    
    def run_no_deconstruct_experiment(self, sentence: str) -> AblationResult:
        """消融实验1：移除解构阶段，直接端到端生成"""
        start_time = time.time()
        logger.info("=== 消融实验：移除解构阶段 ===")
        
        try:
            # 使用端到端处理器
            result = self.end2end_processor.process_single(sentence)
            
            processing_time = time.time() - start_time
            
            # 转换TokenUsage类型
            dsv_token_usage = TokenUsage(
                prompt_tokens=result.token_usage.prompt_tokens,
                completion_tokens=result.token_usage.completion_tokens,
                total_tokens=result.token_usage.total_tokens
            )
            
            return AblationResult(
                experiment_name="no_deconstruct",
                input_sentence=sentence,
                final_mtl_formula=result.final_mtl_expression,
                success=result.final_mtl_expression is not None,
                processing_time=processing_time,
                token_usage=dsv_token_usage,
                additional_info={
                    "method": "end_to_end_generation",
                    "rag_enabled": result.rag_enabled,
                    "agent_response": result.agent_response[:500] + "..." if len(result.agent_response) > 500 else result.agent_response
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"无解构实验失败: {e}")
            
            return AblationResult(
                experiment_name="no_deconstruct",
                input_sentence=sentence,
                final_mtl_formula=None,
                success=False,
                processing_time=processing_time,
                token_usage=TokenUsage(),
                additional_info={"error": str(e)}
            )
    
    def run_no_verify_experiment(self, sentence: str) -> AblationResult:
        """消融实验2：移除验证阶段，只执行解构和合成"""
        start_time = time.time()
        logger.info("=== 消融实验：移除验证阶段 ===")
        
        try:
            # 重置token统计
            self.dsv_framework.total_token_usage = TokenUsage()
            
            # 阶段一：解构
            deconstruct_result = self.dsv_framework._stage_1_deconstruct(sentence)
            if not deconstruct_result.success:
                raise Exception("解构阶段失败")
            
            # 阶段二：合成
            synthesize_result = self.dsv_framework._stage_2_synthesize(deconstruct_result.stage_output)
            if not synthesize_result.success:
                raise Exception("合成阶段失败")
            
            processing_time = time.time() - start_time
            
            return AblationResult(
                experiment_name="no_verify",
                input_sentence=sentence,
                final_mtl_formula=synthesize_result.stage_output.mtl_formula,
                success=True,
                processing_time=processing_time,
                token_usage=self.dsv_framework.total_token_usage,
                additional_info={
                    "method": "deconstruct_synthesize_only",
                    "semantic_sketch": {
                        "atomic_propositions_count": len(deconstruct_result.stage_output.atomic_propositions),
                        "temporal_relations_count": len(deconstruct_result.stage_output.temporal_relations),
                        "metric_constraints_count": len(deconstruct_result.stage_output.metric_constraints),
                        "global_property": deconstruct_result.stage_output.global_property
                    },
                    "synthesis_reasoning": synthesize_result.stage_output.synthesis_reasoning[:300] + "..." if len(synthesize_result.stage_output.synthesis_reasoning) > 300 else synthesize_result.stage_output.synthesis_reasoning
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"无验证实验失败: {e}")
            
            return AblationResult(
                experiment_name="no_verify",
                input_sentence=sentence,
                final_mtl_formula=None,
                success=False,
                processing_time=processing_time,
                token_usage=self.dsv_framework.total_token_usage,
                additional_info={"error": str(e)}
            )
    
    def run_no_refinement_experiment(self, sentence: str) -> AblationResult:
        """消融实验3：移除修正循环，验证失败后直接报告失败"""
        start_time = time.time()
        logger.info("=== 消融实验：移除修正循环 ===")
        
        try:
            # 使用DSV框架但禁用修正
            result = self.dsv_framework.process(sentence, enable_refinement=False)
            
            processing_time = time.time() - start_time
            
            # 提取验证信息
            verification_info = {}
            for stage_result in result.stage_results:
                if stage_result.stage.value == "verify" and stage_result.stage_output:
                    verification_info = {
                        "similarity_score": stage_result.stage_output.similarity_score,
                        "verification_passed": stage_result.stage_output.verification_passed,
                        "back_translation": stage_result.stage_output.back_translation[:200] + "..." if len(stage_result.stage_output.back_translation) > 200 else stage_result.stage_output.back_translation
                    }
                    break
            
            return AblationResult(
                experiment_name="no_refinement",
                input_sentence=sentence,
                final_mtl_formula=result.final_mtl_formula,
                success=result.success,
                processing_time=processing_time,
                token_usage=result.total_token_usage,
                additional_info={
                    "method": "single_iteration_only",
                    "termination_reason": result.termination_reason,
                    "verification_info": verification_info,
                    "refinement_iterations": result.refinement_iterations
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"无修正实验失败: {e}")
            
            return AblationResult(
                experiment_name="no_refinement",
                input_sentence=sentence,
                final_mtl_formula=None,
                success=False,
                processing_time=processing_time,
                token_usage=TokenUsage(),
                additional_info={"error": str(e)}
            )
    
    def run_full_dsv_experiment(self, sentence: str) -> AblationResult:
        """运行完整的DSV框架作为对照组"""
        start_time = time.time()
        logger.info("=== 完整DSV框架实验 ===")
        
        try:
            result = self.dsv_framework.process(sentence, enable_refinement=True)
            
            processing_time = time.time() - start_time
            
            # 提取详细信息
            stage_info = {}
            for stage_result in result.stage_results:
                stage_name = stage_result.stage.value
                if stage_name not in stage_info:
                    stage_info[stage_name] = []
                
                stage_info[stage_name].append({
                    "success": stage_result.success,
                    "processing_time": stage_result.processing_time,
                    "token_usage": stage_result.token_usage.total_tokens
                })
            
            return AblationResult(
                experiment_name="full_dsv",
                input_sentence=sentence,
                final_mtl_formula=result.final_mtl_formula,
                success=result.success,
                processing_time=processing_time,
                token_usage=result.total_token_usage,
                additional_info={
                    "method": "complete_dsv_framework",
                    "termination_reason": result.termination_reason,
                    "refinement_iterations": result.refinement_iterations,
                    "stage_info": stage_info
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"完整DSV实验失败: {e}")
            
            return AblationResult(
                experiment_name="full_dsv",
                input_sentence=sentence,
                final_mtl_formula=None,
                success=False,
                processing_time=processing_time,
                token_usage=TokenUsage(),
                additional_info={"error": str(e)}
            )
    
    def run_single_sentence_ablation(self, sentence: str) -> AblationComparison:
        """对单个句子运行所有消融实验"""
        logger.info(f"开始单句消融实验: {sentence}")
        
        # 运行完整DSV框架
        full_dsv_result = self.run_full_dsv_experiment(sentence)
        
        # 运行消融实验
        ablation_results = {}
        
        if self.experiments_enabled.get("no_deconstruct", {}).get("enabled", False):
            ablation_results["no_deconstruct"] = self.run_no_deconstruct_experiment(sentence)
        
        if self.experiments_enabled.get("no_verify", {}).get("enabled", False):
            ablation_results["no_verify"] = self.run_no_verify_experiment(sentence)
        
        if self.experiments_enabled.get("no_refinement", {}).get("enabled", False):
            ablation_results["no_refinement"] = self.run_no_refinement_experiment(sentence)
        
        # 基线对比（可以添加其他基线方法）
        baseline_results = {
            "end2end_baseline": self.run_no_deconstruct_experiment(sentence)  # 使用端到端作为基线
        }
        
        return AblationComparison(
            input_sentence=sentence,
            full_dsv_result=full_dsv_result,
            ablation_results=ablation_results,
            baseline_results=baseline_results
        )
    
    def run_batch_ablation(self, sentences: List[str], output_file: str) -> List[AblationComparison]:
        """批量运行消融实验"""
        logger.info(f"开始批量消融实验，共 {len(sentences)} 个句子")
        
        results = []
        
        for i, sentence in enumerate(sentences, 1):
            logger.info(f"处理第 {i}/{len(sentences)} 个句子")
            
            try:
                comparison = self.run_single_sentence_ablation(sentence)
                results.append(comparison)
                
                # 每5个样本保存一次中间结果
                if i % 5 == 0 or i == len(sentences):
                    self.save_ablation_results(results, output_file)
                    logger.info(f"已保存 {len(results)} 个结果到 {output_file}")
                    
            except Exception as e:
                logger.error(f"句子 {i} 处理失败: {e}")
                continue
        
        logger.info(f"批量消融实验完成！共处理 {len(results)} 个句子")
        return results
    
    def save_ablation_results(self, results: List[AblationComparison], output_file: str):
        """保存消融实验结果"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 构建保存数据
        save_data = {
            "experiment_info": {
                "framework": "DSV Ablation Study",
                "total_sentences": len(results),
                "experiments_conducted": list(self.experiments_enabled.keys()),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "config": self.ablation_config,
            "results": []
        }
        
        # 处理每个对比结果
        for comparison in results:
            result_data = {
                "input_sentence": comparison.input_sentence,
                "full_dsv": asdict(comparison.full_dsv_result),
                "ablation_experiments": {
                    name: asdict(result) for name, result in comparison.ablation_results.items()
                },
                "baseline_comparisons": {
                    name: asdict(result) for name, result in comparison.baseline_results.items()
                }
            }
            save_data["results"].append(result_data)
        
        # 保存JSON结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        # 同时保存Excel摘要
        excel_file = output_file.replace('.json', '_summary.xlsx')
        self.save_ablation_summary(results, excel_file)
        
        logger.info(f"消融实验结果已保存到: {output_file} 和 {excel_file}")
    
    def save_ablation_summary(self, results: List[AblationComparison], excel_file: str):
        """保存消融实验摘要到Excel"""
        summary_data = []
        
        for comparison in results:
            base_data = {
                "Input_Sentence": comparison.input_sentence,
                "Full_DSV_Success": comparison.full_dsv_result.success,
                "Full_DSV_Formula": comparison.full_dsv_result.final_mtl_formula,
                "Full_DSV_Time": comparison.full_dsv_result.processing_time,
                "Full_DSV_Tokens": comparison.full_dsv_result.token_usage.total_tokens
            }
            
            # 添加消融实验结果
            for exp_name, result in comparison.ablation_results.items():
                base_data[f"{exp_name}_Success"] = result.success
                base_data[f"{exp_name}_Formula"] = result.final_mtl_formula
                base_data[f"{exp_name}_Time"] = result.processing_time
                base_data[f"{exp_name}_Tokens"] = result.token_usage.total_tokens
            
            # 添加基线结果
            for baseline_name, result in comparison.baseline_results.items():
                base_data[f"{baseline_name}_Success"] = result.success
                base_data[f"{baseline_name}_Formula"] = result.final_mtl_formula
                base_data[f"{baseline_name}_Time"] = result.processing_time
                base_data[f"{baseline_name}_Tokens"] = result.token_usage.total_tokens
            
            summary_data.append(base_data)
        
        # 保存为Excel
        df = pd.DataFrame(summary_data)
        df.to_excel(excel_file, index=False)
    
    def analyze_ablation_results(self, results: List[AblationComparison]) -> Dict[str, Any]:
        """分析消融实验结果"""
        analysis = {
            "total_sentences": len(results),
            "success_rates": {},
            "average_processing_times": {},
            "average_token_usage": {},
            "formula_consistency": {}
        }
        
        # 收集所有实验的结果
        all_experiments = {}
        for comparison in results:
            all_experiments["full_dsv"] = all_experiments.get("full_dsv", []) + [comparison.full_dsv_result]
            
            for exp_name, result in comparison.ablation_results.items():
                all_experiments[exp_name] = all_experiments.get(exp_name, []) + [result]
            
            for baseline_name, result in comparison.baseline_results.items():
                all_experiments[baseline_name] = all_experiments.get(baseline_name, []) + [result]
        
        # 计算统计指标
        for exp_name, exp_results in all_experiments.items():
            # 成功率
            success_count = sum(1 for r in exp_results if r.success)
            analysis["success_rates"][exp_name] = success_count / len(exp_results) if exp_results else 0
            
            # 平均处理时间
            avg_time = sum(r.processing_time for r in exp_results) / len(exp_results) if exp_results else 0
            analysis["average_processing_times"][exp_name] = avg_time
            
            # 平均token使用
            avg_tokens = sum(r.token_usage.total_tokens for r in exp_results) / len(exp_results) if exp_results else 0
            analysis["average_token_usage"][exp_name] = avg_tokens
        
        return analysis

def main():
    """主函数演示"""
    print("=== DSV Ablation Study Demo ===\n")
    
    # 创建消融实验
    ablation_study = DSVAblationStudy()
    
    # 测试句子
    test_sentences = [
        "在传感器A检测到故障后的5到10秒内，警报B必须响起，并持续至少20秒。",
        "After receiving the signal, the system must respond within 10 seconds.",
        "The door should remain locked for at least 30 seconds after the alarm is triggered."
    ]
    
    # 运行批量消融实验
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"data/output/dsv/ablation_results_{timestamp}.json"
    
    results = ablation_study.run_batch_ablation(test_sentences, output_file)
    
    # 分析结果
    analysis = ablation_study.analyze_ablation_results(results)
    
    print("=== 消融实验分析结果 ===")
    print(f"总句子数: {analysis['total_sentences']}")
    print("\n成功率对比:")
    for exp_name, success_rate in analysis["success_rates"].items():
        print(f"  {exp_name}: {success_rate:.2%}")
    
    print("\n平均处理时间对比:")
    for exp_name, avg_time in analysis["average_processing_times"].items():
        print(f"  {exp_name}: {avg_time:.2f}秒")
    
    print("\n平均Token使用对比:")
    for exp_name, avg_tokens in analysis["average_token_usage"].items():
        print(f"  {exp_name}: {avg_tokens:.0f} tokens")

if __name__ == "__main__":
    main()
