"""
DSV框架的单大模型直接生成基线 (Single LLM Baseline)
用作对比实验的基线方法，展示传统端到端生成与DSV框架的差异

特点：
- 单一大模型直接生成MTL公式
- 无结构化分解
- 无验证循环
- 简单直接的prompt工程
"""

import json
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    """Token使用统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class SingleBaselineResult:
    """单模型基线结果"""
    input_sentence: str
    final_mtl_formula: Optional[str]
    agent_response: str
    processing_time: float
    token_usage: TokenUsage
    success: bool
    extraction_method: str
    confidence_score: Optional[float] = None

class DSVSingleBaseline:
    """DSV框架的单大模型基线"""
    
    def __init__(self, config_path: str = "config/dsv_config.json"):
        """初始化单模型基线"""
        self.config = self._load_config(config_path)
        self.client = self._initialize_client()
        self.agent_config = self.config["agents"]["analyst"]  # 使用analyst配置
        
        logger.info("DSV单模型基线初始化完成")
        logger.info(f"使用模型: {self.agent_config['model']}")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 未找到，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "agents": {
                "analyst": {
                    "name": "Single_Baseline_Agent",
                    "model": "deepseek-chat",
                    "temperature": 0.3,
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url_env": "DEEPSEEK_API_URL"
                }
            }
        }
    
    def _initialize_client(self) -> OpenAI:
        """初始化API客户端"""
        load_dotenv()
        
        api_key = os.getenv(self.agent_config["api_key_env"])
        base_url = os.getenv(self.agent_config["base_url_env"])
        
        if not api_key:
            raise ValueError(f"API密钥未找到: {self.agent_config['api_key_env']}")
            
        return OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def _call_llm(self, messages: List[Dict]) -> Tuple[str, TokenUsage]:
        """调用LLM并追踪token使用"""
        try:
            response = self.client.chat.completions.create(
                model=self.agent_config["model"],
                messages=messages,  # type: ignore
                temperature=self.agent_config["temperature"]
            )
            
            # 追踪token使用
            token_usage = TokenUsage()
            if hasattr(response, 'usage') and response.usage:
                token_usage.prompt_tokens = response.usage.prompt_tokens
                token_usage.completion_tokens = response.usage.completion_tokens
                token_usage.total_tokens = response.usage.total_tokens
            
            content = response.choices[0].message.content
            return content.strip() if content else "", token_usage
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return "", TokenUsage()
    
    def _create_single_prompt(self, sentence: str) -> str:
        """创建单模型直接生成的prompt"""
        prompt = f"""
你是一个专业的MTL（Metric Temporal Logic）专家，请将以下自然语言句子直接转换为MTL公式。

输入句子: "{sentence}"

请按照以下要求进行转换：

1. **MTL语法规则**：
   - 使用标准MTL算子：G (Globally), F (Finally), X (Next), U (Until)
   - 时间约束使用下标表示：G_[a,b], F_[a,b], U_[a,b]
   - 逻辑算子：∧ (AND), ∨ (OR), ¬ (NOT), → (IMPLIES)

2. **转换步骤**：
   - 识别句子中的原子命题（基本状态或事件）
   - 确定时序关系和时间约束
   - 构建完整的MTL公式

3. **输出格式**：
   - 提供简要的分析过程
   - 给出最终的MTL公式
   - 评估转换的置信度（0-1之间）

请按以下格式回答：

```
分析过程：
[简要说明识别的原子命题、时序关系和时间约束]

MTL公式：
[完整的MTL公式]

置信度：
[0-1之间的数值，表示对转换结果的信心]
```

注意：
- 确保MTL公式语法正确
- 时间单位与原句保持一致
- 如果句子存在歧义，选择最合理的解释
"""
        return prompt
    
    def _extract_mtl_result(self, response: str) -> Tuple[Optional[str], Optional[float], str]:
        """从回答中提取MTL公式、置信度和分析过程"""
        try:
            # 提取分析过程
            analysis_match = re.search(r'分析过程：\s*(.*?)(?=MTL公式：|$)', response, re.DOTALL)
            analysis = analysis_match.group(1).strip() if analysis_match else ""
            
            # 提取MTL公式
            formula_match = re.search(r'MTL公式：\s*(.*?)(?=置信度：|$)', response, re.DOTALL)
            if not formula_match:
                # 尝试其他格式
                formula_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            
            formula = None
            if formula_match:
                formula = formula_match.group(1).strip()
                # 清理公式中的多余字符
                formula = re.sub(r'\s+', ' ', formula)
                formula = formula.replace('\n', ' ').strip()
            
            # 提取置信度
            confidence_match = re.search(r'置信度：\s*([\d.]+)', response)
            confidence = None
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))  # 确保在0-1范围内
                except ValueError:
                    confidence = None
            
            # 如果没有找到公式，尝试从整个回答中提取包含MTL算子的行
            if not formula:
                lines = response.split('\n')
                for line in lines:
                    if any(op in line for op in ['G(', 'F_[', 'P_[', 'U_[', 'X(', 'G[', 'F[', 'U[', 'P[']):
                        formula = line.strip()
                        break
            
            extraction_method = "structured" if formula_match else "pattern_matching" if formula else "failed"
            
            return formula, confidence, extraction_method
            
        except Exception as e:
            logger.error(f"MTL结果提取失败: {e}")
            return None, None, "error"
    
    def process_single(self, sentence: str) -> SingleBaselineResult:
        """处理单个句子"""
        start_time = time.time()
        logger.info(f"单模型基线处理: {sentence}")
        
        try:
            # 创建prompt
            prompt = self._create_single_prompt(sentence)
            
            # 构建消息
            messages = [
                {
                    "role": "system", 
                    "content": "你是一个专业的MTL专家，擅长将自然语言直接转换为MTL公式。请严格按照要求的格式回答。"
                },
                {"role": "user", "content": prompt}
            ]
            
            # 调用LLM
            response, token_usage = self._call_llm(messages)
            
            # 提取结果
            mtl_formula, confidence, extraction_method = self._extract_mtl_result(response)
            
            processing_time = time.time() - start_time
            success = mtl_formula is not None and mtl_formula != ""
            
            logger.info(f"单模型基线完成: 成功={success}, 用时={processing_time:.2f}s")
            
            return SingleBaselineResult(
                input_sentence=sentence,
                final_mtl_formula=mtl_formula,
                agent_response=response,
                processing_time=processing_time,
                token_usage=token_usage,
                success=success,
                extraction_method=extraction_method,
                confidence_score=confidence
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"单模型基线处理失败: {e}")
            
            return SingleBaselineResult(
                input_sentence=sentence,
                final_mtl_formula=None,
                agent_response=f"处理失败: {str(e)}",
                processing_time=processing_time,
                token_usage=TokenUsage(),
                success=False,
                extraction_method="error"
            )
    
    def process_batch(self, sentences: List[str], output_file: str) -> List[SingleBaselineResult]:
        """批量处理句子"""
        logger.info(f"单模型基线批量处理 {len(sentences)} 个句子")
        
        results = []
        total_tokens = 0
        total_time = 0
        success_count = 0
        
        for i, sentence in enumerate(sentences, 1):
            logger.info(f"处理第 {i}/{len(sentences)} 个句子")
            
            result = self.process_single(sentence)
            results.append(result)
            
            total_tokens += result.token_usage.total_tokens
            total_time += result.processing_time
            
            if result.success:
                success_count += 1
            
            # 每5个样本保存一次中间结果
            if i % 5 == 0 or i == len(sentences):
                self.save_batch_results(results, output_file)
                logger.info(f"已保存 {len(results)} 个结果到 {output_file}")
        
        logger.info(f"单模型基线批量处理完成！成功率: {success_count/len(sentences):.1%}")
        return results
    
    def save_result(self, result: SingleBaselineResult, output_file: str):
        """保存单个结果"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            "method": "DSV_Single_Baseline",
            "input_sentence": result.input_sentence,
            "final_mtl_formula": result.final_mtl_formula,
            "agent_response": result.agent_response,
            "processing_time": result.processing_time,
            "token_usage": asdict(result.token_usage),
            "success": result.success,
            "extraction_method": result.extraction_method,
            "confidence_score": result.confidence_score,
            "agent_config": {
                "name": self.agent_config["name"],
                "model": self.agent_config["model"],
                "temperature": self.agent_config["temperature"]
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"单模型基线结果已保存到: {output_file}")
    
    def save_batch_results(self, results: List[SingleBaselineResult], output_file: str):
        """保存批量结果"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 计算统计信息
        total_tokens = sum(r.token_usage.total_tokens for r in results)
        total_time = sum(r.processing_time for r in results)
        success_count = sum(1 for r in results if r.success)
        avg_confidence = sum(r.confidence_score for r in results if r.confidence_score) / len([r for r in results if r.confidence_score]) if any(r.confidence_score for r in results) else None
        
        # 构建保存数据
        save_data = {
            "method": "DSV_Single_Baseline",
            "summary": {
                "total_sentences": len(results),
                "success_count": success_count,
                "success_rate": success_count / len(results) if results else 0,
                "total_tokens": total_tokens,
                "total_time": total_time,
                "average_time_per_sentence": total_time / len(results) if results else 0,
                "average_confidence": avg_confidence,
                "agent_config": {
                    "name": self.agent_config["name"],
                    "model": self.agent_config["model"],
                    "temperature": self.agent_config["temperature"]
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": []
        }
        
        # 添加详细结果
        for result in results:
            save_data["results"].append({
                "input_sentence": result.input_sentence,
                "final_mtl_formula": result.final_mtl_formula,
                "agent_response": result.agent_response,
                "processing_time": result.processing_time,
                "token_usage": asdict(result.token_usage),
                "success": result.success,
                "extraction_method": result.extraction_method,
                "confidence_score": result.confidence_score
            })
        
        # 保存JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        # 同时保存Excel摘要
        excel_file = output_file.replace('.json', '_summary.xlsx')
        self.save_excel_summary(results, excel_file)
        
        logger.info(f"单模型基线批量结果已保存到: {output_file} 和 {excel_file}")
    
    def save_excel_summary(self, results: List[SingleBaselineResult], excel_file: str):
        """保存Excel摘要"""
        try:
            import pandas as pd
            
            summary_data = []
            for i, result in enumerate(results, 1):
                summary_data.append({
                    "ID": i,
                    "Input_Sentence": result.input_sentence,
                    "Success": result.success,
                    "MTL_Formula": result.final_mtl_formula,
                    "Processing_Time": result.processing_time,
                    "Token_Usage": result.token_usage.total_tokens,
                    "Confidence_Score": result.confidence_score,
                    "Extraction_Method": result.extraction_method
                })
            
            df = pd.DataFrame(summary_data)
            df.to_excel(excel_file, index=False)
            
        except ImportError:
            logger.warning("pandas未安装，跳过Excel文件生成")
        except Exception as e:
            logger.error(f"Excel文件生成失败: {e}")

def main():
    """主函数演示"""
    print("=== DSV Single Baseline Demo ===\n")
    
    # 创建单模型基线
    baseline = DSVSingleBaseline()
    
    # 测试句子
    test_sentences = [
        "在传感器A检测到故障后的5到10秒内，警报B必须响起，并持续至少20秒。",
        "After receiving the signal, the system must respond within 10 seconds.",
        "The door should remain locked for at least 30 seconds after the alarm is triggered.",
        "If the temperature exceeds 80 degrees, the cooling system must activate within 5 seconds.",
        "The backup generator should start within 30 seconds if the main power fails."
    ]
    
    # 单句处理演示
    print("=== 单句处理演示 ===")
    demo_sentence = test_sentences[0]
    print(f"输入: {demo_sentence}")
    print("-" * 60)
    
    result = baseline.process_single(demo_sentence)
    
    print(f"✅ 处理成功: {result.success}")
    print(f"🎯 MTL公式: {result.final_mtl_formula}")
    print(f"📊 置信度: {result.confidence_score}")
    print(f"⏱️  处理时间: {result.processing_time:.2f}秒")
    print(f"🔢 Token使用: {result.token_usage.total_tokens}")
    print(f"🔍 提取方法: {result.extraction_method}")
    
    # 保存单句结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    single_output = f"data/output/dsv/baseline_single_{timestamp}.json"
    baseline.save_result(result, single_output)
    
    # 批量处理演示
    print(f"\n=== 批量处理演示 ===")
    print(f"批量处理 {len(test_sentences)} 个句子...")
    
    batch_output = f"data/output/dsv/baseline_batch_{timestamp}.json"
    batch_results = baseline.process_batch(test_sentences, batch_output)
    
    # 显示批量统计
    success_count = sum(1 for r in batch_results if r.success)
    total_time = sum(r.processing_time for r in batch_results)
    total_tokens = sum(r.token_usage.total_tokens for r in batch_results)
    avg_confidence = sum(r.confidence_score for r in batch_results if r.confidence_score) / len([r for r in batch_results if r.confidence_score]) if any(r.confidence_score for r in batch_results) else 0
    
    print(f"\n📊 批量处理统计:")
    print(f"   📝 总句子数: {len(batch_results)}")
    print(f"   ✅ 成功处理: {success_count}")
    print(f"   📈 成功率: {success_count/len(batch_results):.1%}")
    print(f"   ⏱️  总处理时间: {total_time:.2f}秒")
    print(f"   ⚡ 平均每句: {total_time/len(batch_results):.2f}秒")
    print(f"   🔢 总Token使用: {total_tokens}")
    print(f"   📊 平均置信度: {avg_confidence:.2f}")
    
    print(f"\n🌟 单模型基线特点:")
    print(f"   🎯 直接生成: 单次调用直接生成MTL公式")
    print(f"   ⚡ 高效快速: 无多阶段处理开销")
    print(f"   📊 置信度评估: 模型自评转换质量")
    print(f"   🔍 多种提取策略: 结构化和模式匹配")
    print(f"   📈 适合对比: 作为DSV框架的基线方法")

if __name__ == "__main__":
    main()
