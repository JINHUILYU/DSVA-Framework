"""
增强的循环修复DSV框架
在修复循环中传递上下文信息，实现更智能的迭代修正
"""

import json
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from dotenv import load_dotenv
import logging
import os

# 导入原有的DSV框架组件
from dsv_framework import (
    DSVStage, TokenUsage, SemanticSpecificationSketch, 
    SynthesisResult, VerificationResult, DSVStageResult, DSVProcessResult
)
from example_retrieval import ExampleRetriever, Example

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RefinementContext:
    """修正上下文信息"""
    iteration: int
    previous_results: List[Dict[str, Any]]
    failure_reasons: List[str]
    similarity_scores: List[float]
    best_result: Optional[Dict[str, Any]]
    refinement_strategy: str

class EnhancedRefinementDSV:
    """增强循环修复的DSV框架"""
    
    def __init__(self, config_path: str = "config/dsv_config.json"):
        """初始化增强修复DSV框架"""
        self.config = self._load_config(config_path)
        self.clients = self._initialize_clients()
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.total_token_usage = TokenUsage()
        
        # 初始化示例检索器
        self.example_retriever = ExampleRetriever(config_path)
        
        # 获取配置参数
        self.similarity_threshold = self.config.get("similarity_threshold", 0.12)
        self.max_refinement_iterations = self.config.get("max_refinement_iterations", 3)
        
        # 示例增强配置
        self.example_config = self.config.get("example_retrieval", {})
        self.examples_enabled = self.example_config.get("enabled", True)
        
        logger.info("增强修复DSV框架初始化完成")
        logger.info(f"示例增强启用状态: {self.examples_enabled}")
        logger.info(f"相似度阈值: {self.similarity_threshold}")
        logger.info(f"最大修正迭代次数: {self.max_refinement_iterations}")
    
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
                    "name": "Analyst_Agent",
                    "model": "deepseek-chat",
                    "temperature": 0.3,
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url_env": "DEEPSEEK_API_URL"
                },
                "synthesizer": {
                    "name": "Synthesizer_Agent",
                    "model": "deepseek-chat",
                    "temperature": 0.1,
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url_env": "DEEPSEEK_API_URL"
                },
                "verifier": {
                    "name": "Verifier_Agent",
                    "model": "deepseek-chat",
                    "temperature": 0.2,
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url_env": "DEEPSEEK_API_URL"
                }
            },
            "similarity_threshold": 0.9,
            "max_refinement_iterations": 3,
            "example_retrieval": {
                "enabled": True,
                "top_k": 3,
                "similarity_threshold": 0.3
            }
        }
    
    def _initialize_clients(self) -> Dict[str, OpenAI]:
        """初始化API客户端"""
        load_dotenv()
        clients = {}
        
        for agent_type, agent_config in self.config["agents"].items():
            api_key = os.getenv(agent_config["api_key_env"])
            base_url = os.getenv(agent_config["base_url_env"])
            
            if api_key:
                clients[agent_type] = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
                logger.info(f"{agent_type} 客户端初始化成功")
            else:
                logger.warning(f"API密钥未找到: {agent_config['api_key_env']}")
                
        return clients
    
    def _call_llm(self, agent_type: str, messages: List[Dict]) -> Tuple[str, TokenUsage]:
        """调用LLM并追踪token使用"""
        if agent_type not in self.clients:
            raise ValueError(f"客户端未找到: {agent_type}")
            
        client = self.clients[agent_type]
        agent_config = self.config["agents"][agent_type]
        
        try:
            response = client.chat.completions.create(
                model=agent_config["model"],
                messages=messages,  # type: ignore
                temperature=agent_config["temperature"]
            )
            
            # 追踪token使用
            token_usage = TokenUsage()
            if hasattr(response, 'usage') and response.usage:
                token_usage.prompt_tokens = response.usage.prompt_tokens
                token_usage.completion_tokens = response.usage.completion_tokens
                token_usage.total_tokens = response.usage.total_tokens
            
            # 累计到总使用量
            self.total_token_usage.prompt_tokens += token_usage.prompt_tokens
            self.total_token_usage.completion_tokens += token_usage.completion_tokens
            self.total_token_usage.total_tokens += token_usage.total_tokens
            
            content = response.choices[0].message.content
            return content.strip() if content else "", token_usage
            
        except Exception as e:
            logger.error(f"LLM调用失败 {agent_type}: {e}")
            return "", TokenUsage()
    
    def _build_refinement_context_prompt(self, context: RefinementContext) -> str:
        """构建包含修正上下文的prompt"""
        if context.iteration == 0:
            return ""
        
        context_prompt = f"""
## 修正上下文信息

这是第 {context.iteration + 1} 次处理尝试。以下是之前尝试的情况：

### 历史处理结果：
"""
        
        for i, prev_result in enumerate(context.previous_results):
            context_prompt += f"""
**第 {i + 1} 次尝试：**
- 相似度分数: {context.similarity_scores[i]:.3f}
- 失败原因: {context.failure_reasons[i]}
- MTL公式: {prev_result.get('mtl_formula', 'N/A')}
- 验证翻译: {prev_result.get('back_translation', 'N/A')[:100]}...
"""
        
        if context.best_result:
            context_prompt += f"""
### 当前最佳结果：
- 相似度分数: {context.best_result['similarity_score']:.3f}
- MTL公式: {context.best_result['mtl_formula']}
- 来自第 {context.best_result['iteration']} 次尝试
"""
        
        context_prompt += f"""
### 修正策略建议：
{context.refinement_strategy}

**请根据上述历史信息和失败原因，调整你的处理策略，避免重复之前的错误。**
"""
        
        return context_prompt
    
    def _determine_refinement_strategy(self, context: RefinementContext) -> str:
        """根据历史结果确定修正策略"""
        if not context.previous_results:
            return "首次处理，使用标准策略"
        
        avg_similarity = sum(context.similarity_scores) / len(context.similarity_scores)
        
        strategies = []
        
        if avg_similarity < 0.3:
            strategies.append("语义理解可能存在根本性问题，需要重新审视原子命题的识别")
        elif avg_similarity < 0.6:
            strategies.append("时序关系或度量约束可能不准确，需要更仔细地分析时间逻辑")
        else:
            strategies.append("整体理解基本正确，可能需要微调MTL算子的使用或时间窗口的设置")
        
        # 分析常见失败模式
        failure_patterns = {}
        for reason in context.failure_reasons:
            if "解构" in reason:
                failure_patterns["deconstruct"] = failure_patterns.get("deconstruct", 0) + 1
            elif "合成" in reason:
                failure_patterns["synthesize"] = failure_patterns.get("synthesize", 0) + 1
            elif "验证" in reason:
                failure_patterns["verify"] = failure_patterns.get("verify", 0) + 1
        
        if failure_patterns.get("deconstruct", 0) > 1:
            strategies.append("解构阶段反复失败，建议简化语义组件的提取，专注于核心时序关系")
        
        if failure_patterns.get("synthesize", 0) > 1:
            strategies.append("合成阶段反复失败，建议检查MTL算子的映射规则，确保语法正确性")
        
        return "; ".join(strategies)
    
    def _stage_1_deconstruct_with_context(self, sentence: str, context: RefinementContext) -> DSVStageResult:
        """带上下文的解构阶段"""
        start_time = time.time()
        logger.info(f"=== DSV阶段一：语义解构（第{context.iteration + 1}次尝试） ===")
        
        # 获取相似示例
        examples_text = self._get_examples_for_stage(sentence, "deconstruct")
        
        # 构建包含上下文的prompt
        context_prompt = self._build_refinement_context_prompt(context)
        
        # 构建增强版分析师Agent的prompt
        analyst_prompt = f"""
你是一个专业的语义分析师Agent，负责将自然语言句子解构为构成MTL公式所需的核心语义组件。

{context_prompt}

{examples_text}

请分析以下句子并提取结构化信息：

句子: "{sentence}"

请按照以下JSON格式输出语义规约草图：

```json
{{
    "atomic_propositions": [
        {{
            "id": "ap_1",
            "description": "描述系统状态或事件的基本单元",
            "variable": "对应的变量名"
        }}
    ],
    "temporal_relations": [
        {{
            "type": "关系类型（如after, before, during等）",
            "antecedent": "前件命题ID",
            "consequent": "后件命题ID",
            "description": "时序关系描述"
        }}
    ],
    "metric_constraints": [
        {{
            "applies_to": "约束应用的对象",
            "type": "约束类型（如window, duration, delay等）",
            "value": "具体的时间值或范围",
            "description": "约束描述"
        }}
    ],
    "global_property": "全局属性（如Always, Eventually, Never等）"
}}
```

重要要求：
1. 仔细识别句子中的原子命题（基本状态或事件）
2. 明确时序关系（事件之间的先后顺序）
3. 提取精确的度量约束（时间窗口、持续时间等）
4. 确定全局属性的适用范围
5. 输出必须是有效的JSON格式
6. 变量名使用简洁的英文标识符
7. 参考上述示例和历史上下文，避免重复之前的错误

请开始分析：
"""
        
        messages = [
            {"role": "system", "content": "你是一个专业的语义分析师，擅长将自然语言解构为结构化的语义组件。请根据历史上下文调整策略，避免重复错误。"},
            {"role": "user", "content": analyst_prompt}
        ]
        
        try:
            response, token_usage = self._call_llm("analyst", messages)
            
            # 提取JSON内容
            sketch = self._extract_semantic_sketch(response)
            
            processing_time = time.time() - start_time
            
            return DSVStageResult(
                stage=DSVStage.DECONSTRUCT,
                success=sketch.extraction_success,
                processing_time=processing_time,
                token_usage=token_usage,
                stage_output=sketch,
                agent_response=response,
                error_message=None if sketch.extraction_success else "JSON提取失败"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"解构阶段失败: {e}")
            return DSVStageResult(
                stage=DSVStage.DECONSTRUCT,
                success=False,
                processing_time=processing_time,
                token_usage=TokenUsage(),
                stage_output=None,
                agent_response="",
                error_message=str(e)
            )
    
    def _get_examples_for_stage(self, sentence: str, stage: str) -> str:
        """为指定阶段获取相似示例"""
        if not self.examples_enabled:
            return ""
        
        try:
            retrieval_result = self.example_retriever.retrieve_examples(sentence, stage)
            if retrieval_result.examples:
                formatted_examples = self.example_retriever.format_examples_for_prompt(retrieval_result)
                logger.info(f"为阶段 {stage} 检索到 {len(retrieval_result.examples)} 个示例")
                return formatted_examples
            else:
                logger.info(f"阶段 {stage} 未找到相似示例")
                return ""
        except Exception as e:
            logger.error(f"示例检索失败 {stage}: {e}")
            return ""
    
    def _extract_semantic_sketch(self, response: str) -> SemanticSpecificationSketch:
        """从回答中提取语义规约草图"""
        logger.info(f"开始解析LLM响应，响应长度: {len(response)}")
        logger.debug(f"LLM响应内容: {response[:500]}...")  # 只显示前500字符用于调试
        
        try:
            # 多种JSON提取策略
            json_str = None
            
            # 策略1: 提取```json```包围的内容
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1).strip()
                logger.info("使用策略1提取JSON: ```json```")
            
            # 策略2: 提取```包围的内容
            if not json_str:
                json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                    logger.info("使用策略2提取JSON: ```")
            
            # 策略3: 查找JSON对象结构
            if not json_str:
                json_match = re.search(r'\{[\s\S]*"atomic_propositions"[\s\S]*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0).strip()
                    logger.info("使用策略3提取JSON: 查找JSON对象结构")
            
            # 策略4: 查找任何看起来像JSON的内容
            if not json_str:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0).strip()
                    logger.info("使用策略4提取JSON: 查找任何JSON结构")
            
            if json_str:
                logger.info(f"提取到JSON字符串，长度: {len(json_str)}")
                logger.debug(f"JSON内容: {json_str}")
                
                # 尝试解析JSON
                sketch_data = json.loads(json_str)
                logger.info("JSON解析成功")
                
                return SemanticSpecificationSketch(
                    atomic_propositions=sketch_data.get("atomic_propositions", []),
                    temporal_relations=sketch_data.get("temporal_relations", []),
                    metric_constraints=sketch_data.get("metric_constraints", []),
                    global_property=sketch_data.get("global_property", "Always"),
                    raw_json=json_str,
                    extraction_success=True
                )
            else:
                logger.error("未找到JSON格式的语义规约草图")
                logger.error(f"完整响应内容: {response}")
                return SemanticSpecificationSketch(
                    atomic_propositions=[],
                    temporal_relations=[],
                    metric_constraints=[],
                    global_property="Always",
                    raw_json="",
                    extraction_success=False
                )
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error(f"尝试解析的JSON: {json_str}")
            return SemanticSpecificationSketch(
                atomic_propositions=[],
                temporal_relations=[],
                metric_constraints=[],
                global_property="Always",
                raw_json="",
                extraction_success=False
            )
        except Exception as e:
            logger.error(f"语义规约草图提取异常: {e}")
            return SemanticSpecificationSketch(
                atomic_propositions=[],
                temporal_relations=[],
                metric_constraints=[],
                global_property="Always",
                raw_json="",
                extraction_success=False
            )
    
    def _stage_2_synthesize_with_context(self, sketch: SemanticSpecificationSketch, context: RefinementContext) -> DSVStageResult:
        """带上下文的合成阶段"""
        start_time = time.time()
        logger.info(f"=== DSV阶段二：约束下的语法合成（第{context.iteration + 1}次尝试） ===")
        
        # 获取相似示例
        examples_text = self._get_examples_for_stage("", "synthesize")  # 合成阶段不需要原句子
        
        # 构建包含上下文的prompt
        context_prompt = self._build_refinement_context_prompt(context)
        
        # 构建合成师Agent的prompt
        synthesizer_prompt = f"""
你是一个专业的MTL公式合成师Agent，负责根据结构化的语义规约草图合成语法正确的MTL公式。

{context_prompt}

{examples_text}

你收到的语义规约草图如下：

```json
{sketch.raw_json}
```

请严格按照以下约束进行合成：

1. **只能使用草图中提供的组件**：
   - 原子命题：{[ap.get('variable', ap.get('id', '')) for ap in sketch.atomic_propositions]}
   - 时序关系：{[rel.get('type', '') for rel in sketch.temporal_relations]}
   - 度量约束：{[const.get('type', '') + ':' + str(const.get('value', '')) for const in sketch.metric_constraints]}
   - 全局属性：{sketch.global_property}

2. **MTL算子映射规则**：
   - Always → G (Globally)
   - Eventually → F (Finally)
   - Next → X
   - Until → U
   - 时间窗口使用下标表示，如 F_[5,10], G_[0,20]

3. **合成步骤**：
   - 将全局属性映射到相应的MTL算子
   - 根据时序关系构建逻辑结构（如 P → F_[t1,t2] Q）
   - 整合度量约束到时间算子中
   - 确保语法完全正确

4. **根据历史上下文调整策略**：
   - 参考之前的失败原因和修正建议
   - 避免重复相同的错误
   - 优化MTL算子的选择和时间窗口设置

请提供：
1. 详细的合成推理过程
2. 最终的MTL公式

格式要求：
```
推理过程：
[详细说明每一步的映射和合成逻辑]

最终MTL公式：
[完整的MTL公式]
```
"""
        
        messages = [
            {"role": "system", "content": "你是一个专业的MTL公式合成师，严格按照给定的语义组件构建MTL公式。请根据历史上下文调整策略，避免重复错误。"},
            {"role": "user", "content": synthesizer_prompt}
        ]
        
        try:
            response, token_usage = self._call_llm("synthesizer", messages)
            
            # 提取合成结果
            synthesis_result = self._extract_synthesis_result(response)
            
            processing_time = time.time() - start_time
            
            return DSVStageResult(
                stage=DSVStage.SYNTHESIZE,
                success=synthesis_result.synthesis_success,
                processing_time=processing_time,
                token_usage=token_usage,
                stage_output=synthesis_result,
                agent_response=response,
                error_message=None if synthesis_result.synthesis_success else "MTL公式提取失败"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"合成阶段失败: {e}")
            return DSVStageResult(
                stage=DSVStage.SYNTHESIZE,
                success=False,
                processing_time=processing_time,
                token_usage=TokenUsage(),
                stage_output=None,
                agent_response="",
                error_message=str(e)
            )
    
    def _extract_synthesis_result(self, response: str) -> SynthesisResult:
        """从回答中提取合成结果"""
        try:
            # 提取推理过程
            reasoning_match = re.search(r'推理过程：\s*(.*?)(?=最终MTL公式：|$)', response, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
            # 提取MTL公式
            formula_match = re.search(r'最终MTL公式：\s*(.*?)(?:\n|$)', response, re.DOTALL)
            if not formula_match:
                # 尝试其他格式
                formula_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            
            if formula_match:
                formula = formula_match.group(1).strip()
                # 清理公式中的多余字符
                formula = re.sub(r'\s+', ' ', formula)
                
                return SynthesisResult(
                    mtl_formula=formula,
                    synthesis_reasoning=reasoning,
                    synthesis_success=True
                )
            else:
                logger.error("未找到MTL公式")
                return SynthesisResult(
                    mtl_formula="",
                    synthesis_reasoning=reasoning,
                    synthesis_success=False
                )
                
        except Exception as e:
            logger.error(f"合成结果提取失败: {e}")
            return SynthesisResult(
                mtl_formula="",
                synthesis_reasoning="",
                synthesis_success=False
            )
    
    def _stage_3_verify_with_context(self, original_sentence: str, mtl_formula: str, context: RefinementContext) -> DSVStageResult:
        """带上下文的验证阶段"""
        start_time = time.time()
        logger.info(f"=== DSV阶段三：循环验证与修正（第{context.iteration + 1}次尝试） ===")
        
        # 获取相似示例
        examples_text = self._get_examples_for_stage("", "verify")  # 验证阶段不需要原句子
        
        # 构建包含上下文的prompt
        context_prompt = self._build_refinement_context_prompt(context)
        
        # 构建验证师Agent的prompt（注意：验证师没有看过原始输入）
        verifier_prompt = f"""
你是一个专业的MTL公式验证师Agent，负责将MTL公式翻译回自然语言以验证其语义正确性。

{context_prompt}

{examples_text}

请将以下MTL公式翻译成清晰的自然语言描述：

MTL公式: {mtl_formula}

翻译指导：
1. G(φ): 表示"总是"或"在所有时间"
2. F_[a,b](φ): 表示"在a到b时间单位内最终"
3. →: 表示"如果...那么..."或"after...then..."
4. ∧: 表示"并且"或"and"
5. sensor_A_fault: 传感器A检测到故障
6. alarm_B_sounding: 警报B响起

要求：
1. 使用简洁、直接的表达方式
2. 保持时间约束的准确性
3. 避免冗长的技术解释
4. 翻译结果应该接近日常英语表达习惯
5. 根据历史上下文调整翻译策略，确保语义准确性

请直接给出自然语言翻译，格式如下：

自然语言翻译：
[简洁清晰的自然语言描述]
"""
        
        messages = [
            {"role": "system", "content": "你是一个专业的MTL公式验证师，擅长将形式化公式翻译成自然语言。请根据历史上下文调整策略，确保翻译准确性。"},
            {"role": "user", "content": verifier_prompt}
        ]
        
        try:
            response, token_usage = self._call_llm("verifier", messages)
            
            # 提取验证结果
            verification_result = self._extract_verification_result(response, original_sentence)
            
            processing_time = time.time() - start_time
            
            return DSVStageResult(
                stage=DSVStage.VERIFY,
                success=True,  # 验证阶段总是成功，但结果可能不通过
                processing_time=processing_time,
                token_usage=token_usage,
                stage_output=verification_result,
                agent_response=response,
                error_message=None
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"验证阶段失败: {e}")
            return DSVStageResult(
                stage=DSVStage.VERIFY,
                success=False,
                processing_time=processing_time,
                token_usage=TokenUsage(),
                stage_output=None,
                agent_response="",
                error_message=str(e)
            )
    
    def _extract_verification_result(self, response: str, original_sentence: str) -> VerificationResult:
        """从回答中提取验证结果并计算相似度"""
        try:
            # 提取自然语言翻译
            translation_match = re.search(r'自然语言翻译：\s*(.*?)(?:\n|$)', response, re.DOTALL)
            
            if not translation_match:
                # 如果没有找到标准格式，尝试提取整个回答作为翻译
                # 去掉可能的前缀文字，只保留翻译内容
                lines = response.strip().split('\n')
                # 找到最后一个非空行作为翻译结果
                for line in reversed(lines):
                    if line.strip() and not any(prefix in line for prefix in ['翻译', '推理', '要求', '格式']):
                        translation_match = re.match(r'.*', line.strip())
                        break
                
                # 如果还是没找到，使用整个回答
                if not translation_match:
                    translation_match = re.match(r'.*', response.strip())
            
            if translation_match:
                back_translation = translation_match.group(0).strip()
                
                # 清理翻译结果，去掉可能的格式标记
                back_translation = re.sub(r'^自然语言翻译：\s*', '', back_translation)
                back_translation = back_translation.strip()
                
                logger.info(f"提取的验证翻译: {back_translation}")
                
                # 计算语义相似度
                similarity_score = self._calculate_semantic_similarity(original_sentence, back_translation)
                
                # 判断验证是否通过
                verification_passed = similarity_score >= self.similarity_threshold
                
                logger.info(f"相似度分数: {similarity_score:.4f}, 阈值: {self.similarity_threshold}, 通过: {verification_passed}")
                
                return VerificationResult(
                    back_translation=back_translation,
                    similarity_score=similarity_score,
                    verification_passed=verification_passed,
                    verification_reasoning=""
                )
            else:
                logger.error("未找到自然语言翻译")
                logger.error(f"完整响应: {response}")
                return VerificationResult(
                    back_translation="",
                    similarity_score=0.0,
                    verification_passed=False,
                    verification_reasoning=""
                )
                
        except Exception as e:
            logger.error(f"验证结果提取失败: {e}")
            return VerificationResult(
                back_translation="",
                similarity_score=0.0,
                verification_passed=False,
                verification_reasoning=""
            )
    
    def _calculate_semantic_similarity(self, sentence1: str, sentence2: str) -> float:
        """计算两个句子的语义相似度"""
        try:
            # 编码两个句子
            embeddings = self.sentence_model.encode([sentence1, sentence2], convert_to_tensor=True)
            
            # 计算余弦相似度
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])[0][0]
            return float(similarity.cpu().numpy())
            
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0
    
    def process_with_enhanced_refinement(self, sentence: str, enable_refinement: bool = True) -> DSVProcessResult:
        """带增强修正的完整DSV处理流程"""
        start_time = time.time()
        logger.info(f"开始增强修正DSV处理: {sentence}")
        
        # 重置token统计
        self.total_token_usage = TokenUsage()
        
        stage_results = []
        refinement_iterations = 0
        final_mtl_formula = None
        success = False
        termination_reason = "未知"
        
        # 修正上下文
        refinement_context = RefinementContext(
            iteration=0,
            previous_results=[],
            failure_reasons=[],
            similarity_scores=[],
            best_result=None,
            refinement_strategy=""
        )
        
        try:
            # 主处理循环
            for iteration in range(self.max_refinement_iterations + 1):
                logger.info(f"=== 增强修正DSV处理迭代 {iteration + 1} ===")
                
                # 更新修正上下文
                refinement_context.iteration = iteration
                refinement_context.refinement_strategy = self._determine_refinement_strategy(refinement_context)
                
                # 阶段一：带上下文的解构
                deconstruct_result = self._stage_1_deconstruct_with_context(sentence, refinement_context)
                stage_results.append(deconstruct_result)
                
                if not deconstruct_result.success:
                    failure_reason = "解构阶段失败"
                    refinement_context.failure_reasons.append(failure_reason)
                    refinement_context.similarity_scores.append(0.0)
                    termination_reason = failure_reason
                    break
                
                # 阶段二：带上下文的合成
                synthesize_result = self._stage_2_synthesize_with_context(deconstruct_result.stage_output, refinement_context)
                stage_results.append(synthesize_result)
                
                if not synthesize_result.success:
                    failure_reason = "合成阶段失败"
                    refinement_context.failure_reasons.append(failure_reason)
                    refinement_context.similarity_scores.append(0.0)
                    termination_reason = failure_reason
                    break
                
                # 阶段三：带上下文的验证
                verify_result = self._stage_3_verify_with_context(sentence, synthesize_result.stage_output.mtl_formula, refinement_context)
                stage_results.append(verify_result)
                
                if not verify_result.success:
                    failure_reason = "验证阶段失败"
                    refinement_context.failure_reasons.append(failure_reason)
                    refinement_context.similarity_scores.append(0.0)
                    termination_reason = failure_reason
                    break
                
                # 检查验证结果
                similarity_score = verify_result.stage_output.similarity_score
                refinement_context.similarity_scores.append(similarity_score)
                
                if verify_result.stage_output.verification_passed:
                    # 验证通过，处理成功
                    final_mtl_formula = synthesize_result.stage_output.mtl_formula
                    success = True
                    termination_reason = f"验证通过（相似度: {similarity_score:.3f}）"
                    break
                else:
                    # 验证失败，记录结果并准备下次迭代
                    failure_reason = f"验证失败（相似度: {similarity_score:.3f}）"
                    refinement_context.failure_reasons.append(failure_reason)
                    
                    # 更新最佳结果
                    current_result = {
                        'mtl_formula': synthesize_result.stage_output.mtl_formula,
                        'back_translation': verify_result.stage_output.back_translation,
                        'similarity_score': similarity_score,
                        'iteration': iteration + 1
                    }
                    refinement_context.previous_results.append(current_result)
                    
                    if not refinement_context.best_result or similarity_score > refinement_context.best_result['similarity_score']:
                        refinement_context.best_result = current_result
                    
                    refinement_iterations += 1
                    logger.info(f"验证失败（相似度: {similarity_score:.3f}），低于阈值 {self.similarity_threshold}")
                    
                    if not enable_refinement or iteration >= self.max_refinement_iterations:
                        # 不启用修正或达到最大迭代次数
                        final_mtl_formula = refinement_context.best_result['mtl_formula'] if refinement_context.best_result else synthesize_result.stage_output.mtl_formula
                        termination_reason = f"达到最大修正次数（最佳相似度: {refinement_context.best_result['similarity_score'] if refinement_context.best_result else similarity_score:.3f}）"
                        break
                    else:
                        # 继续修正循环
                        logger.info(f"开始第 {refinement_iterations} 次修正")
                        continue
            
            total_processing_time = time.time() - start_time
            
            return DSVProcessResult(
                input_sentence=sentence,
                final_mtl_formula=final_mtl_formula,
                total_processing_time=total_processing_time,
                total_token_usage=self.total_token_usage,
                stage_results=stage_results,
                refinement_iterations=refinement_iterations,
                success=success,
                termination_reason=termination_reason
            )
            
        except Exception as e:
            total_processing_time = time.time() - start_time
            logger.error(f"增强修正DSV处理失败: {e}")
            
            return DSVProcessResult(
                input_sentence=sentence,
                final_mtl_formula=None,
                total_processing_time=total_processing_time,
                total_token_usage=self.total_token_usage,
                stage_results=stage_results,
                refinement_iterations=refinement_iterations,
                success=False,
                termination_reason=f"系统错误: {str(e)}"
            )

def main():
    """演示增强修正功能"""
    print("=== 增强修正DSV框架演示 ===\n")
    
    enhanced_dsv = EnhancedRefinementDSV()
    
    test_sentence = "Within 5 to 10 seconds after sensor A detects a fault, alarm B must sound for at least 20 seconds."
    
    print(f"测试句子: {test_sentence}")
    print("-" * 80)
    
    result = enhanced_dsv.process_with_enhanced_refinement(test_sentence, enable_refinement=True)
    
    print(f"✅ 处理成功: {result.success}")
    print(f"🎯 最终MTL公式: {result.final_mtl_formula}")
    print(f"🔄 修正迭代次数: {result.refinement_iterations}")
    print(f"📝 终止原因: {result.termination_reason}")
    print(f"⏱️  总处理时间: {result.total_processing_time:.2f}秒")

if __name__ == "__main__":
    main()
