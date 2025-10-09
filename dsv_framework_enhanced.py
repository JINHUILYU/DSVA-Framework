
"""
Enhanced DSV Framework with Example-Based Prompts
增强版DSV框架：为每个agent的prompt中基于相似度插入top-5个示例

核心增强：
- 集成示例检索系统
- 为每个阶段的agent提供相似示例
- 动态构建包含示例的prompt
- 保持原有DSV框架的所有功能
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

class EnhancedDSVFramework:
    """增强版DSV框架主类"""
    
    def __init__(self, config_path: str = "config/dsv_config.json"):
        """初始化增强版DSV框架"""
        self.config = self._load_config(config_path)
        self.clients = self._initialize_clients()
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.total_token_usage = TokenUsage()
        
        # 初始化示例检索器
        self.example_retriever = ExampleRetriever(config_path)
        
        # 获取配置参数
        self.similarity_threshold = self.config.get("similarity_threshold", 0.5)
        self.max_refinement_iterations = self.config.get("max_refinement_iterations", 3)
        
        # 示例增强配置
        self.example_config = self.config.get("example_retrieval", {})
        self.examples_enabled = self.example_config.get("enabled", True)
        
        logger.info("增强版DSV框架初始化完成")
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
                "top_k": 5,
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
    
    def _stage_1_deconstruct(self, sentence: str) -> DSVStageResult:
        """阶段一：语义解构与组件提取（增强版）"""
        start_time = time.time()
        logger.info("=== DSV阶段一：语义解构与组件提取（增强版） ===")
        
        # 获取相似示例
        examples_text = self._get_examples_for_stage(sentence, "deconstruct")
        
        # 构建增强版分析师Agent的prompt
        analyst_prompt = f"""
你是一个专业的语义分析师Agent，负责将自然语言句子解构为构成MTL公式所需的核心语义组件。

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
7. 参考上述示例的处理方式，但要根据当前输入的具体情况进行分析

请开始分析：
"""
        
        messages = [
            {"role": "system", "content": "你是一个专业的语义分析师，擅长将自然语言解构为结构化的语义组件。请参考提供的示例，但要根据具体输入进行独立分析。"},
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
    
    def _extract_semantic_sketch(self, response: str) -> SemanticSpecificationSketch:
        """从回答中提取语义规约草图"""
        try:
            # 尝试提取JSON内容
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
                sketch_data = json.loads(json_str)
                
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
            return SemanticSpecificationSketch(
                atomic_propositions=[],
                temporal_relations=[],
                metric_constraints=[],
                global_property="Always",
                raw_json="",
                extraction_success=False
            )
    
    def _stage_2_synthesize(self, sketch: SemanticSpecificationSketch, original_sentence: str) -> DSVStageResult:
        """阶段二：约束下的语法合成（增强版）"""
        start_time = time.time()
        logger.info("=== DSV阶段二：约束下的语法合成（增强版） ===")
        
        # 获取相似示例
        examples_text = self._get_examples_for_stage(original_sentence, "synthesize")
        
        # 构建增强版合成师Agent的prompt
        synthesizer_prompt = f"""
你是一个专业的MTL公式合成师Agent，负责根据结构化的语义规约草图合成语法正确的MTL公式。

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

请参考上述示例的处理方式，但要根据当前语义规约草图的具体内容进行合成。
"""
        
        messages = [
            {"role": "system", "content": "你是一个专业的MTL公式合成师，严格按照给定的语义组件构建MTL公式。请参考提供的示例，但要根据具体的语义规约草图进行独立合成。"},
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
    
    def _stage_3_verify(self, original_sentence: str, mtl_formula: str) -> DSVStageResult:
        """阶段三：循环验证与修正（增强版）"""
        start_time = time.time()
        logger.info("=== DSV阶段三：循环验证与修正（增强版） ===")
        
        # 获取相似示例（基于MTL公式）
        examples_text = self._get_examples_for_stage(mtl_formula, "verify")
        
        # 构建增强版验证师Agent的prompt
        verifier_prompt = f"""
你是一个专业的MTL公式验证师Agent，负责将MTL公式翻译回自然语言以验证其语义正确性。

{examples_text}

请将以下MTL公式翻译成清晰的自然语言描述：

MTL公式: {mtl_formula}

要求：
1. 不要使用技术术语，用通俗易懂的自然语言表达
2. 准确描述时间约束和逻辑关系
3. 保持语义的完整性和准确性
4. 提供详细的翻译推理过程

请按以下格式回答：

```
翻译推理过程：
[详细说明MTL公式各部分的含义和翻译逻辑]

自然语言翻译：
[完整的自然语言描述]
```

请参考上述示例的处理方式，但要根据当前MTL公式的具体内容进行翻译。
"""
        
        messages = [
            {"role": "system", "content": "你是一个专业的MTL公式验证师，擅长将形式化公式翻译成自然语言。请参考提供的示例，但要根据具体的MTL公式进行独立翻译。"},
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
            # 提取翻译推理过程
            reasoning_match = re.search(r'翻译推理过程：\s*(.*?)(?=自然语言翻译：|$)', response, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
            # 提取自然语言翻译
            translation_match = re.search(r'自然语言翻译：\s*(.*?)(?:\n|$)', response, re.DOTALL)
            if not translation_match:
                # 尝试其他格式
                lines = response.split('\n')
                for line in lines:
                    if line.strip() and not line.startswith('翻译推理过程'):
                        translation_match = re.match(r'.*', line)
                        break
            
            if translation_match:
                back_translation = translation_match.group(0).strip()
                
                # 计算语义相似度
                similarity_score = self._calculate_semantic_similarity(original_sentence, back_translation)
                
                # 判断验证是否通过
                verification_passed = similarity_score >= self.similarity_threshold
                
                return VerificationResult(
                    back_translation=back_translation,
                    similarity_score=similarity_score,
                    verification_passed=verification_passed,
                    verification_reasoning=reasoning
                )
            else:
                logger.error("未找到自然语言翻译")
                return VerificationResult(
                    back_translation="",
                    similarity_score=0.0,
                    verification_passed=False,
                    verification_reasoning=reasoning
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
    
    def process(self, sentence: str, enable_refinement: bool = True) -> DSVProcessResult:
        """完整的增强版DSV处理流程"""
        start_time = time.time()
        logger.info(f"开始增强版DSV处理: {sentence}")
        
        # 重置token统计
        self.total_token_usage = TokenUsage()
        
        stage_results = []
        refinement_iterations = 0
        final_mtl_formula = None
        success = False
        termination_reason = "未知"
        
        # 用于跟踪最佳结果
        best_result = None
        best_similarity = -1.0
        
        try:
            # 主处理循环
            for iteration in range(self.max_refinement_iterations + 1):
                logger.info(f"=== 增强版DSV处理迭代 {iteration + 1} ===")
                
                # 阶段一：解构（增强版）
                deconstruct_result = self._stage_1_deconstruct(sentence)
                stage_results.append(deconstruct_result)
                
                if not deconstruct_result.success:
                    termination_reason = "解构阶段失败"
                    break
                
                # 阶段二：合成（增强版）
                synthesize_result = self._stage_2_synthesize(deconstruct_result.stage_output, sentence)
                stage_results.append(synthesize_result)
                
                if not synthesize_result.success:
                    termination_reason = "合成阶段失败"
                    break
                
                # 阶段三：验证（增强版）
                verify_result = self._stage_3_verify(sentence, synthesize_result.stage_output.mtl_formula)
                stage_results.append(verify_result)
                
                if not verify_result.success:
                    termination_reason = "验证阶段失败"
                    break
                
                # 更新最佳结果（基于相似度）
                current_similarity = verify_result.stage_output.similarity_score
                if current_similarity > best_similarity:
                    best_similarity = current_similarity
                    best_result = {
                        'mtl_formula': synthesize_result.stage_output.mtl_formula,
                        'similarity_score': current_similarity,
                        'iteration': iteration + 1
                    }
                
                # 检查验证结果
                if verify_result.stage_output.verification_passed:
                    # 验证通过，处理成功
                    final_mtl_formula = synthesize_result.stage_output.mtl_formula
                    success = True
                    termination_reason = f"验证通过（相似度: {verify_result.stage_output.similarity_score:.3f}）"
                    break
                else:
                    # 验证失败
                    refinement_iterations += 1
                    logger.info(f"验证失败（相似度: {verify_result.stage_output.similarity_score:.3f}），"
                              f"低于阈值 {self.similarity_threshold}")
                    
                    if not enable_refinement or iteration >= self.max_refinement_iterations:
                        # 不启用修正或达到最大迭代次数，使用最佳结果
                        if best_result:
                            final_mtl_formula = best_result['mtl_formula']
                            termination_reason = f"达到最大修正次数，使用最佳结果（相似度: {best_result['similarity_score']:.3f}，来自第{best_result['iteration']}次迭代）"
                        else:
                            final_mtl_formula = synthesize_result.stage_output.mtl_formula
                            termination_reason = f"达到最大修正次数（相似度: {verify_result.stage_output.similarity_score:.3f}）"
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
            logger.error(f"增强版DSV处理失败: {e}")
            
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
    
    def save_result(self, result: DSVProcessResult, output_file: str):
        """保存增强版DSV处理结果"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 构建保存数据
        save_data = {
            "framework": "Enhanced DSV (Deconstruct, Synthesize, and Verify) with Examples",
            "input_sentence": result.input_sentence,
            "final_mtl_formula": result.final_mtl_formula,
            "success": result.success,
            "termination_reason": result.termination_reason,
            "total_processing_time": result.total_processing_time,
            "total_token_usage": asdict(result.total_token_usage),
            "refinement_iterations": result.refinement_iterations,
            "stage_results": [],
            "config": {
                "similarity_threshold": self.similarity_threshold,
                "max_refinement_iterations": self.max_refinement_iterations,
                "examples_enabled": self.examples_enabled,
                "example_config": self.example_config,
                "agents": self.config["agents"]
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 处理阶段结果
        for stage_result in result.stage_results:
            stage_data = {
                "stage": stage_result.stage.value,
                "success": stage_result.success,
                "processing_time": stage_result.processing_time,
                "token_usage": asdict(stage_result.token_usage),
                "agent_response": stage_result.agent_response,
                "error_message": stage_result.error_message
            }
            
            # 添加阶段特定的输出
            if stage_result.stage_output:
                if isinstance(stage_result.stage_output, SemanticSpecificationSketch):
                    stage_data["semantic_sketch"] = {
                        "atomic_propositions": stage_result.stage_output.atomic_propositions,
                        "temporal_relations": stage_result.stage_output.temporal_relations,
                        "metric_constraints": stage_result.stage_output.metric_constraints,
                        "global_property": stage_result.stage_output.global_property,
                        "extraction_success": stage_result.stage_output.extraction_success
                    }
                elif isinstance(stage_result.stage_output, SynthesisResult):
                    stage_data["synthesis_result"] = {
                        "mtl_formula": stage_result.stage_output.mtl_formula,
                        "synthesis_reasoning": stage_result.stage_output.synthesis_reasoning,
                        "synthesis_success": stage_result.stage_output.synthesis_success
                    }
                elif isinstance(stage_result.stage_output, VerificationResult):
                    stage_data["verification_result"] = {
                        "back_translation": stage_result.stage_output.back_translation,
                        "similarity_score": stage_result.stage_output.similarity_score,
                        "verification_passed": stage_result.stage_output.verification_passed,
                        "verification_reasoning": stage_result.stage_output.verification_reasoning
                    }
            
            save_data["stage_results"].append(stage_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"增强版DSV结果已保存到: {output_file}")

def main():
    """主函数演示"""
    print("=== Enhanced DSV Framework Demo ===\n")
    
    # 创建增强版DSV处理器
    enhanced_dsv = EnhancedDSVFramework()
    
    # 测试句子
    test_sentences = [
        "在传感器A检测到故障后的5到10秒内，警报B必须响起，并持续至少20秒。",
        "After receiving the signal, the system must respond within 10 seconds.",
        "The door should remain locked for at least 30 seconds after the alarm is triggered."
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"=== 测试句子 {i} ===")
        print(f"输入: {sentence}")
        print("-" * 60)
        
        try:
            # 执行增强版DSV处理
            result = enhanced_dsv.process(sentence, enable_refinement=True)
            
            # 显示结果
            print(f"✅ 处理成功: {result.success}")
            print(f"🎯 最终MTL公式: {result.final_mtl_formula}")
            print(f"🔄 修正迭代次数: {result.refinement_iterations}")
            print(f"📝 终止原因: {result.termination_reason}")
            print(f"⏱️  总处理时间: {result.total_processing_time:.2f}秒")
            print(f"🔢 总Token使用: {result.total_token_usage.total_tokens}")
            print(f"🔍 示例增强: {enhanced_dsv.examples_enabled}")
            
            # 显示各阶段结果摘要
            print(f"\n📊 阶段结果摘要:")
            for stage_result in result.stage_results:
                status = "✅" if stage_result.success else "❌"
                print(f"   {status} {stage_result.stage.value}: {stage_result.processing_time:.2f}s, {stage_result.token_usage.total_tokens} tokens")
            
            # 保存结果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"data/output/dsv/enhanced_result_{i}_{timestamp}.json"
            enhanced_dsv.save_result(result, output_file)
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
