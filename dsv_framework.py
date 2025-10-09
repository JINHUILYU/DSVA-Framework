"""
DSV (Deconstruct, Synthesize, and Verify) Framework for NL2MTL
解构-合成-验证框架：一个白盒化、结构驱动的自然语言到MTL转换方法

核心理念：通过结构化分解与约束生成来保证精确性
- 阶段一：语义解构与组件提取 (Deconstruct)
- 阶段二：约束下的语法合成 (Synthesize) 
- 阶段三：循环验证与修正 (Verify)
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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DSVStage(Enum):
    """DSV框架阶段枚举"""
    DECONSTRUCT = "deconstruct"
    SYNTHESIZE = "synthesize"
    VERIFY = "verify"

@dataclass
class TokenUsage:
    """Token使用统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class SemanticSpecificationSketch:
    """语义规约草图"""
    atomic_propositions: List[Dict[str, str]]
    temporal_relations: List[Dict[str, str]]
    metric_constraints: List[Dict[str, str]]
    global_property: str
    raw_json: str
    extraction_success: bool = True

@dataclass
class SynthesisResult:
    """合成结果"""
    mtl_formula: str
    synthesis_reasoning: str
    synthesis_success: bool = True

@dataclass
class VerificationResult:
    """验证结果"""
    back_translation: str
    similarity_score: float
    verification_passed: bool
    verification_reasoning: str

@dataclass
class DSVStageResult:
    """DSV阶段结果"""
    stage: DSVStage
    success: bool
    processing_time: float
    token_usage: TokenUsage
    stage_output: Any  # 可以是SemanticSpecificationSketch, SynthesisResult, 或 VerificationResult
    agent_response: str
    error_message: Optional[str] = None

@dataclass
class DSVProcessResult:
    """DSV处理完整结果"""
    input_sentence: str
    final_mtl_formula: Optional[str]
    total_processing_time: float
    total_token_usage: TokenUsage
    stage_results: List[DSVStageResult]
    refinement_iterations: int
    success: bool
    termination_reason: str

class DSVFramework:
    """DSV框架主类"""
    
    def __init__(self, config_path: str = "config/dsv_config.json"):
        """初始化DSV框架"""
        self.config = self._load_config(config_path)
        self.clients = self._initialize_clients()
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.total_token_usage = TokenUsage()
        
        # 获取配置参数
        self.similarity_threshold = self.config.get("similarity_threshold", 0.9)
        self.max_refinement_iterations = self.config.get("max_refinement_iterations", 3)
        
        logger.info("DSV框架初始化完成")
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
            "max_refinement_iterations": 3
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
    
    def _stage_1_deconstruct(self, sentence: str) -> DSVStageResult:
        """阶段一：语义解构与组件提取"""
        start_time = time.time()
        logger.info("=== DSV阶段一：语义解构与组件提取 ===")
        
        # 构建分析师Agent的prompt
        analyst_prompt = f"""
你是一个专业的语义分析师Agent，负责将自然语言句子解构为构成MTL公式所需的核心语义组件。

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

请开始分析：
"""
        
        messages = [
            {"role": "system", "content": "你是一个专业的语义分析师，擅长将自然语言解构为结构化的语义组件。"},
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
    
    def _stage_2_synthesize(self, sketch: SemanticSpecificationSketch) -> DSVStageResult:
        """阶段二：约束下的语法合成"""
        start_time = time.time()
        logger.info("=== DSV阶段二：约束下的语法合成 ===")
        
        # 构建合成师Agent的prompt
        synthesizer_prompt = f"""
你是一个专业的MTL公式合成师Agent，负责根据结构化的语义规约草图合成语法正确的MTL公式。

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
"""
        
        messages = [
            {"role": "system", "content": "你是一个专业的MTL公式合成师，严格按照给定的语义组件构建MTL公式。"},
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
        """阶段三：循环验证与修正"""
        start_time = time.time()
        logger.info("=== DSV阶段三：循环验证与修正 ===")
        
        # 构建验证师Agent的prompt（注意：验证师没有看过原始输入）
        verifier_prompt = f"""
你是一个专业的MTL公式验证师Agent，负责将MTL公式翻译回自然语言以验证其语义正确性。

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
"""
        
        messages = [
            {"role": "system", "content": "你是一个专业的MTL公式验证师，擅长将形式化公式翻译成自然语言。"},
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
        """完整的DSV处理流程"""
        start_time = time.time()
        logger.info(f"开始DSV处理: {sentence}")
        
        # 重置token统计
        self.total_token_usage = TokenUsage()
        
        stage_results = []
        refinement_iterations = 0
        final_mtl_formula = None
        success = False
        termination_reason = "未知"
        
        try:
            # 主处理循环
            for iteration in range(self.max_refinement_iterations + 1):
                logger.info(f"=== DSV处理迭代 {iteration + 1} ===")
                
                # 阶段一：解构
                deconstruct_result = self._stage_1_deconstruct(sentence)
                stage_results.append(deconstruct_result)
                
                if not deconstruct_result.success:
                    termination_reason = "解构阶段失败"
                    break
                
                # 阶段二：合成
                synthesize_result = self._stage_2_synthesize(deconstruct_result.stage_output)
                stage_results.append(synthesize_result)
                
                if not synthesize_result.success:
                    termination_reason = "合成阶段失败"
                    break
                
                # 阶段三：验证
                verify_result = self._stage_3_verify(sentence, synthesize_result.stage_output.mtl_formula)
                stage_results.append(verify_result)
                
                if not verify_result.success:
                    termination_reason = "验证阶段失败"
                    break
                
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
                        # 不启用修正或达到最大迭代次数
                        final_mtl_formula = synthesize_result.stage_output.mtl_formula
                        termination_reason = f"达到最大修正次数（相似度: {verify_result.stage_output.similarity_score:.3f}）"
                        break
                    else:
                        # 继续修正循环
                        logger.info(f"开始第 {refinement_iterations} 次修正")
                        # 这里可以添加基于验证反馈的修正逻辑
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
            logger.error(f"DSV处理失败: {e}")
            
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
        """保存DSV处理结果"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 构建保存数据
        save_data = {
            "framework": "DSV (Deconstruct, Synthesize, and Verify)",
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
        
        logger.info(f"DSV结果已保存到: {output_file}")

def main():
    """主函数演示"""
    print("=== DSV Framework Demo ===\n")
    
    # 创建DSV处理器
    dsv = DSVFramework()
    
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
            # 执行DSV处理
            result = dsv.process(sentence, enable_refinement=True)
            
            # 显示结果
            print(f"✅ 处理成功: {result.success}")
            print(f"🎯 最终MTL公式: {result.final_mtl_formula}")
            print(f"🔄 修正迭代次数: {result.refinement_iterations}")
            print(f"📝 终止原因: {result.termination_reason}")
            print(f"⏱️  总处理时间: {result.total_processing_time:.2f}秒")
            print(f"🔢 总Token使用: {result.total_token_usage.total_tokens}")
            
            # 显示各阶段结果摘要
            print(f"\n📊 阶段结果摘要:")
            for stage_result in result.stage_results:
                status = "✅" if stage_result.success else "❌"
                print(f"   {status} {stage_result.stage.value}: {stage_result.processing_time:.2f}s, {stage_result.token_usage.total_tokens} tokens")
            
            # 保存结果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"data/output/dsv/result_{i}_{timestamp}.json"
            dsv.save_result(result, output_file)
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
