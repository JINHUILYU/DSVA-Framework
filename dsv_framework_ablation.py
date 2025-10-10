"""
DSV Framework - Ablation Study Version
消融实验版本：移除动态增强模块，保持基础DSV框架功能

This is the baseline DSV framework without dynamic example enhancement for ablation studies.
Core DSV functionality remains intact: Deconstruct → Synthesize → Verify with refinement loop.
"""

import json
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import logging
import os

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not installed, skipping .env loading")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DSVStage(Enum):
    """DSV pipeline stages enumeration."""
    DECONSTRUCT = "deconstruct"
    SYNTHESIZE = "synthesize"
    VERIFY = "verify"


@dataclass
class TokenUsage:
    """Track token usage for API calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class SemanticSpecificationSketch:
    """Structured semantic components extracted from natural language."""
    atomic_propositions: List[Dict] = field(default_factory=list)
    temporal_relations: List[Dict] = field(default_factory=list)
    metric_constraints: List[Dict] = field(default_factory=list)
    global_property: str = "Always"
    raw_json: str = ""
    lexicon: Dict[str, str] = field(default_factory=dict)
    extraction_success: bool = False


@dataclass
class SynthesisResult:
    """Result of MTL formula synthesis."""
    mtl_formula: str
    synthesis_reasoning: str
    synthesis_success: bool


@dataclass
class VerificationResult:
    """Result of MTL formula verification."""
    back_translation: str
    similarity_score: float
    verification_passed: bool
    verification_reasoning: str


@dataclass
class DSVStageResult:
    """Result for a single DSV stage."""
    stage: DSVStage
    success: bool
    processing_time: float
    token_usage: TokenUsage
    stage_output: Any  # SemanticSpecificationSketch, SynthesisResult, or VerificationResult
    agent_response: str
    error_message: Optional[str] = None


@dataclass
class DSVProcessResult:
    """Complete DSV pipeline processing result."""
    input_sentence: str
    final_mtl_formula: Optional[str]
    total_processing_time: float
    total_token_usage: TokenUsage
    stage_results: List[DSVStageResult]
    refinement_iterations: int
    success: bool
    termination_reason: str


class DSVFrameworkAblation:
    """
    DSV Framework - Ablation Study Version
    基础DSV框架，不包含动态增强模块，用于消融实验
    """

    def __init__(self, config_path: str = "config/dsv_config.json"):
        """初始化DSV框架"""
        self.config = self._load_config(config_path)
        self.clients = self._initialize_clients()

        # Initialize sentence transformer for similarity calculation
        if SentenceTransformer is not None:
            try:
                self.sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                logger.info("Sentence transformer model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize sentence transformer: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None

        self.total_token_usage = TokenUsage()
        self.similarity_threshold = self.config.get("similarity_threshold", 0.9)
        self.max_refinement_iterations = self.config.get("max_refinement_iterations", 3)

        logger.info("DSV Framework (Ablation Version) initialized")
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
        logger.info(f"Max refinement iterations: {self.max_refinement_iterations}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using default config")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
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

    def _initialize_clients(self) -> Dict[str, Any]:
        """Initialize API clients"""
        load_dotenv()
        clients = {}
        
        for agent_type, agent_config in self.config.get("agents", {}).items():
            api_key = os.getenv(agent_config.get("api_key_env", ""))
            base_url = os.getenv(agent_config.get("base_url_env", ""))
            
            if api_key and OpenAI is not None:
                try:
                    clients[agent_type] = OpenAI(api_key=api_key, base_url=base_url)
                    logger.info(f"{agent_type} client initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize {agent_type} client: {e}")
            else:
                logger.warning(f"API key not found or OpenAI client unavailable for {agent_type}")
        
        return clients

    def _call_llm(self, agent_type: str, messages: List[Dict]) -> Tuple[str, TokenUsage]:
        """Call LLM and track token usage"""
        if agent_type not in self.clients:
            raise ValueError(f"Client not found: {agent_type}")
        
        client = self.clients[agent_type]
        agent_config = self.config["agents"][agent_type]
        
        try:
            response = client.chat.completions.create(
                model=agent_config["model"],
                messages=messages,  # type: ignore
                temperature=agent_config["temperature"]
            )
            
            token_usage = TokenUsage()
            if hasattr(response, "usage") and response.usage:
                token_usage.prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                token_usage.completion_tokens = getattr(response.usage, "completion_tokens", 0)
                token_usage.total_tokens = getattr(response.usage, "total_tokens", 0)

            # Accumulate total usage
            self.total_token_usage.prompt_tokens += token_usage.prompt_tokens
            self.total_token_usage.completion_tokens += token_usage.completion_tokens
            self.total_token_usage.total_tokens += token_usage.total_tokens

            content = response.choices[0].message.content
            return content.strip() if content else "", token_usage
        except Exception as e:
            logger.error(f"LLM call failed for {agent_type}: {e}")
            return "", TokenUsage()

    def _stage_1_deconstruct(self, sentence: str) -> DSVStageResult:
        """Stage 1: Deconstruct natural language into semantic components"""
        start_time = time.time()
        logger.info("=== DSV Stage 1: Deconstruct ===")

        # Basic analyst prompt without examples (ablation version)
        analyst_prompt = f"""
你是一个专业的语义分析师Agent，负责将自然语言句子解构为构成MTL公式所需的核心语义组件。

请分析以下句子并提取结构化信息：

句子: "{sentence}"

请提供一个JSON格式的语义规约草图，包含以下字段：

1. atomic_propositions: 原子命题列表，每个包含id、description和variable
2. temporal_relations: 时序关系列表，描述原子命题之间的时间关系
3. metric_constraints: 度量约束列表，包含时间窗口、持续时间等约束
4. global_property: 全局属性（如"Always"、"Eventually"等）
5. lexicon: 词汇表，将变量名映射到自然语言描述

确保输出是有效的JSON格式。

输出格式示例：
```json
{{
    "atomic_propositions": [
        {{"id": "ap_1", "description": "描述", "variable": "变量名"}}
    ],
    "temporal_relations": [
        {{"type": "关系类型", "antecedent": "前件", "consequent": "后件", "description": "描述"}}
    ],
    "metric_constraints": [
        {{"applies_to": "适用对象", "type": "约束类型", "value": "约束值", "description": "描述"}}
    ],
    "global_property": "Always",
    "lexicon": {{"变量名": "自然语言描述"}}
}}
```
"""

        messages = [
            {"role": "system", "content": "你是一个专业的语义分析师，专门负责将自然语言解构为结构化的语义组件。严格按照要求输出JSON格式。"},
            {"role": "user", "content": analyst_prompt}
        ]

        try:
            response, token_usage = self._call_llm("analyst", messages)
            semantic_sketch = self._extract_semantic_sketch(response)
            processing_time = time.time() - start_time
            
            return DSVStageResult(
                stage=DSVStage.DECONSTRUCT,
                success=semantic_sketch.extraction_success,
                processing_time=processing_time,
                token_usage=token_usage,
                stage_output=semantic_sketch,
                agent_response=response,
                error_message=None if semantic_sketch.extraction_success else "Failed to extract semantic sketch"
            )
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Deconstruct stage failed: {e}")
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
        """Extract semantic components from analyzer response"""
        try:
            # Try to find JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if not json_match:
                # Try to find JSON-like content
                brace_match = re.search(r'\{[\s\S]*\}', response)
                json_str = brace_match.group(0) if brace_match else ""
            else:
                json_str = json_match.group(1).strip()

            if not json_str:
                logger.error("No JSON-formatted semantic sketch found in response")
                return SemanticSpecificationSketch(extraction_success=False)

            # Parse the JSON
            semantic_info = json.loads(json_str)

            # Extract lexicon for verifier
            lexicon = semantic_info.get("lexicon", {})
            if not lexicon:
                # Create lexicon from atomic propositions if not provided
                for ap in semantic_info.get("atomic_propositions", []):
                    var = ap.get("variable", "")
                    desc = ap.get("description", "")
                    if var and desc:
                        lexicon[var] = desc

            return SemanticSpecificationSketch(
                atomic_propositions=semantic_info.get("atomic_propositions", []),
                temporal_relations=semantic_info.get("temporal_relations", []),
                metric_constraints=semantic_info.get("metric_constraints", []),
                global_property=semantic_info.get("global_property", "Always"),
                raw_json=json_str,
                lexicon=lexicon,
                extraction_success=True
            )
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            return SemanticSpecificationSketch(extraction_success=False)
        except Exception as e:
            logger.error(f"Failed to extract semantic sketch: {e}")
            return SemanticSpecificationSketch(extraction_success=False)

    def _stage_2_synthesize(self, sketch: SemanticSpecificationSketch) -> DSVStageResult:
        """Stage 2: Synthesize MTL formula from semantic sketch"""
        start_time = time.time()
        logger.info("=== DSV Stage 2: Synthesize ===")

        # Basic synthesizer prompt without examples (ablation version)
        synthesizer_prompt = f"""
你是一个专业的MTL公式合成师Agent，负责根据结构化的语义规约草图合成语法正确的MTL公式。

你收到的语义规约草图如下：

```json
{sketch.raw_json}
```

请根据此语义规约草图合成一个语法正确的MTL公式。

MTL语法规则：
- G: Globally (总是)
- F: Finally (最终)
- X: Next (下一个)  
- U: Until (直到)
- 时间区间: [a,b] 表示时间窗口
- 逻辑运算: ∧ (and), ∨ (or), ¬ (not), → (implies)

请提供两个标记的部分：

推理过程:
[详细的推理过程，解释如何从语义组件构建MTL公式]

最终MTL公式:
[合成的MTL公式]
"""

        messages = [
            {"role": "system", "content": "你是一个专业的MTL公式合成师。严格根据提供的语义组件构建MTL公式，不要添加额外的解释或推测。"},
            {"role": "user", "content": synthesizer_prompt}
        ]

        try:
            response, token_usage = self._call_llm("synthesizer", messages)
            synthesis_result = self._extract_synthesis_result(response)
            processing_time = time.time() - start_time
            
            return DSVStageResult(
                stage=DSVStage.SYNTHESIZE,
                success=synthesis_result.synthesis_success,
                processing_time=processing_time,
                token_usage=token_usage,
                stage_output=synthesis_result,
                agent_response=response,
                error_message=None if synthesis_result.synthesis_success else "Failed to extract MTL formula"
            )
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Synthesis stage failed: {e}")
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
        """Extract synthesis result from synthesizer response"""
        try:
            # Extract reasoning
            reasoning_match = re.search(r'推理过程[:：]\s*(.*?)(?=最终MTL公式[:：]|$)', response, re.DOTALL | re.IGNORECASE)
            if not reasoning_match:
                reasoning_match = re.search(r'Reasoning[:：]\s*(.*?)(?=Final MTL formula[:：]|最终MTL公式[:：]|$)', response, re.DOTALL | re.IGNORECASE)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            # Extract MTL formula
            formula_match = re.search(r'最终MTL公式[:：]\s*(.*)', response, re.DOTALL | re.IGNORECASE)
            if not formula_match:
                formula_match = re.search(r'Final MTL formula[:：]\s*(.*)', response, re.DOTALL | re.IGNORECASE)
            
            if not formula_match:
                # Try to find formula in code blocks
                code_block = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                formula = code_block.group(1).strip() if code_block else ""
            else:
                formula = formula_match.group(1).strip()

            # Clean up the formula
            formula = re.sub(r'[`\n\r]', '', formula).strip()

            if not formula:
                logger.error("No MTL formula found in synthesizer response")
                return SynthesisResult(
                    mtl_formula="",
                    synthesis_reasoning=reasoning,
                    synthesis_success=False
                )

            return SynthesisResult(
                mtl_formula=formula,
                synthesis_reasoning=reasoning,
                synthesis_success=True
            )
        except Exception as e:
            logger.error(f"Failed to extract synthesis result: {e}")
            return SynthesisResult(
                mtl_formula="",
                synthesis_reasoning="",
                synthesis_success=False
            )

    def _stage_3_verify(self, original_sentence: str, mtl_formula: str, lexicon: Optional[Dict[str, str]] = None) -> DSVStageResult:
        """Stage 3: Verify MTL formula by back-translation"""
        start_time = time.time()
        logger.info("=== DSV Stage 3: Verify ===")

        lexicon = lexicon or {}
        
        # Format lexicon for prompt
        lexicon_text = ""
        if lexicon:
            lexicon_text = "变量词汇表:\n"
            for var, desc in lexicon.items():
                lexicon_text += f"- {var}: {desc}\n"
            lexicon_text += "\n"

        # Basic verifier prompt without examples (ablation version)
        verifier_prompt = f"""
你是一个专业的MTL公式验证师Agent，负责将MTL公式翻译回自然语言进行验证。

{lexicon_text}待验证的MTL公式: {mtl_formula}

请将这个MTL公式翻译成清晰的自然语言描述。

MTL符号含义：
- G: 总是/全局地
- F: 最终/将来某时
- X: 下一个时刻
- U: 直到
- [a,b]: 时间区间a到b
- ∧: 且
- ∨: 或  
- ¬: 非
- →: 蕴含

请提供两个标记的部分：

推理过程:
[详细解释MTL公式的含义和翻译思路]

自然语言翻译:
[将MTL公式翻译成自然语言的结果]
"""

        messages = [
            {"role": "system", "content": "你是一个专业的MTL公式验证师，擅长将形式化公式翻译成自然语言。请确保翻译准确且易于理解。"},
            {"role": "user", "content": verifier_prompt}
        ]

        try:
            response, token_usage = self._call_llm("verifier", messages)
            verification_result = self._extract_verification_result(response, original_sentence)
            processing_time = time.time() - start_time
            
            return DSVStageResult(
                stage=DSVStage.VERIFY,
                success=True,  # Always successful if we get a response
                processing_time=processing_time,
                token_usage=token_usage,
                stage_output=verification_result,
                agent_response=response,
                error_message=None
            )
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Verification stage failed: {e}")
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
        """Extract verification result from verifier response"""
        try:
            # Extract reasoning
            reasoning_match = re.search(r'推理过程[:：]\s*(.*?)(?=自然语言翻译[:：]|$)', response, re.DOTALL | re.IGNORECASE)
            if not reasoning_match:
                reasoning_match = re.search(r'Reasoning[:：]\s*(.*?)(?=Natural language translation[:：]|自然语言翻译[:：]|$)', response, re.DOTALL | re.IGNORECASE)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            # Extract back translation
            translation_match = re.search(r'自然语言翻译[:：]\s*(.*)', response, re.DOTALL | re.IGNORECASE)
            if not translation_match:
                translation_match = re.search(r'Natural language translation[:：]\s*(.*)', response, re.DOTALL | re.IGNORECASE)

            if translation_match:
                back_translation = translation_match.group(1).strip()
            else:
                # Fallback: try to find the translation in the response
                lines = response.split('\n')
                back_translation = ""
                for line in reversed(lines):
                    if line.strip() and not line.strip().startswith(('推理', 'Reasoning', '自然语言', 'Natural')):
                        back_translation = line.strip()
                        break

            if not back_translation:
                logger.error("No natural language translation found in verifier response")
                return VerificationResult(
                    back_translation="",
                    similarity_score=0.0,
                    verification_passed=False,
                    verification_reasoning=reasoning
                )

            # Calculate semantic similarity
            similarity_score = self._calculate_semantic_similarity(original_sentence, back_translation)
            verification_passed = similarity_score >= self.similarity_threshold

            return VerificationResult(
                back_translation=back_translation,
                similarity_score=similarity_score,
                verification_passed=verification_passed,
                verification_reasoning=reasoning
            )
        except Exception as e:
            logger.error(f"Failed to extract verification result: {e}")
            return VerificationResult(
                back_translation="",
                similarity_score=0.0,
                verification_passed=False,
                verification_reasoning=""
            )

    def _calculate_semantic_similarity(self, sentence1: str, sentence2: str) -> float:
        """Calculate semantic similarity between two sentences"""
        try:
            if not self.sentence_model or util is None:
                logger.debug("Sentence transformer not available; returning similarity 0.0")
                return 0.0

            embeddings = self.sentence_model.encode([sentence1, sentence2], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])[0][0]
            return float(similarity.cpu().numpy())
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.0

    def process(self, sentence: str, enable_refinement: bool = True) -> DSVProcessResult:
        """Process a sentence through the complete DSV pipeline"""
        start_time = time.time()
        logger.info(f"Starting DSV processing (Ablation Version): {sentence}")

        # Reset token usage tracking
        self.total_token_usage = TokenUsage()
        
        stage_results = []
        refinement_iterations = 0
        final_mtl_formula = None
        success = False
        termination_reason = "Unknown"

        try:
            for iteration in range(self.max_refinement_iterations + 1):
                logger.info(f"=== DSV Processing Iteration {iteration + 1} ===")
                
                # Stage 1: Deconstruct
                deconstruct_result = self._stage_1_deconstruct(sentence)
                stage_results.append(deconstruct_result)
                if not deconstruct_result.success:
                    termination_reason = "Deconstruct stage failed"
                    break

                # Stage 2: Synthesize
                synth_result = self._stage_2_synthesize(deconstruct_result.stage_output)
                stage_results.append(synth_result)
                if not synth_result.success:
                    termination_reason = "Synthesize stage failed"
                    break

                # Stage 3: Verify
                lexicon = deconstruct_result.stage_output.lexicon if deconstruct_result.stage_output else {}
                verify_result = self._stage_3_verify(sentence, synth_result.stage_output.mtl_formula, lexicon=lexicon)
                stage_results.append(verify_result)
                if not verify_result.success:
                    termination_reason = "Verify stage failed"
                    break

                # Check verification result
                if verify_result.stage_output.verification_passed:
                    final_mtl_formula = synth_result.stage_output.mtl_formula
                    success = True
                    termination_reason = f"Verification passed (similarity: {verify_result.stage_output.similarity_score:.3f})"
                    break
                else:
                    refinement_iterations += 1
                    similarity = verify_result.stage_output.similarity_score
                    logger.info(f"Verification failed (similarity: {similarity:.3f}), below threshold {self.similarity_threshold}")
                    
                    if not enable_refinement or iteration >= self.max_refinement_iterations:
                        final_mtl_formula = synth_result.stage_output.mtl_formula
                        termination_reason = f"Reached max refinement iterations (similarity: {similarity:.3f})"
                        break
                    else:
                        logger.info(f"Starting refinement iteration {refinement_iterations}")
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
            logger.error(f"DSV processing failed: {e}")
            return DSVProcessResult(
                input_sentence=sentence,
                final_mtl_formula=None,
                total_processing_time=total_processing_time,
                total_token_usage=self.total_token_usage,
                stage_results=stage_results,
                refinement_iterations=refinement_iterations,
                success=False,
                termination_reason=str(e)
            )

    def save_result(self, result: DSVProcessResult, output_file: str) -> None:
        """Save DSV processing result to a JSON file"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            "framework": "DSV (Deconstruct, Synthesize, and Verify) - Ablation Version",
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
                "agents": self.config.get("agents", {}),
                "dynamic_enhancement": False  # Ablation version marker
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        for stage_result in result.stage_results:
            stage_data = {
                "stage": stage_result.stage.value,
                "success": stage_result.success,
                "processing_time": stage_result.processing_time,
                "token_usage": asdict(stage_result.token_usage),
                "agent_response": stage_result.agent_response,
                "error_message": stage_result.error_message
            }
            
            if stage_result.stage_output:
                if isinstance(stage_result.stage_output, SemanticSpecificationSketch):
                    stage_data["semantic_sketch"] = {
                        "atomic_propositions": stage_result.stage_output.atomic_propositions,
                        "temporal_relations": stage_result.stage_output.temporal_relations,
                        "metric_constraints": stage_result.stage_output.metric_constraints,
                        "global_property": stage_result.stage_output.global_property,
                        "lexicon": stage_result.stage_output.lexicon,
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
            
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"DSV result (Ablation Version) saved to: {output_file}")


def main() -> None:
    """Run demo of DSV Framework Ablation Version"""
    print("=== DSV Framework Demo - Ablation Version ===\n")
    print("This version does NOT include dynamic example enhancement for ablation studies.\n")
    
    dsv = DSVFrameworkAblation()
    
    test_sentences = [
        "在传感器A检测到故障后的5到10秒内，警报B必须响起，并持续至少20秒。",
        "After receiving the signal, the system must respond within 10 seconds.",
        "The door should remain locked for at least 30 seconds after the alarm is triggered."
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"=== Test Sentence {i} ===")
        print(f"Input: {sentence}")
        print("-" * 60)
        
        try:
            result = dsv.process(sentence, enable_refinement=True)
            
            # Display results
            print(f"✅ Success: {result.success}")
            print(f"🎯 Final MTL formula: {result.final_mtl_formula}")
            print(f"🔄 Refinement iterations: {result.refinement_iterations}")
            print(f"📝 Termination reason: {result.termination_reason}")
            print(f"⏱️  Total processing time: {result.total_processing_time:.2f}s")
            print(f"🔢 Total tokens: {result.total_token_usage.total_tokens}")
            print(f"🚫 Dynamic enhancement: Disabled (Ablation Version)")
            
            # Save results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = Path("data/output/dsv_ablation")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"result_{i}_{timestamp}.json"
            dsv.save_result(result, str(output_file))
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()