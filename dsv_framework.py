"""DSV (Deconstruct, Synthesize, and Verify) Framework for NL2MTL

This module implements a three-stage pipeline that converts natural language
requirements into Metric Temporal Logic (MTL):
  - Deconstruct: extract a structured semantic sketch from text
  - Synthesize: synthesize an MTL formula from the sketch
  - Verify: back-translate the MTL into natural language and compare

The framework records intermediate artifacts, token usage, and supports a
limited refinement loop driven by verifier feedback.
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
    load_dotenv()  # Load environment variables from .env file
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

class DSVFramework:
    """Main DSV framework orchestrating Deconstruct → Synthesize → Verify."""

    def __init__(self, config_path: str = "config/dsv_config.json"):
        self.config = self._load_config(config_path)
        self.clients = self._initialize_clients()

        # Sentence transformer model (optional)
        if SentenceTransformer is not None:
            try:
                self.sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                self.sentence_model = None
        else:
            self.sentence_model = None

        self.total_token_usage = TokenUsage()
        self.similarity_threshold = self.config.get("similarity_threshold", 0.9)
        self.max_refinement_iterations = self.config.get("max_refinement_iterations", 3)

        logger.info("DSV framework initialized")
        logger.info(f"similarity_threshold: {self.similarity_threshold}")
        logger.info(f"max_refinement_iterations: {self.max_refinement_iterations}")

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using default config")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        return {
            "agents": {
                "analyst": {"name": "Analyst_Agent", "model": "deepseek-chat", "temperature": 0.3, "api_key_env": "DEEPSEEK_API_KEY", "base_url_env": "DEEPSEEK_API_URL"},
                "synthesizer": {"name": "Synthesizer_Agent", "model": "deepseek-chat", "temperature": 0.1, "api_key_env": "DEEPSEEK_API_KEY", "base_url_env": "DEEPSEEK_API_URL"},
                "verifier": {"name": "Verifier_Agent", "model": "deepseek-chat", "temperature": 0.2, "api_key_env": "DEEPSEEK_API_KEY", "base_url_env": "DEEPSEEK_API_URL"}
            },
            "similarity_threshold": 0.9,
            "max_refinement_iterations": 3
        }

    def _initialize_clients(self) -> Dict[str, Any]:
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
        if agent_type not in self.clients:
            raise ValueError(f"Client not found: {agent_type}")
        client = self.clients[agent_type]
        agent_config = self.config["agents"][agent_type]
        try:
            response = client.chat.completions.create(model=agent_config["model"], messages=messages, temperature=agent_config["temperature"])  # type: ignore
            token_usage = TokenUsage()
            if hasattr(response, "usage") and response.usage:
                token_usage.prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                token_usage.completion_tokens = getattr(response.usage, "completion_tokens", 0)
                token_usage.total_tokens = getattr(response.usage, "total_tokens", 0)

            # accumulate
            self.total_token_usage.prompt_tokens += token_usage.prompt_tokens
            self.total_token_usage.completion_tokens += token_usage.completion_tokens
            self.total_token_usage.total_tokens += token_usage.total_tokens

            content = response.choices[0].message.content
            return content.strip() if content else "", token_usage
        except Exception as e:
            logger.error(f"LLM call failed for {agent_type}: {e}")
            return "", TokenUsage()

    def _stage_1_deconstruct(self, sentence: str) -> DSVStageResult:
            start_time = time.time()
            logger.info("=== DSV stage 1: Deconstruct ===")

            analyst_prompt = f"""
    You are a professional semantic analyst agent. Your role is to decompose a natural language sentence into the core semantic components required to build an MTL formula.

    Sentence: "{sentence}"

    Output the semantic specification sketch in the following JSON format (exact JSON only):

    ```json
    {{
      "atomic_propositions": [{{"id": "ap_1", "description": "human-readable description", "variable": "var_name"}}],
      "temporal_relations": [],
      "metric_constraints": [],
      "global_property": "Always"
    }}
    ```

    Requirements:
    1. Output valid JSON only.
    2. Use concise English identifiers for variables.
    """

            messages = [
                {"role": "system", "content": "You are a professional semantic analyst skilled at decomposing natural language into structured semantic components."},
                {"role": "user", "content": analyst_prompt}
            ]

            try:
                response, token_usage = self._call_llm("analyst", messages)
                sketch = self._extract_semantic_sketch(response)
                processing_time = time.time() - start_time
                return DSVStageResult(stage=DSVStage.DECONSTRUCT, success=sketch.extraction_success, processing_time=processing_time, token_usage=token_usage, stage_output=sketch, agent_response=response, error_message=None if sketch.extraction_success else "Failed to extract JSON")
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Deconstruct stage failed: {e}")
                return DSVStageResult(stage=DSVStage.DECONSTRUCT, success=False, processing_time=processing_time, token_usage=TokenUsage(), stage_output=None, agent_response="", error_message=str(e))

    def _extract_semantic_sketch(self, response: str) -> SemanticSpecificationSketch:
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if not json_match:
                # Try to find the first JSON-like substring
                brace_match = re.search(r'\{[\s\S]*\}', response)
                json_str = brace_match.group(0) if brace_match else ""
            else:
                json_str = json_match.group(1).strip()

            if not json_str:
                logger.error("No JSON-formatted semantic sketch found in response")
                return SemanticSpecificationSketch(extraction_success=False)

                sketch_data = json.loads(json_str)
                lex = {}
                for ap in sketch_data.get("atomic_propositions", []):
                    ap_id = ap.get("id") or ap.get("variable")
                    ap_desc = ap.get("description") or ap.get("variable") or ""
                    if ap_id:
                        lex[ap_id] = ap_desc

                return SemanticSpecificationSketch(
                    atomic_propositions=sketch_data.get("atomic_propositions", []),
                    temporal_relations=sketch_data.get("temporal_relations", []),
                    metric_constraints=sketch_data.get("metric_constraints", []),
                    global_property=sketch_data.get("global_property", "Always"),
                    raw_json=json_str,
                    lexicon=lex,
                    extraction_success=True,
                )
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            return SemanticSpecificationSketch(extraction_success=False)
        except Exception as e:
            logger.error(f"Failed to extract semantic sketch: {e}")
            return SemanticSpecificationSketch(extraction_success=False)
        return SemanticSpecificationSketch(extraction_success=False)

    def _stage_2_synthesize(self, sketch: SemanticSpecificationSketch) -> DSVStageResult:
        start_time = time.time()
        logger.info("=== DSV stage 2: Synthesize ===")

        synthesizer_prompt = f"""
You are a professional MTL formula synthesizer agent. Your role is to produce a syntactically correct MTL formula given a structured semantic specification sketch.

```json
{sketch.raw_json}
```

Provide two labeled sections exactly as:
Reasoning:
[detailed reasoning]

Final MTL formula:
    [the formula]
    """

        messages = [
            {"role": "system", "content": "You are a professional MTL formula synthesizer. Strictly construct MTL formulas using only the provided semantic components."},
            {"role": "user", "content": synthesizer_prompt}
        ]

        try:
            response, token_usage = self._call_llm("synthesizer", messages)
            synthesis_result = self._extract_synthesis_result(response)
            processing_time = time.time() - start_time
            return DSVStageResult(stage=DSVStage.SYNTHESIZE, success=synthesis_result.synthesis_success, processing_time=processing_time, token_usage=token_usage, stage_output=synthesis_result, agent_response=response, error_message=None if synthesis_result.synthesis_success else "Failed to extract MTL formula")
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Synthesis stage failed: {e}")
            return DSVStageResult(stage=DSVStage.SYNTHESIZE, success=False, processing_time=processing_time, token_usage=TokenUsage(), stage_output=None, agent_response="", error_message=str(e))

        def _extract_synthesis_result(self, response: str) -> SynthesisResult:
            try:
                reasoning_match = re.search(r'Reasoning:\s*(.*?)(?=Final MTL formula:|$)', response, re.DOTALL | re.IGNORECASE)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

                formula_match = re.search(r'Final MTL formula:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
                if not formula_match:
                    code_block = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                    formula = code_block.group(1).strip() if code_block else ""
                else:
                    formula = formula_match.group(1).strip()

                formula = re.sub(r'\s+', ' ', formula).strip()
                if formula:
                    return SynthesisResult(mtl_formula=formula, synthesis_reasoning=reasoning, synthesis_success=True)
                else:
                    return SynthesisResult(mtl_formula="", synthesis_reasoning=reasoning, synthesis_success=False)
            except Exception as e:
                logger.error(f"Failed to extract synthesis result: {e}")
                return SynthesisResult(mtl_formula="", synthesis_reasoning="", synthesis_success=False)

    def _stage_3_verify(self, original_sentence: str, mtl_formula: str, lexicon: Optional[Dict[str, str]] = None) -> DSVStageResult:
        """Execute the verification stage by translating MTL back to natural language.
        
        Args:
            original_sentence: The original natural language input
            mtl_formula: The MTL formula to verify
            lexicon: Optional mapping of variable IDs to descriptions
            
        Returns:
            DSVStageResult containing verification success/failure and back-translation
        """
        start_time = time.time()
        logger.info("=== DSV stage 3: Verify ===")

        lex_text = json.dumps(lexicon, ensure_ascii=False, indent=2) if lexicon else "{}"
        verifier_prompt = f"""
You are a professional MTL formula verifier agent. Your task is to translate an MTL formula back into clear natural language to verify its semantic correctness.

MTL formula: {mtl_formula}

You do NOT have access to the original input sentence. You may use the following lexicon (ID -> description):
{lex_text}

Please answer in this exact format:
Translation reasoning:
[detailed reasoning]

Natural language translation:
[the translation]
"""

        messages = [
            {"role": "system", "content": "You are a professional MTL formula verifier skilled at translating formal MTL formulas into clear natural language."},
            {"role": "user", "content": verifier_prompt}
        ]

        try:
            response, token_usage = self._call_llm("verifier", messages)
            verification_result = self._extract_verification_result(response, original_sentence)
            processing_time = time.time() - start_time
            success_flag = bool(verification_result.back_translation)
            return DSVStageResult(
                stage=DSVStage.VERIFY,
                success=success_flag,
                processing_time=processing_time, 
                token_usage=token_usage,
                stage_output=verification_result,
                agent_response=response,
                error_message=None if success_flag else "Failed to extract back-translation"
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
        try:
            reasoning_match = re.search(r'Translation reasoning:\s*(.*?)(?=Natural language translation:|$)', response, re.DOTALL | re.IGNORECASE)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            translation_match = re.search(r'Natural language translation:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
            if not translation_match:
                # fallback: first non-empty line after reasoning
                parts = response.split('\n')
                back_translation = ""
                for line in parts[::-1]:
                    if line.strip():
                        back_translation = line.strip()
                        break
            else:
                back_translation = translation_match.group(1).strip()

            similarity_score = self._calculate_semantic_similarity(original_sentence, back_translation)
            verification_passed = similarity_score >= self.similarity_threshold
            return VerificationResult(back_translation=back_translation, similarity_score=similarity_score, verification_passed=verification_passed, verification_reasoning=reasoning)
        except Exception as e:
            logger.error(f"Failed to extract verification result: {e}")
            return VerificationResult(back_translation="", similarity_score=0.0, verification_passed=False, verification_reasoning="")

    def _calculate_semantic_similarity(self, sentence1: str, sentence2: str) -> float:
        """Calculate semantic similarity between two sentences.
        
        Args:
            sentence1: First sentence
            sentence2: Second sentence
            
        Returns:
            Cosine similarity score between sentence embeddings (0-1)
        """
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
        start_time = time.time()
        logger.info(f"Starting DSV processing: {sentence}")

        self.total_token_usage = TokenUsage()
        stage_results: List[DSVStageResult] = []
        refinement_iterations = 0
        final_mtl_formula = None
        success = False
        termination_reason = "Unknown"

        try:
            for iteration in range(self.max_refinement_iterations + 1):
                logger.info(f"=== DSV processing iteration {iteration + 1} ===")
                deconstruct_result = self._stage_1_deconstruct(sentence)
                stage_results.append(deconstruct_result)
                if not deconstruct_result.success:
                    termination_reason = "Deconstruct stage failed"
                    break

                    synth_result = self._stage_2_synthesize(deconstruct_result.stage_output)
                    stage_results.append(synth_result)
                    if not synth_result.success:
                        termination_reason = "Synthesize stage failed"
                        break

                    lexicon = deconstruct_result.stage_output.lexicon if deconstruct_result.stage_output else {}
                    verify_result = self._stage_3_verify(sentence, synth_result.stage_output.mtl_formula, lexicon=lexicon)
                    stage_results.append(verify_result)
                    if not verify_result.success:
                        termination_reason = "Verify stage failed"
                        break

                    if verify_result.stage_output.verification_passed:
                        final_mtl_formula = synth_result.stage_output.mtl_formula
                        success = True
                        termination_reason = f"Verification passed (similarity: {verify_result.stage_output.similarity_score:.3f})"
                        break
                    else:
                        refinement_iterations += 1
                        logger.info(f"Verification failed (similarity: {verify_result.stage_output.similarity_score:.3f}), below threshold {self.similarity_threshold}")
                        if not enable_refinement or iteration >= self.max_refinement_iterations:
                            final_mtl_formula = synth_result.stage_output.mtl_formula
                            termination_reason = f"Reached max refinement iterations (similarity: {verify_result.stage_output.similarity_score:.3f})"
                            break
                        else:
                            logger.info(f"Starting refinement iteration {refinement_iterations}")
                            continue

            total_processing_time = time.time() - start_time
            return DSVProcessResult(input_sentence=sentence, final_mtl_formula=final_mtl_formula, total_processing_time=total_processing_time, total_token_usage=self.total_token_usage, stage_results=stage_results, refinement_iterations=refinement_iterations, success=success, termination_reason=termination_reason)
        except Exception as e:
            total_processing_time = time.time() - start_time
            logger.error(f"DSV processing failed: {e}")
            return DSVProcessResult(input_sentence=sentence, final_mtl_formula=None, total_processing_time=total_processing_time, total_token_usage=self.total_token_usage, stage_results=stage_results, refinement_iterations=refinement_iterations, success=False, termination_reason=str(e))

    def save_result(self, result: DSVProcessResult, output_file: str) -> None:
        """Save DSV processing result to a JSON file.
        
        Args:
            result: The DSVProcessResult to save
            output_file: Path to the output JSON file
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
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
                "agents": self.config.get("agents", {})
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
        
        logger.info(f"DSV result saved to: {output_file}")


def main() -> None:
    """Run demo of DSV Framework with example sentences."""
    print("=== DSV Framework Demo ===\n")
    dsv = DSVFramework()
    
    test_sentences = [
        "Within 5 to 10 seconds after sensor A detects a fault, Alarm B must sound and last for at least 20 seconds.",
        "After receiving the signal, the system must respond within 10 seconds.",
        "The door should remain locked for at least 30 seconds after the alarm is triggered."
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"=== Test sentence {i} ===")
        print(f"Input: {sentence}")
        print("-" * 60)
        
        try:
            result = dsv.process(sentence, enable_refinement=True)
            
            # Display results
            print(f"Success: {result.success}")
            print(f"Final MTL formula: {result.final_mtl_formula}")
            print(f"Refinement iterations: {result.refinement_iterations}")
            print(f"Termination reason: {result.termination_reason}")
            print(f"Total processing time: {result.total_processing_time:.2f}s")
            print(f"Total tokens: {result.total_token_usage.total_tokens}")
            
            # Save results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"data/output/dsv/result_{i}_{timestamp}.json"
            dsv.save_result(result, output_file)
            
        except Exception as e:
            print(f"Processing failed: {e}")
            
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
                
                return VerificationResult(
                    back_translation=back_translation,
                    similarity_score=similarity_score,
                    verification_passed=verification_passed,
                    verification_reasoning=reasoning
                )
            else:
                logger.error("No natural language translation found in verifier response")
                return VerificationResult(
                    back_translation="",
                    similarity_score=0.0,
                    verification_passed=False,
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
        """计算两个句子的语义相似度"""
        try:
            # 编码两个句子
            embeddings = self.sentence_model.encode([sentence1, sentence2], convert_to_tensor=True)
            
            # 计算余弦相似度
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])[0][0]
            return float(similarity.cpu().numpy())
            
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.0
    

        success = False
        termination_reason = "Unknown"
        
        try:
            # Main processing loop
            for iteration in range(self.max_refinement_iterations + 1):
                logger.info(f"=== DSV processing iteration {iteration + 1} ===")
                
                # Stage 1: Deconstruct
                deconstruct_result = self._stage_1_deconstruct(sentence)
                stage_results.append(deconstruct_result)
                
                if not deconstruct_result.success:
                    termination_reason = "Deconstruct stage failed"
                    break
                
                # Stage 2: Synthesize
                synthesize_result = self._stage_2_synthesize(deconstruct_result.stage_output)
                stage_results.append(synthesize_result)
                
                if not synthesize_result.success:
                    termination_reason = "Synthesize stage failed"
                    break
                
                # Stage 3: Verify (blind verifier receives lexicon only)
                lexicon = deconstruct_result.stage_output.lexicon if deconstruct_result.stage_output else {}
                verify_result = self._stage_3_verify(sentence, synthesize_result.stage_output.mtl_formula, lexicon=lexicon)
                stage_results.append(verify_result)
                
                if not verify_result.success:
                    termination_reason = "Verify stage failed"
                    break
                
                # Check verification result
                if verify_result.stage_output.verification_passed:
                    # Verification passed, success
                    final_mtl_formula = synthesize_result.stage_output.mtl_formula
                    success = True
                    termination_reason = f"Verification passed (similarity: {verify_result.stage_output.similarity_score:.3f})"
                    break
                else:
                    # Verification failed
                    refinement_iterations += 1
                    logger.info(f"Verification failed (similarity: {verify_result.stage_output.similarity_score:.3f}), "
                              f"below threshold {self.similarity_threshold}")
                    
                    if not enable_refinement or iteration >= self.max_refinement_iterations:
                        # No refinement enabled or max iterations reached
                        final_mtl_formula = synthesize_result.stage_output.mtl_formula
                        termination_reason = f"Reached max refinement iterations (similarity: {verify_result.stage_output.similarity_score:.3f})"
                        break
                    else:
                        # Continue refinement loop
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
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
            logger.info(f"DSV result saved to: {output_file}")


