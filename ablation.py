import json
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field, replace
from enum import Enum
from pathlib import Path
import logging
import os
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from dotenv import load_dotenv
import openpyxl

load_dotenv()

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
    """Result of LTL formula synthesis."""
    ltl_formula: str
    synthesis_reasoning: str
    synthesis_success: bool


@dataclass
class VerificationResult:
    """Result of LTL formula verification."""
    back_translation: str
    similarity_score: float
    verification_passed: bool
    verification_reasoning: str


@dataclass
class RefinementFeedback:
    """Feedback from a failed refinement iteration."""
    iteration: int
    ltl_formula: str
    back_translation: str
    similarity_score: float
    semantic_sketch_json: str
    issue_analysis: str


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
    final_ltl_formula: Optional[str]
    total_processing_time: float
    total_token_usage: TokenUsage
    stage_results: List[DSVStageResult]
    refinement_iterations: int
    success: bool
    termination_reason: str

# LTL Knowledge Base - Standardized syntax and operators
LTL_KNOWLEDGE_BASE = """
**<Linear Temporal Logic Knowledge Base>**

Use only the following operators and symbols (consistent with LTL standard):

**Temporal operators**:
- `X(φ)` — next (φ holds in the next step)
- `F(φ)` — eventually/finally (φ holds at some point in the future)
- `G(φ)` — globally/always (φ holds at all future points)
- `φ U ψ` — until (φ holds until ψ becomes true)
- `φ R ψ` — release (ψ holds until and including when φ first becomes true, or forever if φ never becomes true)

**Logical connectives**:
- `&` (and), `|` (or), `!` or `~` (not), `->` (implication), `<->` (equivalence)

**Atomic propositions**:
- Simple propositional variables: `a`, `b`, `c`, `p`, `q`, etc.
- The formula should only contain **atomic propositions** and the above operators.
- Use lowercase letters for atomic propositions to maintain consistency.

**Important notes**:
* LTL does not include explicit time bounds or metric constraints.
* Focus on the ordering and occurrence of events, not their timing.
* "In the next step" uses `X`, "eventually" uses `F`, "always" uses `G`.
* For "until" patterns, use the `U` operator.

---

**I. Temporal Operator Mapping (Natural Language → LTL)**:

1. **Basic temporal patterns**:
   * "in the next step" → `X(φ)`
   * "eventually" / "at some point" / "will happen" → `F(φ)`
   * "always" / "at all times" / "never" (as `G(!φ)`) → `G(φ)`
   * "φ until ψ" / "φ holds until ψ" → `φ U ψ`
   * "infinitely often" / "repeatedly" → `G(F(φ))`
   * "from some point on" → `F(G(φ))`

2. **Common patterns**:
   * "if a then eventually b" → `G(a -> F(b))`
   * "a is always followed by b" → `G(a -> X(b))`
   * "whenever a holds, b holds as well" → `G(a -> b)`
   * "a and b never occur at the same time" → `G(!(a & b))`
   * "before a happens, b must have happened" → `(!a) U b`
   * "a happens, then b never happens again" → `G(a -> G(!b))`

3. **Nested temporal operators**:
   * "in the next two steps" → `X(X(φ))`
   * "eventually always" → `F(G(φ))`
   * "always eventually" → `G(F(φ))`

---

**II. Analysis Requirements (applied each time)**:

1. **Sentence Decomposition**: Break into clauses, identify conditions, events, and temporal relationships.
2. **Keyword Identification**: Detect temporal expressions (`always`, `eventually`, `next`, `until`, `never`, `infinitely often`).
3. **Atomic Proposition Extraction**: Map natural-language phrases into simple propositional variables (e.g., `a`, `b`, `request`, `grant`).
4. **LTL Construction & Verification**:
   * Use `X`, `F`, `G`, `U`, `R` appropriately.
   * Ensure proper nesting and operator precedence.
   * Verify formula reflects the temporal ordering faithfully.

---

**III. Simplification Rules**:
- Avoid redundant nesting (e.g., simplify `F(F(φ))` → `F(φ)`, `G(G(φ))` → `G(φ)`).
- Use standard operator precedence: `!` > `&` > `|` > `->` > `<->`.
- Prefer canonical readable forms (e.g., `G(a -> F(b))`).

---
"""


class DSVFrameworkAblation:
    """
    DSV Framework - Ablation Study Version for LTL
    基础DSV框架（LTL版本），不包含动态增强模块，用于消融实验
    """

    def __init__(self, config_path: str = "config/dsva_config.json"):
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
        processing_params = self.config.get("processing_parameters", {})
        self.similarity_threshold = processing_params.get("similarity_threshold", 0.85)
        self.max_refinement_iterations = processing_params.get("max_refinement_iterations", 3)

        logger.info("DSV Framework (Ablation Version) initialized")
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
        logger.info(f"Max refinement iterations: {self.max_refinement_iterations}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

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
        except TimeoutError as e:
            logger.error(f"⏱️  API call timeout for {agent_type} after 120s: {e}")
            return "", TokenUsage()
        except Exception as e:
            logger.error(f"❌ LLM call failed for {agent_type}: {type(e).__name__}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return "", TokenUsage()

    def _analyze_verification_failure(
        self,
        original_sentence: str,
        ltl_formula: str,
        back_translation: str,
        similarity_score: float,
        semantic_sketch_json: str
    ) -> str:
        """Analyze why verification failed and provide specific feedback"""
        analysis_prompt = f"""
You are an expert analyst tasked with identifying why an LTL formula verification failed.

Original Sentence: "{original_sentence}"

Generated LTL Formula: {ltl_formula}

Back-translation: "{back_translation}"

Semantic Similarity Score: {similarity_score:.3f} (Threshold: {self.similarity_threshold})

Semantic Sketch Used:
```json
{semantic_sketch_json}
```

Please analyze the discrepancy between the original sentence and the back-translation. Identify:

1. What semantic information was lost or misinterpreted?
2. What temporal/metric constraints were incorrectly captured?
3. Specific suggestions for correcting the semantic decomposition

Provide a concise analysis focusing on actionable corrections.
"""

        messages = [
            {"role": "system", "content": "You are an expert in temporal logic and semantic analysis. Provide precise, actionable feedback for improving LTL formula generation."},
            {"role": "user", "content": analysis_prompt}
        ]

        try:
            response, _ = self._call_llm("analyst", messages)
            return response
        except Exception as e:
            logger.error(f"Failure analysis failed: {e}")
            return f"Analysis failed: Low similarity score ({similarity_score:.3f}). The back-translation differs significantly from the original sentence."

    def _format_refinement_history(self, refinement_history: List[RefinementFeedback]) -> str:
        """Format refinement history for inclusion in prompts"""
        if not refinement_history:
            return ""
        
        history_text = "\n=== PREVIOUS REFINEMENT ATTEMPTS ===\n"
        history_text += "The following attempts were made but failed verification. Learn from these mistakes:\n\n"
        
        for feedback in refinement_history:
            history_text += f"Attempt {feedback.iteration}:\n"
            history_text += f"- Semantic Sketch:\n```json\n{feedback.semantic_sketch_json}\n```\n"
            history_text += f"- Generated LTL Formula: {feedback.ltl_formula}\n"
            history_text += f"- Back-translation: \"{feedback.back_translation}\"\n"
            history_text += f"- Similarity Score: {feedback.similarity_score:.3f}\n"
            history_text += f"- Issue Analysis:\n{feedback.issue_analysis}\n"
            history_text += "\n" + "-"*60 + "\n\n"
        
        history_text += "Based on these failures, please adjust your approach to avoid repeating the same mistakes.\n"
        history_text += "=== END OF PREVIOUS ATTEMPTS ===\n\n"
        
        return history_text

    def _stage_1_deconstruct(self, sentence: str, refinement_history: Optional[List[RefinementFeedback]] = None) -> DSVStageResult:
        """Stage 1: Deconstruct natural language into semantic components with refinement feedback"""
        start_time = time.time()
        logger.info("=== DSV Stage 1: Deconstruct ===")
        
        refinement_history = refinement_history or []
        
        # Format refinement history if available
        history_context = self._format_refinement_history(refinement_history)

# Enhanced analyst prompt with l't'l knowledge base and refinement feedback
        analyst_prompt = f"""
You are a professional semantic analysis agent tasked with decomposing natural language sentences into core semantic components required to construct LTL formulas.

{LTL_KNOWLEDGE_BASE}

**Object and Predicate Extraction Guidelines**:

1. **Object Naming Conventions**:
   - Use lowercase, concise names: "ego", "other", "sign_306"
   - For sensors/devices: Use CamelCase without spaces: "SensorA", "AlarmB", "StatusLight"
   - Avoid full descriptions in names (use "ego" not "Ego vehicle")

2. **Predicate Naming Conventions**:
   - Use snake_case: "in_front_of", "sudden_braking", "at_intersection", "detects_fault"
   - For relational predicates (2 objects): include preposition
     Examples: in_front_of, in_right_of, yield
   - For unary predicates (1 object): use verb/state
     Examples: sudden_braking, at_intersection, detects_fault, sounds

3. **Related Objects** (CRITICAL for relational predicates):
   - MUST specify "related_object" field for predicates involving 2 entities
   - Example: "in_front_of" requires subject (ego) and object (other)
   - Example: "yield" requires who yields (ego) and to whom (other)

4. **Complete Object List**:
   - Extract ALL entities mentioned in the sentence
   - Each object needs: id, name, type
   - Types: "vehicle", "sensor", "actuator", "indicator", "traffic_sign", etc.

{history_context}Analyze the following sentence and extract structured information:

Sentence: "{sentence}"

Provide a JSON-formatted semantic specification sketch containing the following fields:

1. **objects**:
   A list of ALL identified physical or logical entities, each with:
   - id: unique identifier (e.g., "obj_ego", "obj_other", "obj_sensor_a")
   - name: concise object name (e.g., "ego", "other", "SensorA")
   - type: category (e.g., "vehicle", "sensor", "actuator", "indicator", "traffic_sign")

2. **atomic_propositions**:
   List of atomic propositions, each containing:
   - id: unique identifier (e.g., "ap_1", "ap_2")
   - object: reference to object ID from objects list
   - related_object: (REQUIRED for relational predicates) reference to second object ID
   - predicate: the action or property in snake_case (e.g., "in_front_of", "sudden_braking")
   - variable: short variable name for reference
   - description: human-readable description

3. **temporal_relations**:
   List of temporal relations describing time relationships between atomic propositions

4. **metric_constraints**:
   List of metric constraints, including time window, duration, etc.

5. **global_property**:
   Global property (e.g., "Always", "Eventually")

6. **lexicon**:
   A lexicon mapping variable names to natural language descriptions

Ensure the output adheres to valid JSON format.

Example output format:
```json
{{
    "objects": [
        {{"id": "obj_ego", "name": "ego", "type": "vehicle"}},
        {{"id": "obj_other", "name": "other", "type": "vehicle"}}
    ],
    "atomic_propositions": [
        {{
            "id": "ap_1",
            "object": "obj_ego",
            "related_object": "obj_other",
            "predicate": "in_front_of",
            "variable": "ego_in_front",
            "description": "Ego vehicle is in front of other vehicle"
        }}
    ],
    "temporal_relations": [
        {{"type": "relation_type", "antecedent": "antecedent", "consequent": "consequent", "description": "description"}}
    ],
    "metric_constraints": [
        {{"applies_to": "applies_to", "type": "constraint_type", "value": "constraint_value", "description": "description"}}
    ],
    "global_property": "Always",
    "lexicon": {{"variable_name": "natural language description"}}
}}
```

Please refer to the processing approach in the above example, but analyze it based on the specific context of the current sentence.
"""

        messages = [
            {"role": "system", "content": "You are a professional semantic analyst specializing in deconstructing natural language into structured semantic components. Output strictly in JSON format as required. Refer to the provided examples, but perform independent analysis based on specific inputs."},{"role": "user", "content": analyst_prompt}
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

    def _stage_2_synthesize(self, sketch: SemanticSpecificationSketch, refinement_history: Optional[List[RefinementFeedback]] = None) -> DSVStageResult:
        """Stage 2: Synthesize LTL formula from semantic sketch with refinement feedback"""
        start_time = time.time()
        logger.info("=== DSV Stage 2: Synthesize ===")
        
        refinement_history = refinement_history or []
        
        # Format refinement history if available
        history_context = self._format_refinement_history(refinement_history)

        # Enhanced synthesizer prompt with LTL knowledge base and refinement feedback
        synthesizer_prompt = f"""
You are a professional LTL formula synthesizer agent tasked with generating syntactically correct LTL formulas based on structured semantic specification sketches.

{LTL_KNOWLEDGE_BASE}

**CRITICAL RULES - MUST FOLLOW**:

1. **MANDATORY**: Use predicate(object) format for ALL atomic propositions. NEVER use simple variables.
   ❌ WRONG: ego_in_front, sudden_brake, alarm_active
   ✅ CORRECT: in_front_of(ego,other), sudden_braking(ego), sounds(AlarmB)

2. **For Relational Predicates** (involving 2 objects, with "related_object" field in sketch):
   - Format: predicate(subject, object)
   - Examples: 
     * in_front_of(ego,other)
     * yield(ego,other)
     * in_right_of(other,ego)
   - Order matters: predicate(主体, 客体)

3. **For Unary Predicates** (single object, no "related_object" field):
   - Format: predicate(object)
   - Examples:
     * sudden_braking(ego)
     * at_intersection(ego)
     * detects_fault(SensorA)
     * sounds(AlarmB)

4. **Object Name Extraction**:
   - Extract object names from "objects" list in the semantic sketch
   - Use the "name" field directly (e.g., "ego", "other", "SensorA", "AlarmB")
   - For each atomic proposition:
     a) Find the "object" field (object ID) → map to object "name"
     b) If "related_object" exists → also map its ID to object "name"
     c) Build: predicate(object_name) or predicate(object1_name, object2_name)

5. **Step-by-Step Construction Process**:
   Step 1: Parse the "objects" list to build ID→name mapping
   Step 2: For each atomic proposition:
           - Extract predicate name
           - Map object ID to object name
           - If related_object exists, map it to related object name
           - Construct: predicate(name) or predicate(name1, name2)
   Step 3: Combine predicates using temporal and logical operators
   Step 4: Apply global property (G, F, etc.)

{history_context}The semantic specification sketch you received is as follows:

```json
{sketch.raw_json}
```

**Your Task**:
Synthesize a syntactically correct LTL formula based on this semantic specification sketch.
REMEMBER: Every atomic proposition MUST use predicate(object) format!

LTL Syntax Rules:
- G: Globally (always)
- F: Finally (ultimately)  
- X: Next (subsequent)
- U: Until (until)
- Time Interval: [a,b] denotes a time window
- Logical Operations: & (and), | (or), ~ (not), -> (implication)

Please provide the following two sections:

Reasoning Process:
[Explain step-by-step how you extract objects and construct predicate(object) forms]

Final LTL Formula:
[Synthetic LTL formula using ONLY predicate(object) format]

Please follow the processing approach demonstrated in the above example, but synthesize the formula based on the specific content of the current semantic specification sketch.
"""

        messages = [
            {"role": "system", "content": "You are a professional LTL formula synthesizer. Strictly construct LTL formulas based on the provided semantic components without adding any additional explanations or speculations. Refer to the provided examples, but synthesize independently according to the specific semantic specification sketch."},{"role": "user", "content": synthesizer_prompt}
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
                error_message=None if synthesis_result.synthesis_success else "Failed to extract LTL formula"
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
            reasoning_match = re.search(r'Reasoning Process[:：]\s*(.*?)(?=Final LTL Formula[:：]|$)', response, re.DOTALL | re.IGNORECASE)
            if not reasoning_match:
                reasoning_match = re.search(r'Reasoning[:：]\s*(.*?)(?=Final LTL formula[:：]|Final LTL Formula[:：]|$)', response, re.DOTALL | re.IGNORECASE)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            # Extract LTL formula
            formula_match = re.search(r'Final LTL Formula[:：]\s*(.*)', response, re.DOTALL | re.IGNORECASE)
            if not formula_match:
                formula_match = re.search(r'Final LTL formula[:：]\s*(.*)', response, re.DOTALL | re.IGNORECASE)
            
            if not formula_match:
                # Try to find formula in code blocks
                code_block = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                formula = code_block.group(1).strip() if code_block else ""
            else:
                formula = formula_match.group(1).strip()

            # Clean up the formula
            formula = re.sub(r'[`\n\r]', '', formula).strip()

            if not formula:
                logger.error("No LTL formula found in synthesizer response")
                return SynthesisResult(
                    ltl_formula="",
                    synthesis_reasoning=reasoning,
                    synthesis_success=False
                )

            return SynthesisResult(
                ltl_formula=formula,
                synthesis_reasoning=reasoning,
                synthesis_success=True
            )
        except Exception as e:
            logger.error(f"Failed to extract synthesis result: {e}")
            return SynthesisResult(
                ltl_formula="",
                synthesis_reasoning="",
                synthesis_success=False
            )

    def _stage_3_verify(self, original_sentence: str, ltl_formula: str, lexicon: Optional[Dict[str, str]] = None) -> DSVStageResult:
        """Stage 3: Verify LTL formula by back-translation"""
        start_time = time.time()
        logger.info("=== DSV Stage 3: Verify ===")

        lexicon = lexicon or {}
        
        # Format lexicon for prompt
        lexicon_text = ""
        if lexicon:
            lexicon_text = "Variable Vocabulary List:\n"
            for var, desc in lexicon.items():
                lexicon_text += f"- {var}: {desc}\n"
            lexicon_text += "\n"

        # Enhanced verifier prompt with LTL knowledge base
        verifier_prompt = f"""
You are a professional LTL formula verifier Agent, responsible for translating LTL formulas back into natural language for verification.

{LTL_KNOWLEDGE_BASE}

**Predicate(Object) Interpretation Rules**:

1. **Relational Predicates** (two parameters):
   - Format: predicate(subject, object)
   - in_front_of(ego,other) → "ego is in front of other" or "the ego vehicle is in front of the other vehicle"
   - yield(ego,other) → "ego yields to other" or "ego vehicle yields to other vehicle"
   - in_right_of(other,ego) → "other is on the right side of ego"

2. **Unary Predicates** (one parameter):
   - Format: predicate(object)
   - sudden_braking(ego) → "ego brakes suddenly" or "ego vehicle brakes suddenly"
   - at_intersection(ego) → "ego is at an intersection" or "ego vehicle is at an intersection"
   - detects_fault(SensorA) → "SensorA detects a fault" or "sensor A detects a fault"
   - sounds(AlarmB) → "AlarmB sounds" or "alarm B sounds"

3. **Preserve Object Information**:
   - Always mention the specific objects involved
   - Use natural language equivalents: "ego vehicle", "other vehicle", "sensor A", "alarm B"

{lexicon_text}LTL formula to be verified: {ltl_formula}

Please translate this LTL formula into a clear natural language description.

LTL Symbol Meanings:
- G: Always/Globally
- F: Eventually/At some future time
- X: Next time step
- U: Until
- [a,b]: Time interval from a to b
- &: And
- |: Or  
- ~: Not
- ->: Implication

Provide two marked sections:

Reasoning Process:
[Explain how you interpret each predicate(object) and translate to natural language]

Natural Language Translation:
[Result of translating LTL formula into natural language, preserving object information]

Please follow the processing method from the above example, but adapt the translation based on the specific content of the current LTL formula.
"""

        messages = [
            {"role": "system", "content": "You are a professional LTL formula translator, skilled at converting formal formulas into natural language. Ensure translations are accurate and easy to understand. Refer to the provided examples, but translate each LTL formula independently."},{"role": "user", "content": verifier_prompt}
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
            reasoning_match = re.search(r'Reasoning Process[:：]\s*(.*?)(?=Natural Language Translation[:：]|$)', response, re.DOTALL | re.IGNORECASE)
            if not reasoning_match:
                reasoning_match = re.search(r'Reasoning[:：]\s*(.*?)(?=Natural language translation[:：]|Natural Language Translation[:：]|$)', response, re.DOTALL | re.IGNORECASE)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            # Extract back translation
            translation_match = re.search(r'Natural Language Translation[:：]\s*(.*)', response, re.DOTALL | re.IGNORECASE)
            if not translation_match:
                translation_match = re.search(r'Natural language translation[:：]\s*(.*)', response, re.DOTALL | re.IGNORECASE)

            if translation_match:
                back_translation = translation_match.group(1).strip()
            else:
                # Fallback: try to find the translation in the response
                lines = response.split('\n')
                back_translation = ""
                for line in reversed(lines):
                    if line.strip() and not line.strip().startswith(('Reasoning', 'Natural')):
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
        """Process a sentence through the complete DSV pipeline with refinement feedback"""
        start_time = time.time()
        logger.info(f"Starting DSV processing (Ablation Version with Refinement Feedback): {sentence}")

        # Reset token usage tracking
        self.total_token_usage = TokenUsage()
        
        stage_results = []
        refinement_iterations = 0
        final_ltl_formula = None
        success = False
        termination_reason = "Unknown"
        
        # Track refinement history for feedback
        refinement_history: List[RefinementFeedback] = []

        try:
            for iteration in range(self.max_refinement_iterations + 1):
                logger.info(f"=== DSV Processing Iteration {iteration + 1} ===")
                
                if refinement_history:
                    logger.info(f"Using feedback from {len(refinement_history)} previous attempt(s)")
                
                # Stage 1: Deconstruct (with refinement history)
                deconstruct_result = self._stage_1_deconstruct(sentence, refinement_history=refinement_history)
                stage_results.append(deconstruct_result)
                if not deconstruct_result.success:
                    termination_reason = "Deconstruct stage failed"
                    break

                # Stage 2: Synthesize (with refinement history)
                synth_result = self._stage_2_synthesize(
                    deconstruct_result.stage_output,
                    refinement_history=refinement_history
                )
                stage_results.append(synth_result)
                if not synth_result.success:
                    termination_reason = "Synthesize stage failed"
                    break

                # Stage 3: Verify
                lexicon = deconstruct_result.stage_output.lexicon if deconstruct_result.stage_output else {}
                verify_result = self._stage_3_verify(sentence, synth_result.stage_output.ltl_formula, lexicon=lexicon)
                stage_results.append(verify_result)
                if not verify_result.success:
                    termination_reason = "Verify stage failed"
                    break

                # Check verification result
                if verify_result.stage_output.verification_passed:
                    final_ltl_formula = synth_result.stage_output.ltl_formula
                    success = True
                    termination_reason = f"Verification passed (similarity: {verify_result.stage_output.similarity_score:.3f})"
                    logger.info(f"✅ Success after {iteration + 1} iteration(s)")
                    break
                else:
                    refinement_iterations += 1
                    similarity = verify_result.stage_output.similarity_score
                    logger.info(f"Verification failed (similarity: {similarity:.3f}), below threshold {self.similarity_threshold}")
                    
                    if not enable_refinement or iteration >= self.max_refinement_iterations:
                        final_ltl_formula = synth_result.stage_output.ltl_formula
                        termination_reason = f"Reached max refinement iterations (similarity: {similarity:.3f})"
                        break
                    else:
                        # Analyze failure and create feedback for next iteration
                        logger.info(f"Analyzing failure to improve next iteration...")
                        issue_analysis = self._analyze_verification_failure(
                            original_sentence=sentence,
                            ltl_formula=synth_result.stage_output.ltl_formula,
                            back_translation=verify_result.stage_output.back_translation,
                            similarity_score=similarity,
                            semantic_sketch_json=deconstruct_result.stage_output.raw_json
                        )
                        
                        # Store feedback for next iteration
                        feedback = RefinementFeedback(
                            iteration=iteration + 1,
                            ltl_formula=synth_result.stage_output.ltl_formula,
                            back_translation=verify_result.stage_output.back_translation,
                            similarity_score=similarity,
                            semantic_sketch_json=deconstruct_result.stage_output.raw_json,
                            issue_analysis=issue_analysis
                        )
                        refinement_history.append(feedback)
                        
                        logger.info(f"Starting refinement iteration {refinement_iterations} with feedback")
                        continue

            total_processing_time = time.time() - start_time
            return DSVProcessResult(
                input_sentence=sentence,
                final_ltl_formula=final_ltl_formula,
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
                final_ltl_formula=None,
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
            "final_ltl_formula": result.final_ltl_formula,
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
                        "ltl_formula": stage_result.stage_output.ltl_formula,
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
        
        logger.info(f"DSVA result (Ablation Version) saved to: {output_file}")


def main() -> None:
    """Run demo of DSVA Framework Ablation Version"""
    print("=== DSVA Framework Demo - Ablation Version ===\n")
    print("This version does NOT include dynamic example enhancement for ablation studies.\n")
    
    dsv = DSVFrameworkAblation()
    
    dataset = []
    with open("data/input/dataset.xlsx", "rb") as f:
        # 读取Excel文件
        workbook = openpyxl.load_workbook(f)
        # 选择第一个工作表
        sheet = workbook.active
        if sheet is None:
            raise ValueError("Could not load active sheet from workbook")
        # 读取 Natural Language 列的数据
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if row and row[0]:
                dataset.append(row[0])
    
    for i, sentence in enumerate(dataset, 1):
        print(f"=== Test Sentence {i} ===")
        print(f"Input: {sentence}")
        print("-" * 60)

        try:
            result = dsv.process(sentence, enable_refinement=True)
            
            # Display results
            print(f"✅ Success: {result.success}")
            print(f"🎯 Final LTL formula: {result.final_ltl_formula}")
            print(f"🔄 Refinement iterations: {result.refinement_iterations}")
            print(f"📝 Termination reason: {result.termination_reason}")
            print(f"⏱️  Total processing time: {result.total_processing_time:.2f}s")
            print(f"🔢 Total tokens: {result.total_token_usage.total_tokens}")
            print(f"🚫 Dynamic enhancement: Disabled (Ablation Version)")
            
            # Save results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = Path("data/output/ablation/gpt-4")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"result_{i}_{timestamp}.json"
            dsv.save_result(result, str(output_file))
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
