
# DSV Framework: Deconstruct-Synthesize-Verify

A sophisticated framework for translating natural language specifications into Metric Temporal Logic (MTL) formulas using a three-stage agent-based pipeline with intelligent refinement feedback.

## 🌟 Overview

The DSV Framework employs a **Deconstruct → Synthesize → Verify** approach to convert natural language requirements into formal MTL formulas:

1. **Deconstruct**: Semantic analysis agent breaks down natural language into structured components
2. **Synthesize**: MTL formula synthesizer generates formal logic from semantic sketches
3. **Verify**: Back-translation verifier validates formula correctness through semantic similarity

### Key Features

✅ **Intelligent Refinement Loop**: Learns from verification failures with detailed feedback analysis  
✅ **MTL Knowledge Base Integration**: Standardized temporal logic operators and mappings  
✅ **Dynamic Example Retrieval** (Complete version): Semantic similarity-based few-shot learning  
✅ **Multi-Agent Architecture**: Specialized agents for each DSV stage  
✅ **Comprehensive Tracking**: Token usage, processing time, and refinement history  

---

## 📁 Project Structure

```
DSV-Framework/
├── dsv_framework_complete.py    # Enhanced version with example retrieval
├── dsv_framework_ablation.py    # Baseline version for ablation studies
├── retrieval.py                  # Example retrieval system
├── config/
│   ├── dsv_config.json          # Framework configuration
│   └── single_prompt.txt        # Additional prompt templates
├── data/
│   ├── examples/
│   │   └── dsv_examples.json    # Example dataset for retrieval
│   └── output/
│       ├── dsv_enhanced/        # Enhanced version outputs
│       └── dsv_ablation/        # Ablation version outputs
├── .env                         # API credentials (YOU MUST CREATE THIS)
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd DSV-Framework

# Install dependencies
pip install -r requirements.txt
```

### 2. Create `.env` File

**⚠️ IMPORTANT**: You must create a `.env` file in the project root with your API credentials:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1

# Alternative: Use custom API endpoints
# OPENAI_API_KEY=your-custom-api-key
# OPENAI_BASE_URL=https://your-custom-endpoint.com/v1
```

**Note**: The `.env` file is required for both framework versions to authenticate with the LLM API.

### 3. Run the Framework

#### Option A: Enhanced Version (with Dynamic Examples)

```python
from dsv_framework_complete import EnhancedDSVFramework

# Initialize framework
dsv = EnhancedDSVFramework(config_path="config/dsv_config.json")

# Process a natural language specification
result = dsv.process(
    sentence="Within 5 to 10 seconds after Sensor A detects a fault, "
             "Alarm B must sound and remain active for at least 20 seconds.",
    enable_refinement=True
)

# Display results
print(f"Success: {result.success}")
print(f"MTL Formula: {result.final_mtl_formula}")
print(f"Refinement Iterations: {result.refinement_iterations}")
print(f"Similarity Score: {result.stage_results[-1].stage_output.similarity_score:.3f}")

# Save results
dsv.save_result(result, "data/output/dsv_enhanced/my_result.json")
```

#### Option B: Ablation Version (Baseline)

```python
from dsv_framework_ablation import DSVFrameworkAblation

# Initialize baseline framework
dsv = DSVFrameworkAblation(config_path="config/dsv_config.json")

# Process specification
result = dsv.process(
    sentence="After receiving the signal, the system must respond within 10 seconds.",
    enable_refinement=True
)

# Display results
print(f"Success: {result.success}")
print(f"MTL Formula: {result.final_mtl_formula}")
```

### 4. Run Demo Scripts

```bash
# Enhanced version demo
python dsv_framework_complete.py

# Ablation version demo
python dsv_framework_ablation.py
```

---

## 🏗️ Architecture

### Core Components

#### 1. **DSV Framework Ablation** ([`dsv_framework_ablation.py`](dsv_framework_ablation.py))

The baseline version without dynamic example retrieval, designed for ablation studies.

**Key Classes:**
- [`DSVFrameworkAblation`](dsv_framework_ablation.py:108): Main framework class
- [`SemanticSpecificationSketch`](dsv_framework_ablation.py:44): Structured semantic components
- [`SynthesisResult`](dsv_framework_ablation.py:56): MTL formula synthesis output
- [`VerificationResult`](dsv_framework_ablation.py:64): Back-translation verification output
- [`RefinementFeedback`](dsv_framework_ablation.py:73): Feedback from failed iterations

**Key Methods:**
- [`_stage_1_deconstruct()`](dsv_framework_ablation.py:263): Natural language → Semantic sketch
- [`_stage_2_synthesize()`](dsv_framework_ablation.py:388): Semantic sketch → MTL formula
- [`_stage_3_verify()`](dsv_framework_ablation.py:504): MTL formula → Back-translation + similarity
- [`_analyze_verification_failure()`](dsv_framework_ablation.py:195): Diagnose refinement failures
- [`process()`](dsv_framework_ablation.py:646): Complete DSV pipeline with refinement

#### 2. **Enhanced DSV Framework** ([`dsv_framework_complete.py`](dsv_framework_complete.py))

Full-featured version with dynamic example retrieval for improved performance.

**Additional Features:**
- Semantic similarity-based example retrieval
- Top-k relevant examples injected into agent prompts
- All baseline features + example enhancement

**Key Methods:**
- [`_get_examples_for_stage()`](dsv_framework_complete.py:280): Retrieve relevant examples
- [`toggle_examples()`](dsv_framework_complete.py:887): Enable/disable example enhancement

#### 3. **Example Retrieval System** ([`retrieval.py`](retrieval.py))

Manages semantic similarity-based example retrieval for the enhanced version.

**Key Classes:**
- `ExampleRetriever`: Retrieve top-k similar examples
- `RetrievalResult`: Store retrieval results with similarity scores

---

## 🔧 Configuration

### Main Configuration File: [`config/dsv_config.json`](config/dsv_config.json)

```json
{
  "agents": {
    "analyst": {
      "model": "gpt-4",
      "temperature": 0.0,
      "api_key_env": "OPENAI_API_KEY",
      "base_url_env": "OPENAI_BASE_URL"
    },
    "synthesizer": {
      "model": "gpt-4",
      "temperature": 0.0,
      "api_key_env": "OPENAI_API_KEY",
      "base_url_env": "OPENAI_BASE_URL"
    },
    "verifier": {
      "model": "gpt-4",
      "temperature": 0.0,
      "api_key_env": "OPENAI_API_KEY",
      "base_url_env": "OPENAI_BASE_URL"
    }
  },
  "similarity_threshold": 0.85,
  "max_refinement_iterations": 3,
  "example_retrieval": {
    "enabled": true,
    "top_k": 3,
    "examples_path": "data/examples/dsv_examples.json"
  }
}
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `similarity_threshold` | Minimum similarity score for verification | 0.85 |
| `max_refinement_iterations` | Maximum number of refinement attempts | 3 |
| `example_retrieval.enabled` | Enable dynamic example retrieval | true |
| `example_retrieval.top_k` | Number of examples to retrieve | 3 |
| `agents.*.model` | LLM model for each agent | gpt-4 |
| `agents.*.temperature` | Temperature for LLM generation | 0.0 |

---

## 📊 MTL Knowledge Base

Both framework versions include a comprehensive MTL knowledge base that standardizes:

### Temporal Operators

**Future-time:**
- `F_[a,b](φ)` — Eventually (φ occurs within [a,b])
- `G_[a,b](φ)` — Globally (φ holds throughout [a,b])
- `φ U_[a,b] ψ` — Until (φ holds until ψ within [a,b])
- `X` — Next (discrete next step)

**Past-time:**
- `P_[a,b](φ)` — Previously (φ held within past [a,b])
- `O(φ)` — Once (φ occurred at least once in the past)

### Logical Connectives

- `&` (and), `|` (or), `~` (not), `->` (implication), `<->` (equivalence)

### Common Mappings

| Natural Language | MTL Formula |
|------------------|-------------|
| "within T seconds" | `F_[0,T](φ)` |
| "for T seconds" | `G_[0,T](φ)` |
| "after at least T seconds" | `F_[T,∞)(φ)` |
| "always" | `G(φ)` |
| "eventually" | `F(φ)` |
| "immediately" | `X(φ)` |

---

## 🔄 Intelligent Refinement Mechanism

### The Problem (Before)

Original implementation had a critical flaw:
- Each refinement iteration used identical input
- No learning from previous failures
- Essentially "random retries" without improvement

### The Solution (Now)

**Feedback-Driven Refinement Loop:**

```
Iteration 1: 
  Input: "Within 5 seconds after A, B must occur"
  → Sketch₁ → Formula₁ → Verify (similarity: 0.45) ❌
  → Analyze: "Time constraint [5,5] should be [0,5]"

Iteration 2:
  Input: Original + [Analysis from Iteration 1]
  → Sketch₂ (corrected) → Formula₂ → Verify (similarity: 0.78) ❌
  → Analyze: "Missing 'must' implication operator"

Iteration 3:
  Input: Original + [Analysis from Iterations 1 & 2]
  → Sketch₃ (fully corrected) → Formula₃ → Verify (similarity: 0.92) ✅
```

### Key Improvements

1. **Failure Analysis** ([`_analyze_verification_failure()`](dsv_framework_ablation.py:195))
   - Uses Analyst Agent to diagnose discrepancies
   - Identifies specific semantic/temporal errors
   - Provides actionable correction suggestions

2. **Feedback Accumulation** ([`RefinementFeedback`](dsv_framework_ablation.py:73))
   - Stores each iteration's: formula, back-translation, similarity, analysis
   - Prevents repeating the same mistakes

3. **Context-Aware Agents**
   - Analyst and Synthesizer receive full refinement history
   - Learn from previous attempts
   - Continuously improve output quality

---

## 📈 Usage Examples

### Example 1: Basic Processing

```python
from dsv_framework_complete import EnhancedDSVFramework

dsv = EnhancedDSVFramework()
result = dsv.process("The door must remain locked for 30 seconds after the alarm.")

print(f"MTL Formula: {result.final_mtl_formula}")
# Output: G(alarm -> G_[0,30](door_locked))
```

### Example 2: Accessing Stage Details

```python
result = dsv.process("If signal received, respond within 10 seconds.")

for stage_result in result.stage_results:
    print(f"\nStage: {stage_result.stage.value}")
    print(f"Success: {stage_result.success}")
    print(f"Time: {stage_result.processing_time:.2f}s")
    print(f"Tokens: {stage_result.token_usage.total_tokens}")
```

### Example 3: Analyzing Refinement Process

```python
result = dsv.process(complex_sentence, enable_refinement=True)

print(f"Total Iterations: {result.refinement_iterations + 1}")
print(f"Final Success: {result.success}")
print(f"Termination: {result.termination_reason}")

# Check similarity progression
verify_stages = [s for s in result.stage_results if s.stage.value == "verify"]
for i, stage in enumerate(verify_stages, 1):
    sim = stage.stage_output.similarity_score
    print(f"Iteration {i} similarity: {sim:.3f}")
```

### Example 4: Comparing Enhanced vs Ablation

```python
from dsv_framework_complete import EnhancedDSVFramework
from dsv_framework_ablation import DSVFrameworkAblation

sentence = "Within 5 seconds after A, B must occur."

# Enhanced version (with examples)
enhanced = EnhancedDSVFramework()
result_enhanced = enhanced.process(sentence)

# Ablation version (without examples)
ablation = DSVFrameworkAblation()
result_ablation = ablation.process(sentence)

print("Enhanced:")
print(f"  Success: {result_enhanced.success}")
print(f"  Tokens: {result_enhanced.total_token_usage.total_tokens}")

print("Ablation:")
print(f"  Success: {result_ablation.success}")
print(f"  Tokens: {result_ablation.total_token_usage.total_tokens}")
```

---

## 📝 Output Format

Results are saved as JSON files with comprehensive metadata:

```json
{
  "framework": "Enhanced DSV with Dynamic Examples",
  "input_sentence": "Within 5 seconds...",
  "final_mtl_formula": "G(A -> F_[0,5](B))",
  "success": true,
  "termination_reason": "Verification passed (similarity: 0.923)",
  "total_processing_time": 12.34,
  "total_token_usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801
  },
  "refinement_iterations": 1,
  "stage_results": [
    {
      "stage": "deconstruct",
      "success": true,
      "processing_time": 3.45,
      "semantic_sketch": {...},
      "agent_response": "..."
    },
    {
      "stage": "synthesize",
      "success": true,
      "synthesis_result": {
        "mtl_formula": "G(A -> F_[0,5](B))",
        "synthesis_reasoning": "..."
      }
    },
    {
      "stage": "verify",
      "success": true,
      "verification_result": {
        "back_translation": "Always, when A occurs, B must occur within 5 seconds",
        "similarity_score": 0.923,
        "verification_passed": true
      }
    }
  ],
  "config": {...},
  "timestamp": "2025-01-10 13:45:30"
}
```

---

## 🧪 Ablation Studies

The framework provides two versions specifically designed for ablation studies:

### Purpose

Compare performance with and without dynamic example retrieval to measure the impact of few-shot learning on:
- Translation accuracy (similarity scores)
- Refinement iteration count
- Token efficiency
- Processing time

### Running Ablation Experiments

```python
from dsv_framework_complete import EnhancedDSVFramework
from dsv_framework_ablation import DSVFrameworkAblation
import json

# Test dataset
test_cases = [
    "Within 5 seconds after A, B must occur.",
    "The system must respond within 10 to 20 seconds.",
    "Always, if X happens, Y must follow immediately."
]

# Run enhanced version
enhanced_dsv = EnhancedDSVFramework()
enhanced_results = []
for sentence in test_cases:
    result = enhanced_dsv.process(sentence)
    enhanced_results.append({
        "sentence": sentence,
        "success": result.success,
        "similarity": result.stage_results[-1].stage_output.similarity_score,
        "iterations": result.refinement_iterations,
        "tokens": result.total_token_usage.total_tokens
    })

# Run ablation version
ablation_dsv = DSVFrameworkAblation()
ablation_results = []
for sentence in test_cases:
    result = ablation_dsv.process(sentence)
    ablation_results.append({
        "sentence": sentence,
        "success": result.success,
        "similarity": result.stage_results[-1].stage_output.similarity_score,
        "iterations": result.refinement_iterations,
        "tokens": result.total_token_usage.total_tokens
    })

# Compare results
print("Enhanced Version:")
print(json.dumps(enhanced_results, indent=2))
print("\nAblation Version:")
print(json.dumps(ablation_results, indent=2))
```

### Expected Differences

| Metric | Enhanced (with examples) | Ablation (baseline) |
|--------|--------------------------|---------------------|
| Similarity Score | Higher (0.85-0.95) | Lower (0.75-0.90) |
| Success Rate | Higher | Lower |
| Refinement Iterations | Fewer (0-1) | More (1-3) |
| Token Usage | Higher (examples overhead) | Lower |
| Processing Time | Slightly higher | Slightly lower |

---

## 🔍 Troubleshooting

### Common Issues

#### 1. **API Authentication Error**

```
Error: Invalid API key
```

**Solution**: Ensure your `.env` file exists and contains valid credentials:

```bash
# Check if .env exists
ls -la .env

# Verify format
cat .env
# Should show:
# OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://...
```

#### 2. **Low Similarity Scores**

```
Verification failed: similarity 0.65 below threshold 0.85
```

**Solutions**:
- Check if input sentence is grammatically correct
- Ensure temporal constraints are clearly expressed
- Try adjusting `similarity_threshold` in [`config/dsv_config.json`](config/dsv_config.json)
- Enable refinement: `result = dsv.process(sentence, enable_refinement=True)`

#### 3. **Example Retrieval Errors**

```
FileNotFoundError: data/examples/dsv_examples.json
```

**Solution**: Verify example file path in [`config/dsv_config.json`](config/dsv_config.json):

```json
{
  "example_retrieval": {
    "examples_path": "data/examples/dsv_examples.json"
  }
}
```

#### 4. **High Token Usage**

**Solutions**:
- Reduce `top_k` in example retrieval (default: 3 → 1 or 2)
- Use ablation version if examples are unnecessary
- Switch to a more efficient model (gpt-3.5-turbo)

---

## 📚 API Reference

### EnhancedDSVFramework

```python
class EnhancedDSVFramework:
    def __init__(self, config_path: str = "config/dsv_config.json")
    def process(self, sentence: str, enable_refinement: bool = True) -> DSVResult
    def toggle_examples(self, enabled: bool) -> None
    def save_result(self, result: DSVResult, filepath: str) -> None
```

### DSVFrameworkAblation

```python
class DSVFrameworkAblation:
    def __init__(self, config_path: str = "config/dsv_config.json")
    def process(self, sentence: str, enable_refinement: bool = True) -> DSVResult
    def save_result(self, result: DSVResult, filepath: str) -> None
```

### Data Structures

```python
@dataclass
class SemanticSpecificationSketch:
    """Structured semantic components from natural language"""
    events_objects: Dict[str, str]
    time_constraints: Dict[str, Any]
    temporal_relations: Dict[str, Any]
    logical_conditions: Dict[str, Any]
    quality_constraints: Dict[str, Any]

@dataclass
class SynthesisResult:
    """MTL formula synthesis output"""
    mtl_formula: str
    synthesis_reasoning: str

@dataclass
class VerificationResult:
    """Back-translation verification output"""
    back_translation: str
    similarity_score: float
    verification_passed: bool
    verification_reasoning: str

@dataclass
class RefinementFeedback:
    """Feedback from failed verification iterations"""
    iteration: int
    mtl_formula: str
    back_translation: str
    similarity_score: float
    semantic_sketch_json: str
    issue_analysis: str
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Multi-language support**: Extend beyond English specifications
2. **Additional temporal logics**: STL, LTL, CTL support
3. **Improved example selection**: Better semantic similarity metrics
4. **Performance optimization**: Caching, parallel processing
5. **Enhanced verification**: Multi-metric validation beyond similarity

---

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

---

## 📧 Contact

For questions, issues, or contributions:

- **GitHub Issues**: [Submit an issue](https://github.com/your-repo/issues)
- **Email**: your-email@example.com

---

## 🙏 Acknowledgments

This framework builds upon research in:
- Natural language processing for formal specifications
- Multi-agent systems for complex reasoning
- Metric Temporal Logic formalization
- Few-shot learning with LLMs

---

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{dsv_framework,
  title={DSV Framework: Deconstruct-Synthesize-Verify for NL to MTL Translation},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/dsv-framework}
}
```

---

## ⚡ Quick Reference Card

### Setup Checklist

- [ ] Install Python 3.8+
- [ ] Run `pip install -r requirements.txt`
- [ ] **Create `.env` file with API credentials**
- [ ] Verify `config/dsv_config.json` exists
- [ ] Test with: `python dsv_framework_complete.py`

### Essential Commands

```bash
# Quick test (enhanced version)
python dsv_framework_complete.py

# Quick test (ablation version)
python dsv_framework_ablation.py

# Custom processing
python -c "
from dsv_framework_complete import EnhancedDSVFramework
dsv = EnhancedDSVFramework()
result = dsv.process('Your sentence here')
print(f'Success: {result.success}')
print(f'Formula: {result.final_mtl_formula}')
"
```

### File Checklist

**Must Have:**
- ✅ `.env` (API credentials) - **YOU MUST CREATE THIS**
- ✅ `config/dsv_config.json` (configuration)
- ✅ `requirements.txt` (dependencies)

**Framework Files:**
- ✅ `dsv_framework_complete.py` (enhanced version)
- ✅ `dsv_framework_ablation.py` (baseline version)
- ✅ `retrieval.py` (example retrieval system)

**Data Files:**
- ✅ `data/examples/dsv_examples.json` (example dataset)
- 📁 `data/output/dsv_enhanced/` (enhanced outputs)
- 📁 `data/output/dsv_ablation/` (ablation outputs)

---

**Remember**: Always create your `.env` file before running the framework! 🔑
