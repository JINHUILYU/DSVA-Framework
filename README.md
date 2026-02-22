# DSVA Framework: Natural Language to Temporal Logic Translation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

A sophisticated multi-agent framework for translating natural language specifications into formal temporal logic formulas using a **Deconstruct → Synthesize → Verify → Analyze** pipeline.

---

## 🎯 Overview

The DSVA Framework employs a four-stage approach with intelligent refinement feedback:

1. **Deconstruct**: Semantic analysis agent breaks down natural language into structured components
2. **Synthesize**: Formula synthesizer generates formal temporal logic from semantic sketches
3. **Verify**: Back-translation verifier validates correctness through semantic similarity
4. **Analyze**: Error analyst diagnoses failures and provides targeted refinement feedback

### Key Features

- ✅ **Intelligent Refinement Loop**: Iterative improvement with error analysis
- ✅ **Dynamic Example Retrieval**: Semantic similarity-based few-shot learning
- ✅ **Multi-Agent Architecture**: Specialized agents for each pipeline stage
- ✅ **Multi-LLM Support**: GPT-4, GPT-4o, DeepSeek-v3, DeepSeek-r1, Gemini 2.5 Flash
- ✅ **Comprehensive Tracking**: Token usage, processing time, refinement history
- ✅ **Ablation Study Support**: Baseline version for performance comparison

---

## 📚 Available Implementations

This repository contains two complete implementations of the DSVA framework for different temporal logic targets:

### 🔷 [MTL Implementation (`nl2mtl` branch)](../../tree/nl2mtl)
**Natural Language → Metric Temporal Logic (MTL)**

- Time-bounded temporal operators: `F_[a,b]`, `G_[a,b]`, `U_[a,b]`
- Complex object-predicate semantic structures
- Precise timing constraints (e.g., "within 3 seconds", "for at least 5 steps")
- Suitable for real-time systems and safety-critical applications

**Use Case**: Autonomous driving, robotics specifications with timing requirements

**Example**:
```
Input:  "The ego vehicle must stop within 3 seconds when detecting a pedestrian"
Output: F_[0,3](pedestrian_detected -> stop)
```

```bash
git checkout nl2mtl
```

### 🔶 [LTL Implementation (`nl2ltl` branch)](../../tree/nl2ltl)
**Natural Language → Linear Temporal Logic (LTL)**

- Standard temporal operators: `G`, `F`, `X`, `U`, `R`
- Simple atomic propositions
- Event ordering and causality (without explicit time bounds)
- Based on nl2spec benchmark dataset

**Use Case**: Protocol verification, reactive systems, general temporal properties

**Example**:
```
Input:  "Whenever a holds, b must eventually hold"
Output: G(a -> F(b))
```

```bash
git checkout nl2ltl
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Natural Language Input                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: DECONSTRUCT                                       │
│  └─ Semantic Analyst: Extract structured components         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: SYNTHESIZE                                        │
│  └─ Formula Synthesizer: Generate temporal logic formula    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: VERIFY                                            │
│  └─ Verifier: Back-translate and compute similarity         │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
    [Pass: ≥ 0.70]          [Fail: < 0.70]
           │                       │
           │                       ▼
           │          ┌─────────────────────────────┐
           │          │  Stage 4: ANALYZE           │
           │          │  └─ Error Analyst: Diagnose │
           │          │     and provide feedback    │
           │          └────────────┬────────────────┘
           │                       │
           │                       │ Refinement Feedback
           │                       │ (Max 3 iterations)
           │                       │
           │          ┌────────────▼─────────────────┐
           │          │  Return to Stage 1           │
           │          │  with feedback context       │
           │          └──────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────┐
    │     Final Temporal Logic Formula        │
    └─────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/JINHUILYU/DSVA-Framework.git
cd DSVA-Framework
```

### 2. Choose Your Implementation

#### For MTL (with timing constraints):
```bash
git checkout nl2mtl
```

#### For LTL (standard temporal logic):
```bash
git checkout nl2ltl
```

### 3. Follow Branch-Specific Instructions

Each branch contains its own detailed README with:
- Installation instructions
- Configuration guide
- Usage examples
- Dataset information
- Evaluation results

---

## 📊 Comparison: MTL vs LTL

| Feature | MTL (`nl2mtl`) | LTL (`nl2ltl`) |
|---------|----------------|----------------|
| **Temporal Operators** | `F_[a,b]`, `G_[a,b]`, `U_[a,b]`, `P_[a,b]`, `O` | `G`, `F`, `X`, `U`, `R` |
| **Time Constraints** | ✅ Explicit time bounds | ❌ Event ordering only |
| **Semantic Structure** | Complex (object-predicate) | Simple (atomic propositions) |
| **Use Cases** | Real-time systems, safety specs | Protocol verification, reactive systems |
| **Example Dataset** | Custom traffic/robotics | nl2spec benchmark |
| **Formula Complexity** | Higher (nested intervals) | Lower (standard operators) |

---

## 📖 Documentation

- **MTL Branch**: See [`nl2mtl` README](../../tree/nl2mtl/README.md) for detailed MTL implementation
- **LTL Branch**: See [`nl2ltl` README](../../tree/nl2ltl/README.md) for detailed LTL implementation

---

## 🛠️ Technology Stack

- **Language**: Python 3.8+
- **LLMs**: GPT-4, GPT-4o, DeepSeek-v3, DeepSeek-r1, Gemini 2.5 Flash
- **Key Libraries**:
  - `openai` - LLM API integration
  - `sentence-transformers` - Semantic similarity and example retrieval
  - `python-dotenv` - Environment configuration
  - `pandas` - Data processing

---

## 🌟 Acknowledgments

- nl2spec dataset for LTL benchmarking
- TR2MTL dataset for MTL benchmarking
- OpenAI, DeepSeek, and Google for LLM APIs
- sentence-transformers for semantic similarity models
