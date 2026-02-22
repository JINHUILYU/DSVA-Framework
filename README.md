# DSVA Framework: Deconstruct-Synthesize-Verify-Analyze

A sophisticated framework for translating natural language specifications into Metric Temporal Logic (MTL) formulas using a four-stage pipeline with intelligent refinement feedback and error analysis.

## 🌟 Overview

The DSVA Framework employs a **Deconstruct → Synthesize → Verify → Analyze** approach to convert natural language requirements into formal MTL formulas:

1. **Deconstruct**: Semantic analysis agent breaks down natural language into structured components
2. **Synthesize**: MTL formula synthesizer generates formal logic from semantic sketches  
3. **Verify**: Back-translation verifier validates formula correctness through semantic similarity
4. **Analyze**: Error analyst function diagnoses verification failures and provides targeted feedback for refinement

### Key Features

✅ **Intelligent Refinement Loop**: Learns from verification failures with detailed feedback analysis  
✅ **Error Analyst Function**: Advanced failure diagnosis that identifies semantic gaps and provides actionable correction suggestions  
✅ **MTL Knowledge Base Integration**: Standardized temporal logic operators and mappings  
✅ **Dynamic Example Retrieval**: Semantic similarity-based few-shot learning for improved accuracy  
✅ **Multi-Agent Architecture**: Specialized agents for each DSVA stage  
✅ **Multi-LLM Support**: Tested with GPT-4, GPT-4o, DeepSeek-v3, and Gemini 2.5 Flash  
✅ **Comprehensive Tracking**: Token usage, processing time, and refinement history  
✅ **Ablation Study Support**: Built-in baseline version for performance comparison  

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/JINHUILYU/DSVA-Framework.git
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

```bash
# Enhanced DSVA framework with dynamic examples
python dsva.py

# Ablation version (baseline without examples)
python ablation.py
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

## 🙏 Acknowledgments

This framework builds upon research in:
- Natural language processing for formal specifications
- Multi-agent systems for complex reasoning
- Metric Temporal Logic formalization
- Few-shot learning with LLMs
