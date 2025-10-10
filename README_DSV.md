# DSV Framework - 自然语言到MTL转换系统

## 🏗️ 架构概述

DSV框架实现了一种创新的**解构-合成-验证**流程，将自然语言规约转换为度量时序逻辑（MTL）公式。

### 核心思想

通过三个专门的LLM智能体协同工作：
1. **分析师Agent** - 语义解构专家
2. **合成师Agent** - 形式化语言编码器  
3. **验证师Agent** - 独立的第三方审计员

## 📁 项目结构

```
DSV-Framework/
├── dsv_framework_complete.py      # 🚀 完整增强版（含动态示例检索）
├── dsv_framework_ablation.py      # 🚫 消融实验版（基础DSV框架）
├── dsv_framework_enhanced.py      # 📚 原有增强版本
├── retrieval.py                   # 🔍 示例检索系统
├── test_dsv_framework.py          # 🧪 测试套件
├── config/
│   └── dsv_config.json           # ⚙️ 配置文件
├── data/
│   └── examples/
│       └── dsv_examples.json     # 📋 示例数据库
└── README.md                     # 📖 本文档
```

## 🌟 核心特性

### 1. 三阶段处理流程

#### 阶段一：语义解构 (Deconstruct)
- **输入**：原始自然语言句子
- **Agent**：分析师Agent
- **输出**：结构化语义规约草图（JSON格式）
- **功能**：提取原子命题、时序关系、度量约束等

#### 阶段二：约束下的合成 (Synthesize) 
- **输入**：语义规约草图
- **Agent**：合成师Agent
- **输出**：候选MTL公式
- **功能**：严格按照草图生成语法正确的MTL公式

#### 阶段三：循环验证与修正 (Verify)
- **输入**：MTL公式 + 词汇表
- **Agent**：验证师Agent
- **输出**：自然语言反向翻译
- **功能**：验证语义保真度，支持修正循环

### 2. 动态增强生成 🚀

**增强版框架** (`dsv_framework_complete.py`) 包含：
- 基于语义相似度的示例检索
- 为每个Agent动态注入相关示例
- Top-K相似示例选择
- 智能prompt构建

**消融实验版** (`dsv_framework_ablation.py`) 提供：
- 纯DSV基础功能
- 无示例增强
- 用于对比实验

### 3. 智能验证与修正循环

- 语义相似度计算（Sentence-BERT）
- 可配置相似度阈值
- 最大修正迭代次数限制
- 详细的失败分析

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置API密钥（.env文件）
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_API_URL=https://api.deepseek.com
```

### 2. 运行测试

```bash
# 运行完整测试套件
python test_dsv_framework.py
```

### 3. 使用示例

#### 增强版框架（推荐）

```python
from dsv_framework_complete import EnhancedDSVFramework

# 创建增强版DSV框架
dsv = EnhancedDSVFramework()

# 处理自然语言句子
sentence = "在传感器A检测到故障后的5到10秒内，警报B必须响起，并持续至少20秒。"
result = dsv.process(sentence, enable_refinement=True)

print(f"成功: {result.success}")
print(f"MTL公式: {result.final_mtl_formula}")
print(f"处理时间: {result.total_processing_time:.2f}s")
print(f"动态增强: {'启用' if dsv.examples_enabled else '禁用'}")
```

#### 消融实验版本

```python
from dsv_framework_ablation import DSVFrameworkAblation

# 创建基础DSV框架（无增强）
dsv_ablation = DSVFrameworkAblation()

# 处理相同句子进行对比
result = dsv_ablation.process(sentence, enable_refinement=True)
```

## ⚙️ 配置说明

### 主要配置项 (config/dsv_config.json)

```json
{
  "agents": {
    "analyst": { 
      "model": "deepseek-chat",
      "temperature": 0.3 
    },
    "synthesizer": { 
      "model": "deepseek-chat", 
      "temperature": 0.1 
    },
    "verifier": { 
      "model": "deepseek-chat",
      "temperature": 0.2 
    }
  },
  "similarity_threshold": 0.9,
  "max_refinement_iterations": 3,
  "example_retrieval": {
    "enabled": true,
    "top_k": 3,
    "similarity_threshold": 0.3
  }
}
```

## 📊 消融实验

框架支持完整的消融实验，对比有无动态增强的性能差异：

### 实验设计
- **对照组**：基础DSV框架（无示例增强）
- **实验组**：增强版DSV框架（含动态示例检索）
- **评估指标**：准确率、处理时间、Token使用量、修正迭代次数

### 运行消融实验

```python
# 测试脚本会自动进行对比实验
python test_dsv_framework.py
```

## 📋 输出格式

### 处理结果结构

```json
{
  "framework": "Enhanced DSV with Dynamic Examples",
  "input_sentence": "原始输入句子",
  "final_mtl_formula": "G (fault_A → F_[5,10] (alarm_B_on ∧ G_[0,20] alarm_B_on))",
  "success": true,
  "total_processing_time": 15.42,
  "total_token_usage": {
    "prompt_tokens": 1250,
    "completion_tokens": 380,
    "total_tokens": 1630
  },
  "refinement_iterations": 0,
  "stage_results": [
    {
      "stage": "deconstruct",
      "success": true,
      "processing_time": 5.2,
      "semantic_sketch": { ... }
    },
    {
      "stage": "synthesize", 
      "success": true,
      "processing_time": 4.1,
      "synthesis_result": { ... }
    },
    {
      "stage": "verify",
      "success": true,
      "processing_time": 6.12,
      "verification_result": { ... }
    }
  ]
}
```

## 🔍 示例检索系统

### 示例数据结构

```json
{
  "examples": {
    "deconstruct": [
      {
        "id": "dec_001",
        "input_text": "传感器故障后警报响起",
        "output": "{ structured JSON }",
        "reasoning": "解构推理过程"
      }
    ],
    "synthesize": [ ... ],
    "verify": [ ... ]
  }
}
```

### 检索机制
1. 基于Sentence-BERT计算语义相似度
2. 返回Top-K个最相关示例
3. 动态构建包含示例的prompt
4. 指导Agent进行任务相关的推理

## 🎯 架构优势

### 1. 模块化设计
- 三个Agent职责分离，便于调试和优化
- 每个阶段独立评估和改进

### 2. 可解释性
- 详细的中间产物记录
- 每个阶段的推理过程可追踪
- 语义草图提供可视化理解

### 3. 鲁棒性
- 修正循环机制处理初次失败
- 语义相似度验证确保质量
- 可配置的容错参数

### 4. 动态增强
- 上下文感知的示例选择
- 提升Agent的任务理解能力
- 支持增量学习和改进

## 🧪 测试与验证

### 测试用例类型
1. **基础功能测试**：验证三阶段流程
2. **增强功能测试**：验证示例检索和注入
3. **消融对比测试**：对比有无增强的性能
4. **边界条件测试**：处理复杂和边缘情况

### 评估指标
- **准确性**：MTL公式的正确性
- **效率**：处理时间和资源消耗
- **稳定性**：不同输入的一致性表现
- **可扩展性**：处理复杂场景的能力

## 🔮 未来扩展

1. **多语言支持**：扩展到英文等其他语言
2. **领域适应**：针对特定领域的优化
3. **交互式修正**：人机协作的修正机制
4. **性能优化**：并行处理和缓存机制
5. **可视化界面**：友好的用户交互界面

## 📞 联系与支持

如有问题或需要支持，请通过以下方式联系：

- **GitHub Issues**：项目问题和功能请求
- **文档**：详细的API文档和使用指南
- **测试**：运行 `python test_dsv_framework.py` 验证安装

---

**DSV Framework** - 让自然语言到MTL的转换更智能、更准确、更可靠！ 🚀