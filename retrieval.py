"""
Example Retrieval System for DSV Framework
基于相似度的示例检索系统，为每个agent提供top-5相似示例
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class Example:
    """示例数据结构"""
    id: str
    input_text: str
    stage: str  # 'deconstruct', 'synthesize', 'verify', 'all'
    output: str
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RetrievalResult:
    """检索结果"""
    examples: List[Example]
    similarities: List[float]
    query: str
    stage: str

class ExampleRetriever:
    """示例检索器"""
    
    def __init__(self, config_path: str = "config/dsv_config.json"):
        """初始化示例检索器"""
        self.config = self._load_config(config_path)
        self.sentence_model = None
        self.examples_db = {}  # stage -> List[Example]
        self.embeddings_cache = {}  # stage -> embeddings
        
        # 获取示例配置
        self.example_config = self.config.get("example_retrieval", {})
        self.enabled = self.example_config.get("enabled", True)
        self.top_k = self.example_config.get("top_k", 5)
        self.similarity_threshold = self.example_config.get("similarity_threshold", 0.3)
        
        if self.enabled:
            self._initialize_components()
        
        logger.info(f"示例检索器初始化完成，启用状态: {self.enabled}")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 未找到")
            return {}
    
    def _initialize_components(self):
        """初始化组件"""
        try:
            # 初始化句子模型
            model_name = self.example_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
            self.sentence_model = SentenceTransformer(model_name)
            
            # 加载示例数据
            self._load_examples()
            
            # 预计算嵌入
            self._precompute_embeddings()
            
            logger.info("示例检索组件初始化成功")
            
        except Exception as e:
            logger.error(f"示例检索组件初始化失败: {e}")
            self.enabled = False
    
    def _load_examples(self):
        """加载示例数据"""
        example_sources = self.example_config.get("sources", [])
        
        for source in example_sources:
            try:
                source_path = source.get("path")
                source_type = source.get("type", "json")
                stages = source.get("stages", ["all"])
                
                if source_type == "json":
                    self._load_json_examples(source_path, stages)
                elif source_type == "excel":
                    self._load_excel_examples(source_path, stages)
                elif source_type == "csv":
                    self._load_csv_examples(source_path, stages)
                    
            except Exception as e:
                logger.error(f"加载示例源失败 {source.get('path')}: {e}")
        
        total_examples = sum(len(examples) for examples in self.examples_db.values())
        logger.info(f"共加载 {total_examples} 个示例，覆盖阶段: {list(self.examples_db.keys())}")
    
    def _load_json_examples(self, file_path: str, stages: List[str]):
        """从JSON文件加载示例"""
        if not os.path.exists(file_path):
            logger.warning(f"示例文件不存在: {file_path}")
            return
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both flat list format and categorized format
        if "examples" in data and isinstance(data["examples"], dict):
            # Categorized format: {examples: {stage: [examples]}}
            for stage_name, stage_examples in data["examples"].items():
                if stage_name in stages:
                    for example_data in stage_examples:
                        example = Example(
                            id=example_data.get("id", ""),
                            input_text=example_data.get("input_text", ""),
                            stage=example_data.get("stage", stage_name),
                            output=example_data.get("output", ""),
                            reasoning=example_data.get("reasoning"),
                            metadata=example_data.get("metadata")
                        )
                        
                        if stage_name not in self.examples_db:
                            self.examples_db[stage_name] = []
                        self.examples_db[stage_name].append(example)
        elif "examples" in data and isinstance(data["examples"], list):
            # Flat list format: {examples: [examples]}
            examples = data.get("examples", [])
            for example_data in examples:
                example = Example(
                    id=example_data.get("id", ""),
                    input_text=example_data.get("input_text", ""),
                    stage=example_data.get("stage", "all"),
                    output=example_data.get("output", ""),
                    reasoning=example_data.get("reasoning"),
                    metadata=example_data.get("metadata")
                )
                
                # 添加到对应阶段
                for stage in stages:
                    if stage not in self.examples_db:
                        self.examples_db[stage] = []
                    self.examples_db[stage].append(example)
    
    def _load_excel_examples(self, file_path: str, stages: List[str]):
        """从Excel文件加载示例"""
        if not os.path.exists(file_path):
            logger.warning(f"示例文件不存在: {file_path}")
            return
            
        df = pd.read_excel(file_path)
        
        for idx, row in df.iterrows():
            example = Example(
                id=str(row.get("id", idx)),
                input_text=str(row.get("input_text", "")),
                stage=str(row.get("stage", "all")),
                output=str(row.get("output", "")),
                reasoning=str(row.get("reasoning", "")) if pd.notna(row.get("reasoning")) else None,
                metadata={"source_row": idx}
            )
            
            # 添加到对应阶段
            for stage in stages:
                if stage not in self.examples_db:
                    self.examples_db[stage] = []
                self.examples_db[stage].append(example)
    
    def _load_csv_examples(self, file_path: str, stages: List[str]):
        """从CSV文件加载示例"""
        if not os.path.exists(file_path):
            logger.warning(f"示例文件不存在: {file_path}")
            return
            
        df = pd.read_csv(file_path)
        
        for idx, row in df.iterrows():
            example = Example(
                id=str(row.get("id", idx)),
                input_text=str(row.get("input_text", "")),
                stage=str(row.get("stage", "all")),
                output=str(row.get("output", "")),
                reasoning=str(row.get("reasoning", "")) if pd.notna(row.get("reasoning")) else None,
                metadata={"source_row": idx}
            )
            
            # 添加到对应阶段
            for stage in stages:
                if stage not in self.examples_db:
                    self.examples_db[stage] = []
                self.examples_db[stage].append(example)
    
    def _precompute_embeddings(self):
        """预计算所有示例的嵌入"""
        if not self.sentence_model:
            return
            
        for stage, examples in self.examples_db.items():
            if not examples:
                continue
                
            try:
                # 提取输入文本
                input_texts = [example.input_text for example in examples]
                
                # 计算嵌入
                embeddings = self.sentence_model.encode(input_texts, convert_to_tensor=True)
                self.embeddings_cache[stage] = embeddings
                
                logger.info(f"为阶段 {stage} 预计算了 {len(examples)} 个示例的嵌入")
                
            except Exception as e:
                logger.error(f"预计算嵌入失败 {stage}: {e}")
    
    def retrieve_examples(self, query: str, stage: str, top_k: Optional[int] = None) -> RetrievalResult:
        """检索相似示例"""
        if not self.enabled:
            return RetrievalResult([], [], query, stage)
        
        top_k = top_k or self.top_k
        
        # 检查是否有该阶段的示例
        if stage not in self.examples_db or not self.examples_db[stage]:
            logger.warning(f"阶段 {stage} 没有可用示例")
            return RetrievalResult([], [], query, stage)
        
        try:
            # 编码查询
            query_embedding = self.sentence_model.encode([query], convert_to_tensor=True)
            
            # 获取该阶段的嵌入
            stage_embeddings = self.embeddings_cache.get(stage)
            if stage_embeddings is None:
                logger.error(f"阶段 {stage} 的嵌入未找到")
                return RetrievalResult([], [], query, stage)
            
            # 计算相似度
            similarities = util.pytorch_cos_sim(query_embedding, stage_embeddings)[0]
            similarities = similarities.cpu().numpy()
            
            # 获取top-k索引
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # 直接提取top-k个示例，不使用阈值过滤
            top_examples = []
            top_similarities = []
            
            for idx in top_indices:
                top_examples.append(self.examples_db[stage][idx])
                top_similarities.append(float(similarities[idx]))
            
            logger.info(f"为阶段 {stage} 检索到 top-{len(top_examples)} 个相似示例")
            
            return RetrievalResult(
                examples=top_examples,
                similarities=top_similarities,
                query=query,
                stage=stage
            )
            
        except Exception as e:
            logger.error(f"示例检索失败: {e}")
            return RetrievalResult([], [], query, stage)
    
    def format_examples_for_prompt(self, retrieval_result: RetrievalResult) -> str:
        """将检索结果格式化为prompt文本"""
        if not retrieval_result.examples:
            return ""
        
        formatted_text = "## 相似示例参考\n\n"
        formatted_text += "以下是一些相似的处理示例，供您参考：\n\n"
        
        for i, (example, similarity) in enumerate(zip(retrieval_result.examples, retrieval_result.similarities), 1):
            formatted_text += f"### 示例 {i} (相似度: {similarity:.3f})\n\n"
            formatted_text += f"**输入**: {example.input_text}\n\n"
            formatted_text += f"**输出**: {example.output}\n\n"
            
            if example.reasoning:
                formatted_text += f"**推理过程**: {example.reasoning}\n\n"
            
            formatted_text += "---\n\n"
        
        formatted_text += "请参考以上示例的处理方式，但要根据当前输入的具体情况进行分析。\n\n"
        
        return formatted_text
    
    def add_example(self, example: Example):
        """添加新示例"""
        stage = example.stage
        if stage not in self.examples_db:
            self.examples_db[stage] = []
        
        self.examples_db[stage].append(example)
        
        # 重新计算该阶段的嵌入
        if self.sentence_model and stage in self.examples_db:
            try:
                input_texts = [ex.input_text for ex in self.examples_db[stage]]
                embeddings = self.sentence_model.encode(input_texts, convert_to_tensor=True)
                self.embeddings_cache[stage] = embeddings
                logger.info(f"为阶段 {stage} 添加了新示例并更新了嵌入")
            except Exception as e:
                logger.error(f"更新嵌入失败: {e}")
    
    def save_examples(self, output_path: str):
        """保存示例数据库"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            "config": self.example_config,
            "examples": {}
        }
        
        for stage, examples in self.examples_db.items():
            save_data["examples"][stage] = [
                {
                    "id": ex.id,
                    "input_text": ex.input_text,
                    "stage": ex.stage,
                    "output": ex.output,
                    "reasoning": ex.reasoning,
                    "metadata": ex.metadata
                }
                for ex in examples
            ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"示例数据库已保存到: {output_path}")

def create_default_examples():
    """创建默认示例数据"""
    examples = {
        "deconstruct": [
            {
                "id": "dec_001",
                "input_text": "在传感器A检测到故障后的5到10秒内，警报B必须响起，并持续至少20秒。",
                "stage": "deconstruct",
                "output": """{
    "atomic_propositions": [
        {
            "id": "ap_1",
            "description": "传感器A检测到故障",
            "variable": "fault_A"
        },
        {
            "id": "ap_2",
            "description": "警报B响起",
            "variable": "alarm_B_on"
        }
    ],
    "temporal_relations": [
        {
            "type": "after",
            "antecedent": "ap_1",
            "consequent": "ap_2",
            "description": "ap_2必须在ap_1之后发生"
        }
    ],
    "metric_constraints": [
        {
            "applies_to": "relation_between_ap1_ap2",
            "type": "window",
            "value": "[5, 10]",
            "description": "时间窗口为5到10秒"
        },
        {
            "applies_to": "ap_2",
            "type": "duration",
            "value": ">=20",
            "description": "持续时间至少20秒"
        }
    ],
    "global_property": "Always"
}""",
                "reasoning": "首先识别两个原子命题：传感器A检测到故障和警报B响起。然后确定时序关系：警报B必须在传感器A检测到故障之后响起。最后提取度量约束：5到10秒的时间窗口和至少20秒的持续时间。"
            }
        ],
        "synthesize": [
            {
                "id": "syn_001",
                "input_text": "基于语义规约草图合成MTL公式",
                "stage": "synthesize",
                "output": "G(fault_A → F_[5,10](alarm_B_on ∧ G_[0,20] alarm_B_on))",
                "reasoning": "1. 全局属性'Always'映射为G算子\n2. 时序关系'after'映射为蕴含关系 fault_A → ...\n3. 时间窗口[5,10]映射为F_[5,10]\n4. 持续约束>=20映射为G_[0,20]\n5. 组合得到最终公式"
            }
        ],
        "verify": [
            {
                "id": "ver_001",
                "input_text": "G(fault_A → F_[5,10](alarm_B_on ∧ G_[0,20] alarm_B_on))",
                "stage": "verify",
                "output": "总是，如果fault_A为真，那么在5到10个时间单位内，alarm_B_on将为真，并且在接下来的20个时间单位内保持为真。",
                "reasoning": "将MTL公式逐步翻译：G表示'总是'，→表示'如果...那么'，F_[5,10]表示'在5到10个时间单位内'，∧表示'并且'，G_[0,20]表示'在接下来的20个时间单位内保持'。"
            }
        ]
    }
    
    return examples

def main():
    """主函数演示"""
    print("=== Example Retrieval System Demo ===\n")
    
    # 创建示例检索器
    retriever = ExampleRetriever()
    
    # 如果没有示例，创建默认示例
    if not any(retriever.examples_db.values()):
        print("创建默认示例...")
        default_examples = create_default_examples()
        
        for stage, examples in default_examples.items():
            for example_data in examples:
                example = Example(
                    id=example_data["id"],
                    input_text=example_data["input_text"],
                    stage=example_data["stage"],
                    output=example_data["output"],
                    reasoning=example_data["reasoning"]
                )
                retriever.add_example(example)
    
    # 测试检索
    test_queries = [
        ("在系统启动后的3到5秒内，状态灯必须亮起", "deconstruct"),
        ("根据提取的语义组件构建MTL公式", "synthesize"),
        ("将MTL公式翻译回自然语言", "verify")
    ]
    
    for query, stage in test_queries:
        print(f"=== 测试查询: {query} (阶段: {stage}) ===")
        
        result = retriever.retrieve_examples(query, stage)
        
        if result.examples:
            print(f"检索到 {len(result.examples)} 个相似示例:")
            for i, (example, similarity) in enumerate(zip(result.examples, result.similarities), 1):
                print(f"  {i}. 相似度: {similarity:.3f}")
                print(f"     输入: {example.input_text[:50]}...")
        else:
            print("未找到相似示例")
        
        print()
    
    # 保存示例数据库
    retriever.save_examples("data/examples/example_database.json")

if __name__ == "__main__":
    main()
