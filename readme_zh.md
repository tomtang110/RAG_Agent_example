```markdown
# 简单的 Agent RAG：篮球运动员信息问答系统

![License](https://img.shields.io/badge/license-MIT-blue.svg)

本项目实现了一个简单的 **Agent RAG（检索增强生成）系统**，旨在帮助开发者和研究者理解如何构建一个领域专用的问答系统。当前系统专注于 **篮球相关问题**，数据涵盖 **中国和美国各前50名篮球运动员的概览信息**。

系统集成了路由、嵌入检索（Embedding）、BM25、生成等核心模块，是学习 RAG 架构的理想实践项目。

---

## 📚 数据集说明

系统基于以下两个 Excel 数据文件构建：

- `america_basketball_player.xlsx`：美国前50名篮球运动员的综合信息
- `china_basketball_player.xlsx`：中国前50名篮球运动员的综合信息

> ⚠️ **注意**：本系统**仅支持篮球相关查询**，其他领域问题将被拒绝。

---

## 🧱 系统架构

Agent RAG 流程包含以下核心组件：

| 模块       | 功能说明 |
|------------|--------|
| **Router（路由）** | 判断用户问题是否与篮球相关，过滤无关请求 |
| **Retriever（检索器）** | 融合 **向量检索（Embedding）** 与 **关键词检索（BM25）**，提升召回准确率 |
| **Generator（生成器）** | 基于检索结果，调用大模型生成自然语言回答 |

> 🔧 **后续计划**：将逐步加入查询改写、重排序（reranker）、网络搜索等高级模块。

---

## 🛠️ 依赖库

项目依赖以下核心库和服务：

- `qwen` – 使用通义千问 API 进行文本嵌入和生成
- `langchain` – 构建 LLM 应用的核心框架
- `langgraph` – 用于构建 Agent 状态流和工作流
- `pydantic` – 数据校验与配置管理
- `pandas` – 读取和处理 Excel 数据
- `openpyxl` – 支持 `.xlsx` 文件读取
- 其他常用库：`tqdm`, `numpy` 等

安装依赖：
```bash
pip install -r requirements.txt
```

> 💡 需提前配置 **Qwen API Key**（建议通过环境变量设置）

---

## 📁 项目结构

```
.
├── main.py                   # 主程序入口：实现完整的 RAG 流程
├── models.py                 # 定义嵌入模型和生成模型的 API 接口
├── prompt_set.py             # 包含路由和生成阶段的提示词模板
├── utils.py                  # 工具函数，如数据解析、格式处理等
├── america_basketball_player.xlsx  # 美国球员数据集
├── china_basketball_player.xlsx    # 中国球员数据集
├── README.md / README_zh.md  # 说明文档（中英文）
└── requirements.txt          # 依赖列表
```

---

## 🚀 使用方法

1. **设置 Qwen API Key**：
   ```bash
   export QWEN_API_KEY="your_api_key_here"
   ```

2. **运行主程序**：
   ```bash
   python main.py
   ```

3. **输入问题示例**：
   - “中国排名前三的球员是谁？”
   - “勒布朗·詹姆斯效力于哪支球队？”
   - “姚明的身高和职业生涯得分是多少？”

> ✅ 系统会自动判断问题相关性，非篮球问题将被礼貌拒绝。

---

## 🧪 示例输出

**输入**：  
“列出中国前3名球员的职业生涯总得分”

**输出**：  
根据数据，中国前3名球员的职业生涯得分如下：  
1. 姚明 – NBA与CBA合计超过9200分  
2. 易建联 – 职业生涯累计7500+分  
3. 郭艾伦 – CBA核心后卫，得分超6000分  

> （注：具体数值以实际数据集内容为准）

---

## 📌 当前限制

- 仅支持篮球领域问题
- 暂无查询扩展、重排序等高级功能
- 回答质量依赖数据完整性和模型能力

---

## 🌟 后续优化计划

- [ ] 添加 **查询改写（Query Rewriting）** 提升召回率
- [ ] 集成 **重排序模型（Reranker）** 提高结果相关性
- [ ] 支持 **网络搜索扩展**，应对未知知识
- [ ] 扩展至更多体育项目或通用领域

---

## 🙌 致谢

- [LangChain](https://langchain.com) 与 [LangGraph](https://langchain-ai.github.io/langgraph/) – 强大的流程编排工具
- [通义千问 Qwen](https://qwen.ai) – 提供高质量的 LLM API 支持

---

## 📄 许可证

MIT License，详见 `LICENSE` 文件。

---

> 🤖 由 阿里云 Qwen 打造 | 仅供学习与研究使用
```