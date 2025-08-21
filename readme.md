```markdown
# Simple Agent RAG for Basketball Player Information

![License](https://img.shields.io/badge/license-MIT-blue.svg)

This repository provides a simple **Agent RAG (Retrieval-Augmented Generation)** system designed to help practitioners understand the foundational components of building a domain-specific question-answering system. The current implementation focuses exclusively on **basketball-related queries**, specifically information about the **top 50 Chinese and American basketball players**.

This project is ideal for learning how to integrate core RAG components such as routing, retrieval (embedding + BM25), and generation using modern LLM tooling.

---

## 📚 Dataset Overview

The system is trained on two datasets:

- `america_basketball_player.xlsx`: Overview of the top 50 American basketball players.
- `china_basketball_player.xlsx`: Overview of the top 50 Chinese basketball players.

> ⚠️ **Note**: The RAG system currently **only supports basketball-related queries** due to the limited scope of the dataset.

---

## 🧱 System Architecture

The Agent RAG pipeline includes the following core modules:

| Module       | Description |
|--------------|-----------|
| **Router**   | Determines if the query is basketball-related. If not, it returns a rejection response. |
| **Retriever** | Combines **dense retrieval (embedding-based)** and **sparse retrieval (BM25)** for robust document lookup. |
| **Generator** | Uses a large language model to generate natural language responses based on retrieved context. |

> 🔧 **Future Enhancements**: Support for query rewriting, reranking, and web search will be added in upcoming versions.

---

## 🛠️ Dependencies

This project relies on the following key libraries and services:

- `qwen` – Alibaba's Qwen API for LLM inference (used for generation and embeddings)
- `langchain` – Framework for building LLM-powered applications
- `langgraph` – For orchestrating agent workflows and state management
- `pydantic` – Data validation and settings management
- `pandas` – For loading and processing Excel datasets
- `openpyxl` – Required for reading `.xlsx` files
- Other standard utilities: `tqdm`, `numpy`, etc.

Install dependencies:
```bash
pip install -r requirements.txt
```

> 💡 You must have access to the **Qwen API** and configure your API key (e.g., via environment variables).

---

## 📁 Project Structure

```
.
├── main.py                   # Entry point: Implements the full RAG pipeline
├── models.py                 # Defines the embedding and generation APIs using Qwen
├── prompt_set.py             # Contains prompts for router and generator
├── utils.py                  # Utility functions for parsing and data processing
├── america_basketball_player.xlsx  # Dataset: Top 50 American players
├── china_basketball_player.xlsx    # Dataset: Top 50 Chinese players
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

---

## 🚀 Usage

1. **Set up your Qwen API key**:
   ```bash
   export QWEN_API_KEY="your_api_key_here"
   ```

2. **Run the main application**:
   ```bash
   python main.py
   ```

3. **Ask a question**:
   Example queries:
   - "Who is the tallest player in the Chinese top 50?"
   - "What team does the number 1 ranked American player play for?"

> ✅ Only basketball-related questions will be processed. All others will be filtered by the router.

---

## 🧪 Example Output

**Input**:  
"List the top 3 Chinese basketball players by career points."

**Output**:  
Based on the dataset, the top 3 Chinese basketball players by career points are:  
1. Yao Ming – 9,200+ points in the NBA and CBA combined  
2. Yi Jianlian – Over 7,500 professional points  
3. Guo Ailun – Key guard with over 6,000 CBA points  

*(Note: Actual stats depend on dataset content)*

---

## 📌 Limitations

- Domain-specific: Only answers questions related to the provided basketball datasets.
- No query expansion or advanced reranking yet.
- Performance depends on the quality of the input data and embedding model.

---

## 🌟 Future Work

- [ ] Add **query revision** module to improve retrieval accuracy
- [ ] Integrate a **reranker** (e.g., BERT-based) to improve result relevance
- [ ] Enable **web search fallback** for out-of-dataset queries
- [ ] Support for more sports and broader domains

---

## 🙌 Acknowledgments

- [LangChain](https://langchain.com) & [LangGraph](https://langchain-ai.github.io/langgraph/) – For powerful orchestration tools
- [Alibaba Cloud Qwen](https://qwen.ai) – For providing state-of-the-art LLM APIs

---

## 📄 License

MIT License. See `LICENSE` for details.

---

> 🤖 Built with ❤️ by Qwen | Alibaba Cloud  
> For educational and research purposes.
```