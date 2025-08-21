import pandas as pd
from langchain_core.documents import Document
from models import QwenEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import re
import json
from rank_bm25 import BM25Okapi
import spacy


def df2content(df):
    res = []
    for i, row in df.iterrows():
        name = row['name']
        gender = row['gender']
        height = row['height']
        weight = row['weight']
        position = row['position']
        honor = row['honor']

        conent_prompt = f"""- name: {name}
- gender:{gender}

- height: {height}

- weight: {weight}

- position: {position}

- honor: {honor}
"""
        res.append(conent_prompt)

    return res


def token_text_for_bm25(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    tokens = [token.text.lower() for token in doc if token.is_alpha]
    return tokens

def create_indexstore_from_excel(file_path):
    df = pd.read_excel(file_path)
    print(f"{file_path} has been read completely.")
    contents = df2content(df)
    docs = [Document(page_content=content) for content in contents]
    vector_topk = 5
    bm25_topk = 5

    embeddings = QwenEmbeddings()


    batch_size = 10
    vectorstore = None


    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]

        # 第一次创建向量存储，之后合并
        if vectorstore is None:
            vectorstore = FAISS.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                distance_strategy=DistanceStrategy.COSINE
            )
        else:
            # 为后续批次创建临时向量存储并合并
            temp_store = FAISS.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                distance_strategy=DistanceStrategy.COSINE
            )
            vectorstore.merge_from(temp_store)

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": vector_topk})
    print("vector retriever has established.")
    # BM25
    tokenized_corpus = [token_text_for_bm25(doc.page_content) for doc in docs]
    bm25_model = BM25Okapi(tokenized_corpus)
    def bm25_retrieve(query):
        query_tokens = token_text_for_bm25(query)
        scores = bm25_model.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:bm25_topk]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    {"page_content": docs[idx].page_content,
                        "metadata": docs[idx].metadata,
                        "score": float(scores[idx])}
                )
        return  results

    print("bm25 retriever has established.")
    return vector_retriever,bm25_retrieve


def parse_response(response):
    pattern = r'```json\n(.*?)\n```'

    # 使用非贪婪匹配查找第一个符合条件的内容
    match = re.search(pattern, response, re.DOTALL)


    if match:
        # 重组完整的JSON数组字符串
        json_str = match.group(1)
        try:
            # 解析为JSON对象并返回
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("提取到的内容不是有效的JSON格式")
            return None
    else:
        print("未找到符合格式的JSON内容")
        return None