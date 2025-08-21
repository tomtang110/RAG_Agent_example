import pandas as pd
from langchain_core.documents import Document
from models import QwenEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import re
import json
def df2content(df):
    res = []
    for i, row in df.iterrows():
        name = row['name']
        gender = row['gender']
        height = row['height']
        weight = row['weight']
        position = row['position']
        honor = row['honor']

        conent_prompt = f"""- name 
{name}

- gender
{gender}

- height
{height}

- weight
{weight}

- position
{position}

- honor
{honor}
"""
        res.append(conent_prompt)

    return res

def create_vectorstore_from_excel(file_path):
    df = pd.read_excel(file_path)
    contents = df2content(df)
    docs = [Document(page_content=content) for content in contents]

    embeddings = QwenEmbeddings()

    # vectorstore = FAISS.from_documents(documents=docs,
    #                                    embedding=embeddings,
    #                                    distance_strategy=DistanceStrategy.COSINE
    #                                    )

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

    return vectorstore.as_retriever(search_kwargs={"k": 2})


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