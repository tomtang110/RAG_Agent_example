import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from langchain_core.embeddings import Embeddings
from typing import List, Any, Literal
from http import HTTPStatus
import dashscope

def generation_models(prompt):
    load_dotenv(dotenv_path=r"/code_project/.env", override=True)

    api_k = os.getenv("DASHSCOPE_API_KEY")
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=api_k,  # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen3-8b",  # qwen-plus 属于 qwen3 模型，如需开启思考模式，请参见：https://help.aliyun.com/zh/model-studio/deep-thinking
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ],
        extra_body={"enable_thinking": False},
    )
    output = completion.choices[0].message.content

    return output

def embedding_models(doc):
    load_dotenv(dotenv_path=r"/code_project/.env", override=True)

    api_k = os.getenv("DASHSCOPE_API_KEY")
    client = OpenAI(
        api_key=api_k,  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )

    completion = client.embeddings.create(
        model="text-embedding-v4",
        input=doc,
        dimensions=512, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
        encoding_format="float"
    )

    embeddings_str = completion.model_dump_json()

    embeddings = json.loads(embeddings_str)['data'][0]['embedding']

    return embeddings


class QwenEmbeddings(Embeddings):
    client = None
    model_name = "text-embedding-v4"
    dimensions = 512

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        load_dotenv(dotenv_path=r"../.env", override=True)

        api_k = os.getenv("DASHSCOPE_API_KEY")
        # print(api_k)
        self.client = OpenAI(
        api_key=api_k,  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            dimensions=self.dimensions,
            encoding_format="float"
        )
        result = json.loads(response.model_dump_json())
        return [d['embedding'] for d in result['data']]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def rerank_with_dashscope(query: str, passages: list):
    load_dotenv(dotenv_path=r"../.env", override=True)

    api_k = os.getenv("DASHSCOPE_API_KEY")

    dashscope.api_key = api_k
    resp = dashscope.TextReRank.call(
        model="gte-rerank-v2",
        query=query,
        documents=passages,
        top_n=5,
        return_documents=True
    )
    if resp.status_code == HTTPStatus.OK:
        return resp
    else:
        print(resp)

if __name__ == "__main__":
    input_str = "what is your name"
    # print(embedding_models(input_str))

    # print(generation_models(input_str))
    print(rerank_with_dashscope(input_str,["life is astruggle"]))