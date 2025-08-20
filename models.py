import os
from openai import OpenAI
from dotenv import load_dotenv
import json


load_dotenv(dotenv_path=r"/code_project/.env", override=True)


api_k = os.getenv("DASHSCOPE_API_KEY")

def generation_models(prompt):
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

if __name__ == "__main__":
    input_str = "what is your name"
    # print(embedding_models(input_str))

    print(generation_models(input_str))