import os
import json
from typing import List, Any, Literal,Dict
from pydantic import BaseModel

from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores.utils import DistanceStrategy
import pandas as pd
from langgraph.graph import StateGraph, END,START
import dashscope
from dashscope import Generation  # 用于 Qwen 生成
from utils import create_vectorstore_from_excel,parse_response
from prompt_set import router_prompt,generation_prompt
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dotenv import load_dotenv



load_dotenv(dotenv_path=r"../.env", override=True)


api_k = os.getenv("DASHSCOPE_API_KEY")

dashscope.api_key = api_k

class AgentState(BaseModel):
    question: str
    router:List[str]=[]
    contexts: Dict[str,str]={}
    final_answer: str=""
    retrieved_context: str=""


db_configs = {
    "america": "./data/america_basketball_player.xlsx"
    ,"china":"./data/china_basketball_player.xlsx"
}

retriever_dict = {}

for db_name,file_path in db_configs.items():

    retriever = create_vectorstore_from_excel(file_path)
    if retriever:
        retriever_dict[db_name] = retriever


def route_database(state):
    prompt = router_prompt(state.question)

    response = Generation.call(model='qwen3-8b',
                               messages=[{"role": "user", "content": prompt}],
                               result_format="message",  # ✅ 关键：返回标准 message 格式
                               enable_thinking=False,
                               )

    db_choice = response.output.choices[0].message.content
    print(db_choice)
    router_res = parse_response(db_choice)
    state.router = router_res
    return state


executor = ThreadPoolExecutor(max_workers=4)


def _retrieve_one(db_name,question):
    try:
        docs = retriever_dict[db_name].invoke(question)
        content = "\n\n".join([d.page_content for d in docs])
        return db_name,content
    except Exception as e:
        return db_name, f"[retrieval fail: {str(e)}]"


def retrieve_parallel(state):
    futures = [
        executor.submit(_retrieve_one, db_name, state.question)
        for db_name in state.router  # 注意：你原代码写的是 state.db_list，但实际是 state.router
    ]
    results = [future.result() for future in futures]  # 等待全部完成

    state.contexts = {db: content for db, content in results}
    return state

def aggregate_context(state):
    parts = []
    for db_name,content in state.contexts.items():
        if content.strip() and not content.startswith("[retrieval fail"):
            parts.append(f"[source from {db_name}]\n{content}")

    contexts = []
    for i, part in enumerate(parts):
        contexts.append(f"{i}. {part}")
    state.retrieved_context = "\n\n".join(contexts) if contexts else "retrieve nothing"
    return state

def generate_answer(state):

    query = state.question
    context_list = state.retrieved_context
    text = generation_prompt(query,context_list)
    print(text)
    response = Generation.call(model="qwen3-14b",
                               messages=[{"role": "user", "content": text}],
                               result_format="message",  # ✅ 关键：返回标准 message 格式
                               enable_thinking=False,
                               )
    state.final_answer = response.output.choices[0].message.content
    return state


workflow = StateGraph(AgentState)


workflow.add_node("route_databases",route_database)
workflow.add_node("retrieve_parallel",retrieve_parallel)
workflow.add_node("aggregate_context",aggregate_context)
workflow.add_node("generate_answer",generate_answer)


workflow.add_edge(START, "route_databases")
workflow.add_edge("route_databases","retrieve_parallel")
workflow.add_edge("retrieve_parallel","aggregate_context")
workflow.add_edge("aggregate_context","generate_answer")
workflow.add_edge("generate_answer",END)

app = workflow.compile()



if __name__ == "__main__":
    question = ["who is the best basketball player in USA?"]

    for q in question:
        print(f"query is: {q}")
        final_answer = ""
        inputs = AgentState(question=q)
        for event in app.stream(inputs):
            print(event)
            print('___')
            if "generate_answer" in event:
                final_answer = event["generate_answer"].get("final_answer", "No answer generated")

        print(f"✅ final answer: {final_answer}")