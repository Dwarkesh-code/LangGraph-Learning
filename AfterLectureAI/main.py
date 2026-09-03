import os
import sqlite3

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph , END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict 
from operator import add 
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from state import MainState, RouterState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

import uuid

thread_id = str(uuid.uuid4())


# Rate limiter
rate_limiter = InMemoryRateLimiter(
    requests_per_second= 40/60,
    check_every_n_seconds= 0.1,
    max_bucket_size= 1
)

llm = ChatNVIDIA(model="nvidia/nemotron-3-ultra-550b-a55b", rate_limiter= rate_limiter)

#short memory 
DB_PATH = os.path.join(os.path.dirname(__file__), "AfterLectureAI_Memory.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)

from tools.yt_transcript import fetch_transcripts
from tools.link_extractor import links_extractor
from tools.summarizer import make_summarize_videos
from tools.web_search import searcher
summarizer_videos = make_summarize_videos(llm=llm)

from prompts import MAIN_LLM_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT

tools = [links_extractor, fetch_transcripts, summarizer_videos, searcher]

def router_node(state: RouterState):
    messages = state["messages"]

    if not messages:
        messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)]

    messages = messages + [HumanMessage(content=state["query"])]

    response = llm.bind_tools(tools=tools).invoke(messages)

    if not response.tool_calls:
        return {
            "messages": [response],
            "main_llm_prompt": response.content,
        }

    return {"messages": [response]}

main_llm_tools = [searcher]
def main_llm_node(state: RouterState):
    messages = state.get("main_llm_messages", [])

    if not messages:
        messages = [
            SystemMessage(content=MAIN_LLM_SYSTEM_PROMPT),
            HumanMessage(content=state["main_llm_prompt"]),
        ]

    response = llm.bind_tools(tools=main_llm_tools).invoke(messages)

    return {"main_llm_messages": [response]}


# Router Graph 
router_graph_builder = StateGraph(RouterState)

# nodes

router_graph_builder.add_node("router",router_node)
router_graph_builder.add_node("tools", ToolNode(tools=tools))
#edges

router_graph_builder.add_edge(START, "router")
router_graph_builder.add_node("tools", ToolNode(tools))
router_graph_builder.add_conditional_edges("router", tools_condition)
router_graph_builder.add_edge("tools", "router")
router_graph_builder.add_edge("router", END)


router_graph = router_graph_builder.compile()

def final_node(state:MainState) -> MainState: 
    initial_state = {
        "query": state["query"],
        "links": [],
        "messages": [],
        "chunks": [],
        "transcripts": "",
        "summaries": {},
        "core_keywords": {},
        "main_llm_prompt": "",
    }

    router_result = router_graph.invoke(initial_state)

    return {
        "main_llm_prompt": router_result["main_llm_prompt"],
    }

main_state_graph_builder = StateGraph(MainState)

main_state_graph_builder.add_node("router_step", final_node)
main_state_graph_builder.add_node("main_llm", main_llm_node)
main_state_graph_builder.add_node("main_tools", ToolNode(main_llm_tools))

main_state_graph_builder.add_edge(START, "router_step")
main_state_graph_builder.add_edge("router_step", "main_llm")
main_state_graph_builder.add_conditional_edges("main_llm", tools_condition)
main_state_graph_builder.add_edge("main_tools", "main_llm")
main_state_graph_builder.add_edge("main_llm", END)

main_graph = main_state_graph_builder.compile(checkpointer=checkpointer)

query = "https://www.youtube.com/playlist?list=PLKnIA16_RmvYsvB8qkUQuJmJNuiCUJFPL  suggest me projects for this playlist"

initial_state = {
    "query": query,
    "messages": [],
    "transcript_summary": "",
    "main_llm_prompt": "",
    "final_output": "",
}

config = {"configurable": {"thread_id": thread_id}}

result = main_graph.invoke(initial_state, config=config)

print(result)