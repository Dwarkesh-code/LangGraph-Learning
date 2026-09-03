import os
import sqlite3

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict 
from operator import add 
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from state import MainState, RouterState

load_dotenv()

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

from prompts import ROUTER_SYSTEM_PROMPT

tools = [links_extractor, fetch_transcripts, summarizer_videos, searcher]

def router_node(state: RouterState):
    query = state["query"]
    if not state["messages"] or not isinstance(state["messages"][0], SystemMessage):
        state["messages"] = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)] + state["messages"]

    state["messages"] = [HumanMessage(content=query)] + state["messages"]

    

