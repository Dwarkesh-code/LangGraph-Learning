"""
Core LangGraph chatbot: one LLM node bound to tools, one tool-execution node,
and a SQLite checkpointer for persistence across app restarts.

Graph shape:

    START -> chat_node -> (tools_condition) -> tool_node -> chat_node -> END
                                |
                                +-> END   (when the LLM doesn't call a tool)

This mirrors the "existing chatbot" state from the CampusX video, right
before RAG gets bolted on: streaming works, persistence works, and there
are already two tools wired up. Add a retriever tool in tools.py and this
graph picks it up automatically — no changes needed here.
"""

import os
import sqlite3

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

from tools import TOOLS

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "chatbot_state.db")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatGroq(model="llama-3.3-70b-versatile", streaming=True)
llm_with_tools = llm.bind_tools(TOOLS)


def chat_node(state: ChatState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def build_graph(checkpointer):
    builder = StateGraph(ChatState)
    builder.add_node("chat_node", chat_node)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "chat_node")
    builder.add_conditional_edges("chat_node", tools_condition)
    builder.add_edge("tools", "chat_node")

    return builder.compile(checkpointer=checkpointer)


def get_checkpointer() -> SqliteSaver:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return checkpointer


def list_threads(checkpointer: SqliteSaver) -> list[str]:
    """Return distinct thread_ids that have saved state, most recent first."""
    seen = []
    for checkpoint_tuple in checkpointer.list(None):
        thread_id = checkpoint_tuple.config["configurable"]["thread_id"]
        if thread_id not in seen:
            seen.append(thread_id)
    return seen