import os
import sqlite3

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict 




load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "chatbot_state.db")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



def build_graph(checkpointer, tools):
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    llm_with_tools = llm.bind_tools(tools)

    def chat_node(state: ChatState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(ChatState)
    builder.add_node("chat_node", chat_node)
    builder.add_node("tools", ToolNode(tools))
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