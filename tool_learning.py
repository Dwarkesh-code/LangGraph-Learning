from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import numexpr
load_dotenv()

# ---- state ----
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
# ---- model ----

#tools 
search = DuckDuckGoSearchRun()

@tool
def calculator(expression: str) -> str:
    """Useful for when you need to answer mathematical questions. 
    Input should be a mathematical expression like '2 + 2' or 'sqrt(16) * 3'."""
    try:
        return str(numexpr.evaluate(expression).item())
    except Exception as e:
        return f"Error: {e}"

# tools binding
tools = [search, calculator]

model_with_tools = model.bind_tools(tools=tools) 

tool_node = ToolNode(tools)

# ---- nodes ----
def chat_node(state: ChatState) -> ChatState:
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ---- graph ----
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")


workflow = graph.compile()

if __name__ == "__main__":
    state: ChatState = {"messages": []}
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        state["messages"].append(("user", user_input))
        state = workflow.invoke(state)
        print("Bot:", state["messages"][-1].content)

        print("\n\n", state["messages"], "\n\n\n")