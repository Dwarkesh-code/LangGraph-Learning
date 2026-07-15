from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """Do mulltiply of 2 int
    
    Args : 
        a = fist int,
        b = second int"""
    return a*b

tool_nodes = ToolNode([multiply])

llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools= llm.bind_tools([multiply])

def model(state:MessagesState):
    return {"messages": [llm_with_tools.invoke(state['messages'])]} 
    
memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}

graph = StateGraph(MessagesState)
graph.add_node("model", model)
graph.add_node("tools", tool_nodes)
graph.add_edge(START, "model")
graph.add_conditional_edges("model", tools_condition)
graph.add_edge("tools", "model")
final = graph.compile(checkpointer=memory)

result = final.invoke({"messages": ["multiply 2 and 293 "]}, config)

print("output: \n",result['messages'][-1].content,"\n\n\n")

result = final.invoke({"messages": ["multiply 7 and 34 "]}, config)

print("output: \n",result['messages'][-1].content)
