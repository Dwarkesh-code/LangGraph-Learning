from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from dotenv import load_dotenv
from typing import TypedDict
from langchain.tools import tool


load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """Do mulltiply of 2 int
    
    Args : 
        a = fist int,
        b = second int"""
    return a*b



llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools= llm.bind_tools([multiply])

def model(state:MessagesState):
    return {"messages": [llm_with_tools.invoke(state['messages'])]} 
    


graph = StateGraph(MessagesState)
graph.add_node("model", model)
graph.add_edge(START, "model")
graph.add_edge("model", END)
final = graph.compile()

result = final.invoke({"messages": ["You're worst"]})

print("output: \n",result)