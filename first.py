from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

def call_model(state: MessagesState):
    
    response = model.invoke(state["messages"])

    return {'messages':[response]}

graph = StateGraph(MessagesState)
graph.add_node("model", call_model)
graph.add_edge(START, "model")
graph.add_edge("model", END)
agent = graph.compile()

response = agent.invoke({"messages": [{"role":"user", "content": "You're worst"}]})
print(response["messages"][1].content)
