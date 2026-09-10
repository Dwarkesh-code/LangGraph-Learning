from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

def call_model(state: MessagesState):
    
    response = model.invoke(state["messages"])

    return {'messages':[response]}

graph = StateGraph(MessagesState)
graph.add_node("model", call_model)
graph.add_edge(START, "model")
graph.add_edge("model", END)
agent = graph.compile()

response = agent.invoke({"messages": [{"role":"user", "content": "today's date and time"}]})
print(response["messages"][1].content)
