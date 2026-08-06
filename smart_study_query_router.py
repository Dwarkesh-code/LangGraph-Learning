from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from tools import agent_tools
from typing import TypedDict, Annotated
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages


load_dotenv()

#State 
class State(TypedDict):
    query : str
    router_response : str
    messages: Annotated[list, add_messages]
    final_answer : str


parser = StrOutputParser()
# LLMs 
router_llm = ChatGroq(model="llama-3.3-70b-versatile")
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
llm_with_tools= llm.bind_tools(agent_tools)

# Nodes 
def router_nodes(state: State)-> State :
    prompt =PromptTemplate(
        template= """Generate the best prompt Base on the query for main LLM specify the query, force the llm for calling tools.
        tools name list that main LLM have = '{agent_tools}'
        user query = '{query}'  """,
        input_variables= ["agent_tools", "query"]
    )
    router_chain = prompt | router_llm | parser 
    response = router_chain.invoke({"agent_tools":agent_tools, "query": state["query"]})
    return {"router_response": response,
            "messages": [HumanMessage(content=response)]}
    
def llm_node(state: State) -> State:
    llm_prompt = state["router_response"]
    response = llm.invoke(llm_prompt)
    return {"final_answer": response}


#graph
graph = StateGraph(State)
graph.add_node("router", router_nodes)
graph.add_node("llm", llm_node)
graph.add_node("tools", ToolNode(agent_tools))
graph.add_edge(START, "router")
graph.add_edge("router", "llm")
graph.add_conditional_edges(
    "llm",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)
graph.add_edge("tools","llm")
graph.add_edge("llm", END)

memory = MemorySaver() 

app = graph.compile(checkpointer=memory)

query = input("You:>  ")

initial_state = {
    "query": query,
    "router_response": "",
    "messages": [],
    "final_answer": ""
}
config = {"configurable": {"thread_id": "1"}}

response = app.invoke(initial_state, config=config)

print("AI:> \n", response["final_answer"])


