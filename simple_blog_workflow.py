from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph, END
from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

parser = StrOutputParser()

class BlogState(TypedDict):
    topic : str
    outline : str 
    blog : str
    evalute : str


#outline node 
def outline_node(state: BlogState)-> BlogState:
    #extrat topic
    topic = state['topic']

    #prompt    
    prompt = f'Generate a outline based on topic --> "{topic}"'


    outline =  model.invoke(prompt).content

    state['outline'] = outline

    return state


#blog node
def blog_node(state: BlogState)-> BlogState:
    #extract topic and outline
    topic = state['topic']
    outline = state['outline']

    #prompt
    prompt = f'Generate a blog on "{topic}" based on outline ---> \n {outline}'

     
    
    blog =  model.invoke(prompt).content

    state['blog'] = blog

    return state


#evalute node
def evalute_node(state:BlogState)-> BlogState:
    #extract topic, outline and blog
    topic = state['topic']
    outline = state['outline']
    blog = state['blog']

    #prompt
    prompt = f'Rate this blog in 1 line remeber u have only 1 line to rate it .\n Blog --> \n "{blog}" \n based on topic -->\n "{topic}"\n\n and \n outline ---> \n {outline}'

    
    evalute =  model.invoke(prompt).content

    state['evalute'] = evalute

    return state


graph = StateGraph(BlogState)

#add nodes
graph.add_node("outline", outline_node)
graph.add_node("blog", blog_node)
graph.add_node("evalute", evalute_node)

#add edges
graph.add_edge(START, "outline")
graph.add_edge("outline", "blog")
graph.add_edge("blog", "evalute")
graph.add_edge("evalute", END)

agent = graph.compile()

topic = input("Topic :- ").strip()

intial_state = {"topic": topic, "outline": "", "blog": "", "evalute": ""}

result = agent.invoke(intial_state)

print("Blog = \n", result["blog"])
print("\n\n")
print("Ranking = \n", result["evalute"])