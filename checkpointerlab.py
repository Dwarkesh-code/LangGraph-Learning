from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import json
import os

#state
class CounterState(TypedDict):
    count: int
    log: str


# Nodes (simple, deterministic, no LLM)
def increment_node(state: CounterState) -> CounterState:
    new_count = state["count"] + 1
    return {"count": new_count, "log": f"incremented to {new_count}"}


def double_node(state: CounterState) -> CounterState:
    new_count = state["count"] * 2
    return {"count": new_count, "log": f"doubled to {new_count}"}


def label_node(state: CounterState) -> CounterState:
    return {"log": f"final count is {state['count']}"}


# ---- Graph wiring ----
builder = StateGraph(CounterState)
builder.add_node("increment", increment_node)
builder.add_node("double", double_node)
builder.add_node("label", label_node)

builder.add_edge(START, "increment")
builder.add_edge("increment", "double")
builder.add_edge("double", "label")
builder.add_edge("label", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)



# ---- Your experiments go here ----
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "experiment1"}}

    FILE_NAME = "checkpoint.json"
    if os.path.exists(FILE_NAME):
        with open(FILE_NAME, "r", encoding="utf-8") as file:
            try:
                all_data = json.load(file)
            except json.JSONDecodeError:
                all_data = {}
    else:
        all_data = {}

    thread_id= config["configurable"]["thread_id"]

    if thread_id not in all_data:
        all_data[thread_id] = {}
        initial_state = {
                "count" : 3,
                "log" : ""
            }
            

    else :
        thread_data = all_data.get(thread_id, {})
        initial_state = {
                "count" : all_data.get("count", 0),
                "log" : ""
            }

    
    # res = graph.invoke(initial_state, config=config)
    # print(res)

    # data_json = {
    #     config["configurable"]["thread_id"] : {"count" : res["count"], "log": res["log"]}
    # } 
    # all_data[thread_id].update(data_json)

    # #save data in json
    # with open(FILE_NAME, "w" , encoding="utf-8") as file : 
    #     json.dump(all_data, file, indent=4)        

    print(graph.get_state(config))

    pass