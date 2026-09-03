from typing import Annotated, TypedDict 
from langchain_core.messages import BaseMessage
from operator import add , or_
from langgraph.graph.message import add_messages

# States
class MainState(TypedDict):
    query : str
    messages: Annotated[list[BaseMessage], add_messages]
    transcript_summary : str
    main_llm_prompt : str
    final_output : str

#Sub state
class RouterState(TypedDict):
    query : str
    links : Annotated[list[str], add]
    messages: Annotated[list[BaseMessage], add_messages]
    chunks : Annotated[list[str], add]
    transcripts : str
    summaries : Annotated[dict[str, str], or_]
    main_llm_prompt : str