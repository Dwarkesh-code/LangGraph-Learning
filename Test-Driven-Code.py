from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, Annotated,Literal
from pydantic import BaseModel, Field
from operator import add
from langchain_core.prompts import PromptTemplate


load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")


#state
class TestState(TypedDict):
    query : str
    tests : Annotated[list[str], add]
    precode : Annotated[list[str], add]
    newcode : str
    total_tests : int
    verdict : Literal['approved', 'pending']
    coder_prompt : str
    tester_prompt : str

#structure output 

class StrPrompt(BaseModel): 
    code_prompt: str = Field(description="A clear, standalone prompt for the code-writing model")
    tester_prompt : str = Field(description="A clear, standalone prompt for the test decider  model")


str_prompt_model = model.with_structured_output(StrPrompt)

#nodes 
def summarize_query_node(state: TestState) -> TestState:
    prompt = PromptTemplate.from_template("""
        You are a query reformulation specialist.
        Raw user query: "{query}"
        Rewrite this into TWO clear, self-contained prompts — one for each downstream LLM.
        1. **code_prompt**: A focused instruction for a Python code-writing model.
        Include:
        - The exact function/task the user wants
        - Any constraints, edge cases, or examples they mentioned
        - Expected output format (just the function, no extra commentary)
        2. **tester_prompt**: A focused instruction for a test-case-designer model.
        Include:
        - The function/task being tested
        - The kinds of test cases that would validate it
            (basic, edge cases, tricky inputs)
        - Expected format (e.g. test name + description + input/expected output)
        Both prompts must be STANDALONE — the downstream models should be able
        to do their job from the prompt alone, without ever seeing the original
        user query. No ambiguity, no missing context.
        """)

    chain = prompt | str_prompt_model 
    response = chain.invoke({"query": state["query"]})

    return {
        "coder_prompt": response.code_prompt,
        "tester_prompt": response.tester_prompt
    }



