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
    precode : str
    newcode : str
    total_tests : int
    verdict : Literal['approved', 'pending']
    coder_prompt : str
    tester_prompt : str
    code_feedback : str
    attempt : int

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


def code_writer_node(state:TestState) -> TestState:
    #prompt 
    prompt = PromptTemplate.from_template("""
        You are an expert Python developer who writes production-grade functions.

        TASK:
        {instruction}

        RULES:
        - Output ONLY the function definition (def ... : ... and its body)
        - No prose before or after
        - No markdown code fences (```), no comments outside the function
        - Include a docstring (purpose, args, return value)
        - Handle all edge cases mentioned in the task
        - Use clean naming and idiomatic Python
  
    """)

    if state['attempt'] > 1 :
        prompt1 = PromptTemplate.from_template("""
               Feedback => \n{feedback}
               \n\n

               previous code => \n{precode}    
        """)
        prompt += prompt1

    chain = prompt| model 
    response = chain.invoke({"instruction": state["coder_prompt"], "feedback": state["code_feedback"], "precode": state["precode"]}).content

    return {"newcode": response, "precode": response}


