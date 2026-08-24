from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict,Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


#groq router models names
groq_llms = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "llama-3.1-8b-instant"
]

# gemini llms names 
gemini_llms = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash"
]



model = ChatGroq(model="llama-3.3-70b-versatile", temperature= 0.9)
main_llms_fallback_models = [
    ChatGoogleGenerativeAI(model=model_name, temperature=0.9) 
    for model_name in gemini_llms
]+ [
    ChatGroq(model=model_name, temperature=0.9) 
    for model_name in groq_llms
]


model = model.with_fallbacks(main_llms_fallback_models)

#state
class TestState(TypedDict):
    query : str
    tests : list[str]
    total_tests : int
    coder_prompt : str
    tester_prompt : str
    final_code : str
    verdict : str
    pass_tests : int
    attempt : int
    code_feedback : str

    
class CoderState(TypedDict):
    precode : str
    newcode : str
    code_feedback : str
    attempt : int
    verdict : Literal['approved', 'pending']
    sub_tests : list[str]
    sub_total_tests : int
    pass_tests : int
    sub_coder_prompt : str

#pydantic 

#structure prompt output 

class StrPrompt(BaseModel): 
    code_prompt: str = Field(description="A clear, standalone prompt for the code-writing model")
    tester_prompt : str = Field(description="A clear, standalone prompt for the test decider  model")

str_prompt_model = model.with_structured_output(StrPrompt)


#structure tests output
class StrTester(BaseModel):
    tests : str = Field(description="Clear, Tests based on instruction")
    total_tests : int = Field(description="Provide the test count.", ge=1)

str_tests_model = model.with_structured_output(StrTester)

#structure runner output
class StrRunner(BaseModel):
    feedback: str = Field(description="Generate the feedback according to output, tell what to improve")
    verdict: Literal["approved", "pending"] = Field(description="'approved' if all tests pass, 'pending' if any test fails")
    pass_tests: int = Field(description="Total number of tests that passed", ge=0)

str_runner_model = model.with_structured_output(StrRunner)

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


def code_writer_node(state:CoderState) -> CoderState:
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

    #invoke
    chain = prompt| model 
    raw = chain.invoke({"instruction": state["sub_coder_prompt"], "feedback": state["code_feedback"], "precode": state["precode"]}).content

    if isinstance(raw, list):
        response = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw
        )
    else:
        response = raw


    return {"newcode": response, "precode": response}


def test_designer_node(state:TestState) -> TestState:
    #prompts 
    prompt = PromptTemplate.from_template("""
        You are a senior QA engineer who designs comprehensive test cases for Python functions.

        TASK:
        Design test cases for the function described below. Cover the full input space.

        FUNCTION SPEC:
        {instruction}

        REQUIRED TEST CATEGORIES (must include all):
        1. **Basic / happy path** — typical valid input, function returns expected value
        2. **Empty / zero / None inputs** — boundary cases
        3. **Single element** — list with one item, string with one char, etc.
        4. **All-fail / all-pass edge cases** — inputs where every element triggers or skips the logic
        5. **Type edge cases** — negative numbers, very large numbers, floats vs ints, mixed types
        6. **Tricky / off-by-one cases** — inputs designed to catch common bugs

        Output format: one test case per line in this exact format:
        - test_<short_name>: input=<input_value>, expected=<expected_output>

        Aim for 6-10 test cases minimum. Be ruthless — the goal is to BREAK the implementation, not validate happy paths.
                """)

    #invoke 
    chain = prompt | str_tests_model 
    response = chain.invoke({"instruction": state["tester_prompt"]})

    return {"tests": [response.tests], "total_tests" : response.total_tests}


def code_runner_node(state:CoderState) -> CoderState : 
    #prompt 
    prompt = PromptTemplate.from_template("""
        You are a meticulous Python interpreter. Execute the function below against each test case with perfect accuracy — do not assume, do not approximate.

        FUNCTION:
        {code}

        TOTAL TESTS: {total_tests}

        TEST CASES (one per line):
        {tests}

        For each test case:
        1. Identify inputs
        2. Trace the function step by line
        3. Determine the exact return value
        4. Compare to expected
        5. Mark passed/failed

        ACCURACY RULES:
        - Do not invent behavior not present in the function code
        - If input causes an error (IndexError, TypeError, ZeroDivisionError, etc.), mark passed=false
        - If you cannot determine the output with certainty, mark passed=false and explain why

        OUTPUT:
        - verdict: "approved" if pass_tests == {total_tests}, otherwise "pending"
        - pass_tests: count of tests that passed
        - feedback: One concise paragraph (2-3 lines) explaining what to fix. Reference failed test names. If verdict is "approved", feedback can be "All tests pass."
        """)

    chain = prompt | str_runner_model 
    response = chain.invoke({"code": state['newcode'],"total_tests": state["sub_total_tests"], "tests": state["sub_tests"]})

    return {"code_feedback": response.feedback, "pass_tests": response.pass_tests, "verdict": response.verdict, "attempt" : state["attempt"]+1}

def approve_node(state:CoderState) -> CoderState: 
    if state["sub_total_tests"] == state["pass_tests"] : 
        state["verdict"] = "approved"

    state["verdict"] = state["verdict"]
    return state

MAX_ATTEMPTS = 7

def route_verdict(state: CoderState) -> str:
    if state["verdict"] == "approved":
        return "approved"
    if state["attempt"] > MAX_ATTEMPTS:
        return "approved"   
    return "pending"


#sub graph

codergraph_builder = StateGraph(CoderState)
codergraph_builder.add_node("coder", code_writer_node)
codergraph_builder.add_node("runner", code_runner_node)
codergraph_builder.add_node("approve", approve_node)

#Sub edges 
codergraph_builder.add_edge(START, "coder")
codergraph_builder.add_edge("coder", "runner")
codergraph_builder.add_edge("runner", "approve")
codergraph_builder.add_conditional_edges("approve", route_verdict, {'approved': END, 'pending': "coder"})

#codergraph
codergraph = codergraph_builder.compile()

def final_code_node(state:TestState) -> TestState: 
    coder_initial_state: CoderState = {
        "precode": "",
        "newcode": "",
        "code_feedback": "",
        "attempt": 1,
        "verdict": "pending",
        "sub_tests": state["tests"],
        "sub_total_tests": state["total_tests"],
        "pass_tests": 0,
        "sub_coder_prompt" : state["coder_prompt"]
        }
    
    final_code = codergraph.invoke(coder_initial_state)

    return {
        "final_code": final_code["newcode"],
        "verdict": final_code["verdict"],
        "pass_tests": final_code["pass_tests"],
        "attempt": final_code["attempt"],
        "code_feedback": final_code["code_feedback"],
    }


#graph 

graph = StateGraph(TestState)

#add nodes 
graph.add_node("summarizer", summarize_query_node)
graph.add_node("coder", final_code_node)
graph.add_node("tester", test_designer_node)


#add edges
graph.add_edge(START, "summarizer")
graph.add_edge("summarizer","tester")
graph.add_edge("tester","coder")
graph.add_edge("coder", END)



#workflow 
workflow = graph.compile()


if __name__ == "__main__":
    user_query = input("Enter your query: ")

    initial_state: TestState = {
        "query": user_query,
        "tests": [],
        "total_tests": 0,
        "coder_prompt": "",
        "tester_prompt": "",
        "final_code": ""
    }
    
    result = workflow.invoke(initial_state)
    
    print("=" * 60)
    print(f"FINAL CODE:")
    print("=" * 60)
    print(result['final_code'])
    print("\n" + "=" * 60)
    print(f"Total Tests: {result['total_tests']}")
    print("Tests Generated:")
    for t in result['tests']:
        print(t)
    print("=" * 60)