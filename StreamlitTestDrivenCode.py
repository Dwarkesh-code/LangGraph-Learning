import streamlit as st
from TestDrivenCode import workflow, TestState
import os

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.title("Code Writer + Tester Agent")

user_query = st.text_input("Enter your query")

if st.button("Run"):
    if not user_query.strip():
        st.warning("Please enter a query first.")
    else:
        initial_state: TestState = {
            "query": user_query,
            "tests": [],
            "total_tests": 0,
            "coder_prompt": "",
            "tester_prompt": "",
            "final_code": "",
            "verdict": "pending",
            "pass_tests": 0,
            "attempt": 1,
            "code_feedback": "",
        }

        status_box = st.status("Starting workflow...", expanded=True)
        code_placeholder = st.empty()
        tests_placeholder = st.empty()
        result_placeholder = st.empty()

        final_state = initial_state

        for step in workflow.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in step.items():

                if node_name == "summarizer":
                    status_box.write("Summarizer: broke query into coder + tester prompts")
                    with status_box.expander("Coder prompt"):
                        st.write(node_output.get("coder_prompt", ""))
                    with status_box.expander("Tester prompt"):
                        st.write(node_output.get("tester_prompt", ""))

                elif node_name == "tester":
                    status_box.write(f"Tester: designed {node_output.get('total_tests', 0)} test cases")
                    tests_placeholder.code(
                        "\n".join(node_output.get("tests", [])),
                        language="text"
                    )

                elif node_name == "coder":
                    # this node runs the full internal retry loop (coder -> runner -> approve)
                    # and only reports back once it's fully done
                    status_box.write(
                        f"Coder: finished after {node_output.get('attempt', 1)} attempt(s) "
                        f"-> verdict: {node_output.get('verdict', '')}"
                    )
                    code_placeholder.code(node_output.get("final_code", ""), language="python")
                    with status_box.expander("Final feedback"):
                        st.write(node_output.get("code_feedback", ""))

                final_state = {**final_state, **node_output}

        status_box.update(label="Workflow finished", state="complete", expanded=False)

        result_placeholder.subheader(f"Final Verdict: {final_state['verdict']}")
        st.write(f"Passed: {final_state['pass_tests']}/{final_state['total_tests']}")

        st.subheader(f"Final Code (attempt {final_state['attempt']})")
        st.code(final_state['final_code'], language="python")

        st.subheader("Feedback")
        st.write(final_state['code_feedback'])