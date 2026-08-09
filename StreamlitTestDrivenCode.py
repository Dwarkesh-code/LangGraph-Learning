import streamlit as st
from TestDrivenCode import workflow, TestState

st.title("Code Writer + Tester Agent")

user_query = st.text_input("Enter your query")

if st.button("Run"):
    if not user_query.strip():
        st.warning("Please enter a query first.")
    else:
        initial_state: TestState = {
            "query": user_query,
            "tests": [],
            "precode": "",
            "newcode": "",
            "total_tests": 0,
            "verdict": "pending",
            "coder_prompt": "",
            "tester_prompt": "",
            "code_feedback": "",
            "attempt": 1,
            "pass_tests": 0
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
                    status_box.write(f"Coder: wrote code (attempt {final_state.get('attempt', 1)})")
                    code_placeholder.code(node_output.get("newcode", ""), language="python")

                elif node_name == "runner":
                    status_box.write(
                        f"Runner: {node_output.get('pass_tests', 0)}/{final_state.get('total_tests', 0)} tests passed "
                        f"-> verdict: {node_output.get('verdict', '')}"
                    )
                    with status_box.expander("Runner feedback"):
                        st.write(node_output.get("code_feedback", ""))

                elif node_name == "approve":
                    status_box.write(f"Approve node: final verdict = {node_output.get('verdict', '')}")

                final_state = {**final_state, **node_output}

        status_box.update(label="Workflow finished", state="complete", expanded=False)

        result_placeholder.subheader(f"Final Verdict: {final_state['verdict']}")
        st.write(f"Passed: {final_state['pass_tests']}/{final_state['total_tests']}")

        st.subheader(f"Final Code (attempt {final_state['attempt']})")
        st.code(final_state['newcode'], language="python")

        st.subheader("Feedback")
        st.write(final_state['code_feedback'])