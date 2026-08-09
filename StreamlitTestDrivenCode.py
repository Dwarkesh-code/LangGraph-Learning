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
 
        with st.spinner("Running workflow..."):
            result = workflow.invoke(initial_state)
 
        st.subheader(f"Final Verdict: {result['verdict']}")
        st.write(f"Passed: {result['pass_tests']}/{result['total_tests']}")
 
        st.subheader(f"Final Code (attempt {result['attempt']})")
        st.code(result['newcode'], language="python")
 
        st.subheader("Feedback")
        st.write(result['code_feedback'])
