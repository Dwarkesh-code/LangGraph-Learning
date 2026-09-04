""""
AfterLectureAI — Streamlit UI
Run: streamlit run app.py

Requirements:
    pip install streamlit
"""

import os
import sys
import uuid
import streamlit as st

# Make sure we can import from this folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the compiled graph from main.py
# (main.py ke bottom wale `main_graph.invoke(...)` ko `if __name__ == "__main__":` ke andar rakhna zaroori hai,
#  varna app khulte hi poori pipeline chal jayegi)
from main import main_graph


# ---------- Page config ----------
st.set_page_config(
    page_title="AfterLectureAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",  # no sidebar — bs itna hi
)


# ---------- Session state ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{"role": ..., "content": ...}]


def new_chat():
    """Reset the conversation (new thread_id, empty history)."""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


# ---------- Header ----------
col1, col2 = st.columns([6, 1])
with col1:
    st.title("🎓 AfterLectureAI")
    st.caption("Drop a YouTube playlist link + ask a question → get answers grounded in the videos")
with col2:
    if st.button("🔄 New chat", use_container_width=True, help="Start a fresh conversation"):
        new_chat()
        st.rerun()

st.divider()


# ---------- Chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- Chat input ----------
if prompt := st.chat_input("Paste a YouTube playlist link or ask anything..."):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the graph with a status indicator
    with st.chat_message("assistant"):
        status = st.status("🧠 Processing your request...", expanded=True)
        status.write("🔍 Router is preparing context — links → transcripts → summaries → project ideas...")
        status.write("⏱️ Big playlists can take 10–15 minutes. Hang tight.")

        try:
            initial_state = {
                "query": prompt,
                "messages": [],
                "transcript_summary": "",
                "main_llm_prompt": "",
                "final_output": "",
            }
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            result = main_graph.invoke(initial_state, config=config)

            # Extract final answer from the last AIMessage
            final_answer = ""
            if result.get("messages"):
                last_msg = result["messages"][-1]
                if hasattr(last_msg, "content"):
                    final_answer = last_msg.content

            # Fallback: if for some reason messages is empty, use the handoff prompt
            if not final_answer:
                final_answer = result.get("main_llm_prompt", "_(no response generated)_")

            status.update(label="✅ Done", state="complete")
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            status.update(label="❌ Error", state="error")
            err_text = f"```\n{type(e).__name__}: {str(e)}\n```"
            st.error(err_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ **Error:**\n\n{err_text}",
            })
