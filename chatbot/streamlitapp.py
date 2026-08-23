import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import tempfile
from langgraph.types import Command
from tools import TOOLS as BASE_TOOLS
from rag import gen_vectorstores, make_retriever_tool
from chatbot import build_graph, get_checkpointer, list_threads

st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖")



# file uploader

uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if uploaded_file and "retriever" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.session_state.retriever = gen_vectorstores(tmp_path)

tools = BASE_TOOLS.copy()
if "retriever" in st.session_state:
    tools.append(make_retriever_tool(st.session_state.retriever))

checkpointer = get_checkpointer()
graph = build_graph(checkpointer, tools)

# ---------------------------------------------------------------------------
# Sidebar: thread management (this is the "persistence" piece from the video)
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

st.sidebar.title("Conversations")

if st.sidebar.button("➕ New chat", use_container_width=True):
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()

for tid in list_threads(checkpointer):
    label = tid[:8]
    if st.sidebar.button(label, key=f"thread-{tid}", use_container_width=True):
        st.session_state.thread_id = tid
        st.rerun()

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ---------------------------------------------------------------------------
# Main chat window
# ---------------------------------------------------------------------------

st.title("🤖 LangGraph Chatbot")
st.caption("4 basic tools wired up (calculator, time, web search, word count). RAG goes in next.")

state = graph.get_state(config)
history = state.values.get("messages", []) if state.values else []

for msg in history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    if isinstance(msg, AIMessage) and not msg.content:
        continue
    
    content = msg.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Ask me anything, or give me a sum to calculate...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    current_state = graph.get_state(config)
    if current_state.tasks and current_state.tasks[0].interrupts:
        st.session_state.pending_interrupt = current_state.tasks[0].interrupts[0].value
        st.rerun()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        status = st.empty()
        full_text = ""

        for chunk, metadata in graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node")

            if node == "tools":
                status.caption(f"🔧 using `{getattr(chunk, 'name', 'a tool')}`...")
                continue

            if node == "chat_node" and getattr(chunk, "content", None):
                content = chunk.content
                if isinstance(content, list):
                    content = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                if content:
                    full_text += content
                    placeholder.markdown(full_text)
        status.empty()


if st.session_state.get("pending_interrupt"):
    data = st.session_state.pending_interrupt
    st.warning(data.get("message", "\nApprove this action?"))

    col1, col2 = st.columns(2)
    approve_clicked = col1.button("✅ Approve")
    reject_clicked = col2.button("❌ Reject")

    if approve_clicked or reject_clicked:
        st.session_state.pending_interrupt = None
        for chunk, metadata in graph.stream(
            Command(resume={"approved": approve_clicked}),
            config=config,
            stream_mode="messages",
        ):
            pass
        st.rerun()