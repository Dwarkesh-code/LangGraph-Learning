"""
AfterLectureAI — Streamlit UI with full pipeline observability.

Run: streamlit run app.py

Features:
  - Simple chat (no sidebar)
  - Live node-by-node status while pipeline runs
  - Collapsed dropdowns with:
      * Stats (chunks, tokens, LLM calls — overall + per-node)
      * Node trace (which nodes ran, in order)
      * LLM prompts (system + user + handoff)
  - "New chat" button resets the thread

Requirements:
    pip install streamlit
"""

import os
import sys
import uuid
import streamlit as st

# Make sure we can import from this folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import main_graph
from prompts import ROUTER_SYSTEM_PROMPT, MAIN_LLM_SYSTEM_PROMPT
from langchain_core.messages import AIMessage


# ---------- Page config ----------
st.set_page_config(
    page_title="AfterLectureAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------- Session state ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_obs" not in st.session_state:
    st.session_state.last_obs = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""


def new_chat():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.last_obs = None


def approx_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars/token for English."""
    return len(text) // 4


def run_pipeline(query: str, thread_id: str, status_box):
    """
    Stream the graph; accumulate observability data + grab the final answer
    from main_llm's last AIMessage (no need for a second invoke).
    """
    initial_state = {
        "query": query,
        "messages": [],
        "transcript_summary": "",
        "main_llm_prompt": "",
        "final_output": "",
        "chunks": [],
        "links": [],
    }
    config = {"configurable": {"thread_id": thread_id}}

    obs = {
        "node_trace": [],          # ordered list of nodes that ran
        "llm_calls": 0,
        "token_usage": {
            "input": 0,
            "output": 0,
            "reasoning": 0,
            "total": 0,
        },
        "node_tokens": {},         # per-node breakdown
        "chunks": [],
        "links": [],
        "main_llm_prompt": "",
    }

    final_answer = ""

    for event in main_graph.stream(initial_state, config=config):
        for node_name, node_output in event.items():
            if node_name == "__end__":
                continue

            # Track this node visit
            obs["node_trace"].append(node_name)

            if node_name not in obs["node_tokens"]:
                obs["node_tokens"][node_name] = {
                    "input": 0, "output": 0, "reasoning": 0, "calls": 0,
                }

            status_box.write(f"⚙️ **{node_name}** finished")

            if not isinstance(node_output, dict):
                continue

            # Capture state updates from router_step (chunks, links, handoff prompt)
            if "chunks" in node_output:
                obs["chunks"] = node_output["chunks"]
            if "links" in node_output:
                obs["links"] = node_output["links"]
            if "main_llm_prompt" in node_output:
                obs["main_llm_prompt"] = node_output["main_llm_prompt"]

            # Track LLM calls + token usage from any AIMessage in the output
            for msg in node_output.get("messages", []):
                if not isinstance(msg, AIMessage):
                    continue

                obs["llm_calls"] += 1
                obs["node_tokens"][node_name]["calls"] += 1

                um = msg.usage_metadata or {}
                in_t = um.get("input_tokens", 0) or 0
                out_t = um.get("output_tokens", 0) or 0
                reas_t = (um.get("output_token_details") or {}).get("reasoning", 0) or 0

                obs["token_usage"]["input"] += in_t
                obs["token_usage"]["output"] += out_t
                obs["token_usage"]["reasoning"] += reas_t
                obs["token_usage"]["total"] += um.get("total_tokens", 0) or 0

                obs["node_tokens"][node_name]["input"] += in_t
                obs["node_tokens"][node_name]["output"] += out_t
                obs["node_tokens"][node_name]["reasoning"] += reas_t

                # Last main_llm AIMessage = final answer
                if node_name == "main_llm" and msg.content:
                    final_answer = msg.content

    return obs, final_answer


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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("🧠 Running pipeline...", expanded=True)
        status.write("🔍 Router will extract links → fetch transcripts → summarize → search project ideas...")
        status.write("⏱️ Big playlists can take 10–15 minutes. Hang tight.")

        try:
            obs, final_answer = run_pipeline(prompt, st.session_state.thread_id, status)

            if not final_answer:
                final_answer = obs["main_llm_prompt"] or "_(no response generated)_"

            status.update(label="✅ Done", state="complete")
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

            # Save for the observability section below
            st.session_state.last_obs = obs
            st.session_state.last_query = prompt

        except Exception as e:
            status.update(label="❌ Error", state="error")
            err_text = f"```\n{type(e).__name__}: {str(e)}\n```"
            st.error(err_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ **Error:**\n\n{err_text}",
            })


# ---------- Observability (collapsed dropdowns) ----------
if st.session_state.last_obs:
    obs = st.session_state.last_obs

    st.divider()
    st.subheader("🔬 Pipeline observability")
    st.caption("Click any section to expand 👇")

    # ── 1. Stats ─────────────────────────────────────────────────────────────
    with st.expander("📊 Stats — chunks, tokens, LLM calls", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Chunks", len(obs["chunks"]))
        with c2:
            st.metric("LLM calls", obs["llm_calls"])
        with c3:
            st.metric("Total tokens", f"{obs['token_usage']['total']:,}")
        with c4:
            st.metric("Videos found", len(obs["links"]))

        # Token breakdown
        st.markdown("**Token breakdown (entire run):**")
        st.dataframe(
            {
                "Type": ["Input", "Output (answer)", "Reasoning", "Total"],
                "Tokens": [
                    obs["token_usage"]["input"],
                    obs["token_usage"]["output"] - obs["token_usage"]["reasoning"],
                    obs["token_usage"]["reasoning"],
                    obs["token_usage"]["total"],
                ],
            },
            use_container_width=True,
            hide_index=True,
        )

        # Per-node breakdown
        if obs["node_tokens"]:
            st.markdown("**Per-node breakdown:**")
            rows = []
            for node, t in obs["node_tokens"].items():
                rows.append({
                    "Node": node,
                    "LLM calls": t["calls"],
                    "Input": t["input"],
                    "Output": t["output"],
                    "Reasoning": t["reasoning"],
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

        # Per-chunk size
        if obs["chunks"]:
            st.markdown(f"**Per-chunk size** ({len(obs['chunks'])} chunks):")
            st.dataframe(
                {
                    "#": list(range(1, len(obs["chunks"]) + 1)),
                    "Chars": [len(c) for c in obs["chunks"]],
                    "Approx tokens (~4 chars/token)": [approx_tokens(c) for c in obs["chunks"]],
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No chunks produced (no playlist / transcripts not fetched).")

    # ── 2. Node trace ────────────────────────────────────────────────────────
    with st.expander("🔄 Node trace — which nodes ran, in order", expanded=False):
        st.write(f"**Total node executions:** {len(obs['node_trace'])}")
        st.write("**Order:**  " + " → ".join(f"`{n}`" for n in obs["node_trace"]))
        st.markdown("**Step-by-step:**")
        for i, n in enumerate(obs["node_trace"], 1):
            t = obs["node_tokens"].get(n, {})
            meta = (
                f"  —  {t.get('calls', 0)} LLM call(s), "
                f"~{t.get('input', 0) + t.get('output', 0):,} tokens"
                if t.get("calls", 0)
                else "  —  no LLM call"
            )
            st.write(f"{i}. `{n}`{meta}")

    # ── 3. LLM Prompts ───────────────────────────────────────────────────────
    with st.expander("📝 LLM Prompts — what was sent to the LLM", expanded=False):
        st.markdown("### 🔵 Router System Prompt")
        st.caption(f"({len(ROUTER_SYSTEM_PROMPT):,} chars)")
        st.code(ROUTER_SYSTEM_PROMPT, language="markdown")

        st.markdown("### 🟢 Main LLM System Prompt")
        st.caption(f"({len(MAIN_LLM_SYSTEM_PROMPT):,} chars)")
        st.code(MAIN_LLM_SYSTEM_PROMPT, language="markdown")

        st.markdown("### 👤 User query")
        st.code(st.session_state.last_query, language="text")

        st.markdown("### 🤝 Router → Main LLM handoff prompt")
        handoff = obs["main_llm_prompt"]
        if handoff:
            display = handoff if len(handoff) < 4000 else handoff[:4000] + "\n\n... (truncated for display)"
            st.caption(f"({len(handoff):,} chars total)")
            st.code(display, language="markdown")
        else:
            st.info("No handoff prompt captured.")
