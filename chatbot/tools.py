"""
Tools bound to the chatbot's LLM.

Kept intentionally simple - no external APIs, no extra dependencies.
Add new tools here and register them in TOOLS at the bottom.
This is the exact spot where a `retriever_tool` for RAG will slot in later:
build the FAISS retriever elsewhere, wrap it in a @tool function the same
way these are wrapped, then add it to TOOLS.
"""

from datetime import datetime

from ddgs import DDGS
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (4 + 3) / 2'.
    Supports +, -, *, /, ** and parentheses."""
    allowed = "0123456789+-*/(). "
    if not all(ch in allowed for ch in expression):
        return "Only numbers and + - * / ( ) are allowed."
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"


@tool
def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


@tool
def web_search(query: str) -> str:
    """Search the web for current information and return the top results
    (title + short snippet + link) for a given query."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return f"No results found for '{query}'."
        lines = []
        for r in results:
            lines.append(f"- {r['title']}: {r['body']} ({r['href']})")
        return "\n".join(lines)
    except Exception as e:
        return f"Web search failed for '{query}': {e}"


@tool
def count_words(text: str) -> str:
    """Count the number of words and characters in a piece of text."""
    words = len(text.split())
    chars = len(text)
    return f"{words} words, {chars} characters"


# ---------------------------------------------------------------------------
# Registered tools - append your RAG retriever tool to this list
# ---------------------------------------------------------------------------

TOOLS = [calculator, get_current_time, web_search, count_words]