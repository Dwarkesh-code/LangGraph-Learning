from langgraph.store.sqlite import SqliteStore
from langchain_core.tools import tool
store_cm = SqliteStore.from_conn_string("memory.db")
store = store_cm.__enter__()
store.setup()

@tool
def memory_view(key:str) : 
    """Retrieve a previously saved memory by its exact key.

    Use this when you need the full detail of a specific memory you already
    know the key of (e.g. from a prior memory_add call or a memory search/listing
    tool result). Only use a key that you actually received from one of those —
    never guess or invent a key.

    Args:
        key: The exact key of the memory to look up, exactly as it was returned
            to you (e.g. from memory_add's confirmation or a memory search
            result). MUST be lowercase, hyphen-separated, no spaces — must
            match exactly how it was originally saved.
        """
    item = store.get(("me",), key=key)
    if item is not None:
        return item
    else :
        return "Key is not valid" 

@tool 
def memory_add(key: str, summary: str, detail: str) : 
    """Save a new memory or update an existing one for later recall.

    Use this whenever the user shares a preference, a fact about themselves,
    or any information that should be remembered across future conversations
    (not just useful for the current message).

    Args:
        key: A short unique identifier for this memory. MUST be lowercase,
             use hyphens instead of spaces (e.g. "hinglish-no-fluff", 
             "prefers-sqlite"). No spaces allowed.
        summary: A one-line summary of the memory (a few words to one sentence).
        detail: The full detailed context — enough to fully understand and
                use this memory later without needing the original conversation.
    """
    try :
        store.put(("me",), key, {
            "summary": summary,
            "detail": detail
            })
        return "Memory saved succefully"
    except Exception as e: 
        print("\n\nerror:",e,"\n\n")
        return f"Got an error {e}"