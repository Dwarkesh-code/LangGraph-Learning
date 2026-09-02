"""
transcript_fetcher.py

Fetches the transcript for one or more YouTube video IDs.
Used right after link_extractor resolves raw user input into video IDs.

pip install youtube-transcript-api
"""

from typing import Dict, Optional, Annotated
from langchain_core.tools import tool
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state import RouterState
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

import re

def get_chunks(results) : 
    tokenizer = AutoTokenizer.from_pretrained("nvidia/nemotron-3-8b-chat-4k")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size= 100000,
        chunk_overlap= 1000
    )

    chunks = []
    for link, transcript in results.items():
        if transcript:
            for text in text_splitter.split_text(transcript):
                chunks.append({"link": link, "text": text})
    return chunks 


def get_video_id(url):
    match = re.search(r'(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/))([\w-]{11})', url)
    return match.group(1) if match else None

def _fetch_single_transcript(video_id: str) -> Optional[str]:
    """
    Returns the plain-text transcript for one video, or None if unavailable
    (transcripts disabled, video removed, no captions in any usable language).
    """
    try:
        segments = YouTubeTranscriptApi().fetch(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception:
        # catch-all so one bad video doesn't crash a whole playlist batch
        return None

    return " ".join(seg.text for seg in segments)


@tool
def fetch_transcripts( state: Annotated[RouterState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
        Fetch YouTube transcripts for all video links already stored in state.
        Call this AFTER the link extractor tool has populated `state["links"]`
        with a clean list of YouTube URLs. No arguments needed — the tool pulls
        video IDs and URLs directly from state.
        Returns:
            A Command that updates `state["transcripts"]` with a dict mapping
            each video ID to its transcript text (or None if unavailable).
        """
    links = state['links']
    results: Dict[str, Optional[str]] = {}
    for link in links:
        vid = get_video_id(link)
        results[link]  = _fetch_single_transcript(vid)

    chunks = get_chunks(results)

    return Command(
                update={
                    "transcripts": results,
                    "messages": [
                        ToolMessage(
                            content="Done, all transcripts and chunks get from this tool and store in State ",
                            tool_call_id=tool_call_id,
                        )
                    ],
                    "chunks" : chunks
                }
            )
    


if __name__ == "__main__":
    # quick manual test
    test_ids = ["TlDHVrTXKKw"]  
    print(fetch_transcripts.invoke({"video_ids": test_ids}))