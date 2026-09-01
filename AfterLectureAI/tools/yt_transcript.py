"""
transcript_fetcher.py

Fetches the transcript for one or more YouTube video IDs.
Used right after link_extractor resolves raw user input into video IDs.

pip install youtube-transcript-api
"""

from typing import List, Dict, Optional
from langchain_core.tools import tool

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)


def _fetch_single_transcript(video_id: str) -> Optional[str]:
    """
    Returns the plain-text transcript for one video, or None if unavailable
    (transcripts disabled, video removed, no captions in any usable language).
    """
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception:
        # catch-all so one bad video doesn't crash a whole playlist batch
        return None

    return " ".join(seg["text"] for seg in segments)


@tool
def fetch_transcripts(video_ids: List[str]) -> Dict[str, Optional[str]]:
    """
    Fetch transcripts for a list of YouTube video IDs.

    Returns a dict mapping video_id -> transcript text (or None if that
    video's transcript couldn't be fetched, e.g. captions disabled).

    Call this after the link extractor tool has resolved raw input into
    a clean list of video IDs.
    """
    results: Dict[str, Optional[str]] = {}
    for vid in video_ids:
        results[vid] = _fetch_single_transcript(vid)
    return results


if __name__ == "__main__":
    # quick manual test
    test_ids = ["dQw4w9WgXcQ"]  # replace with a real ID to sanity check
    print(fetch_transcripts.invoke({"video_ids": test_ids}))