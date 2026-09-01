"""
link_extractor.py

Preprocessing tool for "After Lecture AI".
Takes raw user input (a URL, a playlist link, or messy pasted text with
video titles) and returns a clean list of YouTube video IDs.

No LLM call happens here unless the input has no extractable URLs at all
(i.e. it's just plain title text) — deterministic parsing is tried first.
"""

import re
from typing import List, Optional
from pydantic import BaseModel, Field

# youtube-search: no API key needed, good enough for title -> video resolution
# pip install youtube-search
from youtube_search import YoutubeSearch

# pytube: used only to expand a playlist URL into its individual video URLs
# pip install pytube
from pytube import Playlist

from langchain_core.tools import tool


VIDEO_URL_PATTERNS = [
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
]
PLAYLIST_URL_PATTERN = r"youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)"


class ExtractedTitles(BaseModel):
    """Structured output schema for LLM-based title extraction."""
    titles: List[str] = Field(
        description="List of distinct video titles the user pasted or referenced, "
        "cleaned of view counts, channel names, durations, and other metadata noise."
    )


def _find_video_ids(text: str) -> List[str]:
    ids = []
    for pattern in VIDEO_URL_PATTERNS:
        ids.extend(re.findall(pattern, text))
    return list(dict.fromkeys(ids))  # dedupe, preserve order


def _find_playlist_ids(text: str) -> List[str]:
    return list(dict.fromkeys(re.findall(PLAYLIST_URL_PATTERN, text)))


def _expand_playlist(playlist_url: str) -> List[str]:
    """Given a playlist URL, return all video IDs in it."""
    pl = Playlist(playlist_url)
    ids = []
    for url in pl.video_urls:
        match = re.search(VIDEO_URL_PATTERNS[0], url)
        if match:
            ids.append(match.group(1))
    return ids


def _resolve_title_to_id(title: str) -> Optional[str]:
    """Search YouTube for a title and return the top result's video ID."""
    results = YoutubeSearch(title, max_results=1).to_dict()
    if not results:
        return None
    return results[0]["id"]


def _extract_titles_with_llm(raw_text: str, llm) -> List[str]:
    """
    Fallback for when the input has no URLs at all — just pasted title text
    (e.g. copy-pasted from a YouTube search/recommendation page).
    `llm` should be a LangChain chat model already bound with rate limiter.
    """
    structured_llm = llm.with_structured_output(ExtractedTitles)
    prompt = (
        "The user pasted text copied from a YouTube page. Extract only the "
        "actual video titles, ignoring view counts, upload dates, channel "
        "names, and durations.\n\nText:\n" + raw_text
    )
    result: ExtractedTitles = structured_llm.invoke(prompt)
    return result.titles


def _extract_video_ids_impl(raw_input: str, llm) -> List[str]:
    """Core logic, kept separate from the @tool wrapper so `llm` can be
    bound via closure instead of being an LLM-facing argument."""
    video_ids: List[str] = []

    playlist_ids = _find_playlist_ids(raw_input)
    if playlist_ids:
        for pid in playlist_ids:
            playlist_url = f"https://www.youtube.com/playlist?list={pid}"
            video_ids.extend(_expand_playlist(playlist_url))

    direct_ids = _find_video_ids(raw_input)
    video_ids.extend(direct_ids)

    if not video_ids:
        if llm is None:
            raise ValueError(
                "No URLs found in input and no LLM provided to extract "
                "titles from plain text."
            )
        titles = _extract_titles_with_llm(raw_input, llm)
        for title in titles:
            vid = _resolve_title_to_id(title)
            if vid:
                video_ids.append(vid)

    return list(dict.fromkeys(video_ids))  # final dedupe, preserve order


def make_link_extractor_tool(llm):
    """
    Factory that returns the @tool-decorated function with `llm` bound via
    closure. Call this once when building the graph:

        link_extractor = make_link_extractor_tool(my_rate_limited_llm)
        tools = [link_extractor, ...]
    """

    @tool
    def extract_video_ids(raw_input: str) -> List[str]:
        """
        Extract YouTube video IDs from user input of any form: a direct
        video URL, a playlist URL, or messy pasted text containing video
        titles (e.g. copied from a YouTube search/recommendations page).

        Order of attempts:
          1. Playlist URL present -> expand it (pytube), no LLM needed
          2. Direct video URL(s) present -> regex extract, no LLM needed
          3. Neither -> assume raw text is pasted titles -> LLM extraction
             -> resolve each title via YouTube search

        Returns a deduped list of video IDs.
        """
        return _extract_video_ids_impl(raw_input, llm)

    return extract_video_ids


if __name__ == "__main__":
    # quick manual test — replace with a real playlist URL, video URL, or
    # pasted title block to sanity check each branch
    link_extractor = make_link_extractor_tool(llm=None)
    sample = "https://www.youtube.com/playlist?list=REPLACE_ME"
    print(link_extractor.invoke({"raw_input": sample}))