"""
link_extractor.py

Preprocessing tool for "After Lecture AI".
Two deterministic extraction paths — no LLM involved at all:

  1. extract_links_from_playlist(url) -> all video URLs from a playlist
  2. extract_links_from_raw_text(text) -> any video URLs found inside
     arbitrary/messy pasted text (regex based)
"""

import re
from typing import List, Annotated

from pytube import Playlist
from state import RouterState
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage

VIDEO_URL_PATTERN = r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})"
PLAYLIST_URL_PATTERN = r"youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)"


def extract_links_from_playlist(playlist_url: str) -> List[str]:
    """
    Takes a YouTube playlist URL and returns all video URLs (full https
    links) contained in it, in playlist order, with duplicates removed.
    """
    if not re.search(PLAYLIST_URL_PATTERN, playlist_url):
        raise ValueError("No valid playlist ID found in the given URL.")

    playlist = Playlist(playlist_url)
    video_urls = list(dict.fromkeys(playlist.video_urls))  # dedupe, preserve order
    return video_urls


def extract_links_from_raw_text(raw_text: str) -> List[str]:
    """
    Takes any raw pasted text (a single URL, multiple URLs, or a messy
    block copied from a YouTube page with titles/views/channel names
    mixed in) and returns the distinct video URLs found inside it.
    """
    video_ids = list(dict.fromkeys(re.findall(VIDEO_URL_PATTERN, raw_text)))
    video_urls = [f"https://www.youtube.com/watch?v={vid}" for vid in video_ids]
    return video_urls


@tool
def links_extractor(
    state: Annotated[RouterState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    raw_data: str,
    func_choose: bool,
) -> Command:
    """
    Extract YouTube video links and store them in state.

    Use this tool whenever the user provides either a YouTube playlist
    link or raw text/links that need video URLs extracted from them.

    Args:
        raw_data: The playlist URL, or the raw pasted text/link(s)
            to extract video URLs from.
        func_choose: True if raw_data is a playlist URL (extracts every
            video in the playlist). False if raw_data is raw text or a
            direct link (extracts video URLs found inside it via regex).
    """
    if func_choose:
        return Command(
            update={
                "links": extract_links_from_playlist(raw_data),
                "messages": [
                    ToolMessage(
                        content="Done, links were extracted from the playlist link successfully.",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return Command(
        update={
            "links": extract_links_from_raw_text(raw_data),
            "messages": [
                ToolMessage(
                    content="Done, links were extracted from the raw text successfully.",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


if __name__ == "__main__":
    text_or_url = input("Link/text : ")
    if "playlist?list=" in text_or_url:
        print(extract_links_from_playlist(text_or_url))
    else:
        print(extract_links_from_raw_text(text_or_url))