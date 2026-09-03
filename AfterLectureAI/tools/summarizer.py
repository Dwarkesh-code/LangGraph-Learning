from langchain_core.tools import tool
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state import RouterState
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field
from typing import Annotated
from langchain_core.prompts import PromptTemplate
import json


class StructureOutput(BaseModel):
    summaries: dict[str, str] = Field(
        description=(
            "One entry per video, in the same order as the input videos. "
            "Key: a very short summary (under 10 words) capturing the core/gist of the video. "
            "Value: the full, detailed summary of the video's actual content."
        )
    )
    core_keywords: dict[str, list[str]] = Field(
        description=(
            "One entry per video, using the SAME keys as `summaries`. "
            "Value: a list of 3-5 core topic/technology/tool keywords extracted from that "
            "video's summary — specific enough to be used later for targeted project-idea search "
            "(e.g. 'RAG pipeline', 'FastAPI', 'vector database'), not generic words like 'AI' or 'tutorial'."
        )
    )


def make_summarize_videos(llm):
    struc_llm = llm.with_structured_output(StructureOutput)

    @tool
    def summarize_videos(state: Annotated[RouterState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """
        Summarize all video transcripts stored in the state, and extract core keywords.

        Reads the pre-chunked transcripts from `state["chunks"]` and, for each chunk,
        generates a concise summary AND a list of core topic/tech keywords per video
        using the LLM's structured output. Results from all chunks are merged into
        a single `summaries` dict and a single `core_keywords` dict (same keys).

        Use this tool when the video transcripts have already been fetched and
        chunked, and the next step is to generate summaries and keywords for each video.

        Updates state:
            - summaries: dict[str, str] — merged summaries from all chunks
            - core_keywords: dict[str, list[str]] — merged core keywords from all chunks
            - messages: confirmation ToolMessage containing both summaries and keywords

        Returns:
            Command with the updated state.
        """

        prompt = PromptTemplate.from_template("""
            You are given multiple video transcripts below. Each is prefixed with a [VIDEO N: <link>] marker.
            For EACH video, generate:
            1. A concise summary
            2. A list of 3-5 core topic/technology/tool keywords from that video

            Return the summaries and keywords in the EXACT same order as the videos appear below,
            using the same key for a video in both `summaries` and `core_keywords`.

            For each summary:
            - Summary length should be 1/10th of the input transcript length (e.g., 1000 words -> ~100 words)
            - Focus on the main topic, key points, and actionable insights
            - Preserve important facts, numbers, names, and examples

            For each keyword list:
            - Extract specific, searchable terms (tools, techniques, concepts) mentioned in the video
            - Avoid generic/vague words like "AI", "tutorial", "programming"

            Videos:
            {videos}
        """)

        chain = prompt | struc_llm

        chunks = state["chunks"]

        summaries = {}
        core_keywords = {}

        for c in chunks:
            result = chain.invoke({"videos": c})
            summaries.update(result.summaries)
            core_keywords.update(result.core_keywords)

        return Command(
            update={
                "summaries": summaries,
                "core_keywords": core_keywords,
                "messages": [
                    ToolMessage(
                        content=(
                            f"Done, all summaries and core keywords generated from Chunks and transcripts, stored in State.\n\n"
                            f"Summaries =>\n{json.dumps(summaries, indent=2)}\n\n"
                            f"Core Keywords =>\n{json.dumps(core_keywords, indent=2)}"
                        ),
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    return summarize_videos