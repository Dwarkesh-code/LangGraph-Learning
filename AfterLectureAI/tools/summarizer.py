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


class StructureOutput(BaseModel):
    summaries: dict[str, str] = Field(
        description=(
            "One entry per video, in the same order as the input videos. "
            "Key: a very short summary (under 10 words) capturing the core/gist of the video. "
            "Value: the full, detailed summary of the video's actual content."
        )
    )

def make_summarize_videos(llm):
    struc_llm = llm.with_structured_output(StructureOutput)

    @tool
    def summarize_videos(state: Annotated[RouterState, InjectedState],tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """
        Summarize all video transcripts stored in the state.

        Reads the pre-chunked transcripts from `state["chunks"]` and generates a
        concise summary for each video using the LLM. For every chunk, it invokes
        the structured-output chain to produce a dict of {short_summary_key: full_summary}
        pairs, then merges all results into a single summaries dict.

        Use this tool when the video transcripts have already been fetched and
        chunked, and the next step is to generate summaries for each video.

        Updates state:
            - summaries: dict[str, str] — merged summaries from all chunks
            - messages: confirmation ToolMessage

        Returns:
            Command with the updated state.
        """


        prompt = PromptTemplate.from_template("""
            You are given multiple video transcripts below. Each is prefixed with a [VIDEO N: <link>] marker.
            Generate a separate concise summary for EACH video. Return the summaries in the EXACT same order as the videos appear below.
            For each summary:
            - Summary length should be 1/10th of the input transcript length (e.g., 1000 words -> ~100 words)
            - Focus on the main topic, key points, and actionable insights
            - Preserve important facts, numbers, names, and examples
            Videos:
            {videos}
            Return your response strictly as a list of summaries, one per video, in the same order.

        """)

        chain = prompt | struc_llm

        chunks = state["chunks"]

        summaries = {}

        for c in chunks :
            result = chain.invoke({"videos": c}).summaries
            summaries.update(result)

        return Command(
            update={
                "summaries" : summaries,
                "messages" : [
                        ToolMessage(
                            content="Done, all summaries generated from this tool using Chunks and transcripts and store in State ",
                            tool_call_id=tool_call_id,
                        )
                    ]
            }
        )
