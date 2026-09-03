from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
load_dotenv()




@tool
def searcher(queries: list[str], max_results: int) -> str:
    """
    Search the web for real, specific project ideas or general knowledge you don't have.
    Always send ALL your queries together in a single call as a list — even if you only
    have one query, still send it as a list with one item. Do NOT call this tool
    multiple times for multiple queries.

    Use this tool in two scenarios:
    1. PROJECT IDEA DISCOVERY: After extracting core keywords/topics from video summaries,
        build one targeted query per keyword, combining it with relevant platforms where
        real people discuss real project ideas. Example query format:
            "{keyword} project ideas site:reddit.com OR site:news.ycombinator.com"
        This surfaces actual, specific, recent ideas people have built or discussed —
        NOT generic textbook project lists.
    2. GENERAL KNOWLEDGE GAP: If you don't have enough knowledge about a topic/tool/library
        mentioned in the video summaries to make a good suggestion, add a query for it too
        in the same list.

    Args:
        queries: A list of all search queries you want run, sent together in one call.
            For project ideas, combine each core keyword with site: filters
            (reddit.com, news.ycombinator.com, github.com, devpost.com) using OR to
            search multiple platforms per query. Keep each query specific and
            topic-grounded — avoid vague queries like "AI project ideas".
        max_results: How many results to fetch PER query. Decide based on how broad or
            niche the topics are — narrow/niche keywords need fewer (2-3), broad topics
            may need more (5-6). This applies uniformly to every query in the list.

    Returns:
        A formatted string with results grouped under each query.
    """
    tavily = TavilySearch(
        max_results=max_results,
    )

    all_formatted = []

    for query in queries:
        raw_results = tavily.invoke({"query": query})
        results = raw_results.get("results", [])

        if not results:
            all_formatted.append(f"### Query: '{query}'\nNo results found.\n")
            continue

        formatted = [f"### Query: '{query}'"]
        for i, r in enumerate(results, start=1):
            formatted.append(
                f"{i}. {r.get('title', 'No title')}\n"
                f"   URL: {r.get('url', 'N/A')}\n"
                f"   Content: {r.get('content', 'N/A')}\n"
            )
        all_formatted.append("\n".join(formatted))

    return "\n\n".join(all_formatted)
