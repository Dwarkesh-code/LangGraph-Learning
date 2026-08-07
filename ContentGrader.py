import streamlit as st
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from operator import add
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")


# ---------- Structured output schema ----------

class FeedScorSchema(BaseModel):
    feedback: str = Field(description="Detailed Feedback on given paragraph")
    score: int = Field(description="Score out of 10 based on given paragraph", ge=0, le=10)


structure_model = model.with_structured_output(FeedScorSchema)


# ---------- State ----------

class ContentState(TypedDict):
    paragraph: str
    clarity_feed: str
    depth_feed: str
    engage_feed: str
    individual_scores: Annotated[list[int], add]
    overall_feedback: str


# ---------- Nodes ----------

def clarity_node(state: ContentState) -> ContentState:
    prompt = PromptTemplate(
        template="""Evaluate the clarity and non-technical person's understanding of the
            following paragraph and provide a feedback and score out of 10
            Paragraph => \n{paragraph}""",
        input_variables=['paragraph']
    )
    chain = prompt | structure_model
    output = chain.invoke({"paragraph": state['paragraph']})
    return {"clarity_feed": output.feedback, "individual_scores": [output.score]}


def depth_node(state: ContentState) -> ContentState:
    prompt = PromptTemplate(
        template="""Evaluate the depth, technical terms and accuracy of the
            following paragraph and provide a feedback and score out of 10
            Paragraph => \n{paragraph}""",
        input_variables=['paragraph']
    )
    chain = prompt | structure_model
    output = chain.invoke({"paragraph": state['paragraph']})
    return {"depth_feed": output.feedback, "individual_scores": [output.score]}


def engage_node(state: ContentState) -> ContentState:
    prompt = PromptTemplate(
        template="""Evaluate the engagement/hook of the
            following paragraph and provide a feedback and score out of 10
            Paragraph => \n{paragraph}""",
        input_variables=['paragraph']
    )
    chain = prompt | structure_model
    output = chain.invoke({"paragraph": state['paragraph']})
    return {"engage_feed": output.feedback, "individual_scores": [output.score]}


def overall_node(state: ContentState) -> ContentState:
    prompt = PromptTemplate(
        template="""Generate overall feedback of paragraph based on feedbacks and scores
            \nParagraph => \n{paragraph}
            \nClarity Feedback => {clarity}
            \nTechnical Depth Feedback => {depth}
            \nEngagement/hook Feedback => {engage}
            \nIndividual scores => {scores}""",
        input_variables=['paragraph', 'clarity', 'depth', 'engage', 'scores']
    )
    chain = prompt | model
    output = chain.invoke({
        "paragraph": state['paragraph'],
        "clarity": state['clarity_feed'],
        "depth": state['depth_feed'],
        "engage": state["engage_feed"],
        "scores": state['individual_scores']
    })
    return {"overall_feedback": output.content}


# ---------- Graph (cached so it's built once) ----------

@st.cache_resource
def get_workflow():
    graph = StateGraph(ContentState)

    graph.add_node("clarity", clarity_node)
    graph.add_node("depth", depth_node)
    graph.add_node("engage", engage_node)
    graph.add_node("overall", overall_node)

    graph.add_edge(START, "clarity")
    graph.add_edge(START, "depth")
    graph.add_edge(START, "engage")

    graph.add_edge("clarity", "overall")
    graph.add_edge("depth", "overall")
    graph.add_edge("engage", "overall")
    graph.add_edge("overall", END)

    return graph.compile()


# ---------- UI ----------

st.set_page_config(page_title="Content Grader", page_icon="📝")
st.title("ContentGrader")
st.write("Paste a paragraph (LinkedIn post, essay, writeup) and get it graded on clarity, depth, and engagement — in parallel.")

workflow = get_workflow()

paragraph = st.text_area("Your paragraph", height=200, placeholder="Paste your writeup here...")

if st.button("Grade it", disabled=not paragraph.strip()):
    initial_state = {
        "paragraph": paragraph,
        "clarity_feed": "",
        "depth_feed": "",
        "engage_feed": "",
        "individual_scores": [],
        "overall_feedback": ""
    }

    with st.spinner("Running parallel evaluation..."):
        res = workflow.invoke(initial_state)

    st.subheader("Overall Feedback")
    st.write(res["overall_feedback"])

    avg_score = sum(res["individual_scores"]) / len(res["individual_scores"])
    st.metric("Average Score", f"{avg_score:.1f} / 10")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Clarity**")
        st.write(res["clarity_feed"])
    with col2:
        st.markdown("**Depth**")
        st.write(res["depth_feed"])
    with col3:
        st.markdown("**Engagement**")
        st.write(res["engage_feed"])

    with st.expander("Individual scores"):
        st.write(res["individual_scores"])