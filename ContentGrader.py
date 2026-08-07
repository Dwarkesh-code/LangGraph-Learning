from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import TypedDict, Optional, Annotated
from pydantic import BaseModel, Field
from operator import add 
from langchain_core.prompts import PromptTemplate  

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

#Feedback and scores Structure output
class FeedScorSchema(BaseModel):
    feedback: str = Field(description="Detailed Feedback on given paragraph")
    score : int = Field(description="Score out of 10 based on given paragraph", ge=0, le=10)

structure_model = model.with_structured_output(FeedScorSchema)

#state 
class ContentState(TypedDict):
    paragraph : str
    cliarity_feed : str
    depth_feed : str
    engage_feed : str
    indivisual_scores : Annotated[list[int] ,add]
    overall_feedback : str 


#nodes

def cliarity_node(state: ContentState)-> ContentState: 
    #prompt
    prompt = PromptTemplate(
        template= """Evalute the cliarity and Non technical person's understanding, of the 
            following Paragaraph and provide a feedback and score out of 10 
            Paragraph => \n{paragraph}  """,
        input_variables= ['paragraph']
    )

    chain = prompt| structure_model

    output = chain.invoke({"paragraph": state['paragraph']})

    return {"cliarity_feed": output.feedback, "indivisual_scores": [output.score]}

def depth_node(state: ContentState)-> ContentState: 
    #prompt
    prompt = PromptTemplate(
        template= """Evalute the Depth, Technical terms and accuracy , of the 
            following Paragaraph and provide a feedback and score out of 10 
            Paragraph => \n{paragraph}  """,
        input_variables= ['paragraph']
    )

    chain = prompt| structure_model

    output = chain.invoke({"paragraph": state['paragraph']})

    return {"depth_feed": output.feedback, "indivisual_scores": [output.score]}


def engage_node(state: ContentState)-> ContentState: 
    #prompt
    prompt = PromptTemplate(
        template= """Evalute the engagement/hook of the 
            following Paragaraph and provide a feedback and score out of 10 
            Paragraph => \n{paragraph}  """,
        input_variables= ['paragraph']
    )

    chain = prompt| structure_model

    output = chain.invoke({"paragraph": state['paragraph']})

    return {"engage_feed": output.feedback, "indivisual_scores": [output.score]}



def overall_node(state: ContentState)-> ContentState: 
    #prompt
    prompt = PromptTemplate(
        template= """Generate Overall feedback of paragraph based on feedbacks and scores  
            \nParagraph => \n{paragraph}
            \nCliarity Feedback => {cliarity}
            \nTechnical Depth Feedback => {depth}
            \nEngagement/hook Feedback => {engage}
            \nIndivisual scores => {scores}""",
        input_variables= ['paragraph', 'cliarity', 'depth', 'engage', "scores"]
    )

    chain = prompt| model

    output = chain.invoke({
        "paragraph": state['paragraph'],
        "cliarity": state['cliarity_feed'],
        "depth": state['depth_feed'],
        "engage": state["engage_feed"],
        "scores": state['indivisual_scores']
        })

    return {"overall_feedback": output.content}


#graph 
graph = StateGraph(ContentState)

# add nodes
graph.add_node("cliarity", cliarity_node)
graph.add_node("depth", depth_node)
graph.add_node("engage", engage_node)
graph.add_node("overall", overall_node)

#add edges

graph.add_edge(START, "cliarity")
graph.add_edge(START, "depth")
graph.add_edge(START, "engage")


graph.add_edge("cliarity", "overall")
graph.add_edge("depth", "overall")
graph.add_edge("engage", "overall")

#workflow

workflow = graph.compile()


paragraph1 = """🔍 LimitLens
Real-time Claude.ai session limit & context window monitor

Edge Add-on Manifest V3 JavaScript License: MIT

Install from Microsoft Edge Store →  ·  Report a Bug  ·  Request a Feature

LimitLens Demo

Overview
LimitLens is a lightweight browser extension that injects directly into Claude.ai to give you real-time visibility into your session consumption and context window usage — without any page refreshes.

Power users hitting Claude's message limits mid-conversation know the frustration of being cut off unexpectedly. LimitLens solves that with a clean, non-intrusive progress bar that lives inside the Claude interface and updates with every message.

Know exactly where you stand before Claude cuts you off.

Features
Feature	Description
Real-Time Session Tracking	Live percentage display of your message limit consumption. Updates instantly on every response.
Context Window Monitoring	See how much of the context window your current chat is utilizing.
Peak / Off-Peak Indicator	Know at a glance whether Claude's servers are under high traffic — this directly affects your rate limits.
Session History Logging	Automatic session logs stored locally and viewable from the extension popup.
Native UI Integration	Progress bar injected seamlessly into the Claude interface — not a floating overlay.
Installation
✅ Microsoft Edge Store (Recommended)
Install directly from the Edge Add-ons Store — one click, no setup required.

→ Add to Microsoft Edge

Store ID: 0RDCK9G8C2FR  ·  Version: 2.1.0

🛠 Developer Mode (Chrome / Edge)
Run the latest source code locally:

Clone or download this repository and extract the folder.
Open chrome://extensions (Chrome) or edge://extensions (Edge).
Enable Developer Mode using the toggle in the top-right corner.
Click Load unpacked and select the extracted folder.
Pin LimitLens to your browser toolbar.
How It Works
LimitLens intercepts Claude.ai's internal network responses entirely within the browser. It watches for usage metadata included in Claude's existing API responses and parses it in real-time — no external servers, no polling a third-party API.

Data flow:

Claude.ai page
      │
  injector.js          ← bridge between extension context and page world
      │
  injected.js          ← runs in page world; intercepts fetch + SSE streams
      │
  content.js           ← receives parsed data; renders UI; writes to storage
      │
  popup_usage.js       ← reads chrome.storage.local; renders popup history
Script	Execution Context	Role
injector.js	Extension context	Dynamically injects injected.js into the page world; acts as a message bridge
injected.js	Page world	Hooks into fetch and parses Server-Sent Event (SSE) streams to extract usage data
content.js	Content script	Receives parsed data via window.postMessage; renders the progress bar; persists sessions
popup_usage.js	Extension popup	Reads persisted history from chrome.storage.local; renders the popup UI
Tech Stack
Technology	Purpose
Manifest V3	Extension architecture — required for Chrome/Edge store compliance
Vanilla JavaScript	Core logic — zero external dependencies, minimal footprint
Server-Sent Events (SSE)	Intercepting Claude's streaming token responses
REST API interception	Fallback parsing for non-streaming responses
MutationObserver	Detecting Claude UI changes and new chat turns
chrome.storage.local	Cross-tab persistence of session history
Privacy
LimitLens is built with a local-only architecture:

✅ No external servers. Zero network requests are made by the extension itself.
✅ No analytics or telemetry. No tracking pixels, no usage reporting.
✅ Conversation content is never read. The extension only reads usage metadata from response headers.
✅ All data stays in your browser. Session history is stored in chrome.storage.local — on your device only.
Contributing
Contributions are welcome. If you find a bug or want to suggest a feature, please open an issue.

For pull requests:

# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
# Open a Pull Request
Browser Support
Browser	Status
Microsoft Edge	✅ Available on Edge Add-ons Store
Google Chrome	✅ Supported via Developer Mode
Firefox	🔄 Manifest included (manifest_firefox.json) — store submission pending
Safari	🔄 Build script included (build_safari.sh) — in progress
License
This project is licensed under the MIT License.

Author
Built by Dwarkesh

GitHub LinkedIn Discord""" 

#paragraph = input("User > ")

intial_state = {"paragraph" : paragraph1,
    "cliarity_feed" : "",
    "depth_feed" : "",
    "engage_feed" : "",
    "indivisual_scores" : [], 
    "overall_feedback" : ""}

res = workflow.invoke(intial_state)

print("\n\n\nOverall Feedback => ", res["overall_feedback"])
print("\n\n\nCliarity => ", res["cliarity_feed"])
print("\n\n\nDepth => ", res["depth_feed"])
print("\n\n\nEngage => ", res["engage_feed"])
print("\n\n\nScores => ", res["indivisual_scores"])
