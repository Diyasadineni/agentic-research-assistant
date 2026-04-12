Agentic Research Assistant
An autonomous AI research agent that takes a research question, plans a search strategy, fetches live web data, and generates a structured cited report — with zero human intervention.
Built with LangGraph, Groq (Llama 3.3 70B), and Tavily Search API.

Demo

Input: "What are the latest trends in AI agents in 2025?"
Output: A full structured report with Summary, Key Findings, Details, and cited Sources — generated autonomously in under 30 seconds.


How It Works
User Question → Agent Plans Queries → Web Search (Tavily) → Page Fetching → Synthesize → Structured Report
The agent uses a ReAct loop — it reasons, calls tools, observes results, and decides the next step autonomously. It's not hardcoded — the LLM decides what to search and when to stop.

Features

Autonomous planning — breaks one question into 3-4 targeted search queries
Live web retrieval — fetches real-time web content via Tavily Search API
Deep page reading — extracts full content from relevant URLs
Structured reports — always outputs Summary → Key Findings → Details → Sources
LLM-as-judge evaluation — custom evaluation pipeline scoring faithfulness and answer relevancy
Streamlit UI — clean interface with live agent status and evaluation scores dashboard


Tech Stack
ToolPurposeLangGraphAgentic reasoning loop (ReAct pattern)Groq APILLM inference — Llama 3.3 70BTavily Search APILive web search + page extractionLangChainTool definitions and prompt managementStreamlitFrontend UIPython-dotenvEnvironment variable management

Project Structure
research-agent/
├── agent.py          # LangGraph agent with ReAct loop
├── tools.py          # search_web and fetch_page tools
├── app.py            # Streamlit UI
├── evaluate.py       # LLM-as-judge evaluation framework
└── .env              # API keys (not pushed to GitHub)

Getting Started
1. Clone the repo
bashgit clone https://github.com/YOURUSERNAME/agentic-research-assistant.git
cd agentic-research-assistant
2. Create virtual environment
bashpython3 -m venv venv
source venv/bin/activate
3. Install dependencies
bashpip install langchain langchain-groq langchain-community langchain-core langgraph tavily-python streamlit python-dotenv
4. Set up API keys
Create a .env file in the root directory:
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
Get your free API keys:

Groq → https://console.groq.com
Tavily → https://app.tavily.com

5. Run the app
bashstreamlit run app.py

Evaluation
Run the evaluation framework to score the agent:
bashpython3 evaluate.py
Scores are saved to eval_results.json and displayed live in the Streamlit UI.
MetricDescriptionFaithfulnessAre answers factually grounded in sources?Answer RelevancyDoes the answer address the question?

Why This Project
Most RAG projects use a static vector database — upload documents, ask questions. This project is different:

No pre-loaded knowledge base — retrieves live information every time
Agentic architecture — the LLM decides what to search, not hardcoded logic
Real citations — every report includes actual URLs from fetched pages
Evaluation built in — not just a demo, but a measured system

