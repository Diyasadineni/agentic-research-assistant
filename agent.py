from doc_store import search_documents
from tools import search_web, fetch_page
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def build_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    tools = [search_web, fetch_page, search_documents]

    system_prompt = """You are an expert research assistant. When given a research question:

1. PLAN: Break it into 3-4 specific search queries
2. SEARCH DOCUMENTS: Always check uploaded documents first using search_documents
3. SEARCH WEB: Use search_web for additional current information
4. DEEP DIVE: Use fetch_page on the most relevant URLs
5. SYNTHESIZE: Combine findings from BOTH documents and web into a structured report

Always structure your final report as:
## Summary
## Key Findings (from documents)
## Additional Findings (from web)
## Details
## Sources

Always cite sources with URLs and document names. Never make up facts."""

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )
    return agent

def run_agent(question):
    agent = build_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })
    return result["messages"][-1].content