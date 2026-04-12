import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from tools import search_web, fetch_page

load_dotenv()

def build_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    tools = [search_web, fetch_page]

    system_prompt = """You are an expert research assistant. When given a research question:

1. PLAN: Break it into 3-4 specific search queries
2. SEARCH: Use search_web for each query
3. DEEP DIVE: Use fetch_page on the most relevant URLs
4. SYNTHESIZE: Compile findings into a structured report

Always structure your final report as:
## Summary
## Key Findings
## Details (with subsections)
## Sources

Always cite sources with URLs. Never make up facts."""

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