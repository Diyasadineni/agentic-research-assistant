import streamlit as st
from agent import run_agent
from dotenv import load_dotenv
import json, os

load_dotenv()

st.title("🔍 Agentic Research Assistant")
st.caption("Autonomous AI research agent powered by LLMs and live web search")

question = st.text_input(
    "Enter your research question:",
    placeholder="e.g. What are the latest trends in AI agents?"
)

if st.button("Research") and question:
    with st.status("Agent is working...", expanded=True) as status:
        st.write("Planning research strategy...")
        with st.spinner("Searching and synthesizing..."):
            result = run_agent(question)
        status.update(label="Research complete!", state="complete")
    
    st.markdown("---")
    st.markdown(result)

st.markdown("---")
st.subheader("Model Evaluation Scores")

if os.path.exists("eval_results.json"):
    with open("eval_results.json") as f:
        scores = json.load(f)
    col1, col2, col3 = st.columns(3)
    col1.metric("Faithfulness", f"{scores['faithfulness']}/1.0")
    col2.metric("Answer Relevancy", f"{scores['answer_relevancy']}/1.0")
    col3.metric("Overall Score", f"{scores['overall']}/1.0")
else:
    st.caption("Run evaluate.py to generate scores")