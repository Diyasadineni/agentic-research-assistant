import streamlit as st
from agent import run_agent
from doc_store import add_document
from dotenv import load_dotenv
import tempfile, os, json

load_dotenv()

st.title("Agentic Research Assistant")
st.caption("Autonomous AI research agent powered by LLMs and live web search")

# ── Document upload section ──────────────────────────────────
st.subheader("Upload your documents")
uploaded_files = st.file_uploader(
    "Upload PDFs, Word docs, CSVs or text files",
    type=["pdf", "docx", "csv", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        with st.spinner(f"Processing {uploaded_file.name}..."):
            count = add_document(tmp_path)
        st.success(f"{uploaded_file.name} — {count} chunks added to knowledge base")

st.markdown("---")

# ── Research question section ────────────────────────────────
st.subheader("Ask your research question")
question = st.text_input(
    "The agent will search both your documents and the web:",
    placeholder="e.g. What does the contract say about payment terms?"
)

if st.button("Research") and question:
    with st.status("Agent is working...", expanded=True) as status:
        st.write("Searching documents and web...")
        with st.spinner("Synthesizing findings..."):
            result = run_agent(question)
        status.update(label="Research complete!", state="complete")

    st.markdown("---")
    st.markdown(result)

# ── Evaluation scores ────────────────────────────────────────
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