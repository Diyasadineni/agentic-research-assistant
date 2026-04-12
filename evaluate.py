import json
import os
from langchain_groq import ChatGroq
from agent import run_agent
from dotenv import load_dotenv

load_dotenv()

# ─── 1. TEST CASES ───────────────────────────────────────────
test_cases = [
    {
        "question": "What are the latest trends in AI agents in 2025?",
        "ground_truth": "AI agents in 2025 use LLMs to autonomously plan, use tools, browse the web, and complete multi-step tasks"
    },
    {
        "question": "How does RAG improve LLM accuracy?",
        "ground_truth": "RAG improves LLM accuracy by retrieving relevant documents and grounding responses in real sources instead of relying on training data"
    },
    {
        "question": "What are the best vector databases for AI applications in 2025?",
        "ground_truth": "Popular vector databases include Pinecone, Weaviate, Chroma, and FAISS, each suited for different scale and use cases"
    },
    {
        "question": "What is LangGraph and how is it used for building agents?",
        "ground_truth": "LangGraph is a framework for building stateful multi-agent workflows using graph-based execution"
    },
    {
        "question": "What are the risks of hallucination in LLMs and how can they be reduced?",
        "ground_truth": "LLM hallucinations can be reduced using RAG, citations, fact checking, and grounding responses in retrieved documents"
    }
]

# ─── 2. LLM JUDGE USING GROQ ─────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

def score_answer(question, answer, ground_truth):
    prompt = f"""You are an expert evaluator for AI research assistants.

Question: {question}
Expected themes: {ground_truth}
Agent's answer: {answer[:3000]}

Score generously — this is a research agent that writes detailed reports.
A good answer covers the key themes even if worded differently.

Score each from 0.0 to 1.0:
- faithfulness: Are the facts accurate and grounded in real sources?
- answer_relevancy: Does the answer address the question and cover key themes?

Reply ONLY with this JSON, nothing else:
{{"faithfulness": 0.0, "answer_relevancy": 0.0}}"""

    response = llm.invoke(prompt)
    text = response.content.strip()
    print(f"Raw LLM response: {text}")
    start = text.find("{")
    end = text.rfind("}") + 1
    scores = json.loads(text[start:end])
    return scores

# ─── 3. RUN EVALUATION ───────────────────────────────────────
def run_evaluation():
    all_scores = []
    
    print("Running evaluation...\n")
    print("=" * 50)
    
    for i, test in enumerate(test_cases):
        print(f"\nQuestion {i+1}/{len(test_cases)}: {test['question']}")
        
        try:
            print("Step 1 - Getting agent answer...")
            answer = run_agent(test["question"])
            print(f"Step 1 DONE - answer length: {len(answer)}")
            print(f"Answer preview: {answer[:200]}")
            
            print("Step 2 - Scoring...")
            scores = score_answer(
                test["question"],
                answer,
                test["ground_truth"]
            )
            print(f"Step 2 DONE - scores: {scores}")
            
            all_scores.append(scores)
            print(f"Faithfulness: {scores['faithfulness']} | Relevancy: {scores['answer_relevancy']} ✅")
            
        except Exception as e:
            import traceback
            print(f"FAILED ON QUESTION {i+1}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            traceback.print_exc()
            all_scores.append({"faithfulness": 0.0, "answer_relevancy": 0.0})
    
    return all_scores

# ─── 4. SAVE RESULTS ─────────────────────────────────────────
def save_results(all_scores):
    avg_faithfulness = round(
        sum(s["faithfulness"] for s in all_scores) / len(all_scores), 3
    )
    avg_relevancy = round(
        sum(s["answer_relevancy"] for s in all_scores) / len(all_scores), 3
    )
    overall = round((avg_faithfulness + avg_relevancy) / 2, 3)
    
    final = {
        "faithfulness": avg_faithfulness,
        "answer_relevancy": avg_relevancy,
        "overall": overall,
        "num_questions": len(all_scores)
    }
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Faithfulness:      {avg_faithfulness} / 1.0")
    print(f"Answer Relevancy:  {avg_relevancy} / 1.0")
    print(f"Overall Score:     {overall} / 1.0")
    print(f"Questions tested:  {len(all_scores)}")
    print("=" * 50)
    
    with open("eval_results.json", "w") as f:
        json.dump(final, f, indent=2)
    
    print("\nResults saved to eval_results.json ✅")
    return final

# ─── 5. RUN ──────────────────────────────────────────────────
if __name__ == "__main__":
    all_scores = run_evaluation()
    save_results(all_scores)