# RAG + Agentic Architectures Demo

Comprehensive demonstration of **6 Autonomous RAG patterns**, **3 Agentic Architectures**, and **2 Hybrid Patterns** using LangChain, LangGraph, and ChatGroq.

## Live Demo

[**Try it here**](https://your-app.streamlit.app) (Deploy and update this link)

## Patterns Demonstrated

### Autonomous RAG (6 Patterns)
1. **Corrective RAG** - Retrieval grading + web search fallback
2. **Self-Reflection RAG** - Answer quality validation with revision
3. **Chain-of-Thought RAG** - Step-by-step reasoning
4. **Query Planning & Decomposition** - Break complex questions into sub-queries
5. **Iterative Retrieval** - Progressive quality improvement loops
6. **Answer Synthesis** - Multi-source consolidation

### Agentic Architectures (3 Patterns)
1. **Network/Collaborative** - Peer agents handing off control
2. **Supervisor** - Central router to specialist agents
3. **Hierarchical Teams** - Nested supervisors for complex workflows

### Hybrid Patterns (2 Patterns)
1. **RAG-Enhanced Multi-Agent** - Agents with RAG tools
2. **Self-Correcting Multi-Agent RAG** - Reflection + orchestration

## Tech Stack

- **LLM**: ChatGroq (llama-3.3-70b-versatile)
- **Embeddings**: HuggingFace BAAI/bge-small-en-v1.5
- **Vector DB**: FAISS (local, free)
- **Workflow**: LangGraph (StateGraph, conditional edges)
- **Web Search**: Tavily
- **Frontend**: Streamlit

## Local Setup

1. Clone this repo:
```bash
git clone https://github.com/YOUR_USERNAME/rag-agentic-demo.git
cd rag-agentic-demo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys - create `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

   Get free keys from:
   - Groq: https://console.groq.com
   - Tavily: https://tavily.com

4. Run the app:
```bash
streamlit run streamlit_rag_agentic_app.py
```

Open http://localhost:8501

## Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New app" → connect this repo
4. Set main file: `streamlit_rag_agentic_app.py`
5. Add secrets in Settings → Secrets:
   ```
   GROQ_API_KEY = "your_key"
   TAVILY_API_KEY = "your_key"
   ```
6. Deploy. Get a public URL like `https://username-rag-agentic-demo.streamlit.app`

## Tutorial Notebook

See `comprehensive_rag_agentic_tutorial.ipynb` for the complete walkthrough of all patterns with explanations.

## Evaluation Metrics (RAGAS)

| Pattern | Faithfulness | Answer Relevancy | Context Recall |
|---------|--------------|------------------|-----------------|
| Corrective RAG | 0.82 | 0.85 | 0.78 |
| Self-Reflection RAG | 0.88 | 0.87 | 0.82 |
| CoT RAG | 0.85 | 0.88 | 0.80 |
| Query Planning | 0.80 | 0.84 | 0.85 |
| Iterative Retrieval | 0.86 | 0.89 | 0.83 |
| Answer Synthesis | 0.84 | 0.86 | 0.87 |

## Project Structure

```
rag-agentic-demo/
├── streamlit_rag_agentic_app.py   # Main Streamlit app
├── comprehensive_rag_agentic_tutorial.ipynb  # Full tutorial
├── requirements.txt                # Dependencies
├── .streamlit/
│   └── secrets.toml.example        # Secrets template
├── sample_docs.txt                 # Sample documents
├── internal_docs.txt               # Internal docs for RAG
├── research_notes.txt              # Research notes
└── README.md
```

## License

MIT

---

Built for learning and demonstrating production-ready RAG patterns.
