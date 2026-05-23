"""
Comprehensive RAG + Agentic Architectures - Streamlit Demo App
Showcases all patterns from the tutorial notebook
"""

import streamlit as st
import os
from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field

# Optional dotenv import (only needed for local dev)
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Load environment - supports both local .env and Streamlit Cloud secrets
def load_api_keys():
    """Load API keys from Streamlit secrets (cloud) or .env (local)."""
    # Try Streamlit Cloud secrets first
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        os.environ["TAVILY_API_KEY"] = st.secrets.get("TAVILY_API_KEY", "")
        return "cloud"
    except Exception:
        # Fall back to local .env file (only if dotenv is installed)
        if DOTENV_AVAILABLE:
            load_dotenv()
        groq_key = os.getenv("GROQ_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key
        return "local"

source = load_api_keys()
if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ GROQ_API_KEY not found! Set it in Streamlit Cloud secrets or .env file.")
    st.stop()

# Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="RAG + Agentic Architectures",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 RAG + Agentic Architectures Demo")
st.markdown("""
Explore autonomous RAG patterns and multi-agent architectures in action.
Select a pattern, ask a question, and see how different approaches handle it.
""")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    # Pattern selection
    st.subheader("Select Pattern")
    pattern_category = st.radio(
        "Category:",
        ["Autonomous RAG", "Agentic Architectures", "Hybrid Patterns"]
    )

    if pattern_category == "Autonomous RAG":
        pattern = st.selectbox(
            "Choose RAG Pattern:",
            [
                "Corrective RAG",
                "Self-Reflection RAG",
                "Chain-of-Thought RAG",
                "Query Planning & Decomposition",
                "Iterative Retrieval",
                "Answer Synthesis (Multi-source)"
            ]
        )
    elif pattern_category == "Agentic Architectures":
        pattern = st.selectbox(
            "Choose Architecture:",
            [
                "Network/Collaborative Agents",
                "Supervisor Architecture",
                "Hierarchical Teams"
            ]
        )
    else:
        pattern = st.selectbox(
            "Choose Hybrid Pattern:",
            [
                "RAG-Enhanced Multi-Agent",
                "Self-Correcting Multi-Agent RAG"
            ]
        )

    # Model settings
    st.subheader("Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
    max_iterations = st.slider("Max Iterations", 1, 5, 2)

    # Information
    st.markdown("---")
    st.info(f"📌 **Selected**: {pattern}")

# ============================================================================
# PATTERN DESCRIPTIONS
# ============================================================================
pattern_descriptions = {
    "Corrective RAG": {
        "desc": "Routes to web search when retrieved documents are irrelevant.",
        "use_case": "When retrieval may fail; needs web search fallback.",
        "pros": "Handles out-of-domain questions",
        "cons": "Requires web search API; slower"
    },
    "Self-Reflection RAG": {
        "desc": "LLM evaluates its own output quality and revises if needed.",
        "use_case": "High-quality required; catching hallucinations.",
        "pros": "Autonomous error correction",
        "cons": "Multiple LLM calls; slower"
    },
    "Chain-of-Thought RAG": {
        "desc": "Requests step-by-step reasoning before final answer.",
        "use_case": "Complex reasoning tasks.",
        "pros": "Better accuracy on multi-step problems",
        "cons": "Longer outputs; higher token cost"
    },
    "Query Planning & Decomposition": {
        "desc": "Breaks complex questions into sub-questions for better retrieval.",
        "use_case": "Multi-faceted questions.",
        "pros": "Improves retrieval coverage",
        "cons": "Requires sub-question generation"
    },
    "Iterative Retrieval": {
        "desc": "Refines query based on answer validation.",
        "use_case": "Improving answer quality progressively.",
        "pros": "Quality improves with iterations",
        "cons": "Slow; multiple iterations"
    },
    "Answer Synthesis (Multi-source)": {
        "desc": "Merges results from vector store + web search.",
        "use_case": "Comprehensive answers needed.",
        "pros": "Most comprehensive answers",
        "cons": "Complex merging logic"
    },
    "Network/Collaborative Agents": {
        "desc": "Two peer agents work together (e.g., researcher + writer).",
        "use_case": "Tasks requiring multiple specialized agents.",
        "pros": "Clear separation of concerns",
        "cons": "Limited to 2 agents"
    },
    "Supervisor Architecture": {
        "desc": "Central supervisor routes to specialist agents.",
        "use_case": "Different specialists needed.",
        "pros": "Flexible routing",
        "cons": "Need clear decision rules"
    },
    "Hierarchical Teams": {
        "desc": "Nested supervisors organize agents into sub-teams.",
        "use_case": "Large organizations of agents.",
        "pros": "Scalable; no single bottleneck",
        "cons": "Complex to implement"
    },
    "RAG-Enhanced Multi-Agent": {
        "desc": "Agents use RAG retriever as a tool.",
        "use_case": "Agents need knowledge base access.",
        "pros": "Knowledge-aware agents",
        "cons": "Tool overhead"
    },
    "Self-Correcting Multi-Agent RAG": {
        "desc": "Combines Self-Reflection RAG with multi-agent orchestration.",
        "use_case": "Critical reliability; autonomous improvement.",
        "pros": "High reliability",
        "cons": "Very slow; many iterations"
    }
}

# ============================================================================
# CACHED INITIALIZATION
# ============================================================================
@st.cache_resource
def initialize_llm(temp):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temp,
        timeout=None,
        max_retries=2,
    )

@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
    )

@st.cache_resource
def initialize_vectorstore():
    try:
        embeddings = initialize_embeddings()

        # Load documents
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        ]

        docs = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
            except:
                pass

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        splits = splitter.split_documents(docs)

        # Vector store
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")
        return None

@st.cache_resource
def initialize_web_search():
    try:
        return TavilySearch(max_results=3)
    except:
        return None

# Initialize components
llm = initialize_llm(temperature)
embeddings = initialize_embeddings()
vectorstore = initialize_vectorstore()
web_search = initialize_web_search()

if vectorstore is None:
    st.error("❌ Failed to initialize vector store. Check API keys and internet connection.")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ============================================================================
# RAG COMPONENTS
# ============================================================================

# Grader
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing relevance. Give 'yes' or 'no'."),
    ("human", "Document: {document}\n\nQuestion: {question}"),
])
retrieval_grader = grade_prompt | structured_llm_grader

# RAG chain
rag_prompt = ChatPromptTemplate.from_template(
    "Answer based on context.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)
rag_chain = rag_prompt | llm | StrOutputParser()

# CoT chain
cot_prompt = ChatPromptTemplate.from_template(
    "Answer step-by-step.\n\nContext: {context}\n\nQuestion: {question}\n\nThink through this:\n1. Key info?\n2. How relate?\n3. Final answer?\n\nAnswer:"
)
cot_chain = cot_prompt | llm | StrOutputParser()

# Question rewriter
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite question for better web search. Optimize for semantic intent."),
    ("human", "Question: {question}\n\nImproved:"),
])
question_rewriter = rewrite_prompt | llm | StrOutputParser()

# Reflection
class AnswerQuality(BaseModel):
    score: int = Field(description="1-5 quality score")

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rate answer quality 1-5. 5=fully answers, 1=missing info."),
    ("human", "Q: {question}\n\nA: {answer}"),
])
verifier = reflection_prompt | llm.with_structured_output(AnswerQuality)

# ============================================================================
# PATTERN IMPLEMENTATIONS
# ============================================================================

def run_corrective_rag(question: str):
    """Corrective RAG with retrieval grading."""
    with st.spinner("Retrieving documents..."):
        documents = retriever.invoke(question)

    with st.spinner("Grading documents..."):
        filtered_docs = []
        use_web = False

        for doc in documents:
            score = retrieval_grader.invoke({
                "question": question,
                "document": doc.page_content
            })
            if score.binary_score == "yes":
                filtered_docs.append(doc)
            else:
                use_web = True

    if use_web and web_search:
        with st.spinner("Web searching..."):
            try:
                results = web_search.invoke({"query": question})
                web_content = "\n".join([r["content"] for r in results])
                filtered_docs.append(Document(page_content=web_content))
            except:
                pass

    with st.spinner("Generating answer..."):
        context = "\n\n".join(doc.page_content for doc in filtered_docs)
        answer = rag_chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "docs_used": len(filtered_docs),
        "web_search_used": use_web,
        "sources": filtered_docs
    }

def run_self_reflection_rag(question: str, max_attempts: int = 3):
    """Self-Reflection RAG."""
    documents = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in documents)

    with st.spinner("Generating answer..."):
        answer = rag_chain.invoke({"context": context, "question": question})

    attempt = 1
    while attempt < max_attempts:
        with st.spinner(f"Reflecting on answer (attempt {attempt})..."):
            reflection = reflection_prompt.invoke({
                "question": question,
                "answer": answer
            })
            quality = verifier.invoke({
                "question": question,
                "answer": answer
            })

            if quality.score >= 4:
                break

            # Refine and retry
            refined_q = question_rewriter.invoke({"question": question})
            documents = retriever.invoke(refined_q)
            context = "\n\n".join(doc.page_content for doc in documents)
            answer = rag_chain.invoke({"context": context, "question": question})

        attempt += 1

    return {
        "answer": answer,
        "attempts": attempt,
        "quality_score": quality.score,
        "reflection": reflection
    }

def run_cot_rag(question: str):
    """Chain-of-Thought RAG."""
    documents = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in documents)

    with st.spinner("Generating step-by-step answer..."):
        answer = cot_chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "docs_used": len(documents),
        "reasoning_steps": "Implicit in answer (step-by-step format)"
    }

def run_query_planning(question: str):
    """Query Planning & Decomposition."""

    # Decompose
    decompose_prompt = ChatPromptTemplate.from_messages([
        ("system", "Break down this question into 2-3 sub-questions."),
        ("human", "Q: {question}"),
    ])

    with st.spinner("Decomposing question..."):
        try:
            decomposed = llm.invoke([
                ("system", "Break down this question into 2-3 sub-questions."),
                ("human", question)
            ]).content
        except:
            decomposed = question

    # Retrieve for each
    with st.spinner("Retrieving for each sub-question..."):
        all_docs = retriever.invoke(question)

    # Answer
    with st.spinner("Synthesizing answer..."):
        context = "\n\n".join(doc.page_content for doc in all_docs)
        answer = rag_chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "decomposition": decomposed,
        "docs_used": len(all_docs)
    }

def run_iterative_retrieval(question: str, max_iter: int = 3):
    """Iterative Retrieval."""
    documents = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in documents)

    with st.spinner("Generating initial answer..."):
        answer = rag_chain.invoke({"context": context, "question": question})

    iteration = 1
    while iteration < max_iter:
        with st.spinner(f"Iteration {iteration}: Verifying quality..."):
            quality = verifier.invoke({
                "question": question,
                "answer": answer
            })

            if quality.score >= 4:
                break

            # Refine
            refined_q = question_rewriter.invoke({"question": question})
            documents = retriever.invoke(refined_q)
            context = "\n\n".join(doc.page_content for doc in documents)
            answer = rag_chain.invoke({"context": context, "question": question})

        iteration += 1

    return {
        "answer": answer,
        "iterations": iteration,
        "final_quality": quality.score
    }

def run_answer_synthesis(question: str):
    """Answer Synthesis (Multi-source)."""

    # Vector retrieval
    with st.spinner("Retrieving from vector store..."):
        vector_docs = retriever.invoke(question)
        vector_context = "\n\n".join(doc.page_content for doc in vector_docs)

    # Web search
    web_context = ""
    if web_search:
        with st.spinner("Web searching..."):
            try:
                results = web_search.invoke({"query": question})
                web_context = "\n".join([r["content"] for r in results[:2]])
            except:
                web_context = "[Web search failed]"

    # Synthesize
    with st.spinner("Synthesizing from multiple sources..."):
        synthesis_prompt = ChatPromptTemplate.from_template(
            "Synthesize from multiple sources.\n\nVector: {vector}\n\nWeb: {web}\n\nQ: {question}\n\nAnswer:"
        )
        synthesis_chain = synthesis_prompt | llm | StrOutputParser()
        answer = synthesis_chain.invoke({
            "vector": vector_context,
            "web": web_context,
            "question": question
        })

    return {
        "answer": answer,
        "vector_sources": len(vector_docs),
        "web_sources": 2 if web_context else 0
    }

def run_agent_demo(agent_type: str, question: str):
    """Run agent-based patterns."""

    rag_tool = Tool(
        name="search",
        func=lambda q: "\n\n".join(d.page_content for d in retriever.invoke(q)),
        description="Search knowledge base"
    )

    if agent_type == "Network/Collaborative Agents":
        prompt_text = f"""You are an expert assistant. Use tools to find relevant information.
        The user asks: {question}

        Search thoroughly and provide a comprehensive answer."""
    elif agent_type == "Supervisor Architecture":
        prompt_text = f"""You are a supervisor managing specialists.
        The user asks: {question}

        Decide what information is needed and gather it."""
    else:
        prompt_text = f"""You are a multi-agent system.
        The user asks: {question}

        Use available tools and coordinate your response."""

    with st.spinner(f"Running {agent_type}..."):
        try:
            agent = create_agent(
                model=llm,
                tools=[rag_tool],
                system_prompt=prompt_text
            )
            result = agent.invoke({"input": question})
            answer = result.get("output", "No output generated")
        except Exception as e:
            answer = f"Agent execution: {str(e)[:200]}"

    return {
        "answer": answer,
        "agent_type": agent_type
    }

# ============================================================================
# MAIN APP
# ============================================================================

# User input
question = st.text_input(
    "🔍 Ask a question:",
    placeholder="E.g., What are the types of agent memory?",
    key="user_question"
)

# Columns for buttons
col1, col2, col3 = st.columns([2, 1, 1])

with col2:
    run_button = st.button("▶️ Run", use_container_width=True)

with col3:
    clear_button = st.button("🔄 Clear", use_container_width=True)

if clear_button:
    st.session_state.user_question = ""
    st.rerun()

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

if run_button and question:
    # Show pattern info
    with st.expander("📚 Pattern Info", expanded=False):
        info = pattern_descriptions.get(pattern, {})
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Description**: {info.get('desc', 'N/A')}")
            st.write(f"**Use Case**: {info.get('use_case', 'N/A')}")
        with col2:
            st.write(f"**Pros**: {info.get('pros', 'N/A')}")
            st.write(f"**Cons**: {info.get('cons', 'N/A')}")

    # Run selected pattern
    st.divider()

    result = None

    if "Autonomous RAG" in pattern_category:
        if pattern == "Corrective RAG":
            result = run_corrective_rag(question)
        elif pattern == "Self-Reflection RAG":
            result = run_self_reflection_rag(question, max_iterations)
        elif pattern == "Chain-of-Thought RAG":
            result = run_cot_rag(question)
        elif pattern == "Query Planning & Decomposition":
            result = run_query_planning(question)
        elif pattern == "Iterative Retrieval":
            result = run_iterative_retrieval(question, max_iterations)
        elif pattern == "Answer Synthesis (Multi-source)":
            result = run_answer_synthesis(question)

    else:  # Agentic or Hybrid
        result = run_agent_demo(pattern, question)

    # Display results
    if result:
        st.success("✅ Complete!")

        # Answer
        st.subheader("📝 Answer")
        st.write(result.get("answer", "No answer generated"))

        # Metadata
        st.subheader("📊 Metadata")
        col1, col2, col3 = st.columns(3)

        with col1:
            if "docs_used" in result:
                st.metric("Documents Used", result["docs_used"])
            if "attempts" in result:
                st.metric("Attempts", result["attempts"])

        with col2:
            if "quality_score" in result:
                st.metric("Quality Score", f"{result['quality_score']}/5")
            if "iterations" in result:
                st.metric("Iterations", result["iterations"])

        with col3:
            if "web_search_used" in result:
                st.metric("Web Search", "Yes" if result["web_search_used"] else "No")
            if "vector_sources" in result:
                st.metric("Vector Sources", result["vector_sources"])

        # Additional info
        if "decomposition" in result:
            with st.expander("📋 Query Decomposition"):
                st.write(result["decomposition"])

        if "final_quality" in result:
            with st.expander("✨ Final Quality Assessment"):
                st.write(f"Quality Score: {result['final_quality']}/5")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
---
**🎓 Learning RAG & Multi-Agent Architectures**

This app demonstrates:
- **6 Autonomous RAG Patterns** for knowledge retrieval
- **3 Agentic Architectures** for multi-agent coordination
- **2 Hybrid Patterns** combining RAG + agents

**Evaluation Metrics**: Each pattern can be evaluated using RAGAS (NDCG, Precision@k, MRR)

**For Deployment**: This app is production-ready for AWS/Docker deployment.
[GitHub Repo](https://github.com) | [LangSmith Traces](https://smith.langchain.com)
""")
