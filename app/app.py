import streamlit as st
import sys
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from rag import (
    create_qdrant_client,
    create_embedding_model,
    create_reranker,
    create_llm,
    answer_question,
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Industrial Maintenance Copilot",
    page_icon="🔧",
    layout="wide",
)

# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #777;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .source-card {
        padding: 0.8rem;
        border-radius: 8px;
        background-color: #f5f5f5;
        margin-bottom: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🔧 Industrial Maintenance Copilot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered assistant for industrial pump maintenance "
    "and troubleshooting using technical documentation."
    "</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ System")

    st.success("RAG system online")

    st.markdown(
        """
        **Pipeline**

        📄 Technical manuals  
        ↓  
        ✂️ Chunking  
        ↓  
        🧠 Embeddings  
        ↓  
        🔎 Qdrant retrieval  
        ↓  
        🎯 Reranking  
        ↓  
        🤖 Gemini  
        ↓  
        💬 Answer + sources
        """
    )

    st.divider()

    st.caption(
        "Industrial Maintenance Copilot"
    )

    st.caption(
        "Built with LangChain, Qdrant, "
        "Sentence Transformers and Gemini."
    )

# --------------------------------------------------
# Load RAG system
# --------------------------------------------------

@st.cache_resource
def load_system():

    client = create_qdrant_client()
    embedding_model = create_embedding_model()
    reranker = create_reranker()
    llm = create_llm()

    return (
        client,
        embedding_model,
        reranker,
        llm,
    )


client, embedding_model, reranker, llm = load_system()

# --------------------------------------------------
# Example questions
# --------------------------------------------------

st.subheader("Ask a maintenance question")

st.write(
    "Describe a pump problem and the Copilot will "
    "search the technical documentation."
)

example_questions = [
    "Why can a pump have excessive discharge pressure?",
    "What can cause abnormal pump vibration?",
    "What should I check if the pump does not start?",
]

st.markdown("**Example questions:**")

cols = st.columns(3)

for i, question in enumerate(example_questions):

    with cols[i]:

        if st.button(
            question,
            use_container_width=True,
        ):

            st.session_state["query"] = question

# --------------------------------------------------
# Query input
# --------------------------------------------------

query = st.text_area(
    "Maintenance question",
    value=st.session_state.get("query", ""),
    placeholder=(
        "Example: Why can a pump have "
        "excessive discharge pressure?"
    ),
    height=100,
)

# --------------------------------------------------
# Ask button
# --------------------------------------------------

if st.button(
    "🔍 Ask Copilot",
    type="primary",
    use_container_width=True,
):

    if not query.strip():

        st.warning(
            "Please enter a maintenance question."
        )

    else:

        with st.spinner(
            "Searching technical documentation and generating answer..."
        ):

            try:

                answer, results = answer_question(
                    query=query,
                    client=client,
                    embedding_model=embedding_model,
                    reranker=reranker,
                    llm=llm,
                )

                st.session_state["answer"] = answer
                st.session_state["results"] = results

            except Exception as e:

                st.error(
                    "The RAG system encountered an error."
                )

                st.exception(e)

# --------------------------------------------------
# Display answer
# --------------------------------------------------

if "answer" in st.session_state:

    st.divider()

    st.subheader("🤖 Answer")

    answer = st.session_state["answer"]

    if answer.startswith("⚠️ **AI generation is temporarily unavailable.**"):
        st.warning(answer)
    else:
        st.markdown(answer)
# --------------------------------------------------
# Display sources
# --------------------------------------------------

if "results" in st.session_state:

    st.subheader("📚 Sources")

    results = st.session_state["results"]

    for i, item in enumerate(
        results,
        start=1
    ):

        result = item["result"]

        source = result.payload.get(
            "source_file",
            "Unknown",
        )

        page = result.payload.get(
            "page",
            "Unknown",
        )

        rerank_score = item.get(
            "rerank_score",
            0,
        )

        with st.expander(
            f"{i}. {source} — Page {page}"
        ):

            st.write(
                result.payload.get(
                    "text",
                    "",
                )
            )