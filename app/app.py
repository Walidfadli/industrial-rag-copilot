import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Industrial Maintenance Copilot",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .main-title {
        font-size: 2.7rem;
        font-weight: 750;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    .status-card {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border: 1px solid #d1d5db;
        margin-bottom: 1rem;
    }

    .source-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.7rem;
        background-color: #fafafa;
    }

    .source-file {
        font-weight: 650;
        font-size: 0.98rem;
    }

    .source-meta {
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }

    .example-label {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "query" not in st.session_state:
    st.session_state["query"] = ""


# ============================================================
# FASTAPI CONNECTION
# ============================================================

def check_api_health():
    """
    Check whether the FastAPI backend is running.
    """

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if response.status_code == 200:
            return True

    except requests.RequestException:
        pass

    return False


def ask_api(question):
    """
    Send a question to the FastAPI backend.
    """

    response = requests.post(
        f"{API_URL}/ask",
        json={
            "question": question
        },
        timeout=180,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    api_online = check_api_health()

    if api_online:
        st.success("FastAPI backend online")
    else:
        st.error("FastAPI backend offline")

    st.divider()

    st.markdown("### Architecture")

    st.markdown(
        """
        **🖥️ Streamlit**

        ↓

        **⚡ FastAPI**

        ↓

        **🔎 RAG Pipeline**

        ↓

        **🧠 Qdrant + Reranker**

        ↓

        **🤖 Gemini**

        ↓

        **📚 Grounded Answer**
        """
    )

    st.divider()

    st.markdown("### System components")

    st.markdown(
        """
        - **Frontend:** Streamlit
        - **Backend:** FastAPI
        - **LLM:** Gemini
        - **Embeddings:** Sentence Transformers
        - **Vector DB:** Qdrant
        - **Reranker:** Cross-Encoder
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state["messages"] = []
        st.session_state["query"] = ""

        st.rerun()

    st.divider()

    st.caption("Industrial Maintenance Copilot")

    st.caption(
        "AI assistant for industrial pump "
        "maintenance and troubleshooting."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    "🔧 Industrial Maintenance Copilot"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered assistant for industrial pump "
    "maintenance and troubleshooting using "
    "technical documentation."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SYSTEM STATUS
# ============================================================

if api_online:

    st.markdown(
        """
        <div class="status-card">
            🟢 <strong>System ready</strong><br>
            FastAPI is connected. Ask a question about
            the industrial pump manuals. The backend
            retrieves relevant documentation, reranks
            the evidence, and generates a grounded answer.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <div class="status-card">
            🔴 <strong>Backend unavailable</strong><br>
            Start the FastAPI server before asking a question.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown(
    '<div class="example-label">'
    "Example questions"
    "</div>",
    unsafe_allow_html=True,
)

example_questions = [
    "Why can a pump have excessive discharge pressure?",
    "What can cause abnormal pump vibration?",
    "What should I check if the pump does not start?",
]

example_cols = st.columns(3)

for i, example in enumerate(example_questions):

    with example_cols[i]:

        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True,
        ):

            st.session_state["query"] = example


# ============================================================
# QUESTION INPUT
# ============================================================

st.markdown("### Ask a maintenance question")

query = st.text_area(
    "Question",
    value=st.session_state.get("query", ""),
    placeholder=(
        "Example: Why can a pump have excessive "
        "discharge pressure?"
    ),
    height=100,
    label_visibility="collapsed",
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔍 Ask Copilot",
    type="primary",
    use_container_width=True,
)


# ============================================================
# SEND QUESTION TO FASTAPI
# ============================================================

if ask_button:

    if not query.strip():

        st.warning(
            "Please enter a maintenance question."
        )

    elif not api_online:

        st.error(
            "FastAPI backend is not running. "
            "Start it with: "
            "`uvicorn api.main:app --reload`"
        )

    else:

        with st.spinner(
            "Searching technical documentation "
            "and generating answer..."
        ):

            try:

                result = ask_api(
                    query.strip()
                )

                status = result.get(
                    "status",
                    "unknown",
                )

                answer = result.get(
                    "answer",
                    "",
                )

                sources = result.get(
                    "sources",
                    [],
                )

                st.session_state["messages"].append(
                    {
                        "question": query.strip(),
                        "answer": answer,
                        "status": status,
                        "sources": sources,
                    }
                )

                st.session_state["query"] = ""

                st.rerun()

            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. "
                    "The AI backend may still be processing "
                    "the request."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI backend."
                )

            except requests.exceptions.HTTPError as e:

                st.error(
                    "FastAPI returned an HTTP error."
                )

                st.exception(e)

            except Exception as e:

                st.error(
                    "The application encountered an unexpected error."
                )

                st.exception(e)


# ============================================================
# CONVERSATION HISTORY
# ============================================================

if st.session_state["messages"]:

    st.divider()

    for message in st.session_state["messages"]:

        question = message["question"]
        answer = message["answer"]
        status = message["status"]
        sources = message["sources"]

        # ----------------------------------------------------
        # QUESTION
        # ----------------------------------------------------

        st.markdown("### 👤 Question")

        st.info(question)

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.markdown("### 🤖 Answer")

        if status == "success":

            st.markdown(answer)

        elif status == "quota_exceeded":

            st.warning(
                "AI generation is temporarily unavailable "
                "because the Gemini API quota has been exceeded."
            )

        else:

            st.error(
                "The AI answer could not be generated."
            )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if sources:

            st.markdown("### 📚 Sources")

            for i, source in enumerate(
                sources,
                start=1,
            ):

                source_file = source.get(
                    "source_file",
                    "Unknown document",
                )

                page = source.get(
                    "page",
                    "Unknown",
                )

                section = source.get(
                    "section",
                    "",
                )

                text = source.get(
                    "text",
                    "",
                )

                title = (
                    f"{i}. "
                    f"{source_file} "
                    f"— Page {page}"
                )

                with st.expander(title):

                    if section:

                        st.caption(
                            f"Section: {section}"
                        )

                    if text:

                        st.markdown(
                            "**Relevant excerpt:**"
                        )

                        st.write(text)

                    else:

                        st.caption(
                            "Source excerpt unavailable."
                        )

        st.divider()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Industrial Maintenance Copilot ·
        Retrieval-Augmented Generation for
        industrial pump documentation
    </div>
    """,
    unsafe_allow_html=True,
)
