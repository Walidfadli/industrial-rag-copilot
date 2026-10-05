import os
import time

from dotenv import load_dotenv
from google import genai

from src.retrieval import (
    create_qdrant_client,
    create_embedding_model,
    search,
)

from src.reranking import (
    create_reranker,
    rerank,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5

MODEL_NAME = "gemini-3.8-flash"

MAX_RETRIES = 3


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Make sure it is defined in your .env file."
    )


# --------------------------------------------------
# Create Gemini client
# --------------------------------------------------

def create_llm():
    return genai.Client(
        api_key=GEMINI_API_KEY
    )


# --------------------------------------------------
# Build context
# --------------------------------------------------

def build_context(reranked_results):

    context_parts = []

    for i, item in enumerate(
        reranked_results,
        start=1
    ):

        result = item["result"]

        source = result.payload.get(
            "source_file",
            "Unknown"
        )

        page = result.payload.get(
            "page",
            "Unknown"
        )

        section = result.payload.get(
            "section",
            "Unknown"
        )

        text = result.payload.get(
            "text",
            ""
        )

        context_parts.append(
            f"""
[SOURCE {i}]
Document: {source}
Page: {page}
Section: {section}

{text}
"""
        )

    return "\n".join(context_parts)


# --------------------------------------------------
# Build prompt
# --------------------------------------------------

def build_prompt(query, context):

    return f"""
You are an AI assistant specialized in industrial
pump maintenance.

Answer the user's question using ONLY the technical
documentation provided below.

GROUNDING RULES:

1. Use only information explicitly supported by the
   provided documentation.

2. The retrieved passages may come from different
   parts of the manuals. Combine information from
   multiple passages when they are relevant to the
   same question.

3. If a retrieved passage directly addresses the
   user's question, USE that evidence to answer.
   Do not refuse to answer simply because the passage
   does not contain every possible detail.

4. Do not use outside knowledge or assumptions.

5. Do not invent causes, procedures, values,
   recommendations, or safety instructions.

6. Include the relevant causes, conditions, checks,
   or remedies that are explicitly supported by the
   documentation.

7. If the documentation provides only part of the
   answer, give the supported part and clearly state
   what information is not available.

8. Only say:

   "I could not find enough information to answer
   this question in the provided manuals."

   when NONE of the retrieved passages provides
   useful evidence for answering the question.

9. Every important technical claim must include a
   citation using this format:

   [document_name, p. X]

10. Only cite document names and page numbers that
    appear in the provided documentation.

11. Keep the answer concise and practical.

12. Do not mention Qdrant, embeddings, reranking,
    retrieval, prompts, or the internal RAG system.

TECHNICAL DOCUMENTATION
=======================

{context}

USER QUESTION
=============

{query}

ANSWER
======
"""


# --------------------------------------------------
# Generate Gemini response
# --------------------------------------------------

def generate_answer(llm, prompt):

    for attempt in range(MAX_RETRIES):

        try:

            response = llm.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            if not response.text:
                return (
                    "generation_failed",
                    "The AI model returned an empty response."
                )

            return "success", response.text

        except Exception as e:

            error_message = str(e)

            # ------------------------------------------
            # Gemini quota exceeded
            # ------------------------------------------

            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            ):

                return (
                    "quota_exceeded",
                    "Gemini's current free-tier quota has "
                    "been reached. The technical documents "
                    "were successfully retrieved, but an "
                    "AI-generated answer cannot be produced "
                    "until the quota resets."
                )

            # ------------------------------------------
            # Temporary Gemini server error
            # ------------------------------------------

            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                if attempt < MAX_RETRIES - 1:

                    wait_time = 2 ** attempt

                    print(
                        "\nGemini temporarily unavailable."
                    )

                    print(
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

                return (
                    "generation_failed",
                    "Gemini is temporarily unavailable. "
                    "The technical documentation was "
                    "successfully retrieved, but the answer "
                    "could not be generated."
                )

            # ------------------------------------------
            # Unexpected error
            # ------------------------------------------

            raise


# --------------------------------------------------
# Extract source information
# --------------------------------------------------

def extract_sources(reranked_results):
    """
    Extract source information from reranked results,
    including the original retrieved text.
    """

    sources = []

    for rank, item in enumerate(
        reranked_results,
        start=1,
    ):

        result = item["result"]

        payload = result.payload or {}

        sources.append(
            {
                "rank": rank,

                "source_file": payload.get(
                    "source_file",
                    "Unknown",
                ),

                "page": payload.get(
                    "page",
                    "Unknown",
                ),

                "section": payload.get(
                    "section",
                    "",
                ),

                "text": payload.get(
                    "text",
                    "",
                ),

                "qdrant_score": float(
                    result.score
                ),

                "rerank_score": float(
                    item.get(
                        "rerank_score",
                        0,
                    )
                ),
            }
        )

    return sources
# --------------------------------------------------
# Full RAG pipeline
# --------------------------------------------------

def answer_question(
    query,
    client,
    embedding_model,
    reranker,
    llm,
):

    # ----------------------------------------------
    # Step 1: Dense retrieval
    # ----------------------------------------------

    results = search(
        query=query,
        client=client,
        model=embedding_model,
        top_k=RETRIEVAL_TOP_K,
    )

    print(
        f"\nRetrieved {len(results)} candidates "
        f"from Qdrant."
    )

    # ----------------------------------------------
    # Step 2: Cross-encoder reranking
    # ----------------------------------------------

    reranked_results = rerank(
        query=query,
        results=results,
        reranker=reranker,
        top_k=RERANK_TOP_K,
    )

    print(
        f"Reranked down to "
        f"{len(reranked_results)} chunks."
    )

    # ----------------------------------------------
    # Step 3: Build context
    # ----------------------------------------------

    context = build_context(
        reranked_results
    )


    # ----------------------------------------------
    # Step 4: Build grounded prompt
    # ----------------------------------------------

    prompt = build_prompt(
        query=query,
        context=context,
    )

    # ----------------------------------------------
    # Step 5: Generate answer
    # ----------------------------------------------

    status, answer = generate_answer(
        llm=llm,
        prompt=prompt,
    )

    # ----------------------------------------------
    # Step 6: Extract sources
    # ----------------------------------------------

    sources = extract_sources(
        reranked_results
    )

    return {
        "status": status,
        "answer": answer,
        "sources": sources,
        "retrieved_count": len(results),
        "reranked_count": len(reranked_results),
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("Starting RAG system...")

    # ----------------------------------------------
    # Create components
    # ----------------------------------------------

    client = create_qdrant_client()

    embedding_model = create_embedding_model()

    reranker = create_reranker()

    llm = create_llm()

    # ----------------------------------------------
    # Test query
    # ----------------------------------------------

    query = (
        "Why can a pump have "
        "excessive discharge pressure?"
    )

    # ----------------------------------------------
    # Run RAG
    # ----------------------------------------------

    result = answer_question(
        query=query,
        client=client,
        embedding_model=embedding_model,
        reranker=reranker,
        llm=llm,
    )

    # ----------------------------------------------
    # Display answer
    # ----------------------------------------------

    print("\n" + "=" * 70)
    print("RAG ANSWER")
    print("=" * 70)

    print(
        f"\nStatus: {result['status']}"
    )

    print(
        f"\n{result['answer']}"
    )

    # ----------------------------------------------
    # Display sources
    # ----------------------------------------------

    print("\n" + "=" * 70)
    print("RERANKED SOURCES")
    print("=" * 70)

    for source in result["sources"]:

        print(
            f"{source['rank']}. "
            f"{source['source_file']} | "
            f"Page {source['page']} | "
            f"Section: {source['section']} | "
            f"Qdrant: {source['qdrant_score']:.4f} | "
            f"Rerank: {source['rerank_score']:.4f}"
        )

    # ----------------------------------------------
    # Close Qdrant
    # ----------------------------------------------

    client.close()

    print(
        "\nRAG pipeline completed."
    )
