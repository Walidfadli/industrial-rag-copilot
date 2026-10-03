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

        text = result.payload.get(
            "text",
            ""
        )

        context_parts.append(
            f"""
SOURCE {i}
Document: {source}
Page: {page}

{text}
"""
        )

    return "\n".join(context_parts)


# --------------------------------------------------
# Build prompt
# --------------------------------------------------

def build_prompt(query, context):

    return f"""
You are an AI assistant specialized in
industrial pump maintenance.

Answer the user's question using ONLY the
technical documentation provided below.

Rules:

1. Answer ONLY using the provided documentation.
2. Do not use outside knowledge or assumptions.
3. Extract all directly relevant causes, conditions,
   and troubleshooting actions from the retrieved
   documentation.
4. Do not omit a relevant troubleshooting cause
   that is explicitly stated in the context.
5. Distinguish between:
   - directly documented causes
   - related operating conditions
   - recommended checks or remedies
6. If the documentation does not provide enough
   evidence for a claim, explicitly say so.
7. Cite the document and page for each important claim.
8. Keep the answer concise and practical.

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
# Generate Gemini response with retry
# --------------------------------------------------

def generate_answer(llm, prompt):

    for attempt in range(MAX_RETRIES):

        try:

            response = llm.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            return response.text

        except Exception as e:

            error_message = str(e)

            # --------------------------------------------------
            # Gemini quota exceeded
            # --------------------------------------------------

            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            ):

                return (
                    "⚠️ **AI generation quota exceeded.**\n\n"
                    "Gemini's current free-tier request quota has been "
                    "reached. The retrieved technical documentation is "
                    "still available, but a new AI-generated answer "
                    "cannot be produced until the quota resets."
                )

            # --------------------------------------------------
            # Temporary Gemini server unavailability
            # --------------------------------------------------

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
                    "⚠️ **AI generation is temporarily unavailable.**\n\n"
                    "The relevant technical documentation was successfully "
                    "retrieved, but the language model could not generate "
                    "the answer at this time.\n\n"
                    "Please try again in a few moments. The retrieved sources "
                    "are still available below."
                )

            # --------------------------------------------------
            # Unexpected error
            # --------------------------------------------------

            raise e


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

    # --------------------------------------------------
    # Step 1: Retrieve candidates
    # --------------------------------------------------

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


    # --------------------------------------------------
    # Step 2: Rerank candidates
    # --------------------------------------------------

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


    # --------------------------------------------------
    # Step 3: Build context
    # --------------------------------------------------

    context = build_context(
        reranked_results
    )


    # --------------------------------------------------
    # Step 4: Build prompt
    # --------------------------------------------------

    prompt = build_prompt(
        query=query,
        context=context,
    )


    # --------------------------------------------------
    # Step 5: Generate answer
    # --------------------------------------------------

    answer = generate_answer(
        llm=llm,
        prompt=prompt,
    )


    return answer, reranked_results


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("Starting RAG system...")


    # --------------------------------------------------
    # Create components
    # --------------------------------------------------

    client = create_qdrant_client()

    embedding_model = create_embedding_model()

    reranker = create_reranker()

    llm = create_llm()


    # --------------------------------------------------
    # User query
    # --------------------------------------------------

    query = (
        "Why can a pump have "
        "excessive discharge pressure?"
    )


    # --------------------------------------------------
    # Run RAG
    # --------------------------------------------------

    answer, results = answer_question(
        query=query,
        client=client,
        embedding_model=embedding_model,
        reranker=reranker,
        llm=llm,
    )


    # --------------------------------------------------
    # Display answer
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("RAG ANSWER")
    print("=" * 70)

    print(answer)


    # --------------------------------------------------
    # Display sources
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("RERANKED SOURCES")
    print("=" * 70)

    for i, item in enumerate(
        results,
        start=1
    ):

        result = item["result"]

        rerank_score = item[
            "rerank_score"
        ]

        source = result.payload.get(
            "source_file",
            "Unknown"
        )

        page = result.payload.get(
            "page",
            "Unknown"
        )

        qdrant_score = result.score

        print(
            f"{i}. {source} | "
            f"Page {page} | "
            f"Qdrant: {qdrant_score:.4f} | "
            f"Rerank: {rerank_score:.4f}"
        )


    # --------------------------------------------------
    # Close Qdrant
    # --------------------------------------------------

    client.close()

    print(
        "\nRAG pipeline completed."
    )

