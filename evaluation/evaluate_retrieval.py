import json
import sys
from pathlib import Path

# ============================================================
# ADD SRC/ TO PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from retrieval import (
    create_qdrant_client,
    create_embedding_model,
    search,
)

from reranking import (
    create_reranker,
    rerank,
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = Path(__file__).parent / "test_questions.json"

RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5


# ============================================================
# LOAD TEST QUESTIONS
# ============================================================

def load_test_questions():

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# CHECK WHETHER A SOURCE MATCHES
# ============================================================

def source_matches(result, expected_source):

    source_file = result.payload.get("source_file")
    page = result.payload.get("page")

    return (
        source_file == expected_source["source_file"]
        and page == expected_source["page"]
    )


# ============================================================
# FIND EXPECTED SOURCES IN RESULTS
# ============================================================

def find_matches(results, expected_sources):

    matches = []

    for expected in expected_sources:

        for rank, item in enumerate(results, start=1):

            # Raw Qdrant result
            if isinstance(item, dict):

                # Reranked result
                result = item["result"]

            else:

                # Normal Qdrant result
                result = item

            if source_matches(result, expected):

                matches.append(
                    {
                        "source_file": expected["source_file"],
                        "page": expected["page"],
                        "rank": rank,
                    }
                )

                break

    return matches


# ============================================================
# CALCULATE MRR
# ============================================================

def calculate_mrr(matches):

    if not matches:
        return 0.0

    # Rank of the first relevant result
    first_rank = min(
        match["rank"]
        for match in matches
    )

    return 1 / first_rank


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("INDUSTRIAL MAINTENANCE RAG")
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load evaluation questions
    # --------------------------------------------------------

    test_questions = load_test_questions()

    print(
        f"\nTest questions: {len(test_questions)}"
    )

    print(
        f"Qdrant Top-K: {RETRIEVAL_TOP_K}"
    )

    print(
        f"Reranker Top-K: {RERANK_TOP_K}"
    )

    # --------------------------------------------------------
    # Create models once
    # --------------------------------------------------------

    client = create_qdrant_client()

    embedding_model = create_embedding_model()

    reranker = create_reranker()

    # --------------------------------------------------------
    # Initialize metrics
    # --------------------------------------------------------

    qdrant_hits = 0
    reranked_hits = 0

    qdrant_mrr_total = 0.0
    reranked_mrr_total = 0.0

    qdrant_recall_total = 0.0
    reranked_recall_total = 0.0

    print("\n" + "=" * 70)

    # ========================================================
    # EVALUATE EACH QUESTION
    # ========================================================

    for test in test_questions:

        question_id = test["id"]

        question = test["question"]

        expected_sources = test["expected_sources"]

        print(
            f"\n{question_id}: {question}"
        )

        # ----------------------------------------------------
        # 1. Dense retrieval
        # ----------------------------------------------------

        retrieved_results = search(
            query=question,
            client=client,
            model=embedding_model,
            top_k=RETRIEVAL_TOP_K,
        )

        # ----------------------------------------------------
        # 2. Evaluate raw Qdrant retrieval
        # ----------------------------------------------------

        raw_matches = find_matches(
            retrieved_results,
            expected_sources,
        )

        # Hit@10
        if raw_matches:

            qdrant_hits += 1

            qdrant_status = "PASS"

        else:

            qdrant_status = "FAIL"

        # MRR@10
        qdrant_mrr_total += calculate_mrr(
            raw_matches
        )

        # Source Recall@10
        qdrant_recall_total += (
            len(raw_matches)
            / len(expected_sources)
        )

        # ----------------------------------------------------
        # 3. Reranking
        # ----------------------------------------------------

        reranked_results = rerank(
            query=question,
            results=retrieved_results,
            reranker=reranker,
            top_k=RERANK_TOP_K,
        )

        # ----------------------------------------------------
        # 4. Evaluate reranked results
        # ----------------------------------------------------

        reranked_matches = find_matches(
            reranked_results,
            expected_sources,
        )

        # Hit@5
        if reranked_matches:

            reranked_hits += 1

            reranked_status = "PASS"

        else:

            reranked_status = "FAIL"

        # MRR@5
        reranked_mrr_total += calculate_mrr(
            reranked_matches
        )

        # Source Recall@5
        reranked_recall_total += (
            len(reranked_matches)
            / len(expected_sources)
        )

        # ----------------------------------------------------
        # 5. Display question results
        # ----------------------------------------------------

        print(
            f"  Qdrant Top-{RETRIEVAL_TOP_K}: "
            f"{qdrant_status}"
        )

        print(
            f"  Reranked Top-{RERANK_TOP_K}: "
            f"{reranked_status}"
        )

        # ----------------------------------------------------
        # Display Qdrant matches
        # ----------------------------------------------------

        if raw_matches:

            print("\n  Qdrant matches:")

            for match in raw_matches:

                print(
                    f"    - {match['source_file']} "
                    f"(page {match['page']}) "
                    f"→ rank {match['rank']}"
                )

        # ----------------------------------------------------
        # Display reranked matches
        # ----------------------------------------------------

        if reranked_matches:

            print("\n  Reranked matches:")

            for match in reranked_matches:

                print(
                    f"    - {match['source_file']} "
                    f"(page {match['page']}) "
                    f"→ rank {match['rank']}"
                )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    total = len(test_questions)

    # --------------------------------------------------------
    # Hit Rate
    # --------------------------------------------------------

    qdrant_hit_rate = (
        qdrant_hits / total
    )

    reranked_hit_rate = (
        reranked_hits / total
    )

    # --------------------------------------------------------
    # Mean Reciprocal Rank
    # --------------------------------------------------------

    qdrant_mrr = (
        qdrant_mrr_total / total
    )

    reranked_mrr = (
        reranked_mrr_total / total
    )

    # --------------------------------------------------------
    # Source Recall
    # --------------------------------------------------------

    qdrant_recall = (
        qdrant_recall_total / total
    )

    reranked_recall = (
        reranked_recall_total / total
    )

    # ========================================================
    # DISPLAY FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)

    print("FINAL RESULTS")

    print("=" * 70)

    # Hit@K
    print(
        f"\nQdrant Hit@{RETRIEVAL_TOP_K}: "
        f"{qdrant_hit_rate:.2%} "
        f"({qdrant_hits}/{total})"
    )

    print(
        f"Reranked Hit@{RERANK_TOP_K}: "
        f"{reranked_hit_rate:.2%} "
        f"({reranked_hits}/{total})"
    )

    # MRR
    print(
        f"\nQdrant MRR@{RETRIEVAL_TOP_K}: "
        f"{qdrant_mrr:.3f}"
    )

    print(
        f"Reranked MRR@{RERANK_TOP_K}: "
        f"{reranked_mrr:.3f}"
    )

    # Source Recall
    print(
        f"\nQdrant Source Recall@{RETRIEVAL_TOP_K}: "
        f"{qdrant_recall:.2%}"
    )

    print(
        f"Reranked Source Recall@{RERANK_TOP_K}: "
        f"{reranked_recall:.2%}"
    )

    print("\n" + "=" * 70)

    # --------------------------------------------------------
    # Close Qdrant
    # --------------------------------------------------------

    client.close()


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()