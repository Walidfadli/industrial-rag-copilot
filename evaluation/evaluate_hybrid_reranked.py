from pathlib import Path
import json

from src.hybrid_search import (
    create_qdrant_client,
    create_embedding_model,
    load_documents_from_qdrant,
    create_bm25_index,
    bm25_search,
    qdrant_search,
    reciprocal_rank_fusion,
)

from src.reranking import (
    create_reranker,
    rerank,
)


TOP_K = 10
RERANK_TOP_K = 5
RRF_K = 60


def load_test_questions():
    path = Path(__file__).parent / "test_questions.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def source_matches(record, expected_source):
    payload = record.payload

    return (
        payload.get("source_file")
        == expected_source["source_file"]
        and
        payload.get("page")
        == expected_source["page"]
    )


def evaluate_question(
    question,
    records,
    bm25,
    client,
    model,
    reranker,
):
    query = question["question"]

    # --------------------------------------------------
    # BM25
    # --------------------------------------------------

    bm25_results = bm25_search(
        query,
        records,
        bm25,
        top_k=TOP_K,
    )

    # --------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------

    dense_results = qdrant_search(
        query,
        client,
        model,
        top_k=TOP_K,
    )

    # --------------------------------------------------
    # Hybrid RRF
    # --------------------------------------------------

    hybrid_results = reciprocal_rank_fusion(
        bm25_results,
        dense_results,
        top_k=TOP_K,
        rrf_k=RRF_K,
    )

    # --------------------------------------------------
    # Convert hybrid results into Qdrant-style results
    # for the existing Cross-Encoder reranker.
    # --------------------------------------------------

    hybrid_records = [
        item["record"]
        for item in hybrid_results
    ]

    # --------------------------------------------------
    # Cross-Encoder reranking
    # --------------------------------------------------

    reranked_results = rerank(
        query,
        hybrid_records,
        reranker,
        top_k=RERANK_TOP_K,
    )

    expected_sources = question["expected_sources"]

    # --------------------------------------------------
    # Evaluate reranked results
    # --------------------------------------------------

    ranks = []

    for rank, item in enumerate(
        reranked_results,
        start=1
    ):
        record = item["result"]

        for expected_source in expected_sources:

            if source_matches(
                record,
                expected_source
            ):
                ranks.append(rank)
                break

    hit_at_5 = len(ranks) > 0

    reciprocal_rank = (
        1 / min(ranks)
        if ranks
        else 0
    )

    matched_sources = set()

    for item in reranked_results:

        record = item["result"]

        for expected_source in expected_sources:

            if source_matches(
                record,
                expected_source
            ):
                matched_sources.add(
                    (
                        expected_source["source_file"],
                        expected_source["page"],
                    )
                )

    source_recall = (
        len(matched_sources)
        / len(expected_sources)
    )

    return {
        "hit_at_5": hit_at_5,
        "reciprocal_rank": reciprocal_rank,
        "source_recall": source_recall,
        "ranks": ranks,
        "results": reranked_results,
    }


def main():

    print("=" * 70)
    print("INDUSTRIAL MAINTENANCE RAG")
    print("HYBRID + RERANKER EVALUATION")
    print("=" * 70)

    questions = load_test_questions()

    print(f"\nTest questions: {len(questions)}")
    print(f"Hybrid Top-K: {TOP_K}")
    print(f"Reranker Top-K: {RERANK_TOP_K}")
    print(f"RRF K: {RRF_K}")

    # --------------------------------------------------
    # Initialize models
    # --------------------------------------------------

    print("\nLoading Qdrant...")

    client = create_qdrant_client()

    print("Loading embedding model...")

    model = create_embedding_model()

    print("Loading reranker...")

    reranker = create_reranker()

    # --------------------------------------------------
    # Load documents
    # --------------------------------------------------

    print("Loading documents...")

    records = load_documents_from_qdrant(client)

    print(
        f"Documents loaded: {len(records)}"
    )

    # --------------------------------------------------
    # Build BM25
    # --------------------------------------------------

    print("\nBuilding BM25 index...")

    bm25 = create_bm25_index(records)

    print("BM25 index ready.")

    print("\n" + "=" * 70)

    results = []

    for question in questions:

        print(
            f"\n{question['id']}: "
            f"{question['question']}"
        )

        evaluation = evaluate_question(
            question,
            records,
            bm25,
            client,
            model,
            reranker,
        )

        results.append(evaluation)

        print(
            f"  Hybrid + Reranker Hit@5: "
            f"{'PASS' if evaluation['hit_at_5'] else 'FAIL'}"
        )

        print(
            f"  MRR contribution: "
            f"{evaluation['reciprocal_rank']:.3f}"
        )

        print(
            f"  Source Recall: "
            f"{evaluation['source_recall']:.2%}"
        )

        print("\n  Matches:")

        for rank in evaluation["ranks"]:

            item = evaluation["results"][rank - 1]

            record = item["result"]
            payload = record.payload

            print(
                f"    - "
                f"{payload.get('source_file')} "
                f"(page {payload.get('page')}) "
                f"→ rank {rank}"
            )

    # --------------------------------------------------
    # Final metrics
    # --------------------------------------------------

    hit_rate = (
        sum(
            result["hit_at_5"]
            for result in results
        )
        / len(results)
    )

    mrr = (
        sum(
            result["reciprocal_rank"]
            for result in results
        )
        / len(results)
    )

    source_recall = (
        sum(
            result["source_recall"]
            for result in results
        )
        / len(results)
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"\nHybrid + Reranker Hit@5: "
        f"{hit_rate:.2%} "
        f"({sum(r['hit_at_5'] for r in results)}/{len(results)})"
    )

    print(
        f"Hybrid + Reranker MRR@5: "
        f"{mrr:.3f}"
    )

    print(
        f"Hybrid + Reranker Source Recall@5: "
        f"{source_recall:.2%}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()