from retrieval import (
    create_qdrant_client,
    create_embedding_model,
    search,
)

from reranking import (
    create_reranker,
    rerank,
)


QUERY = "Why can a pump have excessive discharge pressure?"


if __name__ == "__main__":

    print("Loading models...")

    client = create_qdrant_client()
    embedding_model = create_embedding_model()
    reranker = create_reranker()

    print("\nSearching Qdrant...")

    results = search(
        query=QUERY,
        client=client,
        model=embedding_model,
        top_k=10,
    )

    print(f"Retrieved {len(results)} candidates.")

    print("\nReranking...")

    reranked_results = rerank(
        query=QUERY,
        results=results,
        reranker=reranker,
        top_k=3,
    )

    print("\n" + "=" * 80)
    print("RERANKED RESULTS")
    print("=" * 80)

    for i, item in enumerate(reranked_results, start=1):

        result = item["result"]
        score = item["rerank_score"]

        print(f"\n--- Result {i} ---")
        print(f"Rerank score : {score:.4f}")
        print(f"Qdrant score : {result.score:.4f}")
        print(f"Source       : {result.payload.get('source_file')}")
        print(f"Page         : {result.payload.get('page')}")

        text = result.payload.get("text", "")

        print("\nText:")
        print(text[:800])

    client.close()