from retrieval import (
    create_qdrant_client,
    create_embedding_model,
    search,
)


QUERIES = [
    "Why can a pump have excessive discharge pressure?",
    "What can cause a pump to overheat?",
    "What should I check if the pump does not start?",
    "How can I protect a pump from overpressure?",
    "What should I check if the pump has abnormal vibration?",
]


if __name__ == "__main__":

    client = create_qdrant_client()
    model = create_embedding_model()

    for query in QUERIES:

        print("\n" + "=" * 80)
        print("QUERY:", query)
        print("=" * 80)

        results = search(
            query=query,
            client=client,
            model=model,
            top_k=5
        )

        for i, result in enumerate(results, start=1):

            print(f"\n--- Result {i} ---")
            print(f"Score : {result.score:.4f}")
            print(f"Source: {result.payload.get('source_file')}")
            print(f"Page  : {result.payload.get('page')}")

            text = result.payload.get("text", "")
            print(f"Text  : {text[:500]}")

    client.close()