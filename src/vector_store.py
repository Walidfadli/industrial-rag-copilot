from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ingestion import (
    load_documents,
    assign_sections,
    split_documents,
)

from embeddings import (
    create_embedding_model,
    create_embeddings,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

COLLECTION_NAME = "industrial_maintenance"
VECTOR_SIZE = 384
QDRANT_PATH = "qdrant_storage"


# --------------------------------------------------
# Create Qdrant client
# --------------------------------------------------

def create_qdrant_client():
    return QdrantClient(path=QDRANT_PATH)


# --------------------------------------------------
# Create / recreate collection
# --------------------------------------------------

def create_collection(client):

    if client.collection_exists(COLLECTION_NAME):

        print(
            f"Collection '{COLLECTION_NAME}' already exists."
        )

        print("Deleting old collection...")

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Collection '{COLLECTION_NAME}' created."
    )


# --------------------------------------------------
# Insert documents
# --------------------------------------------------

def insert_documents(
    client,
    chunks,
    embeddings,
):

    points = []

    for i, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        point = PointStruct(
            id=i,

            vector=embedding.tolist(),

            payload={
                # Main chunk content
                "text": chunk.page_content,

                # Document information
                "source_file": chunk.metadata.get(
                    "source_file",
                    "unknown",
                ),

                "page": chunk.metadata.get(
                    "page",
                    "unknown",
                ),

                # Structural metadata
                "section": chunk.metadata.get(
                    "section",
                    "GENERAL",
                ),

                "chunk_index": chunk.metadata.get(
                    "chunk_index",
                    0,
                ),

                "chunk_id": chunk.metadata.get(
                    "chunk_id",
                    f"chunk_{i}",
                ),
            },
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(
        f"{len(points)} chunks inserted into Qdrant."
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("BUILDING INDUSTRIAL MAINTENANCE VECTOR DATABASE")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load documents
    # --------------------------------------------------

    documents = load_documents()

    print(
        f"\nPages loaded: {len(documents)}"
    )

    # --------------------------------------------------
    # 2. Assign sections
    # --------------------------------------------------

    documents = assign_sections(
        documents
    )

    # --------------------------------------------------
    # 3. Split documents into chunks
    # --------------------------------------------------

    chunks = split_documents(
        documents
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    # --------------------------------------------------
    # 4. Load embedding model
    # --------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    model = create_embedding_model()

    # --------------------------------------------------
    # 5. Extract text
    # --------------------------------------------------

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    # --------------------------------------------------
    # 6. Create embeddings
    # --------------------------------------------------

    print(
        "Creating embeddings..."
    )

    embeddings = create_embeddings(
        model,
        texts,
    )

    print(
        f"Embeddings shape: {embeddings.shape}"
    )

    # --------------------------------------------------
    # 7. Create Qdrant client
    # --------------------------------------------------

    client = create_qdrant_client()

    # --------------------------------------------------
    # 8. Recreate collection
    # --------------------------------------------------

    print(
        "\nCreating Qdrant collection..."
    )

    create_collection(
        client
    )

    # --------------------------------------------------
    # 9. Insert chunks + embeddings
    # --------------------------------------------------

    print(
        "\nIndexing chunks..."
    )

    insert_documents(
        client,
        chunks,
        embeddings,
    )

    # --------------------------------------------------
    # 10. Verify collection
    # --------------------------------------------------

    collection_info = client.get_collection(
        COLLECTION_NAME
    )

    print("\n" + "=" * 70)
    print("QDRANT IS READY")
    print("=" * 70)

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Points indexed: "
        f"{collection_info.points_count}"
    )

    print(
        f"Vector size: {VECTOR_SIZE}"
    )

    print(
        "Distance: COSINE"
    )

    # --------------------------------------------------
    # Show example metadata
    # --------------------------------------------------

    print("\nExample metadata:")

    print(
        {
            "source_file": chunks[0].metadata.get(
                "source_file"
            ),
            "page": chunks[0].metadata.get(
                "page"
            ),
            "section": chunks[0].metadata.get(
                "section"
            ),
            "chunk_index": chunks[0].metadata.get(
                "chunk_index"
            ),
            "chunk_id": chunks[0].metadata.get(
                "chunk_id"
            ),
        }
    )

    client.close()

    print(
        "\nVector database rebuild completed."
    )
