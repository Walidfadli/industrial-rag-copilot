from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def create_embedding_model():
    """
    Load and return the Sentence Transformer embedding model.
    """
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(model, texts):
    """
    Convert a list of texts into embedding vectors.

    Args:
        model: Sentence Transformer model.
        texts: List of text strings.

    Returns:
        A matrix of embeddings.
    """
    return model.encode(
        texts,
        show_progress_bar=True
    )


if __name__ == "__main__":
    # Load the embedding model
    model = create_embedding_model()

    # Test text
    texts = [
        "High discharge pressure can be caused by a blocked discharge line.",
        "Check the pump alignment before starting the equipment.",
        "Inspect the suction line for possible blockage."
    ]

    # Generate embeddings
    embeddings = create_embeddings(model, texts)

    # Display results
    print("\nNumber of texts:", len(texts))
    print("Embeddings shape:", embeddings.shape)

    print("\nFirst embedding:")
    print(embeddings[0][:10])