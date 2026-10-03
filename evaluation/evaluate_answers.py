import json
from pathlib import Path

from src.rag import (
    create_llm,
    create_qdrant_client,
    create_embedding_model,
    create_reranker,
    build_context,
    build_prompt,
    generate_answer,
)

from src.retrieval import search
from src.reranking import rerank


# ============================================================
# CONFIGURATION
# ============================================================

QUESTIONS_FILE = Path("evaluation/test_questions.json")
OUTPUT_FILE = Path("evaluation/answer_evaluation.json")

RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5


# ============================================================
# GENERATION FAILURE DETECTION
# ============================================================

def is_generation_failure(answer):
    """
    Detect whether Gemini failed to generate a real answer.

    Generation failures are excluded from:
    - completeness
    - relevance
    - groundedness

    This prevents API failures from being treated as
    poor-quality RAG answers.
    """

    if not answer:
        return True

    failure_markers = [
        # Temporary Gemini/server failure
        "AI generation is temporarily unavailable.",
        "language model could not generate the answer",

        # Gemini quota/rate-limit failure
        "AI generation quota exceeded.",
        "Gemini's current free-tier request quota has been reached",
        "RESOURCE_EXHAUSTED",
        "429",

        # Other known failure wording
        "generation failed",
    ]

    answer_lower = answer.lower()

    return any(
        marker.lower() in answer_lower
        for marker in failure_markers
    )


# ============================================================
# MANUAL EVALUATION
# ============================================================

def ask_yes_no(prompt):
    """
    Ask the evaluator for a yes/no answer.
    """

    while True:

        answer = input(prompt).strip().lower()

        if answer in ["y", "n"]:
            return answer

        print("Please enter y or n.")


def evaluate_successful_answer(answer, expected_points):
    """
    Manually evaluate completeness, relevance
    and groundedness.

    This function is ONLY called when Gemini
    successfully generated a real answer.
    """

    covered_points = []

    # --------------------------------------------------------
    # Expected answer points
    # --------------------------------------------------------

    print("-" * 80)
    print("EXPECTED ANSWER POINTS")
    print("-" * 80)

    for i, point in enumerate(
        expected_points,
        start=1
    ):

        print(f"{i}. {point}")

    # --------------------------------------------------------
    # Completeness evaluation
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("MANUAL EVALUATION")
    print("-" * 80)

    print(
        "\nFor each expected point, decide whether "
        "the generated answer covers it."
    )

    for i, point in enumerate(
        expected_points,
        start=1
    ):

        print(f"\nPoint {i}:")
        print(point)

        result = ask_yes_no(
            "Covered? [y/n]: "
        )

        covered = result == "y"

        covered_points.append(
            covered
        )

        if covered:
            print("✓ Covered")
        else:
            print("✗ Missing")

    # --------------------------------------------------------
    # Calculate completeness
    # --------------------------------------------------------

    total_points = len(
        expected_points
    )

    points_covered = sum(
        covered_points
    )

    if total_points > 0:

        completeness = (
            points_covered
            / total_points
        )

    else:

        completeness = None

    print("\n" + "-" * 80)
    print("COMPLETENESS RESULT")
    print("-" * 80)

    print(
        f"Covered points: "
        f"{points_covered}/{total_points}"
    )

    if completeness is not None:

        print(
            f"Completeness: "
            f"{completeness * 100:.2f}%"
        )

    else:

        print(
            "Completeness: N/A "
            "(no expected points)"
        )

    # --------------------------------------------------------
    # Relevance
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("RELEVANCE")
    print("-" * 80)

    relevance = ask_yes_no(
        "\nDoes the answer directly address "
        "the question? [y/n]: "
    )

    if relevance == "y":

        print("✓ Relevant")

    else:

        print("✗ Not relevant")

    # --------------------------------------------------------
    # Groundedness
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("GROUNDEDNESS")
    print("-" * 80)

    groundedness = ask_yes_no(
        "\nDoes the answer stay grounded in "
        "the retrieved documentation? [y/n]: "
    )

    if groundedness == "y":

        print("✓ Grounded")

    else:

        print("✗ Potentially unsupported")

    # --------------------------------------------------------
    # Return evaluation
    # --------------------------------------------------------

    return {
        "evaluation_status": "evaluated",

        "covered_points": covered_points,

        "points_covered": points_covered,

        "total_points": total_points,

        "completeness": completeness,

        "relevance": relevance == "y",

        "groundedness": groundedness == "y",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("INDUSTRIAL MAINTENANCE RAG")
    print("ANSWER GENERATION EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        questions = json.load(f)

    print(
        f"\nEvaluation questions: "
        f"{len(questions)}"
    )

    print(
        f"Qdrant Top-K: "
        f"{RETRIEVAL_TOP_K}"
    )

    print(
        f"Reranker Top-K: "
        f"{RERANK_TOP_K}"
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("\nLoading models...")

    llm = create_llm()

    qdrant_client = create_qdrant_client()

    embedding_model = create_embedding_model()

    reranker = create_reranker()

    print("Models loaded.")

    # --------------------------------------------------------
    # Evaluation results
    # --------------------------------------------------------

    evaluation_results = []

    # --------------------------------------------------------
    # Main evaluation loop
    # --------------------------------------------------------

    for index, question_data in enumerate(
        questions,
        start=1
    ):

        question_id = question_data["id"]

        question = question_data["question"]

        expected_sources = question_data.get(
            "expected_sources",
            []
        )

        expected_points = question_data.get(
            "expected_points",
            []
        )

        print("\n" + "=" * 80)

        print(
            f"QUESTION "
            f"{index}/{len(questions)}"
        )

        print("=" * 80)

        print(
            f"\nID: {question_id}"
        )

        print(
            f"Question: {question}"
        )

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        results = search(
            question,
            qdrant_client,
            embedding_model,
            top_k=RETRIEVAL_TOP_K,
        )

        # ----------------------------------------------------
        # Reranking
        # ----------------------------------------------------

        reranked_results = rerank(
            question,
            results,
            reranker,
            top_k=RERANK_TOP_K,
        )

        print("\nRetrieved sources:")

        for rank, item in enumerate(
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

            print(
                f"  {rank}. "
                f"{source} "
                f"(page {page})"
            )

        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context = build_context(
            reranked_results
        )

        # ----------------------------------------------------
        # Build prompt
        # ----------------------------------------------------

        prompt = build_prompt(
            question,
            context
        )

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        print(
            "\nGenerating Gemini answer..."
        )

        answer = generate_answer(
            llm,
            prompt
        )

        print("\n" + "-" * 80)
        print("GENERATED ANSWER")
        print("-" * 80)

        print(answer)

        # ----------------------------------------------------
        # Detect generation failure
        # ----------------------------------------------------

        generation_failed = (
            is_generation_failure(answer)
        )

        if generation_failed:

            print("\n" + "-" * 80)
            print("GENERATION STATUS")
            print("-" * 80)

            print(
                "⚠️ Generation failed. "
                "Skipping completeness, relevance "
                "and groundedness evaluation."
            )

            evaluation = {

                "evaluation_status":
                    "generation_failed",

                "covered_points":
                    None,

                "points_covered":
                    None,

                "total_points":
                    len(expected_points),

                "completeness":
                    None,

                "relevance":
                    None,

                "groundedness":
                    None,
            }

        else:

            evaluation = (
                evaluate_successful_answer(
                    answer,
                    expected_points
                )
            )

        # ----------------------------------------------------
        # Save question result
        # ----------------------------------------------------

        result = {

            "id":
                question_id,

            "question":
                question,

            "expected_sources":
                expected_sources,

            "expected_points":
                expected_points,

            "generated_answer":
                answer,

            "generation_success":
                not generation_failed,

            "evaluation":
                evaluation,
        }

        evaluation_results.append(
            result
        )

        print("\n" + "=" * 80)

        print(
            f"{question_id} "
            f"evaluation complete"
        )

        print("=" * 80)

    # ========================================================
    # SUMMARY
    # ========================================================

    successful = [
        r
        for r in evaluation_results
        if r["generation_success"]
    ]

    failed = [
        r
        for r in evaluation_results
        if not r["generation_success"]
    ]

    print(
        "\n\n" + "=" * 80
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 80
    )

    total = len(
        evaluation_results
    )

    print(
        f"\nTotal questions: "
        f"{total}"
    )

    print(
        f"Successful generations: "
        f"{len(successful)}/{total}"
    )

    print(
        f"Generation failures: "
        f"{len(failed)}/{total}"
    )

    # --------------------------------------------------------
    # Answer quality metrics
    # --------------------------------------------------------

    if successful:

        completeness_scores = [
            r["evaluation"]["completeness"]
            for r in successful
            if (
                r["evaluation"]["completeness"]
                is not None
            )
        ]

        relevance_scores = [
            r["evaluation"]["relevance"]
            for r in successful
            if (
                r["evaluation"]["relevance"]
                is not None
            )
        ]

        groundedness_scores = [
            r["evaluation"]["groundedness"]
            for r in successful
            if (
                r["evaluation"]["groundedness"]
                is not None
            )
        ]

        # ----------------------------------------------------
        # Calculate metrics safely
        # ----------------------------------------------------

        if completeness_scores:

            average_completeness = (
                sum(completeness_scores)
                / len(completeness_scores)
            )

        else:

            average_completeness = None

        if relevance_scores:

            relevance_rate = (
                sum(relevance_scores)
                / len(relevance_scores)
            )

        else:

            relevance_rate = None

        if groundedness_scores:

            groundedness_rate = (
                sum(groundedness_scores)
                / len(groundedness_scores)
            )

        else:

            groundedness_rate = None

        # ----------------------------------------------------
        # Display metrics
        # ----------------------------------------------------

        print(
            "\n" + "-" * 80
        )

        print(
            "SUCCESSFUL GENERATIONS ONLY"
        )

        print(
            "-" * 80
        )

        if average_completeness is not None:

            print(
                f"\nAverage completeness: "
                f"{average_completeness * 100:.2f}%"
            )

        else:

            print(
                "\nAverage completeness: N/A"
            )

        if relevance_rate is not None:

            print(
                f"Relevance rate: "
                f"{relevance_rate * 100:.2f}%"
            )

        else:

            print(
                "Relevance rate: N/A"
            )

        if groundedness_rate is not None:

            print(
                f"Groundedness rate: "
                f"{groundedness_rate * 100:.2f}%"
            )

        else:

            print(
                "Groundedness rate: N/A"
            )

    else:

        average_completeness = None

        relevance_rate = None

        groundedness_rate = None

        print(
            "\nNo successful generations available "
            "for answer-quality evaluation."
        )

    # --------------------------------------------------------
    # Failed questions
    # --------------------------------------------------------

    if failed:

        print(
            "\n" + "-" * 80
        )

        print(
            "GENERATION FAILURES"
        )

        print(
            "-" * 80
        )

        for r in failed:

            print(
                f"\n{r['id']}: "
                f"{r['question']}"
            )

            print(
                "Status: generation_failed "
                "(excluded from answer-quality metrics)"
            )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output = {

        "summary": {

            "total_questions":
                total,

            "successful_generations":
                len(successful),

            "generation_failures":
                len(failed),
        },

        "results":
            evaluation_results,
    }

    # --------------------------------------------------------
    # Add answer-quality metrics
    # --------------------------------------------------------

    if successful:

        output["summary"].update({

            "average_completeness":
                average_completeness,

            "relevance_rate":
                relevance_rate,

            "groundedness_rate":
                groundedness_rate,
        })

    # --------------------------------------------------------
    # Write evaluation file
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nResults saved to: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

