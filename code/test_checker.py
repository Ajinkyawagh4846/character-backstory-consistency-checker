"""
Smoke test for ConsistencyChecker using a sample backstory.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure we can import local modules
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from consistency_checker import ConsistencyChecker
except Exception as exc:  # noqa: BLE001
    raise ImportError(f"Failed to import ConsistencyChecker: {exc}") from exc


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY is not set in the environment.")
        return

    books_dir = Path(__file__).resolve().parent.parent / "books"
    book_path = books_dir / "The Count of Monte Cristo.txt"
    if not book_path.exists():
        print(f"[ERROR] Book not found: {book_path}")
        return

    try:
        novel_text = book_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to read book text: {exc}")
        return

    backstory = (
        "Faria lived quietly on a small island from 1800 onward, drafting a vast work on "
        "human intelligence while sending chapters out by secret courier."
    )

    print("=" * 80)
    print("Running ConsistencyChecker smoke test")
    print("=" * 80)

    try:
        checker = ConsistencyChecker(
            novel_text=novel_text,
            backstory_content=backstory,
            book_name="The Count of Monte Cristo",
            character_name="Faria",
        )
        result = checker.make_final_decision()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Consistency check failed: {exc}")
        return

    prediction = result.get("prediction")
    rationale = result.get("rationale", "")
    claim_results = result.get("claim_results", [])

    print("\n--- RESULTS ---")
    print(f"Prediction : {prediction}")
    print(f"Rationale  : {rationale}")
    print(f"Claims analyzed: {len(claim_results)}")

    print("\nIndividual claims:")
    for idx, claim_res in enumerate(claim_results, start=1):
        claim = claim_res.get("claim", "<unknown claim>")
        consistency = claim_res.get("consistency")
        confidence = claim_res.get("confidence")
        reasoning = claim_res.get("reasoning", "")
        print("-" * 40)
        print(f"[{idx}] Claim      : {claim}")
        print(f"     Consistency: {consistency}")
        print(f"     Confidence : {confidence}")
        print(f"     Reasoning  : {reasoning}")

    print("\nDone.")


if __name__ == "__main__":
    main()

