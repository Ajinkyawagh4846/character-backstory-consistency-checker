"""
Main pipeline for KDSH 2026 - Character Backstory Consistency Checker.

This script:
- Optionally validates the pipeline on a small sample from train.csv.
- Runs the full prediction pipeline on test.csv.
- Saves a submission file to results/submission.csv.
"""

import json
import os
import sys
import time
from typing import Dict, Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai

from consistency_checker import ConsistencyChecker

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

if os.environ.get("GEMINI_API_KEY"):
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])


# ---------------------------------------------------------------------------
# Book loading with simple caching
# ---------------------------------------------------------------------------

_BOOK_CACHE: Dict[str, str] = {}


def load_book(book_name: str) -> str:
    """
    Load book text from books/ folder.

    - Tries exact match with .txt extension.
    - Handles case variations (e.g. 'In Search of the Castaways' vs 'In search of the castaways').
    - Looks only in the top-level 'books' directory.
    - Returns the full text or raises FileNotFoundError with a helpful message.
    """
    if not book_name or not book_name.strip():
        raise ValueError("book_name is empty.")

    # Use module location to be robust to current working directory.
    base_dir = os.path.dirname(os.path.dirname(__file__))
    books_dir = os.path.join(base_dir, "books")

    if not os.path.isdir(books_dir):
        raise FileNotFoundError(f"Books directory not found at: {books_dir}")

    # Cache by original name
    if book_name in _BOOK_CACHE:
        return _BOOK_CACHE[book_name]

    target_stem = book_name.strip()
    target_lower = target_stem.lower()

    # First, try exact filename: "<book_name>.txt"
    exact_path = os.path.join(books_dir, f"{target_stem}.txt")
    if os.path.isfile(exact_path):
        with open(exact_path, "r", encoding="utf-8") as f:
            text = f.read()
        _BOOK_CACHE[book_name] = text
        return text

    # Otherwise, search case-insensitively over available .txt files
    candidates = []
    for fname in os.listdir(books_dir):
        if not fname.lower().endswith(".txt"):
            continue
        stem = os.path.splitext(fname)[0]
        candidates.append(stem)
        if stem.lower() == target_lower:
            path = os.path.join(books_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            _BOOK_CACHE[book_name] = text
            return text

    available = ", ".join(sorted(candidates)) if candidates else "No .txt files found"
    raise FileNotFoundError(
        f"Could not find book '{book_name}' in {books_dir}. "
        f"Available book names (stems): {available}"
    )


# ---------------------------------------------------------------------------
# Single-case processing
# ---------------------------------------------------------------------------

def process_single_case(row: pd.Series, books_cache: Dict[str, str]) -> Dict[str, Any]:
    """Process one test or train case into a prediction result."""
    try:
        case_id = row["id"]
        book_name = row["book_name"]
        character = row["char"]
        backstory = row["content"]

        # Load and cache book text
        if book_name not in books_cache:
            print(f"\n[INFO] Loading book text for '{book_name}'...")
            books_cache[book_name] = load_book(book_name)
        novel_text = books_cache[book_name]

        checker = ConsistencyChecker(
            novel_text=novel_text,
            backstory_content=backstory,
            book_name=book_name,
            character_name=character,
        )

        result = checker.make_final_decision()

        return {
            "id": case_id,
            "prediction": result["prediction"],
            "rationale": result["rationale"],
            "book": book_name,
            "character": character,
        }
    except Exception as e:  # noqa: BLE001
        print(f"❌ Error processing case {row.get('id', '<unknown>')}: {e}")
        # Default to "consistent" on errors to avoid over-flagging contradictions.
        return {
            "id": row.get("id", None),
            "prediction": "consistent",
            "rationale": f"Error: {str(e)[:100]}",
            "book": row.get("book_name", ""),
            "character": row.get("char", ""),
        }


# ---------------------------------------------------------------------------
# Validation on train.csv
# ---------------------------------------------------------------------------

def validate_on_train() -> float:
    """Run a small validation on train.csv (first 5 samples) to inspect behavior."""
    print("\n" + "=" * 80)
    print("🧪 VALIDATION ON TRAIN.CSV (First 5 samples)")
    print("=" * 80)

    train_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train.csv")
    train_df = pd.read_csv(train_path)
    sample_df = train_df.head(5)

    books_cache: Dict[str, str] = {}
    correct = 0
    total = 0

    for _, row in sample_df.iterrows():
        result = process_single_case(row, books_cache)
        actual = row["label"]
        predicted = result["prediction"]

        match = "✅" if predicted == actual else "❌"
        print(f"{match} ID {result['id']}: Predicted={predicted}, Actual={actual}")

        if predicted == actual:
            correct += 1
        total += 1

    accuracy = (correct / total) * 100 if total else 0.0
    print(f"\n📊 Validation Accuracy: {accuracy:.1f}% ({correct}/{total})")
    return accuracy


# ---------------------------------------------------------------------------
# Full test.csv processing
# ---------------------------------------------------------------------------

def process_test_set() -> pd.DataFrame:
    """Process the entire test.csv and write results/submission.csv."""
    print("\n" + "=" * 80)
    print("🚀 PROCESSING TEST.CSV")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(__file__))
    test_path = os.path.join(base_dir, "test.csv")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    test_df = pd.read_csv(test_path)
    print(f"📋 Total test cases: {len(test_df)}")

    books_cache: Dict[str, str] = {}
    results = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
        result = process_single_case(row, books_cache)
        results.append(result)
        time.sleep(0.5)  # Respect rate limits

    submission_df = pd.DataFrame(results)
    output_path = os.path.join(results_dir, "submission.csv")
    submission_df.to_csv(output_path, index=False)

    print(f"\n✅ Results saved to: {output_path}")
    print(f"📊 Predictions: {len(submission_df)} cases")
    print(f"   Consistent: {(submission_df['prediction'] == 'consistent').sum()}")
    print(f"   Contradict: {(submission_df['prediction'] == 'contradict').sum()}")

    return submission_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("🎯 KDSH 2026 - Character Backstory Consistency Checker")
    print("=" * 80)

    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY not found in environment!")
        print("Please create a .env file with your API key.")
        sys.exit(1)

    print("✅ API key loaded")

    try:
        validate = input("\n🧪 Run validation on train.csv first? (y/n): ").lower().strip()
    except EOFError:
        validate = "n"

    if validate == "y":
        _ = validate_on_train()
        try:
            proceed = input("\n➡️  Proceed to test set? (y/n): ").lower().strip()
        except EOFError:
            proceed = "n"
        if proceed != "y":
            print("Exiting...")
            sys.exit(0)

    _ = process_test_set()

    print("\n" + "=" * 80)
    print("✨ COMPLETE! Check results/submission.csv")
    print("=" * 80)


