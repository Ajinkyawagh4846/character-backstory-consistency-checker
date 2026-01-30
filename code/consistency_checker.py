"""
Consistency checking for character backstories using Gemini 2.0 and the
NovelRetriever index.
"""

import json
import os
import re
import sys
import time
from typing import Any, Dict, List

import google.generativeai as genai

# Ensure we can import sibling modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from retriever import NovelRetriever  # noqa: E402

# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]


class ConsistencyChecker:
    """
    Verifies whether a character backstory is consistent with novel text.
    """

    def __init__(self, novel_text: str, backstory_content: str, book_name: str, character_name: str):
        if not novel_text or not novel_text.strip():
            raise ValueError("novel_text is empty.")
        if not backstory_content or not backstory_content.strip():
            raise ValueError("backstory_content is empty.")
        if not book_name or not book_name.strip():
            raise ValueError("book_name is required.")
        if not character_name or not character_name.strip():
            raise ValueError("character_name is required.")

        self.book_name = book_name.strip()
        self.character_name = character_name.strip()
        self.backstory_content = backstory_content.strip()

        print(f"[INIT] Building retriever for book '{self.book_name}'...")
        self.retriever = NovelRetriever(novel_text, self.book_name)

        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config={
                "temperature": 0.2,
            },
            safety_settings=SAFETY_SETTINGS,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _call_model_with_retry(self, prompt: str, max_retries: int = 4) -> str:
        delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                response = self.model.generate_content(
                    prompt,
                    safety_settings=SAFETY_SETTINGS,
                )
                return response.text
            except Exception as exc:  # noqa: BLE001
                print(f"[RETRY] Model call failed (attempt {attempt}/{max_retries}): {exc}")
                if attempt == max_retries:
                    raise
                time.sleep(delay)
                delay *= 2

    @staticmethod
    def _safe_json_loads(payload: str) -> Any:
        """
        Extract and parse JSON from text response, handling markdown code blocks
        and extra text around the JSON.
        """
        if not payload or not payload.strip():
            raise ValueError("Empty payload provided for JSON parsing.")
        
        text = payload.strip()
        
        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks (```json ... ```)
        json_patterns = [
            r'```json\s*(\{.*?\}|\[.*?\])\s*```',  # ```json {...} ```
            r'```\s*(\{.*?\}|\[.*?\])\s*```',      # ``` {...} ```
            r'(\{.*\})',                            # {...} anywhere
            r'(\[.*\])',                            # [...] anywhere
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        
        # Try to find JSON object/array boundaries manually
        # Look for first { or [ and last } or ]
        start_idx = -1
        end_idx = -1
        
        for i, char in enumerate(text):
            if char in '{[':
                start_idx = i
                break
        
        if start_idx >= 0:
            # Find matching closing bracket
            open_char = text[start_idx]
            close_char = '}' if open_char == '{' else ']'
            depth = 0
            
            for i in range(start_idx, len(text)):
                if text[i] == open_char:
                    depth += 1
                elif text[i] == close_char:
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # If all else fails, raise with helpful error
        raise ValueError(
            f"Failed to parse JSON from response. "
            f"Response preview: {text[:200]}..."
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract_backstory_claims(self) -> List[str]:
        """
        Extract atomic, verifiable claims from a backstory.
        """
        prompt = f"""
You are extracting atomic claims.
Character: {self.character_name}
Book: {self.book_name}
Backstory:
\"\"\"{self.backstory_content}\"\"\"

Task:
- Extract 5-7 atomic, verifiable claims about this character.
- Focus on traits, past events, relationships, skills, fears, and motivations.
- Each claim should be concise and checkable against the novel text.

Example format (JSON array of strings):
[
  "He trained as a medic during the uprising.",
  "She distrusts the royal court due to past betrayal."
]
"""
        raw = self._call_model_with_retry(prompt)
        parsed = self._safe_json_loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON list of claims.")
        claims = [c for c in parsed if isinstance(c, str) and c.strip()]
        print(f"[CLAIMS] Extracted {len(claims)} claim(s).")
        return claims

    def check_claim_consistency(self, claim: str) -> Dict[str, Any]:
        """
        Check if a claim is consistent with the novel text.
        """
        if not claim or not claim.strip():
            raise ValueError("Claim is empty.")

        passages = self.retriever.retrieve_relevant_passages(
            query=f"{self.character_name}: {claim}", top_k=7
        )
        formatted_passages = []
        for idx, (text, score, meta) in enumerate(passages, start=1):
            formatted_passages.append(
                f"[{idx}] (score={score:.4f}, pos={meta.get('position')}) {text}"
            )
        passages_blob = "\n".join(formatted_passages) if formatted_passages else "No passages found."

        prompt = f"""
You are verifying backstory consistency.

Character: {self.character_name}
Book: {self.book_name}
Claim to verify: "{claim}"

Retrieved passages:
{passages_blob}

Instructions:
- Check for DIRECT CONTRADICTIONS (explicit conflicts).
- Check CAUSAL CONSISTENCY (does this past make future events plausible?).
- Check BEHAVIORAL PATTERNS (does backstory explain actions?).
- Decide if the character could have this backstory given the text.

Examples of CONSISTENT:
- Claim: "She was a skilled navigator." Passages show her guiding ships successfully.
- Claim: "He vowed to protect his sister." Passages show him guarding her in danger.

Examples of INCONSISTENT:
- Claim: "He loves the monarchy." Passages show he led a revolt against the king.
- Claim: "She never left her village." Passages show her traveling abroad for years.

Return JSON with:
{{
  "consistency": "consistent" or "contradict",
  "confidence": float between 0.0 and 1.0,
  "reasoning": "detailed explanation",
  "key_evidence": "most relevant passage"
}}
"""
        raw = self._call_model_with_retry(prompt)
        parsed = self._safe_json_loads(raw)
        # Basic validation
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object for consistency result.")
        consistency = parsed.get("consistency")
        if consistency not in {"consistent", "contradict"}:
            raise ValueError("consistency must be 'consistent' or 'contradict'.")
        if passages and "key_evidence" not in parsed:
            parsed["key_evidence"] = passages[0][0]
        parsed.setdefault("confidence", 0.0)
        parsed.setdefault("reasoning", "")
        return parsed

    def make_final_decision(self) -> Dict[str, Any]:
        """
        Run full pipeline: extract claims, check consistency, and decide label.
        """
        print(f"Analyzing backstory for {self.character_name}...")
        claims = self.extract_backstory_claims()
        print(f"Found {len(claims)} claims to verify.")

        results = []
        for idx, claim in enumerate(claims, start=1):
            print(f"[CHECK] ({idx}/{len(claims)}) {claim}")
            try:
                res = self.check_claim_consistency(claim)
                res["claim"] = claim
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to check claim: {exc}")
                res = {
                    "claim": claim,
                    "consistency": "contradict",
                    "confidence": 0.0,
                    "reasoning": f"Error during check: {exc}",
                    "key_evidence": "",
                }
            results.append(res)

        high_conf_contradictions = [
            r for r in results if r.get("consistency") == "contradict" and r.get("confidence", 0) > 0.65
        ]
        final_label = "contradict" if len(high_conf_contradictions) >= 2 else "consistent"
        rationale = self.generate_rationale(results, final_label)

        return {
            "prediction": final_label,
            "rationale": rationale,
            "claim_results": results,
            "character": self.character_name,
            "book": self.book_name,
        }

    def generate_rationale(self, claim_results: List[Dict[str, Any]], final_decision: str) -> str:
        """
        Build a concise rationale based on claim results.
        """
        if final_decision == "contradict":
            strongest = max(
                claim_results,
                key=lambda r: r.get("confidence", 0) if r.get("consistency") == "contradict" else -1,
                default=None,
            )
            if strongest:
                return (
                    f"Found multiple high-confidence contradictions; strongest evidence: "
                    f"{strongest.get('key_evidence') or strongest.get('reasoning')}"
                )
            return "Multiple claims conflict with the novel text."
        else:
            supportive = next((r for r in claim_results if r.get("consistency") == "consistent"), None)
            if supportive:
                return (
                    f"Backstory aligns with character actions; evidence: "
                    f"{supportive.get('key_evidence') or supportive.get('reasoning')}"
                )
            return "Backstory is generally aligned with the novel."


if __name__ == "__main__":
    try:
        # Minimal smoke test using the first available book
        from pathlib import Path

        books_dir = Path(__file__).resolve().parent.parent / "books"
        book_files = sorted(books_dir.glob("*.txt"))
        if not book_files:
            raise FileNotFoundError("No books found for testing.")
        sample_path = book_files[0]
        sample_text = sample_path.read_text(encoding="utf-8")
        sample_backstory = (
            "The character grew up near the coast, learned navigation early, "
            "lost a sibling in a storm, and now avoids sea voyages."
        )

        checker = ConsistencyChecker(
            novel_text=sample_text,
            backstory_content=sample_backstory,
            book_name=sample_path.stem,
            character_name="Sample Character",
        )
        result = checker.make_final_decision()
        print(json.dumps(result, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}")

