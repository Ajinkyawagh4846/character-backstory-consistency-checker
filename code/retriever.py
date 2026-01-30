"""
Retriever utilities for querying long-form novel text using Google Gemini embeddings
and a local ChromaDB vector store.
"""

import os
import re
import time
from pathlib import Path
from typing import List, Tuple

import chromadb
import google.generativeai as genai
from chromadb.api.types import Documents, Embeddings, Metadatas
from tqdm import tqdm

# Configure Gemini from environment
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


class NovelRetriever:
    """
    Retrieves relevant passages from a novel using Gemini embeddings + ChromaDB.
    """

    def __init__(self, novel_text: str, book_name: str):
        if not novel_text or not novel_text.strip():
            raise ValueError("novel_text is empty; cannot index.")

        self.novel_text = novel_text
        self.book_name = book_name.strip() if book_name else "unknown_book"
        self.collection_name = self._sanitize_collection_name(self.book_name)

        print(f"[INIT] Using collection name: {self.collection_name}")

        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(self.collection_name)

        self.chunks: List[str] = self.chunk_novel(self.novel_text)
        print(f"[INIT] Created {len(self.chunks)} chunk(s) from novel text.")

        self.index_chunks()

    @staticmethod
    def _sanitize_collection_name(name: str) -> str:
        """Sanitize collection name to be filesystem and Chroma safe."""
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower())
        return cleaned or "novel_collection"

    def chunk_novel(self, text: str, chunk_size: int = 3000, overlap: int = 500) -> List[str]:
        """
        Split text into overlapping word-based chunks for better retrieval context.
        """
        words = text.split()
        if not words:
            raise ValueError("Text contains no words to chunk.")

        chunks: List[str] = []
        step = max(1, chunk_size - overlap)
        idx = 0
        while idx < len(words):
            chunk_words = words[idx : idx + chunk_size]
            chunks.append(" ".join(chunk_words))
            idx += step
        return chunks

    def _embed_with_retry(self, content: str, task_type: str, max_retries: int = 5) -> List[float]:
        """
        Embed content with retry logic to handle rate limits or transient failures.
        """
        delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=content,
                    task_type=task_type,
                )
                embedding = response.get("embedding") or response.get("embeddings") or response
                if embedding is None:
                    raise ValueError("No embedding returned from Gemini API.")
                return embedding
            except Exception as exc:  # noqa: BLE001
                print(f"[RETRY] Attempt {attempt}/{max_retries} failed: {exc}")
                if attempt == max_retries:
                    raise
                time.sleep(delay)
                delay *= 2

    def index_chunks(self) -> None:
        """Generate embeddings for each chunk and store them in ChromaDB."""
        if not self.chunks:
            raise ValueError("No chunks to index.")

        documents: Documents = []
        metadatas: Metadatas = []
        ids: List[str] = []
        embeddings: Embeddings = []

        print("[INDEX] Embedding and indexing chunks...")
        for position, chunk in enumerate(tqdm(self.chunks, desc="Indexing chunks")):
            chunk_id = f"{self.collection_name}_{position}"
            try:
                embedding = self._embed_with_retry(
                    content=chunk, task_type="retrieval_document"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[INDEX][WARN] Skipping chunk {position} due to embedding error: {exc}")
                continue

            documents.append(chunk)
            metadatas.append(
                {
                    "chunk_id": chunk_id,
                    "position": position,
                    "length": len(chunk),
                    "book_name": self.book_name,
                }
            )
            ids.append(chunk_id)
            embeddings.append(embedding)

        if not documents:
            raise RuntimeError("Failed to index any chunks due to embedding errors.")

        # Remove existing collection to avoid duplicates, then re-create
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(self.collection_name)

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )
        print(f"[INDEX] Indexed {len(documents)} chunk(s) into collection '{self.collection_name}'.")

    def retrieve_relevant_passages(
        self, query: str, top_k: int = 7
    ) -> List[Tuple[str, float, dict]]:
        """
        Retrieve the most relevant passages for a given query.
        Returns a list of tuples: (passage_text, distance_score, metadata).
        """
        if not query or not query.strip():
            raise ValueError("Query is empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        print(f"[RETRIEVE] Querying for: {query}")
        embedding = self._embed_with_retry(
            content=query, task_type="retrieval_query"
        )

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )

        if not results or not results.get("documents"):
            print("[RETRIEVE] No results found.")
            return []

        documents = results["documents"][0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        passages = []
        for doc, dist, meta in zip(documents, distances, metadatas):
            passages.append((doc, float(dist), meta))
        print(f"[RETRIEVE] Retrieved {len(passages)} passage(s).")
        return passages


def _load_sample_book() -> Tuple[str, str]:
    """Load the first .txt book from the ../books directory for testing."""
    books_dir = Path(__file__).resolve().parent.parent / "books"
    txt_files = sorted(books_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {books_dir}")
    sample_path = txt_files[0]
    print(f"[TEST] Loading sample book: {sample_path.name}")
    text = sample_path.read_text(encoding="utf-8")
    return text, sample_path.stem


if __name__ == "__main__":
    try:
        book_text, book_name = _load_sample_book()
        retriever = NovelRetriever(book_text, book_name)
        sample_query = "Describe the main character's motivations."
        results = retriever.retrieve_relevant_passages(sample_query, top_k=3)
        for i, (passage, score, meta) in enumerate(results, start=1):
            print(f"\n[RESULT {i}] Score: {score:.4f}")
            print(f"Metadata: {meta}")
            print(f"Passage:\n{passage[:500]}{'...' if len(passage) > 500 else ''}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}")

