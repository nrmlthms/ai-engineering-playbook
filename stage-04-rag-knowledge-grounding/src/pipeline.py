"""
End-to-end RAG pipeline.

The grounding problem
──────────────────────
Retrieval gives you relevant text; grounding is the art of making the model USE that
text instead of its parametric memory. Three approaches — pick based on your risk tolerance:

  Strict (for factual Q&A / compliance):
    "Answer only using the provided context. If the answer is not in the context,
     say you don't know."
    + Minimises hallucination
    − Frustrates users when context is incomplete

  Augmented (for assistant tasks):
    "Use the provided context to answer. You may draw on general knowledge,
     but note when you do."
    + More helpful; graceful degradation
    − Harder to audit — model blends context with training knowledge

  Cited (for research / auditable pipelines):
    "Answer using the context. Cite the source number [N] for every claim."
    + Transparent, auditable
    − Verbosity; model must track source attribution per sentence

build_rag_prompt() is your contribution — Exercise 1.
"""

from dataclasses import dataclass, field
from typing import Any

from store import SearchResult


@dataclass
class Document:
    """A source document to be chunked and ingested into the vector store."""

    text: str
    doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    answer: str
    sources: list[SearchResult]
    query: str


def format_context_block(result: SearchResult, index: int) -> str:
    """
    Format one retrieved chunk as a labelled context block.
    Used by build_rag_prompt() to assemble the context section.
    """
    header = f"[{index}] Source: {result.doc_id}" if result.doc_id else f"[{index}]"
    return f"{header}\n{result.text}"


def build_rag_prompt(
    query: str,
    results: list[SearchResult],
    max_context_chars: int = 8_000,
) -> tuple[str, str]:
    """
    Build (system_prompt, user_message) for a retrieval-grounded query.

    TODO: Implement this — it is Exercise 1 of Stage 04.

    This function shapes the grounding contract: how faithful is the model to the
    retrieved content, and how does it handle gaps in that content?

    Approach A — Strict (recommended for factual Q&A):
      system = "Answer only using the provided context. If the answer is not in
                the context, say 'I don't know.'"
      user   = "<context>\\n{chunks}\\n</context>\\n\\nQuestion: {query}"

    Approach B — Augmented (recommended for assistant tasks):
      system = "Use the provided context as your primary source. You may supplement
                with general knowledge but clearly indicate when you do."
      user   = same structure as A

    Approach C — Cited (recommended for research / compliance):
      system = "Answer using the provided context. Cite sources with [N] notation
                matching the context block numbers."
      user   = same structure as A

    Implementation hints:
      - Use format_context_block(result, i+1) to format each chunk.
      - Truncate the assembled context to max_context_chars before adding it to the user message.
      - Place the question AFTER the context ("context then question" improves grounding).

    Returns:
        (system_prompt, user_message) — pass directly to client.complete(system=, messages=).
    """
    raise NotImplementedError(
        "Implement build_rag_prompt() in pipeline.py — see the docstring for the three approaches."
    )


class RAGPipeline:
    """
    End-to-end RAG: ingest documents → answer questions.

    Usage:
        pipeline = RAGPipeline(retriever=retriever, client=anthropic_client)
        pipeline.ingest([Document(text="...", doc_id="readme.md")])
        response = await pipeline.query("What does Stage 04 cover?")
        print(response.answer)
        print([r.doc_id for r in response.sources])
    """

    def __init__(
        self,
        retriever: Any,  # Retriever from retriever.py
        client: Any,  # AnthropicClient or OpenAIClient from Stage 02
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.client = client
        self.top_k = top_k

    def ingest(self, documents: list[Document], chunk_size: int = 512) -> int:
        """
        Chunk, embed, and store all documents.
        Returns the total number of chunks added to the vector store.
        """
        from chunker import chunk_by_tokens_sliding

        total = 0
        for doc in documents:
            chunks = chunk_by_tokens_sliding(
                doc.text, chunk_size=chunk_size, overlap=64, doc_id=doc.doc_id
            )
            if not chunks:
                continue

            embeddings = self.retriever.embedder.embed([c.text for c in chunks])
            ids = [f"{doc.doc_id}::{c.index}" for c in chunks]
            metadatas = [{**doc.metadata, "chunk_index": c.index} for c in chunks]

            self.retriever.store.add(
                ids=ids,
                embeddings=embeddings,
                texts=[c.text for c in chunks],
                metadatas=metadatas,
            )
            total += len(chunks)

        return total

    async def query(self, question: str, system: str = "") -> RAGResponse:
        """
        Retrieve relevant chunks, build a grounding prompt, and generate an answer.
        Requires build_rag_prompt() to be implemented (Exercise 1).
        """
        results = self.retriever.retrieve(question, top_k=self.top_k)
        grounding_system, user = build_rag_prompt(question, results)
        combined_system = f"{system}\n\n{grounding_system}".strip() if system else grounding_system

        response = await self.client.complete(
            messages=[{"role": "user", "content": user}],
            system=combined_system,
        )

        return RAGResponse(answer=response.content, sources=results, query=question)
