"""Reproducible scholarship retrieval benchmark (standard library only)."""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from scripts.check_corpus import CORPUS, read_document
from src import (
    ChunkingStrategyComparator,
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    KnowledgeBaseAgent,
    RecursiveChunker,
    SentenceChunker,
)
from src.chunking import compute_similarity


ROOT = Path(__file__).resolve().parent
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def features(text: str) -> list[str]:
    words = TOKEN_RE.findall(text.casefold())
    return words + [f"{left}_{right}" for left, right in zip(words, words[1:])]


class TfidfEmbedder:
    """Fit one vocabulary on whole source documents for every strategy."""

    def __init__(self, texts: list[str]) -> None:
        document_frequency = Counter()
        for text in texts:
            document_frequency.update(set(features(text)))
        self.vocabulary = {term: index for index, term in enumerate(sorted(document_frequency))}
        count = len(texts)
        self.idf = {
            term: math.log((count + 1) / (frequency + 1)) + 1
            for term, frequency in document_frequency.items()
        }

    def __call__(self, text: str) -> list[float]:
        vector = [0.0] * len(self.vocabulary)
        for term, count in Counter(features(text)).items():
            index = self.vocabulary.get(term)
            if index is not None:
                vector[index] = (1 + math.log(count)) * self.idf[term]
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class HeadingChunker:
    """Preserve the heading in every child when a section is too long."""

    def __init__(self, chunk_size: int = 320) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
        chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= self.chunk_size:
                chunks.append(section)
                continue
            heading, newline, body = section.partition("\n")
            if not newline or not heading.startswith("#"):
                chunks.extend(RecursiveChunker(chunk_size=self.chunk_size).chunk(section))
                continue
            body_size = max(1, self.chunk_size - len(heading) - 1)
            for part in RecursiveChunker(chunk_size=body_size).chunk(body):
                chunks.append(f"{heading}\n{part}")
        return chunks


class ParagraphChunker:
    """Group neighboring Markdown paragraphs without breaking their boundaries."""

    def __init__(self, chunk_size: int = 360) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        chunks = []
        current = ""
        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(RecursiveChunker(chunk_size=self.chunk_size).chunk(paragraph))
                continue
            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if len(candidate) > self.chunk_size:
                chunks.append(current)
                current = paragraph
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks


@dataclass(frozen=True)
class Query:
    question: str
    gold_doc_id: str
    markers: tuple[str, ...]
    metadata_filter: dict[str, str] | None = None
    institution_scope: str | None = None


QUERIES = [
    Query(
        "Học bổng President’s Excellence của VinUni chi trả những gì?",
        "undergraduate-scholarships", ("chi phí sinh hoạt",),
    ),
    Query(
        "Sinh viên VinUni cần GPA tối thiểu bao nhiêu để duy trì học bổng 100%?",
        "scholarship-renewal-policy", ("3,2",),
    ),
    Query(
        "Ở UET, học bổng loại Giỏi cho khóa QH-2023 đến QH-2025 là bao nhiêu mỗi tháng?",
        "uet-merit-scholarship-2025-2026", ("3.500.000đ/tháng",),
    ),
    Query(
        "Sinh viên RMIT Việt Nam đang học cần bao nhiêu tín chỉ và GPA để xin học bổng thành tích 2026?",
        "rmit-current-student-scholarship-2026", ("96 tín chỉ", "3,4/4,0"),
    ),
    Query(
        "Ở UEH, mức hỗ trợ tài chính tối đa cho một học kỳ là bao nhiêu?",
        "ueh-learning-support-scholarship", ("100% học phí trung bình của 15 tín chỉ",),
        {"audience": "student"}, "ueh",
    ),
]


STRATEGIES = {
    "fixed_220": FixedSizeChunker(chunk_size=220, overlap=30),
    "sentence_2": SentenceChunker(max_sentences_per_chunk=2),
    "recursive_280": RecursiveChunker(chunk_size=280),
    "heading_320": HeadingChunker(chunk_size=320),
    "paragraph_360": ParagraphChunker(chunk_size=360),
}


SIMILARITY_PAIRS = [
    ("Học bổng toàn phần chi trả học phí.", "Học bổng 100% hỗ trợ học phí."),
    ("Sinh viên cần GPA để duy trì học bổng.", "Điểm trung bình tối thiểu để giữ suất tài trợ là gì?"),
    ("UEH cấp học bổng cho sinh viên.", "UEH hỗ trợ tài chính giảng viên."),
    ("RMIT yêu cầu 96 tín chỉ và GPA 3,4.", "UET cấp học bổng loại Giỏi 3.500.000đ/tháng."),
    ("Học bổng hỗ trợ học tập UEH.", "Học bổng khuyến khích học tập UET."),
]


def evidence_rank(results: list[dict], query: Query) -> int | None:
    for rank, result in enumerate(results, 1):
        content = result["content"].casefold()
        if result["metadata"].get("doc_id") == query.gold_doc_id and all(
            marker.casefold() in content for marker in query.markers
        ):
            return rank
    return None


def extractive_answer(prompt: str, embedder: TfidfEmbedder) -> str:
    """A transparent demo answerer: select one line from retrieved context."""
    question = prompt.rsplit("Question: ", 1)[-1].split("\nAnswer:", 1)[0]
    query_vector = embedder(question)
    best_score = -1.0
    best_line = "Không tìm thấy thông tin trong ngữ cảnh."
    best_citation = ""
    matches = re.finditer(
        r"\[(\d+)\] Source: [^\n]*\n(.*?)(?=\n\n\[\d+\] Source:|\n\nQuestion:)",
        prompt, flags=re.DOTALL,
    )
    for match in matches:
        for line in match.group(2).splitlines():
            line = line.strip().strip("| ")
            if not line or line.startswith("#") or line.startswith("| ---"):
                continue
            line_vector = embedder(line)
            score = sum(left * right for left, right in zip(query_vector, line_vector))
            if re.search(r"\d", line) and any(word in question.casefold() for word in ("bao nhiêu", "mức", "gpa")):
                score += 0.03
            if score > best_score:
                best_score = score
                best_line = line
                best_citation = f" [{match.group(1)}]"
    return best_line + best_citation


class FilteredStore:
    def __init__(self, store: EmbeddingStore, metadata_filter: dict[str, str]) -> None:
        self.store = store
        self.metadata_filter = metadata_filter

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        return self.store.search_with_filter(query, top_k, self.metadata_filter)


def run() -> str:
    source_documents = [read_document(path) for path in sorted(CORPUS.glob("*.md"))]
    embedder = TfidfEmbedder([body for _, body in source_documents])
    lines = ["BENCHMARK HOC BONG - 2026-09-19", "Backend: TF-IDF tu vung co dinh tren 7 van ban; khong dung API/LLM."]

    lines.append("\nBASELINE chunk_size=200, front matter da loai bo")
    baseline_names = ("undergraduate-scholarships", "scholarship-renewal-policy", "uet-merit-scholarship-2025-2026")
    for doc_id in baseline_names:
        body = next(body for metadata, body in source_documents if metadata["doc_id"] == doc_id)
        comparison = ChunkingStrategyComparator().compare(body, chunk_size=200)
        for strategy, stats in comparison.items():
            lines.append(f"{doc_id} | {strategy} | count={stats['count']} | avg_length={stats['avg_length']:.1f}")

    lines.append("\nSIMILARITY PAIRS: TF-IDF + compute_similarity, high >= 0.20")
    for index, (left, right) in enumerate(SIMILARITY_PAIRS, 1):
        lines.append(f"P{index}: {compute_similarity(embedder(left), embedder(right)):.4f} | {left} | {right}")

    for strategy_name, chunker in STRATEGIES.items():
        store = EmbeddingStore(collection_name=strategy_name, embedding_fn=embedder)
        documents = []
        for metadata, body in source_documents:
            for index, chunk in enumerate(chunker.chunk(body)):
                documents.append(Document(
                    id=f"{metadata['doc_id']}#{index}",
                    content=chunk,
                    metadata={**metadata, "doc_id": metadata["doc_id"]},
                ))
        store.add_documents(documents)
        institution_stores = {}
        for institution in {query.institution_scope for query in QUERIES if query.institution_scope}:
            scoped_store = EmbeddingStore(collection_name=f"{strategy_name}_{institution}", embedding_fn=embedder)
            scoped_store.add_documents([doc for doc in documents if doc.metadata.get("institution") == institution])
            institution_stores[institution] = scoped_store
        lines.append(f"\nSTRATEGY {strategy_name} | chunks={store.get_collection_size()} | avg_length={sum(len(d.content) for d in documents) / len(documents):.1f}")
        total = 0
        for number, query in enumerate(QUERIES, 1):
            query_store = institution_stores[query.institution_scope] if query.institution_scope else store
            results = query_store.search_with_filter(query.question, 3, query.metadata_filter)
            rank = evidence_rank(results, query)
            agent_store = FilteredStore(query_store, query.metadata_filter) if query.metadata_filter else query_store
            answer = KnowledgeBaseAgent(agent_store, lambda prompt: extractive_answer(prompt, embedder)).answer(query.question)
            answer_has_marker = all(marker.casefold() in answer.casefold() for marker in query.markers)
            score = 2 if rank == 1 and answer_has_marker else 1 if rank is not None else 0
            total += score
            lines.append(f"Q{number}: evidence_rank={rank or '-'} score={score} answer_has_marker={answer_has_marker} scope={query.institution_scope or '-'} filter={query.metadata_filter or '-'}")
            lines.append(f"  answer: {answer}")
            for position, result in enumerate(results, 1):
                snippet = result["content"].replace("\n", " ")[:100]
                lines.append(f"  top{position}: {result['metadata']['doc_id']}#{result['id'].rsplit('#', 1)[-1]} score={result['score']:.3f} | {snippet}")
            if query.metadata_filter:
                unfiltered = query_store.search(query.question, top_k=3)
                lines.append("  AB institution-only: " + ", ".join(
                    f"{result['metadata']['doc_id']}({result['metadata']['audience']})" for result in unfiltered
                ))
                lines.append("  AB filtered: " + ", ".join(
                    f"{result['metadata']['doc_id']}({result['metadata']['audience']})" for result in results
                ))
        lines.append(f"TOTAL rubric score={total}/10")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    output = run()
    (ROOT / "ket_qua_benchmark.txt").write_text(output, encoding="utf-8")
    print(output, end="")
