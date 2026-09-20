from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if self._store.get_collection_size() == 0:
            return "Không có tài liệu nào trong knowledge base để trả lời câu hỏi này."

        results = self._store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin liên quan trong knowledge base."

        context_lines = []
        for i, result in enumerate(results, start=1):
            doc_id = result["metadata"].get("doc_id", "unknown")
            context_lines.append(f"[{i}] (nguồn: {doc_id}) {result['content']}")
        context = "\n\n".join(context_lines)

        prompt = (
            "Bạn là trợ lý trả lời câu hỏi dựa trên ngữ cảnh được cung cấp dưới đây. "
            "Chỉ sử dụng thông tin trong ngữ cảnh để trả lời. "
            "Nếu ngữ cảnh không chứa thông tin cần thiết, hãy nói rõ là không tìm thấy, "
            "không được tự bịa ra câu trả lời. "
            "Khi trả lời, hãy trích dẫn số nguồn tương ứng, ví dụ [1], [2].\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Trả lời:"
        )

        return self._llm_fn(prompt)