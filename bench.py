"""
bench.py — Cong cu do luong rieng cua Phung Dinh Trien.
Chien luoc: FixedSizeChunker (chunk_size=300, overlap=30)

Cach chay:
    python bench.py
Ket qua duoc in ra man hinh va ghi vao file ket_qua_benchmark.txt
"""

from __future__ import annotations

import re
from pathlib import Path

from src.chunking import FixedSizeChunker
from src.models import Document
from src.store import EmbeddingStore

DATA_DIR = Path("data/shopee-returns")
CHUNK_SIZE = 300
OVERLAP = 30

# 5 cau hoi CHUNG cua nhom (do R2 - Giap soan), khong tu doi
QUERIES = [
    {
        "question": "Thời gian tối đa để gửi yêu cầu Trả hàng/Hoàn tiền cho đơn hàng thực phẩm tươi sống là bao lâu?",
        "gold_answer": "Trong vòng 24 giờ kể từ lúc đơn hàng được cập nhật trạng thái Giao hàng thành công.",
        "expect_keyword": "24 giờ",
        "filter": None,
    },
    {
        "question": "Đơn hàng có giá trị trên bao nhiêu tiền thì KHÔNG được áp dụng chương trình Đồng kiểm?",
        "gold_answer": "Đơn hàng có giá trị lớn hơn 3.000.000 VND (3 Triệu Đồng).",
        "expect_keyword": "3.000.000",
        "filter": None,
    },
    {
        "question": "Người bán bị phạt bao nhiêu điểm Sao Quả Tạ nếu có tỷ lệ giao hàng trễ (LSR) từ 10% trở lên và trên 30 đơn?",
        "gold_answer": "2 điểm phạt Sao Quả Tạ.",
        "expect_keyword": "LSR",
        "filter": None,
    },
    {
        "question": "Shop đăng bán sản phẩm hàng giả/hàng nhái trên Shopee sẽ phải chịu những chế tài xử lý nào?",
        "gold_answer": "Sản phẩm bị khóa/xóa, cộng điểm phạt Sao Quả Tạ, tạm thời đóng băng hoặc khóa vĩnh viễn tài khoản nếu tái phạm.",
        "expect_keyword": "Hàng Giả",
        "filter": None,
    },
    {
        "question": "Thời hạn khiếu nại quyết định Trả hàng/Hoàn tiền là bao nhiêu ngày?",
        "gold_answer": "Người mua có 15 ngày kể từ ngày giao hàng; Người bán có 2 ngày kể từ khi nhận được thông báo để khiếu nại.",
        "expect_keyword": "khiếu nại",
        "filter": {"audience": "buyer"},
    },
]


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Tach frontmatter YAML don gian va phan noi dung con lai."""
    parts = text.split("---")
    if len(parts) < 3:
        return {}, text
    fm_raw = parts[1]
    body = "---".join(parts[2:]).strip()
    fm = {}
    for key, val in re.findall(r"^(\w+):\s*(.+)$", fm_raw, re.M):
        fm[key] = val.strip().strip('"').strip("'")
    return fm, body


def load_documents() -> list[Document]:
    """Doc tung file .md, chunk bang FixedSizeChunker, tra ve list Document."""
    chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    docs: list[Document] = []

    for path in sorted(DATA_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(raw)
        doc_id = fm.get("doc_id", path.stem)

        chunks = chunker.chunk(body)
        for i, chunk_text in enumerate(chunks):
            docs.append(
                Document(
                    id=f"{doc_id}#{i}",
                    content=chunk_text,
                    metadata={**fm, "doc_id": doc_id},
                )
            )
    return docs


def run_benchmark() -> str:
    lines: list[str] = []
    lines.append("=" * 50)
    lines.append("BENCHMARK RUN: Strategy = FixedSizeChunker")
    lines.append("=" * 50)

    docs = load_documents()
    store = EmbeddingStore(collection_name="bench_fixed_size")
    store.add_documents(docs)

    md_files = list(DATA_DIR.glob("*.md"))
    lines.append(f"Loaded {len(md_files)} files -> Total chunks indexed: {store.get_collection_size()}")
    lines.append("")

    pass_count = 0
    for i, q in enumerate(QUERIES, start=1):
        lines.append(f"--- Query {i}: {q['question']} ---")
        if q["filter"]:
            lines.append(f"  [Filter]: {q['filter']}")
        lines.append(f"  Gold Answer: {q['gold_answer']}")

        if q["filter"]:
            results = store.search_with_filter(q["question"], top_k=3, metadata_filter=q["filter"])
        else:
            results = store.search(q["question"], top_k=3)

        found = False
        for rank, r in enumerate(results, start=1):
            doc_name = r["metadata"].get("doc_id", "unknown")
            preview = r["content"].replace("\n", " ").strip()[:70]
            lines.append(f"  Top-{rank} (Score: {r['score']:.4f}, Doc: {doc_name}): {preview}...")
            if q["expect_keyword"].lower() in r["content"].lower():
                found = True

        status = "PASS (Found relevant context in Top-3)" if found else "MISS"
        if found:
            pass_count += 1
        lines.append(f"  -> Evaluation: {status}")
        lines.append("")

    lines.append(f"Summary for FixedSizeChunker: {pass_count}/{len(QUERIES)} relevant in Top-3")
    return "\n".join(lines)


if __name__ == "__main__":
    output = run_benchmark()
    print(output)
    Path("ket_qua_benchmark.txt").write_text(output, encoding="utf-8")
    print("\n[Da ghi ket qua vao ket_qua_benchmark.txt]")