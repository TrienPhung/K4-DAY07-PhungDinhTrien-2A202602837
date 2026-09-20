# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Phùng Đình Triển
**Nhóm:** TEN-NHU-CU 
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector embedding có hướng gần nhau (góc giữa chúng nhỏ, giá trị tiến về 1), tức là hai đoạn văn có ý nghĩa hoặc chủ đề gần nhau, dù có thể dùng từ ngữ khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: Tôi muốn trả hàng và hoàn tiền cho đơn hàng này.
- Câu B: Cho mình gửi yêu cầu hoàn tiền cho đơn hàng vừa nhận.
- Tại sao tương đồng: Cùng một ý định (yêu cầu trả hàng/hoàn tiền) trong cùng ngữ cảnh mua sắm, chỉ khác cách diễn đạt.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Chính sách đổi trả hàng của Shopee.
- Câu B: Công thức nấu phở bò truyền thống.
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn không liên quan (thương mại điện tử và nấu ăn), không chia sẻ ngữ cảnh.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine chỉ đo hướng của vector (tức là nghĩa), không phụ thuộc độ dài vector, nên không bị lệch bởi độ dài văn bản hay cách chuẩn hóa. Khoảng cách Euclid bị ảnh hưởng bởi độ lớn, nên hai đoạn cùng chủ đề nhưng khác độ dài có thể bị coi là xa nhau.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Phép tính:* bước nhảy giữa hai chunk liên tiếp = chunk_size − overlap = 500 − 50 = 450. Số chunk = ceil((10000 − 50) / 450) = ceil(22,11) = 23.
> *Đáp án:* **23 chunks**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Bước nhảy giảm còn 400 nên số chunk tăng lên ceil((10000 − 100) / 400) = ceil(24,75) = 25 chunks. Overlap lớn hơn giúp thông tin nằm ở ranh giới giữa hai chunk không bị cắt mất ngữ cảnh: một câu hoặc ý bị cắt đôi vẫn xuất hiện trọn vẹn trong ít nhất một chunk.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng `re.split(r"(?<=[.!?])\s+", text.strip())`: regex lookbehind cắt ngay sau dấu `.`, `!`, `?` khi theo sau là khoảng trắng hoặc xuống dòng, nên dấu câu vẫn nằm trong câu và các số như `3.000.000` (dấu chấm không theo sau bởi khoảng trắng) không bị cắt nhầm. Sau đó lọc bỏ câu rỗng, gom mỗi `max_sentences_per_chunk` câu (mặc định 3) thành một chunk bằng `" ".join`. Edge case: văn bản rỗng trả về `[]`, và `max_sentences_per_chunk` nhỏ hơn 1 được nâng lên thành 1.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thử các separator theo thứ tự ưu tiên `["\n\n", "\n", ". ", " ", ""]`. `_split` cắt văn bản theo separator hiện tại, gộp dần các mảnh liền kề vào một buffer chừng nào độ dài còn ≤ `chunk_size`; mảnh nào tự nó vẫn dài hơn `chunk_size` thì được đệ quy với các separator còn lại. Base case: đoạn có độ dài ≤ `chunk_size` được trả về nguyên (rỗng thì trả `[]`), và khi hết separator thì cắt cứng theo `chunk_size`, nên `separators=[]` vẫn chạy được mà không lỗi.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Bản cài đặt chỉ dùng bộ nhớ trong (in-memory, không nhánh ChromaDB). `_make_record` sao chép metadata, gán `metadata["doc_id"] = doc.id`, tạo id dạng `"{doc.id}#{chỉ số tăng dần}"` và tính embedding bằng `embedding_fn` (mặc định `_mock_embed`), rồi `add_documents` thêm record vào `self._store` (hàm này không tự chunk, chunk phải làm trước). `search` gọi `_search_records`: tính tích vô hướng (`_dot`) giữa embedding của câu hỏi và của từng record, sắp xếp giảm dần theo điểm và lấy `top_k`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Lọc trước rồi mới tìm kiếm: nếu có `metadata_filter` thì chỉ giữ các record có `metadata.get(k) == v` với mọi cặp trong filter (nếu filter rỗng thì dùng toàn bộ store), sau đó chạy `_search_records` trên tập đã lọc để `top_k` không bị chiếm bởi record không hợp lệ. `delete_document` dựng lại `self._store` chỉ gồm các record có `doc_id` khác `doc_id` cần xóa, rồi trả về `True` nếu kích thước store giảm, `False` nếu không.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Nếu store rỗng hoặc `search` không trả kết quả thì trả ngay thông báo tiếng Việt mà không gọi LLM. Ngược lại, lấy `top_k` chunk, đánh số từng chunk thành dòng `[i] (nguồn: doc_id) nội dung`, ghép vào prompt gồm chỉ dẫn (chỉ dùng ngữ cảnh, nếu thiếu thông tin thì nói rõ là không tìm thấy, không được bịa, trích dẫn số nguồn như [1], [2]), phần "Ngữ cảnh", câu hỏi, rồi gọi `llm_fn(prompt)` để lấy câu trả lời.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
(.venv) PS E:\K4-DAY07-PhungDinhTrien-2A202602837> pytest tests/ -v
===================================== test session starts =====================================
platform win32 -- Python 3.10.11, pytest-9.1.1, pluggy-1.6.0 -- E:\K4-DAY07-PhungDinhTrien-2A202602837\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: E:\K4-DAY07-PhungDinhTrien-2A202602837
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

====================================== 42 passed in 0.14s ======================================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Điểm thực tế tính bằng `compute_similarity` trên embedding của `MockEmbedder` (backend mặc định của lab). Dự đoán "cao" nghĩa là kỳ vọng điểm gần 1, "thấp" là gần 0 hoặc âm.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Tôi muốn trả hàng và hoàn tiền | Cho tôi yêu cầu hoàn tiền đơn hàng | cao | 0.0010 | Sai |
| 2 | Shopee phạt người bán hàng giả | Bán hàng nhái sẽ bị xử phạt | cao | 0.0223 | Sai |
| 3 | Chính sách đổi trả hàng | Công thức nấu phở bò | thấp | -0.0941 | Đúng |
| 4 | Đơn hàng giao thành công | Đơn hàng giao không thành công | cao | 0.0443 | Sai |
| 5 | Hôm nay trời mưa to | Người bán bị phạt điểm Sao Quả Tạ | thấp | -0.0678 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là cặp 1 và cặp 2: hai câu gần nghĩa hoàn toàn (cùng ý trả hàng/hoàn tiền, cùng ý phạt hàng giả) nhưng điểm chỉ khoảng 0.00 và 0.02, gần như trực giao. Cả 5 cặp đều nằm trong khoảng -0.09 đến 0.04, nên `MockEmbedder` không phân biệt được cặp giống nghĩa với cặp khác chủ đề; hai dự đoán "thấp" đúng ở cặp 3 và 5 chỉ là ngẫu nhiên. Điều này cho thấy embedding chỉ biểu diễn được ý nghĩa khi được huấn luyện để làm việc đó, còn mock băm ký tự chỉ so khớp bề mặt chuỗi, và đó cũng là lý do benchmark ở mục 5 cho 0/5.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Cấu hình chạy: chiến lược `FixedSizeChunker` (chunk_size=300, overlap=30), embedder `MockEmbedder`, dữ liệu `data/shopee-returns` (10 file, 214 chunk). 5 câu hỏi dùng chung với nhóm.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Thời gian tối đa để gửi yêu cầu Trả hàng/Hoàn tiền cho đơn thực phẩm tươi sống | `about-change-of-mind-returns#23`: đoạn nói không hỗ trợ trả hàng với lý do đổi ý | 0.3123 | Không | Chưa chạy với LLM thật. Top-3 không chứa "24 giờ" nên agent không có căn cứ; theo prompt sẽ phải trả lời không tìm thấy. |
| 2 | Đơn hàng trên bao nhiêu tiền thì không áp dụng Đồng kiểm | `general-returns-policy#21`: đoạn về hoàn lại Voucher | 0.3046 | Không | Chưa chạy với LLM thật. Top-3 không chứa "3.000.000" nên agent không có căn cứ; theo prompt sẽ phải trả lời không tìm thấy. |
| 3 | Bị phạt bao nhiêu điểm Sao Quả Tạ khi LSR ≥ 10% và trên 30 đơn | `shopee-returns-policy#37`: đoạn về quyết định xử lý trả hàng của Shopee | 0.3645 | Không | Chưa chạy với LLM thật. Top-3 không có chunk về LSR nên agent không có căn cứ; theo prompt sẽ phải trả lời không tìm thấy. |
| 4 | Chế tài với shop đăng bán hàng giả/hàng nhái | `about-change-of-mind-returns#11`: đoạn điều khoản thay đổi theo quyết định của Shopee | 0.3692 | Không | Chưa chạy với LLM thật. Top-3 không có chunk về chế tài hàng giả nên agent không có căn cứ; theo prompt sẽ phải trả lời không tìm thấy. |
| 5 | Thời hạn khiếu nại quyết định Trả hàng/Hoàn tiền (filter `audience: buyer`) | `shopee-chinh-sach-bao-hanh#11`: đoạn về bảo hành kèm sản phẩm | 0.4296 | Không | Chưa chạy với LLM thật. Top-3 (toàn chunk bảo hành) không có thời hạn khiếu nại nên agent không có căn cứ; theo prompt sẽ phải trả lời không tìm thấy. |

Ghi chú: `main.py` dùng `demo_llm` (LLM giả, chỉ in lại phần đầu prompt) nên tác tử chưa sinh được câu trả lời thật trong môi trường lab; cột "Câu trả lời của Agent" ở trên là đánh giá dựa trên ngữ cảnh top-3 mà agent nhận được.

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 0 / 5

**Nhận xét:** `MockEmbedder` băm ký tự nên không hiểu nghĩa, các chunk lọt top-3 chủ yếu là chunk chung chung (chân trang, danh sách ngành hàng), score sát nhau (0.25 – 0.43). Với Query 3, đáp án (LSR ≥ 10% và ≥ 30 đơn: 2 điểm phạt) có trong `shopee-he-thong-sao-qua-ta` nhưng với `chunk_size=300` dòng này bị tách sang chunk khác với tiêu đề "Tỷ lệ giao hàng trễ (LSR)" và lẫn các dòng NFR có cùng con số, nên không lọt top-3. Với Query 5, filter `buyer` loại luôn file `seller` chứa đáp án "2 ngày" (`shopee-quy-dinh-xu-ly-khieu-nai-nguoi-ban`); khi bỏ filter thì top-1 và top-2 vẫn giữ nguyên, cho thấy nút thắt nằm ở xếp hạng của mock embedding.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Qua bài của Nguyễn Quang Đạo (cùng nhóm), `HeadingChunker` (chunk theo tiêu đề, gắn lại tiêu đề vào từng chunk con) kết hợp filter `audience` cho kết quả tốt nhất nhóm: theo báo cáo nhóm, cả 5 câu đều có chunk liên quan trong top-3, trong khi `FixedSizeChunker` + `MockEmbedder` của tôi chỉ đạt 0/5. Tôi học được rằng chunk nên giữ trọn một mục cùng tiêu đề (ở Query 3, dòng đáp án của tôi bị tách khỏi tiêu đề LSR), và filter `audience` phải khớp đối tượng của câu hỏi (ở Query 5, filter `buyer` loại mất file `seller` chứa đáp án).

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) |10 / 10 |
| Hoàn thiện code (Core Implementation — tests) |30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) |5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) |10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |