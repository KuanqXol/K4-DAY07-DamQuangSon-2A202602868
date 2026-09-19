# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Đàm Quang Sơn
**Nhóm:** G15
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
Hai vector biểu diễn hai đoạn văn bản cùng hướng, thường cho thấy nội dung gần nhau về mặt ngữ nghĩa theo mô hình embedding đang dùng. Điểm gần 1 không tự bảo đảm hai câu đều đúng về mặt thực tế.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên cần GPA bao nhiêu để duy trì học bổng 100%?"
- Câu B: "Điều kiện điểm trung bình để giữ học bổng toàn phần là gì?"
- Tại sao tương đồng: Cả hai cùng hỏi ngưỡng học tập để tiếp tục nhận học bổng toàn phần.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Sinh viên cần GPA bao nhiêu để duy trì học bổng 100%?"
- Câu B: "Cách khởi tạo môi trường ảo Python trên Windows."
- Tại sao khác: Hai câu nói về hai chủ đề và mục đích khác nhau.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
Cosine đo hướng của vector, nên ít bị ảnh hưởng bởi độ lớn vector do độ dài văn bản hoặc cách mô hình tạo embedding. Khoảng cách Euclid phụ thuộc cả hướng lẫn độ lớn, nên hai vector cùng hướng vẫn có thể bị coi là xa.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
`ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23`.

**Đáp án:** 23 chunks.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
`ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25` chunks, tăng 2 chunks. Overlap lớn hơn giúp thông tin nằm sát ranh giới vẫn xuất hiện đầy đủ ở ít nhất một chunk, đổi lại tốn thêm dung lượng và lượt embedding.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
Dùng `(?<=[.!?])\s+` để tách sau dấu kết câu và giữ dấu trong kết quả. Bỏ các phần rỗng, gom tối đa `max_sentences_per_chunk` câu, rồi chuẩn hóa khoảng trắng quanh chunk. Cách đơn giản này có thể tách sai chữ viết tắt như `TS.` và không nhận diện mọi quy ước dấu câu.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
Ưu tiên tách theo đoạn, dòng, dấu chấm và khoảng trắng; mảnh còn quá dài mới chuyển xuống dấu phân cách tiếp theo. Các mảnh liền nhau được gom lại miễn không vượt `chunk_size`. Text rỗng trả `[]`, mảnh đã đủ nhỏ được giữ nguyên, hết separator thì cắt cứng theo kích thước.

**Chiến lược tôi được gán để đánh giá — `paragraph_360`:**
`ParagraphChunker` trong [`bench.py`](../bench.py) tách tại dòng trắng, gom các đoạn liên tiếp đến 360 ký tự và dùng `RecursiveChunker` nếu một đoạn quá dài. Tôi chọn cấu hình này để cố giữ cùng nhau hàng bảng và đoạn điều kiện học bổng; nhược điểm là chunk đầu một tài liệu có thể chỉ chứa tiêu đề, còn bảng dài vẫn bị cắt. Các kết quả bên dưới là một lần chạy từ mã chung của repo, không phải bằng chứng mỗi thành viên đã chạy độc lập.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
Mỗi `Document` được nhúng một lần và lưu thành một record trong bộ nhớ cùng nội dung, id và bản sao metadata; `add_documents` không tự chia chunk. `search` nhúng query, tính tích vô hướng với mọi vector tài liệu, rồi sắp xếp giảm dần và lấy `top_k`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
`search_with_filter` chọn trước các record có metadata khớp toàn bộ điều kiện, rồi mới xếp hạng trên tập đó; nhờ vậy tài liệu không hợp lệ không chiếm chỗ trong top-k. Mỗi record có `doc_id` của tài liệu gốc, nên `delete_document` có thể xóa mọi chunk thuộc cùng tài liệu và trả về liệu có xóa được gì.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
Agent lấy `top_k` kết quả rồi đưa từng chunk vào prompt với số `[1]`, `[2]` và nguồn tài liệu. Prompt yêu cầu chỉ trả lời theo ngữ cảnh, dẫn số chunk và nói rõ khi thiếu thông tin, sau đó chuyển prompt cho `llm_fn`.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
python -m pytest tests/ -q
..........................................                               [100%]
42 passed in 0.09s
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Học bổng toàn phần chi trả học phí. | Học bổng 100% hỗ trợ học phí. | Cao | 0,3339 (cao) | Có |
| 2 | Sinh viên cần GPA để duy trì học bổng. | Điểm trung bình tối thiểu để giữ suất tài trợ là gì? | Cao | 0,1216 (thấp) | Không |
| 3 | UEH cấp học bổng cho sinh viên. | UEH hỗ trợ tài chính giảng viên. | Thấp | 0,1581 (thấp) | Có |
| 4 | RMIT yêu cầu 96 tín chỉ và GPA 3,4. | UET cấp học bổng loại Giỏi 3.500.000đ/tháng. | Thấp | 0,0404 (thấp) | Có |
| 5 | Học bổng hỗ trợ học tập UEH. | Học bổng khuyến khích học tập UET. | Thấp | 0,2476 (cao) | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
Cặp 2 cùng hỏi điều kiện duy trì học bổng nhưng điểm thấp vì TF-IDF không hiểu `GPA` gần nghĩa với `điểm trung bình`, hay `duy trì` với `giữ`. Ngược lại, cặp 5 khác trường và loại học bổng nhưng lặp nhiều từ nên điểm khá cao. Đây là thử nghiệm với `TfidfEmbedder` trong `bench.py` và `compute_similarity()` của `src/chunking.py`, dùng cùng từ vựng của 7 tài liệu; quy ước minh họa `cao ≥ 0,20`, không phải ngưỡng chuẩn cho mọi mô hình. Điểm được ghi trong mục `SIMILARITY PAIRS` của [`ket_qua_benchmark.txt`](../ket_qua_benchmark.txt).

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | President’s Excellence VinUni chi trả gì? | `undergraduate-scholarships#1`: học phí và sinh hoạt | 2/2 | Có, top-1 | Trích đúng toàn bộ học phí và chi phí sinh hoạt. |
| 2 | GPA tối thiểu để giữ học bổng VinUni 100%? | `scholarship-renewal-policy#0`: tiêu đề và giới thiệu | 1/2 | Có, nhưng hàng `3,2` ở top-3 | Agent chọn hàng WIT 5%, trả lời sai ngưỡng. |
| 3 | UET loại Giỏi QH-2023 đến QH-2025 bao nhiêu? | `uet-merit-scholarship-2025-2026#4`: hàng chương trình khác | 1/2 | Có, hàng đúng ở top-2 | Trích hàng `Chuẩn QH-2023 đến QH-2025`, gồm giá trị `3.500.000đ/tháng` ở cột Giỏi. |
| 4 | RMIT cần bao nhiêu tín chỉ và GPA? | `rmit-current-student-scholarship-2026#0`: chỉ tiêu học bổng | 1/2 | Có, điều kiện ở top-3 | Trích đúng `96 tín chỉ` và `3,4/4,0` từ chunk [3]. |
| 5 | UEH hỗ trợ tài chính tối đa một học kỳ? | `ueh-learning-support-scholarship#0`: mức toàn phần | 1/2 | Có, top-1 sau lọc | Agent lại trích điều kiện hoàn cảnh ở chunk [2], thiếu mức `100% học phí trung bình của 15 tín chỉ`. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5. **Điểm theo rubric:** 6 / 10. Q5 giới hạn tập ứng viên ở UEH rồi dùng đúng `metadata_filter={"audience": "student"}`; trong nhánh A/B chỉ bỏ `audience`, tài liệu UEH dành cho giảng viên đứng top-1. Top-3 có bằng chứng không đồng nghĩa agent trả lời đúng.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
Từ so sánh cấu hình trong nhóm, tôi thấy `heading_320` đưa cả điều kiện RMIT lên top-1 nhờ lặp tiêu đề ở chunk con, trong khi `paragraph_360` đưa thông tin đó lên top-3. Tôi sẽ giữ nhãn mục và tiêu đề cột bảng khi chia đoạn để tăng khả năng chọn đúng điều khoản. Chưa có bằng chứng đã dự demo với nhóm khác, nên chưa ghi nhận xét từ buổi đó.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 9 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 6 / 10 |
| **Tổng phần cá nhân, tự đánh giá** | **55 / 60** |

Điểm tự đánh giá dựa trên kết quả chạy được trong repo; giảng viên có thể chấm khác khi xem chất lượng câu trả lời và phần trình bày.
