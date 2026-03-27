"""
Lab NLP: Tự động gán nhãn & phân tách từ cho bình luận Facebook tiếng Việt
===========================================================================
File bài tập — Sinh viên hoàn thành các phần TODO bên dưới.

Hướng dẫn:
- Tìm tất cả dòng có "# TODO" và viết code thay thế
- Chạy app bằng: streamlit run app_auto_label_todo.py
- Tham khảo tài liệu underthesea: https://github.com/undertheseanlp/underthesea
"""

import re
import streamlit as st
import pandas as pd
from underthesea import word_tokenize, sentiment


# ============================================================================
# HÀM PHÁT HIỆN SPAM
# ============================================================================

spam_keywords = ["liên hệ", "inbox", "dm", "giá rẻ", "miễn phí", "zalo"]
spam_pattern = re.compile("|".join(map(re.escape, spam_keywords)))  

def detect_spam(text: str) -> bool:
    """
    Phát hiện comment spam dựa trên các đặc điểm phổ biến.
    Trả về True nếu là spam, False nếu không.

    Gợi ý kiểm tra:
    - Chứa link (http, www, .com, .vn, bit.ly ...)
    - Chứa số điện thoại (chuỗi 10-11 chữ số)
    - Chứa từ khóa quảng cáo (inbox, giá rẻ, miễn phí, zalo ...)
    - Lặp ký tự bất thường (aaaaaaa, !!!!!!)
    """
    t = text.lower()

    if re.search(r"https?://|www\.|\.com|\.vn|\.net|bit\.ly", t):
        return True

    if re.search(r"(\d[\d\.\-]{8,}\d)", t):
        return True

    if spam_pattern.search(t):
        return True

    if re.search(r"(.)\1{5,}", t):
        return True

    return False


# ============================================================================
# GIAO DIỆN STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="Lab NLP - Tự động gán nhãn",
    layout="wide"
)

st.title("Lab NLP: Tự động gán nhãn & phân tách từ cho bình luận Facebook tiếng Việt")

st.markdown(
    "Upload file CSV (cột `id`, `text`) — ứng dụng sẽ dùng **underthesea** để:\n"
    "- **Phân tách từ** (word segmentation)\n"
    "- **Gán nhãn cảm xúc** tự động (positive / negative / neutral)"
)

uploaded_file = st.file_uploader("Chọn file CSV", type="csv")  # <-- thay dòng này

if uploaded_file is None:
    st.info("Vui lòng upload file CSV để bắt đầu.")
    st.stop()

df = pd.read_csv(uploaded_file)

if not {"id", "text"}.issubset(df.columns):
    st.error("File CSV phải chứa cột 'id' và 'text'. Vui lòng kiểm tra lại.")
    st.stop()
st.success(f"Đã load {len(df)} dòng. Đang xử lý...")

progress = st.progress(0)

tokenized_list = []
sentiment_list = []

for i, row in df.iterrows():
    text = str(row["text"])

    # TODO 9: Dùng word_tokenize để phân tách từ (format="text")
    tokens = word_tokenize(text, format="text")  # <-- thay dòng này
    tokenized_list.append(tokens)

    # TODO 10: Dùng hàm sentiment() để gán nhãn cảm xúc
    # label = sentiment(...)
    label = sentiment(tokens)  # <-- thay dòng này
    sentiment_list.append(label)

    progress.progress((i + 1) / len(df))


# ============================================================================
# GÁN KẾT QUẢ VÀO DATAFRAME
# ============================================================================
df["tokenized"] = tokenized_list
df["sentiment_label"] = sentiment_list

# TODO 11: Dùng hàm detect_spam để tạo cột "spam" (True/False)
# df["spam"] = df["text"].apply(lambda x: detect_spam(str(x)))
df["spam"] = df["text"].apply(lambda x: detect_spam(str(x)))  # <-- thay dòng này

# TODO 12: Tạo cột spam_label ("spam" / "không spam") từ cột spam
# Gợi ý: df["spam_label"] = df["spam"].map({True: "spam", False: "không spam"})
df["spam_label"] = df["spam"].map({True: "spam", False: "không spam"})  # <-- thay dòng này

# TODO 13: Tạo cột spam_label_vn ("Spam" / "Không spam") từ cột spam
df["spam_label_vn"] = df["spam_label"].map({"spam": "Spam", "không spam": "Không spam"})  # <-- thay dòng này

# TODO 14: Tạo dict map sentiment tiếng Anh -> tiếng Việt và tạo cột sentiment_label_vn
sentiment_vn_map = {"positive": "Tích cực", "negative": "Tiêu cực", "neutral": "Trung lập"}
df["sentiment_label_vn"] = df["sentiment_label"].map(sentiment_vn_map).fillna("Trung lập")

progress.empty()
st.success("Hoàn tất xử lý!")


# ============================================================================
# HIỂN THỊ THỐNG KÊ & KẾT QUẢ
# ============================================================================

# TODO 15: Hiển thị biểu đồ thống kê cảm xúc và spam
# Gợi ý: dùng st.subheader, st.columns, st.bar_chart
col1, col2 = st.columns(2)
with col1: st.bar_chart(df["sentiment_label_vn"].value_counts())
with col2: st.bar_chart(df["spam_label_vn"].value_counts())

# Hiển thị bảng kết quả
st.subheader("Kết quả chi tiết")
display_cols = ["id", "text", "tokenized", "spam_label", "spam_label_vn", "sentiment_label", "sentiment_label_vn"]
st.dataframe(df[display_cols], use_container_width=True)

# TODO 16: Xuất CSV và tạo nút download
# Gợi ý:
csv_data = df[display_cols].to_csv(index=False, encoding="utf-8-sig")
st.download_button(label="⬇ Tải về", data=csv_data, file_name="auto_labels_output.csv", mime="text/csv")
