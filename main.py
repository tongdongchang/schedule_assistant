import streamlit as st
import sqlite3
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import re

# ==========================
# CẤU HÌNH ỨNG DỤNG
# ==========================
st.set_page_config(
    page_title="Trợ lý phân loại cảm xúc tiếng Việt",
    page_icon="😊",
    layout="centered"
)

# Khởi tạo session state để theo dõi trạng thái
if 'last_analyzed' not in st.session_state:
    st.session_state.last_analyzed = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# ==========================
# KHỞI TẠO MODEL
# ==========================
@st.cache_resource
def load_model():
    try:
        model_name = "mr4/phobert-base-vi-sentiment-analysis"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        return classifier
    except Exception as e:
        st.error(f"Lỗi khi load model: {str(e)}")
        return None


# ==========================
# TIỀN XỬ LÝ TIẾNG VIỆT
# ==========================
def preprocess_text(text):
    """Chuẩn hóa văn bản tiếng Việt"""
    if not text or len(text.strip()) < 3:
        return None

    # Chuyển thành chữ thường
    text = text.lower().strip()

    # Sửa các từ viết tắt/thường gặp
    replacements = {
        ' rat ': ' rất ',
        ' dc ': ' được ',
        ' ko ': ' không ',
        ' k ': ' không ',
        ' nt ': ' như thế ',
        ' ntn ': ' như thế nào ',
        ' bt ': ' bình thường ',
        ' do ': ' dở ',
        ' ng ': ' người ',
        ' hom nay': ' hôm nay',
        ' hom qua': ' hôm qua',
        ' hom sau': ' hôm sau'
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ==========================
# XỬ LÝ DATABASE
# ==========================
def init_db():
    """Khởi tạo database SQLite"""
    conn = sqlite3.connect('sentiment_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def save_to_db(text, sentiment):
    """Lưu kết quả vào database"""
    conn = sqlite3.connect('sentiment_history.db')
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute(
        "INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)",
        (text, sentiment, timestamp)
    )
    conn.commit()
    conn.close()


def get_history():
    """Lấy lịch sử phân loại"""
    conn = sqlite3.connect('sentiment_history.db')
    c = conn.cursor()
    c.execute(
        "SELECT text, sentiment, timestamp FROM sentiments ORDER BY timestamp DESC LIMIT 50"
    )
    results = c.fetchall()
    conn.close()
    return results


# ==========================
# GIAO DIỆN CHÍNH
# ==========================
def main():
    # Khởi tạo database
    init_db()

    # Header
    st.title("😊 Trợ lý phân loại cảm xúc tiếng Việt")
    st.markdown("---")
    
    # Load model với progress bar
    with st.spinner("Đang khởi tạo model..."):
        classifier = load_model()

    if classifier is None:
        st.error("Không thể khởi tạo model. Vui lòng thử lại!")
        return

    # Khu vực nhập liệu
    st.subheader("📝 Nhập câu tiếng Việt cần phân loại")
    
    # Sử dụng session state để giữ giá trị textarea
    user_input = st.text_area(
        "Nhập câu của bạn tại đây:",
        placeholder="Ví dụ: Hôm nay tôi rất vui...",
        height=100,
        key="input_text"  # Thêm key để quản lý widget
    )
    
    # Nút phân loại
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🔍 Phân loại cảm xúc", use_container_width=True, type="primary")

    # Xử lý khi nhấn nút phân loại
    if analyze_btn:
        st.session_state.last_analyzed = user_input
        st.session_state.user_input = user_input
        
        if not user_input or not user_input.strip():
            st.warning("⚠️ Vui lòng nhập câu trước khi phân tích!")
            st.stop()
        elif len(user_input.strip()) < 3:
            st.warning("⚠️ Câu quá ngắn, vui lòng nhập ít nhất 3 ký tự!")
            st.stop()
        else:
            with st.spinner("Đang phân tích cảm xúc..."):
                # Tiền xử lý văn bản
                processed_text = preprocess_text(user_input)

                if not processed_text:
                    st.error("❌ Câu nhập vào không hợp lệ!")
                    return

                try:
                    # Phân loại cảm xúc
                    result = classifier(processed_text)[0]

                    # Ánh xạ nhãn cảm xúc
                    label_map = {
                        'Tích cực': 'TÍCH CỰC 😊',
                        'Tiêu cực': 'TIÊU CỰC 😞',
                        'Trung tính': 'TRUNG TÍNH 😐'
                    }

                    sentiment = result['label']
                    score = result['score']

                    # Hiển thị kết quả
                    st.markdown("---")
                    st.subheader("🎯 Kết quả phân loại")

                    # Hiển thị với màu sắc tương ứng
                    if sentiment == 'Tích cực':
                        st.success(f"**Cảm xúc:** {label_map[sentiment]}")
                    elif sentiment == 'Tiêu cực':
                        st.error(f"**Cảm xúc:** {label_map[sentiment]}")
                    else:
                        st.info(f"**Cảm xúc:** {label_map[sentiment]}")
                    
                    st.write(f"**Độ tin cậy:** {score:.2%}")
                    st.write(f"**Câu đã xử lý:** {processed_text}")
                    st.write(f"**Câu gốc:** {user_input}")

                    # Lưu vào database
                    save_to_db(user_input, sentiment)

                    st.success("✅ Đã lưu kết quả vào lịch sử!")

                except Exception as e:
                    st.error(f"❌ Lỗi khi phân tích: {str(e)}")
    
    # Hiển thị lại kết quả lần trước nếu có và không phải là lần nhấn đầu tiên
    elif st.session_state.last_analyzed and st.session_state.last_analyzed != "":
        st.markdown("---")
        st.info("ℹ️ Kết quả phân tích lần trước vẫn được hiển thị bên dưới.")
    
    # Hiển thị lịch sử
    st.markdown("---")
    st.subheader("📊 Lịch sử phân loại")
    history = get_history()
    if history:
        # Hiển thị dưới dạng bảng
        for i, (text, sentiment, timestamp) in enumerate(history[:10], 1):
            # Định dạng thời gian
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%d/%m/%Y %H:%M")

            # Hiển thị với icon tương ứng
            icons = {'Tích cực': '😊', 'Tiêu cực': '😞', 'Trung tính': '😐'}
            icon = icons.get(sentiment, '❓')

            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{i}. {text}**")
                with col2:
                    st.write(f"{icon} {sentiment} - {time_str}")
                st.markdown("---")
    else:
        st.info("📝 Chưa có lịch sử phân loại nào.")


# ==========================
# CHẠY ỨNG DỤNG
# ==========================
if __name__ == "__main__":
    main()
