import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# ====== CẤU HÌNH WEB ======
st.set_page_config(
    page_title="Weather AI",
    page_icon="🌦️",
    layout="wide"
)

# ====== STYLE (CSS) ======
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        font-size:40px;
        font-weight: bold;
        color: #2c3e50;
    }
    .card {
        padding:20px;
        border-radius:10px;
        background-color:white;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ====== HEADER ======
st.markdown('<p class="title">🌤️ Weather AI Prediction</p>', unsafe_allow_html=True)
st.write("Dự đoán thời tiết bằng AI (Machine Learning)")

# ====== LOAD DATA ======
data = pd.read_csv("weather.csv")

X = data[["temperature", "humidity", "wind"]]
y_temp = data["temperature"]
y_rain = data["rain"]

X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2)
_, _, y_rain_train, y_rain_test = train_test_split(X, y_rain, test_size=0.2)

model_temp = LinearRegression()
model_rain = DecisionTreeClassifier()

model_temp.fit(X_train, y_temp_train)
model_rain.fit(X_train, y_rain_train)

# ====== LAYOUT 2 CỘT ======
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📥 Nhập dữ liệu")

    temp = st.slider("🌡️ Nhiệt độ", 20, 40, 30)
    humidity = st.slider("💧 Độ ẩm", 40, 100, 70)
    wind = st.slider("🌬️ Gió", 0, 15, 5)

    predict_btn = st.button("🚀 Dự đoán")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Kết quả")

    if predict_btn:
        input_data = [[temp, humidity, wind]]

        pred_temp = model_temp.predict(input_data)[0]
        pred_rain = model_rain.predict(input_data)[0]

        st.success(f"🌡️ Nhiệt độ dự đoán: {pred_temp:.2f}°C")

        if pred_rain == 1:
            st.error("🌧️ Có mưa")
        else:
            st.info("☀️ Không mưa")

    else:
        st.write("👉 Nhập dữ liệu và bấm dự đoán")

    st.markdown('</div>', unsafe_allow_html=True)

# ====== BIỂU ĐỒ ======
st.subheader("📈 Dữ liệu thời tiết")
st.line_chart(data)
st.markdown(
    """
    <style>
    body {
        background-image: url("https://images.unsplash.com/photo-1502082553048-f009c37129b9");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)