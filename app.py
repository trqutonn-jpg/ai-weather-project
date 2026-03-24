import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# Load data
data = pd.read_csv("weather.csv")

# Features
X = data[["temperature", "humidity", "wind"]]

# Labels
y_temp = data["temperature"]
y_rain = data["rain"]

# Train
X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2)
_, _, y_rain_train, y_rain_test = train_test_split(X, y_rain, test_size=0.2)

model_temp = LinearRegression()
model_rain = DecisionTreeClassifier()

model_temp.fit(X_train, y_temp_train)
model_rain.fit(X_train, y_rain_train)

# UI
st.title("🌦️ AI Dự Báo Thời Tiết")

temp = st.slider("Nhiệt độ hiện tại", 20, 40, 30)
humidity = st.slider("Độ ẩm", 40, 100, 70)
wind = st.slider("Gió", 0, 15, 5)

if st.button("Dự đoán"):
    input_data = [[temp, humidity, wind]]

    pred_temp = model_temp.predict(input_data)[0]
    pred_rain = model_rain.predict(input_data)[0]

    st.write(f"Nhiệt độ ngày mai: {pred_temp:.2f}°C")

    if pred_rain == 1:
        st.write("🌧️ Có mưa")
    else:
        st.write("☀️ Không mưa")