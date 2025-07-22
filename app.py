import streamlit as st
import re
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect
from deep_translator import GoogleTranslator
import pandas as pd

# Emotion labels and emojis
emotions = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
emoji_dict = {
    "joy": "😂", "fear": "😱", "anger": "😠",
    "sadness": "😢", "disgust": "😒",
    "shame": "😳", "guilt": "😳"
}

# --- Function Definitions ---
def read_data(file):
    data = []
    for line in file:
        line = line.strip()
        if not line or not line.startswith("["):
            continue
        label = ' '.join(line[1:line.find("]")].strip().split())
        text = line[line.find("]") + 1:].strip()
        data.append([label, text])
    return data

def convert_label(label, emotion_list):
    items = list(map(float, label.split()))
    for idx in range(len(items)):
        if items[idx] == 1:
            return emotion_list[idx]
    return "unknown"

def make_label_vector(emotion):
    return [1 if e == emotion else 0 for e in emotions]

def preprocess_text(texts, labels):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
    X = vectorizer.fit_transform(texts)
    return X, labels, vectorizer

def detect_and_translate(text):
    lang = detect(text)
    if lang != "en":
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except:
            return text
    return text

# --- Streamlit App UI ---
st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("😄 Emotion Detection App")

# Sidebar: Upload Dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your `.txt` dataset", type="txt")

# Read dataset
if uploaded_file:
    lines = uploaded_file.read().decode("utf-8").splitlines()
    data = read_data(lines)
else:
    with open("text.txt", "r", encoding="utf-8") as f:
        data = read_data(f)

# Prepare data
texts = [text for _, text in data]
labels = [convert_label(label, emotions) for label, _ in data]
X, y, vectorizer = preprocess_text(texts, labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# --- 🔍 Predict Emotion ---
st.subheader("🔍 Predict Emotion")
user_input = st.text_input("Enter text in any language:")
if user_input:
    translated = detect_and_translate(user_input)
    vec = vectorizer.transform([translated])
    prediction = model.predict(vec)[0]
    st.markdown(f"**Predicted Emotion:** `{prediction}` {emoji_dict.get(prediction, '')}`")

    # Save to log
    with open("emotion_log.txt", "a", encoding="utf-8") as log:
        log.write(f"{datetime.now()}\t{user_input}\t{prediction}\n")

# --- 📅 Emotion Log ---
st.subheader("📅 Emotion Log")

log_file = "emotion_log.txt"
log_exists = os.path.exists(log_file)

if log_exists:
    with open(log_file, "r", encoding="utf-8") as log:
        log_content = log.read()

    st.text_area("Prediction Log", log_content, height=150)

    if st.button("🗑️ Clear Log"):
        open(log_file, "w").close()
        st.success("Prediction log cleared! Please refresh.")
else:
    st.info("No predictions logged yet.")

# --- 📝 Add New Training Sentence ---
st.subheader("📝 Add New Training Sentence")
new_text = st.text_input("New sentence:")
new_label = st.selectbox("Choose the correct emotion label:", emotions)
if st.button("Add and Retrain"):
    new_label_vector = make_label_vector(new_label)
    line = f"[{' '.join(map(str, new_label_vector))}] {new_text}\n"
    with open("text.txt", "a", encoding="utf-8") as f:
        f.write(line)
    st.success("New sentence added! Please refresh the page to retrain.")

# --- Sidebar: Model Accuracy ---
st.sidebar.markdown("### Model Accuracy")
train_score = accuracy_score(y_train, model.predict(X_train))
test_score = accuracy_score(y_test, model.predict(X_test))
st.sidebar.write(f"Training Accuracy: `{train_score:.2f}`")
st.sidebar.write(f"Test Accuracy: `{test_score:.2f}`")
