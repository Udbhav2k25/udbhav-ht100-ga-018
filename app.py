import streamlit as st
from emotion_model import EmotionModel
from utils import create_emotion_graph
import re
import json
import os
from datetime import datetime
import shutil

# ---------------------------
# Configuration
# ---------------------------
STORAGE_DIR = "chat_files"
os.makedirs(STORAGE_DIR, exist_ok=True)

# ---------------------------
# Sentence Splitter
# ---------------------------
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]

# ---------------------------
# Suggestions
# ---------------------------
SUGGESTIONS = {
    "sadness": "It seems you're feeling low. Try talking to someone you trust, journaling your thoughts, or taking a short walk.",
    "anger": "You might be feeling frustrated. Try taking deep breaths or stepping away for a moment.",
    "fear": "It seems something is worrying you. Try grounding techniques or talking to someone you feel safe with.",
    "joy": "You're feeling good! Keep doing what makes you happy and share your joy.",
    "neutral": "Your emotions seem balanced. Continue your routine and stay mindful.",
    "disgust": "Something seems to be bothering you. Try identifying the cause and distancing yourself from it.",
    "surprise": "Something unexpected happened! Reflect on it — was it positive or challenging?",
}

# ---------------------------
# File Utilities
# ---------------------------
def list_chat_files():
    return sorted([f for f in os.listdir(STORAGE_DIR) if f.endswith(".json")])

def get_file_path(filename):
    return os.path.join(STORAGE_DIR, filename)

def load_chat_file(filename):
    try:
        path = get_file_path(filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        return []
    return []

def save_chat_file(filename, history):
    path = get_file_path(filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def create_new_file(name):
    safe = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip()
    if not safe:
        return None
    fname = safe + ".json"
    path = get_file_path(fname)
    if not os.path.exists(path):
        save_chat_file(fname, [])
    return fname

def delete_file(name):
    path = get_file_path(name)
    if os.path.exists(path):
        os.remove(path)

# ---------------------------
# Model + Streamlit Session
# ---------------------------
model = EmotionModel()

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------
# Layout Config
# ---------------------------
st.set_page_config(page_title="File-based Emotion Chat", layout="wide")

# ---------------------------
# Sidebar (File Manager)
# ---------------------------
with st.sidebar:
    st.header("📁 Chat Files")

    st.session_state.files = list_chat_files()

    # Open File
    if st.session_state.files:
        chosen = st.selectbox("Open file:", ["(Choose)"] + st.session_state.files)
        if chosen != "(Choose)":
            st.session_state.current_file = chosen
            st.session_state.chat_history = load_chat_file(chosen)
            st.success(f"Opened {chosen}")

    # Create File
    st.markdown("---")
    new_name = st.text_input("Create new file:")
    if st.button("➕ Create"):
        created = create_new_file(new_name)
        if created:
            st.session_state.current_file = created
            st.session_state.chat_history = []
            st.success(f"Created and opened {created}")

    st.markdown("---")

    # Delete File
    if st.session_state.current_file:
        if st.button("🗑 Delete active file"):
            delete_file(st.session_state.current_file)
            st.session_state.current_file = None
            st.session_state.chat_history = []
            st.success("File deleted.")

    # Refresh
    if st.button("🔄 Refresh file list"):
        st.session_state.files = list_chat_files()
        st.rerun()

    st.caption("Files stored in ./chat_files/*.json")

# ---------------------------
# Main Page UI
# ---------------------------
st.title("💬 File-Based Emotion Chat Analyzer")

if not st.session_state.current_file:
    st.info("Create or open a file from the sidebar to begin.")
    st.stop()

st.subheader(f"Active file: {st.session_state.current_file}")

# ---------------------------
# Chat Input
# ---------------------------
user_message = st.chat_input("Type your message here...")

if user_message:

    sentences = split_into_sentences(user_message)
    emotions = []
    confidences = []

    for s in sentences:
        label, probs = model.predict_emotion(s)
        emotions.append(label)
        confidences.append(max(probs.values()))

    dominant = max(set(emotions), key=emotions.count)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "paragraph": user_message,
        "sentences": sentences,
        "sentence_emotions": emotions,
        "confidence_scores": confidences,
        "dominant": dominant
    }

    st.session_state.chat_history.append(entry)
    save_chat_file(st.session_state.current_file, st.session_state.chat_history)

# ---------------------------
# Controls
# ---------------------------
col1, col2 = st.columns([1,1])
with col1:
    if st.button("🧹 Clear RAM History (file unchanged)"):
        st.session_state.chat_history = []
        st.success("RAM history cleared!")

with col2:
    show_history = st.toggle("📜 Show Chat History")

# ---------------------------
# Display History
# ---------------------------
if show_history:

    st.markdown("## 🗂 Chat History")

    for chat in st.session_state.chat_history:

        st.markdown(
            f"""
            <div style="background:#F3F4F6;padding:12px;border-radius:8px;margin-top:10px;">
                <b>🕒 {chat['timestamp']}</b><br>
                <b>Message:</b> {chat['paragraph']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🔍 Sentence-by-sentence")

        for i, sentence in enumerate(chat["sentences"]):
            emo = chat["sentence_emotions"][i]
            conf = chat["confidence_scores"][i]
            st.write(f"*Sentence {i+1}:* {sentence}")
            st.write(f"→ Emotion: *{emo}* (Confidence: {conf:.2f})")
            st.write("---")

        st.success(f"🌟 Dominant Emotion: *{chat['dominant'].upper()}*")

        st.info(f"💡 Suggestion: {SUGGESTIONS.get(chat['dominant'], '')}")

        # Graph
        labels = chat["sentence_emotions"]
        if len(labels) == 1:
            labels = labels + labels

        graph_path = "emotion_plot.png"
        create_emotion_graph(labels, graph_path)
        st.image(graph_path)

        st.markdown("---")
