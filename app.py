import streamlit as st
from emotion_model import EmotionModel
from utils import create_emotion_graph

# Initialize model
model = EmotionModel()

st.title(" Chat-Based Emotion Detector")
st.write("Paste your chat conversation below (one message per line).")

# Chat input
chat_history = st.text_area("Chat History:")

# Analyze button
if st.button("Analyze Emotion From Chat"):
    if chat_history.strip() == "":
        st.warning("Please enter chat messages to analyze.")
    else:
        # Split chat into individual messages
        messages = chat_history.strip().split("\n")

        emotion_scores = []
        emotion_labels = []

        st.subheader(" Message-by-Message Emotion Detection")

        # Analyze every line
        for i, msg in enumerate(messages):
            label, probs = model.predict_emotion(msg)
            emotion_labels.append(label)
            emotion_scores.append(max(probs.values()))

            st.write(f"**Message {i+1}:** {msg}")
            st.write(f"Emotion → {label}")
            st.write("---")

        # Handle single-message chat (to create graph)
        if len(emotion_scores) == 1:
            emotion_scores.append(emotion_scores[0])

        # Create graph
        graph_path = "emotion_plot.png"
        create_emotion_graph(emotion_scores, graph_path)
        st.image(graph_path, caption="📈 Emotion Timeline")

        # Summary (dominant emotion)
        st.subheader("📝 Chat Mood Summary")
        dominant = max(set(emotion_labels), key=emotion_labels.count)
        st.success(f"Dominant Emotion: **{dominant}**")
