import streamlit as st
from emotion_model import EmotionModel
from utils import create_emotion_graph
import nltk

# Download sentence tokenizer (only first time)
nltk.download('punkt')

# Initialize model
model = EmotionModel()

st.title("🧠 Paragraph & Chat-Based Emotion Analyzer")
st.write("Paste a paragraph or chat, and the system will detect emotions sentence-by-sentence.")

# Input box
user_text = st.text_area("Enter paragraph or chat here:")

if st.button("Analyze Emotion"):
    if user_text.strip() == "":
        st.warning("⚠ Please enter some text to analyze.")
    else:
        # STEP 1: Split text to sentences using NLTK sentence tokenizer
        sentences = nltk.sent_tokenize(user_text)

        st.subheader("🔍 Sentence-by-Sentence Emotion Analysis")

        emotion_scores = []
        emotion_labels = []

        for i, sentence in enumerate(sentences):
            label, probs = model.predict_emotion(sentence)
            confidence = max(probs.values())
            emotion_scores.append(confidence)
            emotion_labels.append(label)

            st.write(f"**Sentence {i+1}:** {sentence}")
            st.write(f"Emotion → **{label}** | Confidence: {round(confidence, 2)}")
            st.write("---")

        # Avoid graph error if only one sentence
        if len(emotion_scores) == 1:
            emotion_scores.append(emotion_scores[0])

        # Create and display graph
        graph_path = "emotion_plot.png"
        create_emotion_graph(emotion_scores, graph_path)
        st.image(graph_path, caption="📈 Emotion Timeline")

        # Summary Section
        st.subheader("📝 Overall Summary")
        dominant = max(set(emotion_labels), key=emotion_labels.count)
        st.success(f"Dominant Emotion Detected: **{dominant}**")

        st.write(
            "This graph shows how emotional tone changes across each sentence in the input text."
        )
