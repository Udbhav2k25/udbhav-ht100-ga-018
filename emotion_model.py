from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class EmotionModel:
    """Wrapper around a HuggingFace RoBERTa-based emotion classifier."""

    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        
        # Load tokenizer & model properly
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Force CPU & safe model load
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32
        ).to("cpu")

        # Emotion labels in correct model order
        self.labels = [
            "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
        ]

    def predict_emotion(self, text: str):
        """Return (top_label, probs_dict)"""
        if not text or not text.strip():
            return "neutral", {label: 0.0 for label in self.labels}

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to("cpu")

        # Predict safely (avoid meta tensor)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.detach().to("cpu")
            probs = torch.nn.functional.softmax(logits, dim=1).numpy()[0]

        # Convert to readable format
        probs_dict = {label: float(probs[idx]) for idx, label in enumerate(self.labels)}
        top_label = self.labels[int(np.argmax(probs))]

        return top_label, probs_dict
