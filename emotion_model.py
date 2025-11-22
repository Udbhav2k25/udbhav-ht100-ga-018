from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class EmotionModel:
    """Wrapper around a HuggingFace RoBERTa-based emotion classifier."""

    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.labels = [
            "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
        ]

    def predict_emotion(self, text: str):
        """Return (top_label, probs_dict)"""
        if not text or not text.strip():
            return "neutral", {k: 0.0 for k in self.labels}

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        probs_dict = {label: float(probs[idx]) for idx, label in enumerate(self.labels)}
        top_label = self.labels[int(probs.argmax())]

        return top_label, probs_dict
