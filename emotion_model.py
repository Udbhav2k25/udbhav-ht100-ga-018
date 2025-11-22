from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class EmotionModel:
    """Wrapper around a HuggingFace RoBERTa-based emotion classifier with safe device handling."""

    def _init_(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        # choose device (cpu by default; use cuda if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # load model onto CPU first, then move to desired device to avoid "meta" placements
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        # move model to device (CPU or GPU)
        self.model.to(self.device)
        self.model.eval()

        # label mapping (use model config)
        self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]

    def predict_emotion(self, text: str):
        """Return (top_label, probs_dict). Probabilities are floats between 0 and 1."""

        if not text or not text.strip():
            # default for empty input
            return "neutral", {k: 0.0 for k in self.labels}

        # tokenize and move tensors to model device
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        # move each tensor to the same device as the model
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        # run model safely
        with torch.no_grad():
            outputs = self.model(**inputs)

            # ensure logits are a real tensor (not meta) and detach to cpu
            logits = outputs.logits
            # if logits are meta tensors (some exotic setups), try forcing to cpu
            if getattr(logits, "is_meta", False):
                # try to move model to CPU and re-run (safer fallback)
                self.model.to(torch.device("cpu"))
                self.device = torch.device("cpu")
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits

            probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()[0]

        probs_dict = {label: float(probs[idx]) for idx, label in enumerate(self.labels)}
        top_label = self.labels[int(probs.argmax())]

        return top_label, probs_dict
