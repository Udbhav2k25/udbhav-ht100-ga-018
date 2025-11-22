import matplotlib.pyplot as plt
import numpy as np

def create_emotion_graph(emotion_scores, output_path="emotion_plot.png"):
    """
    Create and save a simple emotion confidence timeline graph.

    emotion_scores: list of floats
    """
    if not emotion_scores:
        raise ValueError("emotion_scores list is empty.")

    x = list(range(1, len(emotion_scores) + 1))
    y = emotion_scores

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker='o')
    plt.xlabel("Message Number")
    plt.ylabel("Emotion Confidence Score")
    plt.title("Emotion Timeline")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
