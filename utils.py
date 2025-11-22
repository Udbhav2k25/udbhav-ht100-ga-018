import matplotlib.pyplot as plt

def create_emotion_graph(emotion_labels, output_path="emotion_plot.png"):
    """
    Graph showing categorized emotion timeline (no confidence).
    """

    EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

    # Convert emotion labels → numeric positions
    y_positions = [EMOTIONS.index(label) for label in emotion_labels]

    x = list(range(1, len(emotion_labels) + 1))

    plt.figure(figsize=(12, 5))
    plt.plot(x, y_positions, marker='o', markersize=10, linewidth=2)

    plt.yticks(range(len(EMOTIONS)), EMOTIONS)
    plt.xlabel("Message Number")
    plt.ylabel("Emotion")
    plt.title("Emotion Timeline (Categorical)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()
