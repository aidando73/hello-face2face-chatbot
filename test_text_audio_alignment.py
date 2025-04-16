from alignment_training import AudioTextAlignment
from audio_qwen_integration import AudioQwenModel

if __name__ == "__main__":
    model = AudioQwenModel()

    alignment_model = AudioTextAlignment(model)
    alignment_model.load("checkpoints/epoch_10")

    predicted_text = alignment_model(["./audio-sample.wav"])

    print(predicted_text)
