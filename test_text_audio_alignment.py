from alignment_training import AudioTextAlignment
from audio_qwen_integration import AudioQwenModel
import os
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/20250420_2235")
    parser.add_argument("--epoch", type=int, default=5)
    args = parser.parse_args()

    model = AudioQwenModel()

    alignment_model = AudioTextAlignment(model)
    alignment_model.load(os.path.join(args.checkpoint, f"epoch_{args.epoch}"))

    predicted_text = alignment_model(["./audio-sample.wav"])

    print(predicted_text)
