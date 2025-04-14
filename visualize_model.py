import torch
from torchviz import make_dot
from audio_qwen_integration import AudioQwenModel
import os

def visualize_model():
    # Initialize the model
    model = AudioQwenModel()
    
    # Create dummy input
    batch_size = 1
    input_dim = 80
    seq_len = 1600
    
    # Process through audio encoder directly
    audio_emb = model.process_audio("audio-sample.wav")
    
    # Create attention mask
    attention_mask = torch.ones(
        (audio_emb.shape[0], audio_emb.shape[1]),
        device=audio_emb.device,
        dtype=torch.long
    )
    
    # Get model outputs
    outputs = model.model(
        inputs_embeds=audio_emb,
        attention_mask=attention_mask
    )

    # print(outputs.logits.grad_fn)
    
    # Create visualization with just audio encoder parameters
    dot = make_dot(outputs.logits, 
                  params=dict(model.audio_encoder.named_parameters()),
                  show_attrs=True,
                  show_saved=True)
    
    # Save the visualization
    dot.render('model_graph', format='svg', cleanup=True)
    print("Model graph saved as 'model_graph.svg'")

if __name__ == "__main__":
    visualize_model() 