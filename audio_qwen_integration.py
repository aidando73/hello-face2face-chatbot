import torch
from audio_encoder import AudioEncoder
from load_qwen import load_qwen_model
from mel_filter_bank_block import MelFilterBank

class AudioQwenModel:
    def __init__(self):
        # Load Qwen model and tokenizer
        self.model, self.tokenizer = load_qwen_model()
        
        # Initialize audio components
        self.mel_filter = MelFilterBank()
        self.audio_encoder = AudioEncoder(
            input_dim=80,
            hidden_dim=512,
            num_heads=8,
            num_layers=24,
            text_embed_dim=self.model.config.hidden_size  # Use Qwen's hidden size
        )
        
        # Move components to the same device and dtype as Qwen
        self.audio_encoder = self.audio_encoder.to(self.model.device)
        self.audio_encoder = self.audio_encoder.to(self.model.dtype)  # Convert to float16
        
    def process_audio(self, audio_path: str) -> torch.Tensor:
        # Process audio through mel filter bank
        mel_spec = self.mel_filter.process_audio(audio_path)
        mel_spec = mel_spec.to(self.model.device)
        mel_spec = mel_spec.to(self.model.dtype)  # Convert to float16
        
        # Process through audio encoder
        audio_embeddings = self.audio_encoder(mel_spec)
        return audio_embeddings
    
    def generate_response(self, audio_path: str, text_prompt: str = "", max_new_tokens: int = 100):
        # Process audio
        audio_embeddings = self.process_audio(audio_path)
        
        # Prepare text inputs
        text_inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.model.device)
        
        # Get text embeddings in the correct dtype
        text_embeddings = self.model.get_input_embeddings()(text_inputs.input_ids)
        text_embeddings = text_embeddings.to(self.model.dtype)
        
        # Combine audio and text embeddings
        combined_embeddings = torch.cat([
            audio_embeddings,  # Shape: (1, seq_len/16, hidden_size)
            text_embeddings  # Shape: (1, text_len, hidden_size)
        ], dim=1)
        
        # Create attention mask for combined sequence
        audio_mask = torch.ones(audio_embeddings.shape[:2], device=self.model.device)
        combined_mask = torch.cat([audio_mask, text_inputs.attention_mask], dim=1)
        
        # Generate response
        outputs = self.model.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def main():
    # Initialize the integrated model
    model = AudioQwenModel()
    
    # Example usage
    audio_path = "audio-sample.wav"
    text_prompt = "The sky is blue because:"
    
    response = model.generate_response(audio_path, text_prompt)
    print(f"Response: {response}")

if __name__ == "__main__":
    main() 