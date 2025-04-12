import torch
import torch.nn as nn
import torch.nn.functional as F
from audio_qwen_integration import AudioQwenModel
import os
import wandb
from datetime import datetime

class AudioTextAlignment(nn.Module):
    def __init__(self, model: AudioQwenModel):
        super().__init__()
        self.model = model
        self.temperature = 0.07  # Temperature for contrastive learning
        
        # Cross-attention layer for audio-text alignment
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model.model.config.hidden_size,
            num_heads=8,
            batch_first=True
        )
        
    def save(self, save_dir: str):
        """Save the alignment model and audio encoder"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the audio encoder
        torch.save(self.model.audio_encoder.state_dict(), 
                  os.path.join(save_dir, 'audio_encoder.pt'))
        
        # Save the cross-attention layer
        torch.save(self.cross_attention.state_dict(),
                  os.path.join(save_dir, 'cross_attention.pt'))
        
    def load(self, save_dir: str):
        """Load the alignment model and audio encoder"""
        # Load the audio encoder
        self.model.audio_encoder.load_state_dict(
            torch.load(os.path.join(save_dir, 'audio_encoder.pt'))
        )
        
        # Load the cross-attention layer
        self.cross_attention.load_state_dict(
            torch.load(os.path.join(save_dir, 'cross_attention.pt'))
        )
        
    def forward(self, audio_paths, text_prompts, text_targets):
        # Process audio
        audio_embeddings = []
        for audio_path in audio_paths:
            audio_emb = self.model.process_audio(audio_path)
            audio_embeddings.append(audio_emb)
        audio_embeddings = torch.stack(audio_embeddings)
        
        # Process text prompts
        text_inputs = self.model.tokenizer(text_prompts, padding=True, return_tensors="pt")
        text_inputs = {k: v.to(self.model.model.device) for k, v in text_inputs.items()}
        text_embeddings = self.model.model.get_input_embeddings()(text_inputs.input_ids)
        
        # Apply cross-attention
        attended_audio, attention_weights = self.cross_attention(
            query=text_embeddings,
            key=audio_embeddings,
            value=audio_embeddings
        )
        
        # Combine embeddings
        combined_embeddings = torch.cat([attended_audio, text_embeddings], dim=1)
        
        # Get attention mask
        audio_mask = torch.ones(audio_embeddings.shape[:2], device=self.model.model.device)
        combined_mask = torch.cat([audio_mask, text_inputs.attention_mask], dim=1)
        
        # Generate predictions
        outputs = self.model.model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_mask,
            labels=text_inputs.input_ids
        )
        
        return outputs.loss, attention_weights

    def contrastive_loss(self, audio_embeddings, text_embeddings):
        # Normalize embeddings
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(audio_embeddings, text_embeddings.t()) / self.temperature
        
        # Create labels for contrastive learning
        labels = torch.arange(similarity.size(0), device=similarity.device)
        
        # Compute loss
        loss_audio = F.cross_entropy(similarity, labels)
        loss_text = F.cross_entropy(similarity.t(), labels)
        
        return (loss_audio + loss_text) / 2

def train_alignment(model, train_loader, num_epochs=10, learning_rate=1e-4, save_dir='checkpoints'):
    # Initialize wandb
    wandb.init(
        project="jarvis-social-iq-module",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "model": "Qwen2.5-7B",
            "optimizer": "AdamW"
        }
    )
    
    alignment_model = AudioTextAlignment(model)
    
    # Freeze Qwen model parameters
    for param in alignment_model.model.model.parameters():
        param.requires_grad = False
    
    # Only optimize the audio encoder and cross-attention
    optimizer = torch.optim.AdamW([
        {'params': alignment_model.model.audio_encoder.parameters()},
        {'params': alignment_model.cross_attention.parameters()}
    ], lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        total_attention_variance = 0  # Track attention pattern diversity
        
        for batch_idx, batch in enumerate(train_loader):
            audio_paths, text_prompts, text_targets = batch
            
            # Forward pass
            loss, attention_weights = alignment_model(audio_paths, text_prompts, text_targets)
            
            # Calculate attention pattern metrics
            attention_variance = torch.var(attention_weights, dim=-1).mean()
            total_attention_variance += attention_variance.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch metrics
            wandb.log({
                "batch_loss": loss.item(),
                "attention_variance": attention_variance.item(),
                "epoch": epoch,
                "batch": batch_idx
            })
        
        # Calculate and log epoch metrics
        epoch_loss = total_loss / num_batches
        avg_attention_variance = total_attention_variance / num_batches
        
        wandb.log({
            "epoch_loss": epoch_loss,
            "avg_attention_variance": avg_attention_variance,
            "epoch": epoch
        })
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Attention Variance: {avg_attention_variance:.4f}")
        
        # Save checkpoint every epoch
        alignment_model.save(os.path.join(save_dir, f'epoch_{epoch+1}'))
    
    wandb.finish()
    return alignment_model

if __name__ == "__main__":
    # Example usage
    model = AudioQwenModel()
    # You would need to create a DataLoader with your audio-text pairs
    # train_loader = create_your_dataloader()
    # train_alignment(model, train_loader) 