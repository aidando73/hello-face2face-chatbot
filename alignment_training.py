import torch
import torch.nn as nn
import torch.nn.functional as F
from audio_qwen_integration import AudioQwenModel
import os
import wandb
from datetime import datetime
from tqdm import tqdm

class AudioTextAlignment(nn.Module):
    def __init__(self, model: AudioQwenModel):
        super().__init__()
        self.model = model
        
    def forward(self, audio_paths, text_prompts, text_targets):
        # Process each audio file separately
        losses = []
        
        print("Model hidden size:", self.model.model.config.hidden_size)
        
        for audio_path, text_target in zip(audio_paths, text_targets):
            # Process audio using the model's existing functionality
            audio_emb = self.model.process_audio(audio_path)
            audio_emb = audio_emb.to(self.model.model.device).to(self.model.model.dtype)
            
            print("audio_emb.shape", audio_emb.shape)

            # Take first 30 frames (or pad if shorter)
            target_frames = 30
            if audio_emb.shape[1] < target_frames:
                padding = torch.zeros(1, target_frames - audio_emb.shape[1], audio_emb.shape[2], 
                                    device=audio_emb.device, dtype=audio_emb.dtype)
                audio_emb = torch.cat([audio_emb, padding], dim=1)
            else:
                audio_emb = audio_emb[:, :target_frames, :]
            
            # Tokenize the target text (for loss computation)
            text_inputs = self.model.tokenizer(text_target, padding=True, return_tensors="pt")
            target_ids = text_inputs['input_ids'].long().to(self.model.model.device)
            
            # Generate text from audio embeddings
            outputs = self.model.model(
                inputs_embeds=audio_emb,
                labels=target_ids  # This will make the model try to predict the target text
            )

            print("outputs.loss", outputs.loss)
            
            losses.append(outputs.loss)
        
        # Average the losses
        avg_loss = torch.mean(torch.stack(losses))
        return avg_loss

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
    
    # Only optimize the audio encoder
    optimizer = torch.optim.AdamW([
        {'params': alignment_model.model.audio_encoder.parameters()}
    ], lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Batches (Epoch {epoch+1}/{num_epochs})", total=len(train_loader)):
            # Extract paths and targets from batch
            audio_paths = batch['audio_paths']
            text_prompts = batch['text_prompts']
            text_targets = batch['text_targets']
            
            # Forward pass
            loss = alignment_model(audio_paths, text_prompts, text_targets)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print("Warning: NaN loss detected!")
                print("Audio paths:", audio_paths)
                print("Text targets:", text_targets)
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch metrics
            wandb.log({
                "batch_loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })
        
        # Calculate and log epoch metrics
        epoch_loss = total_loss / num_batches
        
        wandb.log({
            "epoch_loss": epoch_loss,
            "epoch": epoch
        })
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        # Save checkpoint every epoch
        alignment_model.save(os.path.join(save_dir, f'epoch_{epoch+1}'))
    
    wandb.finish()
    return alignment_model

if __name__ == "__main__":
    # Example usage
    model = AudioQwenModel()
    # You would need to create a DataLoader with your audio-text pairs
    # train_loader = create_your_dataloader
    import dataset_loader
    train_loader = dataset_loader.create_dataloader(data_dir="data/librispeech/LibriSpeech/", subset='dev-clean')
    train_alignment(model, train_loader) 