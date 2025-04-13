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

            # Process transcription (text_target)
            text_inputs = self.model.tokenizer(text_target, padding=True, return_tensors="pt")
            input_ids = text_inputs['input_ids'].long().to(self.model.model.device)
            attention_mask = text_inputs['attention_mask'].to(self.model.model.device)

            print("text_inputs", text_inputs)
            print("input_ids.shape", input_ids.shape)
            print("attention_mask.shape", attention_mask.shape)
            
            # Get text embeddings
            text_embeddings = self.model.model.get_input_embeddings()(input_ids)
            text_embeddings = text_embeddings.to(self.model.model.device).to(self.model.model.dtype)

            print("text_embeddings.shape", text_embeddings.shape)
            
            # Combine embeddings
            # audio_emb is [1, 80, X], we need to reshape it to match text_embeddings
            audio_emb = audio_emb.mean(dim=1)  # Average over the 80 channels
            audio_emb = audio_emb.unsqueeze(1)  # Add sequence dimension
            combined_embeddings = torch.cat([audio_emb, text_embeddings], dim=1)

            print("audio_emb.shape", audio_emb.shape)
            print("combined_embeddings.shape", combined_embeddings.shape)
            
            # Get attention mask
            audio_mask = torch.ones(1, 1, device=self.model.model.device)  # [batch_size, 1]
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(0)
            combined_mask = torch.cat([audio_mask, attention_mask], dim=1)
            
            # Generate predictions
            # Pad labels to match combined_embeddings length
            padded_labels = torch.cat([
                torch.full((1, 1), -100, device=self.model.model.device),  # -100 is ignored in loss
                input_ids
            ], dim=1)
            
            outputs = self.model.model(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_mask,
                labels=padded_labels
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