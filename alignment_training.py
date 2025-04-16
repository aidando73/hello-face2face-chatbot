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
        
    def forward(self, audio_paths, text_targets = None):
        # Process each audio file separately
        losses = []
        if os.environ.get("DEBUG"):
            print("Model hidden size:", self.model.model.config.hidden_size)
        
        # if text_targets is None:
        #     audio_emb = self.model.process_audio(audio_paths[0])
        #     outputs = self.model.model(
        #         inputs_embeds=audio_emb,
        #     )
        #     predicted_text = self.model.tokenizer.batch_decode(
        #         outputs.logits.argmax(dim=-1), 
        #         skip_special_tokens=True
        #     )
        #     return predicted_text
        
        for audio_path, text_target in zip(audio_paths, text_targets):
            # Process audio using the model's existing functionality
            audio_emb = self.model.process_audio(audio_path)
            audio_emb = audio_emb.to(self.model.model.device).to(self.model.model.dtype)
            
            if os.environ.get("DEBUG"):
                print("audio_emb.shape", audio_emb.shape)

            # Tokenize the target text (for loss computation)
            text_inputs = self.model.tokenizer(text_target, padding=True, return_tensors="pt")
            text_input_ids = text_inputs['input_ids'].long().to(self.model.model.device)
            text_emb = self.model.model.get_input_embeddings()(text_input_ids)
            
            input_embeds = torch.cat([audio_emb, text_emb], dim=1)
            labels = torch.full((1, input_embeds.shape[1]), -100, device=self.model.model.device)
            # Set labels for text positions only (shifted by 1 for next-token prediction)
            labels[:, audio_emb.shape[1]:-1] = text_input_ids[:, 1:]
            # Last position predicts EOS token
            labels[:, -1] = self.model.model.config.eos_token_id

            if os.environ.get("DEBUG"):
                print("input_embeds.shape", input_embeds.shape)
                print("labels.shape", labels.shape)
            
            # Generate text from audio embeddings
            outputs = self.model.model(
                inputs_embeds=input_embeds,
                labels=labels,
                # attention_mask=audio_attention_mask,
                # output_hidden_states=True,
            )

            if os.environ.get("DEBUG"):
                print("outputs.logits.shape", outputs.logits.shape)
                # print("num hidden states", len(outputs.hidden_states))
                # print("outputs.hidden_states shapes", [h.shape for h in outputs.hidden_states])

            # Decode the model's output logits to get the predicted tokens
            logits = outputs.logits
            
            # Get the predicted token IDs (take the argmax along the vocabulary dimension)
            predicted_token_ids = torch.argmax(logits, dim=-1)
            
            # Convert the predicted token IDs back to text
            predicted_text = self.model.tokenizer.batch_decode(
                predicted_token_ids, 
                skip_special_tokens=True
            )

            print("\nSample prediction:")
            print(f"Target: {text_target}")
            print(f"Prediction: {predicted_text[0]}")
            print(f"Loss: {outputs.loss.item():.4f}")
            
            losses.append(outputs.loss)
        
        # Average the losses
        avg_loss = torch.mean(torch.stack(losses))
        return avg_loss
    
    def save(self, path):
        """
        Save the audio encoder model to the specified path.
        
        Args:
            path (str): Directory path where the model will be saved
        """
        os.makedirs(path, exist_ok=True)
        
        # Save only the audio encoder state dict
        torch.save(
            self.model.audio_encoder.state_dict(),
            os.path.join(path, "audio_encoder.pt")
        )
        
        print(f"Audio encoder saved to {os.path.join(path, 'audio_encoder.pt')}")
    
    def load(self, path):
        """
        Load the audio encoder model from the specified path.
        
        Args:
            path (str): Directory path where the model will be loaded from
        """
        self.model.audio_encoder.load_state_dict(
            torch.load(os.path.join(path, "audio_encoder.pt"))
        )
        print(f"Audio encoder loaded from {os.path.join(path, 'audio_encoder.pt')}")

def train_alignment(model, train_loader, num_epochs=10, learning_rate=1e-7, save_dir='checkpoints'):
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
    
    for param in alignment_model.model.audio_encoder.cnn_layers.parameters():
        param.requires_grad = False
    
    for param in alignment_model.model.audio_encoder.transformer.parameters():
        param.requires_grad = False
    
    for param in alignment_model.model.audio_encoder.connector.parameters():
        param.requires_grad = True
    
    # Only optimize the audio encoder
    optimizer = torch.optim.AdamW([
        {'params': alignment_model.model.audio_encoder.connector.parameters()}
    ], lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Batches (Epoch {epoch+1}/{num_epochs})", total=len(train_loader)):
            # Extract paths and targets from batch
            audio_paths = batch['audio_paths']
            text_targets = batch['text_targets']
            
            # Forward pass
            loss = alignment_model(audio_paths, text_targets)
            
            # Check for NaN loss
            if os.environ.get("DEBUG") and torch.isnan(loss):
                print("Warning: NaN loss detected!")
                print("Audio paths:", audio_paths)
                print("Text targets:", text_targets)
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # for n, p in alignment_model.model.audio_encoder.named_parameters():
            #     if p.requires_grad:
            #         print(f"{n}: grad_exists={p.grad is not None}, grad_nan={torch.isnan(p.grad).any() if p.grad is not None else 'N/A'}")
            
            # Calculate gradient norm
            grad_norm = 0.0
            for p in alignment_model.model.audio_encoder.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            # Print gradient statistics for each layer
            if os.environ.get("DEBUG"):
                print("\nGradient statistics per layer:")
                for name, param in alignment_model.model.audio_encoder.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: mean={param.grad.mean().item():.4f}, std={param.grad.std().item():.4f}")
            
            # Clip gradients
            max_grad_norm = 0.1
            if grad_norm > max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    alignment_model.model.audio_encoder.parameters(),
                    max_grad_norm
                )
                grad_norm = max_grad_norm
                if os.environ.get("DEBUG"):
                    print(f"Gradients clipped from {grad_norm:.4f} to {max_grad_norm}")
            
            # Log gradient norm
            wandb.log({
                "batch_loss": loss.item(),
                "grad_norm": grad_norm,
                "epoch": epoch,
                "batch": batch_idx
            })
            
            if os.environ.get("DEBUG"):
                print(f"Gradient norm: {grad_norm:.4f}")
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch metrics
            wandb.log({
                "batch_loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })

            print("--------------------------------")
            print(f"Batch {batch_idx} loss: {loss.item():.4f}")
            print("--------------------------------")
        
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
    import dataset_loader
    train_loader = dataset_loader.create_dataloader(data_dir="data/librispeech/LibriSpeech/", subset='dev-clean')
    train_alignment(model, train_loader)