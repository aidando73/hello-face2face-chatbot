import torch
import torch.nn as nn
import torch.nn.functional as F
from audio_qwen_integration import AudioQwenModel
import os
import wandb
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import dataset_loader
class AudioTextAlignment(nn.Module):
    def __init__(self, model: AudioQwenModel):
        super().__init__()
        self.model = model
        
    def forward(self, audio_paths, text_targets = None):
        # Process each audio file separately
        losses = []
        if os.environ.get("DEBUG"):
            print("Model hidden size:", self.model.model.config.hidden_size)
        
        if text_targets is None:
            audio_emb = self.model.process_audio(audio_paths[0])
            
            # Use generate method instead of just getting logits
            outputs = self.model.model.generate(
                inputs_embeds=audio_emb,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
            
            # Decode only the generated tokens
            generated_text = self.model.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            return generated_text

        # Process audio_embeddings individually
        audio_embeddings = []
        for audio_path in audio_paths:
            audio_emb = self.model.process_audio(audio_path)
            audio_emb = audio_emb.to(self.model.model.device).to(self.model.model.dtype)
            audio_embeddings.append(audio_emb)
        
        text_inputs = self.model.tokenizer(text_targets, padding=True, return_tensors="pt")
        text_input_ids = text_inputs['input_ids'].long().to(self.model.model.device)
        text_mask = text_inputs['attention_mask'].to(self.model.model.device)
        text_embeddings = self.model.model.get_input_embeddings()(text_input_ids)

        combined_inputs = []
        combined_masks = []

        for i, (audio_emb, text_emb, mask) in enumerate(zip(audio_embeddings, text_embeddings, text_mask)):
            print(audio_emb.shape, text_emb.shape, mask.shape)
            input_embeds = torch.cat([audio_emb, text_emb], dim=1)
            combined_inputs.append(input_embeds)

            combined_mask = torch.cat([torch.ones_like(audio_emb), mask], dim=1)
            combined_masks.append(combined_mask)

        # Pad the combined inputs and masks to the max length
        labels = []
        max_length = max(len(input_embeds) for input_embeds in combined_inputs)
        for i, (input_embeds, mask) in enumerate(zip(combined_inputs, combined_masks)):
            if len(input_embeds) < max_length:
                combined_inputs[i] = torch.cat([input_embeds, torch.zeros((1, max_length - len(input_embeds)), device=self.model.model.device)], dim=1)
                combined_masks[i] = torch.cat([mask, torch.zeros((1, max_length - len(mask)), device=self.model.model.device)], dim=1)
                
            labels = torch.full((1, max_length), -100, device=self.model.model.device)
            labels[:, audio_emb.shape[1]:-1] = text_input_ids[:, 1:]
            labels[:, len(input_embeds)] = self.model.model.config.eos_token_id
            labels.append(labels)

        combined_inputs = torch.cat(combined_inputs, dim=0)
        combined_masks = torch.cat(combined_masks, dim=0)
        labels = torch.cat(labels, dim=0)
        # Generate text from audio embeddings
        outputs = self.model.model(
            inputs_embeds=combined_inputs,
            labels=labels,
            attention_mask=combined_masks,
        )
        print(outputs.logits.shape)

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

def train_alignment(
        model,
        num_epochs=2,
        learning_rate=1e-5,
        batch_size=32,
        save_dir='checkpoints',
        val_every=700,
        tracking_enabled=False,
        debug=False,
    ):
    train_loader = dataset_loader.create_dataloader(data_dir="data", subset='train-clean-100', batch_size=batch_size)
    val_loader = dataset_loader.create_dataloader(data_dir="data", subset='test-clean', batch_size=batch_size)

    # Initialize wandb
    if tracking_enabled:
        wandb.init(
            project="jarvis-social-iq-module",
            config={
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "batch_size": train_loader.batch_size,
                "model": "Qwen2.5-7B",
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealing"
            }
        )
    
    alignment_model = AudioTextAlignment(model)
    
    # Freeze Qwen model parameters
    for param in alignment_model.model.model.parameters():
        param.requires_grad = False

    # Only optimize the audio encoder
    # optimizer = torch.optim.AdamW([
    #     {'params': alignment_model.model.audio_encoder.connector[0].parameters()}
    # ], lr=learning_rate)
    optimizer = torch.optim.SGD([
        {'params': alignment_model.model.audio_encoder.parameters()},
    ], lr=learning_rate, momentum=0.99)

    # # Add cosine annealing scheduler
    # scheduler = CosineAnnealingLR(
    #     optimizer, 
    #     T_max=num_epochs,
    #     eta_min=learning_rate / 10
    # )

    timestamp = datetime.now().astimezone(timezone(timedelta(hours=11))).strftime('%Y%m%d_%H%M')
    save_dir = f"checkpoints/{timestamp}"
    
    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        alignment_model.train(True)
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

            # Print out gradient statistics
            if os.environ.get("DEBUG"):
                print("\nGradient statistics:")
                for name, param in alignment_model.model.audio_encoder.connector.named_parameters():
                    print(f"{name}: gradient - mean={param.grad.mean().item():.4f}, std={param.grad.std().item():.4f}, min={param.grad.min().item():.4f}, max={param.grad.max().item():.4f}")

            # Calculate gradient norm
            pre_clipped_grad_norm = 0.0
            for p in alignment_model.model.audio_encoder.parameters():
                if p.grad is not None:
                    pre_clipped_grad_norm += p.grad.data.norm(2).item() ** 2
            pre_clipped_grad_norm = pre_clipped_grad_norm ** 0.5
            
            # Print gradient statistics for each layer
            if os.environ.get("DEBUG"):
                print("\nGradient statistics per layer:")
                for name, param in alignment_model.model.audio_encoder.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: mean={param.grad.mean().item():.4f}, std={param.grad.std().item():.4f}")

            post_clipped_grad_norm = 0.0
            for p in alignment_model.model.audio_encoder.parameters():
                if p.grad is not None:
                    post_clipped_grad_norm += p.grad.data.norm(2).item() ** 2
            post_clipped_grad_norm = post_clipped_grad_norm ** 0.5
            
            # Log gradient norm
            wandb.log({
                "batch_loss": loss.item(),
                "grad_norm/pre_clip": pre_clipped_grad_norm,
                "grad_norm/post_clip": post_clipped_grad_norm,
                "epoch": epoch,
                "batch": batch_idx,
            })
            
            if os.environ.get("DEBUG"):
                print(f"Gradient norm: {pre_clipped_grad_norm:.4f}")
                print("\nConnector model parameters - before step:")
                for name, param in alignment_model.model.audio_encoder.connector.named_parameters():
                    print(f"{name}: shape={param.shape}, mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}, min={param.data.min().item():.6f}, max={param.data.max().item():.6f}")

            optimizer.step()

            # Print parameters of the connector model
            if os.environ.get("DEBUG"):
                print("\nConnector model parameters - after step:")
                for name, param in alignment_model.model.audio_encoder.connector.named_parameters():
                    print(f"{name}: shape={param.shape}, mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}, min={param.data.min().item():.6f}, max={param.data.max().item():.6f}")
            
            total_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % val_every == 0:
                epoch_val_loss = 0
                for batch_idx, batch in tqdm(enumerate(val_loader), desc=f"Validation Batches", total=len(val_loader)):
                    audio_paths = batch['audio_paths']
                    text_targets = batch['text_targets']
                    alignment_model.eval()
                    loss = alignment_model(audio_paths, text_targets)
                    epoch_val_loss += loss.item() * len(batch['audio_paths'])
                epoch_val_loss = epoch_val_loss / len(val_loader.dataset)

            # Log batch metrics
            wandb.log({
                "loss/batch_train": loss.item(),
                "loss/val": epoch_val_loss,
                "epoch": epoch,
                "batch": batch_idx,
                # "learning_rate": scheduler.get_last_lr()[0]
                # For compatibility with old logging
                "batch_loss": loss.item(),
            })

            print("--------------------------------")
            print(f"Batch {batch_idx} loss: {loss.item():.4f}")
            print("--------------------------------")

        # Calculate and log epoch metrics
        epoch_loss = total_loss / num_batches
        
        wandb.log({
            "loss/epoch_train": epoch_loss,
            "epoch": epoch,
            # "learning_rate": scheduler.get_last_lr()[0]
            # For compatibility with old logging
            "epoch_loss": epoch_loss,
        })
        
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.8f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        # Step the scheduler at the end of each epoch
        # scheduler.step()
        
        # Save checkpoint every epoch
        alignment_model.save(os.path.join(save_dir, f'epoch_{epoch+1}'))
    
    wandb.finish()
    return alignment_model

if __name__ == "__main__":
    # Example usage
    model = AudioQwenModel()

    train_alignment(model)