import torch
import torch.nn as nn
import torch.nn.functional as F
from audio_qwen_integration import AudioQwenModel
import torch.distributed as dist
import os
import wandb
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import dataset_loader
import torch.multiprocessing as mp

class AudioTextAlignment(nn.Module):
    def __init__(self, model: AudioQwenModel):
        super().__init__()
        self.model = model
        
    def forward(self, audio_paths, text_targets = None):
        # Process each audio file separately
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
            text_emb = text_emb.unsqueeze(0)
            input_embeds = torch.cat([audio_emb, text_emb], dim=1)
            combined_inputs.append(input_embeds)

            combined_mask = torch.cat([torch.ones(audio_emb.shape[1]).to(self.model.model.device), mask])
            combined_mask = combined_mask.unsqueeze(0)
            combined_masks.append(combined_mask)

        # Pad the combined inputs and masks to the max length
        combined_labels = []
        max_length = max(input_embeds.shape[1] for input_embeds in combined_inputs)
        for i, (input_embeds, mask, text_ids, audio_emb) in enumerate(zip(combined_inputs, combined_masks, text_input_ids, audio_embeddings)):
            if len(input_embeds) < max_length:
                combined_inputs[i] = torch.cat([input_embeds, torch.zeros((1, max_length - input_embeds.shape[1], input_embeds.shape[2]), device=self.model.model.device)], dim=1)
                combined_masks[i] = torch.cat([mask, torch.zeros((1, max_length - mask.shape[1]), device=self.model.model.device)], dim=1)
            labels = torch.full((1, max_length), -100, device=self.model.model.device, dtype=torch.long)
            labels[:, audio_emb.shape[1]:audio_emb.shape[1]+len(text_ids) - 1] = text_ids[1:]
            labels[:, len(input_embeds) - 1] = self.model.model.config.eos_token_id
            combined_labels.append(labels)

        combined_inputs = torch.cat(combined_inputs, dim=0).to(dtype=torch.float16)
        combined_masks = torch.cat(combined_masks, dim=0).to(dtype=torch.int32)
        combined_labels = torch.cat(combined_labels, dim=0)

        # Generate text from audio embeddings
        outputs = self.model.model(
            inputs_embeds=combined_inputs,
            labels=combined_labels,
            attention_mask=combined_masks,
        )

        # Decode the model's output logits to get the predicted tokens
        logits = outputs.logits
            
        # Get the predicted token IDs (take the argmax along the vocabulary dimension)
        predicted_token_ids = torch.argmax(logits[0], dim=-1)
        
        # Convert the predicted token IDs back to text
        predicted_text = self.model.tokenizer.batch_decode(
            predicted_token_ids, 
            skip_special_tokens=True
        )

        print("\nSample prediction:")
        print(f"Target: {text_targets[0]}")
        print(f"Prediction: {''.join(predicted_text)}")
        print(f"Loss: {outputs.loss.item():.4f}")
        
        # Average the losses
        return outputs.loss
    
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
        num_epochs=1,
        learning_rate=1e-5,
        batch_size=8,
        save_dir='checkpoints',
        val_every=300,
        tracking_enabled=True,
        debug=False,
    ):
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK")))
    world_size = int(os.environ.get("WORLD_SIZE"))
    dist.init_process_group(backend="nccl")
    rank = int(dist.get_rank())
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    print(f"World size: {world_size}, Rank: {rank}, Device ID: {device_id}")

    train_loader = dataset_loader.create_dataloader(subset='train-clean-100', world_size=world_size, rank=rank, batch_size=batch_size)
    val_loader = dataset_loader.create_dataloader(subset='test-clean', world_size=world_size, rank=rank, batch_size=batch_size)

    # Initialize wandb
    if tracking_enabled and rank == 0:
        wandb.init(
            project="jarvis-social-iq-module",
            config={
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "batch_size": train_loader.batch_size,
                "val_every": val_every,
                "save_dir": save_dir,
                "model": "Qwen2.5-7B",
                "optimizer": "SGD"
            }
        )

    model = AudioQwenModel().to(device_id)
    alignment_model = AudioTextAlignment(model)
    alignment_model = alignment_model.to(device_id)
    # Freeze Qwen model parameters
    for param in alignment_model.model.model.parameters():
        param.requires_grad = False
    
    alignment_model = torch.nn.parallel.DistributedDataParallel(alignment_model, device_ids=[device_id], output_device=device_id)

    optimizer = torch.optim.SGD([
        {'params': alignment_model.module.model.audio_encoder.parameters()},
    ], lr=learning_rate, momentum=0.99)

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
            
            # Backward pass
            loss.backward()

            # Calculate gradient norm
            pre_clipped_grad_norm = 0.0
            for p in alignment_model.module.model.audio_encoder.parameters():
                if p.grad is not None:
                    pre_clipped_grad_norm += p.grad.data.norm(2).item() ** 2
            pre_clipped_grad_norm = pre_clipped_grad_norm ** 0.5
            
            post_clipped_grad_norm = 0.0
            for p in alignment_model.module.model.audio_encoder.parameters():
                if p.grad is not None:
                    post_clipped_grad_norm += p.grad.data.norm(2).item() ** 2
            post_clipped_grad_norm = post_clipped_grad_norm ** 0.5
            
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

            train_loss = loss.item()
            train_batch_idx = batch_idx

            # Clear memory before validation
            del loss
            optimizer.zero_grad()

            val_loss = None
            if global_step % val_every == 0 or global_step == len(train_loader) - 1:
                val_loss = 0
                alignment_model.eval()
                for batch_idx, batch in tqdm(enumerate(val_loader), desc=f"Validation Batches", total=len(val_loader)):
                    audio_paths = batch['audio_paths']
                    text_targets = batch['text_targets']
                    the_loss = alignment_model(audio_paths, text_targets)
                    val_loss += the_loss.item() * len(batch['audio_paths'])
                val_loss_tensor = torch.tensor([val_loss], device=f'cuda:{device_id}')
                dist.all_reduce(val_loss_tensor)
                val_loss = val_loss_tensor.item() / len(val_loader)

            # Log batch metrics
            if tracking_enabled and rank == 0:
                wandb.log({
                    "loss/batch_train": train_loss,
                    "loss/val": val_loss,
                    "grad_norm/pre_clip": pre_clipped_grad_norm,
                    "grad_norm/post_clip": post_clipped_grad_norm,
                    "epoch": epoch,
                    "batch": train_batch_idx,
                })

            print("--------------------------------")
            print(f"Batch {batch_idx} loss: {train_loss:.4f}")
            print("--------------------------------")

        # Calculate and log epoch metrics
        epoch_loss = total_loss / num_batches
        if tracking_enabled and rank == 0:
            wandb.log({
                "loss/epoch_train": epoch_loss,
                "epoch": epoch,
            })
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        if rank == 0:
            alignment_model.module.save(os.path.join(save_dir, f'epoch_{epoch+1}'))
    
    if tracking_enabled and rank == 0:
        wandb.finish()
    
    dist.destroy_process_group()
    return alignment_model

if __name__ == "__main__":
    train_alignment()