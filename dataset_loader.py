import os
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import wandb
from torch.utils.data.distributed import DistributedSampler
from pprint import pprint
def load_librispeech_to_dataframe(data_dir = "data/", subset='dev-clean'):
    """
    Load LibriSpeech data into a pandas DataFrame.
    
    Args:
        data_dir (str): Path to the LibriSpeech root directory
        subset (str): Which subset to load (e.g., 'train-clean-100', 'dev-clean', etc.)
    
    Returns:
        pd.DataFrame: DataFrame with columns ['audio_path', 'text', 'speaker_id', 'chapter_id']
    """
    data = []
    subset_dir = os.path.join(data_dir, subset, "LibriSpeech", subset)
    
    # Walk through the directory structure
    for speaker_dir in os.listdir(subset_dir):
        speaker_path = os.path.join(subset_dir, speaker_dir)
        if not os.path.isdir(speaker_path):
            continue
            
        for chapter_dir in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_dir)
            if not os.path.isdir(chapter_path):
                continue
                
            # Read the transcript file
            trans_file = os.path.join(chapter_path, f"{speaker_dir}-{chapter_dir}.trans.txt")
            if not os.path.exists(trans_file):
                continue
                
            with open(trans_file, 'r') as f:
                for line in f:
                    # Each line is in format: "speaker-chapter-utterance text"
                    parts = line.strip().split(' ', 1)
                    if len(parts) != 2:
                        continue
                        
                    utterance_id, text = parts
                    
                    # Construct audio file path
                    audio_path = os.path.join(chapter_path, f"{utterance_id}.flac")
                    if not os.path.exists(audio_path):
                        continue
                        
                    data.append({
                        'audio_path': audio_path,
                        'text': text.lower(),
                        'speaker_id': speaker_dir,
                        'chapter_id': chapter_dir,
                        'utterance_id': utterance_id
                    })
    
    return pd.DataFrame(data)

class LibriSpeechDataset(Dataset):
    def __init__(self, data_dir, subset='dev-clean'):
        self.df = load_librispeech_to_dataframe(data_dir, subset)
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'audio_path': row['audio_path'],
            'text_prompt': '',  # Empty prompt for pure transcription
            'text_target': row['text']
        }

def create_dataloader(data_dir="data", subset='dev-clean', world_size=1, rank=0, batch_size=32, num_workers=4, seed=None):
    dataset = LibriSpeechDataset(data_dir, subset)

    # Create a generator with the specified seed if provided
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        generator=generator,
        sampler=sampler,
        collate_fn=lambda batch: {
            'audio_paths': [item['audio_path'] for item in batch],
            'text_prompts': [item['text_prompt'] for item in batch],
            'text_targets': [item['text_target'] for item in batch]
        }
    )
    
    return dataloader

def evaluate_model(model, dataloader, metrics=['bleu', 'rouge', 'wer']):
    """Evaluate the model on the given dataloader"""
    from nltk.translate.bleu_score import sentence_bleu
    from rouge import Rouge
    import numpy as np
    from jiwer import wer
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Generate predictions
            predictions = model.generate_response(
                batch['audio_paths'],
                batch['text_prompts']
            )
            
            all_predictions.extend(predictions)
            all_targets.extend(batch['text_targets'])
    
    # Compute metrics
    results = {}
    
    if 'bleu' in metrics:
        bleu_scores = []
        for pred, target in zip(all_predictions, all_targets):
            score = sentence_bleu([target.split()], pred.split())
            bleu_scores.append(score)
        results['bleu'] = np.mean(bleu_scores)
    
    if 'rouge' in metrics:
        rouge = Rouge()
        rouge_scores = rouge.get_scores(all_predictions, all_targets, avg=True)
        results['rouge'] = rouge_scores
    
    if 'wer' in metrics:
        # Word Error Rate (standard metric for speech recognition)
        wer_score = wer(all_targets, all_predictions)
        results['wer'] = wer_score
    
    return results

if __name__ == "__main__":
    train_loader = create_dataloader(subset='test-clean', world_size=2, rank=1, batch_size=2)
    print(f"\nCreated dataloader with {len(train_loader)} batches")

    i = 0
    for batch in train_loader:
        print(f"Batch {i}")
        pprint([text_target[:10] for text_target in batch['text_targets']])
        i += 1
        if i >= 2:
            break