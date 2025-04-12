import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import os

class LibriSpeechDataset(Dataset):
    def __init__(self, split='train', cache_dir='./cache'):
        self.dataset = load_dataset('openslr/librispeech_asr', split=split, cache_dir=cache_dir)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get audio and transcription
        audio = item['audio']
        transcription = item['text']
        
        # Save audio to file (required by your audio encoder)
        audio_path = os.path.join(self.cache_dir, f'audio_{idx}.wav')
        audio.save(audio_path)
        
        return {
            'audio_path': audio_path,
            'text_prompt': '',  # Empty prompt for pure transcription
            'text_target': transcription
        }

def create_dataloader(batch_size=8, num_workers=4, split='train'):
    dataset = LibriSpeechDataset(split=split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True if split == 'train' else False,
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
    # Example usage
    train_loader = create_dataloader(split='train')
    val_loader = create_dataloader(split='validation')
    
    # Example evaluation
    # model = AudioQwenModel()
    # results = evaluate_model(model, val_loader)
    # print(f"Evaluation results: {results}") 