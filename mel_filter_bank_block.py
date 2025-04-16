import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from audio_encoder import AudioEncoder

class MelFilterBank:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        
        # Initialize mel spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
        
    def process_audio(self, audio_path: str) -> torch.Tensor:
        # Load audio file
        if os.environ.get("DEBUG"):
            print(f"Loading audio file: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Debug: Print waveform stats
        if os.environ.get("DEBUG"):
            print(f"Waveform stats - mean: {waveform.mean().item():.4f}, std: {waveform.std().item():.4f}, min: {waveform.min().item():.4f}, max: {waveform.max().item():.4f}")
        
        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = T.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Debug: Print resampled waveform stats
        if os.environ.get("DEBUG"):
            print(f"Resampled waveform stats - mean: {waveform.mean().item():.4f}, std: {waveform.std().item():.4f}, min: {waveform.min().item():.4f}, max: {waveform.max().item():.4f}")
        
        # Compute mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Debug: Print raw mel spectrogram stats
        if os.environ.get("DEBUG"):
            print(f"Raw mel spectrogram stats - mean: {mel_spec.mean().item():.4f}, std: {mel_spec.std().item():.4f}, min: {mel_spec.min().item():.4f}, max: {mel_spec.max().item():.4f}")
        
        # Add small epsilon to prevent log(0)
        mel_spec = mel_spec + 1e-6
        
        # Check for NaN or Inf values
        if os.environ.get("DEBUG") and (torch.isnan(mel_spec).any() or torch.isinf(mel_spec).any()):
            print("Warning: NaN or Inf values detected in mel spectrogram!")
            print(f"NaN count: {torch.isnan(mel_spec).sum().item()}")
            print(f"Inf count: {torch.isinf(mel_spec).sum().item()}")
            # Replace NaN and Inf with small values
            mel_spec = torch.nan_to_num(mel_spec, nan=1e-6, posinf=1e6, neginf=-1e6)
        
        # Convert to log scale with numerical stability
        mel_spec = torch.log(mel_spec)
        
        # Debug: Print log mel spectrogram stats
        if os.environ.get("DEBUG"):
            print(f"Log mel spectrogram stats - mean: {mel_spec.mean().item():.4f}, std: {mel_spec.std().item():.4f}, min: {mel_spec.min().item():.4f}, max: {mel_spec.max().item():.4f}")
        
        # Check for NaN or Inf values after log
        if torch.isnan(mel_spec).any() or torch.isinf(mel_spec).any():
            print("Warning: NaN or Inf values detected after log transform!")
            print(f"NaN count: {torch.isnan(mel_spec).sum().item()}")
            print(f"Inf count: {torch.isinf(mel_spec).sum().item()}")
            # Replace NaN and Inf with small values
            mel_spec = torch.nan_to_num(mel_spec, nan=-10.0, posinf=10.0, neginf=-10.0)
        
        # Normalize to have zero mean and unit variance
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        
        # Final debug print
        if os.environ.get("DEBUG"):
            print(f"Final mel spectrogram stats - mean: {mel_spec.mean().item():.4f}, std: {mel_spec.std().item():.4f}, min: {mel_spec.min().item():.4f}, max: {mel_spec.max().item():.4f}")
        
        return mel_spec
    
    def visualize(self, mel_spec: torch.Tensor, save_path: str = None):
        # Convert to numpy for visualization
        mel_spec_np = mel_spec.squeeze().numpy()
        
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec_np, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

def main():
    # Initialize mel filter bank
    mel_filter = MelFilterBank()
    
    # Process audio file
    mel_spec = mel_filter.process_audio('audio-sample.wav')
    
    # Print mel spectrogram information
    if os.environ.get("DEBUG"): 
        print("\nMel Spectrogram:")
        print(f"Shape: {mel_spec.shape}")
        print(f"Mean: {mel_spec.mean():.4f}")
        print(f"Std: {mel_spec.std():.4f}")
        print(f"Min: {mel_spec.min():.4f}")
        print(f"Max: {mel_spec.max():.4f}")
    
    # Visualize mel spectrogram
    mel_filter.visualize(mel_spec, 'mel_spectrogram.png')
    
    # Initialize and run audio encoder
    audio_encoder = AudioEncoder(
        input_dim=80,  # matches n_mels
        hidden_dim=512,
        num_heads=8,
        num_layers=24
    )
    
    # Process through audio encoder
    encoded_audio = audio_encoder(mel_spec)
    
    # Print encoded audio information
    if os.environ.get("DEBUG"): 
        print("\nEncoded Audio:")
        print(f"Shape: {encoded_audio.shape}")
        print(f"Mean: {encoded_audio.mean():.4f}")
        print(f"Std: {encoded_audio.std():.4f}")
        print(f"Min: {encoded_audio.min():.4f}")
        print(f"Max: {encoded_audio.max():.4f}")
    
    return mel_spec, encoded_audio

if __name__ == "__main__":
    main()
