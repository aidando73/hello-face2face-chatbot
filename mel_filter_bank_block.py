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
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = T.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Compute mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to log scale
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        
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
    print("\nEncoded Audio:")
    print(f"Shape: {encoded_audio.shape}")
    print(f"Mean: {encoded_audio.mean():.4f}")
    print(f"Std: {encoded_audio.std():.4f}")
    print(f"Min: {encoded_audio.min():.4f}")
    print(f"Max: {encoded_audio.max():.4f}")
    
    return mel_spec, encoded_audio

if __name__ == "__main__":
    main()
