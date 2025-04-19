import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import torchaudio.compliance.kaldi as ta_kaldi
import numpy as np
from audio_encoder import AudioEncoder
import os

class MelFilterBank:
    def __init__(
        self,
        sampling_rate: int = 16000,
        num_mel_bins: int = 80,
    ):
        self.sampling_rate = sampling_rate
        self.num_mel_bins = num_mel_bins
        
    def process_audio(self, audio_path: str) -> torch.Tensor:
        # Load audio file
        if os.environ.get("DEBUG"):
            print(f"Loading audio file: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Debug: Print waveform stats
        if os.environ.get("DEBUG"):
            print(f"Waveform stats - mean: {waveform.mean().item():.4f}, std: {waveform.std().item():.4f}, min: {waveform.min().item():.4f}, max: {waveform.max().item():.4f}")
        
        # Resample if necessary
        if sample_rate != self.sampling_rate:
            resampler = T.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
            print(f"Resampling from {sample_rate} to {self.sampling_rate}")
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print(f"Converted to mono: {waveform.shape}")
        
        # Debug: Print resampled waveform stats
        if os.environ.get("DEBUG"):
            print(f"Resampled waveform stats - mean: {waveform.mean().item():.4f}, std: {waveform.std().item():.4f}, min: {waveform.min().item():.4f}, max: {waveform.max().item():.4f}")
        
        
        waveform = waveform * (2**15) # Kaldi compliance: 16-bit signed integers
        # Compute mel spectrogram
        mel_spec = ta_kaldi.fbank(
            waveform,
            num_mel_bins=self.num_mel_bins,
            sample_frequency=self.sampling_rate,
            frame_length=25,
            frame_shift=10,
            dither=1.0,
            energy_floor=0.0,
        )

        # Transpose the mel spectrogram to match the expected shape [1, 80, T]
        # Current shape is [T, 80], where T is the time dimension (1037 in this example)
        mel_spec = mel_spec.transpose(0, 1)
        mel_spec = mel_spec.unsqueeze(0)
        print(f"Mel spec shape: {mel_spec.shape}")

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
        plt.tight_layout()
        
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


if __name__ == "__main__":
    main()
