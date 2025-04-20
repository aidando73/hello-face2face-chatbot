import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import OrderedDict

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=1024, num_heads=8, num_layers=24, text_embed_dim=4096):
        super().__init__()
        
        # CNN downsampling layers
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.intermediate_size = hidden_dim * (((input_dim - 1) // 2 - 1) // 2)
        print("intermediate_size", self.intermediate_size)
        self.out = nn.Linear(self.intermediate_size, hidden_dim)

        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Two-layer MLP connector for audio-text modality
        # self.connector = nn.Sequential(OrderedDict([
        #     # ("layernorm1", nn.LayerNorm(hidden_dim, eps=1e-2)),
        #     ("linear1", nn.Linear(hidden_dim, text_embed_dim)),
        # ]))
        self.connector = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(hidden_dim, hidden_dim * 2)),
            ("layernorm1", nn.LayerNorm(hidden_dim * 2)),  # Add normalization after first linear
            ("gelu1", nn.GELU()),
            ("layernorm2", nn.LayerNorm(hidden_dim * 2)),
            ("linear2", nn.Linear(hidden_dim * 2, text_embed_dim)),
            ("layernorm3", nn.LayerNorm(text_embed_dim))  # Add normalization to final output
        ]))

        # # Initialize CNN layers
        # for m in self.cnn_layers.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        
        # # Initialize connector with small weights
        # for m in self.connector.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0.0, std=0.001)  # Very small initialization
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim, seq_len)
        print(f"Audio encoder input shape: {x.shape}")

        # Debug: Print input stats
        if os.environ.get("DEBUG"):
            print(f"Audio encoder input stats - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}, min: {x.min().item():.4f}, max: {x.max().item():.4f}")

        # Apply CNN downsampling
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)
        print(f"Audio encoder input shape after transpose: {x.shape}")
        x = self.cnn_layers(x)

        x = x.transpose(1, 2)
        b, t, h, m = x.size()
        x = x.contiguous().view(b, t, h * m)
        x = self.out(x)

        if True or os.environ.get("DEBUG"):
            print("CNN output shape:", x.shape)

        x = self.embedding(x)

        # Apply transformer
        x = self.transformer(x)
        
        # Apply modality connector
        if os.environ.get("DEBUG"):
            print(f"Pre-connector stats: min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}")

        x = self.connector(x)  # (batch_size, seq_len/16, text_embed_dim)

        if os.environ.get("DEBUG"):
            print(f"Post-connector stats: min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}")
        
        # Debug: Print final output stats
        if os.environ.get("DEBUG"):
            print(f"Final output stats - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}, min: {x.min().item():.4f}, max: {x.max().item():.4f}")
        
        # Check for NaN or Inf values in final output
        if os.environ.get("DEBUG") and (torch.isnan(x).any() or torch.isinf(x).any()):
            print("Warning: NaN or Inf values detected in final output!")
            print(f"NaN count: {torch.isnan(x).sum().item()}")
            print(f"Inf count: {torch.isinf(x).sum().item()}")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return x * 0.1


def apply_local_cmvn(features, epsilon=1e-8):
    """Apply CMVN to features tensor for each sample independently.
    
    Args:
        features: Tensor of shape [batch_size, n_features, time]
        epsilon: Small value to avoid division by zero
        
    Returns:
        Normalized features tensor of same shape
    """
    # Calculate mean and std along time dimension (dim=2)
    mean = torch.mean(features, dim=2, keepdim=True)
    std = torch.std(features, dim=2, keepdim=True) + epsilon
    
    # Normalize
    return (features - mean) / std


if __name__ == "__main__":
    audio_encoder = AudioEncoder()

    x = torch.randn(1, 80, 1037)
    x = audio_encoder(x)
    print(f"Audio encoder output shape: {x.shape}")