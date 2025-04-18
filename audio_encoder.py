import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import OrderedDict

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=512, num_heads=8, num_layers=12, text_embed_dim=4096):
        super().__init__()
        
        # CNN downsampling layers
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
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

        if os.environ.get("DEBUG"):
            print(f"Audio encoder pre CMVN - global mean: {x.mean().item():.4f}, std: {x.std().item():.4f}, min: {x.min().item():.4f}, max: {x.max().item():.4f}")
            print(f"Channel means min/max: {x.mean(dim=2).min().item():.4f}/{x.mean(dim=2).max().item():.4f}")
            print(f"Channel stds min/max: {x.std(dim=2).min().item():.4f}/{x.std(dim=2).max().item():.4f}")

        # x = apply_local_cmvn(x)

        if os.environ.get("DEBUG"):
            print(f"Audio encoder post CMVN - global mean: {x.mean().item():.4f}, std: {x.std().item():.4f}, min: {x.min().item():.4f}, max: {x.max().item():.4f}")
            print(f"Channel means min/max: {x.mean(dim=2).min().item():.4f}/{x.mean(dim=2).max().item():.4f}")
            print(f"Channel stds min/max: {x.std(dim=2).min().item():.4f}/{x.std(dim=2).max().item():.4f}")

        # Debug: Print input stats
        if os.environ.get("DEBUG"):
            print(f"Audio encoder input stats - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}, min: {x.min().item():.4f}, max: {x.max().item():.4f}")
        
        # Check for NaN or Inf values in input
        if os.environ.get("DEBUG") and (torch.isnan(x).any() or torch.isinf(x).any()):
            print("Warning: NaN or Inf values detected in audio encoder input!")
            print(f"NaN count: {torch.isnan(x).sum().item()}")
            print(f"Inf count: {torch.isinf(x).sum().item()}")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply CNN downsampling
        x = self.cnn_layers(x)  # (batch_size, hidden_dim, seq_len/16)
        
        if os.environ.get("DEBUG"):
            print("CNN output shape:", x.shape)

        # Debug: Print CNN output stats
        if os.environ.get("DEBUG"):
            print(f"CNN output stats - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}, min: {x.min().item():.4f}, max: {x.max().item():.4f}")
        
        # Check for NaN or Inf values after CNN
        if os.environ.get("DEBUG") and (torch.isnan(x).any() or torch.isinf(x).any()):
            print("Warning: NaN or Inf values detected after CNN!")
            print(f"NaN count: {torch.isnan(x).sum().item()}")
            print(f"Inf count: {torch.isinf(x).sum().item()}")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Reshape for transformer
        x = x.permute(0, 2, 1)  # (batch_size, seq_len/16, hidden_dim)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Debug: Print transformer output stats
        if os.environ.get("DEBUG"):
            print(f"Transformer output stats - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}, min: {x.min().item():.4f}, max: {x.max().item():.4f}")
        
        # Check for NaN or Inf values after transformer
        if os.environ.get("DEBUG") and (torch.isnan(x).any() or torch.isinf(x).any()):
            print("Warning: NaN or Inf values detected after transformer!")
            print(f"NaN count: {torch.isnan(x).sum().item()}")
            print(f"Inf count: {torch.isinf(x).sum().item()}")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
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