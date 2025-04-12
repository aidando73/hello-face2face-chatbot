import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=512, num_heads=8, num_layers=24, text_embed_dim=4096):
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
        self.connector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, text_embed_dim)  # Project to language model's embedding dimension
        )
        
    def forward(self, x):
        # x shape: (batch_size, input_dim, seq_len)
        
        # Apply CNN downsampling
        x = self.cnn_layers(x)  # (batch_size, hidden_dim, seq_len/16)
        
        # Reshape for transformer
        x = x.permute(0, 2, 1)  # (batch_size, seq_len/16, hidden_dim)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Apply modality connector
        x = self.connector(x)  # (batch_size, seq_len/16, text_embed_dim)
        
        return x
