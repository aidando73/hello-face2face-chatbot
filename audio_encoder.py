from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import OrderedDict
import math
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)

class WhaleAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_dim, num_heads=16, dropout=0.1, qk_normalization=False, use_relative_pe=True, layer_norm_eps=1e-05):
        super().__init__()
        self.embed_dim = hidden_dim
        self.num_heads = num_heads
        self.use_flash_attn = False
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5
        self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_v = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_out = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_drop = nn.Dropout(dropout)

        self.qk_normalization = qk_normalization

        if self.qk_normalization:
            self.q_norm = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)
            self.k_norm = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)

        self.linear_out = nn.Linear(self.embed_dim, self.embed_dim)

        self.use_relative_pe = use_relative_pe
        if self.use_relative_pe:

            self.linear_pos = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            # these two learnable bias are used in matrix c and matrix d
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            self.pos_bias_u = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
            self.pos_bias_v = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
            nn.init.xavier_uniform_(self.pos_bias_u)
            nn.init.xavier_uniform_(self.pos_bias_v)


    def _naive_attn(self, x, attention_mask=None, pos_embeds=None):
        B, N, C = x.shape
        q = self.linear_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.linear_k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.linear_v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        if self.use_relative_pe:

            q = q.transpose(1, 2)
            batch_size = pos_embeds.size(0)
            p = self.linear_pos(pos_embeds.to(q.dtype)).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            query_with_bias_u = (q + self.pos_bias_u.to(q.device)).transpose(1, 2)
            query_with_bias_v = (q + self.pos_bias_v.to(q.device)).transpose(1, 2)

            # compute attention score
            # first compute matrix a and matrix c
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            matrix_ac = torch.matmul(query_with_bias_u, k.transpose(-2, -1))
            # compute matrix b and matrix d
            matrix_bd = torch.matmul(query_with_bias_v, p.transpose(-2, -1))
            attn = (matrix_ac + matrix_bd) * self.scale

        else:
            attn = ((q * self.scale) @ k.transpose(-2, -1))

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~attention_mask.bool(), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.linear_out(x)
        return x


    def forward(
            self, 
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor = None,
            pos_embeds: torch.Tensor = None
        ) -> torch.Tensor:
        x = self._naive_attn(hidden_states, attention_mask, pos_embeds)
        return x

    
class WhaleMLP(nn.Module):
    def __init__(self, hidden_dim, intermediate_size=4096, dropout=0.1, act="relu"):
        super().__init__()
        self.act = nn.ReLU()
        self.w_1 = nn.Linear(hidden_dim,
                                        intermediate_size,
                                        bias=True)
        self.w_2 = nn.Linear(intermediate_size,
                                     hidden_dim,
                                     bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.w_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.w_2(hidden_states)
        return hidden_states


class WhaleAudioEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        intermediate_size=4096,
        dropout=0.1,
        act="relu",
        num_heads=16,
        qk_normalization=False,
        use_relative_pe=True,
        layer_norm_eps=1e-05,
        concat_after=False,
        normalize_before=True
    ):
        super().__init__()
        self.embed_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout
        self.normalize_before = normalize_before
        self.concat_after = concat_after

        self.attn = WhaleAttention(hidden_dim, num_heads, dropout, qk_normalization, use_relative_pe, layer_norm_eps)
        self.feed_forward = WhaleMLP(hidden_dim, intermediate_size, dropout, act)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        self.dropout = nn.Dropout(dropout)

        if self.concat_after:
            self.concat_linear = nn.Linear(self.embed_dim * 2, self.embed_dim)
        else:
            self.concat_linear = nn.Identity()


    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            pos_emb: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.norm1(hidden_states)
        if self.concat_after:
            hidden_states = torch.cat(
                [hidden_states, self.attn(hidden_states, attention_mask, pos_emb)],
                dim=-1
            )
            hidden_states = self.concat_linear(hidden_states) + residual
        else:
            hidden_states = self.dropout(self.attn(hidden_states, attention_mask, pos_emb)) + residual
        if not self.normalize_before:
            hidden_states = self.norm1(hidden_states)

        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.norm2(hidden_states)
        hidden_states = self.dropout(self.feed_forward(hidden_states)) + residual
        if not self.normalize_before:
            hidden_states = self.norm2(hidden_states)

        return hidden_states


class WhaleAudioEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].
    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_size=4096,
        dropout=0.1,
        act="relu",
        num_heads=16,
        qk_normalization=False,
        use_relative_pe=True,
        layer_norm_eps=1e-05,
        concat_after=False,
        normalize_before=True,
        num_layers=16
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            WhaleAudioEncoderLayer(hidden_dim, num_heads, dropout, qk_normalization, use_relative_pe, layer_norm_eps, concat_after, normalize_before) for idx in range(num_layers)])
        self.gradient_checkpointing = True

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.layer_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

    def forward(
            self,
            inputs_embeds,
            attention_mask: Optional[torch.FloatTensor] = None,
            pos_embeds: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = False
        return_dict = True

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    encoder_layer,
                    hidden_states,
                    attention_mask,
                    pos_embeds,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    pos_embeds,
                )
            hidden_states = layer_outputs
        
        if self.normalize_before:
            hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )


class AudioEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=1024, num_heads=8, num_layers=24, text_embed_dim=4096):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_embed_dim = text_embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
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

        self.pe_max_len = 5000
        self.pe = torch.zeros(self.pe_max_len, hidden_dim, dtype=torch.float32)
        position = torch.arange(0, self.pe_max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
        
        self.encoder = WhaleAudioEncoder(hidden_dim, num_heads=num_heads, num_layers=num_layers)

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
        
        self.connector = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(hidden_dim, hidden_dim * 2)),
            ("layernorm1", nn.LayerNorm(hidden_dim * 2)),  # Add normalization after first linear
            ("gelu1", nn.GELU()),
            ("layernorm2", nn.LayerNorm(hidden_dim * 2)),
            ("linear2", nn.Linear(hidden_dim * 2, text_embed_dim)),
            ("layernorm3", nn.LayerNorm(text_embed_dim))  # Add normalization to final output
        ]))

        
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

        xscale = math.sqrt(self.hidden_dim)
        x = x * xscale
        pos_emb = self.pe[:, :x.size(1)].to(x.device).type(x.dtype)
        x = x + pos_emb

        # Apply transformer
        encoder_outputs = self.encoder(x, pos_embeds=pos_emb)
        x = encoder_outputs.last_hidden_state
        
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
        
        return x


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