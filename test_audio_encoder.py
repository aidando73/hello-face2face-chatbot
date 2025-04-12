import torch
from audio_encoder import AudioEncoder

def test_audio_encoder():
    # Create dummy data
    batch_size = 4
    input_dim = 80  # typical mel-spectrogram dimension
    seq_len = 1600  # arbitrary sequence length
    hidden_dim = 512
    
    # Create random input tensor
    dummy_input = torch.randn(batch_size, input_dim, seq_len)
    print("\nInput tensor:")
    print(f"Shape: {dummy_input.shape}")
    print(f"Mean: {dummy_input.mean():.4f}")
    print(f"Std: {dummy_input.std():.4f}")
    print(f"Min: {dummy_input.min():.4f}")
    print(f"Max: {dummy_input.max():.4f}")
    
    # Initialize the encoder
    encoder = AudioEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=8,
        num_layers=24
    )
    
    # Forward pass
    output = encoder(dummy_input)
    
    print("\nOutput tensor:")
    print(f"Shape: {output.shape}")
    print(f"Mean: {output.mean():.4f}")
    print(f"Std: {output.std():.4f}")
    print(f"Min: {output.min():.4f}")
    print(f"Max: {output.max():.4f}")
    
    # Verify output shape
    expected_seq_len = seq_len // 16  # 4 CNN layers with stride 2
    assert output.shape == (batch_size, expected_seq_len, hidden_dim), \
        f"Expected shape {(batch_size, expected_seq_len, hidden_dim)}, got {output.shape}"
    
    # Verify no NaN values
    assert not torch.isnan(output).any(), "Output contains NaN values"
    
    # Verify no Inf values
    assert not torch.isinf(output).any(), "Output contains Inf values"
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_audio_encoder() 