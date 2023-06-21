import torch
from torch.nn import MultiheadAttention

if __name__ == '__main__':
    # Example parameters
    batch_size = 3
    seq_len = 5
    embed_dim = 4
    num_heads = 2
    dropout = 0.0

    # Set the random seed for reproducibility
    torch.manual_seed(0)

    # Create a random tensor of shape (batch_size, seq_len, embed_dim)
    x = torch.rand((batch_size, seq_len, embed_dim))

    # Create a multi_head attention layer
    multi_head_attention = MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)

    attention_output, attention_weights = multi_head_attention(x, x, x)

    print('x', x.shape)
    print('attention_output', attention_output.shape)
    print('attention_weights', attention_weights.shape)
    print("Output: ", attention_output)
    print("Attention weights: ", attention_weights)
