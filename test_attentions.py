import torch
from torch.nn import MultiheadAttention
from self_attention_from_scratch import SingleHeadAttentionFromScratch
from multi_head_self_attention_from_scratch import MultiHeadAttentionFromScratch


# Check that the output of MultiHeadAttention using only a single head and SingleHeadAttentionFromScratch are the same
def test_single_head_attention():
    # Parameters for the test
    batch_size = 3
    seq_len = 5
    embed_dim = 4

    # Set the random seed to get the same results
    torch.manual_seed(0)

    # Create a random tensor of shape (batch_size, seq_len, embed_dim)
    x = torch.rand((batch_size, seq_len, embed_dim))

    # Reset the random seed to get the same results
    torch.manual_seed(0)

    # Create a multi_head attention layer
    # Create a multi_head attention layer
    attention = MultiheadAttention(embed_dim, 1, 0.0, batch_first=True)

    attention_output, attention_weights = attention(x, x, x)

    # Reset the random seed to get the same results
    torch.manual_seed(0)

    # Create a single-head attention layer
    single_head_attention = SingleHeadAttentionFromScratch(embed_dim)

    # Check that the initial weights are the same
    assert torch.allclose(attention.in_proj_weight, single_head_attention.in_proj_weight, atol=1e-4)
    assert torch.allclose(attention.in_proj_bias, single_head_attention.in_proj_bias, atol=1e-4)
    assert torch.allclose(attention.out_proj.weight, single_head_attention.out_proj.weight, atol=1e-4)
    assert torch.allclose(attention.out_proj.bias, single_head_attention.out_proj.bias, atol=1e-4)

    single_attention_output, single_attention_weights = single_head_attention(x)

    # Check that the outputs are the same
    assert torch.allclose(attention_output, single_attention_output, atol=1e-4)
    assert torch.allclose(attention_weights, single_attention_weights, atol=1e-4)


# Check that the output of MultiHeadAttention using only a single head and MultiHeadAttentionFromScratch are the same
def test_multi_head_attention():
    # Parameters for the test
    batch_size = 3
    seq_len = 5
    embed_dim = 4
    num_heads = 2

    # Set the random seed to get the same results
    torch.manual_seed(0)

    # Create a random tensor of shape (batch_size, seq_len, embed_dim)
    x = torch.rand((batch_size, seq_len, embed_dim))

    # Reset the random seed to get the same results
    torch.manual_seed(0)

    # Create a multi_head attention layer
    attention = MultiheadAttention(embed_dim, num_heads, 0.0, batch_first=True)

    attention_output, attention_weights = attention(x, x, x)

    # Reset the random seed to get the same results
    torch.manual_seed(0)

    # Create a multi_head attention layer
    multi_head_attention = MultiHeadAttentionFromScratch(embed_dim, num_heads)

    # Check that the initial weights are the same
    assert torch.allclose(attention.in_proj_weight, multi_head_attention.in_proj_weight, atol=1e-4)
    assert torch.allclose(attention.in_proj_bias, multi_head_attention.in_proj_bias, atol=1e-4)
    assert torch.allclose(attention.out_proj.weight, multi_head_attention.out_proj.weight, atol=1e-4)
    assert torch.allclose(attention.out_proj.bias, multi_head_attention.out_proj.bias, atol=1e-4)

    multi_head_attention_output, multi_head_attention_weights = multi_head_attention(x)

    assert torch.allclose(attention_output, multi_head_attention_output, atol=1e-4)
    assert torch.allclose(attention_weights, multi_head_attention_weights, atol=1e-4)
