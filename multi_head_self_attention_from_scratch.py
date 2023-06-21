import math
import torch
import torch.nn as nn

from self_attention_from_scratch import SingleHeadAttentionFromScratch, calc_attn_scores


class MultiHeadAttentionFromScratch(SingleHeadAttentionFromScratch):
    def __init__(self, embed_dim, n_heads):
        super().__init__(embed_dim)
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

    # Calculate the attention scores and output of the multi-head attention layer
    def forward(self, x):
        # Get the batch size and sequence length
        batch_sz, seq_sz, embedding_size = x.shape

        # q, k, v are calculated the same way
        q, k, v = self.calculate_qkv(x)

        # Calculate the attention scores. Each head has its own attention scores. This is done by folding the different
        # heads into the batch dimension and then calculating the attention scores
        q = q.view(seq_sz, batch_sz * self.n_heads, self.head_dim)
        q = q.transpose(0, 1)
        k = k.view(seq_sz, batch_sz * self.n_heads, self.head_dim)
        k = k.transpose(0, 1)
        v = v.view(seq_sz, batch_sz * self.n_heads, self.head_dim)
        v = v.transpose(0, 1)

        # Calculate the attention scores
        attn_scores = calc_attn_scores(q, k)

        # Calculate the attention output
        attn_output = torch.bmm(attn_scores, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_sz * batch_sz, embedding_size)
        attn_output = nn.functional.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(seq_sz, batch_sz, attn_output.size(1))
        attn_output = attn_output.transpose(0, 1)

        # Fix the shape of the attention scores
        attn_scores = attn_scores.view(batch_sz, self.n_heads, seq_sz, seq_sz)
        attn_scores = attn_scores.mean(dim=1)

        return attn_output, attn_scores


if __name__ == '__main__':
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
    multi_head_attention = MultiHeadAttentionFromScratch(embed_dim, num_heads)
    multi_head_attention_output, multi_head_attention_weights = multi_head_attention(x)

    # print(multi_head_attention_output.shape)
    print(multi_head_attention_weights.shape)
    # print(multi_head_attention_output)
    print(multi_head_attention_weights)
