import math
import torch
import torch.nn as nn


# Implementation of Multi Head Attention from a scratch

# Calculate the attention matrix from q and k\
def calc_attn_scores(q, k):
    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)
    attn_scores = torch.bmm(q_scaled, k.transpose(-2, -1))
    attn_scores = nn.functional.softmax(attn_scores, dim=-1)
    return attn_scores


class SingleHeadAttentionFromScratch(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.head_dim = embed_dim

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        nn.init.xavier_uniform_(self.in_proj_weight)
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, ))
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    # Calculate the query, key and value matrices from the input x
    def calculate_qkv(self, x):
        query = x.transpose(0, 1)
        seq_sz, batch_sz, embedding_size = query.shape

        # pass through the linear layers
        proj = nn.functional.linear(query, self.in_proj_weight, bias=self.in_proj_bias)
        proj = proj.unflatten(-1, (3, embedding_size))
        proj = proj.unsqueeze(0)
        proj = proj.transpose(0, -2)
        proj = proj.squeeze(-2).contiguous()
        q, k, v = proj[0], proj[1], proj[2]

        return q, k, v

    # Calculate the attention scores

    def forward(self, x):
        # Transpose to have seq_len first and then batch_size and then embed_dim
        batch_sz, seq_sz, embedding_size = x.shape

        # Calculate the query, key and value matrices
        q, k, v = self.calculate_qkv(x)

        # Setup q, k, v to calculate attention by making batch dim 0
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # Calculate the attention scores
        attn_scores = calc_attn_scores(q, k)

        # Calculate the output
        attn_output = torch.bmm(attn_scores, v)
        attn_output = nn.functional.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output, attn_scores


if __name__ == '__main__':
    # Example parameters
    batch_size = 3
    seq_len = 5
    embed_dim = 4
    num_heads = 1

    # Set the random seed for reproducibility
    torch.manual_seed(0)

    # Create a random tensor of shape (batch_size, seq_len, embed_dim)
    x = torch.rand((batch_size, seq_len, embed_dim))

    # Create a multi_head attention layer
    multi_head_attention = SingleHeadAttentionFromScratch(embed_dim)

    attention_output, attention_weights = multi_head_attention(x)

    print('x', x.shape)
    print('attention_output', attention_output.shape)
    print('attention_weights', attention_weights.shape)
    print("Output: ", attention_output)
    print("Attention weights: ", attention_weights)
