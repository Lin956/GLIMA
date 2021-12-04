import torch
import torch.nn as nn
import math


#layers
# shape of input: [D, T, embed_size]
# shape of output: [D, T, embed_size]
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = embed_size // heads
        self.queries = nn.Linear(self.per_dim, self.per_dim)
        self.keys = nn.Linear(self.per_dim, self.per_dim)
        self.values = nn.Linear(self.per_dim, self.per_dim)
        self.linears = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        D, T, E = x.shape

        x = x.view(D, T, self.heads, self.per_dim)

        # compute queries, keys and values
        queries = self.queries(x)
        queries = queries.transpose(1, 2)
        keys = self.keys(x)
        keys = keys.transpose(1, 2)
        values = self.values(x)
        values = values.transpose(1, 2)  # [D, heads, T, per_dim]

        # scaled dot-product
        attn = torch.softmax(torch.matmul(queries, keys.transpose(2, 3))
                        / (self.embed_size ** (1 / 2)), dim=-1)  # [D, heads, T, T]
        # print(attn.shape)
        out = torch.matmul(attn, values)  # [D, heads, T, per_dim]
        # print(out.shape)

        out = out.reshape(D, T, self.heads*self.per_dim)
        out = self.linears(out)
        return out


# input: [C, T, 1]
# output: [C, T, embed_size]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class CrossSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CrossSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = embed_size // heads
        self.queries = nn.Linear(self.per_dim, self.per_dim)
        self.keys = nn.Linear(self.per_dim, self.per_dim)
        self.values = nn.Linear(self.per_dim, self.per_dim)
        # self.linears = nn.Linear(embed_size, embed_size)

    def forward(self, x, task):
        D, T, E = x.shape

        x = x.view(D, T, self.heads, self.per_dim)
        task = task.view(D, T, self.heads, self.per_dim)

        # compute queries, keys and values
        queries = self.queries(task)
        queries = queries.transpose(1, 2)
        keys = self.keys(x)
        keys = keys.transpose(1, 2)
        values = self.values(x)
        values = values.transpose(1, 2)  #[D, heads, T, per_dim]

        # scaled dot-product
        attn = torch.softmax(torch.matmul(queries, keys.transpose(2, 3))
                        / (self.embed_size ** (1 / 2)), dim=-1)  # [D, heads, T, T]
        # print(attn.shape)
        out = torch.matmul(attn, values)  # [D, heads, T, per_dim]
        # print(out.shape)

        out = out.reshape(D, T, self.heads*self.per_dim)
        # out = self.linears(out)
        return out

