# -*- coding: utf-8 -*-
# @Time    : 2023/8/19 21:11
# @Author  :
# @File    : my_model.py
# @Software: PyCharm
import torch.nn.functional as F
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)


    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out


class my_model(nn.Module):
    def __init__(self, input_size, hid1, hid2, hid3, output_size):
        super(my_model, self).__init__()
        self.self_attention = SelfAttention(input_size, 2)

        self.input_layer = nn.Linear(input_size, hid1)
        self.hid_layer1 = nn.Linear(hid1, hid2)
        self.hid_layer2 = nn.Linear(hid2, hid3)
        self.out_layer = nn.Linear(hid3, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        attention_output = self.self_attention(x, x, x, mask=None)
        x = attention_output

        x = x.squeeze(1)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hid_layer1(x))
        x = F.relu(self.hid_layer2(x))
        x = F.sigmoid(self.out_layer(x))
        return x
