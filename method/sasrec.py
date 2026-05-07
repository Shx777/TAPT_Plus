import torch
import copy
import math
import torch.nn as nn
import torch.nn.functional as F


def clones(layer, depth):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(depth)])


def get_mask(x, bidirectional):
    mask = (x > 0).int().unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
    if bidirectional:
        return mask
    else:
        mask = torch.tril(mask)
        return mask


class SubLayerConnect(nn.Module):
    def __init__(self, features, dropout_ratio):
        super(SubLayerConnect, self).__init__()
        self.norm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, mask=None, dropout=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MHAttn(nn.Module):
    def __init__(self, n_head, d_model, dropout_ratio):
        super(MHAttn, self).__init__()
        self.d_k = d_model // n_head
        self.h = n_head
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.merge = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, query, key, value, mask=None):
        b = query.size(0)
        q, k, v = [l(x).view(b, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        x = self.merge(x)
        return x


class FFN(nn.Module):
    def __init__(self, d_model, exp_factor, dropout):
        super(FFN, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_model * exp_factor)
        self.linear_2 = nn.Linear(d_model * exp_factor, d_model)
        # self.act = nn.GELU()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.act(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_head, exp_factor, dropout_ratio):
        super(TransformerLayer, self).__init__()
        self.multi_head_attn_layer = MHAttn(n_head, d_model, dropout_ratio)
        self.feed_forward_layer = FFN(d_model, exp_factor, dropout_ratio)
        self.sublayer_1 = SubLayerConnect(d_model, dropout_ratio)
        self.sublayer_2 = SubLayerConnect(d_model, dropout_ratio)

    def forward(self, x, mask):
        x = self.sublayer_1(x, lambda e: self.multi_head_attn_layer(e, e, e, mask))
        x = self.sublayer_2(x, self.feed_forward_layer)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, n_head, exp_factor, dropout_ratio, depth):
        super(Transformer, self).__init__()
        self.mixer_layer = TransformerLayer(d_model, n_head, exp_factor, dropout_ratio)
        self.mixer = clones(self.mixer_layer, depth)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.mixer:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model, dropout_ratio):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, x):
        lookup_table = self.pos_embedding.weight[:x.size(1), :]
        x += lookup_table.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = self.dropout(x)
        return x


class SASRec(nn.Module):
    def __init__(self, n_loc, **kwargs):
        super(SASRec, self).__init__()
        d_model = kwargs['d_model']
        max_len = kwargs['max_len']
        p_drop = kwargs['p_drop']
        depth = kwargs['depth']
        n_head = kwargs['n_head']
        exp_factor = kwargs['exp_factor']
        self.emb_loc = nn.Embedding(n_loc, d_model, padding_idx=0)
        self.emb_pos = PositionalEmbedding(max_len, d_model, p_drop)
        self.mixer = Transformer(d_model, n_head, exp_factor, p_drop, depth)
        self.out = nn.Linear(d_model, n_loc)

    def forward(self, seq, data_size):
        x = self.emb_loc(seq)
        x = self.emb_pos(x)
        mask = get_mask(seq, bidirectional=False)
        mixer_output = self.mixer(x, mask)

        if self.training:
            output = self.out(mixer_output)
        else:
            output = mixer_output[torch.arange(data_size.size(0)), data_size - 1, :].detach()
            output = self.out(output)
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))