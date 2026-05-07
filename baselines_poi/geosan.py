import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model, drop_ratio):
        super().__init__()
        self.drop = nn.Dropout(drop_ratio)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        if len(x.shape) == 3:
            lookup_table = self.pe.weight[:x.size(1), :]
            x = x + lookup_table.unsqueeze(0).repeat(x.size(0), 1, 1)
        else:
            lookup_table = self.pe.weight[:x.size(2), :]
            x = x + lookup_table.unsqueeze(0).unsqueeze(0).repeat(x.size(0), 1, 1, 1)
        x = self.drop(x)
        return x


class FFN(nn.Module):
    def __init__(self, d_model, drop_ratio):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_model * 4)
        self.linear_2 = nn.Linear(d_model * 4, d_model)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.drop(self.act(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def scaled_dot_product(q, k, v, mask, drop):
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

    seq_len = scores.shape[1]
    if mask is not None:
        mask = mask.unsqueeze(1)  # 将 mask 的维度从 (B, L_k) 扩展为 (B, 1, L_k)
        mask = mask.repeat(1, seq_len, 1)
        #print(scores.shape)
        #print(mask.shape)
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if drop is not None:
        p_attn = drop(p_attn)
    return torch.matmul(p_attn, v)


class Attn(nn.Module):
    def __init__(self, d_model, drop_ratio):
        super().__init__()
        self.map_q = nn.Linear(d_model, d_model)
        self.map_k = nn.Linear(d_model, d_model)
        self.map_v = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, x, mask):
        q, k, v = self.map_q(x), self.map_k(x), self.map_v(x)
        x = scaled_dot_product(q, k, v, mask, self.drop)
        del q, k, v
        return x


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SkipConnection(nn.Module):
    def __init__(self, d_model,  drop_ratio):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, x, layer):
        x = x + self.drop(layer(self.norm(x)))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, drop_ratio):
        super().__init__()
        self.attn_layer = Attn(d_model, drop_ratio)
        self.ffn_layer = FFN(d_model, drop_ratio)
        self.sublayer = clones(SkipConnection(d_model, drop_ratio), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.attn_layer(x, mask))
        x = self.sublayer[1](x, self.ffn_layer)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, drop_ratio, depth):
        super().__init__()
        self.encoder = clones(EncoderLayer(d_model, drop_ratio), depth)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.encoder:
            x = layer(x, mask)
        x = self.norm(x)
        return x

def generate_mask(data_size, seq_length):
    #print(seq_length)
    # 创建一个空的掩码张量，初始化为0
    mask = torch.zeros(len(data_size), seq_length, dtype=torch.uint8).to(torch.device('cuda:0'))

    # 遍历每个样本的data_size，更新相应的掩码行
    for i, size in enumerate(data_size):
        if size >= seq_length:
            mask[i, :] = 1  # 如果size >= seq_length，全为1
        else:
            mask[i, :size] = 1  # 否则，前size个位置为1，其余为0

    return mask

class GeoSAN(nn.Module):
    def __init__(self, n_loc, n_qdk, **kwargs):
        super().__init__()
        d_model = kwargs['d_model']
        max_len = kwargs['max_len']
        drop_ratio = kwargs['drop_ratio']
        depth = kwargs['depth']
        self.device = kwargs['device']
        self.emb_loc = nn.Embedding(n_loc, d_model, padding_idx=0)
        self.emb_qdk = nn.Embedding(n_qdk, d_model, padding_idx=0)
        self.emb_pos = PositionalEmbedding(max_len, d_model, drop_ratio)

        self.gps_encoder = Encoder(d_model, drop_ratio, depth)
        self.seq_encoder = Encoder(d_model * 2, drop_ratio, depth)

        self.drop = nn.Dropout(drop_ratio)
        self.out = nn.Linear(d_model * 2, n_loc)

    def forward(self, src_loc, src_qdk, tgt_loc, tgt_qdk, data_size):
        #torch.Size([512, 100])
        #torch.Size([512, 100])
        #torch.Size([512, 1])
        #torch.Size([512, 1])
        #torch.Size([512])
        batch_size = src_loc.size(0)
        seq_len = src_loc.size(1)
        src_loc_embedding = self.emb_pos.forward(self.emb_loc(src_loc)) #512*100*64
        tgt_loc_embedding = self.drop(self.emb_loc(tgt_loc))    #512*1*64


        src_qdk_embedding = self.emb_pos.forward(self.emb_qdk(src_qdk))
        src_qdk_embedding = self.gps_encoder.forward(src_qdk_embedding, None)
        #src_qdk_embedding = torch.mean(src_qdk_embedding, dim=2)
        tgt_qdk_embedding = self.emb_pos.forward(self.emb_qdk(tgt_qdk))
        tgt_qdk_embedding = self.gps_encoder.forward(tgt_qdk_embedding, None)
        #tgt_qdk_embedding = torch.mean(tgt_qdk_embedding, dim=2)

        src = torch.cat([src_loc_embedding, src_qdk_embedding], dim=-1)
        tgt = torch.cat([tgt_loc_embedding, tgt_qdk_embedding], dim=-1)

        src_mask = generate_mask(data_size, seq_len)

        src = self.seq_encoder.forward(src, src_mask)


        if self.training:
            output = self.out(src)
        else:
            output = src[torch.arange(data_size.size(0)), data_size - 1, :].detach()
            output = self.out(output)
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))