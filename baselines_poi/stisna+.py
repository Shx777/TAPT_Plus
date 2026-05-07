import copy
import torch.nn as nn
import math
import torch
import torch.nn.functional as F

def clones(module, num_sub_layer):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_sub_layer)])


class SubLayerConnect(nn.Module):
    def __init__(self, features):
        super(SubLayerConnect, self).__init__()
        self.norm = nn.LayerNorm(features)

    def forward(self, x, sublayer):
        # (*, d)
        return x + sublayer(self.norm(x))

class FFN(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(features, exp_factor * features)
        self.act = nn.ReLU()
        self.w_2 = nn.Linear(exp_factor * features, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x

class InrAttn(nn.Module):
    def __init__(self, dropout):
        super(InrAttn, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, str_mat, attn_mask):
        scale_term = math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale_term
        str_mat = str_mat.masked_fill(attn_mask == 0.0, -1e9)
        str_mat = F.softmax(str_mat, dim=-1)
        scores += str_mat
        if attn_mask is not None:
            scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        return torch.matmul(prob, value)


class MHInrAttn(nn.Module):
    def __init__(self, features, n_head, dropout):
        super(MHInrAttn, self).__init__()
        self.d_h = features // n_head
        self.n_head = n_head
        self.linears = clones(nn.Linear(features, features), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, str_mat, attn_mask):
        b = x.size(0)
        query, key, value = [l(x).view(b, self.h, -1, self.d_h) for l, x in zip(self.linears, x)]
        scale_term = query.size(-1)
        str_mat = str_mat.masked_fill(attn_mask == 0.0, -1e9)
        str_mat = F.softmax(str_mat, dim=-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale_term + str_mat
        if attn_mask is not None:
            scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        x = torch.matmul(prob, value)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h*self.d_h)
        return self.linears[-1](x)

class RoTAPE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 4 == 0
        # [d/4]
        self.theta = torch.exp(torch.arange(0, d_model, 4, device='cuda:0') * -(math.log(10000.0) / d_model))

    def get_position(self, t_d, t_h):
        # [b, n, d/4]
        pos_day = t_d.unsqueeze(-1) * self.theta
        pos_hour = t_h.unsqueeze(-1) * self.theta
        cos_day = torch.cos(pos_day).unsqueeze(1)
        sin_day = torch.sin(pos_day).unsqueeze(1)
        cos_hour = torch.cos(pos_hour).unsqueeze(1)
        sin_hour = torch.sin(pos_hour).unsqueeze(1)
        return cos_day, sin_day, cos_hour, sin_hour

    def forward(self, x, t_d, t_h):
        cos_day, sin_day, cos_hour, sin_hour = self.get_position(t_d, t_h)
        # [b, 1, n, d/4]
        x_1, x_2, x_3, x_4 = x[..., 0::4], x[..., 1::4], x[..., 2::4], x[..., 3::4]
        x_day_1 = x_1 * cos_day + x_2 * sin_day
        x_day_2 = x_2 * cos_day - x_1 * sin_day
        x_hour_3 = x_3 * cos_hour + x_4 * sin_hour
        x_hour_4 = x_4 * cos_hour - x_3 * sin_hour
        x = torch.cat([x_day_1, x_day_2, x_hour_3, x_hour_4], dim=-1)
        return x

class STRMemoryLayer(nn.Module):
    def __init__(self, src_len, trg_len, dropout):
        super(STRMemoryLayer, self).__init__()
        self.mem_v = FFN(trg_len, src_len, dropout)
        self.mem_h = FFN(src_len, trg_len, dropout)
        self.sublayer_1 = SubLayerConnect(trg_len)
        self.sublayer_2 = SubLayerConnect(src_len)

    def forward(self, x):
        # [b, n, k] -> [b, n, k*n] -> [b, n, k]
        x = self.sublayer_1.forward(x, self.mem_v)
        # [b, k, n]
        x = x.transpose(-2, -1)
        # [b, k, n] -> [b, k, k*n] -> [b, k, n]
        x = self.sublayer_2.forward(x, self.mem_h)
        # [b, n, k]
        x = x.transpose(-2, -1)
        return x

class STRMemory(nn.Module):
    def __init__(self, features, layer, depth):
        super(STRMemory, self).__init__()
        self.layers = clones(layer, depth)
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        # [b, n, k]
        for layer in self.layers:
            x = layer(x)
        # [b, k, n]
        x = x.transpose(-2, -1)
        x = self.norm(x)
        return x