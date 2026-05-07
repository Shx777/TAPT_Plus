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
        x = x + lookup_table.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = self.dropout(x)
        return x

# SASRec Backbone

class TAPT_PLUS(nn.Module):
    def __init__(self, n_loc, **kwargs):
        super(TAPT_PLUS, self).__init__()

        # backbone sasrec 所需参数
        d_model = kwargs['d_model']
        max_len = kwargs['max_len']
        p_drop = kwargs['p_drop']
        depth = kwargs['depth']
        n_head = kwargs['n_head']
        exp_factor = kwargs['exp_factor']

        # tapt_plus 额外所需参数
        d_time = kwargs['d_time']
        d_all = kwargs['d_all']
        s_experts = kwargs['s_experts']
        p_experts = kwargs['p_experts']
        t_experts = kwargs['t_experts']
        num_layers = kwargs['num_layers']

        self.emb_loc = nn.Embedding(n_loc, d_model, padding_idx=0)
        self.emb_pos = PositionalEmbedding(max_len, d_model, p_drop)
        self.mixer = Transformer(d_model, n_head, exp_factor, p_drop, depth)
        self.out = nn.Linear(d_model, n_loc)

        # --- JRL embeddings ---
        self.hour_emb = nn.Embedding(24, d_time // 3)
        self.min_emb = nn.Embedding(60, d_time // 3)
        self.sec_emb = nn.Embedding(60, d_time // 3)
        self.joint_fc = nn.Linear(d_model + d_time, d_all)
        self.relu = nn.ReLU()

        # --- PLE experts ---
        self.shared_experts = nn.ModuleList([nn.Linear(d_all, d_all) for _ in range(s_experts)])
        self.poi_experts = nn.ModuleList([nn.Linear(d_all, d_all) for _ in range(p_experts)])
        self.time_experts = nn.ModuleList([nn.Linear(d_all, d_all) for _ in range(t_experts)])

        # Task-specific gates
        self.gate_p = nn.Linear(d_all, s_experts + p_experts)
        self.gate_t = nn.Linear(d_all, s_experts + t_experts)

        # Task heads
        self.poi_head = nn.Linear(d_all, d_model)

        # time decoding
        self.time_proj = nn.Linear(d_model, d_time)

        self.hour_head = nn.Linear(d_time // 3, 24)
        self.min_head = nn.Linear(d_time // 3, 60)
        self.sec_head = nn.Linear(d_time // 3, 60)

        # Dynamic Gating Network
        self.dgn_fc1 = nn.Linear(2 * d_all, d_all)
        self.dgn_fc2 = nn.Linear(d_all, 2)

    def forward(self, poi_seq, time_seq, data_size):
        poi_emb = self.emb_loc(poi_seq)

        h_e = self.hour_emb(time_seq[:, :, 0])
        m_e = self.min_emb(time_seq[:, :, 1])
        s_e = self.sec_emb(time_seq[:, :, 2])
        time_emb = torch.cat([h_e, m_e, s_e], dim=-1)  # [B, L, dt]

        joint_emb = self.relu(self.joint_fc(torch.cat([poi_emb, time_emb], dim=-1)))  # [B, L, d]

        #print(joint_emb.shape)

        x = self.emb_pos(joint_emb)
        mask = get_mask(poi_seq, bidirectional=False)
        E = self.mixer(x, mask)

        shared_outs = torch.stack([F.relu(expert(E)) for expert in self.shared_experts], dim=1)  # [B, s, L, d]
        poi_outs = torch.stack([F.relu(expert(E)) for expert in self.poi_experts], dim=1)  # [B, p, L, d]
        time_outs = torch.stack([F.relu(expert(E)) for expert in self.time_experts], dim=1)  # [B, t, L, d]

        Ep = torch.cat([shared_outs, poi_outs], dim=1)  # [B, s+p, L, d]
        Et = torch.cat([shared_outs, time_outs], dim=1)  # [B, s+t, L, d]

        # print(Ep.shape)
        # print(Et.shape)

        # --- Task-specific gating ---
        last_e = E[:, -1, :]  # [B, d], use last position for gating

        g_p = F.softmax(self.gate_p(last_e), dim=-1)  # [B, s+p]
        g_t = F.softmax(self.gate_t(last_e), dim=-1)  # [B, s+t]

        # print(g_p.shape)
        # print(g_t.shape)

        # Weighted sum over experts at last position
        # h_p = torch.einsum('bp,bpld->bd', g_p, Ep[:, :, -1, :])  # [B, d]
        # h_t = torch.einsum('bt,btld->bd', g_t, Et[:, :, -1, :])  # [B, d]

        h_p = torch.einsum('bp,bpld->bld', g_p, Ep)  # [512, 100, 64]
        h_t = torch.einsum('bp,bpld->bld', g_t, Et)  # [512, 100, 64]

        # print(h_p.shape)
        # print(h_t.shape)

        # --- Task heads ---
        zp = self.poi_head(h_p)  # [B, dp]
        t_feat = self.time_proj(h_t)  # [B, d_time]
        t_h, t_m, t_s = torch.chunk(t_feat, 3, dim=-1)
        pred_h = self.hour_head(t_h)  # [B, 24]
        pred_m = self.min_head(t_m)  # [B, 60]
        pred_s = self.sec_head(t_s)  # [B, 60]

        p_h = F.softmax(pred_h, dim=-1)  # [B, 24]
        p_m = F.softmax(pred_m, dim=-1)  # [B, 60]
        p_s = F.softmax(pred_s, dim=-1)  # [B, 60]
        hour_idx = torch.arange(24, device=p_h.device).float()  # [24]
        min_idx = torch.arange(60, device=p_m.device).float()  # [60]
        sec_idx = torch.arange(60, device=p_s.device).float()  # [60]
        hour_hat = (p_h * hour_idx).sum(dim=-1)  # [B]
        min_hat = (p_m * min_idx).sum(dim=-1)  # [B]
        sec_hat = (p_s * sec_idx).sum(dim=-1)  # [B]
        yt_hat = hour_hat + min_hat / 60.0 + sec_hat / 3600.0  # [B]


        # --- Dynamic Gating Network ---
        z_joint = torch.cat([h_p, h_t], dim=-1)  # [B, 2d]
        gate = F.softmax(self.dgn_fc2(F.tanh(self.dgn_fc1(z_joint))), dim=-1)  # [B, 2]

        # print(zp.shape) # 预测POI的embedding   torch.Size([512, 100, 64])
        # print(yt_hat.shape) # 预测时间戳  torch.Size([512, 100])
        # print(gate.shape)   # 每个位置的loss dgn torch.Size([512, 100, 2])

        if self.training:
            zp = self.out(zp)

            return zp, yt_hat, gate
        else:
            idx = torch.arange(data_size.size(0), device=data_size.device)
            last_idx = data_size - 1

            zp = zp[idx, last_idx, :].detach()
            yt_hat = yt_hat[idx, last_idx, ...].detach()
            gate = gate[idx, last_idx, :].detach()

            zp = self.out(zp)

            # print(zp.shape)
            # print(yt_hat.shape)
            # print(gate.shape)

            return zp, yt_hat, gate

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))