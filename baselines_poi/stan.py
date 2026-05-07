import numpy as np
import torch
from math import radians, cos, sin, asin, sqrt
import joblib
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.nn import functional as F
hours = 24*7

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def euclidean(point, each):
    lon1, lat1, lon2, lat2 = point[2], point[1], each[2], each[1]
    return np.sqrt((lon1 - lon2)**2 + (lat1 - lat2)**2)


def rst_mat1(traj, poi):
    # traj (*M, [u, l, t]), poi(L, [l, lat, lon])
    mat = np.zeros((len(traj), len(traj), 2))
    for i, item in enumerate(traj):
        for j, term in enumerate(traj):
            poi_item, poi_term = poi[item[1] - 1], poi[term[1] - 1]  # retrieve poi by loc_id
            mat[i, j, 0] = haversine(lon1=poi_item[2], lat1=poi_item[1], lon2=poi_term[2], lat2=poi_term[1])
            mat[i, j, 1] = abs(item[2] - term[2])
    return mat  # (*M, *M, [dis, tim])


def rs_mat2s(poi, l_max):
    # poi(L, [l, lat, lon])
    candidate_loc = np.linspace(1, l_max, l_max)  # (L)
    mat = np.zeros((l_max, l_max))  # mat (L, L)
    for i, loc1 in enumerate(candidate_loc):
        print(i) if i % 100 == 0 else None
        for j, loc2 in enumerate(candidate_loc):
            poi1, poi2 = poi[int(loc1) - 1], poi[int(loc2) - 1]  # retrieve poi by loc_id
            mat[i, j] = haversine(lon1=poi1[2], lat1=poi1[1], lon2=poi2[2], lat2=poi2[1])
    return mat  # (L, L)


def rt_mat2t(traj_time):  # traj_time (*M+1) triangle matrix
    # construct a list of relative times w.r.t. causality
    mat = np.zeros((len(traj_time)-1, len(traj_time)-1))
    for i, item in enumerate(traj_time):  # label
        if i == 0:
            continue
        for j, term in enumerate(traj_time[:i]):  # data
            mat[i-1, j] = np.abs(item - term)
    return mat  # (*M, *M)


class Attn(nn.Module):
    def __init__(self, emb_loc, loc_max, max_len, device, d_model):
        super(Attn, self).__init__()
        self.value = nn.Linear(max_len, 1, bias=False)
        self.emb_loc = emb_loc
        self.loc_max = loc_max
        self.device = device
        self.out = nn.Linear(d_model, loc_max)

    def forward(self, self_attn, self_delta, traj_len):
        '''
        # self_attn (N, M, emb), candidate (N, L, emb), self_delta (N, M, L, emb), len [N]
        self_delta = torch.sum(self_delta, -1).transpose(-1, -2)  # squeeze the embed dimension

        [N, L, M] = self_delta.shape

        candidates = torch.linspace(0, int(self.loc_max) - 1, int(self.loc_max)).long()  # (L)
        candidates = candidates.unsqueeze(0).expand(N, -1).to(self.device)  # (N, L)
        emb_candidates = self.emb_loc(candidates)  # (N, L, emb)
        attn = torch.mul(torch.bmm(emb_candidates, self_attn.transpose(-1, -2)), self_delta)  # (N, L, M)
        #transposed_tensor = attn.transpose(1, 2)
        #print(self.value.weight.dtype)  # 查看权重类型
        #print(attn.dtype)  # 查看 attn 的数据类型
        attn_out = self.value(attn.to(torch.float32)).view(N, L)  # 转换为 Double 类型
        '''
        N, M, emb = self_attn.shape
        self_delta_squeezed = self_delta.sum(-2)  # (N, M, emb)

        # 使用 torch.mul 逐元素相乘
        attn = torch.mul(self_attn, self_delta_squeezed)  # (N, M, emb)

        # 使用线性层映射到目标维度
        attn_out = self.out((attn.to(torch.float32)).view(N * M, -1))  # (N * M, n_loc)
        attn_out = attn_out.view(N, M, -1)  # (N, M, n_loc)
        return attn_out  # (N, L)


class SelfAttn(nn.Module):
    def __init__(self, emb_size, output_size, dropout=0.1):
        super(SelfAttn, self).__init__()
        self.query = nn.Linear(emb_size, output_size, bias=False)
        self.key = nn.Linear(emb_size, output_size, bias=False)
        self.value = nn.Linear(emb_size, output_size, bias=False)

    def forward(self, joint, delta, traj_len):
        delta = torch.sum(delta, -1)  # squeeze the embed dimension
        # joint (N, M, emb), delta (N, M, M, emb), len [N]
        # construct attention mask
        mask = torch.zeros_like(delta, dtype=torch.float32)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        attn = torch.add(torch.bmm(self.query(joint), self.key(joint).transpose(-1, -2)), delta)  # (N, M, M)
        attn = F.softmax(attn, dim=-1) * mask  # (N, M, M)
        joint = joint.float()
        attn = attn.float()
        attn_out = torch.bmm(attn, self.value(joint))  # (N, M, emb)

        return attn_out  # (N, M, emb)


def rs_row(poi, l_max, row_idx):
    """
    计算指定行的POI距离，并返回该行。
    """
    mat_row = np.zeros(l_max, dtype=np.float32)
    for j in range(l_max):
        print(row_idx)
        poi1 = poi[row_idx]  # 当前行的POI
        poi2 = poi[j]        # 当前列的POI
        mat_row[j] = haversine(lon1=poi1[1], lat1=poi1[0], lon2=poi2[1], lat2=poi2[0])
    return mat_row

class Embed(nn.Module):
    def __init__(self, ex, emb_size, loc_max, embed_layers):
        super(Embed, self).__init__()
        _, _, _, self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size
        self.loc_max = loc_max
        #self.idx_to_loc = idx_to_loc

    def forward(self, traj_loc, mat2, vec, traj_len):
        # traj_loc (N, M), mat2 (L, L), vec (N, M), delta_t (N, M, L)
        #print(vec.shape)
        delta_t = vec.unsqueeze(-1).expand(-1, -1, self.loc_max)
        #print(delta_t.shape)
        delta_s = torch.zeros_like(delta_t, dtype=torch.float32)
        mask = torch.zeros_like(delta_t, dtype=torch.long)
        for i in range(mask.shape[0]):  # N
            mask[i, 0:traj_len[i]] = 1
            delta_s[i, :traj_len[i]] = torch.index_select(mat2, 0, (traj_loc[i]-1)[:traj_len[i]])
            #row_idx = traj_loc[i] - 1  # 因为是从1开始的索引
            #mat_row = rs_row(self.idx_to_loc, self.loc_max, row_idx)  # 获取该行的距离
            #delta_s[i, :traj_len[i]] = torch.tensor(mat_row[:traj_len[i]], dtype=torch.float32)

        # pdb.set_trace()

        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        vsl, vsu, vtl, vtu = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)

        space_interval = (esl * vsu + esu * vsl) / (self.su - self.sl)
        time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl)
        delta = space_interval + time_interval  # (N, M, L, emb)

        return delta


class MultiEmbed(nn.Module):
    def __init__(self, ex, emb_size, embed_layers):
        super(MultiEmbed, self).__init__()
        self.emb_t, self.emb_l, self.emb_u, \
        self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size

    def forward(self, traj, mat, traj_len):
        # traj (N, M, 3), mat (N, M, M, 2), len [N]
        traj[:, :, 2] = (traj[:, :, 2]-1) % hours + 1  # segment time by 24 hours * 7 days
        time = self.emb_t(traj[:, :, 2])  # (N, M) --> (N, M, embed)
        loc = self.emb_l(traj[:, :, 1])  # (N, M) --> (N, M, embed)
        user = self.emb_u(traj[:, :, 0])  # (N, M) --> (N, M, embed)
        joint = time + loc + user  # (N, M, embed)
        delta_s, delta_t = mat[:, :, :, 0], mat[:, :, :, 1]  # (N, M, M)
        mask = torch.zeros_like(delta_s, dtype=torch.long)
        #print(mask.shape)
        for i in range(mask.shape[0]):
            #print("traj_len:", traj_len)
            #print("traj_len[i]:", traj_len[i])
            #print("traj_len dtype:", traj_len.dtype)
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        vsl, vsu, vtl, vtu = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)

        space_interval = (esl*vsu+esu*vsl) / (self.su-self.sl)
        time_interval = (etl*vtu+etu*vtl) / (self.tu-self.tl)
        delta = space_interval + time_interval  # (N, M, M, emb)

        return joint, delta

class STAN(nn.Module):
    def __init__(self, t_dim, l_dim, u_dim, ex, **kwargs):
        super(STAN, self).__init__()
        embed_dim = kwargs['d_model']
        max_len = kwargs['max_len']
        self.device = kwargs['device']
        emb_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
        emb_l = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        emb_u = nn.Embedding(u_dim, embed_dim, padding_idx=0)
        emb_su = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_sl = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tu = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tl = nn.Embedding(2, embed_dim, padding_idx=0)
        embed_layers = emb_t, emb_l, emb_u, emb_su, emb_sl, emb_tu, emb_tl

        self.MultiEmbed = MultiEmbed(ex, embed_dim, embed_layers)
        self.SelfAttn = SelfAttn(embed_dim, embed_dim)
        self.Embed = Embed(ex, embed_dim, l_dim, embed_layers)
        self.Attn = Attn(emb_l, l_dim, max_len, self.device, embed_dim)

    def forward(self, traj, mat1, mat2s, vec, traj_len):
        # long(N, M, [u, l, t]), float(N, M, M, 2), float(L, L), float(N, M), long(N)

        joint, delta = self.MultiEmbed.forward(traj, mat1, traj_len)  # (N, M, emb), (N, M, M, emb)

        self_attn = self.SelfAttn.forward(joint, delta, traj_len)  # (N, M, emb)

        self_delta = self.Embed.forward(traj[:, :, 1], mat2s, vec, traj_len)  # (N, M, L, emb)

        out = self.Attn.forward(self_attn, self_delta, traj_len)  # (N, L)


        if self.training:
            output = out
        else:
            output = out[torch.arange(traj_len.size(0)), traj_len - 1, :].detach()
        #print(output.shape)
        return output


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))