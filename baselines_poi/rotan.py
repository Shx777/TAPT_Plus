import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import copy


def rotate(head, relation, hidden, device):
    pi = 3.14159265358979323846

    re_head, im_head = torch.chunk(head, 2, dim=-1)
    #print(re_head.shape)
    # Make phases of relations uniformly distributed in [-pi, pi]
    embedding_range = nn.Parameter(
        torch.Tensor([(24.0 + 2.0) / hidden]),
        requires_grad=False
    ).to(device)

    phase_relation = relation / (embedding_range / pi)

    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation

    score = torch.cat([re_score, im_score], dim=1)
    return score
def rotate_batch(head, relation, hidden, device):
    pi = 3.14159265358979323846

    re_head, im_head = torch.chunk(head, 2, dim=2)

    # Make phases of relations uniformly distributed in [-pi, pi]
    embedding_range = nn.Parameter(
        torch.Tensor([(24.0 + 2.0) / hidden]),
        requires_grad=False
    ).to(device)

    phase_relation = relation / (embedding_range / pi)

    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation

    score = torch.cat([re_score, im_score], dim=2)
    return score

class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed

class TimeIntervalSim(nn.Module):
    def __init__(self, features) -> None:
        super(TimeIntervalSim, self).__init__()

        self.fea = features
        self.fn = nn.Linear(features, 1)

    def forward(self, time):
        return self.fn(time)

class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed

class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, poi_embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), dim=-1))
        x = self.leaky_relu(x)
        return x

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class OriginUserTime2Vec(nn.Module):
    def __init__(self, activation, out_dim, user_num):
        super(OriginUserTime2Vec, self).__init__()
        self.user_bias = nn.parameter.Parameter(torch.zeros(user_num, out_dim))
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x, user_idx):
        fea = x.view(-1, 1)
        pre = self.l1(fea)
        user_prefer = self.user_bias[user_idx]
        pre += user_prefer
        return pre


class OriginTime2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(OriginTime2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        fea = x.view(-1, 1)
        return self.l1(fea)


class Time2Vec(nn.Module):
    def __init__(self, out_dim) -> None:
        super(Time2Vec, self).__init__()
        self.w = nn.parameter.Parameter(torch.randn(1, out_dim))
        self.b = nn.parameter.Parameter(torch.randn(1, out_dim))
        self.f = torch.cos

    def forward(self, time):
        """_summary_

        Args:
            time (1d tensor): shape is seq_len

        Returns:
            time embeddings (2d tensor): shape is seq_len * time_dim
        """
        vec_time = time.view(-1, 1)
        out = torch.matmul(vec_time, self.w) + self.b
        v1 = out[:, 0].view(-1, 1)
        v2 = out[:, 1:]
        v2 = self.f(v2)
        return torch.cat((v1, v2), dim=-1)


class CatTime2Vec(nn.Module):
    def __init__(self, cat_num, out_dim) -> None:
        super(CatTime2Vec, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(cat_num, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(cat_num, 1))

        self.w = nn.parameter.Parameter(torch.randn(cat_num, out_dim - 1))
        self.b = nn.parameter.Parameter(torch.randn(cat_num, out_dim - 1))

    # cat_idx : s
    # norm_time : s
    def forward(self, cat_idx, norm_time):
        w = self.w[cat_idx]
        b = self.b[cat_idx]
        w0 = self.w0[cat_idx]
        b0 = self.b0[cat_idx]

        norm_time_ = norm_time.view(-1, 1)
        v1 = torch.sin(norm_time_ * w + b)
        v2 = norm_time_ * w0 + b0
        return torch.cat((v1, v2), dim=-1)


class RightPositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=600):
        super(RightPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, num_poi, embed_size,max_len, nhead, nhid, nlayers, target_time_dim, d_model, gps_d_model, time_embed_dim, dropout=0.5):
        super(TransformerModel, self).__init__()
        user_time_dim = int(0.5 * (d_model + d_model))
        self.user_embed_dim = d_model
        self.poi_embed_dim = d_model
        self.gps_embed_dim = gps_d_model
        self.time_embed_dim = time_embed_dim

        self.pos_encoder1 = RightPositionalEncoding(d_model + d_model, dropout, max_len)
        self.pos_encoder2 = RightPositionalEncoding(d_model + d_model, dropout, max_len)

        encoder_layers1 = TransformerEncoderLayer(d_model + d_model, nhead, nhid, dropout,
                                                  batch_first=True)
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, nlayers)

        encoder_layers2 = TransformerEncoderLayer(d_model + d_model, nhead, nhid, dropout,
                                                  batch_first=True)
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, nlayers)


        self.device = 'cuda:0'

        self.decoder_poi1 = nn.Linear(d_model + 2 * d_model, num_poi)
        self.decoder_poi2 = nn.Linear(d_model + 2 * gps_d_model, num_poi)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def attention_aggregation(self, src, target_time, traj_time):
        seq_len = src.shape[1]
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = ~mask
        mask = mask.to(self.device)

        out = self.attn(target_time, traj_time, traj_time, need_weights=True, attn_mask=mask)
        return out[1]

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi1.bias.data.zero_()
        self.decoder_poi1.weight.data.uniform_(-initrange, initrange)
        self.decoder_poi2.bias.data.zero_()
        self.decoder_poi2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src1, src2, src_mask, target_hour, target_day, poi_embeds, gps_embeds):
        src1 = src1 * math.sqrt(self.user_embed_dim + self.poi_embed_dim)
        src1 = self.pos_encoder1.forward(src1)
        src1 = self.transformer_encoder1(src1, src_mask)

        user_time_dim = int(0.5 * (self.user_embed_dim + self.poi_embed_dim))

        src1_hour = rotate_batch(src1, target_hour[:, :, :user_time_dim], user_time_dim, self.device)
        src1_day = rotate_batch(src1, target_day[:, :, :user_time_dim], user_time_dim, self.device)


        src1 = 0.7 * src1_hour + 0.3 * src1_day
        src1 = torch.cat((src1, poi_embeds), dim=-1)

        out_poi_prob1 = self.decoder_poi1(src1)

        src2 = src2 * math.sqrt(self.poi_embed_dim + self.gps_embed_dim)
        src2 = self.pos_encoder2(src2)
        src2 = self.transformer_encoder2(src2, src_mask)

        src2_hour = rotate_batch(src2, target_hour[:, :, user_time_dim:], 2 * self.time_embed_dim,
                                 self.device)
        src2_day = rotate_batch(src2, target_day[:, :, user_time_dim:], 2 * self.time_embed_dim, self.device)


        src2 = 0.7 * src2_hour + 0.3 * src2_day
        src2 = torch.cat((src2, gps_embeds), dim=-1)

        out_poi_prob2 = self.decoder_poi2(src2)

        out_poi_prob = 0.7 * out_poi_prob1 + 0.3 * out_poi_prob2

        return out_poi_prob


class PoiEmbeddings(nn.Module):
    def __init__(self, num_pois, embedding_dim):
        super(PoiEmbeddings, self).__init__()

        self.poi_embedding = nn.Embedding(
            num_embeddings=num_pois,
            embedding_dim=embedding_dim
        )

    def forward(self, poi_idx):
        embed = self.poi_embedding(poi_idx)
        return embed


class RotationTime(nn.Module):
    def __init__(self, dim) -> None:
        super(RotationTime, self).__init__()
        self.ln = nn.Linear(2, dim)

    def forward(self, time):
        c = torch.cos(time)
        s = torch.sin(time)
        t = torch.stack([c, s]).t()
        return self.ln(t)


class TimeEmbeddings(nn.Module):
    def __init__(self, num_times, embedding_dim) -> None:
        super(TimeEmbeddings, self).__init__()

        self.time_embedding = nn.Embedding(
            num_embeddings=num_times,
            embedding_dim=embedding_dim
        )

    def forward(self, time_idx):
        embed = self.time_embedding(time_idx)
        return embed


def clones(module, num_sub_layer):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_sub_layer)])


class GPSEmbeddings(nn.Module):
    def __init__(self, num_gps, embedding_dim) -> None:
        super(GPSEmbeddings, self).__init__()
        self.gps_embedding = nn.Embedding(
            num_embeddings=num_gps,
            embedding_dim=embedding_dim
        )

    def forward(self, gps_idx):
        embed = self.gps_embedding(gps_idx)
        return embed


class GPSEncoder(nn.Module):
    def __init__(self, embed_size, nhead, nhid, nlayers, dropout) -> None:
        super(GPSEncoder, self).__init__()
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.norm = nn.LayerNorm(embed_size)

    # s*l*d
    def forward(self, src):
        src = src * math.sqrt(self.embed_size)
        x = self.transformer_encoder(src)
        #x = torch.mean(x, -2)
        return self.norm(x)


class SimplePredict(nn.Module):
    def __init__(self, input_embed, out_embed) -> None:
        super(SimplePredict, self).__init__()
        self.trans = nn.Linear(input_embed, out_embed)

    def forward(self, src, candidate_poi_embeddings):
        src = self.trans(src)
        # print(f'src shape is {src.shape}')
        # print(f'candidate_poi_embeddings is {candidate_poi_embeddings.shape}')
        out_poi = torch.matmul(src, candidate_poi_embeddings.transpose(0, 1))

        return out_poi


class TimeEncoder(nn.Module):
    r"""
    This is a trainable encoder to map continuous time value into a low-dimension time vector.
    Ref: https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs/blob/master/module.py

    The input of ts should be like [E, 1] with all time interval as values.
    """

    def __init__(self, embedding_dim):
        super(TimeEncoder, self).__init__()
        self.time_dim = embedding_dim
        self.expand_dim = self.time_dim
        self.use_linear_trans = True

        self.basis_freq = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = nn.Parameter(torch.zeros(self.time_dim).float())
        if self.use_linear_trans:
            self.dense = nn.Linear(self.time_dim, self.expand_dim, bias=False)
            nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        if ts.dim() == 1:
            dim = 1
            edge_len = ts.size().numel()
        else:
            edge_len, dim = ts.size()
        ts = ts.view(edge_len, dim)
        map_ts = ts * self.basis_freq.view(1, -1)
        map_ts += self.phase.view(1, -1)
        harmonic = torch.cos(map_ts)
        if self.use_linear_trans:
            harmonic = harmonic.type(self.dense.weight.dtype)
            harmonic = self.dense(harmonic)
        return harmonic


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, input_size)
        self.layer2 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

class ROTAN(nn.Module):
    def __init__(self, num_users, num_pois, **kwargs):
        super(ROTAN, self).__init__()
        self.d_model = kwargs['d_model']
        self.time_embed_dim = kwargs['time_embed_dim']
        self.gps_embed_dim = kwargs['gps_embed_dim']
        self.device = 'cuda:0'
        self.max_len = kwargs['max_len']
        # Model1: User model
        self.user_embed_model = UserEmbeddings(num_users, self.d_model)
        self.poi_embed_model = PoiEmbeddings(num_pois, self.d_model)

        # Model2: Time Model
        self.time_embed_model_user = OriginTime2Vec('sin', int(0.5 * (self.d_model + self.d_model)))
        self.time_embed_model_user_tgt = OriginTime2Vec('sin', int(0.5 * (self.d_model + self.d_model)))

        self.time_embed_model_user_day = OriginTime2Vec('sin', int(0.5 * (self.d_model + self.d_model)))
        self.time_embed_model_user_day_tgt = OriginTime2Vec('sin', int(0.5 * (self.d_model + self.d_model)))

        self.time_embed_model_poi = OriginTime2Vec('sin', 2 * self.time_embed_dim)
        self.time_embed_model_poi_tgt = OriginTime2Vec('sin', 2 * self.time_embed_dim)

        self.time_embed_model_poi_day = OriginTime2Vec('sin', 2 * self.time_embed_dim)
        self.time_embed_model_poi_day_tgt = OriginTime2Vec('sin', 2 * self.time_embed_dim)

        # Model3: Geography model
        self.gps_embed_model = GPSEmbeddings(4096, self.gps_embed_dim)
        self.gps_encoder = GPSEncoder(self.gps_embed_dim, 1, 2 * self.gps_embed_dim, 2, 0.3)
        # Model4: Sequence model
        self.seq_input_embed = self.d_model + self.d_model
        self.seq_model = TransformerModel(num_pois, self.seq_input_embed, self.max_len, kwargs['transformer_nhead'],kwargs['transformer_nhid'],kwargs['transformer_nlayers'],self.time_embed_dim,self.d_model,
                                     self.gps_embed_dim, self.time_embed_dim, dropout=0.4)

    def forward(self, input_seq, data_size):
        # 解构 input_seq

        input_seq_users = input_seq[0].to(self.device)  #512

        input_seq_items = input_seq[1].to(self.device)  #512 * 100

        input_seq_time = input_seq[2].to(self.device)   #512 * 100

        input_seq_day_time = input_seq[3].to(self.device)   #512 * 100

        input_next_time = input_seq[4].to(self.device)  #512 * 100

        input_next_day_time = input_seq[5].to(self.device)  #512 * 100

        input_seq_gps = torch.tensor(input_seq[6]).to(self.device).long()


        batch_size, seq_length = input_seq_items.shape


        # GPS embeddings
        input_seq_gps_embeddings = self.gps_embed_model(input_seq_gps)

        input_seq_gps_embeddings = self.gps_encoder(input_seq_gps_embeddings)

        # User embeddings
        user_embedding = self.user_embed_model.forward(input_seq_users)
        user_embedding = torch.squeeze(user_embedding)
        #print(user_embedding.shape)  # 确保没有不小心去除掉的维度

        if user_embedding.dim() == 1:
            user_embedding = user_embedding.unsqueeze(0)  # 如果是单一维度，转化为 (1, 64)
        elif user_embedding.dim() == 2:
            # 如果已经是二维，无需改变，直接保持
            pass

        user_embedding_expanded = user_embedding.unsqueeze(1)
        user_embedding_repeated = user_embedding_expanded.repeat(1, seq_length, 1)


        # Time embeddings
        user_times = self.time_embed_model_user(input_seq_time)
        #user_times = user_times.view(batch_size, seq_length, self.d_model)
        user_day_times = self.time_embed_model_user_day(input_seq_day_time)
        #user_day_times = user_day_times.view(batch_size, seq_length, self.d_model)

        # POI embeddings
        poi_embeds = self.poi_embed_model(input_seq_items)


        # Next time embeddings
        user_next_times = self.time_embed_model_user_tgt(input_next_time)
        #print(user_next_times.shape)
        user_next_day_times = self.time_embed_model_user_day_tgt(input_next_day_time)

        # POI next time embeddings
        poi_next_times = self.time_embed_model_poi_tgt(input_next_time)
        poi_next_day_times = self.time_embed_model_poi_day_tgt(input_next_day_time)

        # POI time embeddings
        poi_times = self.time_embed_model_poi(input_seq_time)
        poi_day_times = self.time_embed_model_poi_day(input_seq_day_time)

        # Concatenate user and POI embeddings
        #print(user_embedding_repeated.shape)
        #print(poi_embeds.shape)
        user_embeddings = torch.cat((user_embedding_repeated, poi_embeds), dim=-1)
        batch_size, seq_length, embedding_dim = user_embeddings.shape
        user_embeddings = user_embeddings.view(batch_size * seq_length, embedding_dim)

        # User rotation embeddings
        user_rotate_hour = rotate(user_embeddings, user_times, int(0.5 * (self.d_model + self.d_model)), self.device)
        user_rotate_day = rotate(user_embeddings, user_day_times, int(0.5 * (self.d_model + self.d_model)), self.device)
        user_rotate = 0.7 * user_rotate_hour + 0.3 * user_rotate_day

        # Concatenate POI embeddings and GPS embeddings
        seq_poi_embeddings = torch.cat((poi_embeds, input_seq_gps_embeddings), dim=-1)
        batch_size, seq_length, embedding_dim = seq_poi_embeddings.shape
        seq_poi_embeddings = seq_poi_embeddings.view(batch_size * seq_length, embedding_dim)

        # POI rotation embeddings
        poi_rotate_hour = rotate(seq_poi_embeddings, poi_times, 2 * self.time_embed_dim, self.device)
        poi_rotate_day = rotate(seq_poi_embeddings, poi_day_times, 2 * self.time_embed_dim, self.device)
        poi_rotate = 0.7 * poi_rotate_hour + 0.3 * poi_rotate_day

        # Final sequence embeddings
        seq_embedding1 = user_rotate
        seq_embedding2 = poi_rotate
        seq_embedding3 = torch.cat((user_next_times, poi_next_times), dim=-1)
        seq_embedding4 = torch.cat((user_next_day_times, poi_next_day_times), dim=-1)


        seq_embedding1 = seq_embedding1.view(batch_size, seq_length, embedding_dim)
        seq_embedding2 = seq_embedding2.view(batch_size, seq_length, embedding_dim)
        seq_embedding3 = seq_embedding3.view(batch_size, seq_length, embedding_dim)
        seq_embedding4 = seq_embedding4.view(batch_size, seq_length, embedding_dim)

        src_mask = self.seq_model.generate_square_subsequent_mask(seq_embedding1.size(1)).to(self.device)

        y_pred_poi = self.seq_model.forward(seq_embedding1, seq_embedding2, src_mask, seq_embedding3, seq_embedding4, poi_embeds,input_seq_gps_embeddings)

        if self.training:
            output = y_pred_poi
        else:
            output = y_pred_poi[torch.arange(data_size.size(0)), data_size - 1, :].detach()

        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))