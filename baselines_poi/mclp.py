import math
import torch
import torch.nn as nn
from torch.nn import init
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim
        pos_encoding = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
        self.dropout = nn.Dropout(0.1)

    def forward(self, out):
        out = out + self.pos_encoding[:, :out.size(1)].detach()
        out = self.dropout(out)
        return out


class MyEmbedding(nn.Module):
    def __init__(self, n_loc, n_user, d_model):
        super(MyEmbedding, self).__init__()

        self.num_locations = n_loc
        self.base_dim = d_model
        self.num_users = n_user

        self.user_embedding = nn.Embedding(self.num_users, self.base_dim, padding_idx=0)
        self.location_embedding = nn.Embedding(self.num_locations, self.base_dim, padding_idx=0)
        self.timeslot_embedding = nn.Embedding(24, self.base_dim)

    def forward(self, location_x):

        loc_embedded = self.location_embedding(location_x)
        #print(self.num_users)
        user_embedded = self.user_embedding(torch.arange(end=self.num_users, dtype=torch.int, device=location_x.device))

        timeslot_embedded = self.timeslot_embedding(torch.arange(end=24, dtype=torch.int, device=location_x.device))

        return loc_embedded, timeslot_embedded, user_embedded

class TransEncoder(nn.Module):
    def __init__(self, d_model):
        super(TransEncoder, self).__init__()
        input_dim = d_model

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim,
                                                   activation='gelu',
                                                   batch_first=True,
                                                   dim_feedforward=input_dim,
                                                   nhead=4,
                                                   dropout=0.1)

        encoder_norm = nn.LayerNorm(input_dim)

        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=4,
                                             norm=encoder_norm)
        self.initialize_parameters()

    def forward(self, embedded_out, src_mask):
        out = self.encoder(embedded_out, mask=src_mask)

        return out

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)


class LSTMEncoder(nn.Module):
    def __init__(self, d_model):
        super(LSTMEncoder, self).__init__()
        input_dim = d_model

        self.encoder = nn.LSTM(input_size=input_dim,
                               hidden_size=input_dim,
                               num_layers=2,
                               dropout=0.1,
                               batch_first=True)
        self.initialize_parameters()

    def forward(self, out):
        out, _ = self.encoder(out)

        return out

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

class MyFullyConnect(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyFullyConnect, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim*2, input_dim),
            nn.Dropout(0.1),
        )

        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.drop = nn.Dropout(0.1)

        num_locations = output_dim
        self.linear_class1 = nn.Linear(input_dim, num_locations)

    def forward(self, out):
        x = out
        out = self.block(out)
        out = out + x
        out = self.batch_norm(out)
        out = self.drop(out)

        return self.linear_class1(out)

class ArrivalTime(nn.Module):
    def __init__(self, d_model, n_user, at_type):
        super(ArrivalTime, self).__init__()
        self.at_type = at_type
        self.base_dim = d_model
        self.num_heads = 4
        self.head_dim = self.base_dim // self.num_heads
        self.num_users = n_user
        self.timeslot_num = 24

        if at_type == 'attn':
            self.user_preference = nn.Embedding(self.num_users, self.base_dim, padding_idx=0)
            self.w_q = nn.ModuleList(
                [nn.Linear(self.base_dim + self.base_dim, self.head_dim) for _ in range(self.num_heads)])
            self.w_k = nn.ModuleList(
                [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
            self.w_v = nn.ModuleList(
                [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
            self.unify_heads = nn.Linear(self.base_dim, self.base_dim)

    def forward(self, timeslot_embedded, user_x1, hour_x1, timeslot_y1,hour_mask1 ):
        user_x = user_x1
        hour_x = hour_x1
        batch_size, sequence_length = hour_x.shape

        auxiliary_y = timeslot_y1
        hour_mask = hour_mask1.view(batch_size * sequence_length, -1)
        if self.at_type == 'truth':
            at_embedded = timeslot_embedded[auxiliary_y]
            return at_embedded
        if self.at_type == 'attn':
            hour_x = hour_x.view(batch_size * sequence_length)
            head_outputs = []
            user_preference = self.user_preference(user_x).unsqueeze(1).repeat(1, sequence_length, 1)
            user_feature = user_preference.view(batch_size * sequence_length, -1)
            time_feature = timeslot_embedded[hour_x]
            query = torch.cat([user_feature, time_feature], dim=-1)
            key = timeslot_embedded
            for i in range(self.num_heads):
                query_i = self.w_q[i](query)
                key_i = self.w_k[i](key)
                value_i = self.w_v[i](key)
                attn_scores_i = torch.matmul(query_i, key_i.T)
                scale = 1.0 / (key_i.size(-1) ** 0.5)
                attn_scores_i = attn_scores_i * scale
                attn_scores_i = attn_scores_i.masked_fill(hour_mask == 1, float('-inf'))
                attn_scores_i = torch.softmax(attn_scores_i, dim=-1)
                weighted_values_i = torch.matmul(attn_scores_i, value_i)
                head_outputs.append(weighted_values_i)
            head_outputs = torch.cat(head_outputs, dim=-1)
            head_outputs = head_outputs.view(batch_size, sequence_length, -1)
            return self.unify_heads(head_outputs)
        '''
        if self.at_type == 'static':
            time_trans_prob_mat = batch_data['prob_matrix_time_individual']
            at_embedded_user = torch.matmul(time_trans_prob_mat, timeslot_embedded)
            batch_indices = torch.arange(batch_size).view(-1, 1).expand_as(hour_x)
            at_embedded = at_embedded_user[batch_indices, hour_x, :]
            return at_embedded
        '''
class UserNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UserNet, self).__init__()
        self.topic_num = input_dim
        self.output_dim = output_dim

        self.block = nn.Sequential(
            nn.Linear(self.topic_num, self.topic_num * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.topic_num * 2, self.topic_num),
        )
        self.final = nn.Sequential(
            nn.LayerNorm(self.topic_num),
            nn.Linear(self.topic_num, self.output_dim)
        )

    def forward(self, topic_vec):
        x = topic_vec
        topic_vec = self.block(topic_vec)
        topic_vec = x + topic_vec

        return self.final(topic_vec)

class MCLP(nn.Module):
    def __init__(self, n_loc, n_user=1032, **kwargs):
        super(MCLP, self).__init__()
        self.d_model = kwargs['d_model']
        self.max_len = kwargs['max_len']
        self.topic_num = kwargs['topic_num']
        self.encoder_type = kwargs['encoder_type']
        self.at_type = kwargs['at_type']
        self.device = 'cuda:0'
        self.embedding_layer = MyEmbedding(n_loc, n_user, self.d_model)

        if self.encoder_type == 'trans':
            emb_dim = self.d_model
            self.positional_encoding = PositionalEncoding(emb_dim=emb_dim, max_len=self.max_len)
            self.encoder = TransEncoder(self.d_model)
        if self.encoder_type == 'lstm':
            self.encoder = LSTMEncoder(self.d_model)

        fc_input_dim = self.d_model + self.d_model

        if self.at_type != 'none':
            self.at_net = ArrivalTime(self.d_model, n_user, self.at_type)
            fc_input_dim += self.d_model

        if self.topic_num > 0:
            self.user_net = UserNet(input_dim=self.topic_num, output_dim=self.d_model)
            fc_input_dim += self.d_model

        self.fc_layer = MyFullyConnect(input_dim=fc_input_dim,
                                       output_dim=n_loc)
        self.out_dropout = nn.Dropout(0.1)

    def forward(self, batch_data, data_size):
        user_x = batch_data['user'].to(self.device)
        loc_x = batch_data['location_x'].to(self.device)
        hour_x = batch_data['hour'].to(self.device)
        timeslot_y = batch_data['timeslot_y'].to(self.device)
        hour_mask = batch_data['hour_mask'].to(self.device)

        if self.topic_num > 0:
            pre_embedded = batch_data['user_topic_loc'].to(self.device)
        batch_size, sequence_length = loc_x.shape

        loc_embedded, timeslot_embedded, user_embedded = self.embedding_layer.forward(loc_x)

        time_embedded = timeslot_embedded[hour_x]

        lt_embedded = loc_embedded + time_embedded

        if self.encoder_type == 'trans':
            future_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(lt_embedded.device)
            future_mask = future_mask.masked_fill(future_mask == 1, float('-inf')).bool()
            encoder_out = self.encoder.forward(self.positional_encoding.forward(lt_embedded * math.sqrt(self.d_model)),src_mask=future_mask)
        if self.encoder_type == 'lstm':
            encoder_out = self.encoder.forward(lt_embedded)
        combined = encoder_out + lt_embedded

        user_embedded = user_embedded[user_x]

        if self.at_type != 'none':
            at_embedded = self.at_net.forward(timeslot_embedded, user_x, hour_x, timeslot_y,hour_mask )
            combined = torch.cat([combined, at_embedded], dim=-1)

        user_embedded = user_embedded.unsqueeze(1).repeat(1, sequence_length, 1)
        combined = torch.cat([combined, user_embedded], dim=-1)

        if self.topic_num > 0:
            pre_embedded_be = self.user_net.forward(pre_embedded.float()).squeeze(0)
            if pre_embedded_be.dim() == 1:
                pre_embedded_be = pre_embedded_be.unsqueeze(0)  # 如果是单一维度，转化为 (1, 64)
            elif pre_embedded_be.dim() == 2:
                # 如果已经是二维，无需改变，直接保持
                pass
            pre_embedded = pre_embedded_be.unsqueeze(1).repeat(1, sequence_length, 1)

            #print(combined.shape)
            #print(pre_embedded.shape)
            combined = torch.cat([combined, pre_embedded], dim=-1)

        #mixer_output = combined.view(batch_size * sequence_length, combined.shape[2])

        if self.training:
            mixer_output = combined.view(batch_size * sequence_length, combined.shape[2])
            output = self.fc_layer.forward(mixer_output)
        else:
            output = combined[torch.arange(data_size.size(0)), data_size - 1, :].detach()
            output = self.fc_layer.forward(output)
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))