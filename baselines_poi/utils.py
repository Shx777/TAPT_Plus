from torch.utils.data import Sampler
from torch.backends import cudnn
import datetime
from tqdm import tqdm
from gensim import corpora, models
from itertools import product
import math
from nltk import ngrams
from collections import defaultdict
from datetime import datetime
import torch.nn as nn
import os
import re
import glob
import torch
import random
import collections
import numpy as np
import pandas as pd

SSTBatch = collections.namedtuple(
    "SSTBatch", ["graph", "features", "time", "label", "mask", "mask2", "type"]
)
EarthRadius = 6378137
MinLatitude = -85.05112878
MaxLatitude = 85.05112878
Minlnggitude = -180
Maxlnggitude = 180

#33333333333333333

def clip(n, minValue, maxValue):
    return min(max(n, minValue), maxValue)

def map_size(levelOfDetail):
    return 256 << levelOfDetail

def latlon2pxy(latitude, longitude, levelOfDetail):
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    longitude = clip(longitude, Minlnggitude, Maxlnggitude)

    x = (longitude + 180) / 360
    sinLatitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)

    mapSize = map_size(levelOfDetail)
    pixelX = int(clip(x * mapSize + 0.5, 0, mapSize - 1))
    pixelY = int(clip(y * mapSize + 0.5, 0, mapSize - 1))
    return pixelX, pixelY

def txy2quadkey(tileX, tileY, levelOfDetail):
    quadKey = []
    for i in range(levelOfDetail, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tileX & mask) != 0:
            digit += 1
        if (tileY & mask) != 0:
            digit += 2
        quadKey.append(str(digit))

    return ''.join(quadKey)

def pxy2txy(pixelX, pixelY):
    tileX = pixelX // 256
    tileY = pixelY // 256
    return tileX, tileY

def latlon2quadkey(lat,lon,level):
    pixelX, pixelY = latlon2pxy(lat, lon, level)
    tileX, tileY = pxy2txy(pixelX, pixelY)
    return txy2quadkey(tileX, tileY,level)



def pad_sequence(seq, max_len):
    seq = list(seq)
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[-max_len:]

    return torch.tensor(seq)

def pad_sequence_2d(seq, max_len):

    seq = list(seq)  # 确保输入是列表
    feature_dim = len(seq[0])  # 获取每个数据点的维度

    if len(seq) < max_len:
        # 如果序列长度小于 max_len，则进行填充
        padding_elem = [0] * feature_dim  # 用零向量填充
        seq.extend([padding_elem] * (max_len - len(seq)))
    else:
        # 如果序列长度大于 max_len，则进行截断
        seq = seq[-max_len:]
    #print(torch.tensor(seq).shape)
    return torch.tensor(seq)

def get_all_permutations_dict(length):
    characters = ['0', '1', '2', '3']

    # 生成所有可能的长度为6的字符串
    all_permutations = [''.join(p) for p in product(characters, repeat=length)]

    premutation_dict = dict(zip(all_permutations,range(len(all_permutations))))

    return premutation_dict


def get_ngrams_of_quadkey(quadkey, n, permutations_dict):
    region_quadkey_bigram = ' '.join([''.join(x) for x in ngrams(quadkey, n)])
    region_quadkey_bigram = region_quadkey_bigram.split()
    region_quadkey_bigram = [permutations_dict[each] for each in region_quadkey_bigram]
    return sum(region_quadkey_bigram) / len(region_quadkey_bigram)  # 取均值

def add_true_node(tree, trajectory, index, parent_node_id, nary):
    for i in range(nary - 1, 0, -1):
        if index - i >= 0:
            node_id = tree.number_of_nodes()
            node = trajectory[index - i]
            tree.add_node(node_id, x=node['features'], time=node['time'], y=node['labels'], mask=1, mask2=0, type=2)
            tree.add_edge(node_id, parent_node_id)
        else:  # empty node
            node_id = tree.number_of_nodes()
            tree.add_node(node_id, x=[0] * 3, time=0, y=[-1] * 2, mask=0, mask2=0, type=-1)
            tree.add_edge(node_id, parent_node_id)

    sub_parent_node_id = tree.number_of_nodes()
    tree.add_node(sub_parent_node_id, x=[0] * 3, time=0, y=[-1] * 2, mask=0, mask2=0, type=-1)
    tree.add_edge(sub_parent_node_id, parent_node_id)

    if index - (nary - 1) > 0:
        add_true_node(tree, trajectory, index - (nary - 1), sub_parent_node_id, nary)
        tree.add_node(sub_parent_node_id, x=[0] * 3, time=0, y=trajectory[index - (nary - 1)]['labels'], mask=0,
                      mask2=0, type=-1)


def add_period_node(tree, trajectory, nary):
    node_id = tree.number_of_nodes()
    period_label = trajectory[len(trajectory) - 1]['labels'] if len(trajectory) > 0 else [-1] * 2
    tree.add_node(node_id, x=[0] * 3, time=0, y=period_label, mask=0, mask2=1, type=1)

    if len(trajectory) > 0:
        add_true_node(tree, trajectory, len(trajectory), node_id, nary)

    return node_id


def add_day_node(tree, trajectory, labels, index, nary):
    node_id = tree.number_of_nodes()
    tree.add_node(node_id, x=[0] * 3, time=0, y=labels[index], mask=0, mask2=1, type=0)
    if index > 0:  # recursion
        child_node_id = add_day_node(tree, trajectory, labels, index - 1, nary)
        tree.add_edge(child_node_id, node_id)
    else:
        fake_node_id = tree.number_of_nodes()
        tree.add_node(fake_node_id, x=[0] * 3, time=0, y=[-1] * 2, mask=0, mask2=0, type=-1)
        tree.add_edge(fake_node_id, node_id)

    day_trajectory = trajectory[index]
    for i in range(len(day_trajectory)):  # Four time periods， 0-6， 6-12， 12-18， 18-24
        period_node_id = add_period_node(tree, day_trajectory[i], nary)
        tree.add_edge(period_node_id, node_id)

    return node_id


def construct_MobilityTree(trajectory, labels, nary):
    tree = nx.DiGraph()

    add_day_node(tree, trajectory, labels, len(trajectory) - 1, nary)
    dgl_tree = dgl.from_networkx(tree, node_attrs=['x', 'time', 'y', 'mask', 'mask2', 'type'])
    return dgl_tree

def process_hour_mask(hour_mask, max_len):

    # 将 hour_mask 转为 numpy 数组，如果它是 list
    hour_mask = np.array(hour_mask)

    # 如果 hour_mask 长度小于 max_len，进行填充
    if len(hour_mask) < max_len:
        padded_hour_mask = np.zeros((max_len, 24), dtype=np.int32)  # 创建填充的数组
        padded_hour_mask[:len(hour_mask)] = hour_mask  # 复制现有的 hour_mask 值
    # 如果 hour_mask 长度大于 max_len，进行截断
    else:
        padded_hour_mask = hour_mask[-max_len:]  # 截取最近的 max_len 个元素
        # 如果截断后的数据维度不是 (max_len, 24)，需要补充形状
        if padded_hour_mask.shape[0] < max_len:
            padded_hour_mask = np.pad(padded_hour_mask, ((max_len - padded_hour_mask.shape[0]), (0, 0)),
                                      mode='constant', constant_values=0)

    return padded_hour_mask

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class LadderSampler(Sampler):
    def __init__(self, data_source, batch_size, fix_order=False):
        super(LadderSampler, self).__init__(data_source)
        self.data = [len(e[0]) for e in data_source]
        self.batch_size = batch_size * 100
        self.fix_order = fix_order

    def __iter__(self):
        if self.fix_order:
            d = zip(self.data, np.arange(len(self.data)), np.arange(len(self.data)))
        else:
            d = zip(self.data, np.random.permutation(len(self.data)), np.arange(len(self.data)))
        d = sorted(d, key=lambda e: (e[1] // self.batch_size, e[0]), reverse=True)
        return iter(e[2] for e in d)

    def __len__(self):
        return len(self.data)

import numpy as np
import torch
from math import radians, cos, sin, asin, sqrt
import joblib

max_len = 100  # max traj len; i.e., M


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
    #print(len(traj))
    mat = np.zeros((len(traj), len(traj), 2), dtype=np.float32)
    #print(traj)
    for i, item in enumerate(traj):
        for j, term in enumerate(traj):
            poi_item, poi_term = poi[item[1]-1], poi[term[1]-1]  # retrieve poi by loc_id
            mat[i, j, 0] = haversine(lon1=poi_item[1], lat1=poi_item[0], lon2=poi_term[1], lat2=poi_term[0])
            mat[i, j, 1] = abs(item[2] - term[2])
    return mat  # (*M, *M, [dis, tim])

from scipy.sparse import lil_matrix


def haversine_gpu(lat1, lon1, lat2, lon2):
    # 将角度转换为弧度
    pi_div_180 = torch.tensor(3.141592653589793 / 180, device=lat1.device)
    lat1, lon1, lat2, lon2 = lat1 * pi_div_180, lon1 * pi_div_180, lat2 * pi_div_180, lon2 * pi_div_180

    # Haversine公式的实现
    dlat = lat2 - lat1  # 纬度差
    dlon = lon2 - lon1  # 经度差
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    R = 6371  # 地球半径，单位：公里
    return R * c


# 修改后的 GPU 计算矩阵函数
def rs_mat2s(poi, l_max, device="cuda"):
    # 确保 poi 是正确格式，例如 [[lat1, lon1], [lat2, lon2], ...]
    if isinstance(poi, dict):
        poi = [poi[idx] for idx in range(0, l_max)]  # 提取字典的值，并转换为列表

    # 转换为 PyTorch Tensor
    poi = torch.tensor(poi, dtype=torch.float16, device=device)  # (L, [lat, lon])
    # 获取所有点对的纬度和经度
    lat = poi[:, 0]  # (L,)
    lon = poi[:, 1]  # (L,)

    # 广播纬度和经度用于矩阵计算
    lat1, lat2 = lat.unsqueeze(1), lat.unsqueeze(0)  # (L, 1), (1, L)
    lon1, lon2 = lon.unsqueeze(1), lon.unsqueeze(0)  # (L, 1), (1, L)

    # 计算距离矩阵
    mat = haversine_gpu(lat1, lon1, lat2, lon2)  # (L, L)
    cpu_tensor = mat.cpu()
    return cpu_tensor  # 返回 GPU 上的张量

#aaaaaaaaaaaaaaaaaaaaaa

def pad_or_truncate_sequences(seq, max_len, padding_value=[0, 1, 0]):

    seq = list(seq)

    if len(seq) < max_len:
        # 如果序列长度不足，则用元组填充
        seq = seq + ([padding_value] * (max_len - len(seq)))
    elif len(seq) > max_len:
        # 如果序列长度超出，则截取
        seq = seq[-max_len:]
    else:
        # 如果序列长度正好，则不处理
        seq = seq

    return seq


def rt_mat2t(traj_time):  # traj_time (*M+1) triangle matrix
    #print(traj_time.shape)
    # construct a list of relative times w.r.t. causality
    mat = np.zeros((len(traj_time), len(traj_time)), dtype=np.float32)
    for i, item in enumerate(traj_time):  # label
        if i == 0:
            continue
        for j, term in enumerate(traj_time[:i]):  # data
            mat[i-1, j] = np.abs(item - term)
    return mat  # (*M, *M)
def gen_train_batch(batch, data_source, max_len, n_poi, n_user, poi, mat2s):
    '''
    src_seq, trg_seq = zip(*batch)
    items, data_size, qdk = [], [], []

    batch_data = {
        'user': [],
        'location_x': [],
        'hour': [],
        'hour_mask': [],
        'timeslot_y': [],
        'user_topic_loc': []
    }

    # users, src_time, src_day_time, src_gps, data_size = [], [], [], [], []
    for e in src_seq:
        # user_idx, poi_index, timestamp, location, True
        u_, i_, t_, l_, _ = zip(*e)

        batch_data['user'].append(u_[0])
        batch_data['location_x'].append(pad_sequence(i_, max_len).tolist())
        batch_data['hour'].append(pad_sequence(datetime_to_features1(t_), max_len).tolist())
        occur_time_user = occur_time_individual[u_[0]]
        # Generate hour_mask
        hour_mask = []
        for timestamp in t_:
            mask = np.zeros(24, dtype=np.int32)
            mask[occur_time_user == 0] = 1
            hour_mask.append(mask)
        # Padding hour_mask to [sequence_length, 24]
        padded_hour_mask = process_hour_mask(hour_mask, max_len)
        batch_data['hour_mask'].append(padded_hour_mask)


        users.append(u_[0])
        src_time.append(pad_sequence(t_, max_len))
        src_day_time.append(pad_sequence(t_, max_len))
        src_gps.append(pad_sequence_2d(l_, max_len))

        items.append(pad_sequence(i_, max_len))
        data_size.append(len(_))
        region_idx_tuple = tuple(location_to[loc] if loc in location_to else -1 for loc in l_)
        qdk.append(pad_sequence(region_idx_tuple, max_len))

    for e in trg_seq:
        # user_idx, poi_index, timestamp, location, True
        _, i_, t_, _, _ = zip(*e)
        batch_data['timeslot_y'].append(pad_sequence(datetime_to_features1(t_), max_len).tolist())

    batch_data['user'] = torch.tensor(batch_data['user'])
    batch_data['location_x'] = torch.tensor(batch_data['location_x'])
    batch_data['hour'] = torch.tensor(batch_data['hour'])
    hour_mask_array = np.array(batch_data['hour_mask'])
    batch_data['hour_mask'] = torch.tensor(hour_mask_array)
    batch_data['timeslot_y'] = torch.tensor(batch_data['timeslot_y'])
    selected_user_topics = user_topics[batch_data['user']]
    if topic_num > 0:
        batch_data['user_topic_loc'] = torch.tensor(selected_user_topics)

    src_items = torch.stack(items)
    data_size = torch.tensor(data_size)
    src_qdk = torch.stack(qdk)
    items, trg_time, trg_day_time, qdk = [], [], [], []
    for e in trg_seq:
        # user_idx, poi_index, timestamp, location, True
        u_, i_, t_, l_, _ = zip(*e)
        items.append(pad_sequence(i_, max_len))
        # trg_time.append(pad_sequence(t_, max_len))
        # trg_day_time.append(pad_sequence(t_, max_len))
        region_idx_tuple = tuple(location_to[loc] if loc in location_to else -1 for loc in l_)
        qdk.append(pad_sequence(region_idx_tuple, max_len))

    src_quad_keys = []
    permutations_dict = get_all_permutations_dict(6)
    for gps_seq in src_gps:
        quad_keys = [latlng2quadkey(loc[0], loc[1], 25) for loc in gps_seq]
        quad_keys = [get_ngrams_of_quadkey(quad_key, 6, permutations_dict) for quad_key in quad_keys]
        src_quad_keys.append(quad_keys)

    # users_tensors = [torch.tensor(u) for u in users]
    # users = torch.stack(users_tensors)
    # src_time = torch.stack(src_time)
    # src_day_time = torch.stack(src_day_time)
    # trg_time = torch.stack(trg_time)
    # trg_day_time = torch.stack(trg_day_time)
    trg_items = torch.stack(items)
    trg_qdk = torch.stack(qdk)

    input_seq = [
        users,  # input_seq[0]
        src_items,  # input_seq[1]
        src_time,  # input_seq[2]
        src_day_time,  # input_seq[3]
        trg_time,  # input_seq[4]
        trg_day_time,  # input_seq[5]
        src_quad_keys  # input_seq[6]
    ]
    return src_items, trg_items, data_size, src_qdk, trg_qdk
    '''
    src_seq, trg_seq = zip(*batch)
    trajs, mat1, mat2t, labels, lens, labels_for = [], [], [], [], [], []

    for e, t in zip(src_seq, trg_seq):
        u_, l_, t_ = zip(*[(record[0], record[1], record[2]) for record in e])  # src_seq
        u_t_, l_t_, t_t_ = zip(*[(record[0], record[1], record[2]) for record in t])  # trg_seq

        user_traj = [[u, l, t] for u, l, t in zip(u_, l_, t_)]  # 使用列表替代元组
        trajs.append(torch.LongTensor(pad_or_truncate_sequences(user_traj, max_len)))

        # 生成空间和时间矩阵
        mat1.append(
            rst_mat1(pad_or_truncate_sequences(user_traj, max_len), poi))  # Generate spatial and temporal matrix
        mat2t.append(rt_mat2t(pad_sequence(t_, max_len)))  # Temporal matrix for each user

        labels.append(pad_sequence(l_t_, max_len))
        labels_for.append(pad_sequence(l_t_, max_len))
        lens.append(len(l_))

    # 将列表转换为张量
    trajs = torch.stack(trajs)
    mat1 = [torch.tensor(x) if isinstance(x, np.ndarray) else x for x in mat1]
    mat1 = torch.stack(mat1)
    mat2 = [torch.tensor(x) if isinstance(x, np.ndarray) else x for x in mat2t]
    mat2t = torch.stack(mat2)
    labels = torch.stack(labels)
    labels_for = torch.stack(labels_for)
    lens = torch.tensor(lens)
    # 获取完整的 POI 距离矩阵
    #mat2s = torch.tensor(rs_mat2s(poi, n_poi))  # Get full matrix for POI distances

    return trajs, mat1, mat2s, mat2t, lens, labels

def gen_eval_batch(batch, data_source, max_len, n_poi, n_user, poi, mat2s):
    '''
    src_seq, trg_seq = zip(*batch)
    items, data_size, qdk = [], [], []

    batch_data = {
        'user': [],
        'location_x': [],
        'hour': [],
        'hour_mask': [],
        'timeslot_y': [],
        'user_topic_loc': []
    }

    #users, src_time, src_day_time, src_gps, data_size = [], [], [], [], []
    for e in src_seq:
        # user_idx, poi_index, timestamp, location, True
        u_, i_, t_, l_, _ = zip(*e)

        batch_data['user'].append(u_[0])
        batch_data['location_x'].append(pad_sequence(i_, max_len).tolist())
        batch_data['hour'].append(pad_sequence(datetime_to_features1(t_), max_len).tolist())
        occur_time_user = occur_time_individual[u_[0]]
        # Generate hour_mask
        hour_mask = []
        for timestamp in t_:
            mask = np.zeros(24, dtype=np.int32)
            mask[occur_time_user == 0] = 1
            hour_mask.append(mask)
        # Padding hour_mask to [sequence_length, 24]
        padded_hour_mask = process_hour_mask(hour_mask, max_len)
        batch_data['hour_mask'].append(padded_hour_mask)


        users.append(u_[0])
        src_time.append(pad_sequence(t_, max_len))
        src_day_time.append(pad_sequence(t_, max_len))
        src_gps.append(pad_sequence_2d(l_, max_len))

        items.append(pad_sequence(i_, max_len))
        data_size.append(len(_))
        region_idx_tuple = tuple(location_to[loc] if loc in location_to else -1 for loc in l_)
        qdk.append(pad_sequence(region_idx_tuple, max_len))

    for e in trg_seq:
        # user_idx, poi_index, timestamp, location, True
        _, i_, t_, _, _ = zip(*e)
        batch_data['timeslot_y'].append(pad_sequence(datetime_to_features1(t_), max_len).tolist())

    batch_data['user'] = torch.tensor(batch_data['user'])
    batch_data['location_x'] = torch.tensor(batch_data['location_x'])
    batch_data['hour'] = torch.tensor(batch_data['hour'])
    hour_mask_array = np.array(batch_data['hour_mask'])
    batch_data['hour_mask'] = torch.tensor(hour_mask_array)
    batch_data['timeslot_y'] = torch.tensor(batch_data['timeslot_y'])
    selected_user_topics = user_topics[batch_data['user']]
    if topic_num > 0:
        batch_data['user_topic_loc'] = torch.tensor(selected_user_topics)

    src_items = torch.stack(items)
    data_size = torch.tensor(data_size)
    src_qdk = torch.stack(qdk)
    items, trg_time, trg_day_time, qdk = [], [], [], []
    for e in trg_seq:
        # user_idx, poi_index, timestamp, location, True
        u_, i_, t_, l_, _ = zip(*e)
        items.append(pad_sequence(i_, 1))
        # trg_time.append(pad_sequence(t_, max_len))
        # trg_day_time.append(pad_sequence(t_, max_len))
        region_idx_tuple = tuple(location_to[loc] if loc in location_to else -1 for loc in l_)
        qdk.append(pad_sequence(region_idx_tuple, 1))

    src_quad_keys = []
    permutations_dict = get_all_permutations_dict(6)
    for gps_seq in src_gps:
        quad_keys = [latlng2quadkey(loc[0], loc[1], 25) for loc in gps_seq]
        quad_keys = [get_ngrams_of_quadkey(quad_key, 6, permutations_dict) for quad_key in quad_keys]
        src_quad_keys.append(quad_keys)

    # users_tensors = [torch.tensor(u) for u in users]
    # users = torch.stack(users_tensors)
    # src_time = torch.stack(src_time)
    # src_day_time = torch.stack(src_day_time)
    # trg_time = torch.stack(trg_time)
    # trg_day_time = torch.stack(trg_day_time)
    trg_items = torch.stack(items)
    trg_qdk = torch.stack(qdk)

    input_seq = [
        users,  # input_seq[0]
        src_items,  # input_seq[1]
        src_time,  # input_seq[2]
        src_day_time,  # input_seq[3]
        trg_time,  # input_seq[4]
        trg_day_time,  # input_seq[5]
        src_quad_keys  # input_seq[6]
    ]


    trajectories = defaultdict(list)  # 自动初始化为空列表
    labels = defaultdict(list)  # 自动初始化为空列表

    for user_seq in src_seq:
        timestamp_now = datetime.strptime(str(datetime.fromtimestamp(float(user_seq[0][2]))), "%Y-%m-%d %H:%M:%S")
        cur_day_of_year = timestamp_now.timetuple().tm_yday  # 用于记录当天的日期
        user_id = user_seq[0][0]  # 用户ID，假设每个用户的签到序列的第一个元素包含 user_id

        trajectories[user_id] = []  # 初始化标签
        labels[user_id] = []  # 初始化标签
        trajectories[user_id].append([[] for _ in range(model_config['n_time_slot'])])

        for i in range(len(user_seq) - 1):  # 遍历每个用户签到记录，且只到倒数第二条记录
            src_record = user_seq[i]
            trg_record = user_seq[i + 1]
            user_id, poi_index, timestamp, location, valid = src_record
            next_user_id, next_poi_index, next_timestamp, next_location, next_valid = trg_record
            coo = location_to_cluster[location]  # 假设你有一个 location 到 cluster 的映射
            next_coo = location_to_cluster[next_location]

            features = [user_id, poi_index, coo]
            timestamp_obj = datetime.strptime(str(datetime.fromtimestamp(float(timestamp))), "%Y-%m-%d %H:%M:%S")
            next_timestamp_obj = datetime.strptime(str(datetime.fromtimestamp(float(next_timestamp))),"%Y-%m-%d %H:%M:%S")
            time_info = timestamp_obj.hour * 4 + int(timestamp_obj.minute / 15)
            labels_info = [next_poi_index, next_coo]

            checkin = {'features': features, 'time': time_info, 'labels': labels_info}

            if next_timestamp_obj.timetuple().tm_yday != timestamp_obj.timetuple().tm_yday or i == len(user_seq) - 2:
                labels[user_id].append(labels_info)

            if timestamp_obj.timetuple().tm_yday == cur_day_of_year:
                time_slot_index = int(timestamp_obj.hour / (24 / model_config['n_time_slot']))
                trajectories[user_id][-1][time_slot_index].append(checkin)
            else:
                cur_day_of_year = timestamp_obj.timetuple().tm_yday
                trajectories[user_id].append([[] for _ in range(model_config['n_time_slot'])])
                trajectories[user_id][-1][int(timestamp_obj.hour / (24 / model_config['n_time_slot']))].append(checkin)

    MT_batcher = []

    for (user_id_1, trajectory), (user_id_2, label) in zip(trajectories.items(), labels.items()):
        # 构造 MobilityTree
        mobility_tree = construct_MobilityTree(trajectory, label, model_config['n_time_slot'] + 1)
        # 转换到目标设备
        print("???")
        MT_batcher.append(mobility_tree.to(model_config['device']))

    # 合并所有 MobilityTree 图为批次
    MT_batch = dgl.batch(MT_batcher).to(model_config['device'])

    # 构造输入数据
    MT_input = SSTBatch(
        graph=MT_batch,
        features=MT_batch.ndata["x"].to(model_config['device']),
        time=MT_batch.ndata["time"].to(model_config['device']),
        label=MT_batch.ndata["y"].to(model_config['device']),
        mask=MT_batch.ndata["mask"].to(model_config['device']),
        mask2=MT_batch.ndata["mask2"].to(model_config['device']),
        type=MT_batch.ndata["type"].to(model_config['device']),
    )

    return src_items, trg_items, data_size, src_qdk, trg_qdk
    '''
    src_seq, trg_seq = zip(*batch)
    trajs, mat1, mat2t, labels, lens, labels_for = [], [], [], [], [], []

    for e, t in zip(src_seq, trg_seq):

        u_, l_, t_ = zip(*[(record[0], record[1], record[2]) for record in e])  # src_seq
        u_t_, l_t_, t_t_ = zip(*[(record[0], record[1], record[2]) for record in t])  # trg_seq


        user_traj = [[u, l, t] for u, l, t in zip(u_, l_, t_)]  # 使用列表替代元组
        trajs.append(torch.LongTensor(pad_or_truncate_sequences(user_traj, max_len)))

        # 生成空间和时间矩阵
        mat1.append(rst_mat1(pad_or_truncate_sequences(user_traj, max_len), poi))  # Generate spatial and temporal matrix
        mat2t.append(rt_mat2t(pad_sequence(t_, max_len)))  # Temporal matrix for each user

        labels.append(pad_sequence(l_t_, 1))
        labels_for.append(pad_sequence(l_t_, max_len))
        lens.append(len(l_))

    # 将列表转换为张量
    trajs = torch.stack(trajs)
    mat1 = [torch.tensor(x) if isinstance(x, np.ndarray) else x for x in mat1]
    mat1 = torch.stack(mat1)
    mat2 = [torch.tensor(x) if isinstance(x, np.ndarray) else x for x in mat2t]
    mat2t = torch.stack(mat2)
    labels = torch.stack(labels)
    labels_for = torch.stack(labels_for)
    lens = torch.tensor(lens)
    # 获取完整的 POI 距离矩阵
    #mat2s = torch.tensor(rs_mat2s(poi, n_poi))  # Get full matrix for POI distances

    #print(mat2t.shape)
    return trajs, mat1, mat2s, mat2t, lens, labels



def calculate_metrics(num_classes, count):
    array = np.zeros(num_classes)
    for k, v in count.items():
        array[k] = v
    hr = array.cumsum()
    ndcg = 1 / np.log2(np.arange(0, len(array)) + 2)
    ndcg = ndcg * array
    ndcg = ndcg.cumsum() / hr.max()
    hr = hr / hr.max()
    return hr, ndcg


def count(logits, trg, count):
    output = logits.clone()
    for i in range(trg.size(0)):
        output[i][0] = logits[i][trg[i]]
        output[i][trg[i]] = logits[i][0]
    idx = output.sort(descending=True, dim=-1)[1]
    order = idx.topk(k=1, dim=-1, largest=False)[1]
    #print(order)
    count.update(order.squeeze().tolist())
    return count

def datetime_to_features1(timestamps):
    hours = []
    for timestamp in timestamps:
        dt = datetime.datetime.fromtimestamp(int(timestamp) // 1000)  # 转换为 datetime 对象
        hour = dt.hour  # 获取小时
        hours.append(hour)  # 将小时添加到结果列表
    return hours

def datetime_to_features2(timestamp):
    dt = datetime.datetime.fromtimestamp(int(timestamp) // 1000)
    hour = dt.hour
    return hour