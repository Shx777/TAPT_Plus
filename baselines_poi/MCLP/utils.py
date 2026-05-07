import torch
import random
import numpy as np
from torch.utils.data import Sampler
from torch.backends import cudnn
import datetime
from tqdm import tqdm
from gensim import corpora, models

def pad_sequence(seq, max_len):
    seq = list(seq)
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[-max_len:]
    return torch.tensor(seq)


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


def compute_user_matrices(src_seq, topic_num, n_poi, n_user):
    # 确定用户数和地点数

    num_users = n_user
    num_locations = n_poi
    # 初始化张量
    trans_time_individual = []
    occur_time_individual = np.zeros((num_users, 24), dtype=np.float32)
    user_loc_matrix = np.zeros((num_users, num_locations))
    diff_data = []

    # 遍历用户签到序列
    for e in tqdm(src_seq, desc="Processing user sequences"):
        # 解包签到数据
        u_, i_, t_, _, _ = zip(*e)  # 用户索引、POI索引、时间戳等

        # 初始化用户转移矩阵
        trans_matrix_time = np.ones((24, 24))

        # 遍历签到点
        for idx in range(len(i_) - 1):
            user = u_[0]  # 当前 batch 的用户索引
            location, timestamp = i_[idx], t_[idx]
            next_location, next_timestamp = i_[idx + 1], t_[idx + 1]

            # 提取时间特征
            hour = datetime_to_features2(timestamp)
            next_hour = datetime_to_features2(next_timestamp)

            # 统计时间差
            diff_data.append(abs(next_hour - hour))

            # 更新转移矩阵和签到统计
            trans_matrix_time[hour, next_hour] += 1
            occur_time_individual[user, hour] += 1
            user_loc_matrix[user, location] += 1

            # 如果是最后一个签到点，统计下一个签到点
            if idx == len(i_) - 2:
                occur_time_individual[user, next_hour] += 1
                user_loc_matrix[user, next_location] += 1

        # 归一化转移矩阵
        time_row_sums = trans_matrix_time.sum(axis=1)
        trans_matrix_time = trans_matrix_time / time_row_sums[:, np.newaxis]
        trans_time_individual.append(trans_matrix_time)

    # 转为 numpy 数组
    trans_time_individual = np.array(trans_time_individual)

    # 构造用户主题分布
    dictionary = corpora.Dictionary([[str(i)] for i in range(num_locations)])
    corpus = []

    for user in user_loc_matrix:
        user_doc = [str(loc) for loc, count in enumerate(user) for _ in range(int(count))]
        corpus.append(dictionary.doc2bow(user_doc))

    user_topics = np.zeros((num_users, topic_num))
    if topic_num > 0:
        print(f"Generating a probability distribution... topic: {topic_num}")
        lda = models.LdaModel(corpus, num_topics=topic_num, random_state=42)
        for i, user in enumerate(user_loc_matrix):
            user_doc = [str(loc) for loc, count in enumerate(user) for _ in range(int(count))]
            for item in lda[dictionary.doc2bow(user_doc)]:
                j = item[0]
                prob = item[1]
                user_topics[i, j] = prob

    return user_topics, trans_time_individual, occur_time_individual, user_loc_matrix

def gen_train_batch(batch, data_source, max_len, n_poi, n_user, topic_num=1500):
    src_seq, trg_seq = zip(*batch)
    items, data_size = [], []
    batch_data = {
        'user': [],
        'location_x': [],
        'hour': [],
        'hour_mask': [],
        'timeslot_y': [],
        'user_topic_loc': []
    }

    user_topics, _, occur_time_individual, _ = compute_user_matrices(src_seq, topic_num,n_poi, n_user,)
    for e in src_seq:
        # user_idx, poi_index, timestamp, location, True
        u_, i_, t_, _, _ = zip(*e)
        batch_data['user'].append(u_[0])
        batch_data['location_x'].append(pad_sequence(i_, max_len).tolist())
        batch_data['hour'].append(pad_sequence(datetime_to_features1(t_), max_len).tolist())
        occur_time_user = occur_time_individual[u_[0]]
        # Generate hour_mask
        hour_mask = []
        for timestamp in t_:
            hour = datetime_to_features2(timestamp)
            mask = np.zeros(24, dtype=np.int32)
            mask[occur_time_user == 0] = 1
            hour_mask.append(mask)
        # Padding hour_mask to [sequence_length, 24]
        padded_hour_mask = process_hour_mask(hour_mask, max_len)
        batch_data['hour_mask'].append(padded_hour_mask)

        items.append(pad_sequence(i_, max_len))
        data_size.append(len(_))

    for e in trg_seq:
        # user_idx, poi_index, timestamp, location, True
        _, i_, t_, _, _ = zip(*e)
        batch_data['timeslot_y'].append(pad_sequence(datetime_to_features1(t_), max_len).tolist())
        items.append(pad_sequence(i_, max_len))


    batch_data['user'] = torch.tensor(batch_data['user'])
    batch_data['location_x'] = torch.tensor(batch_data['location_x'])
    batch_data['hour'] = torch.tensor(batch_data['hour'])
    batch_data['hour_mask'] = torch.tensor(batch_data['hour_mask'])
    batch_data['timeslot_y'] = torch.tensor(batch_data['timeslot_y'])

    # 使用 batch_data['user'] 作为索引从 user_topics 中选择相关用户的主题信息
    selected_user_topics = user_topics[batch_data['user']]

    # 如果 topic_num > 0，生成随机 user_topic_loc
    #if topic_num > 0:
    #    batch_data['user_topic_loc'].append(user_topics.tolist())

    if topic_num > 0:
        batch_data['user_topic_loc'] = torch.tensor(selected_user_topics)

    src_items = torch.stack(items)
    data_size = torch.tensor(data_size)
    items = []
    for e in trg_seq:
        _, i_, _, _, _ = zip(*e)
        items.append(pad_sequence(i_, max_len))
    trg_items = torch.stack(items)
    return src_items, trg_items, data_size, batch_data


def gen_eval_batch(batch, data_source, max_len, n_poi, n_user,topic_num=1500):
    src_seq, trg_seq = zip(*batch)
    items, data_size = [], []
    batch_data = {
        'user': [],
        'location_x': [],
        'hour': [],
        'hour_mask': [],
        'timeslot_y': [],
        'user_topic_loc': []
    }
    user_topics, _, occur_time_individual, _ = compute_user_matrices(src_seq, topic_num, n_poi, n_user)

    for e in src_seq:
        # user_idx, poi_index, timestamp, location, True
        u_, i_, t_, _, _ = zip(*e)
        batch_data['user'].append(u_[0])
        batch_data['location_x'].append(pad_sequence(i_, max_len).tolist())
        batch_data['hour'].append(pad_sequence(datetime_to_features1(t_), max_len).tolist())
        occur_time_user = occur_time_individual[u_[0]]
        # Generate hour_mask
        hour_mask = []
        for timestamp in t_:
            hour = datetime_to_features2(timestamp)
            mask = np.zeros(24, dtype=np.int32)
            mask[occur_time_user == 0] = 1
            hour_mask.append(mask)
        # Padding hour_mask to [sequence_length, 24]
        padded_hour_mask = process_hour_mask(hour_mask, max_len)
        batch_data['hour_mask'].append(padded_hour_mask)

        items.append(pad_sequence(i_, max_len))
        data_size.append(len(_))

    for e in trg_seq:
        # user_idx, poi_index, timestamp, location, True
        _, i_, t_, _, _ = zip(*e)
        batch_data['timeslot_y'].append(pad_sequence(datetime_to_features1(t_), max_len).tolist())
        items.append(pad_sequence(i_, max_len))

    batch_data['user'] = torch.tensor(batch_data['user'])
    batch_data['location_x'] = torch.tensor(batch_data['location_x'])
    batch_data['hour'] = torch.tensor(batch_data['hour'])
    batch_data['hour_mask'] = torch.tensor(batch_data['hour_mask'])
    batch_data['timeslot_y'] = torch.tensor(batch_data['timeslot_y'])

    # 使用 batch_data['user'] 作为索引从 user_topics 中选择相关用户的主题信息
    selected_user_topics = user_topics[batch_data['user']]

    # 如果 topic_num > 0，生成随机 user_topic_loc
    #if topic_num > 0:
    #    batch_data['user_topic_loc'].append(user_topics.tolist())


    if topic_num > 0:
        batch_data['user_topic_loc'] = torch.tensor(selected_user_topics)

    src_items = torch.stack(items)
    data_size = torch.tensor(data_size)
    items = []
    for e in trg_seq:
        # user_idx, poi_index, timestamp, location, True
        _, i_, _, _, _ = zip(*e)
        items.append(pad_sequence(i_, 1))
    trg_items = torch.stack(items)
    return src_items, trg_items, data_size, batch_data


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