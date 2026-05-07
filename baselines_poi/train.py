import copy
import time
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
import torch
from torch.utils.data import DataLoader
from collections import Counter
import sys
import os
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_factory.lbsn_data import un_serialize

LOD =17
hours = 24*7

def generate_location_cluster(user_seqs, n_clusters):
    # Step 1: 收集所有用户签到记录中的地理位置 (经纬度)，去重
    all_locations = set()  # 使用集合避免重复位置

    for user_seq in user_seqs:
        for record in user_seq:
            _, _, _, location, _ = record  # 解压user_seq
            all_locations.add(location)  # 将地理位置添加到集合中

    # Step 2: 将所有唯一的地理位置转换为numpy数组形式
    locations = np.array(list(all_locations))  # 需要是一个二维数组，形状为 (n_locations, 2)

    # Step 3: 使用 KMeans 对地理位置进行聚类
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(locations)  # 聚类

    # Step 4: 为每个地理位置生成簇标签字典，并将簇索引从 0 转换为 1
    location_to_cluster = {}  # 存储每个地理位置对应的簇标签
    for i, location in enumerate(all_locations):
        location_to_cluster[location] = kmeans.labels_[i] + 1  # 将簇标签加1，使其从1开始

    return location_to_cluster

def compute_user_matrices_first(dataset, topic_num, n_poi, n_user):
    # 确定用户数和地点数
    num_users = n_user
    num_locations = n_poi

    # 初始化张量
    trans_time_individual = []
    occur_time_individual = np.zeros((num_users, 24), dtype=np.float32)
    user_loc_matrix = np.zeros((num_users, num_locations))
    diff_data = []

    # 遍历用户的签到序列
    for user_seq in dataset.user_seq:  # 这里移除了desc信息
        # user_seq 是一个用户的签到记录，格式为：(user_idx, poi_index, timestamp, location, True)
        #print(user_seq)
        # 初始化用户转移矩阵
        trans_matrix_time = np.ones((24, 24))  # 24个小时的转移矩阵

        # 提取 POI 和时间戳信息
        user_idx = user_seq[0][0]

        i_ = [record[1] for record in user_seq]  # POI 索引
        t_ = [record[2] for record in user_seq]  # 时间戳
        locations = [record[3] for record in user_seq]  # (lat, lon) 二元组

        # 遍历签到点，计算转移矩阵和统计信息
        for idx in range(len(i_) - 1):
            location, timestamp = i_[idx], t_[idx]
            next_location, next_timestamp = i_[idx + 1], t_[idx + 1]

            # 提取时间特征
            hour = datetime_to_features2(timestamp)
            next_hour = datetime_to_features2(next_timestamp)

            # 统计时间差
            diff_data.append(abs(next_hour - hour))

            # 更新转移矩阵和签到统计
            trans_matrix_time[hour, next_hour] += 1
            occur_time_individual[user_idx, hour] += 1
            user_loc_matrix[user_idx, location] += 1

            # 如果是最后一个签到点，统计下一个签到点
            if idx == len(i_) - 2:
                occur_time_individual[user_idx, next_hour] += 1
                user_loc_matrix[user_idx, next_location] += 1

        # 归一化转移矩阵
        time_row_sums = trans_matrix_time.sum(axis=1)
        trans_matrix_time = trans_matrix_time / time_row_sums[:, np.newaxis]
        trans_time_individual.append(trans_matrix_time)

    # 转为 numpy 数组
    trans_time_individual = np.array(trans_time_individual)

    # 构造用户主题分布
    dictionary = corpora.Dictionary([[str(i)] for i in range(num_locations)])
    corpus = []

    # 为每个用户构造文档
    for user in user_loc_matrix:
        user_doc = [str(loc) for loc, count in enumerate(user) for _ in range(int(count))]
        corpus.append(dictionary.doc2bow(user_doc))

    user_topics = np.zeros((num_users, topic_num))
    if topic_num > 0:
        # 使用 LDA 生成用户主题分布
        lda = models.LdaModel(corpus, num_topics=topic_num, random_state=42)
        for i, user in enumerate(user_loc_matrix):
            user_doc = [str(loc) for loc, count in enumerate(user) for _ in range(int(count))]
            for item in lda[dictionary.doc2bow(user_doc)]:
                j = item[0]
                prob = item[1]
                user_topics[i, j] = prob

    return user_topics, occur_time_individual

def generate_location_region(user_seqs):
    region2idx = {}  # 存储每个区域对应的索引
    idx2region = {}  # 存储索引对应的区域
    n_region = 1  # 区域索引从0开始

    location_to = {}
    for user_seq in user_seqs:
        for record in user_seq:
            _, _, _, location, _ = record
            lat, lon = location

            # 调试信息：查看坐标和生成的区域
            region = latlon2quadkey(lat, lon, LOD)
            #print(f"Latitude: {lat}, Longitude: {lon}, Region: {region}")

            # 检查该区域是否已存在，若不存在则为其分配一个新的索引
            if region not in region2idx:
                region2idx[region] = n_region  # 为该区域分配新的索引
                idx2region[n_region] = region  # 将索引映射回区域
                n_region += 1  # 增加索引

            # 记录每个位置对应的区域索引，确保不会覆盖已有的值
            if location not in location_to:
                region_idx = region2idx[region]
                location_to[location] = region_idx  # 记录 (lat, lon) -> 区域索引 的映射

    return location_to, n_region

def generate_poi_info(user_seqs):
    poi_info = set()  # 使用 set 来确保去重

    # 遍历所有用户的轨迹数据
    for user_seq in user_seqs:
        for record in user_seq:
            _, poi_index, _, location, _ = record  # 提取信息，忽略其他不需要的部分

            # 提取 location 中的纬度和经度
            lat, lng = location

            # 将 poi_index, lat, lng 作为一个元组加入到集合中
            poi_info.add((poi_index, lat, lng))

    # 将结果从 set 转换为 list，并按需要排序（例如按 poi_index 排序）
    poi_info = list(poi_info)
    poi_info.sort(key=lambda x: x[0])  # 如果需要按 poi_index 排序

    return poi_info


def calculate_ex(user_seqs):
    # 初始化最大最小值
    max_distance = float('-inf')
    min_distance = float('inf')
    max_time_diff = float('-inf')
    min_time_diff = float('inf')

    for user_seq in user_seqs:
        # 遍历每个用户的轨迹
        prev_location = None
        prev_time = None

        for record in user_seq:
            _, poi_index, timestamp, location, _ = record
            lat, lon = location

            if prev_location is not None:
                # 计算位置之间的距离
                lat_prev, lon_prev = prev_location
                distance = haversine(lat_prev, lon_prev, lat, lon)
                max_distance = max(max_distance, distance)
                min_distance = min(min_distance, distance)

            if prev_time is not None:
                # 计算时间间隔
                time_diff = timestamp - prev_time
                max_time_diff = max(max_time_diff, time_diff)
                min_time_diff = min(min_time_diff, time_diff)

            # 更新上一位置和时间
            prev_location = location
            prev_time = timestamp

    ex = (max_distance, min_distance, max_time_diff, min_time_diff)
    return ex

class TrainFactory:
    def __init__(self, model, exp_config, model_config, train_config, test_config):
        super(TrainFactory, self).__init__()

        self.model = model
        self.exp_config = exp_config
        self.model_config = model_config
        self.train_config = train_config
        self.test_config = test_config
        self.data_path, self.result_path, self.model_path = self.make_path()
        self.n_poi, self.n_user, self.train_loader, self.test_loader, self.ex = self.prepare_data()
        self.best_metric_dict, self.best_epoch_dict = self.make_metric_dict()
        self.best_model = None
        self.bad_count = 0
        self.early_stop_flag = False

    def make_metric_dict(self):
        metric = {}
        epoch ={'Best Metric Epoch': 0}
        for k in self.test_config['k_list']:
            metric['HR@{}'.format(k)] = 0.0
            metric['NDCG@{}'.format(k)] = 0.0
        return metric, epoch

    def make_path(self):
        data_name = self.exp_config['data_name']
        scenario = self.exp_config['scenario']
        model_name = self.model_config['model_name']

        data_path_prefix = '/home/admin/hongx/TPPRec/data_factory/dataset/'
        data_path_suffix = '/checkins.data'
        data_path = data_path_prefix + data_name + data_path_suffix


        result_path_prefix = '/home/admin/hongx/TPPRec/results/'
        result_path_prefix_data_include = result_path_prefix + data_name + '/'
        if not os.path.exists(result_path_prefix_data_include):
            os.makedirs(result_path_prefix_data_include)
        result_path_suffix = model_name + '_rec_perfm_' + scenario + '.txt'
        result_path = result_path_prefix_data_include + result_path_suffix

        model_path_prefix = '/home/admin/hongx/TPPRec/models/'
        model_path_prefix_data_include = model_path_prefix + data_name + '/'
        if not os.path.exists(model_path_prefix_data_include):
            os.makedirs(model_path_prefix_data_include)
        model_path_suffix = model_name + '_' + scenario + '.pkl'
        model_path = model_path_prefix_data_include + model_path_suffix


        return data_path, result_path, model_path

    def prepare_data(self):
        dataset = un_serialize(self.data_path)
        n_poi = dataset.n_poi
        n_user = dataset.n_user
        #user_topics, occur_time_individual = compute_user_matrices_first(dataset, self.model_config['topic_num'], n_poi, n_user)
        #location_to_cluster = generate_location_cluster(dataset.user_seq, self.model_config['K_cluster'])
        #location_to, n_region = generate_location_region(dataset.user_seq)
        #poi_info = generate_poi_info(dataset.user_seq)
        ex = calculate_ex(dataset.user_seq)
        idx_to_loc = dataset.idx2loc

        mat2s = rs_mat2s(idx_to_loc, n_poi)  # Get full matrix for POI distances
        #print(mat2s.shape)
        '''
        file_path = '/home/admin/hongx/TPPRec/poi_distances_bri.pt'  # 保存路径

        # 检查路径是否存在，若不存在则创建
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        # 检查文件是否存在
        if os.path.exists(file_path):
            print(f"文件 {file_path} 已存在，直接加载。")
            mat2s = torch.load(file_path)  # 加载已保存的矩阵
        else:
            print(f"文件 {file_path} 不存在，正在计算并保存。")
            mat2s = torch.tensor(rs_mat2s(idx_to_loc, n_poi))  # 计算矩阵
            torch.save(mat2s, file_path)  # 保存到文件
        '''
        max_len = self.exp_config['max_len']
        scenario = self.exp_config['scenario']
        train_batch_size = self.train_config['batch_size']
        test_batch_size = self.test_config['batch_size']

        train_data, test_data = dataset.partition(max_len, scenario)
        train_loader = DataLoader(dataset=train_data, batch_size=train_batch_size,
                                  sampler=LadderSampler(train_data, train_batch_size),
                                  num_workers=4,
                                  prefetch_factor=2,
                                  collate_fn=lambda e: gen_train_batch(e, train_data, max_len, n_poi, n_user, idx_to_loc, mat2s))
        test_loader = DataLoader(dataset=test_data,
                                 batch_size=test_batch_size,
                                 num_workers=4,
                                 prefetch_factor=2,
                                 collate_fn=lambda e: gen_eval_batch(e, test_data, max_len, n_poi, n_user, idx_to_loc, mat2s))
        return n_poi, n_user, train_loader, test_loader, ex

    def construct_model(self):
        n_poi = self.n_poi
        n_user = self.n_user
        ex = self.ex
        model = self.model(hours+1, n_poi, n_user, ex, **self.model_config)
        #if os.path.exists(self.model_path):
        #    model.load(self.model_path)
        return model

    def train_function(self, dataloader, model, optimizer, epoch):
        device = self.exp_config['device']
        start_time = time.time()
        model.train()
        running_loss = 0.0
        processed_batch = 0
        tqdm_title = '>>> ' + 'Training Epoch: {} >>>'.format(epoch)
        batch_iterator = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc=tqdm_title)
        for batch_idx, (trajs, mat1, mat2s, m2t, lens, trg) in batch_iterator:
            optimizer.zero_grad()
            trajs = trajs.to(device)
            mat1 = mat1.to(device)
            mat2s = mat2s.to(device)
            m2t = m2t.to(device)
            lens = lens.to(device)
            '''
            batch_size, max_len, num_class = trajs.shape
            all_logits = torch.zeros((batch_size, max_len, self.n_poi), dtype=torch.float32).to(device)

            # 定义掩码，用于输入数据的逐步处理
            input_mask = torch.zeros((batch_size, max_len, trajs.size(-1)), dtype=torch.long).to(device)
            m1_mask = torch.zeros((batch_size, max_len, max_len, mat1.size(-1)), dtype=torch.float32).to(device)

            # 按照最大长度逐步生成掩码并处理数据
            for mask_len in range(1, max_len + 1):  # 遍历每个时间步
                input_mask[:, :mask_len] = 1
                m1_mask[:, :mask_len, :mask_len] = 1

                # 应用掩码过滤输入
                train_input = trajs * input_mask
                train_m1 = mat1 * m1_mask
                train_m2t = m2t[:, mask_len - 1]
                train_len = torch.zeros(size=(batch_size,), dtype=torch.long).to(device) + mask_len

                # 调用模型进行预测
                logits_one = model(train_input, train_m1, mat2s, train_m2t, train_len)  # (N, L)
                all_logits[:, mask_len - 1, :] = logits_one  # 将 (N, L) 填充到 (N, M, L)

            #logits = model(trajs, mat1, mat2s, m2t, lens)
            logits = all_logits.reshape(-1, all_logits.size(-1))
            '''
            logits = model(trajs, mat1, mat2s, m2t[:, max_len-1], lens)
            logits = logits.view(-1, logits.size(-1))
            trg = trg.to(device)
            trg = trg.view(-1)
            loss = F.cross_entropy(logits, trg, ignore_index=0)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().item()
            processed_batch = processed_batch + 1
            batch_iterator.set_postfix_str('Loss={:.4f}'.format(loss.item()))
        cost_time = time.time() - start_time
        avg_loss = running_loss / processed_batch
        print('Time={:.4f}, Average Loss={:.4f}'.format(cost_time, avg_loss))

    def test_function(self, n_poi, dataloader, model, epoch):
        device = self.exp_config['device']
        k_list = self.test_config['k_list']

        cnt = Counter()
        model.eval()
        if epoch is None:
            tqdm_title = '>>> ' + 'Evaluating Model >>>'
        else:
            tqdm_title = '>>> ' + 'Evaluating Epoch: {} >>>'.format(epoch)
        batch_iterator = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc=tqdm_title)
        with torch.no_grad():
            for batch_idx, (trajs, mat1, mat2s, m2t, lens, trg) in batch_iterator:
                trajs = trajs.to(device)
                mat1 = mat1.to(device)
                mat2s = mat2s.to(device)
                m2t = m2t.to(device)
                lens = lens.to(device)
                '''
                batch_size, max_len, num_class = trajs.shape
                all_logits = torch.zeros((batch_size, max_len, self.n_poi), dtype=torch.float32).to(device)

                # 定义掩码，用于输入数据的逐步处理
                input_mask = torch.zeros((batch_size, max_len, trajs.size(-1)), dtype=torch.long).to(device)
                m1_mask = torch.zeros((batch_size, max_len, max_len, mat1.size(-1)), dtype=torch.float32).to(device)

                # 按照最大长度逐步生成掩码并处理数据
                for mask_len in range(1, max_len + 1):  # 遍历每个时间步
                    input_mask[:, :mask_len] = 1
                    m1_mask[:, :mask_len, :mask_len] = 1

                    # 应用掩码过滤输入
                    train_input = trajs * input_mask
                    train_m1 = mat1 * m1_mask
                    train_m2t = m2t[:, mask_len - 1]
                    train_len = torch.zeros(size=(batch_size,), dtype=torch.long).to(device) + mask_len

                    # 调用模型进行预测
                    logits_one = model(train_input, train_m1, mat2s, train_m2t, train_len)  # (N, L)
                    #print(logits_one.shape)
                    # 将当前时间步的输出存入 all_logits
                    all_logits[:, mask_len - 1, :] = logits_one  # 将 (N, L) 填充到 (N, M, L)
                #print(all_logits.shape)
                #print(trg.shape)
                output = all_logits[torch.arange(lens.size(0)), lens - 1, :].detach()
                '''
                scores = model(trajs, mat1, mat2s, m2t[:, max_len-1], lens)
                cnt = count(scores, trg, cnt)
        hr, ndcg = calculate_metrics(n_poi, cnt)
        metric = {}
        for k in k_list:
            metric['HR@{}'.format(k)] = hr[k - 1]
            metric['NDCG@{}'.format(k)] = ndcg[k - 1]
        return metric, model

    def early_stop_function(self, epoch, metric_dict, model):
        curr_metric_sum = 0.0
        best_metric_sum = 0.0
        for value in metric_dict.values():
            curr_metric_sum += value
        for value in self.best_metric_dict.values():
            best_metric_sum += value

        curr_metric_mean = curr_metric_sum / len(metric_dict)
        best_metric_mean = best_metric_sum / len(self.best_metric_dict)

        if curr_metric_mean > best_metric_mean:
            self.bad_count = 0
            self.best_model = copy.deepcopy(model)
            self.best_metric_dict = metric_dict
            self.best_epoch_dict['Best Metric Epoch'] = epoch
            self.best_model.save(self.model_path)
            f = open(self.result_path, 'w')
            print('Dataset: {}'.format(self.exp_config['data_name']))
            print('Model: {}'.format(self.model_config['model_name']))
            for key, value in self.best_metric_dict.items():
                print('Best {} achieved at Epoch {}: {:.2f}%'.format(key, epoch, value * 100))
                print('Best {} achieved at Epoch {}: {:.2f}%'.format(key, epoch, value * 100), file=f)
            print('>>>')
        else:
            self.bad_count += 1
            print('No improvement for {} Epochs'.format(epoch - self.best_epoch_dict['Best Metric Epoch']))

        tolerance = self.train_config['tolerance']
        if self.bad_count >= tolerance:
            self.early_stop_flag = True

        return self.early_stop_flag

    def train_model(self):
        device = self.exp_config['device']
        model = self.construct_model()
        model.to(device)
        seed = self.train_config['seed']
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_config['lr'])
        fix_random_seed_as(seed)
        self.best_metric_dict, self.best_model = self.test_function(self.n_poi, self.test_loader, model, epoch=0)
        print('Dataset: {}'.format(self.exp_config['data_name']))
        print('Model: {}'.format(self.model_config['model_name']))
        for key, value in self.best_metric_dict.items():
            print('Metric {}: {:.2f}%'.format(key, value * 100))
        epoch = 1
        while self.early_stop_flag is False:
            self.train_function(self.train_loader, model, optimizer, epoch)
            current_metric, current_model = self.test_function(self.n_poi, self.test_loader, model, epoch)
            self.early_stop_flag = self.early_stop_function(epoch, current_metric, current_model)
            epoch += 1
        print('Training Done!')
        print('Dataset: {}'.format(self.exp_config['data_name']))
        print('Model: {}'.format(self.model_config['model_name']))
        for key, value in self.best_metric_dict.items():
            print('Best {} achieved at Epoch {}: {:.2f}%'.format(key, epoch, value * 100))

    def validate_model(self):
        device = self.exp_config['device']
        model = self.construct_model()
        model.load(self.model_path)
        model.to(device)
        metric, _ = self.test_function(self.n_poi, self.test_loader, model, epoch=None)
        print('Dataset: {}'.format(self.exp_config['data_name']))
        print('Model: {}'.format(self.model_config['model_name']))
        for key, value in self.best_metric_dict.items():
            print('Metric {}: {:.2f}%'.format(key, value * 100))
