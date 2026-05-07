import os
import json
import copy
import math
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime
try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


def serialize(obj, path, in_json=False):
    if in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            _pickle.dump(obj, file)


def un_serialize(path):
    suffix = os.path.basename(path).split(".")[-1]

    print(path)

    if suffix == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:
            return _pickle.load(file)


def convert_to_timestamp(time_str):
    dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
    timestamp = dt.timestamp()
    return timestamp


class LBSNData(Dataset):
    def __init__(self, path, poi_threshold, user_threshold):
        super(LBSNData, self).__init__()
        self.poi2idx = {'<pad>': 0}
        self.idx2poi = {0: '<pad>'}
        self.poi2count = {}
        self.n_poi = 1
        self.poi2loc = {'<pad>': (0.0, 0.0)}
        self.idx2loc = {0: (0.0, 0.0)}

        self.build_poi_vocabulary(path, poi_threshold)
        self.n_user, self.user2idx, self.user_seq = self.process(path, user_threshold)

    def build_poi_vocabulary(self, path, poi_threshold):
        """
        In LBSN Raw Check-in Datasets, each entry is [0:user, 1:time, 2:lat, 3:lon, 4:poi]
        """
        for line in open(path):
            line = line.strip().split('\t')
            if len(line) < 5:
                continue
            lat = float(line[2])
            lon = float(line[3])
            loc = (lat, lon)
            poi = line[4]
            self.add_poi(poi, loc)
        if poi_threshold > 0:
            self.n_poi = 1
            self.poi2idx = {'<pad>': 0}
            self.idx2poi = {0: '<pad>'}
            self.idx2loc = {0: (0.0, 0.0)}
            for poi in self.poi2count:
                if self.poi2count[poi] >= poi_threshold:
                    loc = self.poi2loc[poi]
                    self.add_poi(poi, loc)

    def add_poi(self, poi, loc):
        if poi not in self.poi2idx:
            self.poi2idx[poi] = self.n_poi
            self.poi2loc[poi] = loc
            self.idx2poi[self.n_poi] = poi
            self.idx2loc[self.n_poi] = loc
            if poi not in self.poi2count:
                self.poi2count[poi] = 1
            self.n_poi += 1
        else:
            self.poi2count[poi] += 1

    def process(self, path, user_threshold):
        n_user = 1
        user2idx = {}
        user_seq = {}
        user_seq_array = list()
        for line in open(path):
            line = line.strip().split('\t')
            if len(line) < 5:
                continue
            user, time, _, _, poi = line
            if poi not in self.poi2idx:
                continue
            poi_idx = self.poi2idx[poi]
            loc = self.idx2loc[poi_idx]
            timestamp = convert_to_timestamp(time)
            if user not in user_seq:
                user_seq[user] = list()
            user_seq[user].append([timestamp, loc, poi_idx])

        for user, seq in user_seq.items():
            if len(seq) >= user_threshold:
                user2idx[user] = n_user
                user_idx = n_user
                new_seq = []
                tmp_set = set()
                count = 0
                for timestamp, location, poi_index in sorted(seq, key=lambda x: x[0]):
                    if poi_index in tmp_set:
                        new_seq.append((user_idx, poi_index, timestamp, location, True))
                    else:
                        new_seq.append((user_idx, poi_index, timestamp, location, False))
                        tmp_set.add(poi_index)
                        count += 1
                if count >= (user_threshold // 2):
                    n_user += 1
                    user_seq_array.append(new_seq)
        return n_user, user2idx, user_seq_array

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        return self.user_seq[idx]

    def partition(self, max_len, mode):
        train_data = copy.copy(self)
        eval_data = copy.copy(self)
        train_seq = []
        eval_seq = []
        for user in range(len(self)):
            seq = self[user]
            if mode == 'last':
                i = len(seq) - 1
            else:
                for i in reversed(range(len(seq))):
                    if mode == 'exp':
                        if not seq[i][-1]:
                            break
            trg = seq[i: i+1]
            src = seq[max(0, i-max_len): i]
            eval_seq.append((src, trg))

            n_sample = math.floor((i+max_len-1)/max_len)
            for k in range(n_sample):
                if (i-k*max_len) > max_len*1.1:
                    trg = seq[i-(k+1)*max_len: i-k*max_len]
                    src = seq[i-(k+1)*max_len-1: i-k*max_len-1]
                    train_seq.append((src, trg))
                else:
                    trg = seq[1: i-k*max_len]
                    src = seq[0: i-k*max_len-1]
                    train_seq.append((src, trg))
                    break
        train_data.user_seq = train_seq
        eval_data.user_seq = eval_seq
        return train_data, eval_data

