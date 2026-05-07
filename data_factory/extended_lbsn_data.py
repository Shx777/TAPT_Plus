import os
import numpy as np
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from data_factory.lbsn_data import LBSNData, serialize, un_serialize

class ExtendedLBSNData(LBSNData):
    def __init__(self, path, poi_threshold, user_threshold):
        super(ExtendedLBSNData, self).__init__(path, poi_threshold, user_threshold)
        self.poi_category2idx = {'<unknown>': 0}  # 初始化 POI 种类字典
        self.idx2poi_category = {0: '<unknown>'}
        self.poi2category = {}  # 映射每个 POI 到其种类
        self.n_poi_category = 1

        self.assign_poi_categories(path)  # 增加种类处理
        self.en_user_seq = self.enrich_user_seq()

    def assign_poi_categories(self, path):
        """
        为每个 POI 分配种类 ID。假设原始数据中第 5 列是 POI， 第 6 列是 POI 的种类。
        """
        for line in open(path):
            line = line.strip().split('\t')
            if len(line) < 6:  # 检查是否有种类列
                continue
            poi = line[4]
            category = line[5]  # 假设 POI 种类在第 6 列
            if poi not in self.poi2idx:  # 只处理有效 POI
                continue
            if category not in self.poi_category2idx:
                self.poi_category2idx[category] = self.n_poi_category
                self.idx2poi_category[self.n_poi_category] = category
                self.n_poi_category += 1
            self.poi2category[poi] = self.poi_category2idx[category]


    def enrich_user_seq(self):
        """
        处理 self.user_seq，为每条访问记录添加对应的 POI 类别。
        """
        enriched_seq = []  # 存储经过处理后的用户序列
        for user_idx, poi_idx, timestamp, location, is_repeat in self.user_seq:
            # 获取 POI 名称
            poi = self.idx2poi[poi_idx]
            cat = self.poi2category[poi]
            enriched_seq.append((user_idx, poi_idx, timestamp, location, is_repeat, cat))

        return enriched_seq