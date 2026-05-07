import os
import numpy as np
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from data_factory.lbsn_data import LBSNData, serialize, un_serialize
from data_factory.extended_lbsn_data import ExtendedLBSNData
def data_statistics(dataset, record2txt, path):
    count = 0
    length = []
    for seq in dataset.user_seq:
        count += len(seq)
        length.append(len(seq))
    print("Dataset:", data_name)
    print("#Users:", '{:,}'.format(dataset.n_user - 1))
    print("#POIs:", '{:,}'.format(dataset.n_poi - 1))
    print("#Checkins:", '{:,}'.format(count))
    print("Average Sequence Length:", '{:.2f}'.format(np.mean(np.array(length))))
    print("Sparsity:", '{:.2f}%'.format((1 - count / ((dataset.n_user - 1) * (dataset.n_poi - 1))) * 100))

    if record2txt:
        f = open(path, 'w')
        print("Dataset:", data_name, file=f)
        print("#Users:", '{:,}'.format(dataset.n_user - 1), file=f)
        print("#POIs:", '{:,}'.format(dataset.n_poi - 1), file=f)
        print("#Checkins:", '{:,}'.format(count), file=f)
        print("Average Sequence Length:", '{:.2f}'.format(np.mean(np.array(length))), file=f)
        print("Sparsity:", '{:.2f}%'.format((1 - count / ((dataset.n_user - 1) * (dataset.n_poi - 1))) * 100), file=f)
        f.close()


if __name__ == '__main__':
    # 'Brightkite', 'Gowalla', 'Weeplaces', 'NYC', 'TKY', 'SIN'
    data_name = 'NYC'
    raw_file_prefix = '/home/admin/hongx/code/LBSN_Checkins/'
    raw_file_suffix = '/raw_checkins.txt'
    new_file_prefix = '/home/admin/hongx/code/data_factory/dataset/'
    new_data_file_suffix = 'checkins.data'
    new_stat_file_suffix = 'statistics.txt'

    poi_threshold = 10
    user_threshold = 20

    raw_data_path = raw_file_prefix + data_name + raw_file_suffix
    new_file_path = new_file_prefix + data_name + '/'
    new_data_path = new_file_path + new_data_file_suffix
    new_stat_path = new_file_path + new_stat_file_suffix

    if os.path.isfile(new_data_path):
        dataset = un_serialize(new_data_path)
        data_statistics(dataset, record2txt=False, path=new_stat_path)
    else:
        if not os.path.exists(raw_data_path):
            os.makedirs(new_file_path)
        serialize(LBSNData(raw_data_path, poi_threshold, user_threshold), new_data_path)
        dataset = un_serialize(new_data_path)
        data_statistics(dataset, record2txt=True, path=new_stat_path)
