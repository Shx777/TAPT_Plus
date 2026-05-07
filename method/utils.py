import torch
import random
import numpy as np
from torch.utils.data import Sampler
from torch.backends import cudnn


def pad_sequence(seq, max_len):
    seq = list(seq)
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[-max_len:]
    return torch.tensor(seq)


def pad_sequence_2d(seq, max_len, pad_value=(0, 0, 0)):
    seq = list(seq)

    if len(seq) < max_len:
        seq = seq + [pad_value] * (max_len - len(seq))
    else:
        seq = seq[-max_len:]

    return torch.tensor(seq, dtype=torch.long)  # [max_len, 3]


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

import time
def parse_hms(ts):
    """
    ts: timestamp (int / float / datetime)
    return: (hour, minute, second)
    """
    if hasattr(ts, 'hour'):  # datetime
        return ts.hour, ts.minute, ts.second
    else:  # unix timestamp (seconds)
        t = time.localtime(ts)
        return t.tm_hour, t.tm_min, t.tm_sec



def gen_train_batch(batch, data_source, max_len):
    src_seq, trg_seq = zip(*batch)

    # ===== source =====
    src_items, src_times, data_size = [], [], []
    for e in src_seq:
        _, i_, t_, _, _ = zip(*e)

        src_items.append(pad_sequence(i_, max_len))
        data_size.append(len(i_))

        # 提取 h/m/s
        hms = [parse_hms(ts) for ts in t_]
        src_times.append(pad_sequence_2d(hms, max_len))

    src_items = torch.stack(src_items)                 # [B, L]
    src_times = torch.stack(src_times)  # [B, L, 3]
    data_size = torch.tensor(data_size)                # [B]

    # ===== target =====
    trg_items, trg_times = [], []
    for e in trg_seq:
        _, i_, t_, _, _ = zip(*e)

        trg_items.append(pad_sequence(i_, max_len))

        hms = [parse_hms(ts) for ts in t_]
        trg_times.append(pad_sequence_2d(hms, max_len))

    trg_items = torch.stack(trg_items)                  # [B, L]
    trg_times = torch.stack(trg_times)  # [B, L, 3]

    return src_items, src_times, trg_items, trg_times, data_size


def gen_eval_batch(batch, data_source, max_len):
    src_seq, trg_seq = zip(*batch)

    # ===== source =====
    src_items, src_times, data_size = [], [], []
    for e in src_seq:
        _, i_, t_, _, _ = zip(*e)

        src_items.append(pad_sequence(i_, max_len))
        data_size.append(len(i_))

        hms = [parse_hms(ts) for ts in t_]
        src_times.append(pad_sequence_2d(hms, max_len))

    # print(len(src_items))
    # print(src_items)
    src_items = torch.stack(src_items)                       # [B, L]
    #print(len(src_times))
    src_times = torch.stack(src_times)    # [B, L, 3]
    data_size = torch.tensor(data_size)                      # [B]

    # ===== target (only next step) =====
    trg_items, trg_times = [], []
    for e in trg_seq:
        _, i_, t_, _, _ = zip(*e)

        trg_items.append(pad_sequence(i_, 1))

        hms = [parse_hms(ts) for ts in t_]
        trg_times.append(pad_sequence_2d(hms, 1))

    trg_items = torch.stack(trg_items)                       # [B, 1]
    trg_times = torch.stack(trg_times)    # [B, 1, 3]

    return src_items, src_times, trg_items, trg_times, data_size


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
