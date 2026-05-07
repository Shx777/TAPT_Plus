import copy
import os
import time
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from torch.utils.data import DataLoader
from collections import Counter
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_factory.lbsn_data import un_serialize

class TrainFactory:
    def __init__(self, model, exp_config, model_config, train_config, test_config):
        super(TrainFactory, self).__init__()

        self.model = model
        self.exp_config = exp_config
        self.model_config = model_config
        self.train_config = train_config
        self.test_config = test_config
        self.data_path, self.result_path, self.model_path = self.make_path()
        self.n_poi, self.train_loader, self.test_loader = self.prepare_data()
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
        max_len = self.exp_config['max_len']
        scenario = self.exp_config['scenario']
        train_batch_size = self.train_config['batch_size']
        test_batch_size = self.test_config['batch_size']

        train_data, test_data = dataset.partition(max_len, scenario)
        train_loader = DataLoader(dataset=train_data, batch_size=train_batch_size,
                                  sampler=LadderSampler(train_data, train_batch_size),
                                  num_workers=12,
                                  prefetch_factor=2,
                                  collate_fn=lambda e: gen_train_batch(e, train_data, max_len))
        test_loader = DataLoader(dataset=test_data,
                                 batch_size=test_batch_size,
                                 num_workers=12,
                                 prefetch_factor=2,
                                 collate_fn=lambda e: gen_eval_batch(e, test_data, max_len))
        return n_poi, train_loader, test_loader

    def construct_model(self):
        n_poi = self.n_poi
        model = self.model(n_poi, **self.model_config)
        if os.path.exists(self.model_path):
            model.load(self.model_path)
        return model

    def train_function(self, dataloader, model, optimizer, epoch):
        device = self.exp_config['device']
        start_time = time.time()
        model.train()
        running_loss = 0.0
        processed_batch = 0
        tqdm_title = '>>> ' + 'Training Epoch: {} >>>'.format(epoch)
        batch_iterator = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc=tqdm_title)
        for batch_idx, (src, trg, dsz) in batch_iterator:
            optimizer.zero_grad()
            src = src.to(device)
            trg = trg.to(device)
            dsz = dsz.to(device)
            logits = model(src, dsz)
            logits = logits.view(-1, logits.size(-1))
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
            for batch_idx, (src, trg, dsz) in batch_iterator:
                src = src.to(device)
                trg = trg.to(device)
                dsz = dsz.to(device)
                scores = model(src, dsz)
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
