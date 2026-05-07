from train import TrainFactory
from mclp import MCLP
from rotan import ROTAN
from geosan import GeoSAN
from stan import STAN
exp_config = {'data_name': None, 'scenario': None, 'max_len': None, 'device': None}
train_config = {'batch_size': 128, 'tolerance': 10, 'lr': 1e-3, 'seed': 42}
test_config = {'batch_size': 128, 'k_list': [1, 5, 10]}

model_list = {'MCLP': {'model': MCLP, 'config': {'d_model': 64, 'topic_num': 10, 'encoder_type': 'trans', 'at_type': 'attn'}},
              'ROTAN': {'model': ROTAN, 'config': {'d_model': 64, 'time_embed_dim': 32, 'gps_embed_dim': 64, 'transformer_nhead': 2, 'transformer_nhid': 1024, 'transformer_nlayers': 2, 'max_len': 100}},
              'GeoSAN': {'model': GeoSAN, 'config': {'d_model': 64, 'max_len': 300, 'drop_ratio': 0.5, 'depth': 2, 'device': 'cuda:0'}},
              'STAN': {'model': STAN, 'config': {'d_model': 10, 'max_len': 300, 'device': 'cuda:0'}},
              }
#aaa

if __name__ == '__main__':
    # 'Brightkite', 'Gowalla', 'Weeplaces', 'NYC', 'TKY', 'SIN'
    for data_name in ['Brightkite']:
        if data_name in ['NYC', 'TKY', 'SIN']:
            train_config['batch_size'] = 12
            test_config['batch_size'] = 12
        else:
            train_config['batch_size'] = 12
            test_config['batch_size'] = 12

        exp_config['data_name'] = data_name
        exp_config['scenario'] = 'exp'
        train_config['tolerance'] = 20
        if exp_config['data_name'] in ['NYC', 'TKY', 'SIN', 'Gowalla']:
            exp_config['max_len'] = 100
        else:
            exp_config['max_len'] = 300
        exp_config['device'] = 'cuda:0'

        # 'GRU4Rec', 'SASRec', 'TriMLP', 'FMLP4Rec', 'NextItNet'
        for model_name in ['STAN']:
            model = model_list[model_name]['model']
            model_config = model_list[model_name]['config']
            model_config['model_name'] = model_name
            model_config['max_len'] = exp_config['max_len']

            train_factory = TrainFactory(model, exp_config, model_config, train_config, test_config)
            train_factory.train_model()
            train_factory.validate_model()
