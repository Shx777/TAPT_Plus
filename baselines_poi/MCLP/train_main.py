from baselines_poi.MCLP.train import TrainFactory
from baselines_poi.MCLP.mclp import MCLP


exp_config = {'data_name': None, 'scenario': None, 'max_len': None, 'device': None}
train_config = {'batch_size': 128, 'tolerance': 10, 'lr': 1e-3, 'seed': 42}
test_config = {'batch_size': 128, 'k_list': [1, 5, 10]}

model_list = {'MCLP': {'model': MCLP, 'config': {'d_model': 64, 'topic_num': 1500, 'encoder_type': 'trans', 'at_type': 'attn'}}}

if __name__ == '__main__':
    # 'Brightkite', 'Gowalla', 'Weeplaces', 'NYC', 'TKY', 'SIN'
    for data_name in ['NYC']:
        if data_name in ['NYC', 'TKY', 'SIN']:
            train_config['batch_size'] = 1024
            test_config['batch_size'] = 512
        else:
            train_config['batch_size'] = 30
            test_config['batch_size'] = 30

        exp_config['data_name'] = data_name
        exp_config['scenario'] = 'exp'
        train_config['tolerance'] = 20
        if exp_config['data_name'] in ['NYC', 'TKY', 'SIN', 'Gowalla']:
            exp_config['max_len'] = 100
        else:
            exp_config['max_len'] = 300
        exp_config['device'] = 'cuda:0'

        # 'GRU4Rec', 'SASRec', 'TriMLP', 'FMLP4Rec', 'NextItNet'
        for model_name in ['MCLP']:
            model = model_list[model_name]['model']
            model_config = model_list[model_name]['config']
            model_config['model_name'] = model_name
            model_config['max_len'] = exp_config['max_len']

            train_factory = TrainFactory(model, exp_config, model_config, train_config, test_config)
            train_factory.train_model()
            train_factory.validate_model()
