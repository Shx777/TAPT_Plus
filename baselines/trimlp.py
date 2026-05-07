import torch
import torch.nn as nn


def global_kernel(seq_len):
    mask = torch.triu(torch.ones([seq_len, seq_len]))
    matrix = torch.ones([seq_len, seq_len])
    matrix = matrix.masked_fill(mask == 0.0, -1e9)
    kernel = nn.parameter.Parameter(matrix, requires_grad=True)
    return kernel


def local_kernel(seq_len, n_session):
    mask = torch.zeros([seq_len, seq_len])
    for i in range(0, seq_len, seq_len // n_session):
        mask[i:i + seq_len // n_session, i:i + seq_len // n_session] = torch.ones(
            [seq_len // n_session, seq_len // n_session])
    mask = torch.triu(mask)
    matrix = torch.ones([seq_len, seq_len])
    matrix = matrix.masked_fill(mask == 0.0, -1e9)
    kernel = nn.parameter.Parameter(matrix, requires_grad=True)
    return kernel


class TriMixer(nn.Module):
    def __init__(self, seq_len, n_session, act=nn.ReLU()):
        super().__init__()
        assert seq_len % n_session == 0
        self.l = seq_len
        self.n_s = n_session
        self.act = act
        self.local_mixing = local_kernel(self.l, self.n_s)
        self.global_mixing = global_kernel(self.l)

    def forward(self, x):
        x = self.act(torch.matmul(x.permute(0, 2, 1), self.global_mixing.softmax(dim=-1)))
        x = self.act(torch.matmul(x, self.local_mixing.softmax(dim=-1))).permute(0, 2, 1)
        return x


class TriMLP(nn.Module):
    def __init__(self, n_item, **kwargs):
        super().__init__()
        d_model = kwargs['d_model']
        seq_len = kwargs['max_len']
        p_drop = kwargs['p_drop']
        n_session = kwargs['n_session']
        self.emb_item = nn.Embedding(n_item, d_model, padding_idx=0)
        self.drop = nn.Dropout(p_drop)
        self.mixer = TriMixer(seq_len, n_session)
        self.final_layer = nn.Linear(d_model, n_item)

    def forward(self, x, dsz):
        x = self.drop(self.emb_item(x))
        mixer_output = self.mixer(x)
        if self.training:
            output = self.final_layer(mixer_output)
        else:
            output = mixer_output[torch.arange(dsz.size(0)), dsz - 1, :]
            output = self.final_layer(output).detach()
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))