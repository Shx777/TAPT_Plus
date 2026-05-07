import math
import torch
import torch.nn as nn
from basemodel import BaseModel


class RMTPPModel(BaseModel):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.layer_temporal_embed = nn.Linear(1, self.hidden_size)
        self.layer_rnn = nn.RNN(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                num_layers=1,
                                batch_first=True)
        self.hidden_to_intensity_logits = nn.Linear(self.hidden_size, self.num_event_types)

        self.w_t = nn.Parameter(torch.zeros(1, self.num_event_types))
        self.b_t = nn.Parameter(torch.zeros(1, self.num_event_types))

        nn.init.xavier_uniform_(self.w_t)
        nn.init.xavier_uniform_(self.b_t)

    def evolve_and_get_intensity(self, hidden_seq, delta_time_seq):
        # [b, n-1, 1, N]
        past_influence = self.hidden_to_intensity_logits(hidden_seq[..., None, :])
        curr_influence = self.w_t[None, None, :] * delta_time_seq[..., None]
        base_intensity = self.b_t[None, None, :]
        intensity = past_influence + curr_influence + base_intensity
        intensity = intensity.clamp(max=math.log(1e5)).exp()
        return intensity

    def forward(self, batch):
        time_seq, delta_time_seq, mark_seq, _, _ = batch
        mark_embedding = self.layer_type_emb(mark_seq)
        time_embedding = self.layer_temporal_embed(time_seq[..., None])
        # [b, n, d]
        hidden_seq, _ = self.layer_rnn(mark_embedding + time_embedding)
        # [b, n-1, 1, N]
        intensity = self.evolve_and_get_intensity(hidden_seq=hidden_seq[:, :-1, :],
                                                  delta_time_seq=delta_time_seq[:, 1:][..., None])
        # [b, n-1, N
        intensity = intensity.squeeze(dim=-2)
        return intensity, hidden_seq

    def loglike_loss(self, batch):
        time_seq, delta_time_seq, mark_seq, batch_non_pad_mask, _ = batch
        # hidden: [b, n, d]; intensity: [b, n-1, N]
        intensity, hidden_seq = self.forward(batch)
        # hidden: [b, n-1, d]
        hidden_seq_right_shift = hidden_seq[..., :-1, :]

        # [b, n-1, num_samples]
        delta_time_sample_right_shift = self.make_dtime_loss_samples(time_delta_seq=delta_time_seq[:, 1:])
        # [b, n-1, 1, N]
        intensity_right_shift = self.evolve_and_get_intensity(hidden_seq=hidden_seq_right_shift,
                                                              delta_time_seq=delta_time_sample_right_shift)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=intensity,
            lambdas_loss_samples=intensity_right_shift,
            time_delta_seq=delta_time_seq[:, 1:],
            seq_mask=batch_non_pad_mask[:, 1:],
            type_seq=mark_seq[:, 1:]
        )

        loss = -(event_ll - non_event_ll).sum()

        return loss, num_events





















