import torch
import torch.nn as nn
import torch.nn.functional as F

class TAPTPlus(nn.Module):
    def __init__(self, num_pois, dp, dt, d,
                 s_experts=2, p_experts=3, t_experts=3,
                 num_layers=2, device='cpu'):
        super().__init__()
        self.device = device
        self.dp = dp
        self.dt = dt
        self.d = d
        self.s, self.p, self.t = s_experts, p_experts, t_experts

        # --- JRL embeddings ---
        self.poi_emb = nn.Embedding(num_pois, dp, padding_idx=0)
        self.hour_emb = nn.Embedding(24, dt//3)
        self.min_emb = nn.Embedding(60, dt//3)
        self.sec_emb = nn.Embedding(60, dt//3)
        self.joint_fc = nn.Linear(dp + dt, d)
        self.relu = nn.ReLU()

        # --- SASRec backbone ---
        encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=4)
        self.sasrec = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- PLE experts ---
        self.shared_experts = nn.ModuleList([nn.Linear(d, d) for _ in range(s_experts)])
        self.poi_experts = nn.ModuleList([nn.Linear(d, d) for _ in range(p_experts)])
        self.time_experts = nn.ModuleList([nn.Linear(d, d) for _ in range(t_experts)])

        # Task-specific gates
        self.gate_p = nn.Linear(d, s_experts + p_experts)
        self.gate_t = nn.Linear(d, s_experts + t_experts)

        # Task heads
        self.poi_head = nn.Linear(d, dp)  # project to POI embedding space externally
        self.time_head = nn.Linear(d, 1)

        # Dynamic Gating Network
        self.dgn_fc1 = nn.Linear(2*d, d)
        self.dgn_fc2 = nn.Linear(d, 2)

    def forward(self, poi_seq, time_seq, seq_lengths=None):
        """
        poi_seq: [B, L]
        time_seq: [B, L, 3] -> hour, minute, second
        seq_lengths: [B], optional, actual length of each sequence
        """
        B, L = poi_seq.size()

        # --- JRL: embeddings ---
        poi_e = self.poi_emb(poi_seq)  # [B, L, dp]
        h_e = self.hour_emb(time_seq[:,:,0])
        m_e = self.min_emb(time_seq[:,:,1])
        s_e = self.sec_emb(time_seq[:,:,2])
        time_e = torch.cat([h_e, m_e, s_e], dim=-1)  # [B, L, dt]

        joint_e = self.relu(self.joint_fc(torch.cat([poi_e, time_e], dim=-1)))  # [B, L, d]

        # --- SASRec backbone ---
        E = self.sasrec(joint_e.permute(1,0,2)).permute(1,0,2)  # [B, L, d]

        # --- PLE: experts ---
        shared_outs = torch.stack([F.relu(expert(E)) for expert in self.shared_experts], dim=1)  # [B, s, L, d]
        poi_outs = torch.stack([F.relu(expert(E)) for expert in self.poi_experts], dim=1)       # [B, p, L, d]
        time_outs = torch.stack([F.relu(expert(E)) for expert in self.time_experts], dim=1)     # [B, t, L, d]

        Ep = torch.cat([shared_outs, poi_outs], dim=1)  # [B, s+p, L, d]
        Et = torch.cat([shared_outs, time_outs], dim=1) # [B, s+t, L, d]

        # --- Task-specific gating ---
        last_e = E[:, -1, :]  # [B, d], use last position for gating

        g_p = F.softmax(self.gate_p(last_e), dim=-1)  # [B, s+p]
        g_t = F.softmax(self.gate_t(last_e), dim=-1)  # [B, s+t]

        # Weighted sum over experts at last position
        h_p = torch.einsum('bp,bpld->bd', g_p, Ep[:,:,-1,:])  # [B, d]
        h_t = torch.einsum('bt,btld->bd', g_t, Et[:,:,-1,:])  # [B, d]

        # --- Task heads ---
        zp = self.poi_head(h_p)       # [B, dp]
        yt_hat = self.time_head(h_t).squeeze(-1)  # [B]

        # --- Dynamic Gating Network ---
        z_joint = torch.cat([h_p, h_t], dim=-1)  # [B, 2d]
        gate = F.softmax(self.dgn_fc2(F.tanh(self.dgn_fc1(z_joint))), dim=-1)  # [B, 2]

        if self.training:
            return zp, yt_hat, gate
        else:
            # In inference, only return last position predictions
            return zp.detach(), yt_hat.detach(), gate.detach()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
