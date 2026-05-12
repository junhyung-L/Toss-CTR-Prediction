import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID = 0

def emb_dim_from_card(card, cap=64):
    return int(min(cap, max(8, round(1.6 * (card ** 0.25)))))

class CrossLayerMix(nn.Module):
    """CrossNetMix layer (low-rank + experts, simplified)"""
    def __init__(self, dim, low_rank=32, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.U = nn.Parameter(torch.randn(num_experts, dim, low_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(num_experts, low_rank, dim) * 0.01)
        self.C = nn.Parameter(torch.zeros(num_experts, dim))
        self.gating = nn.Linear(dim, num_experts, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x0, x):
        gate = torch.softmax(self.gating(x), dim=-1)          # (B,E)
        Xu = torch.einsum("bd,edr->ber", x, self.U)           # (B,E,r)
        Xuv = torch.einsum("ber,erd->bed", Xu, self.V)        # (B,E,d)
        Xuv = Xuv + self.C                                  # (B,E,d)
        mix = torch.einsum("be,bed->bd", gate, Xuv)           # (B,d)
        out = x0 * mix + x + self.bias                          # cross + residual
        return out

class CrossNetMix(nn.Module):
    def __init__(self, dim, num_layers=3, low_rank=32, num_experts=4):
        super().__init__()
        self.layers = nn.ModuleList([CrossLayerMix(dim, low_rank, num_experts) for _ in range(num_layers)])
        
    def forward(self, x):
        x0 = x
        xl = x
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl

class DINActivationUnit(nn.Module):
    """w_i = MLP([q, k_i, q-k_i, q*k_i])"""
    def __init__(self, dim_q, dim_k, hidden=[64, 32], dropout=0.0):
        super().__init__()
        in_dim = dim_q + dim_k + dim_q + dim_k
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)
        
    def forward(self, q, K):
        B, L, Dk = K.shape
        q_exp = q.unsqueeze(1).expand(-1, L, -1)        # (B,L,Dq)
        feats = [q_exp, K, q_exp - K, q_exp * K]
        z = torch.cat(feats, dim=2)                   # (B,L,·)
        w = self.net(z).squeeze(2)                    # (B,L)
        return w

class DCN_DIN_Model(nn.Module):
    def __init__(self, cont_dim, cat_cards, seq_vocab_size, target_name=None,
                 seq_emb_dim=64, deep_units=[512, 256, 128],
                 cross_layers=3, cross_low_rank=32, cross_num_experts=4, dropout=0.2):
        super().__init__()
        self.has_cont = cont_dim > 0
        if self.has_cont:
            self.bn = nn.BatchNorm1d(cont_dim)

        # categorical embeddings
        self.cat_embs = nn.ModuleDict()
        for name, card in cat_cards.items():
            dim = emb_dim_from_card(card + 1)
            self.cat_embs[name] = nn.Embedding(card + 1 + 1, dim)  # +1 OOV
        self.cat_total_dim = sum(emb.embedding_dim for emb in self.cat_embs.values())

        # base tabular x0
        self.tab_dim = (cont_dim if self.has_cont else 0) + self.cat_total_dim

        # DCN-V2
        self.cross = CrossNetMix(self.tab_dim, num_layers=cross_layers,
                                 low_rank=cross_low_rank, num_experts=cross_num_experts)

        # Deep tower
        deep_layers = []
        in_dim = self.tab_dim
        for h in deep_units:
            deep_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.deep = nn.Sequential(*deep_layers)

        # Sequence + DIN(activation unit)
        self.seq_emb = nn.Embedding(seq_vocab_size, seq_emb_dim, padding_idx=PAD_ID)
        self.target_name = target_name if (target_name in self.cat_embs) else None
        if self.target_name is not None:
            tdim = self.cat_embs[self.target_name].embedding_dim
        else:
            tdim = seq_emb_dim
        self.proj_t = nn.Linear(tdim, seq_emb_dim, bias=False) if tdim != seq_emb_dim else nn.Identity()
        self.din_act = DINActivationUnit(seq_emb_dim, seq_emb_dim, hidden=[64, 32], dropout=dropout)

        # Final head: [cross_out, deep_out, interest_vec]
        final_in = self.tab_dim + deep_units[-1] + seq_emb_dim
        self.head = nn.Sequential(
            nn.Linear(final_in, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, xc, xcats, seq_ids):
        # ----- Tabular -----
        outs = []
        if self.has_cont:
            outs.append(self.bn(xc))
        if len(self.cat_embs) > 0:
            outs.append(torch.cat([self.cat_embs[n](xcats[n]) for n in self.cat_embs], dim=1))
        x0 = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]   # (B, tab_dim)

        # DCN-V2 cross / Deep tower
        x_cross = self.cross(x0)                                  # (B, tab_dim)
        x_deep = self.deep(x0)                                   # (B, deep_dim)

        # ----- DIN -----
        K = self.seq_emb(seq_ids)                                 # (B,L,D)
        if self.target_name is not None:
            q = self.cat_embs[self.target_name](xcats[self.target_name])  # (B,Dt)
            q = self.proj_t(q)                                    # (B,D)
        else:
            mask = (seq_ids != PAD_ID).float().unsqueeze(-1)
            q = (K * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)        # mean fallback

        # Activation unit -> attention logits
        logits_w = self.din_act(q, K)                             # (B,L)
        maskL = (seq_ids != PAD_ID)

        # fp16에서도 안전한 음수값으로 PAD 마스킹
        neg = torch.finfo(logits_w.dtype).min
        logits_w = logits_w.masked_fill(~maskL, neg)

        # 전부 PAD인 행 안전 처리 (alpha=0)
        valid = maskL.any(dim=1, keepdim=True)                    # (B,1)
        maxv = torch.where(valid,
                           logits_w.max(dim=1, keepdim=True).values,
                           torch.zeros_like(logits_w[:, :1]))
        logits_w = torch.where(valid, logits_w - maxv, torch.zeros_like(logits_w))
        alpha = torch.where(valid, torch.softmax(logits_w, dim=1), torch.zeros_like(logits_w))
        alpha = torch.nan_to_num(alpha, nan=0.0)

        interest = torch.sum(K * alpha.unsqueeze(2), dim=1)       # (B,D)

        # 출력 결합
        z = torch.cat([x_cross, x_deep, interest], dim=1)
        return self.head(z).squeeze(1)
