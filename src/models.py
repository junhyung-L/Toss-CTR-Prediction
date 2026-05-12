import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

# =============== Sequence Backbones ===============

class DINBackbone(nn.Module):
    """Standard DIN Backbone (Attention over embeddings)"""
    def __init__(self, vocab_size, emb_dim, padding_idx=0, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.att = DINActivationUnit(emb_dim, emb_dim, hidden=[64, 32], dropout=dropout)
        
    def forward(self, seq_ids, q_vec):
        K = self.emb(seq_ids)                 # (B,L,D)
        logits_w = self.att(q_vec, K)          # (B,L)
        maskL = (seq_ids != PAD_ID)
        neg = torch.finfo(logits_w.dtype).min
        logits_w = logits_w.masked_fill(~maskL, neg)
        valid = maskL.any(dim=1, keepdim=True)
        maxv = torch.where(valid,
                           logits_w.max(dim=1, keepdim=True).values,
                           torch.zeros_like(logits_w[:, :1]))
        logits_w = torch.where(valid, logits_w - maxv, torch.zeros_like(logits_w))
        alpha = torch.where(valid, torch.softmax(logits_w, dim=1), torch.zeros_like(logits_w))
        alpha = torch.nan_to_num(alpha, nan=0.0)
        interest = torch.sum(K * alpha.unsqueeze(2), dim=1)   # (B,D)
        return interest

class DIENBackbone(nn.Module):
    """DIEN Backbone (GRU + Attention over hidden states)"""
    def __init__(self, vocab_size, emb_dim, padding_idx=0, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
        self.att = DINActivationUnit(emb_dim, emb_dim, hidden=[64, 32], dropout=dropout)
        
    def forward(self, seq_ids, q_vec):
        K0 = self.emb(seq_ids)                 # (B,L,D)
        H, _ = self.gru(K0)                    # (B,L,D)
        logits_w = self.att(q_vec, H)          # (B,L)
        maskL = (seq_ids != PAD_ID)
        neg = torch.finfo(logits_w.dtype).min
        logits_w = logits_w.masked_fill(~maskL, neg)
        valid = maskL.any(dim=1, keepdim=True)
        maxv = torch.where(valid,
                           logits_w.max(dim=1, keepdim=True).values,
                           torch.zeros_like(logits_w[:, :1]))
        logits_w = torch.where(valid, logits_w - maxv, torch.zeros_like(logits_w))
        alpha = torch.where(valid, torch.softmax(logits_w, dim=1), torch.zeros_like(logits_w))
        alpha = torch.nan_to_num(alpha, nan=0.0)
        interest = torch.sum(H * alpha.unsqueeze(2), dim=1)   # (B,D)
        return interest

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1,max_len,d)
        
    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]

class BSTBackbone(nn.Module):
    """BST Backbone (Transformer Encoder + Attention)"""
    def __init__(self, vocab_size, emb_dim, nhead=4, num_layers=2, dim_ff=128, dropout=0.2, padding_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.pos = PositionalEncoding(emb_dim, max_len=2048)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=dim_ff,
                                                   batch_first=True, dropout=dropout, activation="relu", norm_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.att = DINActivationUnit(emb_dim, emb_dim, hidden=[64, 32], dropout=dropout)
        
    def forward(self, seq_ids, q_vec):
        K0 = self.emb(seq_ids)                 # (B,L,D)
        K0 = self.pos(K0)
        key_pad_mask = (seq_ids == PAD_ID)     # True for PAD (B,L)
        H = self.enc(K0, src_key_padding_mask=key_pad_mask)  # (B,L,D)
        logits_w = self.att(q_vec, H)          # (B,L)
        neg = torch.finfo(logits_w.dtype).min
        logits_w = logits_w.masked_fill(key_pad_mask, neg)
        valid = (~key_pad_mask).any(dim=1, keepdim=True)
        maxv = torch.where(valid,
                           logits_w.max(dim=1, keepdim=True).values,
                           torch.zeros_like(logits_w[:, :1]))
        logits_w = torch.where(valid, logits_w - maxv, torch.zeros_like(logits_w))
        alpha = torch.where(valid, torch.softmax(logits_w, dim=1), torch.zeros_like(logits_w))
        alpha = torch.nan_to_num(alpha, nan=0.0)
        interest = torch.sum(H * alpha.unsqueeze(2), dim=1)   # (B,D)
        return interest

# =============== Unified Model ===============

class DCN_SEQ_Model(nn.Module):
    def __init__(self, cont_dim, cat_cards, seq_vocab_size, target_name=None,
                 seq_emb_dim=64, seq_backbone="din",
                 bst_cfg=None, deep_units=[512, 256, 128],
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

        # Query(target) 임베딩
        self.target_name = target_name if (target_name in self.cat_embs) else None
        if self.target_name is not None:
            tdim = self.cat_embs[self.target_name].embedding_dim
        else:
            tdim = seq_emb_dim
        self.proj_t = nn.Linear(tdim, seq_emb_dim, bias=False) if tdim != seq_emb_dim else nn.Identity()

        # Sequence backbone
        self.seq_backbone = seq_backbone
        if seq_backbone == "bst":
            cfg = bst_cfg or {"nhead": 4, "num_layers": 2, "dim_ff": 128}
            self.seq_enc = BSTBackbone(seq_vocab_size, seq_emb_dim,
                                       nhead=cfg["nhead"], num_layers=cfg["num_layers"], dim_ff=cfg["dim_ff"],
                                       dropout=dropout, padding_idx=PAD_ID)
        elif seq_backbone == "dien":
            self.seq_enc = DIENBackbone(seq_vocab_size, seq_emb_dim, padding_idx=PAD_ID, dropout=dropout)
        else: # "din" fallback
            self.seq_enc = DINBackbone(seq_vocab_size, seq_emb_dim, padding_idx=PAD_ID, dropout=dropout)

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

        # Query(target) vector
        if self.target_name is not None:
            q = self.cat_embs[self.target_name](xcats[self.target_name])  # (B,Dt)
            q = self.proj_t(q)                                    # (B,D)
        else:
            # Fallback: zero vector or mean would be handled in backbone if needed, 
            # here we pass zero vector as query if no target_name is available
            q = torch.zeros(seq_ids.size(0), self.proj_t.out_features if hasattr(self.proj_t, 'out_features') else seq_ids.size(-1), device=seq_ids.device)

        # Sequence → interest vector
        interest = self.seq_enc(seq_ids, q)                        # (B,D)

        # 출력 결합
        z = torch.cat([x_cross, x_deep, interest], dim=1)
        return self.head(z).squeeze(1)

# Keep old model for backward compatibility or direct use
class DCN_DIN_Model(nn.Module):
    def __init__(self, cont_dim, cat_cards, seq_vocab_size, target_name=None,
                 seq_emb_dim=64, deep_units=[512, 256, 128],
                 cross_layers=3, cross_low_rank=32, cross_num_experts=4, dropout=0.2):
        super().__init__()
        self.base_model = DCN_SEQ_Model(cont_dim, cat_cards, seq_vocab_size, target_name,
                                         seq_emb_dim, seq_backbone="din",
                                         deep_units=deep_units, cross_layers=cross_layers,
                                         cross_low_rank=cross_low_rank, cross_num_experts=cross_num_experts,
                                         dropout=dropout)
    def forward(self, xc, xcats, seq_ids):
        return self.base_model(xc, xcats, seq_ids)
