import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def parse_seq_string(s: str):
    if s is None: return []
    s = str(s).strip()
    if not s or s.lower() == "nan": return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
        toks = [t.strip().strip("'\"") for t in s.split(",")]
    else:
        s = re.sub(r"[^\d]+", ",", s)
        toks = [t for t in s.split(",") if t]
    out = []
    for t in toks:
        try: out.append(int(t))
        except Exception:
            try: out.append(int(float(t)))
            except Exception: pass
    return out

def seq_to_ids_hash(lst, max_len=50, hash_buckets=262144, pad_id=0, seq_base=2):
    ids = [seq_base + (int(t) % hash_buckets) for t in lst][-max_len:]
    if len(ids) < max_len:
        ids = [pad_id] * (max_len - len(ids)) + ids
    return np.array(ids, dtype=np.int32)

class CTRDataset(Dataset):
    def __init__(self, df, cont_cols, cat_cols, cat_maps, seq_col, max_seq_len, hash_buckets, cat_cards, label_col=None):
        self.df = df.reset_index(drop=True)
        self.cont_cols, self.cat_cols = cont_cols, cat_cols
        self.cat_maps, self.seq_col = cat_maps, seq_col
        self.max_seq_len = max_seq_len
        self.hash_buckets = hash_buckets
        self.cat_cards = cat_cards
        self.has_label = label_col is not None
        self.label_col = label_col
        
        self.Xc = self.df[self.cont_cols].astype(np.float32).fillna(0.0).values if self.cont_cols else None
        self.Xcats = {c: self.df[c].map(self.cat_maps[c]).fillna(self.cat_cards[c]).astype(np.int64).values for c in self.cat_cols}
        if self.has_label:
            self.y = self.df[self.label_col].astype(np.float32).values
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        xc = torch.from_numpy(self.Xc[idx]) if self.Xc is not None else torch.empty(0)
        cats = {c: torch.tensor(self.Xcats[c][idx], dtype=torch.long) for c in self.cat_cols}
        lst = parse_seq_string(self.df.at[idx, self.seq_col])
        seq = torch.from_numpy(seq_to_ids_hash(lst, self.max_seq_len, self.hash_buckets)).long()
        
        if self.has_label:
            return xc, cats, seq, torch.tensor(self.y[idx], dtype=torch.float32)
        return xc, cats, seq
