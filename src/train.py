import os
import json
import gc
import warnings
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from data import CTRDataset
from models import DCN_SEQ_Model

warnings.filterwarnings("ignore")

def set_seed(seed=42):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def infer_device(arg="auto"):
    if arg == "cpu": return torch.device("cpu")
    if arg == "cuda" or (arg == "auto" and torch.cuda.is_available()):
        print("Using CUDA")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")

def main(args):
    set_seed(args.seed)
    device = infer_device(args.device)

    # Load data
    print("[1/7] Loading data...")
    if not os.path.exists(args.train_path) or not os.path.exists(args.test_path):
        raise FileNotFoundError("Train or Test parquet file not found. Please provide valid paths.")
        
    train = pd.read_parquet(args.train_path)
    test = pd.read_parquet(args.test_path)
    
    assert args.label_col in train.columns and args.seq_col in train.columns

    if 0 < args.sample_subset < 1.0:
        print(f"Subsampling to {args.sample_subset*100:.1f}%")
        train = train.sample(frac=args.sample_subset, random_state=args.seed).reset_index(drop=True)

    must_drop = {args.label_col, args.seq_col, args.id_col}
    base_cols = [c for c in train.columns if c not in must_drop]

    # Target selection & feature separation
    fixed_cats = [c for c in ["gender", "age_group", "inventory_id", "day_of_week", "hour"] if c in base_cols]
    all_lfeats = [c for c in base_cols if c.startswith("l_feat_")]

    preferred_targets = ["ad_id", "creative_id", "l_feat_14"]
    present_targets = [c for c in preferred_targets if c in base_cols]
    target_name = present_targets[0] if len(present_targets) > 0 else None
    if target_name is None and "l_feat_14" in all_lfeats:
        target_name = "l_feat_14"

    cand_cats = fixed_cats.copy()
    tr_tmp, va_tmp = train_test_split(train, test_size=args.test_size, random_state=args.seed, stratify=train[args.label_col])
    for c in all_lfeats:
        try:
            nunq = max(tr_tmp[c].nunique(dropna=True), va_tmp[c].nunique(dropna=True))
        except Exception:
            nunq = pd.concat([tr_tmp[c], va_tmp[c]], axis=0).nunique(dropna=True)
        if nunq <= args.cat_card_max:
            cand_cats.append(c)

    if target_name is not None and target_name not in cand_cats and target_name in base_cols:
        cand_cats.append(target_name)

    cont_cols = [c for c in base_cols if c not in cand_cats]
    print(f"[1.1] target_name={target_name} | continuous={len(cont_cols)} | categorical={len(cand_cats)}")

    # Split
    tr, va = train_test_split(train, test_size=args.test_size, random_state=args.seed, stratify=train[args.label_col])
    del train; gc.collect()

    # Categorical maps
    cat_maps, cat_cards = {}, {}
    for c in cand_cats:
        cats = pd.Categorical(tr[c])
        cat_maps[c] = {v: i for i, v in enumerate(cats.categories)}
        cat_cards[c] = len(cats.categories)

    # Datasets
    PAD_ID, SEQ_BASE = 0, 2
    seq_vocab_size = args.hash_buckets + SEQ_BASE

    ds_tr = CTRDataset(tr, cont_cols, cand_cats, cat_maps, args.seq_col, args.max_seq_len, args.hash_buckets, cat_cards, label_col=args.label_col)
    ds_va = CTRDataset(va, cont_cols, cand_cats, cat_maps, args.seq_col, args.max_seq_len, args.hash_buckets, cat_cards, label_col=args.label_col)
    ds_te = CTRDataset(test, cont_cols, cand_cats, cat_maps, args.seq_col, args.max_seq_len, args.hash_buckets, cat_cards, label_col=None)

    n_pos = float((tr[args.label_col] == 1).sum())
    n_neg = float(len(tr) - n_pos)
    pos_weight = torch.tensor(max(1.0, n_neg / max(1.0, n_pos)), dtype=torch.float32, device=device)
    print(f"[2/7] Train label: pos={int(n_pos)} neg={int(n_neg)} | pos_weight={pos_weight.item():.3f}")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    # Model
    print(f"[3/7] Building model with backbone={args.seq_backbone}...")
    bst_cfg = {"nhead": args.bst_nhead, "num_layers": args.bst_layers, "dim_ff": args.bst_ffn}
    model = DCN_SEQ_Model(
        cont_dim=len(cont_cols), cat_cards=cat_cards, seq_vocab_size=seq_vocab_size,
        target_name=target_name, seq_emb_dim=args.seq_emb_dim, seq_backbone=args.seq_backbone,
        bst_cfg=bst_cfg, deep_units=args.deep_units,
        cross_layers=args.cross_layers, cross_low_rank=args.cross_low_rank, cross_num_experts=args.cross_num_experts,
        dropout=args.dropout
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_auc, best_state, wait = -1.0, None, 0
    final_epoch, final_prauc = 0, 0.0

    print("[4/7] Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xc, cats, seq, y in dl_tr:
            xc, seq, y = xc.to(device, non_blocking=True), seq.to(device, non_blocking=True), y.to(device, non_blocking=True)
            cats_dev = {k: v.to(device, non_blocking=True) for k, v in cats.items()}
            
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(xc, cats_dev, seq)
                loss = criterion(logits, y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * y.size(0)
            
        tr_loss /= len(ds_tr)
        scheduler.step()

        # Validation
        model.eval()
        va_loss = 0.0
        ys, ps = [], []
        with torch.no_grad():
            for xc, cats, seq, y in dl_va:
                xc, seq = xc.to(device, non_blocking=True), seq.to(device, non_blocking=True)
                cats_dev = {k: v.to(device, non_blocking=True) for k, v in cats.items()}
                with autocast(enabled=(device.type == "cuda")):
                    logits = model(xc, cats_dev, seq)
                    loss = criterion(logits, y.to(device))
                    prob = torch.sigmoid(logits)
                    
                p = prob.detach().cpu().numpy().astype(np.float64)
                p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
                va_loss += loss.item() * len(y)
                ys.append(y.numpy())
                ps.append(p)
                
        va_loss /= len(ds_va)
        y_true = np.concatenate(ys)
        y_prob = np.concatenate(ps)
        auc = roc_auc_score(y_true, y_prob)
        prauc = average_precision_score(y_true, y_prob)
        print(f"Epoch {epoch:02d} | Train Loss: {tr_loss:.5f} | Val Loss: {va_loss:.5f} | AUC: {auc:.6f} | PR-AUC: {prauc:.6f}")

        if auc > best_auc + 1e-5:
            best_auc, final_prauc, final_epoch = auc, prauc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping. Best AUC={best_auc:.6f} at epoch {final_epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Inference
    print("[5/7] Predicting test set...")
    model.eval()
    probs = []
    with torch.no_grad():
        for xc, cats, seq in dl_te:
            xc, seq = xc.to(device, non_blocking=True), seq.to(device, non_blocking=True)
            cats_dev = {k: v.to(device, non_blocking=True) for k, v in cats.items()}
            with autocast(enabled=(device.type == "cuda")):
                logits = model(xc, cats_dev, seq)
                prob = torch.sigmoid(logits)
            p = prob.detach().cpu().numpy().astype(np.float64)
            p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
            probs.append(p)
    probs = np.concatenate(probs)

    # Save
    print("[6/7] Saving outputs...")
    pd.DataFrame({args.id_col: test[args.id_col].values, "clicked": probs}).to_csv(args.output_path, index=False)
    
    meta = {
        "columns": {"continuous": cont_cols, "categorical": cand_cats},
        "seq_vocab": {"type": "hash", "buckets": args.hash_buckets, "pad_id": 0, "oov_id": 1},
        "hyperparameters": {
            "sample_subset": args.sample_subset, "max_seq_len": args.max_seq_len,
            "seq_emb_dim": args.seq_emb_dim, "dropout": args.dropout,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "cross_layers": args.cross_layers, "cross_low_rank": args.cross_low_rank, "cross_num_experts": args.cross_num_experts,
            "deep_units": args.deep_units,
            "seq_backbone": args.seq_backbone,
            "bst_layers": args.bst_layers, "bst_nhead": args.bst_nhead, "bst_ffn": args.bst_ffn
        },
        "performance": {"best_epoch": int(final_epoch), "AUC": float(best_auc), "PR_AUC": float(final_prauc)},
        "target_name": target_name
    }
    
    with open(args.meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[7/7] Done. \n - {args.output_path}\n - {args.meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTR Prediction Training Pipeline")
    parser.add_argument("--train_path", type=str, default="../train.parquet")
    parser.add_argument("--test_path", type=str, default="../test.parquet")
    parser.add_argument("--output_path", type=str, default="./submit_dcn_seq.csv")
    parser.add_argument("--meta_path", type=str, default="./meta_dcn_seq.json")
    parser.add_argument("--label_col", type=str, default="clicked")
    parser.add_argument("--seq_col", type=str, default="seq")
    parser.add_argument("--id_col", type=str, default="ID")
    parser.add_argument("--sample_subset", type=float, default=1.0)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--cat_card_max", type=int, default=200000)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--hash_buckets", type=int, default=262144)
    parser.add_argument("--seq_emb_dim", type=int, default=64)
    parser.add_argument("--deep_units", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--cross_layers", type=int, default=3)
    parser.add_argument("--cross_low_rank", type=int, default=32)
    parser.add_argument("--cross_num_experts", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    
    # New arguments for sequence backbones
    parser.add_argument("--seq_backbone", type=str, default="din", choices=["din", "dien", "bst"],
                        help="Sequence backbone architecture")
    parser.add_argument("--bst_layers", type=int, default=2, help="Number of Transformer layers for BST")
    parser.add_argument("--bst_nhead", type=int, default=4, help="Number of heads for BST")
    parser.add_argument("--bst_ffn", type=int, default=128, help="Hidden dim of FFN in BST")
    
    args = parser.parse_args()
    main(args)
