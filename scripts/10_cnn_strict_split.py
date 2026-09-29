"""
模块 10：在严格划分（去邻近）下重跑 CNN vs 组成基线
—— 补上最后一个方法论缺口

模块 09 的结论（随机划分下 CNN ≈ 4-mer）是在随机划分上得到的。
本模块换成去邻近划分（整块基因组区域划分），看结论是否依然成立。
"""
import csv, os, re, sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

BASE = Path(r"D:\code\nap-strep")
PROC = BASE / "data" / "processed"
OUT_ROOT = Path(os.environ.get("NAP_OUT_ROOT", str(BASE)))
TABDIR = OUT_ROOT / "results" / "tables"
FIGDIR = OUT_ROOT / "results" / "figures"
TABDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

TIERS = ["N1", "N2", "N3a", "N3b", "N4", "N5"]
TIER_LABEL = {
    "N1": "N1 random", "N2": "N2 GC-matched", "N3a": "N3a mono shuffle",
    "N3b": "N3b dinuc shuffle", "N4": "N4 flanking", "N5": "N5 extreme AT",
}
BASES = "ACGT"
LEN = 201
KMER = 4
SPLIT = 0.8
BLOCKS = 10
SEED = 42
SEEDS = [0, 1, 2, 3, 4]
EPOCHS, PATIENCE, BATCH, LR_ = 25, 4, 128, 1e-3

torch.set_num_threads(4)


def read_fasta(path):
    recs, name, chunks = [], None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(chunks)))
                name, chunks = line[1:], []
            else:
                chunks.append(line)
    if name is not None:
        recs.append((name, "".join(chunks)))
    return recs


def position_of(name):
    """名字形如 FR845719_3811_4024[_flank+400|_rep1]，取第 2 段作位置"""
    parts = name.split("_")
    if len(parts) < 2:
        return -1
    m = re.match(r"\d+", parts[1])
    return int(m.group()) if m else -1


def one_hot(seqs):
    idx = {b: i for i, b in enumerate(BASES)}
    out = np.zeros((len(seqs), 4, LEN), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s.upper()[:LEN]):
            k = idx.get(ch)
            if k is not None:
                out[i, k, j] = 1.0
    return out


def kmer_feat(seqs, k=KMER):
    keys, vocab = [], {}
    for j in range(4 ** k):
        t, x = [], j
        for _ in range(k):
            t.append(BASES[x % 4]); x //= 4
        keys.append("".join(t))
    vocab = {s: i for i, s in enumerate(keys)}
    out = np.zeros((len(seqs), 4 ** k), dtype=np.float32)
    for i, s in enumerate(seqs):
        u = s.upper()
        for j in range(len(u) - k + 1):
            p = vocab.get(u[j:j + k])
            if p is not None:
                out[i, p] += 1
    return out / max(len(seqs[0]) - k + 1, 1)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 64, 11, padding=5), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(64 * 50, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_cnn(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                  torch.from_numpy(ytr.astype(np.float32))),
                    batch_size=BATCH, shuffle=True, num_workers=0)
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR_)
    lossf = nn.BCEWithLogitsLoss()
    Xva_t = torch.from_numpy(Xva)
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xva_t)).numpy()
        auc = roc_auc_score(yva, p)
        if auc > best_auc + 1e-4:
            best_auc, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    return model, best_auc


def proximity_split(positions, frac, seed):
    """把位置量化成区块，整块划分"""
    lo, hi = positions.min(), positions.max()
    grp = np.clip(((positions - lo) / max(hi - lo, 1) * BLOCKS).astype(int),
                  0, BLOCKS - 1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(BLOCKS)
    target = int(len(positions) * frac)
    take, tot = [], 0
    for b in order:
        if tot >= target:
            break
        take.append(b); tot += int((grp == b).sum())
    mask = np.isin(grp, take)
    return np.flatnonzero(~mask), np.flatnonzero(mask)


# ================= 读数据 =================
pos_rec = read_fasta(PROC / "positives.fa")
pos = [s for _, s in pos_rec]
pos_p = np.array([position_of(n) for n, _ in pos_rec])
tier = {}
for t in TIERS:
    rec = read_fasta(PROC / f"negatives_{t}.fa")
    tier[t] = ([s for _, s in rec], np.array([position_of(n) for n, _ in rec]))

print("=" * 80)
print("严格划分（去邻近）下的 CNN vs 组成基线")
print("=" * 80)
print(f"正样本 {len(pos)} 条；区块数 {BLOCKS}；CNN 种子数 {len(SEEDS)}")

Xpos_hot = one_hot(pos)
Xpos_km = kmer_feat(pos)

rows = []
res_kmer, res_cnn = {}, {}
for t in TIERS:
    neg, neg_p = tier[t]
    seqs = pos + neg
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    positions = np.r_[pos_p, neg_p]

    tr, te = proximity_split(positions, SPLIT, SEED)

    # ---- 4-mer + 逻辑回归 ----
    X = np.vstack([Xpos_km, kmer_feat(neg)])
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X[tr], y[tr])
    a_km = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
    res_kmer[t] = a_km

    # ---- CNN ----
    Xh = np.concatenate([Xpos_hot, one_hot(neg)])
    # 从训练集里再切一小块做早停
    r = np.random.default_rng(SEED)
    idx = r.permutation(len(tr)); k = int(len(tr) * 0.15)
    va, tr2 = tr[idx[:k]], tr[idx[k:]]
    aucs = []
    t0 = time.time()
    for sd in SEEDS:
        m, _ = train_cnn(Xh[tr2], y[tr2], Xh[va], y[va], sd)
        with torch.no_grad():
            p = torch.sigmoid(m(torch.from_numpy(Xh[te]))).numpy()
        aucs.append(roc_auc_score(y[te], p))
    res_cnn[t] = (float(np.mean(aucs)), float(np.min(aucs)), float(np.max(aucs)))
    print(f"{TIER_LABEL[t]:<18} 4-mer {a_km:.4f} | "
          f"CNN {res_cnn[t][0]:.4f} ({res_cnn[t][1]:.4f}~{res_cnn[t][2]:.4f}) "
          f"| Δ {res_cnn[t][0]-a_km:+.4f}  [{time.time()-t0:.0f}s]")
    rows.append(dict(tier=t, split="proximity",
                     auc_4mer=round(a_km, 4),
                     auc_cnn_mean=round(res_cnn[t][0], 4),
                     auc_cnn_min=round(res_cnn[t][1], 4),
                     auc_cnn_max=round(res_cnn[t][2], 4)))

tab = TABDIR / "10_cnn_strict_split.csv"
with tab.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print()
print("表已保存：", tab)

# ================= 画图 =================
fig, ax = plt.subplots(figsize=(10.5, 5))
x = np.arange(len(TIERS)); w = 0.38
ax.bar(x - w/2, [res_kmer[t] for t in TIERS], w, label="4-mer (composition)", color="#2b6cb0")
ax.bar(x + w/2, [res_cnn[t][0] for t in TIERS], w, label="CNN (sequence)", color="#38a169")
ax.errorbar(x + w/2, [res_cnn[t][0] for t in TIERS],
            yerr=[[res_cnn[t][0]-res_cnn[t][1] for t in TIERS],
                  [res_cnn[t][2]-res_cnn[t][0] for t in TIERS]],
            fmt="none", ecolor="black", capsize=4)
ax.axhline(0.5, ls="--", lw=1.2, color="grey")
ax.set_xticks(x)
ax.set_xticklabels([TIER_LABEL[t] for t in TIERS], rotation=25, ha="right", fontsize=9)
ax.set_ylabel("AUC (proximity-aware split)")
ax.set_ylim(0, 1.05)
ax.set_title("Composition vs sequence model under a strict split")
ax.legend(fontsize=10)
ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(FIGDIR / "10_cnn_strict_split.png", dpi=160)
print("图已保存：", FIGDIR / "10_cnn_strict_split.png")

# ================= 结论 =================
print()
print("=" * 80)
print("核心结论（严格划分下）")
print("=" * 80)
d = res_cnn["N2"][0] - res_kmer["N2"]
print(f"N2 上：4-mer {res_kmer['N2']:.4f}  CNN {res_cnn['N2'][0]:.4f}  Δ {d:+.4f}")
lo, hi = res_cnn["N2"][1], res_cnn["N2"][2]
if lo <= res_kmer["N2"] <= hi:
    print("→ CNN 的种子区间覆盖了 4-mer 的值：严格划分下结论不变，顺序仍无额外贡献")
elif d > 0.02:
    print("→ 严格划分下 CNN 明显超过 4-mer：顺序信息的价值在难划分下才显现")
else:
    print("→ 严格划分下 CNN 未超过 4-mer：结论不变")