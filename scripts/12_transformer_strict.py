"""
模块 12：Transformer 在严格划分（去邻近）下的表现

我们已经知道：随机划分会高估复杂模型（CNN 被高估 0.24）。
本模块看 Transformer 被高估多少，并汇总三种方法的"虚高幅度"。
"""
import csv, os, re, sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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

# 前面对照：随机划分的结果
RANDOM = {
    "4mer": {"N1": 0.8606, "N2": 0.7343, "N3a": 0.9117, "N3b": 0.8327, "N4": 0.8638, "N5": 0.8741},
    "cnn":  {"N1": 0.8723, "N2": 0.7387, "N3a": 0.9298, "N3b": 0.7903, "N4": 0.8623, "N5": 0.8415},
    "tf":   {"N1": 0.8474, "N2": 0.7025, "N3a": 0.8976, "N3b": 0.8355, "N4": 0.8541, "N5": 0.8361},
}
# 前面对照：严格划分下 4-mer 与 CNN 的结果（模块 10）
STRICT_REF = {
    "4mer": {"N1": 0.8220, "N2": 0.5863, "N3a": 0.8869, "N3b": 0.8131, "N4": 0.8296, "N5": 0.8667},
    "cnn":  {"N1": 0.7987, "N2": 0.5024, "N3a": 0.6363, "N3b": 0.5848, "N4": 0.8103, "N5": 0.7861},
}

BASES = "ACGT"
KMER, LEN = 3, 201
N_TOK = LEN // KMER
VOCAB = 4 ** KMER
SPLIT, SEED, BLOCKS = 0.8, 42, 10
SEEDS = [0, 1, 2, 3, 4]
EPOCHS, PATIENCE, BATCH, LR_ = 30, 5, 128, 1e-3

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
    parts = name.split("_")
    if len(parts) < 2:
        return -1
    m = re.match(r"\d+", parts[1])
    return int(m.group()) if m else -1


def proximity_split(positions, frac, seed):
    lo, hi = positions.min(), positions.max()
    grp = np.clip(((positions - lo) / max(hi - lo, 1) * BLOCKS).astype(int), 0, BLOCKS - 1)
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


def tokenize(seqs):
    lut = {b: i for i, b in enumerate(BASES)}
    out = np.zeros((len(seqs), N_TOK), dtype=np.int64)
    for i, s in enumerate(seqs):
        u = s.upper()
        for j in range(N_TOK):
            code, ok = 0, True
            for ch in u[j * KMER:(j + 1) * KMER]:
                k = lut.get(ch)
                if k is None:
                    ok = False; break
                code = code * 4 + k
            out[i, j] = code if ok else 0
    return out


class Block(nn.Module):
    def __init__(self, d, nhead, ff, p):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=p, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.ln2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.ln1(x + self.drop(a))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


class DNATransformer(nn.Module):
    def __init__(self, d=64, nhead=4, nl=2, ff=128, p=0.1):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, N_TOK + 1, d))
        self.blocks = nn.ModuleList([Block(d, nhead, ff, p) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, idx):
        x = self.tok(idx)
        b = x.size(0)
        x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1)
        x = x + self.pos[:, :x.size(1)]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x[:, 0])).squeeze(1)


def train_tf(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                  torch.from_numpy(ytr.astype(np.float32))),
                    batch_size=BATCH, shuffle=True, num_workers=0)
    model = DNATransformer()
    opt = torch.optim.Adam(model.parameters(), lr=LR_)
    lossf = nn.BCEWithLogitsLoss()
    Xva_t = torch.from_numpy(Xva)
    best, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xva_t)).numpy()
        auc = roc_auc_score(yva, p)
        if auc > best + 1e-4:
            best, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    return model


# ================= 主流程 =================
pos_rec = read_fasta(PROC / "positives.fa")
pos = [s for _, s in pos_rec]
pos_p = np.array([position_of(n) for n, _ in pos_rec])
tier = {}
for t in TIERS:
    rec = read_fasta(PROC / f"negatives_{t}.fa")
    tier[t] = ([s for _, s in rec], np.array([position_of(n) for n, _ in rec]))

print("=" * 82)
print("模块 12：Transformer 在严格划分（去邻近）下")
print("=" * 82)

Xpos = tokenize(pos)
res = {}
rows = []
for t in TIERS:
    neg, neg_p = tier[t]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    tr, te = proximity_split(np.r_[pos_p, neg_p], SPLIT, SEED)

    X = np.concatenate([Xpos, tokenize(neg)])
    r = np.random.default_rng(SEED)
    idx = r.permutation(len(tr)); k = int(len(tr) * 0.15)
    va, tr2 = tr[idx[:k]], tr[idx[k:]]

    aucs, t0 = [], time.time()
    for sd in SEEDS:
        m = train_tf(X[tr2], y[tr2], X[va], y[va], sd)
        with torch.no_grad():
            p = torch.sigmoid(m(torch.from_numpy(X[te]))).numpy()
        aucs.append(roc_auc_score(y[te], p))
    res[t] = (float(np.mean(aucs)), float(np.min(aucs)), float(np.max(aucs)))
    print(f"{TIER_LABEL[t]:<18} Transformer {res[t][0]:.4f} "
          f"({res[t][1]:.4f}~{res[t][2]:.4f})  [{time.time()-t0:.0f}s]")
    rows.append(dict(tier=t, auc_mean=round(res[t][0], 4),
                     auc_min=round(res[t][1], 4), auc_max=round(res[t][2], 4)))

tab = TABDIR / "12_transformer_strict.csv"
with tab.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print()
print("表已保存：", tab)

# ================= 三种方法 × 两种划分 =================
print()
print("=" * 82)
print("★ 三种方法在两种划分下的对比")
print("=" * 82)
print(f"{'档次':<18}{'4mer随机':>10}{'4mer严格':>10}{'CNN随机':>10}{'CNN严格':>10}"
      f"{'TF随机':>10}{'TF严格':>10}")
print("-" * 82)
for t in TIERS:
    print(f"{TIER_LABEL[t]:<18}{RANDOM['4mer'][t]:>10.4f}{STRICT_REF['4mer'][t]:>10.4f}"
          f"{RANDOM['cnn'][t]:>10.4f}{STRICT_REF['cnn'][t]:>10.4f}"
          f"{RANDOM['tf'][t]:>10.4f}{res[t][0]:>10.4f}")

print()
print("=" * 82)
print("★ 被随机划分高估的幅度")
print("=" * 82)
print(f"{'档次':<18}{'4-mer':>12}{'CNN':>12}{'Transformer':>14}")
print("-" * 82)
for t in TIERS:
    d1 = RANDOM['4mer'][t] - STRICT_REF['4mer'][t]
    d2 = RANDOM['cnn'][t] - STRICT_REF['cnn'][t]
    d3 = RANDOM['tf'][t] - res[t][0]
    print(f"{TIER_LABEL[t]:<18}{d1:>+12.4f}{d2:>+12.4f}{d3:>+14.4f}")
print()
avg = lambda k, ref: np.mean([RANDOM[k][t] - ref[t] for t in TIERS])
print(f"平均高估幅度：4-mer {avg('4mer', STRICT_REF['4mer']):+.4f}  "
      f"CNN {avg('cnn', STRICT_REF['cnn']):+.4f}  "
      f"Transformer {avg('tf', res):+.4f}")

# ================= 画图 =================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(TIERS)); w = 0.26
axes[0].bar(x - w, [RANDOM['tf'][t] for t in TIERS], w, label="Transformer (random split)", color="#c53030")
axes[0].bar(x, [res[t][0] for t in TIERS], w, label="Transformer (strict split)", color="#805ad5")
axes[0].bar(x + w, [STRICT_REF['4mer'][t] for t in TIERS], w, label="4-mer (strict split)", color="#2b6cb0")
axes[0].axhline(0.5, ls="--", lw=1.2, color="grey")
axes[0].set_xticks(x)
axes[0].set_xticklabels([TIER_LABEL[t] for t in TIERS], rotation=25, ha="right", fontsize=9)
axes[0].set_ylabel("AUC")
axes[0].set_ylim(0, 1.05)
axes[0].set_title("A. Transformer under two splits")
axes[0].legend(fontsize=8.5)
axes[0].grid(alpha=0.25, axis="y")

models = ["4-mer", "CNN", "Transformer"]
d1 = [RANDOM['4mer'][t] - STRICT_REF['4mer'][t] for t in TIERS]
d2 = [RANDOM['cnn'][t] - STRICT_REF['cnn'][t] for t in TIERS]
d3 = [RANDOM['tf'][t] - res[t][0] for t in TIERS]
axes[1].bar(np.arange(3) - 0.26, [np.mean(d1)] * 3, 0.26, color="#2b6cb0")
axes[1].bar(np.arange(3), [np.mean(d2)] * 3, 0.26, color="#38a169")
axes[1].bar(np.arange(3) + 0.26, [np.mean(d3)] * 3, 0.26, color="#805ad5")
axes[1].set_xticks(range(3))
axes[1].set_xticklabels(models)
axes[1].axhline(0, lw=1.2, color="black")
axes[1].set_ylabel("Average AUC overestimation by random split")
axes[1].set_title("B. How much does random splitting inflate each model?")
axes[1].grid(alpha=0.25, axis="y")

plt.tight_layout()
plt.savefig(FIGDIR / "12_transformer_strict.png", dpi=160)
print()
print("图已保存：", FIGDIR / "12_transformer_strict.png")

# ================= 结论 =================
print()
print("=" * 82)
print("结论")
print("=" * 82)
print(f"N2 上的最终对比（严格划分）：")
print(f"  4-mer        {STRICT_REF['4mer']['N2']:.4f}")
print(f"  CNN          {STRICT_REF['cnn']['N2']:.4f}")
print(f"  Transformer  {res['N2'][0]:.4f}  (范围 {res['N2'][1]:.4f}~{res['N2'][2]:.4f})")
print()
if res['N2'][0] < STRICT_REF['4mer']['N2']:
    print(f"→ 严格划分下 Transformer 依然低于组成基线 "
          f"（{res['N2'][0] - STRICT_REF['4mer']['N2']:+.4f}）")
else:
    print("→ 严格划分下 Transformer 超过了组成基线")

print()
print("=" * 82)
print("★ 一个有意思的地方")
print("=" * 82)
g_4mer = RANDOM['4mer']['N3a'] - STRICT_REF['4mer']['N3a']
g_cnn = RANDOM['cnn']['N3a'] - STRICT_REF['cnn']['N3a']
g_tf = RANDOM['tf']['N3a'] - res['N3a'][0]
print("以 N3a（单核苷酸打乱）为例，随机划分高估了：")
print(f"  4-mer        {g_4mer:+.4f}")
print(f"  CNN          {g_cnn:+.4f}")
print(f"  Transformer  {g_tf:+.4f}")