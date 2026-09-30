"""
模块 11：自己实现一个小 Transformer —— 注意力机制能否带来额外信息？

设计参照 DNABERT 的思路：把 DNA 切成 k-mer 当作"词"，
然后自己实现 Transformer 编码器（词嵌入 + 位置嵌入 + 多头自注意力 + 前馈网络）。

不依赖任何预训练模型，纯 PyTorch，CPU 可跑。

对比链条：GC(1维) → 4-mer(256维) → CNN → Transformer
"""
import csv, os, sys, time
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
# 前面模块的结果（固定种子，直接引用）
GC_AUC   = {"N1": 0.8601, "N2": 0.5516, "N3a": 0.4684, "N3b": 0.4669, "N4": 0.8431, "N5": 0.7868}
KMER_AUC = {"N1": 0.8606, "N2": 0.7343, "N3a": 0.9117, "N3b": 0.8327, "N4": 0.8638, "N5": 0.8741}

BASES = "ACGT"
KMER = 3
LEN = 201
N_TOK = LEN // KMER          # 67 个 token
VOCAB = 4 ** KMER            # 64
SPLIT, SEED = 0.8, 42
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


def tokenize(seqs):
    """把每条序列切成 67 个 3-mer，每个映射到 0..63 的整数"""
    lut = {b: i for i, b in enumerate(BASES)}
    out = np.zeros((len(seqs), N_TOK), dtype=np.int64)
    for i, s in enumerate(seqs):
        u = s.upper()
        for j in range(N_TOK):
            code = 0
            ok = True
            for ch in u[j * KMER:(j + 1) * KMER]:
                k = lut.get(ch)
                if k is None:
                    ok = False
                    break
                code = code * 4 + k
            out[i, j] = code if ok else 0
    return out


class Block(nn.Module):
    """一个 Transformer 编码器层：多头自注意力 + 前馈网络，都带残差和 LayerNorm"""

    def __init__(self, d, nhead, ff, p):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=p, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.ln2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        a, _ = self.attn(x, x, x)            # 自注意力：Q=K=V=x
        x = self.ln1(x + self.drop(a))       # 残差连接 + 归一化
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


class DNATransformer(nn.Module):
    def __init__(self, d=64, nhead=4, nl=2, ff=128, p=0.1):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)                 # 词嵌入
        self.cls = nn.Parameter(torch.zeros(1, 1, d))     # 特殊的 [CLS] 标记
        self.pos = nn.Parameter(torch.zeros(1, N_TOK + 1, d))  # 位置嵌入
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
        return self.head(self.ln(x[:, 0])).squeeze(1)     # 用 [CLS] 的输出做分类


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
pos = [s for _, s in read_fasta(PROC / "positives.fa")]
neg = {t: [s for _, s in read_fasta(PROC / f"negatives_{t}.fa")] for t in TIERS}

rng = np.random.default_rng(SEED)
def split(n):
    idx = rng.permutation(n); k = int(n * SPLIT)
    return idx[:k], idx[k:]
pos_tr, pos_te = split(len(pos))
neg_idx = {t: split(len(neg[t])) for t in TIERS}

print("=" * 78)
print("模块 11：自己实现的 Transformer")
print("=" * 78)
print(f"分词：{KMER}-mer，每条 {N_TOK} 个 token，词表 {VOCAB}")
print(f"模型：2 层编码器，d_model=64，4 头注意力，5 个随机种子")

Xpos = tokenize(pos)
Xneg = {t: tokenize(neg[t]) for t in TIERS}

rows, res = [], {}
for t in TIERS:
    tr, te = neg_idx[t]
    Xtr_all = np.concatenate([Xpos[pos_tr], Xneg[t][tr]])
    ytr_all = np.r_[np.ones(len(pos_tr)), np.zeros(len(tr))]
    Xte = np.concatenate([Xpos[pos_te], Xneg[t][te]])
    yte = np.r_[np.ones(len(pos_te)), np.zeros(len(te))]

    r = np.random.default_rng(SEED)
    idx = r.permutation(len(ytr_all)); k = int(len(ytr_all) * 0.15)
    va, tr2 = idx[:k], idx[k:]

    aucs, t0 = [], time.time()
    for sd in SEEDS:
        m = train_tf(Xtr_all[tr2], ytr_all[tr2], Xtr_all[va], ytr_all[va], sd)
        with torch.no_grad():
            p = torch.sigmoid(m(torch.from_numpy(Xte))).numpy()
        aucs.append(roc_auc_score(yte, p))
    res[t] = (float(np.mean(aucs)), float(np.min(aucs)), float(np.max(aucs)))
    print(f"{TIER_LABEL[t]:<18} Transformer {res[t][0]:.4f} "
          f"({res[t][1]:.4f}~{res[t][2]:.4f})  [{time.time()-t0:.0f}s]")
    rows.append(dict(tier=t, auc_mean=round(res[t][0], 4),
                     auc_min=round(res[t][1], 4), auc_max=round(res[t][2], 4)))

tab = TABDIR / "11_transformer.csv"
with tab.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print()
print("表已保存：", tab)

# ================= 四者对比 =================
print()
print("=" * 78)
print("四种方法的对比")
print("=" * 78)
print(f"{'档次':<18}{'GC':>9}{'4-mer':>9}{'Transformer':>14}")
print("-" * 78)
for t in TIERS:
    print(f"{TIER_LABEL[t]:<18}{GC_AUC[t]:>9.4f}{KMER_AUC[t]:>9.4f}{res[t][0]:>14.4f}")

# ================= 画图 =================
fig, ax = plt.subplots(figsize=(10.5, 5))
x = np.arange(len(TIERS)); w = 0.3
ax.bar(x - w, [GC_AUC[t] for t in TIERS], w, label="GC (1 dim)", color="#c53030")
ax.bar(x, [KMER_AUC[t] for t in TIERS], w, label="4-mer (256 dim)", color="#2b6cb0")
ax.bar(x + w, [res[t][0] for t in TIERS], w, label="Transformer (self-implemented)", color="#805ad5")
ax.errorbar(x + w, [res[t][0] for t in TIERS],
            yerr=[[res[t][0] - res[t][1] for t in TIERS],
                  [res[t][2] - res[t][0] for t in TIERS]],
            fmt="none", ecolor="black", capsize=3)
ax.axhline(0.5, ls="--", lw=1.2, color="grey")
ax.set_xticks(x)
ax.set_xticklabels([TIER_LABEL[t] for t in TIERS], rotation=25, ha="right", fontsize=9)
ax.set_ylabel("AUC")
ax.set_ylim(0, 1.05)
ax.set_title("Composition baselines vs a self-implemented Transformer")
ax.legend(fontsize=9)
ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(FIGDIR / "11_transformer.png", dpi=160)
print()
print("图已保存：", FIGDIR / "11_transformer.png")

# ================= 结论 =================
print()
print("=" * 78)
print("结论")
print("=" * 78)
d = res["N2"][0] - KMER_AUC["N2"]
print(f"N2（最难假货）上的对比：")
print(f"  GC（1 维）    {GC_AUC['N2']:.4f}")
print(f"  4-mer（256 维）{KMER_AUC['N2']:.4f}")
print(f"  Transformer    {res['N2'][0]:.4f}  范围 {res['N2'][1]:.4f}~{res['N2'][2]:.4f}")
print()
if res["N2"][1] <= KMER_AUC["N2"] <= res["N2"][2]:
    print("→ Transformer 的种子区间覆盖了组成基线：没有证据表明注意力带来额外信息")
elif d > 0.02:
    print(f"→ Transformer 比组成基线高 {d:+.4f}：注意力机制带来了额外信息")
else:
    print(f"→ Transformer 未超过组成基线（{d:+.4f}）")