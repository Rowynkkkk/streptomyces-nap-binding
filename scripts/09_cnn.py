"""
模块 09：一维卷积网络（CNN）—— 序列顺序有没有额外贡献？

和前面的组成基线对照：
  GC（1 维）   —— 只看 GC 含量
  4-mer（256 维）—— 完整的四核苷酸组成，但**不包含顺序信息**
  CNN           —— 能看到位置和顺序，理论上可以学 motif

核心问题（只在 N2 上有意义，因为其他档太容易）：

  ★ 面对 GC 匹配的真实假货，CNN 能不能超过 4-mer 的 0.7343？
     - 超不过 → 组成（含高阶组成）就是全部，顺序没有额外贡献
     - 超过   → 顺序确实携带信息，这是真正的发现

设计沿用前面：A 匹配训练 / B 固定模型（只在 N1 上训练）

输出：
  results/tables/09_cnn.csv
  results/figures/09_cnn_vs_baseline.png
"""

import csv
import os
import sys
import time
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
# 前面模块算出来的组成基线（固定种子，可直接引用）
GC_AUC = {"N1": 0.8601, "N2": 0.5516, "N3a": 0.4684,
          "N3b": 0.4669, "N4": 0.8431, "N5": 0.7868}
KMER4_AUC = {"N1": 0.8606, "N2": 0.7343, "N3a": 0.9117,
             "N3b": 0.8327, "N4": 0.8638, "N5": 0.8741}

SPLIT = 0.8
SEED = 42
BASES = "ACGT"
LEN = 201
EPOCHS = 25
PATIENCE = 4
BATCH = 128
LR = 1e-3
SEEDS = [0, 1, 2, 3, 4]          # 修改为跑五个种子

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


def one_hot(seqs):
    """DNA 序列 -> (n, 4, LEN) 的 one-hot 数组"""
    idx = {b: i for i, b in enumerate(BASES)}
    out = np.zeros((len(seqs), 4, LEN), dtype=np.float32)
    for i, s in enumerate(seqs):
        u = s.upper()
        for j, ch in enumerate(u[:LEN]):
            k = idx.get(ch)
            if k is not None:
                out[i, k, j] = 1.0
    return out


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
    """训练并返回在验证集上最优的模型状态"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32)))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()

    Xva_t = torch.from_numpy(Xva)
    best_auc, best_state, bad = -1.0, None, 0

    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()

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

    model.load_state_dict(best_state)
    model.eval()
    return model, best_auc, ep + 1


# ================= 读数据 =================
pos = [s for _, s in read_fasta(PROC / "positives.fa")]
neg = {t: [s for _, s in read_fasta(PROC / f"negatives_{t}.fa")] for t in TIERS}

rng = np.random.default_rng(SEED)


def split(n):
    idx = rng.permutation(n)
    k = int(n * SPLIT)
    return idx[:k], idx[k:]


pos_tr, pos_te = split(len(pos))
neg_idx = {t: split(len(neg[t])) for t in TIERS}

print("=" * 78)
print("编码序列（one-hot）...")
Xp_all = one_hot(pos)
Xn_all = {t: one_hot(neg[t]) for t in TIERS}
print("  完成")

# 训练时再切出一小块做早停验证（用训练集内部）
def make_xy(tier):
    tr, te = neg_idx[tier]
    Xtr_all = np.concatenate([Xp_all[pos_tr], Xn_all[tier][tr]])
    ytr_all = np.r_[np.ones(len(pos_tr)), np.zeros(len(tr))]
    Xte = np.concatenate([Xp_all[pos_te], Xn_all[tier][te]])
    yte = np.r_[np.ones(len(pos_te)), np.zeros(len(te))]
    return Xtr_all, ytr_all, Xte, yte


def carve_val(X, y, frac=0.15, seed=SEED):
    r = np.random.default_rng(seed)
    idx = r.permutation(len(y))
    k = int(len(y) * frac)
    va, tr = idx[:k], idx[k:]
    return X[tr], y[tr], X[va], y[va]


# ================= 设计 A：匹配训练 =================
print()
print("=" * 78)
print("设计 A · 匹配训练（每档单独训练 CNN）")
print("=" * 78)
print(f"{'档次':<18}{'种子':>6}{'epoch':>7}{'val_AUC':>10}{'test_AUC':>10}{'耗时(s)':>9}")
print("-" * 78)

rows = []
resA = {}
for t in TIERS:
    Xtr_all, ytr_all, Xte, yte = make_xy(t)
    Xtr, ytr, Xva, yva = carve_val(Xtr_all, ytr_all)
    aucs = []
    for sd in SEEDS:
        t0 = time.time()
        model, va, ep = train_cnn(Xtr, ytr, Xva, yva, sd)
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(Xte))).numpy()
        auc = roc_auc_score(yte, p)
        aucs.append(auc)
        dt = time.time() - t0
        print(f"{TIER_LABEL[t]:<18}{sd:>6}{ep:>7}{va:>10.4f}{auc:>10.4f}{dt:>9.1f}")
        rows.append(dict(design="matched", tier=t, seed=sd, auc=round(auc, 4)))
    resA[t] = float(np.mean(aucs))

print()
print(f"{'档次':<18}{'GC':>9}{'4-mer':>9}{'CNN':>9}")
print("-" * 78)
for t in TIERS:
    print(f"{TIER_LABEL[t]:<18}{GC_AUC[t]:>9.4f}{KMER4_AUC[t]:>9.4f}{resA[t]:>9.4f}")

# ================= 设计 B：固定模型 =================
print()
print("=" * 78)
print("设计 B · 固定模型（只在 N1 上训练）")
print("=" * 78)

Xtr_all, ytr_all, _, _ = make_xy("N1")
Xtr, ytr, Xva, yva = carve_val(Xtr_all, ytr_all)
models = []
for sd in SEEDS:
    m, va, ep = train_cnn(Xtr, ytr, Xva, yva, sd)
    models.append(m)
    print(f"  种子 {sd} 训练完成（epoch {ep}, val AUC {va:.4f}）")

print()
print(f"{'档次':<18}{'CNN(fixed)':>12}")
print("-" * 78)
resB = {}
for t in TIERS:
    _, _, Xte, yte = make_xy(t)
    aucs = []
    for m in models:
        with torch.no_grad():
            p = torch.sigmoid(m(torch.from_numpy(Xte))).numpy()
        aucs.append(roc_auc_score(yte, p))
    resB[t] = float(np.mean(aucs))
    print(f"{TIER_LABEL[t]:<18}{resB[t]:>12.4f}")
    rows.append(dict(design="fixed_from_N1", tier=t, seed=-1, auc=round(resB[t], 4)))

tab = TABDIR / "09_cnn.csv"
with tab.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=["design", "tier", "seed", "auc"])
    w.writeheader()
    w.writerows(rows)
print()
print("表已保存：", tab)

# ================= 画图 =================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
x = np.arange(len(TIERS))

axes[0].bar(x - 0.27, [GC_AUC[t] for t in TIERS], 0.27, label="GC (1 dim)", color="#c53030")
axes[0].bar(x, [KMER4_AUC[t] for t in TIERS], 0.27, label="4-mer (256 dim)", color="#2b6cb0")
axes[0].bar(x + 0.27, [resA[t] for t in TIERS], 0.27, label="CNN", color="#38a169")
axes[0].axhline(0.5, ls="--", lw=1.2, color="grey")
axes[0].set_xticks(x)
axes[0].set_xticklabels([TIER_LABEL[t] for t in TIERS], rotation=25, ha="right", fontsize=9)
axes[0].set_ylabel("AUC")
axes[0].set_ylim(0, 1.05)
axes[0].set_title("A. Matched training: composition vs sequence model")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.25, axis="y")

for i, t in enumerate(TIERS):
    axes[0].text(i + 0.27, resA[t] + 0.012, f"{resA[t]:.2f}", ha="center", fontsize=8)

axes[1].plot(x, [GC_AUC[t] for t in TIERS], "o-", label="GC", color="#c53030")
axes[1].plot(x, [KMER4_AUC[t] for t in TIERS], "s-", label="4-mer", color="#2b6cb0")
axes[1].plot(x, [resB[t] for t in TIERS], "^-", label="CNN (fixed from N1)", color="#38a169")
axes[1].axhline(0.5, ls="--", lw=1.2, color="grey")
axes[1].set_xticks(x)
axes[1].set_xticklabels([TIER_LABEL[t] for t in TIERS], rotation=25, ha="right", fontsize=9)
axes[1].set_ylabel("AUC")
axes[1].set_ylim(0, 1.02)
axes[1].set_title("B. Fixed model: does it survive hard decoys?")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.25)

plt.tight_layout()
figpath = FIGDIR / "09_cnn_vs_baseline.png"
plt.savefig(figpath, dpi=160)
print("图已保存：", figpath)

# ================= 结论 =================
print()
print("=" * 78)
print("核心结论")
print("=" * 78)
print(f"N2（唯一真正难的假货）上的对比：")
print(f"  GC（1 维）    {GC_AUC['N2']:.4f}")
print(f"  4-mer（256 维）{KMER4_AUC['N2']:.4f}")
print(f"  CNN            {resA['N2']:.4f}")
d = resA["N2"] - KMER4_AUC["N2"]
print()
if d > 0.02:
    print(f"→ CNN 比 4-mer 高 {d:+.4f}：序列顺序带来了额外信息")
elif d < -0.02:
    print(f"→ CNN 比 4-mer 低 {d:+.4f}：顺序模型没有超过组成基线")
else:
    print(f"→ CNN 与 4-mer 相当（{d:+.4f}）：没有证据表明顺序带来额外信息")
