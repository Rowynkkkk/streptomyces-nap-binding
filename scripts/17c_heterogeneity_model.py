"""
模块 17c：把"内部波动"当成特征，单独建模

问题：正样本的内部组成波动确实更大（SD 9.75 vs N2 的 8.79），
      但 4-mer 模型的分数并不跟随这个波动（r = 0.046）。

所以：这种波动本身是不是一个有效的判别特征？

做法：只用 4 个"内部波动"特征训练逻辑回归，看 AUC。
     并在 5 种不同的严格划分下重复，得到稳定的区间。
"""
import csv, os, re, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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

BASES = "ACGT"
LEN, SUB, KMER = 201, 20, 4
N_SUB = LEN // SUB
SPLIT, BLOCKS = 0.8, 10
SPLIT_SEEDS = [42, 1, 2, 3, 4]
TIERS = ["N1", "N2"]


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


def sub_gc(seqs):
    """返回 (n, N_SUB) 的子窗口 GC 矩阵"""
    n = len(seqs)
    out = np.zeros((n, N_SUB))
    for i, s in enumerate(seqs):
        u = s.upper()
        for j in range(N_SUB):
            w = u[j * SUB:(j + 1) * SUB]
            out[i, j] = (w.count("G") + w.count("C")) / len(w)
    return out


KEYS = []
for j in range(4 ** KMER):
    t, x = [], j
    for _ in range(KMER):
        t.append(BASES[x % 4]); x //= 4
    KEYS.append("".join(t))
VOCAB = {s: i for i, s in enumerate(KEYS)}


def feat4(seqs):
    out = np.zeros((len(seqs), 4 ** KMER), dtype=np.float32)
    for i, s in enumerate(seqs):
        u = s.upper()
        for j in range(len(u) - KMER + 1):
            p = VOCAB.get(u[j:j + KMER])
            if p is not None:
                out[i, p] += 1
    return out / (len(seqs[0]) - KMER + 1)


def build_features(seqs):
    """返回四种特征集"""
    sub = sub_gc(seqs)
    gc = sub.mean(axis=1, keepdims=True)                  # 整体 GC
    sd = sub.std(axis=1, keepdims=True)                   # 内部标准差
    het = np.hstack([sub.std(axis=1, keepdims=True),
                     (sub.max(axis=1) - sub.min(axis=1))[:, None],
                     sub.min(axis=1, keepdims=True),
                     sub.max(axis=1, keepdims=True)])     # 四个内部波动特征
    return dict(gc=gc, sd=sd, het=het, kmer4=feat4(seqs)), sub


# ================= 读数据 =================
pos_rec = read_fasta(PROC / "positives.fa")
pos = [s for _, s in pos_rec]
pos_p = np.array([position_of(n) for n, _ in pos_rec])
Fpos, sub_pos = build_features(pos)

data = {}
for t in TIERS:
    rec = read_fasta(PROC / f"negatives_{t}.fa")
    seqs = [s for _, s in rec]
    F, sub = build_features(seqs)
    data[t] = dict(seqs=seqs, pos=np.array([position_of(n) for n, _ in rec]),
                   F=F, sub=sub)

# ================= 特征之间的关系 =================
print("=" * 80)
print("先确认：内部波动特征和整体 GC 是不是同一个东西")
print("=" * 80)
all_sub = np.vstack([sub_pos, data["N2"]["sub"]])
gc_all = all_sub.mean(axis=1)
sd_all = all_sub.std(axis=1)
print(f"整体 GC vs 内部 SD 的相关性：r = {np.corrcoef(gc_all, sd_all)[0, 1]:+.4f}")
print(f"（相关性低说明它们是两个不同的量）")

# ================= 各特征集的 AUC =================
print()
print("=" * 80)
print("各特征集在 N2 上的 AUC（5 种严格划分）")
print("=" * 80)

FEATURE_SETS = [("整体 GC（1 维）", "gc"),
                ("内部 SD（1 维）", "sd"),
                ("四个内部波动特征（4 维）", "het"),
                ("4-mer 组成（256 维）", "kmer4")]

results = {}
rows = []
for t in TIERS:
    neg = data[t]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg["seqs"]))]
    positions = np.r_[pos_p, neg["pos"]]
    print()
    print(f"--- {t} ---")
    print(f"{'特征集':<26}{'ACU 均值':>10}{'范围':>20}")
    print("-" * 80)
    for label, key in FEATURE_SETS:
        X = np.vstack([Fpos[key], neg["F"][key]])
        aucs = []
        for sd_seed in SPLIT_SEEDS:
            tr, te = proximity_split(positions, SPLIT, sd_seed)
            clf = LogisticRegression(max_iter=3000)
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        m, lo, hi = float(np.mean(aucs)), float(np.min(aucs)), float(np.max(aucs))
        results[(t, key)] = aucs
        print(f"{label:<26}{m:>10.4f}{f'{lo:.4f} ~ {hi:.4f}':>20}")
        rows.append(dict(tier=t, features=key, auc_mean=round(m, 4),
                         auc_min=round(lo, 4), auc_max=round(hi, 4)))

tab = TABDIR / "17c_heterogeneity_model.csv"
with tab.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print()
print("表已保存：", tab)

# ================= 画图 =================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
keys = [k for _, k in FEATURE_SETS]
labels = [l for l, _ in FEATURE_SETS]

for ax, t in ((axes[0], "N2"), (axes[1], "N1")):
    means = [np.mean(results[(t, k)]) for k in keys]
    los = [np.min(results[(t, k)]) for k in keys]
    his = [np.max(results[(t, k)]) for k in keys]
    ax.bar(range(len(keys)), means, yerr=[np.array(means) - np.array(los),
                                          np.array(his) - np.array(means)],
           capsize=4, color=["#c53030", "#dd6b20", "#805ad5", "#2b6cb0"])
    ax.axhline(0.5, ls="--", lw=1.2, color="grey")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("AUC (proximity-aware split)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{t}: which feature set carries the signal?")
    ax.grid(alpha=0.25, axis="y")
    for i, m in enumerate(means):
        ax.text(i, m + 0.015, f"{m:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIGDIR / "17c_heterogeneity_model.png", dpi=160)
print("图已保存：", FIGDIR / "17c_heterogeneity_model.png")

# ================= 结论 =================
print()
print("=" * 80)
print("结论")
print("=" * 80)
m_sd = np.mean(results[("N2", "sd")])
m_het = np.mean(results[("N2", "het")])
m_gc = np.mean(results[("N2", "gc")])
m_k4 = np.mean(results[("N2", "kmer4")])
print(f"N2 上：整体 GC {m_gc:.4f} ｜ 内部 SD {m_sd:.4f} ｜ 四特征 {m_het:.4f} ｜ 4-mer {m_k4:.4f}")
print()
if m_het > 0.55:
    print("→ 内部波动特征本身有判别力：这说明'波动'是有效特征，只是 4-mer 模型没抓到它")
elif m_het > 0.52:
    print("→ 内部波动特征有微弱的判别力，但远低于 4-mer 组成")
else:
    print("→ 内部波动特征几乎无判别力：正负样本在这个指标上的差异不足以区分类别")