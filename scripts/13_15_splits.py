"""
模块 13–15：三种划分方式的对比 —— 随机划分有没有高估性能？

三种划分：
  1. random      随机划分（前面模块一直用的，作为基准）
  2. proximity   去邻近：把序列按基因组位置排序，切成若干区块，整块划分
                 → 保证相邻的峰不会一个进训练、一个进测试
  3. homology    去同源：按 Mash 距离聚类，整簇划分
                 → 保证相似序列不会分到两边

Mash 距离（比对无关的序列距离）：
  J = k-mer 集合的 Jaccard 指数
  d = -ln(2J / (1 + J)) / k        ← k 为 k-mer 长度
  d ≤ 0.05 约相当于 95% 以上的序列一致性

每条序列的"基因组位置"来源：
  正样本 / N1 / N2 / N5  —— 名字里带坐标
  N4                     —— 名字里带源峰的坐标
  N3a / N3b              —— 名字是"源峰名_repN"，取源峰坐标

输出：
  results/tables/13_15_splits.csv
  results/figures/13_15_splits.png
"""

import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
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

TIERS = ["N1", "N2", "N3a", "N3b", "N4", "N5"]
TIER_LABEL = {
    "N1": "N1 random", "N2": "N2 GC-matched", "N3a": "N3a mono shuffle",
    "N3b": "N3b dinuc shuffle", "N4": "N4 flanking", "N5": "N5 extreme AT",
}
KMER_FEAT = 4          # 用 4-mer 频率当特征（和模块 07 一致）
MASH_K = 15            # Mash 用的 k-mer 长度（短序列取 15）
MASH_D = 0.05          # 聚类的 Mash 距离阈值
SPLIT = 0.8
SEED = 42
BLOCKS = 10            # 去邻近划分切成的区块数


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
    """从名字里取出基因组起点（1-based）。

    名字的格式有四种：
        FR845719_3811_4024              正样本
        FR845719_12345                  N1 / N2 / N5
        FR845719_3811_4024_flank+400    N4
        FR845719_3811_4024_rep1         N3a / N3b
    共同点是：下划线分割后，第 2 段就是位置。
    （注意不能直接抓第一个数字串——染色体名 FR845719 里也有数字。）
    """
    parts = name.split("_")
    if len(parts) < 2:
        return -1
    m = re.match(r"\d+", parts[1])
    return int(m.group()) if m else -1


_KIDX = {}


def kmer_sets(seqs, k):
    sets = []
    for s in seqs:
        u = s.upper()
        sets.append({u[i:i + k] for i in range(len(u) - k + 1)})
    return sets


def kmer_matrix(seqs, k):
    """稀疏 0/1 矩阵：行=序列，列=k-mer（只在用到时建索引）"""
    all_sets = kmer_sets(seqs, k)
    vocab = {}
    rows, cols = [], []
    for i, st in enumerate(all_sets):
        for km in st:
            j = vocab.get(km)
            if j is None:
                j = len(vocab)
                vocab[km] = j
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)),
                             shape=(len(seqs), len(vocab)))


def mash_close_matrix(seqs):
    """返回布尔方阵：两两序列的 Mash 距离是否 <= MASH_D"""
    M = kmer_matrix(seqs, MASH_K)
    shared = (M @ M.T).toarray()               # 共享 k-mer 数
    n = np.asarray(M.sum(axis=1)).ravel()      # 各自 k-mer 数
    union = n[:, None] + n[None, :] - shared
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, shared / union, 0.0)
        d = -np.log(2 * J / (1 + J)) / MASH_K
    d[~np.isfinite(d)] = 1.0
    return d <= MASH_D


def cluster_labels(close):
    """贪心聚类：与任一代表距离够近就并入，否则新开一簇"""
    labels = -np.ones(len(close), dtype=int)
    nxt = 0
    for i in range(len(close)):
        if labels[i] >= 0:
            continue
        same = np.flatnonzero(close[i] & (labels < 0))
        labels[same] = nxt
        nxt += 1
    return labels


def group_split(groups, frac, seed):
    """按组划分：整组进训练或测试，尽量贴近 frac"""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse, minlength=len(uniq))
    order = rng.permutation(len(uniq))
    target = int(len(groups) * frac)

    take, tot = [], 0
    for k in order:
        if tot >= target:
            break
        take.append(k)
        tot += sizes[k]
    mask = np.isin(inverse, take)
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def feat4(seqs):
    """4-mer 频率特征（256 维）"""
    h = kmer_matrix(seqs, KMER_FEAT).toarray()
    return h / max(len(seqs[0]) - KMER_FEAT + 1, 1)


# ================= 读数据 =================
pos_names_seqs = read_fasta(PROC / "positives.fa")
pos = [s for _, s in pos_names_seqs]
pos_pos = np.array([position_of(n) for n, _ in pos_names_seqs])

tier_data = {}
for t in TIERS:
    recs = read_fasta(PROC / f"negatives_{t}.fa")
    seqs = [s for _, s in recs]
    poss = np.array([position_of(n) for n, _ in recs])
    tier_data[t] = (seqs, poss)

print("=" * 78)
print("数据")
print("=" * 78)
print(f"正样本 {len(pos)} 条，位置范围 {pos_pos.min()} ~ {pos_pos.max()}")
for t in TIERS:
    s, p = tier_data[t]
    print(f"  {TIER_LABEL[t]:<18} {len(s):>5} 条，位置范围 {p.min()} ~ {p.max()}")

# ================= 顺序统计：多少峰挨得很近 =================
order = np.argsort(pos_pos)
gaps = np.diff(pos_pos[order])
print()
print(f"相邻峰间距：中位 {np.median(gaps):.0f} bp"
      f"；< 1 kb 的有 {(gaps < 1000).sum()} 对"
      f"（{(gaps < 1000).mean() * 100:.1f}%）")

# ================= 三种划分 × 六档负样本 =================
rows = []
print()
print("=" * 92)
print("三种划分下的 AUC（4-mer 特征，256 维）")
print("=" * 92)
print(f"{'档次':<18}{'random':>10}{'proximity':>12}{'homology':>11}"
      f"{'n_clusters':>12}{'Δ(prox)':>10}{'Δ(hom)':>10}")
print("-" * 92)

summary = {}
for t in TIERS:
    neg, neg_pos = tier_data[t]
    seqs = pos + neg
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    X = feat4(seqs)

    # 组标签
    grp_rand = np.arange(len(seqs))                     # 每条自成一"组"→ 退化为随机
    grp_prox = np.r_[pos_pos, neg_pos]
    # 把位置量化成区块
    lo, hi = grp_prox.min(), grp_prox.max()
    grp_prox = np.clip(((grp_prox - lo) / max(hi - lo, 1) * BLOCKS).astype(int),
                       0, BLOCKS - 1)

    close = mash_close_matrix(seqs)
    # 顺便报告一下两两距离的分布，这样"没找到同源对"才是有信息量的结论
    if t == "N2":
        M = kmer_matrix(seqs, MASH_K)
        s_ = (M @ M.T).toarray()
        n_ = np.asarray(M.sum(axis=1)).ravel()
        u_ = n_[:, None] + n_[None, :] - s_
        with np.errstate(divide="ignore", invalid="ignore"):
            J_ = np.where(u_ > 0, s_ / u_, 0.0)
            D_ = -np.log(2 * J_ / (1 + J_)) / MASH_K
        D_[~np.isfinite(D_)] = 1.0
        iu = np.triu_indices(len(seqs), 1)
        dv = D_[iu]
        print()
        print(f"两两 Mash 距离分布（N2，共 {len(dv):,} 对）：")
        print(f"  最小 {dv.min():.4f}  1% {np.percentile(dv, 1):.4f}  "
              f"中位 {np.median(dv):.4f}")
        for thr in (0.02, 0.05, 0.10, 0.20):
            print(f"  d <= {thr:.2f} 的对数：{(dv <= thr).sum()}")

    grp_hom = cluster_labels(close)
    n_clu = len(np.unique(grp_hom))

    def run(groups):
        tr, te = group_split(groups, SPLIT, SEED)
        clf = LogisticRegression(max_iter=3000)
        clf.fit(X[tr], y[tr])
        return roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])

    a_rand = run(grp_rand)
    a_prox = run(grp_prox)
    a_hom = run(grp_hom)

    summary[t] = (a_rand, a_prox, a_hom, n_clu)
    print(f"{TIER_LABEL[t]:<18}{a_rand:>10.4f}{a_prox:>12.4f}{a_hom:>11.4f}"
          f"{n_clu:>12}{a_prox - a_rand:>+10.4f}{a_hom - a_rand:>+10.4f}")
    rows += [dict(tier=t, split="random", auc=round(a_rand, 4)),
             dict(tier=t, split="proximity", auc=round(a_prox, 4)),
             dict(tier=t, split="homology", auc=round(a_hom, 4))]

tab = TABDIR / "13_15_splits.csv"
with tab.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=["tier", "split", "auc"])
    w.writeheader()
    w.writerows(rows)
print()
print("表已保存：", tab)

# ================= 画图 =================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(TIERS))
w = 0.27
axes[0].bar(x - w, [summary[t][0] for t in TIERS], w, label="random", color="#c53030")
axes[0].bar(x, [summary[t][1] for t in TIERS], w, label="proximity-aware", color="#2b6cb0")
axes[0].bar(x + w, [summary[t][2] for t in TIERS], w, label="homology-aware", color="#38a169")
axes[0].axhline(0.5, ls="--", lw=1.2, color="grey")
axes[0].set_xticks(x)
axes[0].set_xticklabels([TIER_LABEL[t] for t in TIERS], rotation=25, ha="right", fontsize=9)
axes[0].set_ylabel("AUC (4-mer features)")
axes[0].set_ylim(0, 1.05)
axes[0].set_title("A. AUC under three splitting schemes")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.25, axis="y")

axes[1].bar(x - 0.17, [summary[t][1] - summary[t][0] for t in TIERS], 0.34,
            label="proximity − random", color="#2b6cb0")
axes[1].bar(x + 0.17, [summary[t][2] - summary[t][0] for t in TIERS], 0.34,
            label="homology − random", color="#38a169")
axes[1].axhline(0, lw=1.2, color="black")
axes[1].set_xticks(x)
axes[1].set_xticklabels([TIER_LABEL[t] for t in TIERS], rotation=25, ha="right", fontsize=9)
axes[1].set_ylabel("AUC difference vs random split")
axes[1].set_title("B. Does random splitting inflate performance?")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.25, axis="y")

plt.tight_layout()
figpath = FIGDIR / "13_15_splits.png"
plt.savefig(figpath, dpi=160)
print("图已保存：", figpath)

# ================= 结论 =================
print()
print("=" * 78)
print("结论")
print("=" * 78)
d_prox = np.mean([summary[t][1] - summary[t][0] for t in TIERS])
d_hom = np.mean([summary[t][2] - summary[t][0] for t in TIERS])
print(f"去邻近划分相对随机划分的平均变化：{d_prox:+.4f}")
print(f"去同源划分相对随机划分的平均变化：{d_hom:+.4f}")
if abs(d_prox) < 0.02 and abs(d_hom) < 0.02:
    print("→ 随机划分没有明显高估性能：本数据集的泄漏风险低")
else:
    print("→ 严格划分下性能下降，说明随机划分存在高估")
