"""
模块 17b：内部组成不均匀性检验

模块 16-17 发现：模型的权重集中在 GC 富集的 4-mer 上（r = +0.47），
而不是 AT 富集的——这和"模型只是数 AT"的直觉相反。

本模块检验一个解释：模型检测的是【组成的不均匀性】——
即序列内部有起伏（AT 富集的局部核心 + GC 富集的侧翼），
而不是整体的 AT 含量。

两个分析：
  1. 比较各数据集的"内部 GC 波动"
  2. 看模型分数与内部 GC 波动的相关性（决定性检验）
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
LEN = 201
SUB = 20                     # 子窗口长度
N_SUB = LEN // SUB           # 10 个
KMER = 4
SPLIT, SEED, BLOCKS = 0.8, 42, 10
TIERS = ["N1", "N2", "N5"]


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


def internal_stats(seqs):
    """返回每条序列的内部 GC 统计：均值、标准差、极差、最小子窗口 GC"""
    n = len(seqs)
    sub = np.zeros((n, N_SUB))
    for i, s in enumerate(seqs):
        u = s.upper()
        for j in range(N_SUB):
            w = u[j * SUB:(j + 1) * SUB]
            sub[i, j] = (w.count("G") + w.count("C")) / len(w) * 100
    return dict(
        mean=sub.mean(axis=1),
        sd=sub.std(axis=1),
        rng=sub.max(axis=1) - sub.min(axis=1),
        min_sub=sub.min(axis=1),
        max_sub=sub.max(axis=1),
    )


# ---- 4-mer 特征 ----
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


# ================= 读数据 =================
pos_rec = read_fasta(PROC / "positives.fa")
pos = [s for _, s in pos_rec]
pos_p = np.array([position_of(n) for n, _ in pos_rec])
pos_st = internal_stats(pos)

data = {}
for t in TIERS:
    rec = read_fasta(PROC / f"negatives_{t}.fa")
    seqs = [s for _, s in rec]
    data[t] = dict(seqs=seqs,
                   pos=np.array([position_of(n) for n, _ in rec]),
                   st=internal_stats(seqs))

# ================= 分析一：各数据集的内部波动 =================
print("=" * 80)
print("分析一：内部 GC 波动（把 201 bp 切成 10 个 20 bp 的子窗口）")
print("=" * 80)
print(f"{'数据集':<14}{'整体GC':>9}{'内部SD':>9}{'内部极差':>10}"
      f"{'最AT子窗':>10}{'最GC子窗':>10}")
print("-" * 80)


def line(label, st):
    print(f"{label:<14}{st['mean'].mean():>9.2f}{st['sd'].mean():>9.2f}"
          f"{st['rng'].mean():>10.2f}{st['min_sub'].mean():>10.2f}"
          f"{st['max_sub'].mean():>10.2f}")


line("正样本", pos_st)
for t in TIERS:
    line(f"{t} 负样本", data[t]["st"])

d_sd = pos_st["sd"].mean() - data["N2"]["st"]["sd"].mean()
print()
print(f"正样本与 N2 的内部 SD 之差：{d_sd:+.3f}")

# ================= 分析二：模型分数 vs 内部波动 =================
print()
print("=" * 80)
print("分析二（决定性检验）：模型分数与内部 GC 波动的相关性")
print("=" * 80)

t = "N2"
neg = data[t]["seqs"]
y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
X = np.vstack([feat4(pos), feat4(neg)])
tr, te = proximity_split(np.r_[pos_p, data[t]["pos"]], SPLIT, SEED)

clf = LogisticRegression(max_iter=3000)
clf.fit(X[tr], y[tr])
score = clf.predict_proba(X[te])[:, 1]
auc = roc_auc_score(y[te], score)
print(f"模型（N2，严格划分）测试集 AUC：{auc:.4f}")

# 测试集的内部波动
sd_all = np.r_[pos_st["sd"], data[t]["st"]["sd"]][te]
gc_all = np.r_[pos_st["mean"], data[t]["st"]["mean"]][te]

r_sd = np.corrcoef(sd_all, score)[0, 1]
r_gc = np.corrcoef(gc_all, score)[0, 1]
print()
print(f"模型分数 vs 内部 GC 标准差：r = {r_sd:+.4f}")
print(f"模型分数 vs 整体 GC 含量  ：r = {r_gc:+.4f}")
print()
if abs(r_sd) > abs(r_gc) * 1.5 and abs(r_sd) > 0.3:
    print("→ 模型分数与【内部波动】的相关性远强于与整体 GC 的相关性")
    print("  支持'模型检测的是组成的不均匀性'这一解释")
elif abs(r_gc) > abs(r_sd) * 1.5:
    print("→ 模型分数主要跟随整体 GC 含量")
else:
    print("→ 两者相关性相近，或都不强：模型可能依赖更复杂的组合模式")

# ================= 画图 =================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

labels = ["positives"] + [f"{t} decoys" for t in TIERS]
vals = [pos_st["sd"]] + [data[t]["st"]["sd"] for t in TIERS]
axes[0].boxplot(vals, tick_labels=labels, showfliers=False)
axes[0].set_ylabel("Internal GC std (across 10 x 20 bp sub-windows)")
axes[0].set_title("A. Internal composition heterogeneity")
axes[0].grid(alpha=0.25, axis="y")

axes[1].scatter(sd_all, score, s=12, alpha=0.35, c=["#c53030" if v else "#2b6cb0"
                                                    for v in y[te]])
axes[1].set_xlabel("Internal GC std")
axes[1].set_ylabel("Model predicted probability")
axes[1].set_title(f"B. Model score vs internal heterogeneity\nPearson r = {r_sd:+.3f}")
axes[1].grid(alpha=0.25)

plt.tight_layout()
plt.savefig(FIGDIR / "17b_heterogeneity.png", dpi=160)
print()
print("图已保存：", FIGDIR / "17b_heterogeneity.png")

# ================= 导出 =================
tab = TABDIR / "17b_heterogeneity.csv"
with tab.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset", "mean_gc", "internal_sd", "internal_range",
                "min_subwindow_gc", "max_subwindow_gc"])
    for label, st in [("positives", pos_st)] + [(t, data[t]["st"]) for t in TIERS]:
        w.writerow([label, round(st["mean"].mean(), 3), round(st["sd"].mean(), 3),
                    round(st["rng"].mean(), 3), round(st["min_sub"].mean(), 3),
                    round(st["max_sub"].mean(), 3)])
print("表已保存：", tab)