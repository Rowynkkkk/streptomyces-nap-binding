"""
模块 16-17：模型可解释性 —— 那个 AUC 0.586 的组成模型，到底在看什么？

两个分析：
  16. 特征重要性：4-mer 逻辑回归的 256 个系数，哪些最重要？它们的 GC 含量如何？
  17. 系数的 GC 分箱：把 k-mer 按自身 GC 含量分组，看各组的系数分布
      —— 如果权重集中在 AT 富集的 k-mer 上，说明模型就是在"数 AT"

用严格划分（去邻近），与前面的结论口径一致。
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
KMER = 4
LEN = 201
SPLIT, SEED, BLOCKS = 0.8, 42, 10
TIER = "N2"          # 只分析最关键的那一档


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


# 4-mer 词表
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


# ================= 读数据、训练 =================
pos_rec = read_fasta(PROC / "positives.fa")
pos = [s for _, s in pos_rec]
pos_p = np.array([position_of(n) for n, _ in pos_rec])
neg_rec = read_fasta(PROC / f"negatives_{TIER}.fa")
neg = [s for _, s in neg_rec]
neg_p = np.array([position_of(n) for n, _ in neg_rec])

y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
X = np.vstack([feat4(pos), feat4(neg)])
tr, te = proximity_split(np.r_[pos_p, neg_p], SPLIT, SEED)

clf = LogisticRegression(max_iter=3000)
clf.fit(X[tr], y[tr])
auc = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
coef = clf.coef_[0]

print("=" * 76)
print(f"模块 16-17：4-mer 模型的可解释性（{TIER}，严格划分）")
print("=" * 76)
print(f"测试集 AUC：{auc:.4f}")
print(f"系数个数：{len(coef)}")

# ================= 分析一：最重要的 k-mer =================
def gc_of_kmer(k):
    return (k.count("G") + k.count("C")) / len(k) * 100

order = np.argsort(-coef)
print()
print("=" * 76)
print("权重最高的 15 个 k-mer（指向'是结合位点'）")
print("=" * 76)
print(f"{'k-mer':<8}{'系数':>10}{'GC%':>8}")
for i in order[:15]:
    print(f"{KEYS[i]:<8}{coef[i]:>10.4f}{gc_of_kmer(KEYS[i]):>8.0f}")

print()
print("权重最低的 15 个 k-mer（指向'不是结合位点'）")
print("=" * 76)
print(f"{'k-mer':<8}{'系数':>10}{'GC%':>8}")
for i in order[-15:]:
    print(f"{KEYS[i]:<8}{coef[i]:>10.4f}{gc_of_kmer(KEYS[i]):>8.0f}")

top_gc = np.mean([gc_of_kmer(KEYS[i]) for i in order[:20]])
bot_gc = np.mean([gc_of_kmer(KEYS[i]) for i in order[-20:]])
print()
print(f"权重最高 20 个 k-mer 的平均 GC：{top_gc:.1f}%")
print(f"权重最低 20 个 k-mer 的平均 GC：{bot_gc:.1f}%")
print(f"全部 256 个 k-mer 的平均 GC ：{np.mean([gc_of_kmer(k) for k in KEYS]):.1f}%")

# ================= 分析二：按 k-mer 的 GC 含量分箱 =================
km_gc = np.array([gc_of_kmer(k) for k in KEYS])
print()
print("=" * 76)
print("按 k-mer 自身 GC 含量分组，看系数的分布")
print("=" * 76)
print(f"{'k-mer GC 区间':<16}{'个数':>6}{'系数均值':>12}{'系数总和':>12}")
print("-" * 76)
for lo, hi in [(0, 26), (26, 51), (51, 76), (76, 101)]:
    m = (km_gc >= lo) & (km_gc < hi)
    if m.sum():
        print(f"{f'{lo}-{hi-1}%':<16}{int(m.sum()):>6}{coef[m].mean():>12.5f}{coef[m].sum():>12.4f}")

# 相关系数
corr = np.corrcoef(km_gc, coef)[0, 1]
print()
print(f"k-mer 的 GC 含量 vs 其系数：Pearson r = {corr:+.4f}")

# ================= 画图 =================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

axes[0].scatter(km_gc, coef, s=22, alpha=0.6, color="#2b6cb0")
axes[0].axhline(0, lw=1, color="black")
axes[0].set_xlabel("GC content of the k-mer (%)")
axes[0].set_ylabel("Logistic regression coefficient")
axes[0].set_title(f"A. Do the model's weights favor AT-rich k-mers?\nPearson r = {corr:+.3f}")
axes[0].grid(alpha=0.25)

# 按 GC 分组画箱线
groups, labels = [], []
for lo, hi in [(0, 26), (26, 51), (51, 76), (76, 101)]:
    m = (km_gc >= lo) & (km_gc < hi)
    if m.sum():
        groups.append(coef[m]); labels.append(f"{lo}-{hi-1}%")
axes[1].boxplot(groups, tick_labels=labels, showfliers=False)
axes[1].axhline(0, lw=1, color="black")
axes[1].set_xlabel("GC content of the k-mer")
axes[1].set_ylabel("Coefficient")
axes[1].set_title("B. Coefficient distribution by k-mer GC")
axes[1].grid(alpha=0.25, axis="y")

plt.tight_layout()
plt.savefig(FIGDIR / "16_17_interpret.png", dpi=160)
print()
print("图已保存：", FIGDIR / "16_17_interpret.png")

# ================= 导出系数表 =================
tab = TABDIR / "16_17_kmer_coefficients.csv"
with tab.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["kmer", "gc_percent", "coefficient"])
    for i in order:
        w.writerow([KEYS[i], round(gc_of_kmer(KEYS[i]), 1), round(float(coef[i]), 6)])
print("系数表已保存：", tab)

# ================= 结论 =================
print()
print("=" * 76)
print("结论")
print("=" * 76)
if corr < -0.3:
    print(f"→ 系数与 k-mer 的 GC 呈明显负相关（r = {corr:+.3f}）：")
    print("  模型确实更看重 AT 富集的 k-mer，与'偏好 AT'的生物学先验一致")
elif corr > 0.3:
    print(f"→ 系数与 GC 呈正相关（r = {corr:+.3f}）：模型偏好 GC 富集的 k-mer")
else:
    print(f"→ 系数与 k-mer 的 GC 相关性弱（r = {corr:+.3f}）：")
    print("  模型的判断不能简单归结为'看 AT'，可能依赖更细的组合模式")