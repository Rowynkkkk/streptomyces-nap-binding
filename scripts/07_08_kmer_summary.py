"""
模块 07 + 08：k-mer 基线与汇总表 —— 把组成信息榨干，M1 里程碑

模块 06 只用了 GC 一个特征。这里把它扩展到完整的碱基组成：
  1-mer（4 维）→ 2-mer（16 维）→ 3-mer（64 维）→ 4-mer（256 维）

要回答的问题：
  1. 光靠组成，最多能做到多好？（组成的天花板）
  2. 从 1 维加到 256 维，涨了多少？值不值得？
  3. 在 GC 匹配（N2）和打乱（N3a/N3b）的难假货上，高阶组成能不能救回来？

两套设计沿用模块 06：
  A · 匹配训练：每档单独训练
  B · 固定模型：只在 N1 上训练，然后测所有档

输出：
  results/tables/07_kmer_baseline.csv
  results/tables/08_summary_M1.csv      <- 里程碑 M1 的汇总表
  results/figures/07_kmer_baseline.png
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             roc_auc_score)

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
FEATS = [("gc", 1), ("1mer", 4), ("2mer", 16), ("3mer", 64), ("4mer", 256)]
SPLIT = 0.8
SEED = 42
BASES = "ACGT"


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


_KMER_INDEX = {}


def kmer_features(seqs, k):
    """返回 (n, 4^k) 的频率矩阵"""
    if k == 0:
        out = np.empty((len(seqs), 1))
        for i, s in enumerate(seqs):
            s = s.upper()
            out[i, 0] = (s.count("G") + s.count("C")) / len(s)
        return out

    if k not in _KMER_INDEX:
        keys = []
        for j in range(4 ** k):
            t, x = [], j
            for _ in range(k):
                t.append(BASES[x % 4])
                x //= 4
            keys.append("".join(t))
        _KMER_INDEX[k] = {s: i for i, s in enumerate(keys)}
    idx = _KMER_INDEX[k]

    out = np.zeros((len(seqs), 4 ** k))
    for i, s in enumerate(seqs):
        u = s.upper()
        for j in range(len(u) - k + 1):
            p = idx.get(u[j:j + k])
            if p is not None:
                out[i, p] += 1
    return out / max(len(seqs[0]) - k + 1, 1)


def features(seqs, kind):
    if kind == "gc":
        return kmer_features(seqs, 0)
    return kmer_features(seqs, int(kind[0]))


def evaluate(y, s):
    return (roc_auc_score(y, s), average_precision_score(y, s),
            matthews_corrcoef(y, (s >= 0.5).astype(int)))


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

print("=" * 76)
print("数据")
print("=" * 76)
print(f"正样本 {len(pos)} 条；训练 {len(pos_tr)} / 测试 {len(pos_te)}")
for t in TIERS:
    print(f"  {TIER_LABEL[t]:<18} {len(neg[t]):>5} 条")

# ================= 预计算特征 =================
print()
print("预计算特征矩阵 ...")
FX = {}
for kind, dim in FEATS:
    Xp_tr = features([pos[i] for i in pos_tr], kind)
    Xp_te = features([pos[i] for i in pos_te], kind)
    Xn_tr, Xn_te = {}, {}
    for t in TIERS:
        tr, te = neg_idx[t]
        Xn_tr[t] = features([neg[t][i] for i in tr], kind)
        Xn_te[t] = features([neg[t][i] for i in te], kind)
    FX[kind] = (Xp_tr, Xp_te, Xn_tr, Xn_te)
    print(f"  {kind:<5} ({dim:>3} 维)  完成")

# ================= 设计 A：匹配训练 =================
print()
print("=" * 90)
print("设计 A · 匹配训练")
print("=" * 90)
header = f"{'档次':<18}" + "".join(f"{k:>10}" for k, _ in FEATS)
print(header)
print("-" * 90)

rows = []
resA = {}
for t in TIERS:
    line = f"{TIER_LABEL[t]:<18}"
    resA[t] = {}
    for kind, _ in FEATS:
        Xp_tr, Xp_te, Xn_tr, Xn_te = FX[kind]
        Xtr = np.vstack([Xp_tr, Xn_tr[t]])
        ytr = np.r_[np.ones(len(pos_tr)), np.zeros(len(Xn_tr[t]))]
        Xte = np.vstack([Xp_te, Xn_te[t]])
        yte = np.r_[np.ones(len(pos_te)), np.zeros(len(Xn_te[t]))]
        clf = LogisticRegression(max_iter=3000)
        clf.fit(Xtr, ytr)
        auc, ap, mcc = evaluate(yte, clf.predict_proba(Xte)[:, 1])
        resA[t][kind] = auc
        line += f"{auc:>10.4f}"
        rows.append(dict(design="matched", tier=t, feature=kind,
                         auc=round(auc, 4), pr_auc=round(ap, 4), mcc=round(mcc, 4)))
    print(line)

# ================= 设计 B：固定模型 =================
print()
print("=" * 90)
print("设计 B · 固定模型（只在 N1 上训练）")
print("=" * 90)
print(header)
print("-" * 90)

resB = {}
for t in TIERS:
    line = f"{TIER_LABEL[t]:<18}"
    resB[t] = {}
    for kind, _ in FEATS:
        Xp_tr, Xp_te, Xn_tr, Xn_te = FX[kind]
        Xtr = np.vstack([Xp_tr, Xn_tr["N1"]])
        ytr = np.r_[np.ones(len(pos_tr)), np.zeros(len(Xn_tr["N1"]))]
        clf = LogisticRegression(max_iter=3000)
        clf.fit(Xtr, ytr)
        Xte = np.vstack([Xp_te, Xn_te[t]])
        yte = np.r_[np.ones(len(pos_te)), np.zeros(len(Xn_te[t]))]
        auc, ap, mcc = evaluate(yte, clf.predict_proba(Xte)[:, 1])
        resB[t][kind] = auc
        line += f"{auc:>10.4f}"
        rows.append(dict(design="fixed_from_N1", tier=t, feature=kind,
                         auc=round(auc, 4), pr_auc=round(ap, 4), mcc=round(mcc, 4)))
    print(line)

tab = TABDIR / "07_kmer_baseline.csv"
with tab.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=["design", "tier", "feature", "auc", "pr_auc", "mcc"])
    w.writeheader()
    w.writerows(rows)
print()
print("表已保存：", tab)

# ================= 里程碑 M1 汇总 =================
summary = []
for t in TIERS:
    summary.append(dict(
        tier=TIER_LABEL[t],
        auc_gc_matched=round(resA[t]["gc"], 4),
        auc_4mer_matched=round(resA[t]["4mer"], 4),
        delta_matched=round(resA[t]["4mer"] - resA[t]["gc"], 4),
        auc_gc_fixed=round(resB[t]["gc"], 4),
        auc_4mer_fixed=round(resB[t]["4mer"], 4),
        delta_fixed=round(resB[t]["4mer"] - resB[t]["gc"], 4),
    ))
sumtab = TABDIR / "08_summary_M1.csv"
with sumtab.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

print()
print("=" * 90)
print("里程碑 M1 汇总：GC（1 维） vs 4-mer（256 维）")
print("=" * 90)
print(f"{'档次':<20}{'GC(match)':>11}{'4mer(match)':>13}{'Δ':>9}"
      f"{'GC(fixed)':>11}{'4mer(fixed)':>13}{'Δ':>9}")
print("-" * 90)
for s in summary:
    print(f"{s['tier']:<20}{s['auc_gc_matched']:>11.4f}{s['auc_4mer_matched']:>13.4f}"
          f"{s['delta_matched']:>+9.4f}{s['auc_gc_fixed']:>11.4f}"
          f"{s['auc_4mer_fixed']:>13.4f}{s['delta_fixed']:>+9.4f}")
print()
print("表已保存：", sumtab)

# ================= 画图 =================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
x = np.arange(len(TIERS))
for ax, res, title in ((axes[0], resA, "A. Design A: matched training"),
                       (axes[1], resB, "B. Design B: fixed model (trained on N1)")):
    for kind, color in zip(["gc", "2mer", "4mer"], ["#c53030", "#dd6b20", "#2b6cb0"]):
        ax.plot(x, [res[t][kind] for t in TIERS], "o-", lw=2, ms=7,
                color=color, label=kind)
    ax.axhline(0.5, ls="--", lw=1.2, color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABEL[t] for t in TIERS], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

plt.tight_layout()
figpath = FIGDIR / "07_kmer_baseline.png"
plt.savefig(figpath, dpi=160)
print()
print("图已保存：", figpath)

# ================= 结论 =================
print()
print("=" * 90)
print("结论")
print("=" * 90)
print(f"组成的天花板（N1，4-mer）：AUC = {resA['N1']['4mer']:.4f}")
print(f"GC 单独达到（N1）        ：AUC = {resA['N1']['gc']:.4f}")
print(f"→ 从 1 维加到 256 维，提升 {resA['N1']['4mer'] - resA['N1']['gc']:+.4f}")
print()
print(f"难假货上的组成天花板：")
print(f"  N2（GC 匹配）  GC {resA['N2']['gc']:.4f} → 4-mer {resA['N2']['4mer']:.4f}")
print(f"  N3a（打乱）    GC {resA['N3a']['gc']:.4f} → 4-mer {resA['N3a']['4mer']:.4f}")
print(f"  N3b（双核苷酸打乱）GC {resA['N3b']['gc']:.4f} → 4-mer {resA['N3b']['4mer']:.4f}")
