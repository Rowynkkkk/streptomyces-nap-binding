"""
模块 06：GC-only 基线 —— 只用碱基组成，能做到多好？

这是 H1 / H2 的直接检验。

设计两套实验：

  【设计 A · 匹配训练】
    对每一档负样本 T，都用"正样本 + T"训练一个新模型，在同分布上测试。
    回答：如果任务就是区分"正样本 vs T"，光靠 GC 能做到多少？

  【设计 B · 固定模型】（★ 关键）
    只在"正样本 + N1（最简单的随机背景）"上训练一次，
    然后拿这同一个模型去测每一档负样本。
    回答：一个靠组成建起来的模型，遇到难假货时会怎样？
    预期：在 N2 / N3a / N3b / N5 上崩塌到接近随机。

输出：
  results/tables/06_gc_baseline.csv
  results/figures/06_gc_baseline.png
"""

import sys
from pathlib import Path
import os

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
# 输出位置。默认写到项目里；设置环境变量 NAP_OUT_ROOT 可以改到别处（用于测试）。
OUT_ROOT = Path(os.environ.get("NAP_OUT_ROOT", str(BASE)))
TABDIR = OUT_ROOT / "results" / "tables"
FIGDIR = OUT_ROOT / "results" / "figures"
TABDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

TIERS = ["N1", "N2", "N3a", "N3b", "N4", "N5"]
TIER_LABEL = {
    "N1": "N1 random",
    "N2": "N2 GC-matched",
    "N3a": "N3a mono shuffle",
    "N3b": "N3b dinuc shuffle",
    "N4": "N4 flanking",
    "N5": "N5 extreme AT",
}
SPLIT = 0.8
SEED = 42


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


def gc_features(seqs):
    """只用 GC 含量这一个特征，返回 (n, 1) 数组"""
    out = np.empty((len(seqs), 1), dtype=np.float64)
    for i, s in enumerate(seqs):
        s = s.upper()
        out[i, 0] = (s.count("G") + s.count("C")) / len(s)
    return out


def gc_of(seqs):
    return np.array([(s.upper().count("G") + s.upper().count("C")) / len(s)
                     for s in seqs]) * 100


def evaluate(y_true, scores):
    return (roc_auc_score(y_true, scores),
            average_precision_score(y_true, scores),
            matthews_corrcoef(y_true, (scores >= 0.5).astype(int)))


# ================= 读数据 =================
pos = [s for _, s in read_fasta(PROC / "positives.fa")]
neg = {}
for t in TIERS:
    neg[t] = [s for _, s in read_fasta(PROC / f"negatives_{t}.fa")]

print("=" * 74)
print("数据概览")
print("=" * 74)
print(f"正样本：{len(pos)} 条   GC 均值 {gc_of(pos).mean():.2f}%")
for t in TIERS:
    print(f"  {t:<4} {len(neg[t]):>5} 条   GC 均值 {gc_of(neg[t]).mean():.2f}%")

# ================= 划分 =================
rng = np.random.default_rng(SEED)


def split(n):
    idx = rng.permutation(n)
    k = int(n * SPLIT)
    return idx[:k], idx[k:]


pos_tr, pos_te = split(len(pos))
neg_idx = {t: split(len(neg[t])) for t in TIERS}

Xpos_tr, Xpos_te = gc_features([pos[i] for i in pos_tr]), gc_features([pos[i] for i in pos_te])

print()
print(f"划分：训练 {len(pos_tr)} 正 / 测试 {len(pos_te)} 正（{SPLIT:.0%}）")

# ================= 设计 A：匹配训练 =================
print()
print("=" * 74)
print("设计 A · 匹配训练（每档单独训练一个模型）")
print("=" * 74)
print(f"{'档次':<18}{'n_neg_tr':>9}{'n_neg_te':>9}{'AUC':>9}{'PR-AUC':>9}{'MCC':>9}")
print("-" * 74)

rows = []
resA = {}
for t in TIERS:
    tr, te = neg_idx[t]
    Xtr = np.vstack([Xpos_tr, gc_features([neg[t][i] for i in tr])])
    ytr = np.r_[np.ones(len(pos_tr)), np.zeros(len(tr))]
    Xte = np.vstack([Xpos_te, gc_features([neg[t][i] for i in te])])
    yte = np.r_[np.ones(len(pos_te)), np.zeros(len(te))]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    auc, ap, mcc = evaluate(yte, s)
    resA[t] = (auc, ap, mcc)
    print(f"{TIER_LABEL[t]:<18}{len(tr):>9}{len(te):>9}{auc:>9.4f}{ap:>9.4f}{mcc:>9.4f}")
    rows.append(dict(tier=t, design="matched", auc=round(auc, 4),
                     pr_auc=round(ap, 4), mcc=round(mcc, 4)))

# ================= 设计 B：固定模型（在 N1 上训练）=================
print()
print("=" * 74)
print("设计 B · 固定模型（只在 N1 上训练，然后测所有档）")
print("=" * 74)

t0_tr, _ = neg_idx["N1"]
Xtr = np.vstack([Xpos_tr, gc_features([neg["N1"][i] for i in t0_tr])])
ytr = np.r_[np.ones(len(pos_tr)), np.zeros(len(t0_tr))]
clf_fixed = LogisticRegression(max_iter=1000)
clf_fixed.fit(Xtr, ytr)

boundary = -clf_fixed.intercept_[0] / clf_fixed.coef_[0, 0] * 100
print(f"（模型学到的 GC 分界点：{boundary:.2f}%）")
print(f"{'档次':<18}{'AUC':>9}{'PR-AUC':>9}{'MCC':>9}")
print("-" * 74)

resB = {}
for t in TIERS:
    _, te = neg_idx[t]
    Xte = np.vstack([Xpos_te, gc_features([neg[t][i] for i in te])])
    yte = np.r_[np.ones(len(pos_te)), np.zeros(len(te))]
    s = clf_fixed.predict_proba(Xte)[:, 1]
    auc, ap, mcc = evaluate(yte, s)
    resB[t] = (auc, ap, mcc, s, yte)
    print(f"{TIER_LABEL[t]:<18}{auc:>9.4f}{ap:>9.4f}{mcc:>9.4f}")
    rows.append(dict(tier=t, design="fixed_from_N1", auc=round(auc, 4),
                     pr_auc=round(ap, 4), mcc=round(mcc, 4)))

# 保存表格
import csv
tab = TABDIR / "06_gc_baseline.csv"
with tab.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=["tier", "design", "auc", "pr_auc", "mcc"])
    w.writeheader()
    w.writerows(rows)
print()
print("表已保存：", tab)

# ================= 画图 =================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

x = np.arange(len(TIERS))
w = 0.38
axes[0].bar(x - w / 2, [resA[t][0] for t in TIERS], w,
            label="Design A: matched training", color="#2b6cb0")
axes[0].bar(x + w / 2, [resB[t][0] for t in TIERS], w,
            label="Design B: fixed model (trained on N1)", color="#c53030")
axes[0].axhline(0.5, ls="--", lw=1.2, color="grey")
axes[0].text(0.02, 0.52, "random", transform=axes[0].transAxes,
             fontsize=9, color="grey")
axes[0].set_xticks(x)
axes[0].set_xticklabels([TIER_LABEL[t] for t in TIERS], rotation=25, ha="right", fontsize=9)
axes[0].set_ylabel("AUC (GC content only)")
# 纵轴必须从 0 开始，否则 N5 设计 B 那根 0.21 的柱子会被裁掉
axes[0].set_ylim(0, 1.05)
axes[0].set_title("A. GC-only baseline across negative tiers")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.25, axis="y")

# 在每根柱子上标数值
for i, t in enumerate(TIERS):
    axes[0].text(i - w / 2, resA[t][0] + 0.015, f"{resA[t][0]:.2f}",
                 ha="center", fontsize=8, color="#2b6cb0")
    axes[0].text(i + w / 2, resB[t][0] + 0.015, f"{resB[t][0]:.2f}",
                 ha="center", fontsize=8, color="#c53030")

# 右图：N2 上的分数分布（固定模型）
_, _, _, s2, y2 = resB["N2"]
axes[1].hist(s2[y2 == 1] * 100, bins=30, alpha=0.75, color="#c53030",
             label=f"positives (n={int((y2 == 1).sum())})")
axes[1].hist(s2[y2 == 0] * 100, bins=30, alpha=0.75, color="#2b6cb0",
             label=f"N2 GC-matched decoys (n={int((y2 == 0).sum())})")
axes[1].set_xlabel("Predicted probability of being a peak (%)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"B. Design B on N2: the two distributions overlap\n"
                  f"AUC = {resB['N2'][0]:.3f}")
axes[1].legend(fontsize=9)

plt.tight_layout()
figpath = FIGDIR / "06_gc_baseline.png"
plt.savefig(figpath, dpi=160)
print("图已保存：", figpath)

# ================= 结论 =================
print()
print("=" * 74)
print("H1 / H2 的判断")
print("=" * 74)
a1 = resA["N1"][0]
b2 = resB["N2"][0]
b3a = resB["N3a"][0]
print(f"H1（GC 在随机背景 N1 上能拿高分）：AUC = {a1:.4f}  "
      f"{'成立' if a1 >= 0.75 else '不成立'}")
print(f"H2（固定模型在 GC 匹配的 N2 上崩塌）：AUC = {b2:.4f}  "
      f"{'成立' if abs(b2 - 0.5) < 0.08 else '不成立'}")
print(f"    在打乱样本 N3a 上：AUC = {b3a:.4f}  "
      f"{'成立' if abs(b3a - 0.5) < 0.08 else '不成立'}")
print(f"    在极端 AT 诱饵 N5 上：AUC = {resB['N5'][0]:.4f}  "
      f"（远低于 0.5，说明判断方向被负样本集带反了）")
