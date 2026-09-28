"""模块 03：基因组背景 GC 分布，以及与结合位点的对比"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FASTA  = r"D:\code\nap-strep\data\reference\FR845719.fasta"
POS    = r"D:\code\nap-strep\data\processed\positives.fa"
OUTDIR = Path(r"D:\code\nap-strep\results\figures")
OUTDIR.mkdir(parents=True, exist_ok=True)
WIN = 201

# ---------- 1. 读基因组 ----------
header, chunks = None, []
with open(FASTA, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is None:
                header = line
        else:
            chunks.append(line)
genome = "".join(chunks).upper()

gc_all = (genome.count("G") + genome.count("C")) / len(genome)
n_bad = len(genome) - sum(genome.count(b) for b in "ACGT")
print("=" * 62)
print("1. 基因组整体")
print("=" * 62)
print("长度    ：", len(genome))
print(f"整体 GC ：{gc_all * 100:.2f}%")
print("非 ACGT ：", n_bad)

# ---------- 2. 切成非重叠窗口 ----------
n_win = len(genome) // WIN
gcs = []
skipped = 0
for i in range(n_win):
    w = genome[i * WIN:(i + 1) * WIN]
    if set(w) - set("ACGT"):        # 含 N 的窗口丢掉
        skipped += 1
        continue
    gcs.append((w.count("G") + w.count("C")) / WIN)
gcs = np.array(gcs) * 100

print()
print("=" * 62)
print("2. 基因组背景（201 bp 非重叠窗口）")
print("=" * 62)
print("窗口总数：", n_win, "  含 N 跳过：", skipped, "  有效：", len(gcs))
print(f"背景 GC：均值 {gcs.mean():.2f}%  中位 {np.median(gcs):.2f}%  "
      f"范围 {gcs.min():.2f}% ~ {gcs.max():.2f}%")
print(f"最 AT 富集的 1% 窗口，GC 低于 {np.percentile(gcs, 1):.2f}%")

# ---------- 3. 读正样本 ----------
pos = []
with open(POS, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line and not line.startswith(">"):
            pos.append((line.count("G") + line.count("C")) / len(line) * 100)
pos = np.array(pos)

print()
print("=" * 62)
print("3. Lsr2 结合位点（正样本）")
print("=" * 62)
print(f"正样本 GC：均值 {pos.mean():.2f}%  中位 {np.median(pos):.2f}%  "
      f"范围 {pos.min():.2f}% ~ {pos.max():.2f}%")
print(f"与基因组整体相差：{gc_all * 100 - pos.mean():.2f} 个百分点")

# ---------- 4. ★核心数字：AT 富集区在这里有多稀少 ----------
thr = float(np.median(pos))
frac = float((gcs <= thr).mean())
print()
print("=" * 62)
print("4. ★核心数字★")
print("=" * 62)
print(f"正样本 GC 中位数：{thr:.2f}%")
print(f"背景窗口中 GC ≤ 该值的比例：{frac * 100:.2f}%")
print(f"→ 在 72% GC 的基因组里，'和结合位点一样 AT 富集'的区域约占 {frac * 100:.1f}%")
print(f"→ 换句话说，随机挑一段基因组，只有约 {frac * 100:.0f}% 的机会比结合位点更 AT 富集")

# ---------- 5. 画图 ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

axes[0].hist(gcs, bins=60, color="#9aa5b1", alpha=0.75, density=True,
             label="genome background")
axes[0].hist(pos, bins=40, color="#c53030", alpha=0.7, density=True,
             label="Lsr2 peaks")
axes[0].axvline(gc_all * 100, color="#2b6cb0", ls="--", lw=2,
                label=f"genome mean {gc_all * 100:.1f}%")
axes[0].axvline(thr, color="#c53030", ls="--", lw=1.5,
                label=f"peak median {thr:.1f}%")
axes[0].set_xlabel("GC content (%)")
axes[0].set_ylabel("Density")
axes[0].set_title("A. Genome background vs Lsr2 peaks")
axes[0].legend(fontsize=8)

s = np.sort(gcs)
cum = np.arange(1, len(s) + 1) / len(s)
axes[1].plot(s, cum * 100, color="#2b6cb0", lw=2)
axes[1].axvline(thr, color="#c53030", ls="--", lw=2,
                label=f"peak median {thr:.1f}%")
axes[1].axhline(frac * 100, color="#c53030", ls=":", lw=1.5)
axes[1].set_xlabel("GC content threshold (%)")
axes[1].set_ylabel("Fraction of genome windows below (%)")
axes[1].set_title("B. How rare are AT-rich regions?")
axes[1].legend(fontsize=9)

plt.tight_layout()
out = OUTDIR / "03_background_vs_peaks.png"
plt.savefig(out, dpi=160)
print()
print("图已保存：", out)