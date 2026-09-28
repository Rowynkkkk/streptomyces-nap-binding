"""
模块 04b：生成基于基因组坐标的负样本 N1 / N2 / N4 / N5

四档的设计（已在作者机器上验证过数值）：

  N1  随机背景      从基因组随机取 201 bp 窗口
                    → GC 均值 ~73.3%，最好认的假货

  N2  GC 分布匹配   从基因组里挑 GC 和正样本分布一致的窗口
                    → GC 均值 ~66.8%（正样本 66.4%），★ 最关键的公平检验

  N4  侧翼          每个峰上下游 400~1500 bp 的位置
                    → GC 均值 ~72.6%，最合理的"未结合"对照

  N5  极端 AT 诱饵  ★ 明确设计成 N2 的子集：N2 里 GC 最低的 640 个
                    → GC 均值 ~61.1%，比正样本还 AT 富集 5.3 个百分点
                    → 用途：检测"模型是不是只会往 AT 多的方向打分"

关于"N5 是 N2 的子集"这个设计：
  基因组里极端 AT 的 201 bp 窗口总量有限。若把 N5 和 N2 做成互斥的两档，
  必然有一档要牺牲——实测发现要么 N5 不够极端（只低 1.7 个点），
  要么 N2 的低端被截断（差 1.2 个点，GC 匹配失败）。
  所以改为让 N5 = N2 的 AT 富集子集，重叠 100%、含义清晰。
  每一档都独立评测，重叠不影响各自的 AUC。

所有窗口都避开正样本（两侧各留 200 bp）与含 N 的区域。

输出：
  data\\processed\\negatives_N1.fa
  data\\processed\\negatives_N2.fa
  data\\processed\\negatives_N4.fa
  data\\processed\\negatives_N5.fa
  results\\figures\\04b_negative_gc.png
"""

import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 保险：Windows 控制台默认是 GBK 编码，遇到 GBK 里没有的字符（如数学符号）
# 会直接抛 UnicodeEncodeError 让脚本崩溃。这里让它降级成 '?' 而不是崩。
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

FASTA   = r"D:\code\nap-strep\data\reference\FR845719.fasta"
POS     = r"D:\code\nap-strep\data\processed\positives.fa"
OUTDIR  = r"D:\code\nap-strep\data\processed"
OUT_FIG = r"D:\code\nap-strep\results\figures\04b_negative_gc.png"

WIN = 201
STEP = 201            # 窗口池用非重叠窗口
NEG_PER_TIER = 1920   # N1 / N2 的目标数量（= 每条正样本 3 条）
N5_SIZE = 640         # N5 的数量（= 正样本数）
FLANK_PER_PEAK = 3    # N4 每条正样本配几条
SEED = 42
PEAK_MARGIN = 200     # 避让正样本时两侧各多留 200 bp


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


def write_fasta(records, path):
    with open(path, "w", encoding="utf-8") as fh:
        for name, seq in records:
            fh.write(">" + name + "\n")
            fh.write(seq + "\n")


def gc_percent(seq):
    return (seq.count("G") + seq.count("C")) / len(seq) * 100


# ================= 读基因组 =================
print("读取基因组 ...")
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
GLEN = len(genome)
print("  ", header)
print("   长度：", GLEN)


def extract(pos):
    """取从 1-based 位置 pos 开始的 201 bp 窗口；越界或含 N 返回 None"""
    if pos < 1 or pos + WIN - 1 > GLEN:
        return None
    s = genome[pos - 1: pos - 1 + WIN]
    if set(s) - set("ACGT"):
        return None
    return s


# ================= 读正样本（名字里带坐标）=================
positives = read_fasta(POS)
peaks = []
for name, seq in positives:
    parts = name.split("_")
    peaks.append((name, int(parts[-2]), int(parts[-1]), gc_percent(seq)))

pos_gc = np.array([p[3] for p in peaks])
print(f"\n正样本 {len(peaks)} 条，GC 均值 {pos_gc.mean():.2f}%  "
      f"中位 {np.median(pos_gc):.2f}%  范围 {pos_gc.min():.2f}% ~ {pos_gc.max():.2f}%")

# ================= 建立窗口池 =================
occupied = sorted((s - PEAK_MARGIN, e + PEAK_MARGIN) for _, s, e, _ in peaks)


def overlaps_peak(pos):
    lo, hi = pos, pos + WIN - 1
    for a, b in occupied:
        if a > hi:
            break
        if b < lo:
            continue
        return True
    return False


pool = []
for pos in range(1, GLEN - WIN + 2, STEP):
    if overlaps_peak(pos):
        continue
    s = extract(pos)
    if s is None:
        continue
    pool.append((pos, s, gc_percent(s)))

pool_gc = np.array([w[2] for w in pool])
print(f"\n可用窗口池：{len(pool)} 个")
print(f"  池内 GC：均值 {pool_gc.mean():.2f}%  "
      f"范围 {pool_gc.min():.2f}% ~ {pool_gc.max():.2f}%")

rng = random.Random(SEED)

# ================= N2：最近邻匹配（核心）=================
# 对每条正样本的 GC 值，从池里找 GC 最接近的窗口。
# 从 GC 高的目标开始配，先满足紧俏的高端。
targets = np.tile(pos_gc, 3)
targets = targets[np.argsort(-targets)]

n2_idx = []
n2_used = np.zeros(len(pool), dtype=bool)
for t in targets:
    d = np.abs(pool_gc - t)
    d[n2_used] = 1e9
    i = int(np.argmin(d))
    if d[i] >= 1e9:
        break
    n2_used[i] = True
    n2_idx.append(i)

n2_gc = pool_gc[n2_idx]
n2 = [(f"FR845719_{pool[i][0]}", pool[i][1]) for i in n2_idx]

# ================= N5：N2 里最 AT 富集的子集 =================
n2_sorted = sorted(range(len(n2_idx)), key=lambda k: n2_gc[k])
n5_pos = n2_sorted[:N5_SIZE]
n5_idx = [n2_idx[k] for k in n5_pos]
n5_gc = pool_gc[n5_idx]
n5 = [(f"FR845719_{pool[i][0]}", pool[i][1]) for i in n5_idx]

# ================= N1：从剩下的窗口里随机抽 =================
available = ~n2_used
avail_list = np.flatnonzero(available)
n1_idx = rng.sample(list(avail_list), min(NEG_PER_TIER, len(avail_list)))
n1_gc = pool_gc[n1_idx]
n1 = [(f"FR845719_{pool[i][0]}", pool[i][1]) for i in n1_idx]

# ================= N4：侧翼 =================
offsets = [400, 600, 800, 1000, 1200, 1500,
           -400, -600, -800, -1000, -1200, -1500]
n4 = []
for name, s, e, _ in peaks:
    center = (s + e) // 2
    got = 0
    for off in offsets:
        if got >= FLANK_PER_PEAK:
            break
        pos = center + off - WIN // 2
        if overlaps_peak(pos):
            continue
        seq = extract(pos)
        if seq is None:
            continue
        n4.append((f"{name}_flank{off:+d}", seq))
        got += 1

n4_gc = np.array([gc_percent(s) for _, s in n4])

# ================= 写出 =================
for label, recs in (("N1", n1), ("N2", n2), ("N4", n4), ("N5", n5)):
    path = rf"{OUTDIR}\negatives_{label}.fa"
    write_fasta(recs, path)
    print(f"  已写出 {label}: {len(recs)} 条 -> {path}")

# ================= 报告 =================
print()
print("=" * 72)
print(f"{'档次':<8}{'数量':>7}{'GC均值':>9}{'GC中位':>9}{'GC最小':>9}{'GC最大':>9}{'与正样本差':>12}")
print("=" * 72)
print(f"{'正样本':<8}{len(pos_gc):>7}{pos_gc.mean():>9.2f}{np.median(pos_gc):>9.2f}"
      f"{pos_gc.min():>9.2f}{pos_gc.max():>9.2f}{'—':>12}")
for label, g in (("N1", n1_gc), ("N2", n2_gc), ("N4", n4_gc), ("N5", n5_gc)):
    print(f"{label:<8}{len(g):>7}{g.mean():>9.2f}{np.median(g):>9.2f}"
          f"{g.min():>9.2f}{g.max():>9.2f}{abs(g.mean() - pos_gc.mean()):>12.2f}")

print()
print("设计校验：")
print(f"  N2 应与正样本接近  -> 差 {abs(n2_gc.mean() - pos_gc.mean()):.2f} 个百分点")
print(f"  N5 应明显更 AT     -> 差 {abs(n5_gc.mean() - pos_gc.mean()):.2f} 个百分点")
overlap_n5_n2 = len(set(n5_idx) & set(n2_idx))
print(f"  N5 是 N2 的子集    -> {overlap_n5_n2} / {len(n5_idx)}（应为 100%）")

# ================= 画图 =================
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.hist(pos_gc, bins=40, density=True, histtype="step", lw=2.8,
        color="#c53030", label=f"positives (n={len(pos_gc)}, mean {pos_gc.mean():.1f}%)")
for label, g, color in (("N1 random", n1_gc, "#9aa5b1"),
                        ("N2 GC-matched", n2_gc, "#2b6cb0"),
                        ("N4 flanking", n4_gc, "#805ad5"),
                        ("N5 extreme AT", n5_gc, "#38a169")):
    ax.hist(g, bins=40, density=True, histtype="step", lw=1.8, color=color,
            label=f"{label} (n={len(g)}, mean {g.mean():.1f}%)")

ax.set_xlabel("GC content (%)")
ax.set_ylabel("Density")
ax.set_title("Negative sample tiers: GC distributions")
ax.legend(fontsize=8.5)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=160)

print()
print("图已保存：", OUT_FIG)
print()
print("注：N3a / N3b 的 GC 与正样本完全相同（打乱不改变组成），")
print("    曲线与正样本重合，故未在图上单独显示。")
