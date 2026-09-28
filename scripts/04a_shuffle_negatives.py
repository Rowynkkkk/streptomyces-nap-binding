"""
模块 04a：生成打乱型负样本 N3a / N3b

N3a  单核苷酸打乱 —— 把序列的字符顺序完全随机化
N3b  二核苷酸打乱 —— 把相邻两个字符当成一组，打乱组的顺序

两者都保证【碱基组成完全不变】（GC 严格守恒），
唯一改变的是碱基的排列顺序 —— 这正是"干净的对照实验"，
因为它只动了一个变量。

输出：
  data\\processed\\negatives_N3a.fa
  data\\processed\\negatives_N3b.fa
  results\\figures\\04a_shuffle_check.png
"""

import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 路径 ----------
POS       = r"D:\code\nap-strep\data\processed\positives.fa"
OUT_N3A   = r"D:\code\nap-strep\data\processed\negatives_N3a.fa"
OUT_N3B   = r"D:\code\nap-strep\data\processed\negatives_N3b.fa"
OUT_FIG   = r"D:\code\nap-strep\results\figures\04a_shuffle_check.png"

SEED = 42


# ---------- 工具函数 ----------
def read_fasta(path):
    """读 FASTA，返回 [(名字, 序列), ...]"""
    recs = []
    name = None
    chunks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(chunks)))
                name = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        recs.append((name, "".join(chunks)))
    return recs


def write_fasta(records, path):
    """写 FASTA。每条两行：标题行 + 序列行"""
    with open(path, "w", encoding="utf-8") as fh:
        for name, seq in records:
            fh.write(">" + name + "\n")
            fh.write(seq + "\n")


def gc_percent(seq):
    """返回 GC 百分比"""
    return (seq.count("G") + seq.count("C")) / len(seq) * 100


def shuffle_mono(seq):
    """N3a：单核苷酸打乱。字符串不能直接打乱，先转成字符列表"""
    chars = list(seq)
    random.shuffle(chars)
    return "".join(chars)


def shuffle_dinucleotide(seq):
    """N3b：二核苷酸打乱。
    把序列每 2 个字符切成一块，打乱这些块的顺序。
    长度 201 是奇数，最后一块只有 1 个字符，单独当成一组。
    """
    pairs = [seq[i:i + 2] for i in range(0, len(seq), 2)]
    random.shuffle(pairs)
    return "".join(pairs)


def position_gc(seqs):
    """逐位置 GC 频率。返回长度 = 序列长度的数组。
    正样本应该有明显起伏（-10 元件），打乱后应该接近平坦。
    """
    L = len(seqs[0])
    counts = np.zeros(L, dtype=np.float64)
    for s in seqs:
        arr = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        is_gc = ((arr == ord("G")) | (arr == ord("C"))).astype(np.float64)
        counts += is_gc[:L]
    return counts / len(seqs)


# ---------- 主流程 ----------
def main():
    positives = read_fasta(POS)
    print("=" * 62)
    print("输入")
    print("=" * 62)
    print("正样本条数：", len(positives))
    print("第一条名字：", positives[0][0])
    lens = [len(s) for _, s in positives]
    print("长度范围：", min(lens), "~", max(lens))

    # ---- N3a ----
    random.seed(SEED)
    n3a = []
    bad_gc = 0
    for name, seq in positives:
        s = shuffle_mono(seq)
        n3a.append((name, s))
        if gc_percent(s) != gc_percent(seq):
            bad_gc += 1

    # ---- N3b ----
    random.seed(SEED)
    n3b = []
    bad_gc_b = 0
    for name, seq in positives:
        s = shuffle_dinucleotide(seq)
        n3b.append((name, s))
        if gc_percent(s) != gc_percent(seq):
            bad_gc_b += 1

    write_fasta(n3a, OUT_N3A)
    write_fasta(n3b, OUT_N3B)

    print()
    print("=" * 62)
    print("生成结果")
    print("=" * 62)
    print(f"N3a 条数：{len(n3a)}   GC 不守恒：{bad_gc}")
    print(f"N3b 条数：{len(n3b)}   GC 不守恒：{bad_gc_b}")
    print("N3a 长度范围：", min(len(s) for _, s in n3a), "~", max(len(s) for _, s in n3a))
    print("N3b 长度范围：", min(len(s) for _, s in n3b), "~", max(len(s) for _, s in n3b))

    # ---- 逐条 GC 严格比对 ----
    max_diff_a = max(abs(gc_percent(s1) - gc_percent(s0))
                     for (_, s0), (_, s1) in zip(positives, n3a))
    max_diff_b = max(abs(gc_percent(s1) - gc_percent(s0))
                     for (_, s0), (_, s1) in zip(positives, n3b))
    print()
    print(f"N3a 逐条 GC 最大偏差：{max_diff_a:.10f}")
    print(f"N3b 逐条 GC 最大偏差：{max_diff_b:.10f}")

    # ---- 位置结构检查 ----
    pg_pos = position_gc([s for _, s in positives])
    pg_a = position_gc([s for _, s in n3a])
    pg_b = position_gc([s for _, s in n3b])

    amp_pos = pg_pos[:101].max() - pg_pos[:101].min()
    amp_a = pg_a[:101].max() - pg_a[:101].min()
    amp_b = pg_b[:101].max() - pg_b[:101].min()

    print()
    print("=" * 62)
    print("位置结构检查（前 101 个位置的 GC 波动幅度）")
    print("=" * 62)
    print(f"正样本：{amp_pos:.4f}")
    print(f"N3a   ：{amp_a:.4f}")
    print(f"N3b   ：{amp_b:.4f}")
    if amp_a < amp_pos / 2 and amp_b < amp_pos / 2:
        print(">>> 通过：打乱后位置结构明显减弱")
    else:
        print(">>> 注意：打乱后仍有较强位置结构，需要检查")

    # ---- 画图 ----
    x = np.arange(len(pg_pos)) - 100
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))

    axes[0].plot(x, pg_pos * 100, lw=1.8, color="#c53030", label="positives")
    axes[0].plot(x, pg_a * 100, lw=1.4, color="#2b6cb0", label="N3a (mono shuffle)")
    axes[0].plot(x, pg_b * 100, lw=1.4, color="#38a169", label="N3b (dinucleotide shuffle)")
    axes[0].axvline(0, color="grey", ls=":", lw=1)
    axes[0].set_xlabel("Position relative to window center (bp)")
    axes[0].set_ylabel("GC frequency (%)")
    axes[0].set_title("A. Positional GC: shuffling removes structure")
    axes[0].legend(fontsize=9)

    axes[1].bar(["positives", "N3a", "N3b"], [amp_pos, amp_a, amp_b],
                color=["#c53030", "#2b6cb0", "#38a169"], width=0.55)
    for i, v in enumerate([amp_pos, amp_a, amp_b]):
        axes[1].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=10)
    axes[1].set_ylabel("GC amplitude over first 101 positions")
    axes[1].set_title("B. Structure amplitude")
    axes[1].set_ylim(0, max(amp_pos, amp_a, amp_b) * 1.25)

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=160)

    print()
    print("已写出：", OUT_N3A)
    print("已写出：", OUT_N3B)
    print("图已保存：", OUT_FIG)


if __name__ == "__main__":
    main()
