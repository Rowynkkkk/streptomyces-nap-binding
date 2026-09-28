"""
模块 04a：生成打乱型负样本 N3a / N3b

N3a  单核苷酸打乱 —— 把序列的字符顺序完全随机化
N3b  二核苷酸打乱 —— 把相邻两个字符当成一组，打乱组的顺序

两者都保证【碱基组成完全不变】（GC 严格守恒），
唯一改变的是碱基的排列顺序 —— 这正是"干净的对照实验"。

每条正样本生成 N_REP 个独立副本。
为什么需要多个副本：单个副本的"聚合位置结构"受随机种子影响很大
（实测二核苷酸打乱用种子 42 时聚合幅度达 0.122，落在零模型的 94 分位）。
多副本可以抹掉种子效应，同时把负样本池扩大 3 倍。

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

# ---------- 配置 ----------
POS     = r"D:\code\nap-strep\data\processed\positives.fa"
OUT_N3A = r"D:\code\nap-strep\data\processed\negatives_N3a.fa"
OUT_N3B = r"D:\code\nap-strep\data\processed\negatives_N3b.fa"
OUT_FIG = r"D:\code\nap-strep\results\figures\04a_shuffle_check.png"

SEED = 42
N_REP = 3


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
    每 2 个字符切一块，打乱块的顺序。
    长度 201 是奇数，最后一块只有 1 个字符，单独当成一组。
    """
    pairs = [seq[i:i + 2] for i in range(0, len(seq), 2)]
    random.shuffle(pairs)
    return "".join(pairs)


def position_gc(seqs):
    """逐位置 GC 频率，返回长度 = 序列长度的数组"""
    L = len(seqs[0])
    counts = np.zeros(L, dtype=np.float64)
    for s in seqs:
        arr = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        is_gc = ((arr == ord("G")) | (arr == ord("C"))).astype(np.float64)
        counts += is_gc[:L]
    return counts / len(seqs)


def amplitude(seqs):
    """前 101 个位置（-100 到 0）的 GC 波动幅度"""
    pg = position_gc(seqs)[:101]
    return float(pg.max() - pg.min())


# ---------- 主流程 ----------
def main():
    positives = read_fasta(POS)
    n_pos = len(positives)

    print("=" * 62)
    print("输入")
    print("=" * 62)
    print("正样本条数：", n_pos)
    print("第一条名字：", positives[0][0])
    lens = [len(s) for _, s in positives]
    print("长度范围：", min(lens), "~", max(lens))

    random.seed(SEED)
    n3a, n3b = [], []
    bad_gc = 0

    for rep in range(1, N_REP + 1):
        for name, seq in positives:
            sa = shuffle_mono(seq)
            n3a.append((f"{name}_rep{rep}", sa))
            if gc_percent(sa) != gc_percent(seq):
                bad_gc += 1

            sb = shuffle_dinucleotide(seq)
            n3b.append((f"{name}_rep{rep}", sb))
            if gc_percent(sb) != gc_percent(seq):
                bad_gc += 1

    write_fasta(n3a, OUT_N3A)
    write_fasta(n3b, OUT_N3B)

    print()
    print("=" * 62)
    print("生成结果")
    print("=" * 62)
    print(f"每条正样本的副本数：{N_REP}")
    print(f"N3a 条数：{len(n3a)}   长度范围 {min(len(s) for _, s in n3a)} ~ {max(len(s) for _, s in n3a)}")
    print(f"N3b 条数：{len(n3b)}   长度范围 {min(len(s) for _, s in n3b)} ~ {max(len(s) for _, s in n3b)}")
    print(f"GC 不守恒总数：{bad_gc}")

    def max_gc_diff(records):
        worst = 0.0
        for rep in range(N_REP):
            block = records[rep * n_pos:(rep + 1) * n_pos]
            for (_, s0), (_, s1) in zip(positives, block):
                worst = max(worst, abs(gc_percent(s1) - gc_percent(s0)))
        return worst

    print()
    print(f"N3a 逐条 GC 最大偏差：{max_gc_diff(n3a):.10f}")
    print(f"N3b 逐条 GC 最大偏差：{max_gc_diff(n3b):.10f}")

    print()
    print("=" * 62)
    print("位置结构检查（前 101 个位置的 GC 波动幅度）")
    print("=" * 62)
    amp_pos = amplitude([s for _, s in positives])
    print(f"正样本      ：{amp_pos:.4f}")

    for label, recs in (("N3a", n3a), ("N3b", n3b)):
        for rep in range(N_REP):
            block = [s for _, s in recs[rep * n_pos:(rep + 1) * n_pos]]
            print(f"{label} 副本{rep + 1}   ：{amplitude(block):.4f}")
        print(f"{label} 合并全部 ：{amplitude([s for _, s in recs]):.4f}   <- 最终值")

    # ---- 画图 ----
    pg_pos = position_gc([s for _, s in positives])
    pg_a = position_gc([s for _, s in n3a])
    pg_b = position_gc([s for _, s in n3b])
    x = np.arange(len(pg_pos)) - 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    axes[0].plot(x, pg_pos * 100, lw=1.8, color="#c53030", label="positives")
    axes[0].plot(x, pg_a * 100, lw=1.3, color="#2b6cb0", label="N3a (mono, pooled)")
    axes[0].plot(x, pg_b * 100, lw=1.3, color="#38a169", label="N3b (dinuc, pooled)")
    axes[0].axvline(0, color="grey", ls=":", lw=1)
    axes[0].set_xlabel("Position relative to window center (bp)")
    axes[0].set_ylabel("GC frequency (%)")
    axes[0].set_title("A. Positional GC profile")
    axes[0].legend(fontsize=9)

    vals = [amplitude([s for _, s in positives]),
            amplitude([s for _, s in n3a]),
            amplitude([s for _, s in n3b])]
    axes[1].bar(["positives", "N3a", "N3b"], vals,
                color=["#c53030", "#2b6cb0", "#38a169"], width=0.55)
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.0015, f"{v:.4f}", ha="center", fontsize=10)
    axes[1].set_ylabel("GC amplitude, first 101 positions")
    axes[1].set_title("B. Structure amplitude (null mean ~0.092)")
    axes[1].set_ylim(0, max(vals) * 1.25)

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=160)

    print()
    print("已写出：", OUT_N3A)
    print("已写出：", OUT_N3B)
    print("图已保存：", OUT_FIG)


if __name__ == "__main__":
    main()
