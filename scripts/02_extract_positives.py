"""模块 02：提取正样本序列（201 bp 中心窗口）"""
import pandas as pd
from pathlib import Path

CSV    = r"D:\code\nap-strep\data\GSE115538_peaks.csv"      # ← 改成实际文件名
FASTA  = r"D:\code\nap-strep\data\reference\FR845719.fasta"
OUTDIR = Path(r"D:\code\nap-strep\data\processed")
OUTDIR.mkdir(parents=True, exist_ok=True)

HALF = 100          # 中心两侧各取 100 bp
WIN  = 2 * HALF + 1 # = 201

# ---------- 1. 读基因组 ----------
# ★ 注意：FASTA 里一条序列是分很多行的，必须先拼接
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
print("基因组头部：", header)
print("基因组长度：", len(genome))
print()

df = pd.read_csv(CSV)
print("读入峰数：", len(df))

# ---------- 2. 逐个峰取中心窗口 ----------
records = []
skipped_edge = 0
for idx, row in df.iterrows():
    start, end = int(row["Start"]), int(row["End"])
    center = (start + end) // 2
    lo, hi = center - HALF, center + HALF
    if lo < 1 or hi > len(genome):       # 落在染色体两端，跳过
        skipped_edge += 1
        continue
    # ★ 坐标换算：表的区间是 1-based 闭区间 [start, end]
    #   Python 切片是 0-based 左闭右开，所以要写成 [lo-1 : hi]
    seq = genome[lo - 1: hi]
    records.append({
        "chr": row["Chr"], "start": start, "end": end,
        "seq": seq, "table_seq": str(row["Sequence"]).upper(),
    })

print("成功提取：", len(records))
print("越界跳过：", skipped_edge)
lens = [len(r["seq"]) for r in records]
print("序列长度范围：", min(lens), "~", max(lens), "（应全为 201）")
print()

# ---------- 3. ★交叉验证：我们提的序列和表里的 Sequence 对得上吗 ----------
print("=" * 60)
print("坐标体系交叉验证")
print("=" * 60)
best = None
for shift in (-2, -1, 0, 1, 2):
    matched = checked = 0
    for r in records:
        t = r["table_seq"]
        if len(t) < WIN:
            continue
        off = (len(t) - WIN) // 2 + shift
        if off < 0 or off + WIN > len(t):
            continue
        checked += 1
        if t[off: off + WIN] == r["seq"]:
            matched += 1
    if checked:
        rate = matched / checked
        print(f"偏移 {shift:+d}：{matched}/{checked} = {rate * 100:.1f}%")
        if best is None or rate > best[1]:
            best = (shift, rate, matched, checked)

print()
if best and best[1] > 0.95:
    print(f">>> 坐标体系确认（最佳偏移 {best[0]:+d}，一致率 {best[1] * 100:.1f}%）")
elif best:
    print(f">>> 一致率偏低（最佳 {best[1] * 100:.1f}%），需要排查")
else:
    print(">>> 没有可比对的峰（所有峰都短于 201 bp？）")

# ---------- 4. 检查 N 区域 ----------
bad = [r for r in records if set(r["seq"]) - set("ACGT")]
print()
print(f"含非 ACGT 字符的窗口：{len(bad)} / {len(records)}")
for r in bad[:5]:
    print(f"   {r['chr']}:{r['start']}-{r['end']}")

# ---------- 5. 峰太短的会有多少 ----------
short = int((df["Size (bp)"] < WIN).sum())
print()
print(f"峰宽 < {WIN} bp 的有 {short} 个（这些峰的中心窗口会超出峰本身）")

# ---------- 6. 写出 ----------
out = OUTDIR / "positives.fa"
with open(out, "w", encoding="utf-8") as fh:
    for r in records:
        fh.write(f">{r['chr']}_{r['start']}_{r['end']}\n{r['seq']}\n")
print()
print("已写出：", out)