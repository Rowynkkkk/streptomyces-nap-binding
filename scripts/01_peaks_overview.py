"""模块 01：摸清 peak 表"""
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = r"D:\code\nap-strep\data\GSE115538_peaks.csv"
CHR_LEN = 8223505          # 染色体 NZ_CP029197.1 的长度

df = pd.read_csv(CSV)

print("=" * 62)
print("1. 基本规模")
print("=" * 62)
print("峰的总数：", len(df))
print("列名：", list(df.columns))

print("\n" + "=" * 62)
print("2. 峰的宽度 Size (bp)")
print("=" * 62)
print(df["Size (bp)"].describe())

print("\n" + "=" * 62)
print("3. 染色体列 Chr")
print("=" * 62)
print("唯一取值：", df["Chr"].unique())
print("取值个数：", df["Chr"].nunique())

print("\n" + "=" * 62)
print("4. 坐标范围  ★关键检查★")
print("=" * 62)
print("Start 范围：", df["Start"].min(), "~", df["Start"].max())
print("End   范围：", df["End"].min(), "~", df["End"].max())
print("染色体长度：", CHR_LEN)
if df["End"].max() <= CHR_LEN:
    print(">>> 通过：坐标没有超出染色体长度")
else:
    print(">>> 警告：坐标超出染色体长度，坐标体系不一致！")

# %GC 那一列是带百分号的字符串，解析成数字
def parse_pct(x):
    m = re.search(r"(-?\d+\.?\d*)", str(x))
    return float(m.group(1)) if m else float("nan")

df["pct_table"] = df["%GC"].apply(parse_pct)

# 用 Sequence 列自己算一遍 GC
def gc_content(s):
    s = str(s).upper()
    return (s.count("G") + s.count("C")) / len(s) if s else float("nan")

df["my_GC"] = df["Sequence"].apply(gc_content)

print("\n" + "=" * 62)
print("5. GC 含量")
print("=" * 62)
print("表里 %GC 的绝对值统计：")
print(df["pct_table"].abs().describe())
print("\n自己算的 my_GC（×100 后应与上面接近）：")
print((df["my_GC"] * 100).describe())

print("\n" + "=" * 62)
print("6. 序列长度是否等于 Size (bp)")
print("=" * 62)
print("Sequence 长度范围：", df["Sequence"].str.len().min(), "~", df["Sequence"].str.len().max())
print("Size (bp)   范围：", df["Size (bp)"].min(), "~", df["Size (bp)"].max())

print("\n" + "=" * 62)
print("7. 其他数值列")
print("=" * 62)
for col in ["Conc", "Fold", "p-value", "FDR"]:
    if col in df.columns:
        v = pd.to_numeric(df[col], errors="coerce")
        print(f"{col:>10} : {v.min():.4g} ~ {v.max():.4g}")

# 画峰宽分布
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(df["Size (bp)"], bins=50, color="#2b6cb0")
axes[0].set_xlabel("Peak width (bp)")
axes[0].set_ylabel("Count")
axes[0].set_title("A. Peak width distribution")
axes[1].hist(df["my_GC"] * 100, bins=50, color="#c53030")
axes[1].set_xlabel("GC content of peak (%)")
axes[1].set_ylabel("Count")
axes[1].set_title("B. Peak GC distribution")
plt.tight_layout()
plt.savefig(r"D:\code\nap-strep\results\figures\01_peaks.png", dpi=150)
print("\n图已保存至 results\\figures\\01_peaks.png")