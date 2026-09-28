"""
第一课：把 Lsr2 结合位点表读明白
"""
from pathlib import Path
import pandas as pd

# ↓↓↓ 改成你的实际文件名 ↓↓↓
CSV_PATH = r"D:\code\nap-strep\data\GSE115538_peaks.csv"

if not Path(CSV_PATH).exists():
    print("找不到文件：", CSV_PATH)
    print("data 目录下现有文件：")
    for p in Path(r"D:\code\nap-strep\data").glob("*"):
        print("   ", p.name)
    raise SystemExit(1)

df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("基本信息")
print("=" * 60)
print("峰的数量：", len(df))
print("列名：", list(df.columns))
print()

print("=" * 60)
print("%GC 列的分布")
print("=" * 60)
print(df["%GC"].describe())
print()

def gc_content(seq):
    s = str(seq).upper()
    return (s.count("G") + s.count("C")) / len(s) if s else float("nan")

df["my_GC"] = df["Sequence"].apply(gc_content)

print("=" * 60)
print("交叉验证：自己算的 GC vs 表里的 %GC")
print("=" * 60)
print("Sequence 长度范围：", df["Sequence"].str.len().min(), "~", df["Sequence"].str.len().max())
print("Size (bp) 范围    ：", df["Size (bp)"].min(), "~", df["Size (bp)"].max())
print()
print("我们自己算的 GC（小数形式）：")
print(df["my_GC"].describe())
print()
print("前 5 行对照：")
print(df[["Size (bp)", "%GC", "my_GC"]].head().to_string())