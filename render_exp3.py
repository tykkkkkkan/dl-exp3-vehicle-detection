# -*- coding: utf-8 -*-
"""把实验3代码与控制台输出渲染为 PNG，供 1.2 / 1.3 节使用。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Consolas", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
CODE_PATH = r"D:\TYKKKKKK\Buddy\2026-10-14-08-40-32\exp3_yolo.py"


def render_lines(lines, path, title, fs=9.0):
    n = len(lines)
    fig_h = max(1.5, n * 0.165 + 0.5)
    fig = plt.figure(figsize=(11.5, fig_h), dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.012, 0.008, 0.976, 0.984])
    ax.set_facecolor("#FAFAFA")
    ax.axis("off")
    ax.text(0.0, 1.0, title, transform=ax.transAxes, fontsize=10.5,
            family=["Microsoft YaHei"], color="white", va="bottom",
            bbox=dict(facecolor="#2F5496", edgecolor="none",
                      boxstyle="square,pad=0.45"))
    for i, ln in enumerate(lines):
        y = 1.0 - (i + 1.6) / (n + 2.0)
        ax.text(0.0, y, ln.rstrip("\n"), transform=ax.transAxes,
                fontsize=fs, family=["Consolas", "Microsoft YaHei"],
                color="#1F1F1F", va="top")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("saved", path)


def render_code():
    with open(CODE_PATH, "r", encoding="utf-8") as f:
        code_lines = f.readlines()
    CHUNK = 120
    parts = [code_lines[i:i + CHUNK] for i in range(0, len(code_lines), CHUNK)]
    for k, part in enumerate(parts, 1):
        render_lines(part, os.path.join(OUT, f"code3_{k}.png"),
                     f"exp3_yolo.py  ({k}/{len(parts)})")


def render_console():
    cpath = os.path.join(OUT, "exp3_console.txt")
    with open(cpath, "r", encoding="utf-8") as f:
        console = f.read().rstrip("\n").splitlines()
    env_block = [
        "=" * 72,
        "运行环境",
        "-" * 72,
        "Python       : 3.13.14",
        "NumPy        : 2.5.2",
        "Matplotlib   : 3.11.2",
        "PyTorch      : 2.14.1+cpu",
        "torchvision  : 0.29.1+cpu",
        "计算设备     : cpu",
        "=" * 72,
        "",
    ]
    render_lines(env_block + console, os.path.join(OUT, "console3.png"),
                 "python exp3_yolo.py  -  运行输出", fs=8.6)
    print("console rendered")


if __name__ == "__main__":
    render_code()
    if os.path.exists(os.path.join(OUT, "exp3_console.txt")):
        render_console()
    print("done")
