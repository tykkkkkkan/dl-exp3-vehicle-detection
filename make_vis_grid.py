# -*- coding: utf-8 -*-
"""把 6 张检测可视化拼成 2x3 网格，供报告 1.3 节使用。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

OUT = "outputs"
paths = [os.path.join(OUT, f"exp3_vis_{i}.png") for i in range(1, 7)]
paths = [p for p in paths if os.path.exists(p)]
if not paths:
    raise SystemExit("no vis images")

fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.0))
for ax in axes.ravel():
    ax.axis("off")
for ax, p in zip(axes.ravel(), paths):
    ax.imshow(mpimg.imread(p))
    ax.axis("off")
fig.suptitle("Vehicle Detection Results on VOC2007 Val (green=GT, red=prediction)",
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(OUT, "exp3_vis_grid.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print("saved", out)
