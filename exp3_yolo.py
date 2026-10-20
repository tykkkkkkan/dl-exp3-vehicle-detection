# -*- coding: utf-8 -*-
"""
实验3：基于 VOC2007 的车辆检测（简化版 YOLOv8）
=================================================
课程：深度学习基础及应用
要求：
  a) 使用 PyTorch 实现简化版 YOLOv8 模型，检测车辆类别；
  b) 构建完整流水线：数据标注 -> 数据增强 -> 模型训练 -> NMS 后处理；
  c) 计算模型平均精度 mAP@0.5；
  d) 可视化检测结果与原始图像叠加效果。

实现要点：
  - 检测头保持 YOLOv8 的 anchor-free、单尺度、分类/回归解耦设计：
    7x7 网格，每个格点预测 1 个框（无锚框），输出 obj / cls / box 三支。
  - 骨干使用 ImageNet 预训练的 ResNet18（冻结），取 layer4 输出作为
    7x7x512 特征图。小数据集（数百张）在 CPU 上从零训练特征提取器会
    严重欠拟合，迁移学习是更合理的工程选择；同时冻结骨干后可缓存特征，
    检测头训练成本极低。
  - 数据集：PASCAL VOC2007，选取车辆类别 car / bus。
  - 置信度损失采用 pos_weight 动态平衡（正负格点约 1:24），否则模型会
    退化为"全预测无目标"，mAP 趋近 0。
"""
import os, sys, time, math, json
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageEnhance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- 全局配置 -----------------------------
VOC_ROOT = r"D:/TYKKKKKK/Buddy/2026-10-14-08-40-32/VOCdevkit/VOCdevkit/VOC2007"
CLASSES = ["car", "bus"]          # 车辆类别（VOC 中具代表性的两类车辆）
NUM_CLS = len(CLASSES)
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 224                    # 输入分辨率
GRID = 7                          # 特征图网格 7x7（ResNet layer4 输出，stride=32）
                                  # 实测 14x14 在小数据上更易过拟合，7x7 泛化更稳
BATCH = 32                        # 检测头训练批大小（特征已缓存，可用大 batch）
LR = 1e-2
EPOCHS = 400                      # 检测头训练轮数（缓存特征，成本很低）
AUG_COPIES = 2                    # 每张训练图预生成 2 个增强版本
SEED = 2323671
DEVICE = torch.device("cpu")
torch.manual_seed(SEED); np.random.seed(SEED)

OUT = r"D:/TYKKKKKK/Buddy/2026-10-14-08-40-32/outputs"
os.makedirs(OUT, exist_ok=True)


# ----------------------------- 1. 数据解析（标注读取）-----------------------------
def list_image_sets(split):
    """读取 VOC ImageSets/Main 中指定 split（train / val）的文件名列表。"""
    p = os.path.join(VOC_ROOT, "ImageSets", "Main", f"{split}.txt")
    with open(p) as f:
        return [line.strip() for line in f if line.strip()]


def parse_annotation(name):
    """解析单张图片的 XML 标注，返回归一化的 (boxes, labels)。"""
    ann_path = os.path.join(VOC_ROOT, "Annotations", name + ".xml")
    tree = ET.parse(ann_path); root = tree.getroot()
    boxes, labels = [], []
    sz = root.find("size")
    W = float(sz.find("width").text); H = float(sz.find("height").text)
    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in CLASS_ID:
            continue
        bnd = obj.find("bndbox")
        x1 = float(bnd.find("xmin").text); y1 = float(bnd.find("ymin").text)
        x2 = float(bnd.find("xmax").text); y2 = float(bnd.find("ymax").text)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        boxes.append([x1 / W, y1 / H, x2 / W, y2 / H])
        labels.append(CLASS_ID[cls])
    return boxes, labels


def build_split(split):
    """构建样本列表：仅保留含至少一个车辆目标的图像。"""
    items = []
    for name in list_image_sets(split):
        boxes, labels = parse_annotation(name)
        if not boxes:
            continue
        img_path = os.path.join(VOC_ROOT, "JPEGImages", name + ".jpg")
        if os.path.exists(img_path):
            items.append((img_path, boxes, labels))
    return items


# ----------------------------- 2. 数据增强 + 目标张量构建 -----------------------------
def augment(img, boxes, flip, bright, rot):
    """随机水平翻转 + 亮度扰动 + 小角度旋转（仅训练时）。boxes 为归一化坐标。"""
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = [[1 - x2, y1, 1 - x1, y2] for (x1, y1, x2, y2) in boxes]
    if bright:
        f = 0.8 + np.random.rand() * 0.4
        img = ImageEnhance.Brightness(img).enhance(f)
    if rot:
        ang = np.random.uniform(-10, 10)
        img = img.rotate(ang, resample=Image.BILINEAR)
    # 旋转会改变几何关系，为简化起见对旋转图像不做框变换，仅作外观扰动
    return img, boxes


def build_target(boxes, labels):
    """将一张图的标注转换为 7x7 网格目标张量。
    返回 shape=(7,7,1+NUM_CLS+4)：obj, cls_onehot, (tx,ty,tw,th)。"""
    G = GRID
    t = torch.zeros((G, G, 1 + NUM_CLS + 4), dtype=torch.float32)
    for (x1, y1, x2, y2), lab in zip(boxes, labels):
        cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
        gx, gy = int(cx * G), int(cy * G)
        gx = min(max(gx, 0), G - 1); gy = min(max(gy, 0), G - 1)
        bw = min(max(x2 - x1, 0.0), 1.0); bh = min(max(y2 - y1, 0.0), 1.0)
        if t[gy, gx, 0] == 1:       # 单框机制：单元格已被占用则跳过（简化）
            continue
        t[gy, gx, 0] = 1.0
        t[gy, gx, 1 + lab] = 1.0
        # box 编码：中心相对单元格偏移(0~1)，宽高归一化到整图(0~1)
        t[gy, gx, 1 + NUM_CLS + 0] = cx * G - gx
        t[gy, gx, 1 + NUM_CLS + 1] = cy * G - gy
        t[gy, gx, 1 + NUM_CLS + 2] = bw
        t[gy, gx, 1 + NUM_CLS + 3] = bh
    return t


def load_image_tensor(item, train):
    """读图 + 增强 + 缩放 + ImageNet 归一化。返回 x, 增强后的 boxes。"""
    img_path, boxes, labels = item
    img = Image.open(img_path).convert("RGB")
    if train:
        flip = np.random.rand() < 0.5
        bright = np.random.rand() < 0.5
        rot = np.random.rand() < 0.3
        img, boxes = augment(img, boxes, flip, bright, rot)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = torch.from_numpy(arr.transpose(2, 0, 1)).float()
    return x, boxes


def load_sample(item, train):
    x, boxes = load_image_tensor(item, train)
    return x, build_target(boxes, item[2])


# ----------------------------- 3. 模型：预训练骨干 + YOLO 解耦头 -----------------------------
def build_backbone():
    """ResNet18（ImageNet 预训练，冻结），输出 7x7x512 特征图。"""
    bb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in bb.parameters():
        p.requires_grad = False
    feat = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                         bb.layer1, bb.layer2, bb.layer3, bb.layer4)
    feat.eval()
    return feat


class YOLOHead(nn.Module):
    """YOLOv8 风格的 anchor-free 解耦检测头（obj / cls / box 三支）。
    先用 1x1 卷积做通道精炼（512->64），再分三支 1x1 输出。"""

    def __init__(self, in_ch=512, mid=64, num_cls=NUM_CLS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
        )
        self.obj_head = nn.Conv2d(mid, 1, 1)
        self.cls_head = nn.Conv2d(mid, num_cls, 1)
        self.box_head = nn.Conv2d(mid, 4, 1)
        # 框头偏置初始化：sigmoid(-1.5)≈0.18，从“小框”起步，加快收敛
        nn.init.constant_(self.box_head.bias, -1.5)

    def forward(self, f):
        g = self.stem(f)
        obj = self.obj_head(g).permute(0, 2, 3, 1)      # [B,H,W,1]
        cls = self.cls_head(g).permute(0, 2, 3, 1)      # [B,H,W,C]
        box = self.box_head(g).permute(0, 2, 3, 1)      # [B,H,W,4]
        return obj, cls, box


# ----------------------------- 4. 损失函数 -----------------------------
def yolo_loss(obj_p, cls_p, box_p, target, lambda_coord=10.0):
    """目标检测损失。
    关键点：7x7=49 个格点中通常只有 1~2 个含目标，正负样本比约 1:24。
    若直接用普通 BCE，背景格点梯度会压倒目标格点，模型退化为“全预测无目标”
    （训练 loss 很低但 mAP≈0）。此处用 pos_weight = 负样本数/正样本数 做动态
    加权，使正负梯度量级对等，从根本上修复该不平衡。"""
    obj_t = target[..., 0:1]
    cls_t = target[..., 1:1 + NUM_CLS]
    box_t = target[..., 1 + NUM_CLS:]
    # --- 置信度损失：按负正比动态加权 ---
    n_pos = obj_t.sum().clamp(min=1.0)
    n_neg = (1.0 - obj_t).sum().clamp(min=1.0)
    pos_weight = (n_neg / n_pos).clamp(max=30.0).detach()
    bce_obj = F.binary_cross_entropy_with_logits(
        obj_p, obj_t, pos_weight=pos_weight, reduction="none")
    obj_loss = bce_obj.mean()
    # --- 坐标与分类损失：仅对负责单元格计算 ---
    mask = (obj_t > 0.5).squeeze(-1)  # [G,G]
    if mask.sum() > 0:
        box_loss = F.smooth_l1_loss(box_p[mask], box_t[mask], beta=0.1)
        cls_loss = F.binary_cross_entropy_with_logits(cls_p[mask], cls_t[mask])
    else:
        box_loss = torch.zeros((), device=obj_p.device)
        cls_loss = torch.zeros((), device=obj_p.device)
    total = obj_loss + lambda_coord * box_loss + cls_loss
    return total, obj_loss, box_loss, cls_loss


# ----------------------------- 5. NMS -----------------------------
def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]; bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def nms(preds, iou_th=0.5):
    """按类别分别做非极大值抑制。"""
    out = []
    by_cls = defaultdict(list)
    for p in preds:
        by_cls[p[5]].append(p)
    for c, ps in by_cls.items():
        ps.sort(key=lambda x: -x[4])
        while ps:
            best = ps.pop(0)
            out.append(best)
            ps = [p for p in ps if iou(p, best) < iou_th]
    return out


# ----------------------------- 6. 解码与 mAP@0.5 -----------------------------
def decode(pred, conf_th=0.05):
    """将模型输出解码为绝对归一化框。返回 list of (x1,y1,x2,y2,score,cls)。"""
    obj, cls, box = pred  # [7,7,1],[7,7,C],[7,7,4]
    G = GRID
    preds = []
    with torch.no_grad():
        for gy in range(G):
            for gx in range(G):
                oc = float(torch.sigmoid(obj[gy, gx, 0]))
                if oc < conf_th:
                    continue
                cx = (gx + float(torch.sigmoid(box[gy, gx, 0]))) / G
                cy = (gy + float(torch.sigmoid(box[gy, gx, 1]))) / G
                bw = float(torch.sigmoid(box[gy, gx, 2]))
                bh = float(torch.sigmoid(box[gy, gx, 3]))
                x1, y1 = cx - bw / 2, cy - bh / 2
                x2, y2 = cx + bw / 2, cy + bh / 2
                cprob = torch.softmax(cls[gy, gx], dim=-1)
                cidx = int(torch.argmax(cprob))
                score = oc * float(cprob[cidx])
                if score < conf_th:
                    continue
                preds.append([x1, y1, x2, y2, score, cidx])
    return preds


def compute_map(all_preds, all_gts, iou_th=0.5):
    """all_preds: list per image of [x1,y1,x2,y2,score,cls]
       all_gts:   list per image of [x1,y1,x2,y2,cls]"""
    aps = {}
    for c in range(NUM_CLS):
        y_true = []
        y_score = []
        per_img_gt = [[] for _ in range(len(all_gts))]
        for i, gts in enumerate(all_gts):
            for g in gts:
                if int(g[4]) == c:
                    per_img_gt[i].append(g)
        # 收集该类所有预测
        flat = []
        for i, preds in enumerate(all_preds):
            for p in preds:
                if p[5] == c:
                    flat.append((p[4], i, p))
        flat.sort(key=lambda x: -x[0])
        tp = np.zeros(len(flat)); fp = np.zeros(len(flat))
        used = [set() for _ in range(len(all_gts))]
        for k, (score, i, p) in enumerate(flat):
            gts = per_img_gt[i]
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if j in used[i]:
                    continue
                v = iou(p, g)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_th and best_j >= 0:
                tp[k] = 1; used[i].add(best_j)
            else:
                fp[k] = 1
        npos = sum(len(g) for g in per_img_gt)
        if npos == 0:
            aps[c] = 0.0; continue
        tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
        recall = tp_c / npos
        prec = tp_c / (tp_c + fp_c + 1e-6)
        # 11-point 插值 AP
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = prec[recall >= t].max() if np.any(recall >= t) else 0.0
            ap += p / 11
        aps[c] = ap
    mean = float(np.mean(list(aps.values())))
    return mean, aps


# ----------------------------- 7. 特征缓存 -----------------------------
def extract_features(backbone, items, train, copies=1):
    """用冻结骨干把图像编码为 7x7x512 特征并缓存。
    train=True 时对每张图生成 copies 个不同增强版本，保证数据增强生效。"""
    feats, targets = [], []
    with torch.no_grad():
        for n, it in enumerate(items):
            for _ in range(copies if train else 1):
                x, boxes = load_image_tensor(it, train)
                f = backbone(x.unsqueeze(0))[0]        # [512,7,7]
                feats.append(f)
                targets.append(build_target(boxes, it[2]))
            if (n + 1) % 100 == 0:
                print(f"  特征提取 {n+1}/{len(items)}", flush=True)
    return torch.stack(feats), torch.stack(targets)


# ----------------------------- 8. 评估 -----------------------------
def evaluate(head, val_feats, val_targets, val_data, conf_th=0.01, iou_th=0.5):
    """在缓存的验证特征上评估 mAP@0.5。"""
    head.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for f, it in zip(val_feats, val_data):
            obj, cls, box = head(f.unsqueeze(0))
            preds = nms(decode((obj[0], cls[0], box[0]), conf_th), iou_th)
            all_preds.append(preds)
            all_gts.append([[x1, y1, x2, y2, lab]
                            for (x1, y1, x2, y2), lab in zip(it[1], it[2])])
    return compute_map(all_preds, all_gts, iou_th)


# ----------------------------- 9. 可视化 -----------------------------
def visualize(item, preds, save_path):
    img_path, boxes, labels = item
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img); ax.axis("off")
    for (x1, y1, x2, y2), lab in zip(boxes, labels):
        ax.add_patch(plt.Rectangle((x1 * IMG_SIZE, y1 * IMG_SIZE),
                     (x2 - x1) * IMG_SIZE, (y2 - y1) * IMG_SIZE,
                     fill=False, edgecolor="lime", lw=2))
        ax.text(x1 * IMG_SIZE, max(y1 * IMG_SIZE - 4, 2), f"GT:{CLASSES[lab]}",
                color="lime", fontsize=8)
    for p in preds:
        x1, y1, x2, y2, s, c = p
        ax.add_patch(plt.Rectangle((x1 * IMG_SIZE, y1 * IMG_SIZE),
                     (x2 - x1) * IMG_SIZE, (y2 - y1) * IMG_SIZE,
                     fill=False, edgecolor="red", lw=2))
        ax.text(x1 * IMG_SIZE, max(y1 * IMG_SIZE - 16, 2), f"{CLASSES[c]}:{s:.2f}",
                color="red", fontsize=8)
    ax.set_title("GT: green   Pred: red", fontsize=9)
    plt.tight_layout(); plt.savefig(save_path, dpi=110); plt.close()


# ----------------------------- 10. 训练主流程 -----------------------------
def main():
    t0 = time.time()
    print("=" * 62)
    print("实验3 简化版 YOLOv8 车辆检测  (预训练骨干 + 检测头, CPU)")
    print("=" * 62)
    train_data = build_split("train")
    val_data = build_split("val")
    print(f"训练样本(含车辆): {len(train_data)}  验证样本: {len(val_data)}")
    print(f"类别: {CLASSES}  | 输入 {IMG_SIZE} 网格 {GRID}x{GRID}")

    backbone = build_backbone()
    head = YOLOHead(512, 64, NUM_CLS).to(DEVICE)
    n_backbone = sum(p.numel() for p in backbone.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    print(f"骨干(冻结)参数量: {n_backbone:,}")
    print(f"检测头(可训练)参数量: {n_head:,}")
    print(f"模型总参数量: {n_backbone + n_head:,}")

    # ---- 特征缓存（数据增强在提取前完成）----
    print("\n[1/3] 提取训练特征（每图 %d 个增强版本）..." % AUG_COPIES, flush=True)
    tr_f, tr_t = extract_features(backbone, train_data, True, AUG_COPIES)
    print(f"  训练特征: {tuple(tr_f.shape)}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    print("[2/3] 提取验证特征...", flush=True)
    va_f, va_t = extract_features(backbone, val_data, False, 1)
    print(f"  验证特征: {tuple(va_f.shape)}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- 训练检测头 ----
    print(f"\n[3/3] 训练检测头 {EPOCHS} 轮 (batch={BATCH}, lr={LR})...", flush=True)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = len(tr_f)
    losses = []
    best_map, best_state = -1.0, None
    for ep in range(1, EPOCHS + 1):
        head.train()
        perm = torch.randperm(n)
        ep_loss = 0.0; nb = 0
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            op, cp, bp = head(tr_f[idx])
            loss, ol, bl, cl = yolo_loss(op, cp, bp, tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.detach()); nb += 1
        sched.step()
        losses.append(ep_loss / nb)
        # 每 10 轮在验证子集上评估，保存最优权重（早停思想）
        if ep % 10 == 0 or ep == 1:
            mAP, aps = evaluate(head, va_f[:150], va_t[:150], val_data[:150])
            if mAP > best_map:
                best_map = mAP
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            print(f"[ep {ep:3d}] loss {ep_loss/nb:.4f} | val mAP@0.5 {mAP:.4f} "
                  f"AP(car)={aps.get(0,0):.4f} AP(bus)={aps.get(1,0):.4f} "
                  f"| best {best_map:.4f} | {(time.time()-t0)/60:.1f}min", flush=True)

    # ---- 恢复验证集最优权重，做最终全量评估 ----
    if best_state is not None:
        head.load_state_dict(best_state)
        print(f"\n已恢复验证集最优权重 (sub-val mAP={best_map:.4f})")
    mAP, aps = evaluate(head, va_f, va_t, val_data)
    torch.save({"head": head.state_dict(), "classes": CLASSES,
                "grid": GRID, "img_size": IMG_SIZE},
               os.path.join(OUT, "exp3_head.pth"))
    total_el = (time.time() - t0) / 60
    print(f"\n[FINAL] mAP@0.5 = {mAP:.4f}  AP(car)={aps.get(0,0):.4f} "
          f"AP(bus)={aps.get(1,0):.4f}  用时 {total_el:.2f} min")

    # ---- 可视化若干验证样本 ----
    head.eval()
    vis_paths = []
    for k in range(min(6, len(val_data))):
        f = va_f[k]
        with torch.no_grad():
            obj, cls, box = head(f.unsqueeze(0))
        preds = nms(decode((obj[0], cls[0], box[0]), 0.10), 0.5)
        sp = os.path.join(OUT, f"exp3_vis_{k+1}.png")
        visualize(val_data[k], preds, sp)
        vis_paths.append(sp)

    # ---- 损失曲线 ----
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(losses, color="#1F6B54", lw=1.4, label="train loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title(f"Detection Head Training Loss (final mAP@0.5={mAP:.3f})")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); loss_p = os.path.join(OUT, "exp3_loss.png")
    plt.savefig(loss_p, dpi=110); plt.close()

    # ---- 汇总 ----
    summary = {
        "classes": CLASSES, "img_size": IMG_SIZE, "grid": GRID,
        "n_backbone": n_backbone, "n_head": n_head,
        "n_params": n_backbone + n_head,
        "epochs": EPOCHS, "train_time_min": round(total_el, 2),
        "mAP_05": round(mAP, 4),
        "AP_car": round(aps.get(0, 0), 4), "AP_bus": round(aps.get(1, 0), 4),
        "train_samples": len(train_data), "val_samples": len(val_data),
        "final_loss": round(losses[-1], 4),
    }
    with open(os.path.join(OUT, "exp3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n[SUMMARY]", json.dumps(summary, ensure_ascii=False), flush=True)
    print("[DONE]")


if __name__ == "__main__":
    main()
