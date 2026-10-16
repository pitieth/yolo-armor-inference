# -*- coding: utf-8 -*-
"""正确性验证：把 infer_onnx_fast.py 的解码结果与 ultralytics 官方后处理逐框比对

对比三路（都跑同一批图片）：
  A = ultralytics + best.pt      （PyTorch 参考）
  B = ultralytics + best.onnx    （官方后处理 + 同一条 ONNX 图）
  M = infer_onnx_fast 的解码     （自研实现）

B vs M 反映的是"我的实现是否正确"（同图、同后处理语义）；
A vs B 反映的是 ONNX 导出相对 PyTorch 的数值差异。

用法：
    python verify_rt.py <图片目录或单个图片> [best.onnx] [conf]
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_onnx_fast import ArmorDetector  # noqa: E402

DEFAULT_DIR = "."
ONNX = sys.argv[2] if len(sys.argv) > 2 else "best.onnx"
CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
IOU = 0.7
PT = "best.pt"


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match(iou, ca, cb, thr=0.5):
    pairs, ua, ub = [], set(), set()
    for i, j in np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]:
        if i in ua or j in ub or ca[i] != cb[j] or iou[i, j] < thr:
            continue
        ua.add(i); ub.add(j); pairs.append((i, j))
    return pairs


def official(model, img):
    r = model.predict(img, imgsz=640, conf=CONF, iou=IOU, device="0", verbose=False)[0]
    return (r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy(),
            r.keypoints.xy.cpu().numpy() if r.keypoints is not None else np.zeros((0, 4, 2)))


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    imgs = sorted(glob.glob(os.path.join(target, "*.jpg"))) if os.path.isdir(target) else [target]
    if not imgs:
        print("没有找到待验证的图片:", target)
        return 1

    from ultralytics import YOLO
    print("加载 A: ultralytics + %s" % PT)
    mA = YOLO(PT)
    print("加载 B: ultralytics + %s" % ONNX)
    mB = YOLO(ONNX, task="pose")
    print("初始化 M: 自研解码")
    mine = ArmorDetector(ONNX, conf=CONF, iou=IOU)
    print("M providers:", mine.sess.get_providers(), " graph:", mine.graph)

    totA = totB = totM = nAB = nBM = 0
    be_AB = ke_AB = be_BM = ke_BM = 0.0
    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            continue
        A, B = official(mA, img), official(mB, img)
        boxes, confs, cls, kpts, *_ = mine.detect(img)
        M = (boxes, cls.astype(int), confs, kpts)
        totA += len(A[0]); totB += len(B[0]); totM += len(M[0])
        for X, Y, tag in ((A, B, "AB"), (B, M, "BM")):
            pr = match(iou_matrix(X[0], Y[0]), X[1], Y[1])
            be = max([float(np.abs(X[0][i] - Y[0][j]).max()) for i, j in pr], default=0.0)
            ke = max([float(np.linalg.norm(X[3][i] - Y[3][j], axis=1).max()) for i, j in pr], default=0.0)
            if tag == "AB":
                be_AB = max(be_AB, be); ke_AB = max(ke_AB, ke); nAB += len(pr)
            else:
                be_BM = max(be_BM, be); ke_BM = max(ke_BM, ke); nBM += len(pr)
        print("%-28s A=%2d B=%2d M=%2d" % (os.path.basename(p), len(A[0]), len(B[0]), len(M[0])))

    print("\n=== 汇总 ===")
    print("检出总数: A(PyTorch)=%d  B(ONNX官方后处理)=%d  M(自研)=%d" % (totA, totB, totM))
    print("B vs M（验证自研实现）:   配对 %d  框最大差 %.3f px  角点最大差 %.3f px" % (nBM, be_BM, ke_BM))
    print("A vs B（ONNX 导出数值差）: 配对 %d  框最大差 %.3f px  角点最大差 %.3f px" % (nAB, be_AB, ke_AB))
    print("\n判定: " + ("✅ 自研实现与官方后处理一致" if (totB == totM == nBM and be_BM < 1.0)
                      else "❌ 存在差异"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
