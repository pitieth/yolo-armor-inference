# -*- coding: utf-8 -*-
"""进阶（可选）：原生 ONNX Runtime + CUDA Graph 推理脚本

与 infer.py 输出完全相同的检测框与 4 个关键点，但绕开 ultralytics 的 Python 前/后处理，
用 CUDA Graph + IOBinding 把单帧推理压到 7.8 ms 左右（RTX 4070 Laptop，约 125 FPS）。

前置：先把 best.pt 导出 ONNX（只需一次）
    yolo export model=best.pt format=onnx imgsz=640

用法：
    python infer_onnx_fast.py --onnx best.onnx --source my_video.mp4 --output result_fast.mp4
    python infer_onnx_fast.py --onnx best.onnx --source my_video.mp4 --device cpu   # 用 CPU EP

注意：CUDA Graph 会话必须使用 IOBinding —— 用普通 session.run() 第 2 次起会静默返回空列表。
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

# CUDA 13 运行时 DLL 随 torch 打包，ORT 的 CUDA EP 需要能找到它们
TORCH_LIB = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "torch", "lib")
TORCH_LIB = os.path.abspath(TORCH_LIB)
if os.path.isdir(TORCH_LIB):
    os.add_dll_directory(TORCH_LIB)
    os.environ["PATH"] = TORCH_LIB + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort  # noqa: E402

CLASS_NAMES = ["B1", "B2", "B3", "R1", "R2", "R3"]
COLORS = [(255, 80, 80), (255, 160, 60), (60, 200, 255),
          (80, 80, 255), (60, 160, 255), (255, 220, 80)]
NK, NDIM = 4, 2
PAD_VALUE = 114
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------- 前处理
def letterbox(img, size=640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return img


def to_blob(img_bgr, size=640):
    lb = letterbox(img_bgr, size)
    x = lb[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0


# ---------------------------------------------------------------- 解码 / NMS / 坐标还原
def _gain_pad(in_shape, out_shape):
    """坐标还原参数：letterbox 是逐边取整缩放，X/Y 实际比例不等，必须分轴"""
    gain = min(in_shape[0] / out_shape[0], in_shape[1] / out_shape[1])
    new_h, new_w = round(out_shape[0] * gain), round(out_shape[1] * gain)
    gain_y, gain_x = new_h / out_shape[0], new_w / out_shape[1]
    return (gain_x, gain_y,
            round((in_shape[1] - new_w) / 2 - 0.1),
            round((in_shape[0] - new_h) / 2 - 0.1))


def scale_boxes(b, in_shape, out_shape):
    gx, gy, px, py = _gain_pad(in_shape, out_shape)
    b = b.astype(np.float32).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - px) / gx
    b[:, [1, 3]] = (b[:, [1, 3]] - py) / gy
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, out_shape[1])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, out_shape[0])
    return b


def scale_coords(c, in_shape, out_shape):
    gx, gy, px, py = _gain_pad(in_shape, out_shape)
    c = c.astype(np.float32).copy()
    c[..., 0] = ((c[..., 0] - px) / gx).clip(0, out_shape[1])
    c[..., 1] = ((c[..., 1] - py) / gy).clip(0, out_shape[0])
    return c


def nms_greedy(boxes, scores, iou_thres, max_det=300):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size and len(keep) < max_det:
        i = int(order[0]); keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        inter = (np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
                 * np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest])))
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, dtype=int)


def postprocess(out, orig_shape, conf=0.25, iou=0.7, max_det=300, imgsz=640):
    """out: (1, 4+nc+4*2, 8400)。关键点在图内已是 640 空间的绝对坐标，直接取用"""
    nc = len(CLASS_NAMES)
    p = out[0].T
    xywh, cls_scores = p[:, :4], p[:, 4:4 + nc]
    kpts = p[:, 4 + nc:].reshape(-1, NK, NDIM)
    confs, cls = cls_scores.max(1), cls_scores.argmax(1)
    m = confs > conf
    if not m.any():
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                np.zeros(0, int), np.zeros((0, NK, NDIM), np.float32))
    xywh, confs, cls, kpts = xywh[m], confs[m], cls[m], kpts[m]
    b = np.empty_like(xywh)
    b[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    b[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    b[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    b[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    keep = []
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        keep.append(idx[nms_greedy(b[idx], confs[idx], iou, max_det)])
    keep = np.concatenate(keep)
    keep = keep[np.argsort(-confs[keep])][:max_det]
    return (scale_boxes(b[keep], (imgsz, imgsz), orig_shape), confs[keep], cls[keep],
            scale_coords(kpts[keep].reshape(-1, NDIM), (imgsz, imgsz),
                         orig_shape).reshape(-1, NK, NDIM))


# ---------------------------------------------------------------- 检测器
class ArmorDetector:
    def __init__(self, onnx, conf=0.25, iou=0.7, imgsz=640, cuda_graph=True, device="cuda"):
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if device == "cpu":
            self.sess = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])
            self.graph = False
        else:
            # providers 与 provider_options 必须等长，否则 ORT 会静默回退到 CPU
            self.sess = ort.InferenceSession(
                onnx, so,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                provider_options=[{"enable_cuda_graph": "1"}, {}] if cuda_graph else None)
            self.graph = cuda_graph and "CUDAExecutionProvider" in self.sess.get_providers()
            if self.sess.get_providers()[0] != "CUDAExecutionProvider":
                print("[警告] CUDA EP 未生效，实际使用:", self.sess.get_providers())
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.out_shape = [d if isinstance(d, int) else 1
                          for d in self.sess.get_outputs()[0].shape]
        if self.graph:
            self.dev_in = ort.OrtValue.ortvalue_from_shape_and_type(
                [1, 3, imgsz, imgsz], np.float32, "cuda", 0)
            self.dev_out = ort.OrtValue.ortvalue_from_shape_and_type(
                self.out_shape, np.float32, "cuda", 0)

    def infer(self, blob):
        if self.graph:
            self.dev_in.update_inplace(blob)
            io = self.sess.io_binding()
            io.bind_ortvalue_input(self.in_name, self.dev_in)
            io.bind_ortvalue_output(self.out_name, self.dev_out)
            self.sess.run_with_iobinding(io)
            return io.get_outputs()[0].numpy()
        return self.sess.run(None, {self.in_name: blob})[0]

    def detect(self, frame):
        t0 = time.perf_counter()
        blob = to_blob(frame, self.imgsz)
        t1 = time.perf_counter()
        out = self.infer(blob)
        t2 = time.perf_counter()
        boxes, confs, cls, kpts = postprocess(out, frame.shape[:2], self.conf,
                                              self.iou, 300, self.imgsz)
        t3 = time.perf_counter()
        return boxes, confs, cls, kpts, (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000


# ---------------------------------------------------------------- 绘制
def draw(frame, boxes, confs, cls, kpts):
    for i in range(len(boxes)):
        cid = int(cls[i]); color = COLORS[cid % len(COLORS)]
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = "%s %.2f" % (CLASS_NAMES[cid], confs[i])
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
        ty = max(y1 - 8, th + 4)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), color, -1)
        cv2.putText(frame, label, (x1 + 3, ty), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        poly = []
        for j in range(NK):
            px, py = int(round(kpts[i][j][0])), int(round(kpts[i][j][1]))
            poly.append((px, py))
            cv2.circle(frame, (px, py), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, str(j), (px + 6, py - 6), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.polylines(frame, [np.array(poly, np.int32).reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
    return len(boxes)


def main():
    ap = argparse.ArgumentParser(description="ONNX Runtime + CUDA Graph 装甲板关键点推理（进阶）")
    ap.add_argument("--onnx", default="best.onnx", help="ONNX 模型路径（由 best.pt 导出）")
    ap.add_argument("--source", required=True, help="输入视频路径")
    ap.add_argument("--output", default="", help="结果视频路径，默认 <源文件名>_fast_result.mp4")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--no-graph", action="store_true", help="关闭 CUDA Graph")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        print("[错误] 找不到 ONNX 模型：%s" % args.onnx)
        print("       请先执行： yolo export model=best.pt format=onnx imgsz=640")
        return 1
    if not os.path.exists(args.source):
        print("[错误] 找不到输入视频：%s" % args.source)
        return 1

    det = ArmorDetector(args.onnx, args.conf, args.iou, args.imgsz,
                        cuda_graph=not args.no_graph, device=args.device)
    print("ONNX 模型 : %s" % args.onnx)
    print("会话      : %s   CUDA Graph: %s" % (det.sess.get_providers(), det.graph))

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("[错误] 无法打开输入视频")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = args.output or (os.path.splitext(os.path.basename(args.source))[0] + "_fast_result.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print("输出      : %s  (%dx%d @ %.2f fps)" % (out_path, w, h, fps))

    idx = t_pre = t_inf = t_post = 0.0
    n_det = 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        boxes, confs, cls, kpts, a, b, c = det.detect(frame)
        t_pre += a; t_inf += b; t_post += c
        n_det += draw(frame, boxes, confs, cls, kpts)
        writer.write(frame)
        if idx % 100 == 0:
            print("  已处理 %d 帧 ..." % idx, flush=True)
    wall = time.perf_counter() - t0
    cap.release(); writer.release()

    print("\n=== 结果汇总 ===")
    print("处理帧数 : %d，检出目标 %d 个" % (idx, n_det))
    print("平均耗时 : %.2f ms/帧 (≈%.1f FPS)   预处理 %.2f / 推理 %.2f / 解码+NMS %.2f"
          % (wall * 1000 / idx, idx / wall, t_pre / idx, t_inf / idx, t_post / idx))
    print("结果视频 : %s" % os.path.abspath(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
