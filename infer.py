# -*- coding: utf-8 -*-
"""装甲板四角点检测推理脚本（YOLO26n-pose / best.pt）

作业要求对应关系：
  1. 加载模型权重、读取自行拍摄的视频        -> --weights / --source
  2. 对视频执行推理                          -> YOLO.predict 逐帧
  3. 显示检测框 / 类别 / 置信度 / 4 个关键点   -> 关键点取自 results.keypoints，非检测框角点
  4. 保存可正常播放的结果视频                -> --output，分辨率与帧率与原视频一致

模型路径、视频路径、输出路径均可通过命令行参数修改（见 --help）。

用法：
    python infer.py --weights best.pt --source my_video.mp4 --output result.mp4
    python infer.py --weights best.pt --source my_video.mp4 --output result.mp4 --device cpu
    python infer.py --weights best.pt --source my_video.mp4 --show          # 边跑边看

依赖：ultralytics, opencv-python, torch（见 requirements.txt）
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

# 六类装甲板，顺序与模型训练时一致（cls 0..5）
CLASS_NAMES = ["B1", "B2", "B3", "R1", "R2", "R3"]
# 每类一个颜色，保证关键点颜色与所属目标的检测框一致
COLORS = [(255, 80, 80), (255, 160, 60), (60, 200, 255),
          (80, 80, 255), (60, 160, 255), (255, 220, 80)]
N_KEYPOINTS = 4          # 每个装甲板 4 个角点
LINE_W = 2               # 检测框线宽
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    p = argparse.ArgumentParser(description="YOLO26n 装甲板四角点视频推理")
    p.add_argument("--weights", default="best.pt",
                   help="模型权重路径（作业指定的 best.pt）")
    p.add_argument("--source", required=True,
                   help="输入视频路径（也可传单张图片）")
    p.add_argument("--output", default="",
                   help="结果视频保存路径；不填则自动生成 <源文件名>_result.mp4")
    p.add_argument("--device", default="",
                   help="推理设备：留空自动选择；'0' 用 GPU；'cpu' 用 CPU")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.7, help="NMS 的 IoU 阈值")
    p.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    p.add_argument("--show", action="store_true", help="同时弹出窗口预览（按 q 退出）")
    p.add_argument("--csv", default="",
                   help="可选：把每帧检测结果导出为 CSV（帧号/类别/置信度/框/4 个关键点）")
    return p.parse_args()


def pick_device(requested):
    """未指定设备时：有可用 CUDA 就用 GPU，否则回退 CPU。"""
    if requested:
        return requested
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def draw_detections(frame, r, kpt_conf_min=0.0):
    """把检测框、类别、置信度和 4 个关键点画到帧上。

    关键点坐标来自 model 的输出 results[0].keypoints.xy（模型直接预测的角点），
    不是用检测框的四个角推出来的。
    """
    n = 0
    boxes = r.boxes
    kpts = r.keypoints                      # 可能为 None（该帧无目标）

    if boxes is not None:
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            color = COLORS[cls_id % len(COLORS)]
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i]]
            n += 1

            # 1) 检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, LINE_W)

            # 2) 类别 + 置信度（带底色，避免深色背景看不清）
            label = "%s %.2f" % (name, conf)
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
            ty = max(y1 - 8, th + 4)
            cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), color, -1)
            cv2.putText(frame, label, (x1 + 3, ty), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # 3) 模型预测的 4 个关键点：画点 + 按顺序连成四边形 + 标出序号
            if kpts is not None and i < len(kpts.xy):
                kp = kpts.xy[i].cpu().numpy()          # (4, 2) 原图像素坐标
                # 关键点置信度（pose 任务的 kpts.conf，shape (4,)）；取不到就忽略
                kc = None
                if getattr(kpts, "conf", None) is not None and len(kpts.conf) > i:
                    kc = kpts.conf[i].cpu().numpy()
                poly = []
                for j in range(min(N_KEYPOINTS, len(kp))):
                    if kc is not None and kc[j] < kpt_conf_min:
                        continue                       # 置信度过低的点不画
                    px, py = int(round(kp[j][0])), int(round(kp[j][1]))
                    poly.append((px, py))
                    cv2.circle(frame, (px, py), 4, color, -1, cv2.LINE_AA)          # 实心点
                    cv2.circle(frame, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)  # 白圈描边，更醒目
                    cv2.putText(frame, str(j), (px + 6, py - 6), FONT, 0.45,
                                (255, 255, 255), 1, cv2.LINE_AA)
                if len(poly) >= 3:
                    cv2.polylines(frame, [np.array(poly, np.int32).reshape(-1, 1, 2)],
                                  True, color, LINE_W, cv2.LINE_AA)
    return n


def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        print("[错误] 找不到权重文件：%s" % args.weights)
        print("       请从作业参考材料下载 best.pt，放到脚本同目录，或用 --weights 指定路径。")
        return 1
    if not os.path.exists(args.source):
        print("[错误] 找不到输入视频：%s" % args.source)
        return 1

    # 延迟导入，这样 --help 不需要装好 torch 也能看
    from ultralytics import YOLO

    device = pick_device(args.device)
    print("权重     : %s" % args.weights)
    print("输入     : %s" % args.source)
    print("推理设备 : %s" % device)

    model = YOLO(args.weights)
    print("模型任务 : %s   类别: %s" % (model.task, list(model.names.values())))

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("[错误] 无法打开输入视频")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("视频规格 : %dx%d  %.2f fps  共 %d 帧" % (w, h, fps, total))

    out_path = args.output
    if not out_path:
        base = os.path.splitext(os.path.basename(args.source))[0]
        out_path = "%s_result.mp4" % base
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")      # mp4v 兼容性最好
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print("[错误] 无法创建输出视频：%s" % out_path)
        return 1
    print("输出     : %s" % out_path)

    idx = 0
    det_frames = 0
    cls_count = {k: 0 for k in CLASS_NAMES}
    t0 = time.perf_counter()

    csv_f = None
    if args.csv:
        csv_f = open(args.csv, "w", encoding="utf-8")
        csv_f.write("frame,cls,name,conf,x1,y1,x2,y2,"
                    + ",".join("kx%d,ky%d" % (i + 1, i + 1) for i in range(N_KEYPOINTS))
                    + "\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1

        r = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                          iou=args.iou, device=device, verbose=False)[0]
        n = draw_detections(frame, r)
        if n > 0:
            det_frames += 1
            for i in range(len(r.boxes)):
                cid = int(r.boxes.cls[i])
                if cid < len(CLASS_NAMES):
                    cls_count[CLASS_NAMES[cid]] += 1
                # 导出逐帧结果（坐标均为原图像素坐标）
                if csv_f is not None:
                    b = r.boxes.xyxy[i].cpu().numpy()
                    kp = r.keypoints.xy[i].cpu().numpy() if r.keypoints is not None else None
                    csv_f.write("%d,%d,%s,%.4f,%.1f,%.1f,%.1f,%.1f,%s\n" % (
                        idx, cid,
                        CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid),
                        float(r.boxes.conf[i]), b[0], b[1], b[2], b[3],
                        ",".join("%.1f,%.1f" % (p[0], p[1]) for p in kp)
                        if kp is not None else ""))

        writer.write(frame)
        if args.show:
            cv2.imshow("infer", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        if idx % 50 == 0:
            print("  已处理 %d/%d 帧 ..." % (idx, total), flush=True)

    wall = time.perf_counter() - t0
    cap.release()
    writer.release()
    if csv_f is not None:
        csv_f.close()
    if args.show:
        cv2.destroyAllWindows()

    print("\n=== 结果汇总 ===")
    print("处理帧数     : %d" % idx)
    print("有目标的帧数 : %d (%.1f%%)" % (det_frames, 100.0 * det_frames / idx if idx else 0))
    print("各目标累计计数: %s" % {k: v for k, v in cls_count.items() if v})
    print("总耗时       : %.1f s  (平均 %.1f ms/帧)" % (wall, wall * 1000 / idx if idx else 0))
    print("结果视频     : %s" % os.path.abspath(out_path))
    if args.csv:
        print("逐帧 CSV     : %s" % os.path.abspath(args.csv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
