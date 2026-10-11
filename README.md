# yolo-armor-inference

基于 YOLO 的装甲板目标检测与推理项目。

## 项目简介

本项目用于对装甲板视频或图片进行目标检测推理，输出检测框、类别与置信度等结果。

## 环境要求

- Python 3.10+
- PyTorch
- Ultralytics YOLO

## 安装

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 使用方法

```bash
python inference.py --source path/to/image-or-video --weights path/to/model.pt
```

> 当前仓库仍在初始化阶段，请根据实际入口文件、模型权重和参数补充使用说明。

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── inference.py
└── weights/
```

## 开发计划

- [ ] 补充推理脚本
- [ ] 添加模型权重与配置说明
- [ ] 支持图片、视频和摄像头输入
- [ ] 增加示例结果

## License

待补充。
